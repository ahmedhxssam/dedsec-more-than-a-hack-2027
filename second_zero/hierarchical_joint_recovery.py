"""Native joint state recovery with three distinct nonnegative transition costs."""
import numpy as np
import torch
from torch import nn
from second_zero import fast_native_joint_recovery as source
from second_zero import joint_corrupted_recovery as old
from second_zero.parallel_path_likelihood import multiply
inputs, targets, batch, joined = source.inputs, source.targets, source.batch, source.joined
RADIUS = source.RADIUS
POS_WEIGHT = 25.


class JointRecovery(source.JointRecovery):
    def __init__(self):
        super().__init__()
        self.transitions = nn.Conv1d(64, 3, 1)
        nn.init.zeros_(self.transitions.weight)
        nn.init.zeros_(self.transitions.bias)

    def forward(self, x):
        z = self.input(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            z = z + .5 * block(z)
        delta = self.output(z)
        return x[:, -2] + delta[:, 0], delta[:, 1], self.transitions(z).transpose(1, 2)


def transition_scores(hazard):
    """Rows are prior state, columns current state; diagonal costs exactly zero."""
    if hazard.shape[-1] != 3:
        raise ValueError('Owner switch, live-to-dead, dead-to-live hazards required')
    cost = nn.functional.softplus(-hazard)
    flip, stop, start = cost.unbind(-1)
    zero = torch.zeros_like(flip)
    return torch.stack([zero, -flip, -stop, -flip, zero, -stop,
                        -start-np.log(2.), -start-np.log(2.), zero], -1).reshape(*hazard.shape[:-1], 3, 3)


def path_nll(node, transition, truth, mask):
    active = mask.bool()
    if not active.any():
        return node.sum()*0 + transition.sum()*0
    b, t, c = node.shape
    if c != 3 or transition.shape != (b, t, 3, 3) or truth.shape != (b, t) or active.shape != (b, t):
        raise ValueError('Aligned native-frame nodes, transitions, targets and mask required')
    eye = torch.eye(3, dtype=torch.bool, device=node.device)
    identity = torch.where(eye, node.new_tensor(0.), node.new_tensor(-torch.inf))
    previous = torch.cat([torch.zeros((b, 1), dtype=torch.bool, device=node.device), active[:, :-1]], 1)
    cont = active & previous
    matrix = node[:, :, None, :] + torch.where(cont[..., None, None], transition, torch.zeros_like(transition))
    matrix = torch.where(active[..., None, None], matrix, identity[None, None])
    while matrix.shape[1] > 1:
        if matrix.shape[1] % 2:
            matrix = torch.cat([matrix, identity[None, None].expand(b, 1, 3, 3)], 1)
        matrix = multiply(matrix[:, 0::2], matrix[:, 1::2])
    partition = torch.logsumexp(matrix[:, 0].reshape(b, -1), 1)-np.log(3.)
    y = truth.long().clamp(0, 2)
    prev_y = torch.cat([y[:, :1], y[:, :-1]], 1)
    node_score = (node.gather(2, y[..., None]).squeeze(-1)*active).sum(1)
    edge_score = (transition.reshape(b, t, 9).gather(2, (3*prev_y+y)[..., None]).squeeze(-1)*cont).sum(1)
    return (partition-node_score-edge_score).sum()/active.sum().clamp_min(1)


def loss(model, v):
    owner, dead, raw_hazard = model(v['x'])
    logp = old.emissions(owner, dead)
    w = v['weight']
    known = (w > 0) & (v['y'] < 3)
    y = v['y'].clamp(0, 2).long()
    err = -logp.gather(2, y[..., None]).squeeze(-1)
    frame = .5*(err*w).sum()/w.sum().clamp_min(1.) + .5*err[known].mean()
    prev = torch.cat([y[:, :1], y[:, :-1]], 1)
    valid = known.clone(); valid[:, 0] = False
    valid[:, 1:] &= v['y'][:, :-1] < 3
    flip_mask = valid & (prev < 2) & (y < 2)
    stop_mask = valid & (prev < 2)
    start_mask = valid & (prev == 2)
    masks = torch.stack([flip_mask, stop_mask, start_mask], -1)
    labels = torch.stack([y != prev, y == 2, y < 2], -1).to(raw_hazard.dtype)
    binary = nn.functional.binary_cross_entropy_with_logits(raw_hazard, labels, reduction='none', pos_weight=raw_hazard.new_tensor(POS_WEIGHT))
    edge = sum((binary[..., i]*masks[..., i]).sum()/masks[..., i].sum().clamp_min(1.) for i in range(3))/3
    # All native central frames contribute, with exactly the same emissions,
    # hazard correction, transition costs and resets used at inference.
    sl = slice(source.MARGIN, -source.MARGIN)
    structured = path_nll(logp[:, sl]/5., transition_scores(raw_hazard[:, sl]-np.log(POS_WEIGHT)), y[:, sl], known[:, sl])
    regularizer = .0001*(owner.square()+dead.square())[known].mean()
    return frame + .1*edge + .25*structured + regularizer


def initial_hazards(target_rows):
    count = np.zeros(3, np.float64); total = np.zeros(3, np.float64)
    for row in target_rows:
        y = row['y']; a, b = y[:-1], y[1:]
        valid = (a < 3) & (b < 3)
        masks = [valid & (a < 2) & (b < 2), valid & (a < 2), valid & (a == 2)]
        labels = [a != b, b == 2, b < 2]
        for i in range(3):
            total[i] += masks[i].sum(); count[i] += (masks[i] & labels[i]).sum()
    p = (count+1)/(total+2)
    return np.log(p/(1-p))+np.log(POS_WEIGHT), {'positive': count.tolist(), 'eligible': total.tolist(), 'smoothing': 'Beta1,1; training physical frames counted once'}


@torch.no_grad()
def predict(model, native, coarse, chunk=4096):
    model.eval()
    n = native.shape[-1]; result = np.empty((n, 5), np.float32)
    for a in range(0, n, chunk):
        b = min(a+chunk, n); lo, hi = max(0, a-RADIUS), min(n, b+RADIUS)
        x = np.stack([joined(native[:, :, lo:hi], coarse[lo:hi], swap) for swap in (False, True)])
        owner, dead, hazard = model(torch.from_numpy(x))
        out = torch.cat([(.5*(owner[0]-owner[1]))[:, None], (.5*(dead[0]+dead[1]))[:, None],
                         .5*(hazard[0]+hazard[1])-np.log(POS_WEIGHT)], -1).numpy()
        result[a:b] = out[a-lo:b-lo]
    return result


def decode(node, hazard):
    node = np.asarray(node, np.float64)
    hazard = np.asarray(hazard, np.float64)
    if node.shape != (len(hazard), 3) or hazard.shape[1:] != (3,) or not len(hazard):
        raise ValueError('Aligned nonempty three-state nodes and three hazards required')
    cost = np.logaddexp(0., -hazard)
    score = node[0]/5.; back = np.zeros((len(node), 3), np.int8)
    for i in range(1, len(node)):
        f, d, s = cost[i]
        transition = np.array([[0., -f, -d], [-f, 0., -d], [-s-np.log(2.), -s-np.log(2.), 0.]])
        moved = score[:, None]+transition
        back[i] = np.argmax(moved, axis=0)
        score = moved[back[i], np.arange(3)]+node[i]/5.
    out = np.empty(len(node), np.int8); out[-1] = np.argmax(score)
    for i in range(len(node)-1, 0, -1): out[i-1] = back[i, out[i]]
    return out


def compose(parent, logits, alpha, change_offset):
    parent = np.asarray(parent)
    if alpha == 0: return parent.copy()
    z = np.asarray(logits, np.float32)
    if change_offset != 0 or z.shape != (len(parent), 5) or not np.isfinite(z).all():
        raise ValueError('Fixed zero offset and aligned finite five-channel logits required')
    tensor = torch.from_numpy(z)
    probs = old.emissions(tensor[:, 0], tensor[:, 1]).exp().numpy()
    prior = np.full_like(probs, .01); known = parent < 3
    prior[np.flatnonzero(known), parent[known]] = .98; prior[~known] = 1/3
    node = np.log(np.clip((1-alpha)*prior+alpha*probs, 1e-8, 1))
    return decode(node, z[:, 2:])
