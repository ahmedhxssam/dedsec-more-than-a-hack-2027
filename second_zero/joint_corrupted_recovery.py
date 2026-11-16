"""Raw-only joint H/A/dead recovery, independent of the incumbent gap graph."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from scipy.ndimage import gaussian_filter1d

STRIDE = 5


def failure_view(raw, seed):
    """Label-independent missing ball, stale tracks, slot switches and outliers.

    These are simulated failures, not an empirically fitted failure distribution.
    Every retained position originates in the training match only.
    """
    if set(raw) != {'ball', 'players', 'team', 'pitch'}:
        raise ValueError('Raw observations only')
    rng = np.random.default_rng(seed)
    out = {k: np.array(v, copy=True) for k, v in raw.items()}
    n = len(out['ball'])
    for start in range(int(rng.integers(0, 250)), n, 500):
        stop = min(n, start + int(rng.integers(50, 401)))
        if rng.random() < .75:
            out['ball'][start:stop] = np.nan
        elif start:
            out['ball'][start:stop] = out['ball'][start - 1]
    out['ball'] += gaussian_filter1d(rng.normal(0, .75, out['ball'].shape).astype(np.float32), 3, axis=0)
    for slot in range(out['players'].shape[1]):
        for start in range(int(rng.integers(0, 1200)), n, 2000):
            stop = min(n, start + int(rng.integers(25, 201)))
            if start and rng.random() < .5:
                out['players'][start:stop, slot] = out['players'][start - 1, slot]
            else:
                out['players'][start:stop, slot] = np.nan
    # Same-team slot switches cannot change team ownership or introduce coordinates.
    for side in np.unique(out['team']):
        slots = np.flatnonzero(out['team'] == side)
        if len(slots) < 2:
            continue
        for start in range(int(rng.integers(0, 3000)), n, 6000):
            a, b = rng.choice(slots, 2, replace=False)
            out['players'][start:, [a, b]] = out['players'][start:, [b, a]]
    return out


def targets(truth):
    y = np.asarray(truth, np.int64)
    if y.ndim != 1 or not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError('Expected native H/A/D/X codes')
    weights = np.zeros(len(y), np.float32)
    bounds = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if y[start] < 3:
            weights[start:stop] = np.clip(np.sqrt(125 / max(1, stop - start)), .25, 4.)
    yd = y[::STRIDE].copy()
    edge = np.zeros(len(yd), np.float32)
    valid = yd < 3
    edge[1:] = ((yd[1:] != yd[:-1]) & valid[1:] & valid[:-1]).astype(np.float32)
    events = np.flatnonzero(edge)
    return {'y': yd, 'weight': weights[::STRIDE], 'edge': edge, 'events': events}


class JointRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(414), nn.Linear(414, 48), nn.GELU())
        self.memory = nn.GRU(48, 48, batch_first=True, bidirectional=True)
        self.owner = nn.Linear(96, 1)
        self.dead = nn.Linear(96, 1)
        self.change = nn.Linear(96, 1)

    def forward(self, x):
        h, _ = self.memory(self.input(x.transpose(1, 2)))
        return tuple(head(h).squeeze(-1) for head in (self.owner, self.dead, self.change))


def emissions(owner, dead):
    # The first perspective's positive ownership logit denotes A, as in v20.
    live = -nn.functional.softplus(dead)
    return torch.stack([live - nn.functional.softplus(owner),
                        live - nn.functional.softplus(-owner),
                        -nn.functional.softplus(-dead)], -1)


def path_nll(logp, change, truth, mask):
    """Exact three-state conditional path likelihood, resetting at ungraded X."""
    active = mask.bool()
    columns = torch.nonzero(active.any(0)).flatten()
    if not len(columns):
        return logp.sum() * 0
    lo, hi = int(columns[0]), int(columns[-1]) + 1
    logp, change, truth, active = (v[:, lo:hi] for v in (logp, change, truth, active))
    alpha = logp.new_zeros((len(logp), 3))
    score = logp.new_zeros(len(logp))
    total = logp.new_zeros(len(logp))
    previous = torch.zeros(len(logp), dtype=torch.bool, device=logp.device)
    last = torch.zeros(len(logp), dtype=torch.long, device=logp.device)
    eye = torch.eye(3, dtype=torch.bool, device=logp.device)[None]
    for i in range(logp.shape[1]):
        take, cont = active[:, i], active[:, i] & previous
        stay = -nn.functional.softplus(change[:, i])
        switch = -nn.functional.softplus(-change[:, i]) - np.log(2.)
        transition = torch.where(eye, stay[:, None, None], switch[:, None, None])
        moved = torch.logsumexp(alpha[:, :, None] + transition, 1)
        next_alpha = logp[:, i] + torch.where(cont[:, None], moved, torch.zeros_like(moved))
        yi = truth[:, i].long().clamp(0, 2)
        next_score = logp[:, i].gather(1, yi[:, None]).squeeze(1)
        next_score += torch.where(cont, score + torch.where(yi == last, stay, switch), torch.zeros_like(score))
        total += torch.where(previous & ~take, torch.logsumexp(alpha, 1) - score, torch.zeros_like(score))
        alpha = torch.where(take[:, None], next_alpha, torch.zeros_like(alpha))
        score = torch.where(take, next_score, torch.zeros_like(score))
        last, previous = yi, take
    total += torch.where(previous, torch.logsumexp(alpha, 1) - score, torch.zeros_like(score))
    return total.sum() / active.sum().clamp_min(1)


def batch(dataset, rng, batch_size=6, length=256):
    values = {k: [] for k in ['x', 'y', 'weight', 'edge']}
    for _ in range(batch_size):
        row = dataset[int(rng.integers(len(dataset)))]
        known = np.flatnonzero(row['y'] < 3)
        centers = row['events'] if rng.random() < .5 and len(row['events']) else known
        center = int(rng.choice(centers))
        ix = np.arange(center - length // 2, center + length // 2)
        valid = (ix >= 0) & (ix < len(row['y']))
        ii = ix.clip(0, len(row['y']) - 1)
        swap = bool(rng.integers(2))
        x = row['x'][int(swap)][:, ii].astype(np.float32)
        x[:, ~valid] = 0
        y = row['y'][ii].copy()
        if swap:
            live = y < 2
            y[live] = 1 - y[live]
        w = row['weight'][ii].copy() * valid
        w[:64] = 0
        w[-64:] = 0
        for k, v in [('x', x), ('y', y), ('weight', w), ('edge', row['edge'][ii].copy())]:
            values[k].append(v)
    return {k: torch.from_numpy(np.stack(v)) for k, v in values.items()}


def loss(model, v):
    owner, dead, change = model(v['x'])
    logp = emissions(owner, dead)
    w = v['weight']
    known = (w > 0) & (v['y'] < 3)
    yi = v['y'].long().clamp(0, 2)
    error = -logp.gather(2, yi[..., None]).squeeze(-1)
    frame = .5 * (error * w).sum() / w.sum().clamp_min(1.)
    frame += .5 * error[known].mean()
    # Exact change positions, not contact labels or a smoothed event surrogate.
    edge_mask = known.clone()
    edge_mask[:, 0] = False
    edge_mask[:, 1:] &= v['y'][:, :-1] < 3
    boundary = nn.functional.binary_cross_entropy_with_logits(change, v['edge'], reduction='none', pos_weight=change.new_tensor(5.))
    boundary = boundary[edge_mask].mean() if edge_mask.any() else boundary.sum() * 0
    structured = path_nll(logp, change - np.log(5.), yi, known)
    return frame + .1 * boundary + .1 * structured + .0001 * (owner.square() + dead.square())[known].mean()


@torch.no_grad()
def predict(model, x, n):
    model.eval()
    size = x.shape[-1]
    result = np.empty((size, 3), np.float32)
    for start in range(0, size, 128):
        ix = np.arange(start - 64, start + 192)
        valid = (ix >= 0) & (ix < size)
        crop = x[:, :, ix.clip(0, size - 1)].astype(np.float32)
        crop[:, :, ~valid] = 0
        owner, dead, change = model(torch.from_numpy(crop))
        owner = .5 * (owner[0] - owner[1])
        dead = .5 * (dead[0] + dead[1])
        change = .5 * (change[0] + change[1]) - np.log(5.)
        stop = min(size, start + 128)
        result[start:stop] = torch.stack([owner, dead, change], -1)[64:64 + stop - start].numpy()
    return np.stack([np.interp(np.arange(n), np.arange(size) * STRIDE, result[:, i]) for i in range(3)], -1).astype(np.float32)


def decode(logp, change):
    """Three-state Viterbi with native-frame emissions scaled to training stride."""
    logp, change = np.asarray(logp), np.asarray(change)
    if logp.shape != (len(change), 3) or not len(change):
        raise ValueError('Aligned nonempty state and transition scores required')
    back = np.zeros((len(change), 3), np.int8)
    score = logp[0].copy() / STRIDE
    eye = np.eye(3, dtype=bool)
    for i in range(1, len(change)):
        stay = -np.logaddexp(0, change[i])
        switch = -np.logaddexp(0, -change[i]) - np.log(2.)
        moved = score[:, None] + np.where(eye, stay, switch)
        back[i] = np.argmax(moved, axis=0)
        score = moved[back[i], np.arange(3)] + logp[i] / STRIDE
    out = np.empty(len(change), np.int8)
    out[-1] = np.argmax(score)
    for i in range(len(change) - 1, 0, -1):
        out[i - 1] = back[i, out[i]]
    return out


def compose(parent, logits, alpha, change_offset):
    parent = np.asarray(parent)
    if alpha == 0:
        return parent.copy()
    z = torch.from_numpy(np.asarray(logits, np.float32))
    probs = emissions(z[:, 0], z[:, 1]).exp().numpy()
    # X carries no reliable state prior and is deliberately editable here.
    prior = np.full_like(probs, .01)
    live = parent < 3
    prior[np.flatnonzero(live), parent[live]] = .98
    prior[~live] = 1 / 3
    logp = np.log(np.clip((1 - alpha) * prior + alpha * probs, 1e-8, 1))
    return decode(logp, logits[:, 2] + change_offset)
