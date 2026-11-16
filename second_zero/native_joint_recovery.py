"""Native-25Hz three-state recovery with a frozen cohort-matched ownership context."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from second_zero import joint_corrupted_recovery as old
from second_zero.temporal_ownership import trajectory_features as owner_features
from second_zero.temporal_live_dead import trajectory_features as status_features

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RADIUS = 2 * sum(DILATIONS)
LENGTH = 3072
MARGIN = 1024
CENTRAL = 1024
CHANNELS = 140
steps_per_match = 2000
compose = old.compose


def inputs(raw):
    if set(raw) != {'ball', 'players', 'team', 'pitch'}:
        raise ValueError('Raw observations only')
    owner = owner_features(raw['ball'], raw['players'], raw['team'], raw['pitch'])
    status = status_features(raw['ball'], raw['players'], raw['team'], raw['pitch'])
    assert owner.shape[1:] == (2, 98) and status.shape == (len(owner), 40)
    x = np.concatenate([owner[:, [1, 0]], np.broadcast_to(status[:, None], (len(owner), 2, 40))], axis=-1)
    x = x.transpose(1, 2, 0).astype(np.float16)
    assert np.isfinite(x).all()
    return x


def targets(truth):
    y = np.asarray(truth, np.int64)
    if y.ndim != 1 or not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError('Expected native H/A/D/X labels')
    weight = np.zeros(len(y), np.float32)
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    for a, b in zip(edges[:-1], edges[1:]):
        if y[a] < 3:
            weight[a:b] = np.clip(np.sqrt(125 / max(1, b-a)), .25, 4.)
    boundary = np.zeros(len(y), np.float32)
    boundary[1:] = ((y[1:] != y[:-1]) & (y[1:] < 3) & (y[:-1] < 3))
    return {'y': y, 'weight': weight, 'edge': boundary, 'events': np.flatnonzero(boundary)}


class JointRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(CHANNELS), nn.Linear(CHANNELS, 64), nn.GELU())
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(64, 64, 5, padding=2*d, dilation=d, groups=64), nn.GELU(),
            nn.Conv1d(64, 64, 1), nn.Dropout(.1)) for d in DILATIONS])
        self.output = nn.Conv1d(64, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        z = self.input(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            z = z + .5 * block(z)
        delta = self.output(z)
        return x[:, -2] + delta[:, 0], delta[:, 1], x[:, -1] + delta[:, 2]


def joined(native, coarse, swap):
    # Coarse context contains v20 antisymmetric owner and symmetric boundary.
    context = np.asarray(coarse, np.float32).T.copy()
    if swap:
        context[0] *= -1
    return np.concatenate([native[int(swap)].astype(np.float32), context], axis=0)


def batch(dataset, rng, batch_size=2):
    rows = {k: [] for k in ('x', 'y', 'weight', 'edge')}
    for _ in range(batch_size):
        row = dataset[int(rng.integers(len(dataset)))]
        known = np.flatnonzero(row['y'] < 3)
        centers = row['events'] if rng.random() < .5 and len(row['events']) else known
        center = int(rng.choice(centers))
        ix = np.arange(center - LENGTH//2, center + LENGTH//2)
        valid = (ix >= 0) & (ix < len(row['y']))
        ii = ix.clip(0, len(row['y']) - 1)
        swap = bool(rng.integers(2))
        x = joined(row['x'][:, :, ii], row['coarse'][ii], swap)
        x[:, ~valid] = 0
        y = row['y'][ii].copy()
        if swap:
            live = y < 2
            y[live] = 1 - y[live]
        weight = row['weight'][ii].copy() * valid
        weight[:MARGIN] = 0
        weight[-MARGIN:] = 0
        for key, value in [('x', x), ('y', y), ('weight', weight), ('edge', row['edge'][ii])]:
            rows[key].append(value)
    return {k: torch.from_numpy(np.stack(v)) for k, v in rows.items()}


def loss(model, v):
    owner, dead, change = model(v['x'])
    logp = old.emissions(owner, dead)
    w = v['weight']
    known = (w > 0) & (v['y'] < 3)
    y = v['y'].clamp(0, 2).long()
    error = -logp.gather(2, y[..., None]).squeeze(-1)
    frame = .5 * (error * w).sum() / w.sum().clamp_min(1.) + .5 * error[known].mean()
    mask = known.clone()
    mask[:, 0] = False
    mask[:, 1:] &= v['y'][:, :-1] < 3
    boundary = nn.functional.binary_cross_entropy_with_logits(change, v['edge'], reduction='none', pos_weight=change.new_tensor(25.))
    boundary = boundary[mask].mean() if mask.any() else boundary.sum() * 0
    # The full 25Hz frame and edge tasks retain short spells. A 5Hz structured
    # regularizer controls long paths without reducing the input/output rate.
    structured = old.path_nll(logp[:, ::5], change[:, ::5] - np.log(5.), y[:, ::5], known[:, ::5])
    return frame + .1 * boundary + .1 * structured + .0001 * (owner.square() + dead.square())[known].mean()


@torch.no_grad()
def predict(model, native, coarse, chunk=4096):
    model.eval()
    n = native.shape[-1]
    result = np.empty((n, 3), np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        lo, hi = max(0, a - RADIUS), min(n, b + RADIUS)
        x = np.stack([joined(native[:, :, lo:hi], coarse[lo:hi], swap) for swap in (False, True)])
        owner, dead, change = model(torch.from_numpy(x))
        out = torch.stack([.5*(owner[0] - owner[1]), .5*(dead[0] + dead[1]),
                           .5*(change[0] + change[1]) - np.log(5.)], -1).numpy()
        result[a:b] = out[a-lo:b-lo]
    return result
