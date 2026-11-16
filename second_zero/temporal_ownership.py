"""Offline team control from ordered, uncertain trajectories, without contact gates.

The public inference API takes positions and team membership only. Provider
possession/status and prior predicted owners are deliberately absent. Channels
are shared across teams; swapping teams exchanges probabilities exactly.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit
import torch
from torch import nn
from .contextual import features as contextual_features, velocity

VERSION = 'temporal-ownership-v1'
DILATIONS = (1, 2, 4, 8, 16, 32)
RADIUS = sum(DILATIONS)


def trajectory_features(ball, players, team, pitch=(105.3, 68), fps=25, contextual=None):
    """Return [frame, perspective team, channel], in source order at 25 Hz.

    Every continuous channel has a separate missing indicator. Reliability is
    derived from measured availability/kinematic consistency, never a label or
    a claim of calibrated detector certainty. Three anonymous neighbors reduce
    reliance on a single fragile nearest-player association.
    """
    if fps != 25:
        raise ValueError('This frozen temporal configuration requires 25 Hz')
    ball = np.asarray(ball, np.float32)
    players = np.asarray(players, np.float32).copy()
    team = np.asarray(team)
    if ball.ndim != 2 or ball.shape[1] != 2 or players.shape[:1] != ball.shape[:1]:
        raise ValueError('Expected aligned frame-major metre trajectories')
    if set(np.unique(team)) != {0, 1}:
        raise ValueError('Both mapped teams are required')
    n = len(ball)
    f = contextual or contextual_features(ball, players, team, pitch, fps)
    pv, bv = velocity(players, fps=fps), velocity(ball, fps=fps)
    ps = np.linalg.norm(pv, axis=2)
    moving = np.linalg.norm(np.diff(players, axis=0, prepend=players[:1]), axis=2) > 1e-4
    usable = np.isfinite(players).all(2)
    usable &= uniform_filter1d(moving.astype(np.float32), 251, axis=0, mode='nearest') > .015
    usable &= (ps < 12) | ~np.isfinite(ps)
    players[~usable] = np.nan
    pv[~usable] = np.nan
    ps[~usable] = np.nan
    relpos = players - ball[:, None]
    dist = np.linalg.norm(relpos, axis=2)
    relvel = pv - bv[:, None]
    relative = np.linalg.norm(relvel, axis=2)
    radial = np.sum(relpos * relvel, axis=2) / np.maximum(dist, .1)
    ball_seen = np.isfinite(ball).all(1)
    ix = np.arange(n)
    # Time to the previous/next measured ball; bounded, bidirectional, no fill.
    prev = np.maximum.accumulate(np.where(ball_seen, ix, -250))
    nxt = np.minimum.accumulate(np.where(ball_seen, ix, n+250)[::-1])[::-1]
    gap = np.c_[np.minimum(ix-prev, 250), np.minimum(nxt-ix, 250)] / 250
    neighborhoods = []
    for t in (0, 1):
        slots = np.flatnonzero(team == t)
        order = np.argsort(np.nan_to_num(dist[:, slots], nan=np.inf), axis=1)[:, :3]
        selected = slots[order]
        rows = ix[:, None]
        associated = np.isfinite(dist[rows, selected])
        vals = [dist[rows, selected]/20, relative[rows, selected]/30,
                radial[rows, selected]/30, ps[rows, selected]/12]
        # A change of selected anonymous slot is uncertain association, not a
        # false player velocity: derivatives above always use the original slot.
        stable = np.full((n, 3), np.nan, np.float32)
        stable[1:] = (selected[1:] == selected[:-1])
        stable[1:][~associated[:-1]] = np.nan
        vals.append(stable)
        for value in vals:
            value[~associated] = np.nan
        td = dist[:, slots]
        count = np.sum(np.isfinite(td), axis=1)
        near = np.c_[(td < 2).sum(1), (td < 5).sum(1), (td < 10).sum(1)] / 11
        near[count == 0] = np.nan
        coverage = usable[:, slots].sum(1, keepdims=True) / 11
        neighborhoods.append(np.c_[*vals, near, coverage])
    scales = np.array([20,20,20,30,30,12,40,100,1], np.float32)
    output = []
    for t in (0, 1):
        base = f['owner'][:, t, :9] / scales
        base[~np.isfinite(base[:,0]),3] = np.nan
        base[~np.isfinite(base[:,0]),5] = np.nan
        base[~np.isfinite(base[:,1]),4] = np.nan
        values = np.c_[base, neighborhoods[t], neighborhoods[1-t], gap]
        missing = ~np.isfinite(values)
        # Fixed physical scales, no statistics fitted on target/dev matches.
        values = np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0), -5, 5)
        output.append(np.c_[values, missing.astype(np.float32)])
    return np.stack(output, axis=1).astype(np.float32)


class TemporalOwner(nn.Module):
    """127-frame receptive field, no stride/pooling or minimum-duration loss."""
    def __init__(self, channels, width=32):
        super().__init__()
        self.input = nn.Conv1d(channels, width, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d)
                                     for d in DILATIONS])
        self.output = nn.Conv1d(width, 1, 1)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        # batch, team, channel, frame -> one anti-symmetric A/H logit.
        b, teams, c, n = x.shape
        z = torch.nn.functional.gelu(self.input(x.reshape(b*teams, c, n)))
        for layer in self.blocks:
            z = z + .5*self.dropout(torch.nn.functional.gelu(layer(z)))
        z = self.output(z).reshape(b, teams, n)
        return z[:, 1] - z[:, 0]


def infer_logits(model, x, chunk=4096):
    """Exact overlap-save inference, independent of chunk boundaries."""
    model.eval()
    padded = np.pad(x, ((RADIUS,RADIUS),(0,0),(0,0)))
    out = np.empty(len(x), np.float32)
    with torch.inference_mode():
        for s in range(0, len(x), chunk):
            e = min(s+chunk, len(x))
            batch = torch.from_numpy(padded[s:e+2*RADIUS].transpose(1,2,0)[None].copy())
            out[s:e] = model(batch)[0,RADIUS:RADIUS+e-s].cpu().numpy()
    return out


def probabilities(logit, temperature=1.):
    p = expit(np.asarray(logit, dtype=np.float64)/temperature)
    return np.c_[1-p, p]


def decode_fixed_live(logit, labels, cost=6.):
    """Replace H/A only; keep every existing D/X and exact island boundary."""
    from .contextual import binary_decode
    result = np.asarray(labels).copy()
    live = result < 2
    bounds = np.r_[0, np.flatnonzero(live[1:] != live[:-1])+1, len(live)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if live[s]:
            result[s:e] = binary_decode(np.asarray(logit)[s:e], cost)
    return result


def evidence_metrics(y, logit, mask=None):
    valid = np.asarray(y) < 2
    if mask is not None:
        valid &= mask
    y, z = np.asarray(y)[valid], np.asarray(logit)[valid]
    if not len(y):
        return {'n': 0}
    p = expit(z)
    correct = (z > 0) == y
    confidence = np.maximum(p, 1-p)
    ece = 0.
    for lo, hi in zip(np.linspace(.5,1,11)[:-1], np.linspace(.5,1,11)[1:]):
        take = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if take.any():
            ece += take.mean()*abs(confidence[take].mean()-correct[take].mean())
    return {'n': len(y), 'accuracy': float(correct.mean()),
            'nll': float(np.mean(np.logaddexp(0,z)-y*z)),
            'brier': float(np.mean((p-y)**2)), 'ece_10_equal_confidence_bins': float(ece)}
