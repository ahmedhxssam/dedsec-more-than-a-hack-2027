"""Ordered live/dead evidence from uncertain observations, independent of ownership.

No status, events, predicted owners or public timestamps enter the network.
Motion is a feature, never a restart rule. Entirely unobserved receptive fields
produce neutral evidence so missingness alone cannot force a dead decision.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from .contextual import features as contextual_features, binary_decode
from .focused_boundaries import touch_decode

VERSION = 'temporal-live-dead-v1'
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
RADIUS = sum(DILATIONS)
VALUE_COUNT = 18
BALL_VALUES = slice(13, 18)
BALL_MISSING = slice(31, 36)
BALL_SEEN, PLAYER_COVERAGE, PREV_GAP, NEXT_GAP = 36, 37, 38, 39
CHANNELS = 40


def trajectory_features(ball, players, team, pitch=(105.3, 68), fps=25):
    """[frame,channel]; anonymous team-neutral kinematics plus measured availability.

    Reuse the existing track filter and velocity estimates, but omit centered
    feature averages: the temporal network learns their order. Physical scales
    are fixed; no target-match normalization or label-derived feature is fitted.
    """
    if fps != 25:
        raise ValueError('Frozen model requires 25 Hz')
    ball = np.asarray(ball, np.float32)
    f = contextual_features(ball, players, team, pitch, fps)
    player = f['player'][:, :13]
    ball_features = f['dead'][:, 52:57]
    scales = np.array([12]*6 + [1, 1, 15, 1, 50, 35, 100]
                      + [60, 100, 35, 30, 60], np.float32)
    values = np.c_[player, ball_features] / scales
    missing = ~np.isfinite(values)
    values = np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0), -5, 5)
    # A physically implausible finite speed is not reliable measured motion.
    seen = f['ball_seen'] & np.isfinite(f['speed'])
    ix = np.arange(len(ball))
    prev = np.maximum.accumulate(np.where(seen, ix, -250))
    nxt = np.minimum.accumulate(np.where(seen, ix, len(ball)+250)[::-1])[::-1]
    coverage = np.where(f['observed_players'] >= 8,
                        np.minimum(f['observed_players']/22., 1.), 0.)
    return np.c_[values, missing.astype(np.float32), seen, coverage,
                 np.minimum(ix-prev, 250)/250.,
                 np.minimum(nxt-ix, 250)/250.].astype(np.float32)


def drop_ball(x):
    """Training augmentation: remove all ball-derived evidence in a sampled window.

    This supplies player-context training in BOTH states, without inventing
    restarts or pretending a synthesized trajectory is a ball measurement.
    """
    x = np.array(x, copy=True)
    x[..., BALL_VALUES] = 0
    x[..., BALL_MISSING] = 1
    x[..., BALL_SEEN] = 0
    x[..., PREV_GAP:NEXT_GAP+1] = 1
    return x


class TemporalLiveDead(nn.Module):
    """255-frame bidirectional receptive field, one output at each native frame."""
    def __init__(self, channels=CHANNELS, width=32):
        super().__init__()
        self.input = nn.Conv1d(channels, width, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d)
                                     for d in DILATIONS])
        self.output = nn.Conv1d(width, 1, 1)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        z = torch.nn.functional.gelu(self.input(x))
        for layer in self.blocks:
            z = z + .5*self.dropout(torch.nn.functional.gelu(layer(z)))
        z = self.output(z)[:, 0]
        support = torch.maximum(x[:, BALL_SEEN:BALL_SEEN+1],
                                x[:, PLAYER_COVERAGE:PLAYER_COVERAGE+1])
        supported = torch.nn.functional.max_pool1d(
            support, 2*RADIUS+1, stride=1, padding=RADIUS)[:, 0] > 0
        return torch.where(supported, z, torch.zeros_like(z))


def infer_logits(model, x, chunk=4096):
    model.eval()
    padded = np.pad(x, ((RADIUS, RADIUS), (0, 0)))
    result = np.empty(len(x), np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), chunk):
            stop = min(start+chunk, len(x))
            batch = torch.from_numpy(padded[start:stop+2*RADIUS].T[None].copy())
            result[start:stop] = model(batch)[0, RADIUS:RADIUS+stop-start].cpu().numpy()
    return result


def integrate(dead_logit, owner_logit, incumbent, regions, touch=None,
              known_dead=(), dead_cost=50., team_cost=6.):
    """Change only live/dead evidence; use incumbent ownership decoder verbatim.

    Labels on new live islands can change because island boundaries change.
    X and existing official half/pre-kickoff metadata remain fixed.
    """
    incumbent = np.asarray(incumbent)
    if (np.asarray(dead_logit).shape != incumbent.shape or
            np.asarray(owner_logit).shape != incumbent.shape or
            not np.isfinite(dead_logit).all() or not np.isfinite(owner_logit).all()):
        raise ValueError('Aligned finite dead and fixed-owner logits required')
    result = incumbent.copy()
    covered = np.zeros(len(result), bool)
    for start, stop in regions:
        covered[start:stop] = True
        # Match existing probability clipping before dead DP.
        result[start:stop] = np.where(binary_decode(
            np.clip(dead_logit[start:stop], -np.log(99), np.log(99)), dead_cost), 2, 0)
    for start, stop in known_dead:
        result[start:stop] = 2
    result[incumbent == 3] = 3
    if np.any((incumbent != 3) & ~covered):
        raise ValueError('Every inference-eligible frame must be covered')
    live = result < 2
    edges = np.r_[0, np.flatnonzero(live[1:] != live[:-1])+1, len(live)]
    for start, stop in zip(edges[:-1], edges[1:]):
        if live[start]:
            result[start:stop] = (binary_decode(owner_logit[start:stop], team_cost)
                                 if touch is None else
                                 touch_decode(owner_logit[start:stop], touch[start:stop], team_cost))
    return result
