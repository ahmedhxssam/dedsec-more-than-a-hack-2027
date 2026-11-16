"""Two bounded hypotheses; no reference labels, fixture IDs or timestamps.

The frozen second-pass decoder stays unchanged. These experimental functions
consume its probabilities plus source tracking and already encoded scene evidence.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .contextual import binary_decode, velocity


def safe_contact_logit(features, owner):
    """Unknown player velocity must not erase an otherwise observed redirection.

    Missing relative motion supplies zero co-motion support. It does not make
    the baseline proximity term or independently finite ball acceleration NaN.
    """
    x = features['owner'][:, :, :9]
    distance, relative, change = x[:, :, 0], x[:, :, 3], x[:, :, 7]
    near = np.exp(-np.minimum(distance, 20) ** 2 / (2 * 1.5 ** 2))
    co_motion = 1 / (1 + np.exp(np.clip((relative - 4) / 1.5, -30, 30)))
    redirect = np.clip(change / 40, 0, 1) * np.exp(-np.minimum(distance, 20) ** 2 / 2)
    quality = np.nan_to_num(near * (.2 + .8 * np.nan_to_num(co_motion))
                            + .75 * np.nan_to_num(redirect), nan=0.)
    model_log = np.log(np.clip(owner[:, 1], .01, .99) / np.clip(owner[:, 0], .01, .99))
    logit = model_log * np.minimum(quality.max(axis=1), 1.) + 2 * (quality[:, 1] - quality[:, 0])
    logit[~features['ball_seen']] = 0
    return logit.astype(np.float32), quality.astype(np.float32)


def touch_support(ball, players, team, fps=25):
    """Approach and departure from the SAME anonymous player slot, +/-0.24s.

    A nearby opponent suppresses support. A straight fly-by has no velocity
    change support. No possession is pruned or required to last this long: the
    temporal context describes the ball trajectory around an instantaneous touch.
    Slot swaps and projected ball errors remain limitations, not solved tracking.
    """
    n = len(ball)
    step = max(1, round(.24 * fps))
    out = np.zeros((n, 2), np.float32)
    if n <= 2 * step:
        return out
    rows = np.arange(step, n - step)
    distance = np.linalg.norm(players - ball[:, None], axis=2)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    near = np.stack([distance[:, team == t].min(axis=1) for t in [0, 1]], axis=1)
    incoming = (ball[rows] - ball[rows - step]) * fps / step
    outgoing = (ball[rows + step] - ball[rows]) * fps / step
    bend = np.clip(np.linalg.norm(outgoing - incoming, axis=1) / 3., 0, 1)
    for t in [0, 1]:
        slots = np.flatnonzero(team == t)
        slot = slots[distance[rows][:, slots].argmin(axis=1)]
        d = distance[rows, slot]
        before = np.linalg.norm(ball[rows - step] - players[rows - step, slot], axis=1)
        after = np.linalg.norm(ball[rows + step] - players[rows + step, slot], axis=1)
        exclusive = np.clip((near[rows, 1 - t] - d) / .75, 0, 1)
        value = (np.exp(-d ** 2 / 2) * exclusive * np.clip((before - d) / .5, 0, 1)
                 * np.clip((after - d) / .5, 0, 1) * bend)
        # A discontinuous slot cannot establish a contact.
        continuity = np.maximum(np.linalg.norm(players[rows, slot] - players[rows - step, slot], axis=1),
                                np.linalg.norm(players[rows + step, slot] - players[rows, slot], axis=1))
        value[continuity > 12 * step / fps] = 0
        out[rows, t] = np.nan_to_num(value, nan=0., posinf=0., neginf=0.)
    return out


def touch_decode(logit, support, cost=6.):
    """Only destination-supported contact can reduce the existing switch cost.

    Cost remains in [cost/2, cost]; isolated weak evidence still has to pay for
    both entry and exit. No global threshold change or duration restriction.
    """
    n = len(logit)
    if not n:
        return np.empty(0, np.int8)
    costs = cost / (1 + np.clip(support, 0, 1))
    back = np.zeros((n, 2), np.int8)
    a, b = 0., float(logit[0])
    for i in range(1, n):
        aa = a
        if b - costs[i, 0] > a:
            aa = b - costs[i, 0]
            back[i, 0] = 1
        bb = b
        back[i, 1] = 1
        if a - costs[i, 1] > b:
            bb = a - costs[i, 1]
            back[i, 1] = 0
        a, b = aa, bb + float(logit[i])
        m = max(a, b)
        a -= m
        b -= m
    result = np.empty(n, np.int8)
    result[-1] = b > a
    for i in range(n - 1, 0, -1):
        result[i - 1] = back[i, result[i]]
    return result


def context_dead(base, player_dead, scene, quality, tags, use_scene=True):
    """Locally resolve head conflicts in sustained waiting/restart context.

    A seed requires >=.95 player-head support, >=.90 mean support over 1s,
    weak instantaneous interaction, and agreeing scene evidence before and after.
    Existing D support or an automatic set-piece tag relaxes only the weak-contact
    seed requirement. Tags never set a label or move a boundary by themselves.
    Connected expansion stops at player-head <=.5; unchanged dead DP resolves
    the resulting emissions. This may delay restarts and is tested as a hypothesis.
    """
    n = len(base)
    # These are summaries of individually encoded images, not video-model outputs.
    before = uniform_filter1d(scene.astype(float), 25, origin=12, mode='nearest')
    after = uniform_filter1d(scene.astype(float), 25, origin=-12, mode='nearest')
    visual = (before > 0) & (after > 0) if use_scene else np.ones(n, bool)
    sustained = uniform_filter1d(player_dead.astype(float), 25, mode='nearest') >= .90
    seed = ((player_dead >= .95) & sustained & visual
            & ((quality.max(axis=1) < .15) | (base > .5) | tags))
    allowed = (player_dead > .5) & visual
    edges = np.diff(np.r_[False, allowed, False].astype(np.int8))
    gate = np.zeros(n, bool)
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if seed[s:e].any():
            gate[s:e] = True
    result = base.copy()
    result[gate] = np.maximum(base[gate], player_dead[gate])
    return result, gate, seed, np.c_[before, after].astype(np.float32)


def decode(dead, logit, touch=None, team_cost=6., dead_cost=50.):
    stopped = binary_decode(np.log(np.clip(dead, .01, .99) / np.clip(1 - dead, .01, .99)), dead_cost)
    labels = np.full(len(dead), 2, np.int8)
    bounds = np.r_[0, np.flatnonzero(stopped[1:] != stopped[:-1]) + 1, len(dead)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if not stopped[s]:
            labels[s:e] = (binary_decode(logit[s:e], team_cost) if touch is None
                           else touch_decode(logit[s:e], touch[s:e], team_cost))
    return labels
