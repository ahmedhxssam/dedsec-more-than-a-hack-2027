"""Geometry-conditioned H/A transition costs with immutable unary evidence.

Boundary features are team invariant and use observed trajectory channels only.
The exact two-state decoder keeps every duration possible and every D/X fixed.
"""
from __future__ import annotations
import numpy as np

VERSION = 'extended-sequence-v1'
WIDTHS = (6, 25)
# Own neighborhood, contextual physics and gap ages; omit duplicate opponent
# neighborhood. Each selected value keeps its original missing indicator.
VALUE_CHANNELS = np.r_[np.arange(28), 47, 48]
CHANNELS = np.r_[VALUE_CHANNELS, VALUE_CHANNELS + 49]


class OwnershipBoundaryFeatures:
    def __init__(self, trajectory):
        x = np.asarray(trajectory)
        if x.ndim != 3 or x.shape[1:] != (2, 98):
            raise ValueError('Expected n by 2 by 98 frozen trajectory features')
        x = x[:, :, CHANNELS].astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Boundary features must be finite')
        self.n = len(x)
        self.prefix = np.concatenate([np.zeros((1, 2, len(CHANNELS))), np.cumsum(x, axis=0)])

    def rows(self, points):
        points = np.asarray(points, dtype=np.int64)
        if np.any(points < 1) or np.any(points >= self.n):
            raise ValueError('Boundary edge must have observations on both sides')
        rows = []
        for width in WIDTHS:
            lo, hi = np.maximum(0, points-width), np.minimum(self.n, points+width)
            a = (self.prefix[points]-self.prefix[lo])/(points-lo)[:, None, None]
            b = (self.prefix[hi]-self.prefix[points])/(hi-points)[:, None, None]
            da, db = a[:, 1]-a[:, 0], b[:, 1]-b[:, 0]
            # Products retain sign reversal evidence while exchanging team names
            # cannot change any boundary feature or boundary prediction.
            rows.extend([a.mean(1), b.mean(1), np.abs(da), np.abs(db),
                         da*db, np.abs(b-a).mean(1)])
        return np.concatenate(rows, axis=1).astype(np.float32)


def boundary_target(labels, radius=3):
    y = np.asarray(labels)
    valid = np.r_[False, (y[1:] < 2) & (y[:-1] < 2)]
    edges = np.flatnonzero(valid & np.r_[False, y[1:] != y[:-1]])
    target = np.zeros(len(y), bool)
    for edge in edges:
        target[max(1, edge-radius):min(len(y), edge+radius+1)] = True
    return target & valid, valid


def boundary_scores(model, trajectory, chunk=4096):
    features = OwnershipBoundaryFeatures(trajectory)
    scores = np.zeros(features.n, np.float64)
    for start in range(1, features.n, chunk):
        points = np.arange(start, min(start+chunk, features.n))
        scores[points] = model.predict_proba(features.rows(points))[:, 1]
    return scores


def transition_costs(boundary_score, alpha, touch_support=None, cost=6.):
    q = np.asarray(boundary_score, np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Expected finite boundary scores in [0,1]')
    if not 0 <= alpha <= .75:
        raise ValueError('Frozen bounded discount requires alpha in [0,.75]')
    costs = np.full((len(q), 2), cost, np.float64)
    if touch_support is not None:
        touch = np.asarray(touch_support)
        if touch.shape != costs.shape or not np.isfinite(touch).all():
            raise ValueError('Touch support must be finite n by 2')
        costs /= 1 + np.clip(touch, 0, 1)
    support = np.maximum(2*q-1, 0)
    return costs * (1-alpha*support[:, None])


def dynamic_decode(logit, costs):
    x, costs = np.asarray(logit), np.asarray(costs)
    if x.ndim != 1 or costs.shape != (len(x), 2):
        raise ValueError('Expected aligned logit and destination transition costs')
    if not np.isfinite(x).all() or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError('Expected finite evidence and nonnegative costs')
    n = len(x)
    if not n:
        return np.empty(0, np.int8)
    back = np.zeros((n, 2), np.int8)
    a, b = 0., float(x[0])
    for i in range(1, n):
        aa, bb = a, b
        back[i, 1] = 1
        if b-costs[i, 0] > a:
            aa = b-costs[i, 0]; back[i, 0] = 1
        if a-costs[i, 1] > b:
            bb = a-costs[i, 1]; back[i, 1] = 0
        a, b = aa, bb+float(x[i])
        maximum = max(a, b); a -= maximum; b -= maximum
    result = np.empty(n, np.int8); result[-1] = b > a
    for i in range(n-1, 0, -1):
        result[i-1] = back[i, result[i]]
    return result


def decode_fixed_live(logit, baseline, boundary_score, alpha, touch_support=None):
    base, z = np.asarray(baseline), np.asarray(logit)
    if base.ndim != 1 or z.shape != base.shape or not np.isin(base, [0, 1, 2, 3]).all():
        raise ValueError('Invalid aligned fixed labels and ownership evidence')
    costs = transition_costs(boundary_score, alpha, touch_support)
    if costs.shape[0] != len(base):
        raise ValueError('Boundary score length differs from baseline')
    out = base.copy(); live = base < 2
    boundaries = np.r_[0, np.flatnonzero(live[1:] != live[:-1])+1, len(base)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if live[start]:
            out[start:stop] = dynamic_decode(z[start:stop], costs[start:stop])
    return out
