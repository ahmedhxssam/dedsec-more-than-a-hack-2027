"""Bounded stoppage endpoint localization without deleting or splitting proposals.

The localizer sees only continuous tracking values/availability and the proposed
endpoint offset. Training proposals are generated out of fold by whole match.
No labels, fixture IDs, event metadata, or global minimum duration enter inference.
"""
from __future__ import annotations

import numpy as np

VERSION = 'overnight-boundaries-v1'
MAX_RADIUS = 125
CONTEXT_WIDTHS = (6, 25, 75)


def dead_intervals(labels):
    d = np.asarray(labels) == 2
    edges = np.diff(np.r_[False, d, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def proposal_endpoints(labels, radius):
    """One onset/restart per D span, bounded by cores, X and live-gap midpoints.

    The central half of short D spans and all frames >radius inside long spans
    are protected. Every original live gap keeps at least one frame, including
    one-frame possessions. Edges adjacent to X or sequence ends are not inferred.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('Expected a one-dimensional H/A/D/X label array')
    if not 1 <= radius <= MAX_RADIUS:
        raise ValueError('Boundary radius must be between 1 and 125 frames')
    spans = dead_intervals(labels)
    rows = []
    for index, (start, stop) in enumerate(spans):
        margin = min(radius, (stop - start - 1) // 4)
        if start > 0 and labels[start - 1] < 2:
            prev = start - 1
            while prev > max(0, start - radius) and labels[prev - 1] < 2:
                prev -= 1
            lo = max(prev, start - radius)
            if index and spans[index - 1][1] >= prev:
                lo = max(lo, (spans[index - 1][1] + start) // 2 + 1)
            rows.append({'interval': index, 'kind': 'onset', 'old': int(start),
                         'lo': int(lo), 'hi': int(start + margin),
                         'span': [int(start), int(stop)]})
        if stop < len(labels) and labels[stop] < 2:
            nxt = stop
            while nxt < min(len(labels) - 1, stop + radius) and labels[nxt + 1] < 2:
                nxt += 1
            hi = min(nxt + 1, stop + radius)
            if index + 1 < len(spans) and spans[index + 1][0] <= nxt + 1:
                hi = min(hi, (stop + spans[index + 1][0]) // 2)
            rows.append({'interval': index, 'kind': 'restart', 'old': int(stop),
                         'lo': int(stop - margin), 'hi': int(hi),
                         'span': [int(start), int(stop)]})
    return rows


class BoundaryFeatures:
    """Prefix sums provide identical before/after features in fitting and inference."""
    def __init__(self, trajectory):
        x = np.asarray(trajectory, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 40 or not np.isfinite(x).all():
            raise ValueError('Expected finite 40-channel trajectory features')
        self.x = x
        self.prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)])

    def rows(self, candidates, proposal):
        candidates = np.asarray(candidates, dtype=int)
        if np.any(candidates < 1) or np.any(candidates >= len(self.x)):
            raise ValueError('Boundary candidate outside the observable timeline')
        result = []
        for width in CONTEXT_WIDTHS:
            lo, hi = np.maximum(0, candidates - width), np.minimum(len(self.x), candidates + width)
            before = (self.prefix[candidates] - self.prefix[lo]) / (candidates - lo)[:, None]
            after = (self.prefix[hi] - self.prefix[candidates]) / (hi - candidates)[:, None]
            result.extend([before, after, after - before])
        result.append(((candidates - proposal) / MAX_RADIUS)[:, None])
        return np.concatenate(result, axis=1).astype(np.float32)

    def supported(self, point):
        lo, hi = max(0, point - 25), min(len(self.x), point + 25)
        # Missing ball does not imply stopped; either actual ball or player
        # observations can support an event hypothesis.
        return bool(np.maximum(self.x[lo:hi, 36], self.x[lo:hi, 37]).mean() >= .25)


def propose_moves(models, x, incumbent, radii):
    extractor = BoundaryFeatures(x)
    all_rows = []
    for kind in ['onset', 'restart']:
        for endpoint in proposal_endpoints(incumbent, radii[kind]):
            if endpoint['kind'] != kind:
                continue
            old = endpoint['old']
            lo, hi = max(1, endpoint['lo']), min(len(x) - 1, endpoint['hi'])
            candidates = np.arange(lo, hi + 1)
            values = models[kind].predict_proba(extractor.rows(candidates, old))[:, 1]
            peak = int(np.argmax(values))
            # Ties retain the nearest endpoint rather than an arbitrary early edge.
            ties = np.flatnonzero(np.isclose(values, values[peak], rtol=0, atol=1e-10))
            peak = int(ties[np.argmin(np.abs(candidates[ties] - old))])
            new = int(candidates[peak])
            old_probability = float(values[np.flatnonzero(candidates == old)[0]])
            gain = max(0., float(values[peak]) - old_probability)
            confidence = float(values[peak]) * gain
            all_rows.append({**endpoint, 'new': new, 'probability': float(values[peak]),
                             'old_probability': old_probability, 'confidence': confidence,
                             'supported': extractor.supported(new)})
    return all_rows


def apply_moves(incumbent, moves, thresholds):
    """Apply only accepted endpoints; no new stoppage and no live hole is possible."""
    incumbent = np.asarray(incumbent)
    spans = dead_intervals(incumbent)
    adjusted = [list(span) for span in spans]
    decisions = []
    for row in moves:
        accepted = (row['new'] != row['old'] and row['supported'] and
                    row['confidence'] > thresholds[row['kind']])
        if not row['lo'] <= row['new'] <= row['hi']:
            raise ValueError('Proposed move escaped its frozen neighborhood')
        if accepted:
            adjusted[row['interval']][0 if row['kind'] == 'onset' else 1] = row['new']
        decisions.append({**row, 'accepted': bool(accepted)})
    out = incumbent.copy()
    out[out == 2] = 0
    for i, (start, stop) in enumerate(adjusted):
        if start >= stop or (i and adjusted[i - 1][1] >= start):
            raise ValueError('Boundary adjustments would erase a D span or live gap')
        if np.any(incumbent[start:stop] == 3):
            raise ValueError('Boundary adjustment crosses an X region')
        out[start:stop] = 2
    out[incumbent == 3] = 3
    if len(dead_intervals(out)) != len(spans):
        raise AssertionError('Stoppage proposal count changed')
    return out, decisions


def reference_boundaries(y, kind):
    y = np.asarray(y)
    if kind == 'onset':
        return np.flatnonzero((y[:-1] < 2) & (y[1:] == 2)) + 1
    if kind == 'restart':
        return np.flatnonzero((y[:-1] == 2) & (y[1:] < 2)) + 1
    raise ValueError('Unknown endpoint kind')
