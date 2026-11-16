"""Soft ownership anchors and two-sided candidate construction.

This module never reads references.  Anchors propose alternatives to a complete
sequence graph; they do not lock frames, remove intervals, or require a minimum
possession duration.  The final graph can reject every anchor-derived proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit
from second_zero.sprint_decoder import segment_decode


@dataclass(frozen=True)
class AnchorConfig:
    confidence: float = .9
    auxiliary_confidence: float = .75
    stable_frames: int = 9
    observation_window: int = 9
    observed_fraction: float = .5
    maximum_dead_probability: float = .15
    maximum_bracket: int = 750
    endpoint_radius: int = 63
    choices: int = 3
    separation: int = 5

    def to_dict(self):
        return asdict(self)


def stretches(labels):
    labels = np.asarray(labels)
    if not len(labels):
        return np.empty((0, 3), dtype=np.int64)
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return np.c_[edges[:-1], edges[1:], labels[edges[:-1]]].astype(np.int64)


def stable_anchors(z, auxiliary, agreement, seen, p_dead, base, config=AnchorConfig()):
    """Find local support, using probabilities, model agreement and play state.

    ``stable_frames`` applies only to an anchor's reliability assessment. A
    possession shorter than this remains eligible in the final graph.
    ``agreement`` is mean signed votes from existing independent model seeds.
    """
    z, auxiliary, agreement, seen, p_dead, base = map(np.asarray, (z, auxiliary, agreement, seen, p_dead, base))
    if any(a.shape != z.shape for a in (auxiliary, agreement, seen, p_dead, base)):
        raise ValueError('Anchor inputs must align')
    team = z > 0
    sign = 2 * team - 1
    observed = uniform_filter1d(seen.astype(float), config.observation_window, mode='nearest')
    supported = ((expit(np.abs(z)) >= config.confidence)
                 & (expit(auxiliary * sign) >= config.auxiliary_confidence)
                 & (agreement * sign >= 1 / 3 - 1e-6)
                 & (observed >= config.observed_fraction)
                 & (p_dead <= config.maximum_dead_probability) & (base < 2))
    labels = np.where(supported, team.astype(int), -1)
    runs = stretches(labels)
    return runs[(runs[:, 2] >= 0) & (runs[:, 1] - runs[:, 0] >= config.stable_frames)]


def maxima(points, score, count=3, separation=5):
    """Select separated local modes, including flat-score ties equivariantly."""
    points, score = np.asarray(points), np.asarray(score)
    if not len(points):
        return np.empty(0, np.int64)
    # The score is team-invariant after orientation. Ties use frame position.
    order = np.lexsort((points, -score))
    picked = []
    for i in order:
        k = int(points[i])
        if all(abs(k - j) >= separation for j in picked):
            picked.append(k)
            if len(picked) == count:
                break
    return np.array(sorted(picked), dtype=np.int64)


def anchored_candidates(z, base, anchors, config=AnchorConfig(), boundary=None):
    """Return frame-accurate handover nodes and multi-change local paths.

    Opposing anchors score the entire uncertain bracket from both sides, not
    a monotonic bisection. Same-owner anchors still receive unrestricted local
    two-state inference. Endpoint priors constrain proposals only; their arcs
    subsequently compete against the unchanged winner graph.
    """
    z, base = np.asarray(z), np.asarray(base)
    prefix = np.r_[0., np.cumsum(z, dtype=float)]
    dead_prefix = np.r_[0, np.cumsum(base >= 2)]
    nodes, proposals, brackets = set(), set(), []
    for a, b in zip(anchors[:-1], anchors[1:]):
        start, stop = int(a[1]), int(b[0])
        if stop < start or stop - start > config.maximum_bracket:
            continue
        left, right = int((a[0] + a[1]) // 2), int((b[0] + b[1]) // 2)
        if dead_prefix[right + 1] != dead_prefix[left]:
            continue
        brackets.append((start, stop, int(a[2]), int(b[2])))
        if a[2] != b[2] and stop >= start:
            points = np.arange(max(left + 1, start), min(right, stop) + 1)
            sign = 2 * int(a[2]) - 1
            scores = sign * (2 * prefix[points] - prefix[left] - prefix[right + 1])
            if boundary is not None:
                # Bounded tie-breaking evidence; ownership remains the main score.
                scores = scores + 2 * boundary[points, int(b[2])]
            for k in maxima(points, scores, config.choices, config.separation):
                nodes.add(int(k))
                proposals.add((left, int(k), int(a[2])))
                proposals.add((int(k), right + 1, int(b[2])))
        # The endpoint priors do not assume exactly one change. In particular,
        # a short opposing island can survive between same-team anchor stretches.
        local = np.asarray(z[left:right + 1], dtype=float).copy()
        local[0] += (2 * int(a[2]) - 1) * 50
        local[-1] += (2 * int(b[2]) - 1) * 50
        for cost in (1.5, 6.):
            costs = np.full((len(local), 2), cost)
            if boundary is not None:
                costs *= 1 - .75 * boundary[left:right + 1]
            path = segment_decode(local, costs)
            for s, e, t in stretches(path):
                s, e = int(s + left), int(e + left)
                proposals.add((s, e, int(t)))
                if s > left:
                    nodes.add(s)
                if e < right + 1:
                    nodes.add(e)
    return np.array(sorted(nodes), dtype=np.int64), np.array(sorted(proposals), dtype=np.int64).reshape(-1, 3), brackets


def boundary_search_segments(path, anchors, config=AnchorConfig()):
    """Full-resolution, two-sided alternatives around a winner boundary.

    Each returned group contains all split frames permitted by neighboring
    locally reliable stretches. A caller scores both resulting segments jointly
    with the frozen quality model and keeps separated plausible modes.
    """
    spans = stretches(path)
    groups = []
    for a, b in zip(spans[:-1], spans[1:]):
        if a[2] >= 2 or b[2] >= 2 or a[2] == b[2]:
            continue
        s, boundary, e = int(a[0]), int(a[1]), int(b[1])
        aa = anchors[(anchors[:, 0] >= s) & (anchors[:, 1] <= boundary) & (anchors[:, 2] == a[2])]
        bb = anchors[(anchors[:, 0] >= boundary) & (anchors[:, 1] <= e) & (anchors[:, 2] == b[2])]
        if not len(aa) or not len(bb):
            continue
        lower = max(s + 1, boundary - config.endpoint_radius, int(aa[-1, 1]))
        upper = min(e - 1, boundary + config.endpoint_radius, int(bb[0, 0]))
        if upper < lower:
            continue
        points = np.arange(lower, upper + 1)
        left = np.c_[np.full(len(points), s), points, np.full(len(points), a[2])]
        right = np.c_[points, np.full(len(points), e), np.full(len(points), b[2])]
        groups.append((points, left.astype(np.int64), right.astype(np.int64)))
    return groups


def connect_nodes(pool, nodes, local_proposals, base, radius=63):
    """Add compatible shifted arcs without deleting any original alternative."""
    result = set(map(tuple, np.asarray(pool).tolist()))
    result.update(map(tuple, np.asarray(local_proposals).reshape(-1, 3).tolist()))
    live_runs = stretches((np.asarray(base) < 2).astype(int))
    for lo, hi, live in live_runs:
        if not live:
            continue
        pp = pool[(pool[:, 0] >= lo) & (pool[:, 1] <= hi)]
        for k in nodes[(nodes > lo) & (nodes < hi)]:
            # Existing quality proposals can now end/start at the new frame.
            left = pp[(np.abs(pp[:, 1] - k) <= radius) & (pp[:, 0] < k)]
            right = pp[(np.abs(pp[:, 0] - k) <= radius) & (pp[:, 1] > k)]
            result.update((int(s), int(k), int(t)) for s, _, t in left)
            result.update((int(k), int(e), int(t)) for _, e, t in right)
    return np.array(sorted(result), dtype=np.int64)


def transition_quality(segments, boundary_probability, base, weight):
    """Bounded directional switch evidence for the graph's existing arc score.

    A log-odds ratio against a fixed 5% reference rate changes transition cost.
    Live-interval starts have no H/A transition term. No duration is forbidden.
    """
    s, _, t = np.asarray(segments).T
    p = np.clip(boundary_probability[s, t], 1e-5, 1 - 1e-5)
    log_odds = np.log(p / (1 - p)) - np.log(.05 / .95)
    valid = (s > 0) & (np.asarray(base)[np.maximum(0, s - 1)] < 2)
    return weight * np.clip(log_odds, -2., 2.) * valid
