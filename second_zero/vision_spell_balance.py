"""Prepared objective experiment; not part of any deployed or trained model.

Use native complete training-half spells to define weights before cropping.
Unlike clipped square-root duration weighting, every complete live spell has
unit total weight. A convex mixture retains ordinary frame calibration. Crop
sampling remains explicit: event-biased batches are not an unbiased full-match
estimate merely because the within-batch loss uses these weights.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def full_spell_weights(labels):
    """One inverse full duration per H/A frame; dead/unknown break spells."""
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError('Expected one complete native-frame half')
    weights = np.zeros(len(y), np.float32)
    if not len(y):
        return weights
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    for start, stop in zip(edges[:-1], edges[1:]):
        if y[start] in (0, 1):
            weights[start:stop] = 1. / (stop - start)
    return weights


def mixed_spell_bce(logits, labels, weights, spell_fraction=.5):
    """Fixed-shaped frame/event mixture without minimum or clipped duration."""
    if not 0 <= spell_fraction <= 1:
        raise ValueError('Mixture must be in [0,1]')
    valid = (labels >= 0) & (labels < 2)
    target = torch.where(valid, labels, 0).to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    frame_weights = valid.to(logits.dtype)
    event_weights = torch.where(valid, weights, 0).to(logits.dtype)
    frame = (losses * frame_weights).sum() / frame_weights.sum().clamp_min(1.)
    event = (losses * event_weights).sum() / event_weights.sum().clamp_min(1e-8)
    return (1 - spell_fraction) * frame + spell_fraction * event, frame, event
