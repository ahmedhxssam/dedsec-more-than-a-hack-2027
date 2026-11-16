"""Native-rate, team-equivariant multiscale possession segmentation.

The model consumes geometry only. Its episode objective compares each complete
training possession with its immediate opponents, keeping brief controls in the
loss without imposing a minimum duration or using reference data at inference.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiscaleBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.local = nn.Conv1d(width, width, 3, padding=1)
        self.context = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.mix = nn.Conv1d(width, width * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.local(x) + self.context(x)
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        value, gate = self.mix(F.gelu(z)).chunk(2, 1)
        return x + .5 * self.dropout(value * torch.sigmoid(gate))


class PossessionSequence(nn.Module):
    def __init__(self, channels=122, width=96, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout=.1):
        super().__init__()
        self.architecture = dict(channels=channels, width=width, dilations=list(dilations), dropout=dropout)
        self.radius = sum(dilations)
        self.input = nn.Conv1d(channels * 2, width, 1)
        self.blocks = nn.ModuleList(MultiscaleBlock(width, d, dropout) for d in dilations)
        self.readout = nn.Sequential(nn.Conv1d(width * 2, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, x):
        b, teams, channels, n = x.shape
        if teams != 2:
            raise ValueError('Expected two team perspectives')
        paired = torch.cat((x, x.flip(1)), 2).reshape(b * 2, channels * 2, n)
        z = F.gelu(self.input(paired))
        for block in self.blocks:
            z = block(z)
        z = z.reshape(b, 2, -1, n)
        score = self.readout(torch.cat((z, z.flip(1)), 2).reshape(b * 2, -1, n)).reshape(b, 2, n)
        return score[:, 1] - score[:, 0]


def episode_masks(labels, maximum=24):
    """Complete live spells and their immediate live neighbors in one crop.

    Edge-clipped spells are excluded from the IoU term; frame loss still covers
    them. The cap limits tensor size, never prediction duration.
    """
    y = np.asarray(labels)
    n = len(y)
    interior = np.zeros((maximum, n), np.float32)
    outside = np.zeros_like(interior)
    team = np.zeros(maximum, np.float32)
    valid = np.zeros(maximum, np.float32)
    cuts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, n]
    spells = [(int(s), int(e), int(y[s])) for s, e in zip(cuts[:-1], cuts[1:])]
    # If there are >maximum complete spells, spread supervision across them.
    eligible = [i for i, (s, e, t) in enumerate(spells) if s > 0 and e < n and t < 2]
    if len(eligible) > maximum:
        eligible = [eligible[i] for i in np.linspace(0, len(eligible) - 1, maximum).astype(int)]
    for j, i in enumerate(eligible):
        s, e, t = spells[i]
        interior[j, s:e] = 1
        if spells[i - 1][2] < 2:
            outside[j, max(spells[i - 1][0], s - 250):s] = 1
        if spells[i + 1][2] < 2:
            outside[j, e:min(spells[i + 1][1], e + 250)] = 1
        team[j] = t
        valid[j] = 1
    return interior, outside, team, valid


def segmentation_loss(logits, labels, weights, masks, iou_weight=.5):
    valid = (labels >= 0) & (labels < 2)
    target = torch.where(valid, labels, 0).to(logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    v = valid.to(logits.dtype)
    frame = (ce * v).sum() / v.sum().clamp_min(1)
    w = weights * v
    episode = (ce * w).sum() / w.sum().clamp_min(1e-8)
    inside, outside, team, eligible = masks
    probability = torch.sigmoid(logits)[:, None]
    probability = torch.where(team[:, :, None] > .5, probability, 1 - probability)
    intersection = (probability * inside).sum(-1)
    union = inside.sum(-1) + (probability * outside).sum(-1)
    iou = ((1 - intersection / union.clamp_min(1)) * eligible).sum() / eligible.sum().clamp_min(1)
    return .5 * (frame + episode) + iou_weight * iou, dict(frame=frame, episode=episode, soft_iou=iou)


def infer(model, values, chunk=4096, device='cpu'):
    """Exact finite-context overlap-save inference, including native edges."""
    model.eval()
    n, radius = len(values), model.radius
    out = np.empty(n, np.float32)
    with torch.inference_mode():
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            lo, hi = max(0, start - radius), min(n, stop + radius)
            x = values[lo:hi].astype(np.float32)
            x = np.pad(x, ((max(0, radius - start), max(0, stop + radius - n)), (0, 0), (0, 0)))
            batch = torch.from_numpy(x.transpose(1, 2, 0)[None].copy()).to(device)
            out[start:stop] = model(batch)[0, radius:radius + stop - start].cpu().numpy()
    return out
