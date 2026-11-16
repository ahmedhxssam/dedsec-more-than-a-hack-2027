"""Temporal region-of-interest evidence for possession interval quality.

This compact detector head sees motion throughout a proposed interval and both
flanks. It complements the existing hand-aggregated segment scorer; it does not
infer control from proximity alone or impose a minimum interval duration.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def roi_positions(segments, frames, inside=48, flank=16):
    starts, ends = segments[:, 0], segments[:, 1]
    duration = ends-starts
    context = np.clip(duration, 25, 125)
    before = starts[:, None]-context[:, None]*(1-(np.arange(flank)+.5)/flank)
    middle = starts[:, None]+duration[:, None]*(np.arange(inside)+.5)/inside
    after = ends[:, None]+context[:, None]*(np.arange(flank)+.5)/flank
    return np.clip(np.concatenate([before, middle, after], axis=1).astype(int), 0, frames-1)


def motion_rois(native, segments):
    positions=roi_positions(segments,len(native))
    values=np.asarray(native[positions],dtype=np.float32)
    teams=segments[:,2]
    same=np.take_along_axis(values,teams[:,None,None,None],axis=2).squeeze(2)
    other=np.take_along_axis(values,(1-teams)[:,None,None,None],axis=2).squeeze(2)
    return np.concatenate([same,other-same],axis=2).transpose(0,2,1).copy()


class IntervalMotion(nn.Module):
    def __init__(self,channels=196,summary=175,width=24,independent=False):
        super().__init__()
        self.independent=independent
        self.encoder=nn.Sequential(nn.Conv1d(channels,width,1),nn.GELU(),
            nn.Conv1d(width,width,3,padding=1),nn.GELU(),
            nn.Conv1d(width,width,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv1d(width,width,3,padding=4,dilation=4),nn.GELU(),
            nn.AdaptiveAvgPool1d(10),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(width*10+summary,64),nn.GELU(),
            nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,motion,summary,base_quality):
        h=self.encoder(motion)
        raw=self.head(torch.cat([h,summary],dim=1)).squeeze(-1)
        if self.independent:
            return torch.sigmoid(raw)
        delta=3*torch.tanh(raw/3)
        base=torch.logit(base_quality.clamp(.02,.98))
        return torch.sigmoid(base+delta)
