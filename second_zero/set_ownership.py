"""Shared player-set encoding for 25 Hz ownership, without player identities.

The model is invariant to player ordering within either team and equivariant to
exchanging team names. Slot derivatives are observation proxies, not identities.
All reductions are mask-aware and temporal decoding remains external.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch
from torch import nn
from .contextual import velocity

VERSION = 'set-owner-v1'
DILATIONS = (1, 2, 4, 8, 16, 32)
RADIUS = sum(DILATIONS)
FEATURES = ['relative_x', 'relative_y', 'distance', 'local_proximity',
    'relative_velocity_x', 'relative_velocity_y', 'player_speed', 'radial_speed',
    'player_pitch_x', 'player_pitch_y', 'distance_change_5f', 'distance_change_12f',
    'position_available', 'velocity_available', 'ball_available', 'relative_available',
    'relative_velocity_available', 'distance_change_5f_available', 'distance_change_12f_available',
    'ball_gap_previous', 'ball_gap_next', 'ball_speed', 'ball_speed_available']


def player_set_features(ball, players, team, pitch=(105.3, 68), fps=25):
    """Return frame/team/player/channel float16 cache and validity as channel 12.

    Positions and original within-slot derivatives are computed before grouping.
    No sorting, top-k, identity embeddings, labels, learned normalization, or
    interpolated positions enter the features. Padding is explicitly masked.
    """
    if fps != 25:
        raise ValueError('Requires native 25 Hz inputs')
    ball = np.asarray(ball, np.float32)
    players = np.asarray(players, np.float32)
    team = np.asarray(team)
    if set(np.unique(team)) != {0, 1} or len(ball) != len(players):
        raise ValueError('Expected aligned positions and two mapped teams')
    n, p, _ = players.shape
    pv, bv = velocity(players, fps=fps), velocity(ball, fps=fps)
    ps = np.linalg.norm(pv, axis=2)
    moving = np.linalg.norm(np.diff(players, axis=0, prepend=players[:1]), axis=2) > 1e-4
    present = np.isfinite(players).all(2)
    present &= uniform_filter1d(moving.astype(np.float32), 251, axis=0, mode='nearest') > .015
    vseen = present & np.isfinite(pv).all(2) & (ps < 12)
    bseen = np.isfinite(ball).all(1)
    bvseen = np.isfinite(bv).all(1)
    rel = players - ball[:, None]
    dist = np.linalg.norm(rel, axis=2)
    rseen = present & bseen[:, None]
    rel[~rseen] = np.nan
    dist[~rseen] = np.nan
    rv = pv - bv[:, None]
    rvseen = vseen & bvseen[:, None]
    rv[~rvseen] = np.nan
    radial = (rel * rv).sum(2) / np.maximum(dist, .1)
    xy = (players - np.asarray(pitch, np.float32)[None, None]/2) / np.asarray(pitch, np.float32)[None, None]
    xy[~present] = np.nan
    ps[~vseen] = np.nan
    changes = []
    for k in (5, 12):
        d = np.full((n, p), np.nan, np.float32)
        d[k:-k] = (dist[2*k:] - dist[:-2*k]) * fps / (2*k*30)
        changes.append(d)
    ix = np.arange(n)
    prev = np.maximum.accumulate(np.where(bseen, ix, -250))
    nxt = np.minimum.accumulate(np.where(bseen, ix, n+250)[::-1])[::-1]
    gap = np.c_[np.minimum(ix-prev, 250), np.minimum(nxt-ix, 250)] / 250
    bspeed = np.linalg.norm(bv, axis=1)/40
    repeat = lambda v: np.broadcast_to(v[:, None], (n,p))
    values = np.stack([rel[...,0]/20, rel[...,1]/20, dist/20, np.exp(-dist/4),
        rv[...,0]/30, rv[...,1]/30, ps/12, radial/30, xy[...,0], xy[...,1],
        changes[0], changes[1], present, vseen, repeat(bseen), rseen, rvseen,
        np.isfinite(changes[0]), np.isfinite(changes[1]), repeat(gap[:,0]), repeat(gap[:,1]),
        repeat(bspeed), repeat(bvseen)], axis=-1)
    values = np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0), -5, 5).astype(np.float16)
    slots = [np.flatnonzero(team == t) for t in (0,1)]
    out = np.zeros((n, 2, max(map(len, slots)), len(FEATURES)), np.float16)
    for t in (0,1):
        out[:,t,:len(slots[t])] = values[:,slots[t]]
    return out


class SetOwner(nn.Module):
    """Deep Sets pools feeding shared own/opponent temporal team scores."""
    def __init__(self, channels=len(FEATURES), player_width=16, width=32):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(channels,24),nn.GELU(),nn.Linear(24,player_width),nn.GELU())
        self.attention = nn.Linear(player_width,1)
        self.input = nn.Conv1d(6*player_width + 2, width, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(width,width,3,padding=d,dilation=d) for d in DILATIONS)
        self.output = nn.Conv1d(width,1,1)
        self.dropout = nn.Dropout(.1)

    def forward(self,x):
        # B,T,team,P,C; player mask goes through every reduction.
        b,n,teams,p,c=x.shape
        mask=x[...,12:13]
        h=self.player(x)
        summed=(h*mask).sum(3)/11
        maximum=h.masked_fill(mask==0,-1e4).amax(3)
        anyseen=mask.sum(3)>0
        maximum=torch.where(anyseen,maximum,torch.zeros_like(maximum))
        a=torch.softmax(self.attention(h).masked_fill(mask==0,-1e4),dim=3)*mask
        a=a/a.sum(3,keepdim=True).clamp_min(1e-8)
        attended=(a*h).sum(3)
        own=torch.cat([summed,maximum,attended,mask.sum(3)/11],dim=-1)
        paired=torch.cat([own,own.flip(2)],dim=-1)
        z=paired.permute(0,2,3,1).reshape(b*teams,paired.shape[-1],n)
        z=torch.nn.functional.gelu(self.input(z))
        for layer in self.blocks:
            z=z+.5*self.dropout(torch.nn.functional.gelu(layer(z)))
        z=self.output(z).reshape(b,teams,n)
        return z[:,1]-z[:,0]


def infer_logits(model,x,chunk=2048):
    model.eval()
    out=np.empty(len(x),np.float32)
    with torch.inference_mode():
        for s in range(0,len(x),chunk):
            e=min(s+chunk,len(x));a=max(0,s-RADIUS);b=min(len(x),e+RADIUS)
            crop=np.asarray(x[a:b],np.float32)
            crop=np.pad(crop,((max(0,RADIUS-s),max(0,e+RADIUS-len(x))),(0,0),(0,0),(0,0)))
            out[s:e]=model(torch.from_numpy(crop[None]))[0,RADIUS:RADIUS+e-s].numpy()
    return out
