"""Bounded linear ball estimates as explicitly uncertain extra owner evidence.

The original observations are never replaced. This module returns only extra
geometry channels plus flags; inference has no access to possession/status.
"""
from __future__ import annotations
import numpy as np
from second_zero.contextual import velocity

MAX_GAP=250
CHANNELS=24

def interpolated_features(ball,players,team,pitch=(105.3,68),fps=25):
    ball=np.asarray(ball,np.float32);players=np.asarray(players,np.float32)
    n=len(ball);ix=np.arange(n);seen=np.isfinite(ball).all(1)
    known=ix[seen];estimated=ball.copy()
    if len(known)<2:return np.zeros((n,2,CHANNELS),np.float32)
    previous=np.maximum.accumulate(np.where(seen,ix,-n))
    following=np.minimum.accumulate(np.where(seen,ix,2*n)[::-1])[::-1]
    span=following-previous
    fill=(~seen)&(previous>=0)&(following<n)&(span<=MAX_GAP)
    for axis in range(2):estimated[fill,axis]=np.interp(ix[fill],known,ball[known,axis])
    pv=velocity(players,fps=fps);bv=velocity(estimated,fps=fps)
    dist=np.linalg.norm(players-estimated[:,None],axis=2)
    rel=np.linalg.norm(pv-bv[:,None],axis=2)
    radial=np.sum((players-estimated[:,None])*(pv-bv[:,None]),axis=2)/np.maximum(dist,.1)
    near=[]
    for t in [0,1]:
        slots=np.flatnonzero(team==t)
        indices=slots[np.argsort(np.nan_to_num(dist[:,slots],nan=np.inf),axis=1)[:,:3]]
        rows=ix[:,None]
        nearest=dist[rows,indices]/20
        rv=rel[rows,indices[:,:1]]/30
        rd=radial[rows,indices[:,:1]]/30
        proximity=np.exp(-dist[:,slots]/3)
        proximity[~np.isfinite(proximity)]=0
        near.append(np.c_[nearest,rv,rd,proximity.max(1)])
    output=[]
    for t in [0,1]:
        values=np.c_[near[t],near[1-t],fill,seen,np.clip(span,0,MAX_GAP)/MAX_GAP,
            np.clip(ix-previous,0,MAX_GAP)/MAX_GAP,np.clip(following-ix,0,MAX_GAP)/MAX_GAP,
            np.linalg.norm(bv,axis=1)/40]
        missing=~np.isfinite(values[:,:6])
        values=np.clip(np.nan_to_num(values,nan=0,posinf=0,neginf=0),-5,5)
        output.append(np.c_[values,missing])
    result=np.stack(output,axis=1).astype(np.float32)
    assert result.shape==(n,2,CHANNELS)
    return result
