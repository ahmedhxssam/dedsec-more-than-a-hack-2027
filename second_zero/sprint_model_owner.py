"""Sprint ownership variants. Inputs are the frozen geometry-only trajectories."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class SprintOwner(nn.Module):
    def __init__(self, channels=98, width=32, dilations=(1,2,4,8,16,32), dropout=.1):
        super().__init__()
        self.radius=sum(dilations)
        self.input=nn.Conv1d(channels,width,1)
        self.blocks=nn.ModuleList([nn.Conv1d(width,width,3,padding=d,dilation=d) for d in dilations])
        self.output=nn.Conv1d(width,1,1)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        b,teams,c,n=x.shape
        z=torch.nn.functional.gelu(self.input(x.reshape(b*teams,c,n)))
        for layer in self.blocks:z=z+.5*self.dropout(torch.nn.functional.gelu(layer(z)))
        z=self.output(z).reshape(b,teams,n)
        return z[:,1]-z[:,0]


def infer_logits(model,x,chunk=4096):
    model.eval();radius=model.radius
    padded=np.pad(x,((radius,radius),(0,0),(0,0)))
    out=np.empty(len(x),np.float32)
    with torch.inference_mode():
        for s in range(0,len(x),chunk):
            e=min(s+chunk,len(x))
            batch=torch.from_numpy(padded[s:e+2*radius].transpose(1,2,0)[None].copy())
            out[s:e]=model(batch)[0,radius:radius+e-s].cpu().numpy()
    return out


def episode_weights(y,mode):
    weights=np.ones(len(y),np.float32)
    bounds=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    for s,e in zip(bounds[:-1],bounds[1:]):
        if y[s]>=2:continue
        if mode=='episode':weights[s:e]=np.clip(np.sqrt(125./(e-s)),.35,4.)
        elif mode=='boundary':
            distance=np.minimum(np.arange(e-s),np.arange(e-s)[::-1])
            weights[s:e]=1+2*np.exp(-distance/12.)
    return weights
