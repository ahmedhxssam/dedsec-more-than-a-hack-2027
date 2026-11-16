"""Bidirectional gap memory with independent, phase-shifted training occlusions."""
import numpy as np
import torch
from torch import nn
from scipy.ndimage import gaussian_filter1d
from second_zero import raw_context_owner as original

inputs=original.inputs
CHANNELS=414


def phase_corruption(raw,phase,seed):
    if set(raw)!={'ball','players','team','pitch'}:raise ValueError('Raw fields only')
    rng=np.random.default_rng(seed);out={k:v.copy() for k,v in raw.items()};n=len(out['ball'])
    for s in range(phase,n,375):out['ball'][s:s+int(rng.integers(25,201))]=np.nan
    out['ball']+=gaussian_filter1d(rng.normal(0,.5,out['ball'].shape).astype(np.float32),3,axis=0)
    for i in range(out['players'].shape[1]):
        for s in range(int(rng.integers(0,2500)),n,2500):out['players'][s:s+100,i]=np.nan
    return out


class GapBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(CHANNELS),nn.Linear(CHANNELS,64),nn.GELU())
        self.memory=nn.GRU(64,64,2,batch_first=True,bidirectional=True,dropout=.1)
        self.owner=nn.Linear(128,1);self.boundary=nn.Linear(128,1)
        nn.init.zeros_(self.owner.weight);nn.init.zeros_(self.owner.bias)

    def forward(self,x,prior):
        h=self.input(x.transpose(1,2));h,_=self.memory(h)
        z=self.owner(h).squeeze(-1)
        return z,self.boundary(h).squeeze(-1),z


def batch(data,rng,batch_size=8,length=256):
    v=original.batch(data,rng,batch_size,length)
    # Match inference: use only the central128 samples of a256-sample window.
    v['weight'][:,:64]=0;v['weight'][:,-64:]=0
    return v


def loss(model,v):
    z,b,absolute=model(v['x'],v['prior']);w=v['weight'];den=w.sum().clamp_min(1.)
    owner=(nn.functional.binary_cross_entropy_with_logits(z,v['y'],reduction='none')*w).sum()/den
    event=(nn.functional.binary_cross_entropy_with_logits(b,v['boundary'],reduction='none',pos_weight=torch.tensor(10.,device=b.device))*(w>0)).sum()/(w>0).sum().clamp_min(1.)
    return owner+.1*event+.001*(absolute.square()*w).sum()/den


@torch.no_grad()
def predict_both(model,x,prior,n):
    model.eval();length=x.shape[-1];owner=np.empty(length,np.float32);boundary=np.empty(length,np.float32)
    for start in range(0,length,128):
        idx=np.arange(start-64,start+192);valid=(idx>=0)&(idx<length);ii=np.clip(idx,0,length-1)
        window=x[:,:,ii].copy();window[:,:,~valid]=0
        z,b,_=model(torch.from_numpy(window),torch.zeros((2,256)))
        stop=min(length,start+128);k=stop-start
        owner[start:stop]=(.5*(z[0,64:64+k]-z[1,64:64+k])).cpu().numpy()
        boundary[start:stop]=(.5*(b[0,64:64+k]+b[1,64:64+k])).cpu().numpy()
    grid=np.arange(length)*5
    return tuple(np.interp(np.arange(n),grid,v).astype(np.float32) for v in [owner,boundary])


def predict(model,x,prior,n,device='cpu'):
    assert device=='cpu'
    return predict_both(model,x,prior,n)[0]
