"""Temporal ownership repair with auxiliary possession-change supervision."""
import numpy as np
import torch
from torch import nn
from scipy.ndimage import uniform_filter1d
from second_zero import global_team_context as context

STRIDE=5
CHANNELS=307
WIDTH=32
DILATIONS=(1,2,4,8,16,32)


class TemporalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(CHANNELS,WIDTH,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(WIDTH,WIDTH,3,padding=d,dilation=d),nn.GELU(),nn.Dropout(.1),nn.Conv1d(WIDTH,WIDTH,1)) for d in DILATIONS])
        self.owner=nn.Conv1d(WIDTH,1,1);self.boundary=nn.Conv1d(WIDTH,1,1)
        nn.init.zeros_(self.owner.weight);nn.init.zeros_(self.owner.bias)

    def forward(self,x,prior):
        h=torch.nn.functional.gelu(self.input(x))
        for block in self.blocks:h=h+.2*block(h)
        delta=self.owner(h).squeeze(1)
        return prior+delta,self.boundary(h).squeeze(1),delta


def inputs(raw,signals,parent):
    g=context.features(raw['players'],raw['team'],raw['pitch'])
    scales=np.array([8,8,2,1,1,8,8,4,4,4,1,1,1],np.float32)
    missing=(~np.isfinite(raw['ball']).all(1)).astype(np.float32);support=uniform_filter1d(missing.astype(np.float64),25,mode='nearest').astype(np.float32)
    perspectives=[]
    for swap in [False,True]:
        sig=signals.copy();p=np.asarray(parent).copy()
        if swap:
            sig[:,[0,1,3,7,8,9]]*=-1;p[p<2]=1-p[p<2]
        x=np.c_[np.clip(g[:,0 if swap else 1],-5,5),np.clip(sig/scales,-5,5),p==0,p==1,missing,support].astype(np.float32)
        perspectives.append(uniform_filter1d(x,5,axis=0,mode='nearest')[::STRIDE].T.copy())
    x=np.stack(perspectives)
    prior=uniform_filter1d(signals[:,0],5,mode='nearest')[::STRIDE].copy()
    if x.shape[1]!=CHANNELS or not np.isfinite(x).all():raise ValueError('Invalid temporal input features')
    return x,prior,support


def targets(truth,parent,support,clean_teacher):
    from second_zero.gap_spell import spells
    y=np.asarray(truth);w=np.zeros(len(y),np.float32)
    for s,e,_ in spells(y):w[s:e]=np.clip(np.sqrt(125/max(1,e-s)),.25,4.)
    w*=((y<2)&(parent<2))*(1+2*support)
    yi=y[::STRIDE];wi=w[::STRIDE];events=np.flatnonzero((y[1:]!=y[:-1])&(y[1:]<2)&(y[:-1]<2))+1
    boundary=np.zeros(len(yi),np.float32)
    for event in events/STRIDE:
        lo=max(0,int(event)-6);hi=min(len(yi),int(event)+7);idx=np.arange(lo,hi)
        boundary[lo:hi]=np.maximum(boundary[lo:hi],np.exp(-.5*((idx-event)/2.)**2))
    teacher=uniform_filter1d(np.asarray(clean_teacher,np.float32),5,mode='nearest')[::STRIDE]
    return {'y':np.clip(yi,0,1).astype(np.float32),'weight':wi,'boundary':boundary,'clean_teacher':teacher,'events':np.round(events/STRIDE).astype(np.int64)}


def batch(data,rng,batch_size=8,length=128):
    values={k:[] for k in ['x','prior','y','weight','boundary','clean_teacher']}
    for _ in range(batch_size):
        d=data[int(rng.integers(len(data)))];n=len(d['y']);live=np.flatnonzero(d['weight']>0);swap=bool(rng.integers(2))
        centers=d['events'] if rng.random()<.5 and len(d['events']) else live
        center=int(rng.choice(centers));idx=np.arange(center-length//2,center+(length-length//2));valid=(idx>=0)&(idx<n);ii=np.clip(idx,0,n-1)
        for k in values:
            v=d[k][int(swap)][:,ii] if k=='x' else d[k][ii].copy()
            if swap and k in ['prior','clean_teacher']:v=-v
            if swap and k=='y':v=1-v
            if k=='weight':v*=valid
            values[k].append(v)
    return {k:torch.from_numpy(np.stack(v).astype(np.float32)) for k,v in values.items()}


def loss(model,v):
    z,b,delta=model(v['x'],v['prior']);w=v['weight'];den=w.sum().clamp_min(1.)
    owner=(torch.nn.functional.binary_cross_entropy_with_logits(z,v['y'],reduction='none')*w).sum()/den
    soft=torch.sigmoid(v['clean_teacher']/2)
    distill=(torch.nn.functional.binary_cross_entropy_with_logits(z/2,soft,reduction='none')*w).sum()/den
    boundary=(torch.nn.functional.binary_cross_entropy_with_logits(b,v['boundary'],reduction='none',pos_weight=torch.tensor(10.,device=b.device))*(w>0)).sum()/(w>0).sum().clamp_min(1.)
    regularizer=(delta.square()*w).sum()/den
    return owner+.25*distill+.05*boundary+.001*regularizer


@torch.no_grad()
def predict(model,x,prior,n,device='cpu'):
    model.eval();pp=np.stack([prior,-prior]);z,_,_=model(torch.from_numpy(x).to(device),torch.from_numpy(pp).to(device));z=(.5*(z[0]-z[1])).cpu().numpy()
    return np.interp(np.arange(n),np.arange(len(z))*STRIDE,z).astype(np.float32)
