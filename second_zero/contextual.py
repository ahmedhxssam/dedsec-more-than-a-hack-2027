"""Separate contextual live/dead and established-team models.

Features are anonymous collective kinematics and ball/player interactions. Models
are fitted only on explicitly named external training matches. Missing ball is
handled by a separate player-context model; no missingness-to-dead rule exists.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d

VERSION='contextual-v1'

def average(x,width):
    finite=np.isfinite(x)
    den=uniform_filter1d(finite.astype(np.float32),width,axis=0,mode='nearest')
    num=uniform_filter1d(np.where(finite,x,0).astype(np.float32),width,axis=0,mode='nearest')
    return np.divide(num,den,out=np.full_like(num,np.nan),where=den>.15)


def velocity(x,step=2,fps=25):
    v=np.full_like(x,np.nan)
    v[step:-step]=(x[2*step:]-x[:-2*step])*(fps/(2*step))
    return v


def features(ball,players,team,pitch=(105.3,68),fps=25):
    """Metres, source-frame order, no labels; call separately for each half."""
    ball=np.asarray(ball,dtype=np.float32);players=np.asarray(players,dtype=np.float32).copy()
    n=len(ball)
    pv=velocity(players,fps=fps);ps=np.linalg.norm(pv,axis=2)
    moving=np.linalg.norm(np.diff(players,axis=0,prepend=players[:1]),axis=2)>1e-4
    usable=np.isfinite(players).all(axis=2)&(uniform_filter1d(moving.astype(np.float32),251,axis=0,mode='nearest')>.015)
    usable&=(ps<12)|~np.isfinite(ps)
    players[~usable]=np.nan;pv[~usable]=np.nan;ps[~usable]=np.nan
    observed=np.isfinite(ps).sum(axis=1)
    # Quantiles and team shape remain meaningful under fragmented slot identities.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        q=np.nanquantile(ps,[.1,.25,.5,.75,.9],axis=1).T
        mean=np.nanmean(ps,axis=1)
        acceleration=np.linalg.norm(velocity(pv,fps=fps),axis=2)
        ac=np.nanmedian(np.where(acceleration<15,acceleration,np.nan),axis=1)
        vmean=np.nanmean(pv,axis=1)
        coherent=np.linalg.norm(vmean,axis=1)/(mean+.1)
        shapes=[]
        for t in [0,1]:
            pos=players[:,team==t]
            center=np.nanmean(pos,axis=1)
            spread=np.nanstd(pos,axis=1)
            shapes.append(np.c_[spread,center])
        shapes=np.stack(shapes,axis=1)
    slow=np.divide(((ps<1)&np.isfinite(ps)).sum(axis=1),observed,out=np.full(n,np.nan),where=observed>=8)
    fast=np.divide(((ps>3)&np.isfinite(ps)).sum(axis=1),observed,out=np.full(n,np.nan),where=observed>=8)
    team_spread=shapes[:,:,:2].mean(axis=1)
    separation=np.linalg.norm(shapes[:,0,2:]-shapes[:,1,2:],axis=1)
    player=np.c_[q,mean,slow,fast,ac,coherent,team_spread,separation]
    player[observed<8]=np.nan
    # Multi-scale temporal context includes walking/repositioning during stoppages.
    player=np.concatenate([player,average(player,13),average(player,51),average(player,126)],axis=1)
    bv=velocity(ball,fps=fps);bs=np.linalg.norm(bv,axis=1)
    bs[bs>60]=np.nan
    change=np.linalg.norm(velocity(bv,fps=fps),axis=1)
    change=np.minimum(change,100)
    d=np.linalg.norm(players-ball[:,None],axis=2)
    d[~np.isfinite(d)]=np.inf
    best=[];relative=[];near_speed=[]
    for t in [0,1]:
        slots=np.flatnonzero(team==t);td=d[:,slots];order=td.argmin(axis=1)
        ix=slots[order];dist=td[np.arange(n),order]
        dist[~np.isfinite(dist)]=np.nan
        best.append(dist)
        relative.append(np.linalg.norm(pv[np.arange(n),ix]-bv,axis=1))
        near_speed.append(ps[np.arange(n),ix])
    best=np.stack(best,axis=1);relative=np.stack(relative,axis=1);near_speed=np.stack(near_speed,axis=1)
    boundary=np.minimum.reduce([ball[:,0],pitch[0]-ball[:,0],ball[:,1],pitch[1]-ball[:,1]])
    bfeat=np.c_[bs,change,boundary,np.minimum(best[:,0],best[:,1]),np.minimum(relative[:,0],relative[:,1])]
    dead=np.c_[player,bfeat,average(bfeat,13),average(bfeat,51)]
    owner=[]
    for t in [0,1]:
        o=1-t
        base=np.c_[best[:,t],best[:,o],best[:,o]-best[:,t],relative[:,t],relative[:,o],near_speed[:,t],bs,change,
                   np.exp(-best[:,t]**2/2)*np.minimum(change/40,1)]
        owner.append(np.c_[base,average(base,5),average(base,13),average(base,51)])
    return {'player':player.astype(np.float32),'dead':dead.astype(np.float32),
            'owner':np.stack(owner,axis=1).astype(np.float32),'ball_seen':np.isfinite(ball).all(axis=1),
            'nearest':best,'speed':bs,'observed_players':observed}


def binary_decode(log_odds,cost):
    """Two-state offline DP without a minimum duration or gap replacement."""
    x=np.asarray(log_odds);n=len(x)
    if not n:return np.zeros(0,dtype=np.int8)
    costs=np.broadcast_to(np.asarray(cost,dtype=float),n)
    back=np.zeros((n,2),dtype=np.int8);a=0.;b=float(x[0])
    for i in range(1,n):
        c=costs[i]
        if b-c>a:aa=b-c;back[i,0]=1
        else:aa=a
        if a-c>b:bb=a-c;back[i,1]=0
        else:bb=b;back[i,1]=1
        a=aa;b=bb+float(x[i]);m=max(a,b);a-=m;b-=m
    out=np.empty(n,dtype=np.int8);out[-1]=b>a
    for i in range(n-1,0,-1):out[i-1]=back[i,out[i]]
    return out


def predict_probabilities(models,f):
    # The player-only head is trained on all states, not just missing-ball examples.
    dp=models['dead'].predict_proba(f['dead'])[:,1]
    pp=models['player'].predict_proba(f['player'])[:,1]
    dp=np.where(f['ball_seen'],dp,pp)
    op=np.stack([models['owner'].predict_proba(f['owner'][:,t])[:,1] for t in [0,1]],axis=1)
    op/=np.maximum(op.sum(axis=1,keepdims=True),1e-8)
    # Unseen ball gives no instantaneous new control claim. Future observed
    # interaction can still constrain the offline sequence at the reappearance.
    op[~f['ball_seen']]=.5
    return dp,op


def decode_probabilities(dead,owner,team_cost=12.,dead_cost=12.,visual_logit=None,visual_weight=0.):
    d=np.log(np.clip(dead,.01,.99)/np.clip(1-dead,.01,.99))
    if visual_logit is not None:d+=visual_weight*np.clip(visual_logit,-2,2)
    alive_dead=binary_decode(d,dead_cost)
    team_log=np.log(np.clip(owner[:,1],.01,.99)/np.clip(owner[:,0],.01,.99))
    state=np.full(len(dead),2,dtype=np.int8)
    # Ownership is decoded within live islands; dead intervals cannot inherit
    # live team emissions, and a prior owner cannot suppress an inferred stoppage.
    edges=np.r_[0,np.flatnonzero(alive_dead[1:]!=alive_dead[:-1])+1,len(dead)]
    for start,stop in zip(edges[:-1],edges[1:]):
        if alive_dead[start]==0:state[start:stop]=binary_decode(team_log[start:stop],team_cost)
    return state
