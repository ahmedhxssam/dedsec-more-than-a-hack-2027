"""Raw-only interval credit model: no learned baseline input during fitting."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
from second_zero.global_team_context import features as global_features
from second_zero.sprint_model_interpolation import interpolated_features

MODEL=dict(max_iter=240,max_leaf_nodes=15,max_depth=5,min_samples_leaf=60,learning_rate=.05,l2_regularization=10.,random_state=20261030,early_stopping=False)
SELECT=np.array([0,1,2,3,4,5,6,18,19,20,21,22,23,24,25,29,58,59,60,61,62,63,64,76,77,78,79,80,81,82,83,87,116,117,118,119,120,121,290,291,293,296,302,303,304,305,306,307,308,309],int)
CONTEXT=SELECT[[0,1,3,5,7,8,11,12,13,14,15,16,18,19,21,23,24,27,28,29,30,31,38,40,46]]


class RawIntervals:
    def __init__(self,raw):
        if set(raw)!={'ball','players','team','pitch'}:raise ValueError('Raw observations only')
        g=global_features(raw['players'],raw['team'],raw['pitch']);linear=interpolated_features(raw['ball'],raw['players'],raw['team'],raw['pitch'])
        self.x=np.concatenate([g,linear],-1);self.n=len(self.x)
        self.prefix=np.concatenate([np.zeros((1,2,self.x.shape[-1])),np.cumsum(self.x,axis=0,dtype=np.float64)])
        squared=self.x[:,:,SELECT]**2
        self.squared=np.concatenate([np.zeros((1,2,len(SELECT))),np.cumsum(squared,axis=0,dtype=np.float64)])
        missing=(~np.isfinite(raw['ball']).all(1)).astype(float)
        self.missing=np.r_[0,np.cumsum(missing)];self.support=uniform_filter1d(missing,25,mode='nearest')

    def means(self,s,e,t,columns=None):
        v=(self.prefix[e,t]-self.prefix[s,t])/np.maximum(1,e-s)[:,None]
        return v if columns is None else v[:,columns]

    def features(self,arcs):
        a,b,t=np.asarray(arcs,np.int64).T;d=b-a
        if np.any((a<0)|(b>self.n)|(d<=0)|((t!=0)&(t!=1))):raise ValueError('Invalid interval')
        mean=self.means(a,b,t);variance=np.maximum(0,(self.squared[b,t]-self.squared[a,t])/d[:,None]-mean[:,SELECT]**2)
        fraction=(self.missing[b]-self.missing[a])/d
        parts=[np.c_[np.log1p(d),d/25,d/125,np.sqrt(d)/25,(d>250),fraction,self.support[a],self.support[b-1]],mean,np.sqrt(variance)]
        for width in [5,25,75]:
            parts.extend([self.means(np.maximum(0,a-width),a,t,CONTEXT),self.means(b,np.minimum(self.n,b+width),t,CONTEXT)])
        parts.extend([self.means(a,np.minimum(b,a+5),t,CONTEXT),self.means(np.maximum(a,b-5),b,t,CONTEXT)])
        f=np.concatenate(parts,axis=1).astype(np.float32)
        if not np.isfinite(f).all():raise FloatingPointError('Nonfinite raw interval features')
        return f

    def predict(self,model,arcs,chunk=8192):
        out=np.empty(len(arcs))
        for s in range(0,len(arcs),chunk):out[s:s+chunk]=model.predict(self.features(arcs[s:s+chunk]))
        return np.clip(out,0,1)


def training_proposals(reference,raw,pm,seed,limit=20000):
    """Fitting-only reference jitter plus raw-grid hard negatives; no teacher."""
    n=len(reference);items=[]
    for s,e,t in pm.possession_spells(reference,pm.valid_runs(reference)):
        for ds in [-25,-10,-5,0,5,10,25]:
            for de in [-25,-10,-5,0,5,10,25]:
                a=max(0,min(n-1,s+ds));b=max(a+1,min(n,e+de))
                items.extend([(a,b,t),(a,b,1-t)])
    starts=np.arange(0,n-1,25)
    for duration in [10,25,50,75,125,250,500,1250]:
        ends=np.minimum(n,starts+duration)
        items.extend(np.c_[starts,ends,np.zeros(len(starts),int)].tolist());items.extend(np.c_[starts,ends,np.ones(len(starts),int)].tolist())
    arcs=np.unique(np.asarray(items,np.int64),axis=0)
    # Corrupted training focuses on intervals overlapping raw gap support;
    # clean views retain the same interval distribution without hidden inputs.
    missing=(~np.isfinite(raw['ball']).all(1)).astype(float)
    if missing.any():
        prefix=np.r_[0,np.cumsum(uniform_filter1d(missing,25,mode='nearest')>0)]
        arcs=arcs[(prefix[arcs[:,1]]-prefix[arcs[:,0]])>0]
    if len(arcs)>limit:
        ix=np.random.default_rng(seed).choice(len(arcs),limit,replace=False);arcs=arcs[np.sort(ix)]
    return arcs
