"""Label-free match adaptation using only coordinates visible in its input."""
from __future__ import annotations
import hashlib,time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts import sprint_ball_imputer as imputer

MODEL=dict(max_iter=140,max_leaf_nodes=15,max_depth=5,min_samples_leaf=50,learning_rate=.05,l2_regularization=5.,early_stopping=False,random_state=20261030)
BLOCK=2500;MARGIN=250


class ResidualBlend:
    def __init__(self,original,local,alpha):self.original=original;self.local=local;self.alpha=float(alpha)
    def predict(self,x):
        old=self.original.predict(x)
        return old if self.alpha==0 else (1-self.alpha)*np.clip(old,-1.25,1.25)+self.alpha*np.clip(self.local.predict(x),-1.25,1.25)


def pseudo_mask(raw,seed):
    ball=np.asarray(raw['ball']);seen=np.isfinite(ball).all(1);n=len(ball);rng=np.random.default_rng(seed);mask=np.zeros(n,bool)
    edges=np.r_[0,np.flatnonzero(seen[1:]!=seen[:-1])+1,n]
    for lo,hi in zip(edges[:-1],edges[1:]):
        if not seen[lo]:continue
        for block in range(lo//BLOCK,(hi-1)//BLOCK+1):
            a=max(lo,block*BLOCK+MARGIN);b=min(hi,(block+1)*BLOCK-MARGIN)
            lengths=[v for v in [25,50,100,150,200] if v+2<=b-a]
            if not lengths:continue
            length=int(rng.choice(lengths));start=int(rng.integers(a+1,b-length));mask[start:start+length]=True
    assert not np.any(mask&~seen)
    return mask


def examples(raw,mask):
    seen=np.isfinite(raw['ball']).all(1)
    if np.any(mask&~seen):raise ValueError('Pseudo targets must be observed in the supplied input')
    hidden={**raw,'ball':raw['ball'].copy()};hidden['ball'][mask]=np.nan
    x,linear,eligible=imputer.features(hidden);indices=np.flatnonzero(eligible);take=mask[indices]&seen[indices]&(indices%4==0);indices=indices[take]
    target=(raw['ball'][indices]-linear[indices])/20
    return x[take],target.astype(np.float32),indices


def fit_axes(x,y):return [HistGradientBoostingRegressor(**MODEL).fit(x,y[:,axis]) for axis in [0,1]]


def adapt(raw,original):
    """Out-of-block coordinate CV chooses the blend; no possession labels API."""
    started=time.monotonic();xx=[];yy=[];ii=[];digests=[]
    for seed in [20261030,20261031,20261032]:
        mask=pseudo_mask(raw,seed);x,y,indices=examples(raw,mask);xx.append(x);yy.append(y);ii.append(indices);digests.append(hashlib.sha256(mask.tobytes()).hexdigest())
    x=np.concatenate(xx);y=np.concatenate(yy);indices=np.concatenate(ii);groups=(indices//BLOCK)%2
    seen=np.isfinite(raw['ball']).all(1);assert seen[indices].all() and np.isfinite(y).all()
    if len(x)<1000 or min(int((groups==f).sum()) for f in [0,1])<300:
        return original,{'alpha':0.,'rows':len(x),'reason':'insufficient observed pseudo gaps','input_only_observed_targets':True,'mask_sha256':digests,'seconds':time.monotonic()-started}
    old=np.column_stack([m.predict(x) for m in original]);local=np.empty_like(old);fits=[]
    for fold in [0,1]:
        train=groups!=fold;test=groups==fold;models=fit_axes(x[train],y[train]);local[test]=np.column_stack([m.predict(x[test]) for m in models]);fits.append({'fit_rows':int(train.sum()),'validation_rows':int(test.sum()),'fit_blocks_parity':1-fold,'validation_blocks_parity':fold})
    # Match the actual bounded residual geometry. Original pipeline applies a
    # fixed .75 residual in the ownership path and full residual physically.
    old=np.clip(old,-1.25,1.25);local=np.clip(local,-1.25,1.25);d=local-old
    alpha=float(np.clip(np.sum(d*(y-old),dtype=np.float64)/max(float(np.sum(d*d,dtype=np.float64)),1e-12),0,1));mixed=(1-alpha)*old+alpha*local
    models=fit_axes(x,y) if alpha>0 else [None,None]
    result=[ResidualBlend(o,m,alpha) for o,m in zip(original,models)]
    stats={'alpha':alpha,'rows':len(x),'target_observed_frames':int(np.unique(indices).size),'input_only_observed_targets':True,'no_possession_or_status_target':True,'mask_sha256':digests,'coordinate_CV':{'baseline_RMSE_m':float(20*np.sqrt(np.mean((old-y)**2))),'local_RMSE_m':float(20*np.sqrt(np.mean((local-y)**2))),'mixture_RMSE_m':float(20*np.sqrt(np.mean((mixed-y)**2)))},'block_fits':fits,'model':MODEL,'seconds':time.monotonic()-started,'raw_input_ball_sha256':hashlib.sha256(np.asarray(raw['ball']).tobytes()).hexdigest(),'original_missing_targets':int((~seen[indices]).sum())}
    return result,stats
