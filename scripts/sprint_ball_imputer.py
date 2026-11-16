"""Learn bounded missing-ball geometry from observed endpoints and player motion.

This is supervised position imputation, never control-label inference. Training
position references are confined to the four existing training matches. WPY
measures imputation error; WQQ/final/public references are not read here.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from second_zero.contextual import velocity
from second_zero.common import json_write,sha256
from scripts.train_contextual import corruption

DATA=ROOT/'private/improvement/external/prepared'
TRAIN=['J03WMX','J03WN1','J03WOH','J03WOY']
MAX_GAP=250

def features(a):
    ball=np.asarray(a['ball'],np.float32);players=np.asarray(a['players'],np.float32)
    n=len(ball);ix=np.arange(n);seen=np.isfinite(ball).all(1)
    prev=np.maximum.accumulate(np.where(seen,ix,-n));nxt=np.minimum.accumulate(np.where(seen,ix,2*n)[::-1])[::-1]
    span=nxt-prev;eligible=(~seen)&(prev>=0)&(nxt<n)&(span<=MAX_GAP)
    rows=np.flatnonzero(eligible);p=prev[rows];q=nxt[rows]
    if not len(rows):return np.empty((0,60),np.float32),ball.copy(),eligible
    fraction=(rows-p)/(q-p);elapsed=(q-p)/25.
    bp=ball[p];bq=ball[q];delta=bq-bp
    estimate=bp+(bq-bp)*fraction[:,None];linear=ball.copy();linear[rows]=estimate
    before=ball[np.maximum(p-5,0)];after=ball[np.minimum(q+5,n-1)]
    vp=(bp-before)*5.;vq=(after-bq)*5.;pathv=delta/elapsed[:,None]
    current=players[rows];pv=velocity(players)[rows]
    distance=np.linalg.norm(current-estimate[:,None],axis=2)
    order=np.argsort(np.nan_to_num(distance,nan=np.inf),axis=1)[:,:8]
    rr=np.arange(len(rows))[:,None]
    relative=(current-estimate[:,None])[rr,order]
    speed=(pv-pathv[:,None])[rr,order]
    d=distance[rr,order]
    pitch=np.asarray(a['pitch'])
    parts=[estimate/pitch,bp/pitch,bq/pitch,delta/50,vp/40,vq/40,pathv/40,
           fraction[:,None],(elapsed/10)[:,None],(fraction*(1-fraction))[:,None],
           relative.reshape(len(rows),-1)/30,speed.reshape(len(rows),-1)/20,d/30,
           (d<2).sum(1)[:,None]/8,(d<5).sum(1)[:,None]/8,
           np.isfinite(d).sum(1)[:,None]/8]
    x=np.column_stack(parts).astype(np.float32);x[np.isinf(x)]=np.nan
    return x,linear,eligible

def fit(folder):
    folder.mkdir(parents=True,exist_ok=False)
    settings=dict(max_iter=240,max_leaf_nodes=15,max_depth=5,min_samples_leaf=80,
        learning_rate=.06,l2_regularization=5.,early_stopping=False,random_state=20261024,loss='squared_error')
    protocol={'kind':'supervised missing-ball coordinate residual','train_matches':TRAIN,
       'seeds':[100,271],'fit_stride_missing':4,'maximum_gap_frames':MAX_GAP,'features':'measured ball endpoints/velocities, interpolation fraction and sorted anonymous nearest8player relative positions/velocities',
       'reference':'raw observed training positions only','validation':'WPY only, no possession/status labels',
       'models':settings,'source_sha256':sha256(Path(__file__))}
    json_write(folder/'protocol.json',protocol);(folder/'source.py').write_text(Path(__file__).read_text())
    xx=[];yy=[];hashes={};started=time.perf_counter()
    for m in TRAIN:
        for h in ['firstHalf','secondHalf']:
            path=DATA/f'{m}-{h}.npz';hashes[str(path)]=sha256(path)
            with np.load(path) as z:raw={k:z[k] for k in z.files}
            for seed in [100,271]:
                a=corruption(raw,seed);x,linear,eligible=features(a);y=(raw['ball'][eligible]-linear[eligible])/20
                keep=np.arange(0,len(x),4);keep=keep[np.isfinite(y[keep]).all(1)]
                xx.append(x[keep]);yy.append(y[keep]);print('IMPUTER prepared',m,h,seed,len(keep),flush=True)
    x=np.concatenate(xx);y=np.concatenate(yy);models=[]
    for axis in (0,1):
        model=HistGradientBoostingRegressor(**settings).fit(x,y[:,axis]);models.append(model)
        print('IMPUTER axis fitted',axis,flush=True)
    with (folder/'models.pkl').open('wb') as f:pickle.dump(models,f)
    json_write(folder/'training.json',{'shape':list(x.shape),'seconds':time.perf_counter()-started,'input_sha256':hashes})
    json_write(folder/'freeze.json',{'models_sha256':sha256(folder/'models.pkl'),'protocol_sha256':sha256(folder/'protocol.json')})

def predict(a,models,blend=1.):
    x,linear,eligible=features(a)
    if len(x):
        correction=np.stack([m.predict(x) for m in models],axis=1)*20
        correction=np.clip(correction,-25,25)
        linear[eligible]+=blend*correction
    return linear,eligible

def evaluate(folder):
    with (folder/'models.pkl').open('rb') as f:models=pickle.load(f)
    result={};all_errors={str(w):[] for w in (0.,.25,.5,.75,1.)}
    for h in ['firstHalf','secondHalf']:
        path=DATA/f'J03WPY-{h}.npz'
        with np.load(path) as z:raw={k:z[k] for k in z.files}
        a=corruption(raw,100);result[h]={}
        for w in (0.,.25,.5,.75,1.):
            estimated,eligible=predict(a,models,w)
            error=np.linalg.norm(estimated[eligible]-raw['ball'][eligible],axis=1)
            all_errors[str(w)].append(error)
            result[h][str(w)]={'frames':len(error),'mean_m':float(np.mean(error)),'median_m':float(np.median(error)),'p90_m':float(np.quantile(error,.9)),'within2m':float(np.mean(error<=2))}
    pooled={}
    for w,err in all_errors.items():
        e=np.concatenate(err);pooled[w]={'mean_m':float(np.mean(e)),'median_m':float(np.median(e)),'p90_m':float(np.quantile(e,.9)),'within2m':float(np.mean(e<=2))}
    json_write(folder/'WPY-position-evaluation.json',{'rows':result,'pooled':pooled,'scope':'geometry validation only; no ownership/status scores'})
    print(json.dumps(pooled,indent=2),flush=True)

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if 'J03WR9' in p or 'J03WQQ' in p or '.truth.npy' in p or '/evaluation/gt/' in p:raise PermissionError('Imputer excludes labels and WQQ/final '+p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-matches',nargs='+',choices=TRAIN,default=TRAIN)
    a=p.parse_args();TRAIN=a.train_matches;sys.addaudithook(guard);fit(a.out);evaluate(a.out)
