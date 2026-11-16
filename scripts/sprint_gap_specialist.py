"""Direct symmetric control classification inside bounded observation gaps.

Uses observed endpoint geometry and uncertain linear/learned position hypotheses.
Neither clean hidden coordinates nor other learned owner outputs enter features.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from second_zero.common import json_write,sha256
from second_zero.evaluate import official_metrics
from second_zero.sprint_model_interpolation import interpolated_features
from scripts.sprint_ball_smooth import predict_smoothed
from scripts.train_contextual import corruption
from scripts.evaluate_temporal_ownership import fixed_state_owner
from scripts.sprint_stack import keys,TRAIN,truth,timeguard
DATA=ROOT/'private/improvement/external/prepared'
FEATURES=ROOT/'runs/ownership-temporal-v1/features'

def features(key,imputers):
    name,condition=key.rsplit('-',1)
    with np.load(DATA/(name+'.npz')) as z:a={k:z[k] for k in z.files}
    if condition=='corrupted':a=corruption(a,100)
    x=np.load(FEATURES/(key+'.npy'),mmap_mode='r');n=len(x);ix=np.arange(n)
    seen=np.isfinite(a['ball']).all(1);prev=np.maximum.accumulate(np.where(seen,ix,-n));nxt=np.minimum.accumulate(np.where(seen,ix,2*n)[::-1])[::-1]
    filled,eligible=predict_smoothed(a,imputers);rows=np.flatnonzero(eligible)
    if not len(rows):return None,eligible
    linear=interpolated_features(a['ball'],a['players'],a['team'],a['pitch'])
    learned=interpolated_features(filled,a['players'],a['team'],a['pitch']);learned[:,:,12:17]=linear[:,:,12:17]
    out=[]
    for t in (0,1):
        pieces=[linear[rows,t],learned[rows,t]]
        for endpoint in [prev[rows],nxt[rows],np.maximum(prev[rows]-25,0),np.minimum(nxt[rows]+25,n-1)]:
            pieces.extend([x[endpoint,t,:28],x[endpoint,t,49:77]])
        # Anonymous current player support near either observed endpoint, from
        # this team and the opponent. No player identities are carried through.
        for endpoint in [prev[rows],nxt[rows]]:
            for team in (t,1-t):
                pts=a['players'][rows][:,a['team']==team]
                dist=np.linalg.norm(pts-a['ball'][endpoint,None],axis=2)
                d=np.sort(np.nan_to_num(dist,nan=100),axis=1)[:,:3]/50
                pieces.append(d)
        out.append(np.column_stack(pieces).astype(np.float32))
    return np.stack(out,axis=1),eligible

def predict(model,f):
    p=[model.predict_proba(f[:,t])[:,1] for t in (0,1)]
    return .5*(logit(np.clip(p[1],1e-6,1-1e-6))-logit(np.clip(p[0],1e-6,1-1e-6)))

def fit(out,fold):
    train=TRAIN if fold=='full' else TRAIN[2:] if fold=='0' else TRAIN[:2]
    evaluation=['J03WQQ'] if fold=='full' else TRAIN[:2] if fold=='0' else TRAIN[2:]
    imputer=ROOT/'runs/two-hour-sprint'/('ball-imputer-v1' if fold=='full' else 'ball-imputer-fold'+fold)
    with (imputer/'models.pkl').open('rb') as h:imputers=pickle.load(h)
    assert json.loads((imputer/'protocol.json').read_text())['train_matches']==train
    out.mkdir(parents=True,exist_ok=False)
    protocol={'train_matches':train,'evaluation_matches':evaluation,'fold':fold,'calibration':'WPYtemperature, eligibleliveframesonly',
        'imputer':str(imputer),'imputer_sha256':sha256(imputer/'models.pkl'),'samples':'everyothereligiblemissingliveframe,bothteamperspectives',
        'features':'linear/smoothedlearnedballhypotheses, observedprev/nextgeometry +/-25frames, anonymous currentteamdistance toobservedanchors',
        'no_other_owner_models_as_features':True,'source_sha256':sha256(Path(__file__))}
    json_write(out/'protocol.json',protocol);(out/'source.py').write_text(Path(__file__).read_text())
    xx=[];yy=[];started=time.perf_counter()
    for key in keys(train):
        if key.endswith('clean'):continue
        timeguard();f,mask=features(key,imputers);y=truth(key)[mask];ix=np.arange(0,len(y),2);ix=ix[y[ix]<2]
        for t in (0,1):xx.append(f[ix,t]);yy.append(y[ix]==t)
        print('GAP prepared',key,len(ix)*2,flush=True)
    x=np.concatenate(xx);y=np.concatenate(yy)
    model=HistGradientBoostingClassifier(max_iter=260,max_leaf_nodes=15,max_depth=5,min_samples_leaf=80,
        l2_regularization=8.,learning_rate=.06,early_stopping=False,random_state=20261024).fit(x,y)
    with (out/'model.pkl').open('wb') as h:pickle.dump(model,h)
    json_write(out/'training.json',{'shape':list(x.shape),'seconds':time.perf_counter()-started})
    zz=[];ys=[]
    for key in keys(['J03WPY']):
        timeguard();f,mask=features(key,imputers);raw=np.zeros(len(mask),np.float32)
        if f is not None:raw[mask]=predict(model,f)
        np.savez_compressed(out/(key+'.npz'),raw_logit=raw,eligible=mask)
        y=truth(key);v=mask&(y<2);zz.append(raw[v]);ys.append(y[v])
    z=np.concatenate(zz);y=np.concatenate(ys)
    opt=minimize_scalar(lambda lt:np.mean(np.logaddexp(0,z/np.exp(lt))-y*z/np.exp(lt)),bounds=(-2.3,2.3),method='bounded')
    temp=float(np.exp(opt.x));json_write(out/'freeze.json',{'temperature':temp,'model_sha256':sha256(out/'model.pkl'),'protocol_sha256':sha256(out/'protocol.json')})
    for key in keys(evaluation):
        timeguard();f,mask=features(key,imputers);raw=np.zeros(len(mask),np.float32)
        if f is not None:raw[mask]=predict(model,f)
        np.savez_compressed(out/(key+'.npz'),raw_logit=raw,logit=raw/temp,eligible=mask)
        print('GAP predicted',key,flush=True)
    return evaluation

def evaluate(out,matches,fold):
    frozen={str(out/(k+'.npz')):sha256(out/(k+'.npz')) for k in keys(matches)};json_write(out/'pre_score_freeze.json',frozen)
    pm=official_metrics();rows={};outputs={str(w):{} for w in (.25,.5,.75,1.)}
    basefolder=ROOT/('runs/fixed-ensemble-v1/A' if fold=='full' else 'runs/oof-uniform-ensemble-v1/ensemble')
    for key in keys(matches):
        with np.load(basefolder/(key+'.npz')) as z:base=z['predicted'];old=z['logit']
        with np.load(out/(key+'.npz')) as z:gap=z['logit'];mask=z['eligible']
        y=truth(key);rows[key]={}
        for w in (0.,.25,.5,.75,1.):
            p=expit(old);p[mask]=(1-w)*p[mask]+w*expit(gap[mask]);z=logit(np.clip(p,1e-8,1-1e-8));pred=fixed_state_owner(base,z)
            metric=pm.evaluate(y,pred,bootstrap_n=0);rows[key][str(w)]={m:float(metric[m]) for m in ['spell_f1_mean','macro_f1']}
            if w:outputs[str(w)][key]=(pred,z)
    means={view:{str(w):{m:float(np.mean([rows[k][str(w)][m] for k in rows if view=='overall' or k.endswith(view)])) for m in ['spell_f1_mean','macro_f1']}
       for w in (0.,.25,.5,.75,1.)} for view in ['overall','clean','corrupted']}
    json_write(out/'evaluation.json',{'rows':rows,'means':means,'interpretation':'adaptiveexternaldevelopment' if fold=='full' else 'crossmatchOOFdevelopment'})
    for w,records in outputs.items():
        dest=out/('blend'+w);dest.mkdir()
        for key,(p,z) in records.items():np.savez_compressed(dest/(key+'.npz'),predicted=p,logit=z)
    print(json.dumps(means,indent=2),flush=True)

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if 'J03WR9' in p or '/evaluation/gt/' in p:raise PermissionError('No public/final in gap experiment')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fold',choices=['full','0','1'],default='full');a=p.parse_args()
    sys.addaudithook(guard);matches=fit(a.out,a.fold);evaluate(a.out,matches,a.fold)
