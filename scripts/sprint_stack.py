"""Two-hour sprint: symmetric OOF-trained temporal ownership stacking.

The input models' training matches exclude each stacking training example's
whole match. WPY calibrates temperatures; WQQ is explicitly adaptive development.
No public or reserved-final labels are loaded by this script.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, json, pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from second_zero.common import sha256, json_write
from second_zero.evaluate import official_metrics
from scripts.evaluate_temporal_ownership import fixed_state_owner

RUN=ROOT/'runs/two-hour-sprint/stack'
REPORT=ROOT/'reports/two-hour-sprint/stack'
DATA=ROOT/'private/improvement/external/prepared'
FEATURES=ROOT/'runs/ownership-temporal-v1/features'
TRAIN=['J03WMX','J03WN1','J03WOH','J03WOY']
OOF=json.loads((ROOT/'runs/oof-uniform-ensemble-v1/protocol.json').read_text())['models']
FULL={
 'uniform20261023':'runs/ownership-temporal-v1',
 'uniform20261024':'runs/next-evidence-seeds/20261024-uniform',
 'uniform20261025':'runs/next-evidence-seeds/20261025-uniform',
 'short25':'runs/extended-short-v2'}
ARMS=list(FULL)
EXTRAS={}
EXTRAS_ADDITIONAL=[]
DEADLINE=datetime.fromisoformat('2026-10-24T11:25:00+00:00').timestamp()

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        s=os.fsdecode(args[0])
        if 'J03WR9' in s or '/evaluation/gt/' in s:
            raise PermissionError('Sprint training excludes public/final labels: '+s)

def timeguard():
    if time.time()>DEADLINE: raise TimeoutError('Experiment deadline reached')

def keys(matches):
    return [f'{m}-{h}-{c}' for m in matches for h in ['firstHalf','secondHalf'] for c in ['clean','corrupted']]

def truth(key):
    return np.load(DATA/('-'.join(key.split('-')[:2])+'.truth.npy'))

def model_logits(folder,key):
    folder=ROOT/folder
    path=folder/(key+'.npz')
    if path.exists():
        with np.load(path) as z:
            if 'logit' in z:return z['logit'].astype(np.float32)
    temp_path=folder/'freeze.json' if (folder/'freeze.json').exists() else folder/'calibration.json'
    t=json.loads(temp_path.read_text())['temperature']
    if path.exists():
        with np.load(path) as z:return (z['raw_logit']/t).astype(np.float32)
    return (np.load(folder/(key+'.raw.npy'))/t).astype(np.float32)

def inputs(key):
    match=key.split('-')[0]
    models=OOF[str(0 if match in TRAIN[:2] else 1)] if match in TRAIN else FULL
    z=np.stack([model_logits(models[arm],key) for arm in ARMS],axis=1)
    for extra_mapping in ([EXTRAS] if EXTRAS else [])+EXTRAS_ADDITIONAL:
        extra=extra_mapping[str(0 if match in TRAIN[:2] else 1)] if match in TRAIN else extra_mapping['full']
        z=np.column_stack([z,model_logits(extra,key)])
    x=np.load(FEATURES/(key+'.npy'),mmap_mode='r')
    return x,z

def shifted(a,offset):
    return a[np.clip(np.arange(len(a))+offset,0,len(a)-1)]

def perspective_features(x,z,team,geometry=True):
    """All channels refer to one team; shared scorer guarantees exchange symmetry."""
    sign=1 if team else -1
    q=np.clip(sign*z,-8,8).astype(np.float32)
    p=expit(q); mean=p[:,:3].mean(axis=1,keepdims=True)
    parts=[q,p,mean,p.std(axis=1,keepdims=True)]
    for width in (7,25,75,151,301):
        parts.extend([uniform_filter1d(q,width,axis=0,mode='nearest'),
                      uniform_filter1d(p,width,axis=0,mode='nearest')])
    for offset in (-150,-75,-25,25,75,150): parts.append(shifted(mean,offset))
    # Endpoint memory bridges *measured* missing trajectories without inventing
    # a ball location. Include both directions because task is offline.
    valid=(x[:,0,49]<.5)&(x[:,0,50]<.5)
    ix=np.arange(len(x)); prev=np.maximum.accumulate(np.where(valid,ix,0))
    nxt=np.minimum.accumulate(np.where(valid,ix,len(x)-1)[::-1])[::-1]
    parts.extend([mean[prev],mean[nxt],np.minimum(ix-prev,500)[:,None]/500,
                  np.minimum(nxt-ix,500)[:,None]/500,valid[:,None]])
    if geometry:
        # Current own/opponent distances, relative velocities, neighbor support,
        # availability and control-like kinematics, in original physical scales.
        sel=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,15,18,21,24,25,26,27,
                      28,29,30,31,34,37,40,43,44,45,46,47,48])
        raw=np.array(x[:,team,sel],dtype=np.float32)
        missing=np.asarray(x[:,team,sel+49],dtype=np.float32)
        parts.extend([raw,missing])
        for width in (25,101):
            parts.append(uniform_filter1d(raw,width,axis=0,mode='nearest'))
        # Direction of approach/control relative to the local center.
        avg=uniform_filter1d(raw[:,:9],51,axis=0,mode='nearest')
        parts.extend([shifted(avg,-25),shifted(avg,25)])
    return np.column_stack(parts).astype(np.float32)

def predict(model,x,z,geometry):
    scores=[]
    for t in (0,1):
        xx=perspective_features(x,z,t,geometry)
        p=model.predict_proba(xx)[:,1]
        scores.append(logit(np.clip(p,1e-6,1-1e-6)))
    return ((scores[1]-scores[0])*.5).astype(np.float32)

def prepare_training(geometry,weighted):
    xx=[];yy=[];ww=[]
    for key in keys(TRAIN):
        timeguard(); x,z=inputs(key); y=truth(key)
        ix=np.arange(0,len(y),8); ix=ix[y[ix]<2]
        durations=np.ones(len(y),np.float32)
        edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
        for a,b in zip(edges[:-1],edges[1:]): durations[a:b]=b-a
        weights=np.clip(np.sqrt(250/durations[ix]),.5,3) if weighted else np.ones(len(ix))
        for t in (0,1):
            f=perspective_features(x,z,t,geometry)
            xx.append(f[ix]); yy.append(y[ix]==t); ww.append(weights)
        print('Prepared',key, len(ix)*2,flush=True)
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(ww)

def train(name,geometry,weighted):
    folder=RUN/name; folder.mkdir(parents=True,exist_ok=False)
    settings=dict(max_iter=220,max_leaf_nodes=15,max_depth=5,min_samples_leaf=100,
                  l2_regularization=8.,learning_rate=.06,random_state=20261024,early_stopping=False)
    protocol={'name':name,'geometry':geometry,'duration_weighted':weighted,'training_matches':TRAIN,
              'base_predictions':'Whole-match-out-of-fold, never in-sample neural outputs',
              'calibration':'WPY only temperature','development':'WQQ repeatedly consulted',
              'models':OOF,'full_models':FULL,'hgb':settings,'sample_stride':8,
              'extra_models':EXTRAS,
              'extra_model_list':([EXTRAS] if EXTRAS else [])+EXTRAS_ADDITIONAL,
              'symmetry':'shared perspective scorer, half logit(teamA)-logit(teamH)',
              'source_sha256':sha256(Path(__file__))}
    json_write(folder/'protocol.json',protocol)
    (folder/'source.py').write_text(Path(__file__).read_text())
    started=time.perf_counter(); x,y,w=prepare_training(geometry,weighted)
    print('Fitting',name,x.shape,flush=True)
    model=HistGradientBoostingClassifier(**settings).fit(x,y,sample_weight=w)
    with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f)
    json_write(folder/'training.json',{'seconds':time.perf_counter()-started,'shape':list(x.shape)})
    del x,y,w
    zz=[];ys=[]
    for key in keys(['J03WPY']):
        timeguard(); x,z=inputs(key); raw=predict(model,x,z,geometry); y=truth(key);v=y<2
        np.savez_compressed(folder/(key+'.npz'),raw_logit=raw)
        zz.append(raw[v]);ys.append(y[v]);print('Calibrating',name,key,flush=True)
    z=np.concatenate(zz);y=np.concatenate(ys)
    opt=minimize_scalar(lambda lt:np.mean(np.logaddexp(0,z/np.exp(lt))-y*z/np.exp(lt)),bounds=(-2.3,2.3),method='bounded')
    temp=float(np.exp(opt.x));json_write(folder/'freeze.json',{'temperature':temp,'model_sha256':sha256(folder/'model.pkl'),'protocol_sha256':sha256(folder/'protocol.json')})
    for key in keys(['J03WQQ']):
        timeguard();x,z=inputs(key);raw=predict(model,x,z,geometry)
        np.savez_compressed(folder/(key+'.npz'),raw_logit=raw,logit=raw/temp)
        print('Predicted',name,key,flush=True)
    return folder

def evaluate(folder):
    # All four development predictions are complete and hashed before labels.
    frozen={str(folder/(k+'.npz')):sha256(folder/(k+'.npz')) for k in keys(['J03WQQ'])}
    json_write(folder/'pre_score_freeze.json',frozen)
    pm=official_metrics(); rows={}; outputs={}
    for key in keys(['J03WQQ']):
        with np.load(ROOT/'runs/fixed-ensemble-v1/A'/(key+'.npz')) as a:
            base=a['predicted'];zbase=a['logit']
        with np.load(folder/(key+'.npz')) as a:z=a['logit']
        y=truth(key); rows[key]={}
        for weight in (0.,.25,.5,.75,1.):
            p=(1-weight)*expit(zbase)+weight*expit(z)
            blended=logit(np.clip(p,1e-8,1-1e-8))
            pred=fixed_state_owner(base,blended)
            m=pm.evaluate(y,pred,bootstrap_n=0)
            rows[key][str(weight)]={k:float(m[k]) for k in ['spell_f1_mean','macro_f1']}
            if weight: outputs.setdefault(str(weight),{})[key]=(pred,blended)
    means={view:{str(w):{m:float(np.mean([rows[k][str(w)][m] for k in rows if view=='overall' or k.endswith(view)]))
           for m in ['spell_f1_mean','macro_f1']} for w in (0.,.25,.5,.75,1.)} for view in ['overall','clean','corrupted']}
    json_write(REPORT/(folder.name+'-results.json'),{'rows':rows,'means':means,'selection_use':'Adaptive WQQ development; not untouched test'})
    for weight,records in outputs.items():
        dest=folder/('blend'+weight);dest.mkdir()
        json_write(dest/'protocol.json',{'parent':str(folder),'weight':float(weight),'base':'ensemble A','freeze_sha256':sha256(folder/'pre_score_freeze.json')})
        for key,(pred,z) in records.items():np.savez_compressed(dest/(key+'.npz'),predicted=pred,logit=z)
    print(json.dumps({'name':folder.name,'means':means},indent=2),flush=True)

def main():
    global EXTRAS
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--no-geometry',action='store_true');p.add_argument('--weighted',action='store_true');p.add_argument('--score-only',action='store_true')
    p.add_argument('--extra-full');p.add_argument('--extra-fold0');p.add_argument('--extra-fold1');a=p.parse_args()
    if a.extra_full or a.extra_fold0 or a.extra_fold1:
        if not(a.extra_full and a.extra_fold0 and a.extra_fold1):raise ValueError('Extra model needs full and both disjoint folds')
        EXTRAS={'full':a.extra_full,'0':a.extra_fold0,'1':a.extra_fold1}
    sys.addaudithook(guard);RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    folder=RUN/a.name if a.score_only else train(a.name,not a.no_geometry,a.weighted)
    evaluate(folder)

if __name__=='__main__':main()
