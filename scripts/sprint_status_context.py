"""Train a long asymmetric-context status model using four external matches.

Calibration and decoder choice use WPY. WQQ is inferred only after that freeze;
the independent sprint evaluator scores it in a separate process. No public or
reserved-final labels are allowed in this runner.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from second_zero.evaluate import official_metrics, sha256
from second_zero.contextual import binary_decode
from second_zero.overnight_boundaries import propose_moves, apply_moves
from scripts.evaluate_temporal_ownership import fixed_state_owner

TRAIN = ['J03WMX','J03WN1','J03WOH','J03WOY']
CAL = 'J03WPY'
DEV = 'J03WQQ'
DATA = ROOT/'private/improvement/external/prepared'
FEATURES = ROOT/'runs/live-dead-temporal-v1/features'
B = ROOT/'runs/overnight-boundary-v1'
WIDTHS = (25,125,501)
READS = []


def now(): return datetime.now(timezone.utc).isoformat()


def guard(event,args):
    if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p = os.fsdecode(args[0])
    if 'J03WR9' in p or '/evaluation/gt/' in p:
        raise PermissionError('Public and reserved reference assets excluded from status experiment')
    if p.endswith('.truth.npy'):
        if not any(Path(p).name.startswith(m+'-') for m in TRAIN+[CAL]):
            raise PermissionError('WQQ labels excluded from status training/calibration/inference')
        READS.append(p)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')


def keys(matches):
    return [f'{m}-{h}-{c}' for m in matches for h in ['firstHalf','secondHalf'] for c in ['clean','corrupted']]


class ContextFeatures:
    """Features at any frames without materializing the full expanded timeline."""
    def __init__(self,x):
        self.x=np.asarray(x,np.float32)
        if self.x.ndim!=2 or self.x.shape[1]!=40 or not np.isfinite(self.x).all():
            raise ValueError('Expected existing finite40-channel observations')
        self.prefix=np.r_[np.zeros((1,40)),np.cumsum(self.x,axis=0,dtype=np.float64)]
        self.squared=np.r_[np.zeros((1,18)),np.cumsum(self.x[:,:18].astype(np.float64)**2,axis=0)]

    def rows(self,ix):
        ix=np.asarray(ix,int); n=len(self.x); values=[self.x[ix]]
        for w in WIDTHS:
            lo=np.maximum(0,ix-w); hi=np.minimum(n,ix+w+1)
            # Include current frame in both sides; edge normalization avoids padded artifacts.
            bn=(ix+1-lo)[:,None]; an=(hi-ix)[:,None]
            before=(self.prefix[ix+1]-self.prefix[lo])/bn
            after=(self.prefix[hi]-self.prefix[ix])/an
            bv=(self.squared[ix+1]-self.squared[lo])/bn-before[:,:18]**2
            av=(self.squared[hi]-self.squared[ix])/an-after[:,:18]**2
            values.extend([before,after,after-before,np.sqrt(np.maximum(0,bv)),np.sqrt(np.maximum(0,av))])
        return np.concatenate(values,axis=1).astype(np.float32)


def owner(key):
    if key.startswith(DEV):
        with np.load(ROOT/'runs/fixed-ensemble-v1/A'/f'{key}.npz') as z:
            return z['logit'],z['predicted']
    parts=[]
    with np.load(ROOT/'runs/ownership-temporal-v1'/f'{key}.npz') as z: parts.append(expit(z['logit']))
    for seed in (20261024,20261025):
        folder=ROOT/f'runs/next-evidence-seeds/{seed}-uniform'
        t=json.loads((folder/'freeze.json').read_text())['temperature']
        parts.append(expit(np.load(folder/f'{key}.raw.npy')/t))
    probability=np.mean(parts,axis=0)
    z=logit(np.clip(probability,1e-12,1-1e-12))
    with np.load(B/f'{key}.npz') as source: status=source['predicted']
    return z,fixed_state_owner(status,z)


def infer(model,x):
    ex=ContextFeatures(x); result=np.empty(len(x),np.float64)
    for lo in range(0,len(x),8192):
        hi=min(len(x),lo+8192)
        result[lo:hi]=model.predict_proba(ex.rows(np.arange(lo,hi)))[:,1]
    return logit(np.clip(result,1e-6,1-1e-6))


def decode(logits,own,cost,x,endpoint,models,config):
    state=np.where(binary_decode(np.clip(logits,-np.log(99),np.log(99)),cost),2,0).astype(np.int8)
    if endpoint:
        moves=propose_moves(models,x,state,config['radii'])
        state,_=apply_moves(state,moves,config['thresholds'])
    return fixed_state_owner(state,own)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=250);p.add_argument('--leaves',type=int,default=31)
    p.add_argument('--stride',type=int,default=8);p.add_argument('--reuse-model',type=Path)
    args=p.parse_args();sys.addaudithook(guard);threadpool_limits(2)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    settings={'max_iter':args.iterations,'max_leaf_nodes':args.leaves,'max_depth':7,
              'min_samples_leaf':100,'l2_regularization':10.,'learning_rate':.07,
              'random_state':20261024,'early_stopping':False}
    protocol={'created_utc':now(),'hypothesis':'20-second ordered before/after player and ball motion distinguishes occluded live play from stoppages',
              'train_matches':TRAIN,'calibration_match':CAL,'development_match':DEV,
              'feature_widths_each_side_frames':WIDTHS,'features':'Current40, before/after means and difference40, before/after standard deviations18 at each width; fixed existing observations only.',
              'model':settings,'training_stride':args.stride,'training_boundary_augmentation':'Every second frame within25f status transitions; union with uniform stride.',
              'calibration':'Positive-slope Platt fit on WPY full valid frames, then WPY official spell/macro selection with frozen A owner.',
              'decoder_grid':{'new_weight':[0,.25,.5,.75,1.],'dead_cost':[12.,25.,50.,100.],'B_endpoints':[False,True]},
              'owner':'Unchanged calibrated three-uniform-member A probability mean; cost6.',
              'threads':2,'reserved_final_accessed':False,'source_sha256':sha256(__file__)}
    write(out/'protocol.json',protocol);started=time.monotonic()
    if args.reuse_model:
        with args.reuse_model.open('rb') as f:model=pickle.load(f)
    else:
        xx=[];yy=[];receipts={}
        for key in keys(TRAIN):
            src=FEATURES/f'{key}.npy';x=np.load(src,mmap_mode='r')
            name='-'.join(key.split('-')[:2]);truth_path=DATA/f'{name}.truth.npy';truth=np.load(truth_path)
            change=np.flatnonzero((truth[1:]==2)!=(truth[:-1]==2))+1
            edges=(change[:,None]+np.arange(-25,26,2)).ravel()
            ix=np.unique(np.r_[np.arange(0,len(x),args.stride),np.clip(edges,0,len(x)-1)])
            ix=ix[truth[ix]<3];xx.append(ContextFeatures(x).rows(ix));yy.append(truth[ix]==2)
            receipts[key]={'features_sha256':sha256(src),'truth_sha256':sha256(truth_path),'training_rows':len(ix)}
            print('prepared',key,len(ix),flush=True)
        x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy
        write(out/'training_inputs.json',receipts)
        print('fit',x.shape,flush=True);fit_start=time.monotonic()
        model=HistGradientBoostingClassifier(**settings).fit(x,y)
        write(out/'training.json',{'rows':len(y),'features':x.shape[1],'seconds':time.monotonic()-fit_start,'source_reads':sorted(set(READS))})
        del x,y
    with (out/'model.pkl').open('xb') as f:pickle.dump(model,f)
    with (B/'models.pkl').open('rb') as f:models=pickle.load(f)
    config=json.loads((B/'freeze.json').read_text())['config']
    calibration={};zz=[];yy=[]
    for key in keys([CAL]):
        x=np.load(FEATURES/f'{key}.npy',mmap_mode='r');z=infer(model,x)
        truth=np.load(DATA/('-'.join(key.split('-')[:2])+'.truth.npy'))
        with np.load(B/f'{key}.npz') as s:legacy=logit(np.clip(s['p_dead'],.01,.99))
        own,baseline=owner(key)
        calibration[key]={'raw':z,'truth':truth,'legacy':legacy,'x':x,'owner':own,'baseline':baseline}
        valid=truth<3;zz.append(z[valid]);yy.append((truth[valid]==2).astype(float))
        print('calibration_inference',key,flush=True)
    z=np.concatenate(zz);y=np.concatenate(yy)
    def objective(ab):
        q=z*np.exp(ab[0])+ab[1]
        return float(np.mean(np.logaddexp(0,q)-y*q))
    fit=minimize(objective,[0.,0.],method='L-BFGS-B',bounds=[(-2.3,2.3),(-5,5)])
    if not fit.success:raise RuntimeError('Platt calibration failed:'+str(fit.message))
    slope=float(np.exp(fit.x[0]));intercept=float(fit.x[1]);pm=official_metrics();grid=[]
    for weight in protocol['decoder_grid']['new_weight']:
        for cost in protocol['decoder_grid']['dead_cost']:
            for endpoint in protocol['decoder_grid']['B_endpoints']:
                rows=[]
                for key,r in calibration.items():
                    mixed=weight*(r['raw']*slope+intercept)+(1-weight)*r['legacy']
                    pred=decode(mixed,r['owner'],cost,r['x'],endpoint,models,config)
                    met=pm.evaluate(r['truth'],pred,bootstrap_n=0)
                    rows.append({k:met[k] for k in ['spell_f1_mean','macro_f1']})
                row={'new_weight':weight,'dead_cost':cost,'B_endpoints':endpoint,
                     **{k:float(np.mean([r[k] for r in rows])) for k in ['spell_f1_mean','macro_f1']}}
                grid.append(row)
    best=max(grid,key=lambda r:(r['spell_f1_mean'],r['macro_f1']))
    baseline_metrics=[pm.evaluate(r['truth'],r['baseline']) for r in calibration.values()]
    cal={'match':CAL,'slope':slope,'intercept':intercept,'raw_nll':objective([0,0]),'calibrated_nll':float(fit.fun),
         'grid':grid,'selected':best,'baseline':{k:float(np.mean([r[k] for r in baseline_metrics])) for k in ['spell_f1_mean','macro_f1']}}
    write(out/'calibration.json',cal)
    print('WPY selected',json.dumps(best),'baseline',cal['baseline'],flush=True)
    freeze={'created_utc':now(),'model_sha256':sha256(out/'model.pkl'),'protocol_sha256':sha256(out/'protocol.json'),
            'calibration_sha256':sha256(out/'calibration.json'),'WQQ_labels_accessed':False,
            'slope':slope,'intercept':intercept,'selected':best}
    write(out/'freeze.json',freeze)
    for key in keys([CAL,DEV]):
        if key in calibration:r=calibration[key];x=r['x'];raw=r['raw'];legacy=r['legacy'];own=r['owner'];base=r['baseline']
        else:
            x=np.load(FEATURES/f'{key}.npy',mmap_mode='r');raw=infer(model,x);own,base=owner(key)
            with np.load(B/f'{key}.npz') as s:legacy=logit(np.clip(s['p_dead'],.01,.99))
        mixed=best['new_weight']*(raw*slope+intercept)+(1-best['new_weight'])*legacy
        pred=decode(mixed,own,best['dead_cost'],x,best['B_endpoints'],models,config)
        np.savez_compressed(out/f'{key}.npz',predicted=pred,incumbent=base,logit=own,
                            raw_dead_logit=raw,dead_logit=mixed,p_dead=expit(mixed),ball_seen=x[:,36]>0,
                            player_coverage=x[:,37])
        print('frozen_inference',key,flush=True)
    write(out/'completion.json',{'created_utc':now(),'seconds':time.monotonic()-started,
          'source_reads':sorted(set(READS)),'WQQ_labels_accessed':False,'reserved_final_accessed':False,
          'prediction_sha256':{p.name:sha256(p) for p in out.glob('*.npz')}})


if __name__=='__main__':main()
