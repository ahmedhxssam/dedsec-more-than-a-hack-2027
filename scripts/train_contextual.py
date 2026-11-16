"""Train/select only on the fixed independent Sportec split; public GT is never read."""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from second_zero.contextual import features,predict_probabilities,decode_probabilities
from second_zero.common import json_write,sha256
from second_zero.evaluate import official_metrics

DATA=ROOT/'private/improvement/external/prepared'
OUT=ROOT/'runs/contextual-external-v1'
SPLIT=ROOT/'private/improvement/external/split.lock.json'
GRID=[{'team_cost':a,'dead_cost':b} for a in [6.,18.,50.] for b in [6.,18.,50.]]

def load(name):
    with np.load(DATA/(name+'.npz')) as z:a={k:z[k] for k in z.files}
    return a,np.load(DATA/(name+'.truth.npy'))

def corruption(a,seed):
    rng=np.random.default_rng(seed);a={k:v.copy() for k,v in a.items()}
    n=len(a['ball'])
    # Fixed contiguous occlusions independent of state labels, plus smooth noise.
    for start in range(150,n,375):
        length=int(rng.integers(25,201));a['ball'][start:start+length]=np.nan
    a['ball']+=gaussian_filter1d(rng.normal(0,.5,a['ball'].shape).astype(np.float32),3,axis=0)
    for i in range(a['players'].shape[1]):
        for start in range(int(rng.integers(0,2500)),n,2500):a['players'][start:start+100,i]=np.nan
    return a

def fextract(a):return features(a['ball'],a['players'],a['team'],a['pitch'])

def metric(pm,y,p):
    m=pm.evaluate(y,p,bootstrap_n=0)
    return {k:m[k] for k in ['spell_f1_mean','macro_f1','per_class','n_gt_spells','n_pred_spells','transitions']}

def baseline(a,halfwidth):
    d=np.linalg.norm(a['players']-a['ball'][:,None],axis=2)
    d[~np.isfinite(d)]=np.inf
    labels=a['team'][d.argmin(axis=1)].astype(np.int8)
    labels[(~np.isfinite(a['ball']).all(axis=1))|(~np.isfinite(d).any(axis=1))]=2
    if halfwidth:
        # Organizer code votes H versus A only: D survives only if neither is
        # observed anywhere in the window. Use exact integer cumulative counts.
        ix=np.arange(len(labels));lo=np.maximum(0,ix-halfwidth);hi=np.minimum(len(labels),ix+halfwidth+1)
        h=np.r_[0,np.cumsum(labels==0)];v=np.r_[0,np.cumsum(labels==1)]
        h=h[hi]-h[lo];v=v[hi]-v[lo]
        labels=np.where(h+v==0,2,np.where(h>=v,0,1)).astype(np.int8)
    return labels

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--out',type=Path,default=OUT);p.add_argument('--train-only',action='store_true');a=p.parse_args()
    OUT=a.out
    split=json.loads(SPLIT.read_text())['split'];pm=official_metrics();OUT.mkdir(parents=True,exist_ok=True)
    if a.final:
        freeze=json.loads((OUT/'freeze.json').read_text())
        if sha256(OUT/'models.pkl')!=freeze['models_sha256']:raise ValueError('Frozen model changed')
        if (OUT/'final.json').exists():raise ValueError('Final audit already used; refusing repeated final optimization')
        with (OUT/'models.pkl').open('rb') as s:models=pickle.load(s)
        scores={}
        for match in split['final']:
            for half in ['firstHalf','secondHalf']:
                name=f'{match}-{half}';raw,y=load(name)
                for variant,inputs in [('clean',raw),('corrupted',corruption(raw,100))]:
                    f=fextract(inputs);dp,op=predict_probabilities(models,f)
                    pred=decode_probabilities(dp,op,**freeze['selected'])
                    key=name+'-'+variant
                    scores[key]={'contextual':metric(pm,y,pred),'baselines':{f'smooth{w}':metric(pm,y,baseline(inputs,w)) for w in [0,25,50,125]}}
        json_write(OUT/'final.json',{'reference':'External original provider labels, held-out whole match; not challenge accuracy','scores':scores,'freeze_sha256':sha256(OUT/'freeze.json')})
        print('Completed one frozen external final evaluation.',flush=True);return
    protocol={'version':1,'split_sha256':sha256(SPLIT),'grid':GRID,
              'selection':'Equal mean official spellF1 across clean and corrupted development halves; macroF1 tie-break.',
              'training_sample_stride':5,'corruption_seed':100,
              'hgb':{'max_iter':160,'max_leaf_nodes':15,'max_depth':5,'min_samples_leaf':80,'l2_regularization':5.,'learning_rate':.08,'random_state':20261022,'early_stopping':False},
              'source_sha256':sha256(ROOT/'second_zero/contextual.py'),'public_reference_access':False}
    if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text())!=protocol:raise ValueError('Protocol changed; use a new experiment directory')
    json_write(OUT/'protocol.json',protocol)
    if not (OUT/'models.pkl').exists():
        xx={k:[] for k in ['dead','player','owner']};yy={k:[] for k in xx}
        for match in split['train']:
            for half in ['firstHalf','secondHalf']:
                name=f'{match}-{half}';raw,y=load(name)
                for variant,inputs in [('clean',raw),('corrupted',corruption(raw,100))]:
                    f=fextract(inputs);idx=np.arange(0,len(y),5);idx=idx[y[idx]!=3]
                    for key in ['dead','player']:xx[key].append(f[key][idx]);yy[key].append(y[idx]==2)
                    live=idx[(y[idx]<2)&f['ball_seen'][idx]]
                    for team in [0,1]:xx['owner'].append(f['owner'][live,team]);yy['owner'].append(y[live]==team)
                print('Prepared external training half',name,flush=True)
        models={}
        for key in xx:
            x=np.concatenate(xx[key]);y=np.concatenate(yy[key]);xx[key]=[];yy[key]=[]
            models[key]=HistGradientBoostingClassifier(**protocol['hgb']).fit(x,y)
            print('Trained',key,'on',len(y),'external samples',flush=True)
        with (OUT/'models.pkl').open('wb') as s:pickle.dump(models,s)
    else:
        with (OUT/'models.pkl').open('rb') as s:models=pickle.load(s)
    if a.train_only:
        print(json.dumps({'training_complete':True,'models_sha256':sha256(OUT/'models.pkl'),'development_or_final_opened':False}),flush=True)
        return
    scores={};base_scores={}
    for match in split['development']:
        for half in ['firstHalf','secondHalf']:
            name=f'{match}-{half}';raw,y=load(name)
            for variant,inputs in [('clean',raw),('corrupted',corruption(raw,100))]:
                key=name+'-'+variant;f=fextract(inputs);dp,op=predict_probabilities(models,f)
                np.savez_compressed(OUT/(key+'.npz'),dead=dp,owner=op)
                scores[key]=[]
                for settings in GRID:scores[key].append(metric(pm,y,decode_probabilities(dp,op,**settings)))
                base_scores[key]={f'smooth{w}':metric(pm,y,baseline(inputs,w)) for w in [0,25,50,125]}
                print('Scored external development',key,flush=True)
    aggregate=[{'config':g,'spell_f1_mean':float(np.mean([v[i]['spell_f1_mean'] for v in scores.values()])),
                'macro_f1':float(np.mean([v[i]['macro_f1'] for v in scores.values()]))} for i,g in enumerate(GRID)]
    chosen=max(aggregate,key=lambda x:(x['spell_f1_mean'],x['macro_f1']))
    json_write(OUT/'development.json',{'aggregate':aggregate,'scores':scores,'baselines':base_scores})
    freeze={'selected':chosen['config'],'external_development':chosen,'models_sha256':sha256(OUT/'models.pkl'),
            'protocol_sha256':sha256(OUT/'protocol.json'),'contextual_source_sha256':sha256(ROOT/'second_zero/contextual.py'),
            'challenge_public_labels_used_for_selection':False,'final_evaluation_opened':False}
    json_write(OUT/'freeze.json',freeze)
    print(json.dumps(freeze,indent=2),flush=True)

if __name__=='__main__':main()
