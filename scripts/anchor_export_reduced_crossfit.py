"""Split-safe reduced complete-pipeline comparison on four external matches.

All learned modules, including those producing quality-training features, use
only the current fold's training side. Quality inputs on that side are in-sample;
this is disclosed rather than quietly using an opposite-fold upstream model
trained on the evaluation side. No full-trained GRU/carrier/quality model is used.
"""
from __future__ import annotations
import argparse
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
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from second_zero.common import sha256
from second_zero.contextual import binary_decode
from second_zero.sprint_model_owner import SprintOwner, infer_logits
from second_zero.sprint_model_interpolation import interpolated_features
from second_zero.evaluate import official_metrics
from scripts.sprint_status_context import infer as status_infer
from scripts.sprint_ball_imputer import predict as impute_ball
from scripts.train_contextual import corruption
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts import sprint_decoder_lattice as lattice
from scripts.anchor_refine_experiment import refine_evidence

SPRINT = ROOT/'runs/two-hour-sprint'
DATA = ROOT/'private/improvement/external/prepared'
FOLDS = [(['J03WOH','J03WOY'], ['J03WMX','J03WN1']),
         (['J03WMX','J03WN1'], ['J03WOH','J03WOY'])]
STATE = {'fold':None, 'phase':'prepare', 'allowed':[], 'reads':[]}
HASHES = {}


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = os.fsdecode(args[0])
    if 'J03WR9' in p or 'J03WQQ' in p or '/evaluation/gt/' in p:
        raise PermissionError('Reduced crossfit excludes WQQ/public/reserved assets')
    if p.endswith('.truth.npy'):
        if not any(Path(p).name.startswith(m+'-') for m in STATE['allowed']):
            raise PermissionError('Heldout truth forbidden during fit/inference: '+p)
        STATE['reads'].append({'fold':STATE['fold'], 'phase':STATE['phase'], 'path':p})


def pin(p):
    p = Path(p)
    if str(p) not in HASHES:
        HASHES[str(p)] = sha256(p)
    return p


def meta(p):
    return json.loads(pin(p).read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as handle:
        json.dump(value, handle, indent=2)


def model_load(folder, train):
    protocol = meta(folder/'protocol.json')
    assert protocol['train_matches'] == train
    ck = torch.load(pin(folder/'model.pt'), weights_only=True, map_location='cpu')
    model = SprintOwner(**ck['architecture'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, meta(folder/'calibration.json')['temperature']


def raw_data(key):
    name, condition = key.rsplit('-', 1)
    with np.load(pin(DATA/(name+'.npz'))) as z:
        a = {k:z[k] for k in ['ball','players','team','pitch']}
    return corruption(a,100) if condition == 'corrupted' else a


def truth(key):
    return np.load(pin(DATA/('-'.join(key.split('-')[:2])+'.truth.npy')))


def raw_geometry(a, raw, imputers):
    filled, _ = impute_ball(a, imputers)
    linear = interpolated_features(a['ball'],a['players'],a['team'],a['pitch'])
    learned = interpolated_features(filled,a['players'],a['team'],a['pitch'])
    unaries, confidence = [], []
    for f, observed in [(raw,True),(linear,False),(learned,False)]:
        d, rv = f[:,:,0]*20, f[:,:,3]*30
        both = (f[:,:,49 if observed else 18]<.5).all(1)
        c = np.where(both,np.exp(-np.min(d,axis=1)/10),0)
        z = np.clip((d[:,0]-d[:,1])/3+.25*(rv[:,0]-rv[:,1])/10,-4,4)*c
        z[~both]=0
        unaries.append(z);confidence.append(c)
    return np.c_[*unaries,*confidence].astype(np.float32)


class FoldEvidence:
    def __init__(self, fold, train, heldout):
        self.fold, self.train, self.heldout = fold, train, heldout
        self.learned_folder = SPRINT/f'model/interp_learned_swap-{fold}'
        self.long_folder = SPRINT/f'model/long-{fold}'
        self.learned, self.learned_t = model_load(self.learned_folder, train)
        self.long, self.long_t = model_load(self.long_folder, train)
        folder = SPRINT/f'ball-imputer-fold{fold}'
        assert meta(folder/'protocol.json')['train_matches'] == train
        with pin(folder/'models.pkl').open('rb') as handle:
            self.imputers = pickle.load(handle)
        folder = SPRINT/f'status-crossfit-v2/fold{fold}'
        frozen = meta(folder/'freeze.json')
        assert frozen['training_matches']==train and frozen['heldout_matches']==heldout
        assert sha256(pin(folder/'model.pkl')) == frozen['model_sha256']
        with (folder/'model.pkl').open('rb') as handle:
            self.status = pickle.load(handle)
        self.cal = meta(folder/'calibration.json')
        with pin(ROOT/f'runs/overnight-boundary-v1/crossfit/fold{fold}-models.pkl').open('rb') as handle:
            self.legacy = pickle.load(handle)
        self.replay = {}

    def owner(self, key, raw, a, folder, model, temperature, learned=False):
        # Every existing owner cache in these folders is WPY or this fold's
        # held-out side; training-side inference is deliberately recomputed.
        p = folder/(key+'.npz')
        if p.exists():
            with np.load(pin(p)) as z:
                return z['logit'].copy()
        x = raw
        if learned:
            linear = interpolated_features(a['ball'],a['players'],a['team'],a['pitch'])
            filled,_ = impute_ball(a,self.imputers,blend=.75)
            extra = interpolated_features(filled,a['players'],a['team'],a['pitch'])
            extra[:,:,12:17]=linear[:,:,12:17]
            x = np.concatenate([raw,extra],axis=-1)
        return infer_logits(model,x)/temperature

    def signals(self, key):
        a = raw_data(key)
        raw = np.load(pin(ROOT/'runs/ownership-temporal-v1/features'/(key+'.npy')),mmap_mode='r')
        status_x = np.load(pin(ROOT/'runs/live-dead-temporal-v1/features'/(key+'.npy')),mmap_mode='r')
        learned = self.owner(key,raw,a,self.learned_folder,self.learned,self.learned_t,True)
        longer = self.owner(key,raw,a,self.long_folder,self.long,self.long_t)
        blend = logit(np.clip(.5*expit(learned)+.5*expit(longer),1e-8,1-1e-8))
        values = np.stack([learned,longer],axis=1)
        physical = raw_geometry(a,raw,self.imputers)
        x = np.c_[blend, learned, values.std(1), np.mean(np.sign(values),axis=1),
                  status_x[:,[36,38,39]], physical].astype(np.float32)
        cache = ROOT/'runs/overnight-boundary-v1/crossfit'
        dead_path = cache/(key+'-dead.npy')
        if dead_path.exists():
            dead_features = np.load(pin(dead_path),mmap_mode='r')
            seen = np.load(pin(cache/(key+'-seen.npy')))
        else:
            from second_zero.contextual import features
            features = features(a['ball'],a['players'],a['team'],a['pitch'])
            dead_features, seen = features['dead'], features['ball_seen']
        legacy_p = np.where(seen,self.legacy['dead'].predict_proba(dead_features)[:,1],
                            self.legacy['player'].predict_proba(dead_features[:,:52])[:,1])
        context = status_infer(self.status,status_x)
        mixed = .75*(context*self.cal['slope']+self.cal['intercept'])+.25*logit(np.clip(legacy_p,.01,.99))
        base = np.where(binary_decode(np.clip(mixed,-np.log(99),np.log(99)),50),2,0).astype(np.int8)
        if key.split('-')[0] in self.heldout:
            p = SPRINT/'status-crossfit-v2/cost50'/(key+'.npz')
            with np.load(pin(p)) as cached:
                assert np.array_equal(base>=2,cached['predicted']>=2),key
                assert np.array_equal(mixed,cached['dead_logit']),key
            self.replay[key]={'exact_existing_crossfit_status_logits_and_mask':True}
        assert x.shape[1] == 13 and np.isfinite(x).all()
        return x, base, expit(mixed)


def pool(x, base):
    return lattice.proposals(x[:,0],base,graph.proposals(x[:,0],base,x[:,4],x[:,7:10]),16)


def graph_prediction(x,base,segments,quality,penalty):
    return graph.graph_decode(x[:,0],base,segments,quality,penalty,.01)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fold',type=int,choices=[0,1],required=True)
    parser.add_argument('--anchors',action='store_true')
    args=parser.parse_args()
    sys.addaudithook(guard)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);threadpool_limits(2)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    train,heldout=FOLDS[args.fold]
    STATE.update(fold=args.fold,phase='train',allowed=train)
    started=time.monotonic()
    protocol={'fold':args.fold,'train_matches':train,'heldout_matches':heldout,'calibration_match':'J03WPY',
        'scope':'Reduced complete pipeline; every fitted component excludes heldout matches. Does not validate full sprint winner architecture.',
        'owners':'Samefold learned-geometry TCN and long TCN; equal probability mean.',
        'ancillary_channels':'Learned owner, two-expert logit std and mean signs replace prior uniform/short ancillary channels; remaining graph feature math unchanged.',
        'status':'Samefold context HGB calibrated on WPY, .75 context+.25 samefold legacy, cost50, no global endpoints/temporal status.',
        'quality_training':'Samefold upstream models applied to their own training side; quality training inputs are in-sample, intentionally avoiding indirect heldout-label leakage through oppositefold feature generators.',
        'quality_model':'Fresh180iteration15leaf depth5 HGB, identical target/objective; no blend with full-trained model.',
        'calibration':'decode cost6/12 baseline and quality penalties .45/.55/.65 on WPY; fixed frame .01 neighbors16. Anchor arm uses graph penalty chosen on WPY, no extra tuning.',
        'no_hard_minimum_duration':True,'source_sha256':sha256(__file__)}
    write(out/'protocol.json',protocol)
    engine=FoldEvidence(args.fold,train,heldout)
    xx,yy=[],[]
    counts={}
    for key in graph.keys(train):
        x,base,_=engine.signals(key)
        seg=pool(x,base)
        target,_=graph.quality_targets(truth(key),seg)
        f=graph.segment_features(x,seg)
        xx.append(f);yy.append(target);counts[key]=len(seg)
        print('Reduced samefold training features',args.fold,key,len(seg),round(time.monotonic()-started,1),flush=True)
    features=np.concatenate(xx);targets=np.concatenate(yy);del xx,yy
    model=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,max_depth=5,min_samples_leaf=40,
        learning_rate=.05,l2_regularization=5.,random_state=20261024,early_stopping=False).fit(features,targets)
    shape=list(features.shape);del features,targets
    with (out/'model.pkl').open('xb') as handle:pickle.dump(model,handle)
    write(out/'training.json',{'shape':shape,'counts':counts,'seconds':time.monotonic()-started})
    STATE.update(phase='calibrate',allowed=['J03WPY'])
    pm=official_metrics();scores={f'decode{cost}':{} for cost in [6,12]}
    scores.update({f'graph{p}':{} for p in [.45,.55,.65]})
    for key in graph.keys(['J03WPY']):
        x,base,pdead=engine.signals(key);y=truth(key);seg=pool(x,base)
        quality=np.clip(model.predict(graph.segment_features(x,seg)),0,1)
        for cost in [6,12]:scores[f'decode{cost}'][key]=graph.metric(pm,y,graph.decode(x[:,0],base,cost=cost))
        for penalty in [.45,.55,.65]:scores[f'graph{penalty}'][key]=graph.metric(pm,y,graph_prediction(x,base,seg,quality,penalty))
        print('Reduced WPY',args.fold,key,flush=True)
    means={arm:{m:float(np.mean([r[m] for r in views.values()])) for m in ['spell','macro']} for arm,views in scores.items()}
    cost=max([6,12],key=lambda c:(means[f'decode{c}']['spell'],means[f'decode{c}']['macro']))
    penalty=max([.45,.55,.65],key=lambda p:(means[f'graph{p}']['spell'],means[f'graph{p}']['macro']))
    write(out/'freeze.json',{'baseline_cost':cost,'graph_penalty':penalty,'frame_weight':.01,'neighbors':16,
        'WPY_means':means,'WPY_views':scores,'model_sha256':sha256(out/'model.pkl'),'heldout_truth_accessed':False})
    print('Reduced frozen',args.fold,cost,penalty,means,flush=True)
    STATE.update(phase='inference',allowed=[])
    arms=['baseline','quality']+(['quality_anchor'] if args.anchors else [])
    for arm in arms:(out/arm).mkdir()
    anchor_meta={}
    for key in graph.keys(heldout):
        x,base,pdead=engine.signals(key);seg=pool(x,base)
        quality=np.clip(model.predict(graph.segment_features(x,seg)),0,1)
        predictions={'baseline':graph.decode(x[:,0],base,cost=cost),
                     'quality':graph_prediction(x,base,seg,quality,penalty)}
        if args.anchors:
            refined,metadata=refine_evidence(x,base,pdead,seg,quality,model)
            predictions['quality_anchor']=graph_prediction(x,base,refined['segments'],refined['quality'],penalty)
            anchor_meta[key]=metadata
        for arm,pred in predictions.items():
            assert np.array_equal(pred[base>=2],base[base>=2])
            np.savez_compressed(out/arm/(key+'.npz'),predicted=pred,logit=x[:,0],status_mask=base)
        print('Reduced heldout predictions',args.fold,key,flush=True)
    write(out/'completion.json',{'seconds':time.monotonic()-started,'input_sha256':HASHES,
        'reference_reads':STATE['reads'],'heldout_references_accessed':False,'status_cache_replay':engine.replay,
        'anchors':anchor_meta,'prediction_sha256':{str(p.relative_to(out)):sha256(p) for arm in arms for p in (out/arm).glob('*.npz')}})


if __name__=='__main__':main()
