"""Corrupted-only exact metric edit learning; clean is an incumbent bypass."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from second_zero import corrupted_spell_edits as edits
from second_zero import gap_spell as gs
from second_zero.full_baseline_sources import scientific_sources
from second_zero.contextual import binary_decode
from scripts import gap_spell_evidence as evidence
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts.aggressive_unseen_pipeline import legacy_features
from scripts.full_baseline_evaluate import join_halves,score_full

b=evidence.base;TRAIN=b.TRAIN;FOLDS=b.FOLDS
RUN=ROOT/'runs/corrupted-recovery-v5';REPORT=ROOT/'reports/corrupted-recovery-v5'
INC=ROOT/'runs/gap-spell-v4-f1';OLD=b.OLD
ARMS=['baseline','proxy','ranker','calibrated']
MODEL=dict(max_iter=220,max_leaf_nodes=15,max_depth=5,min_samples_leaf=60,learning_rate=.05,l2_regularization=10.,random_state=20261030,early_stopping=False)
ALLOWED=set();READS=[];FULL=False;HASH_ONLY=False;STAGE='prepare'


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def pin(path):return {'path':rel(path),'sha256':sha256(path)}
def check(value):
    path=ROOT/value['path'];assert sha256(path)==value['sha256'],str(path);return path
def truth(match,half):return np.load(ROOT/f'private/improvement/external/prepared/{match}-{half}.truth.npy')
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(s in p for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved asset closed')
    if 'J03WQQ' in p and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed before frozen OOF selection')
    if p.endswith('.truth.npy'):
        if Path(p).name.split('-')[0] not in ALLOWED:raise PermissionError('Truth outside current fitting/audit role')
        READS.append({'stage':STAGE,'path':rel(p),'allowed':sorted(ALLOWED)})


def prepare():
    pointer=read(ROOT/'configs/development_best.json');manifest=read(check(pointer['manifest']))
    assert pointer['score']['overall']==.7708462421456541
    code=[Path(__file__),Path(edits.__file__),Path(edits.prior.__file__),Path(evidence.__file__),Path(b.__file__),*scientific_sources(),ROOT/'scripts/full_baseline_evaluate.py',ROOT/'second_zero/gap_spell_model.py',ROOT/'second_zero/gap_spell.py']
    inputs={**manifest['frozen_files'],rel(ROOT/'configs/development_best.json'):sha256(ROOT/'configs/development_best.json'),rel(ROOT/pointer['manifest']['path']):pointer['manifest']['sha256'],rel(REPORT/'headroom.json'):sha256(REPORT/'headroom.json')}
    write(RUN/'protocol.json',{'kind':'corrupted-complete-edit-exact-F1-v5','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':pointer['score'],'OOF_baseline':pointer['strict_OOF'],
      'folds':{str(f):{'train':t,'held':h} for f,(t,h) in enumerate(FOLDS)},'model':MODEL,'arms':ARMS,
      'scope':'Clean exact incumbent bypass. All corrupted H/A regions editable; every existing D/X locked. No old gap fence on corrupted live play.',
      'candidates':'Observation-only canonical uniform3/short/physical graph proposals, both team alternatives, exact parent cover. Up to24 internal opposite insertions per parent spell in four duration bins, reverse merges, and adjacent H/A boundary shifts every5 frames within125 plus ±1. No GT proposal boundaries.',
      'features':'1638: six ordered old/new original175 canonical interval features;572 raw-only changed-span interval features;16 type/count/relative-duration/quality/unary contrasts. Uniform3 probability mean x0 at training/calibration/inference; no final full logit feature.',
      'target':'1000 times exact official full-match five-IoU SpellF1 delta for the complete alternative, both-half counts, greedy one-to-one affected overlap component, fixed rest of timeline. Unknown targets excluded only during fitting.',
      'training':'TRAIN2 outer weights only; corruption100 and271, both halves, equal view mass. Target-excluded otherTRAIN1 uniform3/short/imputer/legacy for fitting features and proposal parents. Full TRAIN4 fitting features use the oppositeTRAIN2 producers. All raw/learned feature ancestors exclude target and outerheld. Candidate quality-ranking model is original sameouter TRAIN2 quality, so its fitting-side scores are in-sample as disclosed.',
      'training_parent_limit':'Training edit parents are target-excluded canonical owners with fixed binary cost6 and target-excluded legacy cost50 status. They approximate the full parent; held comparison always uses the actual complete .770846 incumbent. This can limit learned transfer.',
      'WPY':'Calibration only. Reconstruct exact .7708-family WPY parent from frozen original full-parent bundles, v3 scorer weights/blend and fixed.4/.01 decoder. Analytic nonnegative affine utility calibration by equal-half squared error, slope clipped[0,5], no SpellF1 grid.',
      'inference':'Maximum predicted-utility nonoverlapping complete windows; threshold0 baseline wins ties. Proxy uses inherited sum-quality difference minus.4*delta-spell-count plus.01*unary delta. Raw ranker and WPY affine ranker use identical edit inventories. At most one selected edit per overlapping parent window.',
      'selection':'Seal all16 views with baseline+3 arms before one OOF audit. Strict overall/corrupted gain and nonnegative bothfold deltas; highest eligible mean. One selected new full WQQ candidate only if gate passes, no fallback. Report matched control even if it wins.',
      'development_limits':'Adaptive reused TRAIN4/WPY/WQQ, no untouched or private competition claim; WQQ labels only previously declared headroom diagnostic and selected final audit. WR9/challenge remain closed.',
      'sources':{rel(p):sha256(p) for p in code},'inputs':inputs})
    print('V5 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def quality(which):return b.original_quality(which)
def pool(signals,parent):
    pieces=[gs.spells(parent)]
    for primary in [0,1]:pieces.append(graph.proposals(signals[:,primary],parent,signals[:,4],signals[:,7:10]))
    a=np.unique(np.concatenate(pieces),axis=0);a=np.unique(np.r_[a,np.c_[a[:,:2],1-a[:,2]]],axis=0)
    return gs.pool(parent,parent<2,a)


def build(raw,signals,parent,qmodel):
    arcs=pool(signals,parent);q=b.old_quality(qmodel,signals,arcs);rows=edits.candidates(parent,arcs,q)
    x,proxy=edits.features(raw,signals,parent,rows,qmodel)
    return rows,x,proxy


def fitting_pair(match,view,mode):
    parents=[];raws=[];xs=[];ancestry=[]
    folder,_,labels,manifest=evidence.producer(match,mode)
    with (folder/'legacy/models.pkl').open('rb') as f:legacy=pickle.load(f)
    for half in ['firstHalf','secondHalf']:
        raw=b.raw(match,half,view);x,a=evidence.evidence(match,half,view,mode,raw)
        ctx=legacy_features(raw['ball'],raw['players'],raw['team'],raw['pitch']);dead=np.where(ctx['ball_seen'],legacy['dead'].predict_proba(ctx['dead'])[:,1],legacy['player'].predict_proba(ctx['player'])[:,1])
        parent=binary_decode(x[:,0],6).astype(np.int8);parent[binary_decode(logit(np.clip(dead,.01,.99)),50).astype(bool)]=2
        raws.append(raw);xs.append(x);parents.append(parent);ancestry.append({**a,'legacy':pin(folder/'legacy/models.pkl')})
    return raws,xs,parents,ancestry


def fitted(which):
    freeze=read(RUN/f'models/{which}/freeze.json')
    with check(freeze['model']).open('rb') as f:return pickle.load(f),freeze


def fit(which):
    verify();matches=TRAIN if which=='full' else FOLDS[int(which)][0];mode='full' if which=='full' else 'oof';ALLOWED.update(matches);qmodel=quality(which);pm=official_metrics();xx=[];yy=[];ww=[];stats=[];start=time.monotonic()
    for match in matches:
        ys=[truth(match,h) for h in ['firstHalf','secondHalf']]
        for view in ['100','271']:
            raws,xs,parents,ancestry=fitting_pair(match,view,mode)
            assert all(match not in a['label_matches'] and set(a['label_matches'])<=set(matches) for a in ancestry)
            built=[build(raw,x,p,qmodel) for raw,x,p in zip(raws,xs,parents)];rows=[v[0] for v in built];targets=edits.paired_targets(pm,ys,parents,rows)
            for half,(rows,x,proxy),target,a in zip(['firstHalf','secondHalf'],built,targets,ancestry):
                valid=np.isfinite(target);xx.append(x[valid]);yy.append(target[valid]);ww.append(np.full(int(valid.sum()),1/max(1,int(valid.sum()))))
                stats.append({'match':match,'half':half,'view':view,'rows':int(valid.sum()),'positive':int((target[valid]>0).sum()),'kinds':{str(k):int((rows[:,5]==k).sum()) for k in [0,1,2]},'ancestry':a})
                print('V5 FIT ROWS',which,match,half,view,x.shape,'positive',stats[-1]['positive'],'seconds',time.monotonic()-start,flush=True)
            del raws,xs,parents,built,rows,targets
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean();del xx,yy,ww
    model=HistGradientBoostingRegressor(**MODEL).fit(x,y,sample_weight=w);folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False)
    with (folder/'model.pkl').open('xb') as f:pickle.dump(model,f)
    write(folder/'freeze.json',{'train_matches':matches,'calibration_matches':[],'model':pin(folder/'model.pkl'),'protocol_sha256':sha256(RUN/'protocol.json'),'shape':list(x.shape),'rows':stats,'reference_reads':READS.copy(),'seconds':time.monotonic()-start});ALLOWED.clear();print('V5 FIT COMPLETE',which,x.shape,time.monotonic()-start,flush=True)


def wpy(which):
    folder=evidence.parent_wpy(which);model,freeze=b.fitted(which);cal=read(evidence.RUN/f'models/{which}/calibration.json');qmodel=quality(which);result=[]
    for half in ['firstHalf','secondHalf']:
        path=folder/f'J03WPY-{half}-corrupted.npz'
        with np.load(path,allow_pickle=False) as z:p={k:z[k] for k in z.files}
        raw=b.raw_half(f'J03WPY-{half}-corrupted');x=evidence.component(p['signals'],p);arcs=b.dense.proposals(p['predicted'],p['eligible_mask'],p['segments']);qo=b.old_quality(qmodel,p['signals'],arcs);qr=evidence.EvidenceIntervals(raw,x).predict(model,arcs);q=(1-cal['alpha'])*qo+cal['alpha']*qr
        parent=gs.decode(p['predicted'],p['eligible_mask'],arcs,q,.4,p['signals'][:,0],.01)
        result.append((raw,x,parent,pin(path)))
    return result


def calibrate(which):
    verify();ALLOWED.clear();ALLOWED.add('J03WPY');model,freeze=fitted(which);qmodel=quality(which);views=wpy(which);built=[build(r,x,p,qmodel) for r,x,p,_ in views];ys=[truth('J03WPY',h) for h in ['firstHalf','secondHalf']];targets=edits.paired_targets(official_metrics(),ys,[v[2] for v in views],[v[0] for v in built]);u=[];y=[];w=[]
    for (_,x,_),target in zip(built,targets):
        valid=np.isfinite(target);u.append(model.predict(x[valid]));y.append(target[valid]);w.append(np.full(int(valid.sum()),1/max(1,int(valid.sum()))))
    u=np.concatenate(u);y=np.concatenate(y);w=np.concatenate(w);w/=w.sum();mu=float(w@u);my=float(w@y);slope=float(np.clip((w@((u-mu)*(y-my)))/max(float(w@((u-mu)**2)),1e-12),0,5));intercept=my-slope*mu
    write(RUN/f'models/{which}/calibration.json',{'calibration_matches':['J03WPY'],'slope':slope,'intercept':intercept,'objective':'Equal-half squared exact-F1 utility error, analytic nonnegative affine fit; no F1 search','model':freeze['model'],'parent_inputs':[v[3] for v in views],'reference_reads':READS.copy()});ALLOWED.clear();print('V5 CALIBRATION',which,slope,intercept,flush=True)


def incumbent(match,half,condition):
    folder=INC/'full-predictions' if match=='J03WQQ' else INC/f'oof/fold{0 if match in TRAIN[:2] else 1}'
    path=folder/f'{match}-{half}-{condition}.npz'
    with np.load(path,allow_pickle=False) as z:return z['mixed'].copy()


def prediction(which,match,half,condition,arms):
    parent=incumbent(match,half,condition);out={'baseline':parent.copy()}
    if condition=='clean':out.update({a:parent.copy() for a in arms});return out,{'changed':{a:0 for a in arms},'clean_bypass':True}
    raw=b.raw_half(f'{match}-{half}-{condition}');p=b.ceiling.current(match,half,condition);folder=OLD/'full-transfer/full_baseline' if match=='J03WQQ' else OLD/f'baseline/fold{which}'
    with np.load(folder/f'{match}-{half}-{condition}.npz',allow_pickle=False) as z:owners={f'owner_{i}':z[f'owner_{i}'] for i in range(3)}
    x=evidence.component(p['signals'],owners);rows,features,proxy=build(raw,x,parent,quality(which));utilities={'proxy':proxy}
    if any(a!='proxy' for a in arms):
        model,freeze=fitted(which);cal=read(RUN/f'models/{which}/calibration.json');assert cal['model']==freeze['model'];u=model.predict(features);utilities.update(ranker=u,calibrated=cal['slope']*u+cal['intercept'])
    chosen={}
    for arm in arms:
        selected=edits.prior.select_edits(rows,utilities[arm]);out[arm]=edits.apply(parent,rows,selected);chosen[arm]=len(selected)
    return out,{'candidates':len(rows),'selected':chosen,'changed':{a:int((out[a]!=parent).sum()) for a in arms},'kinds':{str(k):int((rows[:,5]==k).sum()) for k in [0,1,2]}}


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={};start=time.monotonic()
    for f,(_,held) in enumerate(FOLDS):
        for key in b.keys(held):
            match,half,c=key.split('-');v,meta=prediction(str(f),match,half,c,ARMS[1:]);path=RUN/f'oof/fold{f}'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);views[key]=meta;print('V5 OOF',key,meta,'seconds',time.monotonic()-start,flush=True)
    write(RUN/'oof-seal.json',{'protocol_sha256':sha256(RUN/'protocol.json'),'outputs':outputs,'views':views,'held_truth_reads':[],'models':{str(f):{'freeze':pin(RUN/f'models/{f}/freeze.json'),'calibration':pin(RUN/f'models/{f}/calibration.json')} for f in [0,1]}})


def validated_oof():
    verify();seal=read(RUN/'oof-seal.json');assert len(seal['outputs'])==16 and not seal['held_truth_reads'];loaded={}
    for f,(train,held) in enumerate(FOLDS):
        meta=seal['models'][str(f)];freeze=read(check(meta['freeze']));cal=read(check(meta['calibration']));check(freeze['model']);assert freeze['train_matches']==train and cal['calibration_matches']==['J03WPY']
        for row in freeze['rows']:
            a=row['ancestry'];assert row['match'] in train and row['match'] not in a['label_matches'] and set(a['label_matches'])<=set(train);check(a['manifest']);check(a['legacy'])
        for key in b.keys(held):
            path=RUN/f'oof/fold{f}'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            m,h,c=key.split('-');np.testing.assert_array_equal(v['baseline'],incumbent(m,h,c))
            for arm in ARMS:
                assert np.isin(v[arm],[0,1,2,3]).all();fixed=v['baseline']>=2;np.testing.assert_array_equal(v[arm][fixed],v['baseline'][fixed])
                if c=='clean':np.testing.assert_array_equal(v[arm],v['baseline'])
            loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One OOF audit only')
    seal,loaded=validated_oof();ALLOWED.update(TRAIN);pm=official_metrics();details={}
    for m in TRAIN:
        y=join_halves([truth(m,h) for h in ['firstHalf','secondHalf']]);details[m]={c:{a:score_full(pm,y,join_halves([loaded[f'{m}-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ARMS} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in ARMS}
    overall=avg(TRAIN,['clean','corrupted']);conditions={c:avg(TRAIN,[c]) for c in ['clean','corrupted']};folds={str(f):avg(h,['clean','corrupted']) for f,(_,h) in enumerate(FOLDS)};assert abs(overall['baseline']-.7099817195618008)<1e-12
    valid=[a for a in ARMS[1:] if overall[a]>overall['baseline'] and conditions['corrupted'][a]>conditions['corrupted']['baseline'] and all(v[a]>=v['baseline'] for v in folds.values())];selected=max(['baseline']+valid,key=lambda a:overall[a]);result={'overall':overall,'conditions':conditions,'per_fold':folds,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':details,'selected':selected,'passed':selected!='baseline','clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'oof-seal.json'),'protocol_sha256':sha256(RUN/'protocol.json'),'reference_reads':READS,'limitation':'Adaptive development; simplified target-excluded training edit parents, actual full incumbent validation; no new WQQ candidate scored.'};write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_fold','selected','passed']}),flush=True)


def authorize():
    global FULL
    verify();a=read(RUN/'oof-audit.json');check(a['prediction_seal']);validated_oof();assert a['passed'] and a['selected']!='baseline';FULL=True;return a['selected']


def full_run():
    selected=authorize()
    if selected!='proxy':fit('full');calibrate('full')
    write(RUN/'full-selection.json',{'selected':selected,'OOF_audit':pin(RUN/'oof-audit.json'),'no_fallback':True});outputs={};views={};ALLOWED.clear()
    for half in ['firstHalf','secondHalf']:
        for c in ['clean','corrupted']:
            key=f'J03WQQ-{half}-{c}';v,meta=prediction('full','J03WQQ',half,c,[selected]);path=RUN/'full-predictions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);views[key]=meta;print('V5 FULL',key,meta,flush=True)
    write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'selection':pin(RUN/'full-selection.json'),'truth_reads':[]})


def audit_full():
    if (RUN/'full-audit.json').exists():raise FileExistsError('One full audit only')
    selected=authorize();seal=read(RUN/'full-seal.json');check(seal['selection']);assert len(seal['outputs'])==4 and not seal['truth_reads'];loaded={}
    for key in b.keys(['J03WQQ']):
        path=RUN/'full-predictions'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
        with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        m,h,c=key.split('-');np.testing.assert_array_equal(v['baseline'],incumbent(m,h,c));fixed=v['baseline']>=2;np.testing.assert_array_equal(v[selected][fixed],v['baseline'][fixed])
        if c=='clean':np.testing.assert_array_equal(v[selected],v['baseline'])
        loaded[key]=v
    ALLOWED.clear();ALLOWED.add('J03WQQ');pm=official_metrics();y=join_halves([truth('J03WQQ',h) for h in ['firstHalf','secondHalf']]);conditions={c:{a:score_full(pm,y,join_halves([loaded[f'J03WQQ-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ['baseline',selected]} for c in ['clean','corrupted']};overall={a:float(np.mean([conditions[c][a]['spell_f1_mean'] for c in ['clean','corrupted']])) for a in ['baseline',selected]};assert abs(overall['baseline']-.7708462421456541)<1e-12
    result={'overall':overall,'conditions':conditions,'selected':selected,'improved':overall[selected]>overall['baseline'],'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'full-seal.json'),'reference_reads':READS,'no_fallback':True,'interpretation':'One reused development match; no private challenge claim'};write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'improved':result['improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--fold',choices=['0','1']);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(guard)
    if args.stage in ['fit','calibrate']:
        assert args.fold is not None;globals()[args.stage](args.fold)
    else:globals()[args.stage]()
