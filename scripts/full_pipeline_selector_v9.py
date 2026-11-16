"""Crossed full-producer/selector validation with target-excluded actual parents."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from second_zero import full_pipeline_selector as selector
from second_zero import selector_full_parents as parents
from second_zero.full_baseline_sources import scientific_sources
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from scripts.full_baseline_evaluate import join_halves,score_full

old=parents.region;b=parents.b;e=parents.e;s=parents.s
TRAIN=list(s.TRAIN);RUN=ROOT/'runs/full-pipeline-selector-v9';REPORT=ROOT/'reports/full-pipeline-selector-v9'
ARMS=['baseline','selector','conservative'];ALLOWED=set();FULL=False;HASH_ONLY=False;STAGE='prepare';READS=[]


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def pin(path):return {'path':rel(path),'sha256':sha256(path)}
def checked(value):
    p=ROOT/value['path'];assert sha256(p)==value['sha256'],value['path'];return p
def truth(match,half):return np.load(ROOT/f'private/improvement/external/prepared/{match}-{half}.truth.npy')
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.fsdecode(args[0])
    if any(v in path for v in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved assets closed')
    if 'J03WQQ' in path and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed before frozen current OOF advancement')
    if path.endswith('.truth.npy'):
        if Path(path).name.split('-')[0] not in ALLOWED:raise PermissionError('Primary reference outside current fitting/calibration/audit role')
        READS.append({'stage':STAGE,'path':rel(path),'allowed':sorted(ALLOWED)})


def producer(match):
    if match=='J03WQQ':return 'full'
    if match not in TRAIN:raise ValueError('Ambiguous producer for calibration match')
    return '0' if match in TRAIN[:2] else '1'


def split(held):
    if held not in TRAIN:raise ValueError('Invalid selector held match')
    p=producer(held);group=TRAIN[:2] if p=='0' else TRAIN[2:];fit=next(m for m in group if m!=held);primary=list(s.FOLDS[int(p)][0])
    assert held not in primary and fit not in primary and set(primary+[fit])==set(TRAIN)-{held}
    return {'held':held,'selector_fit_matches':[fit],'producer':p,'producer_primary_matches':primary,'all_primary_label_ancestors':sorted(primary+[fit]),'same_actual_producer_for_fit_and_held':True}


def producer_proof(which):
    expected=TRAIN if which=='full' else list(s.FOLDS[int(which)][0]);spec=read(parents.OLD/'full-reference-spec.json' if which=='full' else parents.OLD/f'specs/fold{which}.json')
    if which!='full':assert spec['train_matches']==expected
    credit=read(e.RUN/f'models/{which}/freeze.json');rich=read(s.RUN/f'models/{which}/freeze.json');assert credit['train_matches']==rich['train_matches']==expected
    for row in rich['rows']:
        a=row['ancestor'];assert row['match'] not in a['label_matches'] and set(a['label_matches'])<=set(expected)
    for target,a in credit['strict_crossfit_feature_ancestors'].items():assert target not in a['label_matches'] and set(a['label_matches'])<=set(expected)
    return {'primary_label_matches':expected,'calibration_label_matches':['J03WPY'],'full_baseline_spec':pin(parents.OLD/'full-reference-spec.json' if which=='full' else parents.OLD/f'specs/fold{which}.json'),'credit_freeze':pin(e.RUN/f'models/{which}/freeze.json'),'rich_freeze':pin(s.RUN/f'models/{which}/freeze.json'),'old_complete_ancestry':pin(parents.OLD/'baseline-ancestry.json'),'old_complete_producer':pin(parents.OLD/'producer-protocol.json')}


def prepare():
    current=read(ROOT/'configs/current_development_best.json');best=read(checked(current['versioned_best']));manifest=read(checked(best['manifest']));assert best['score']['overall']==.7708462421456541
    sources=[Path(__file__),Path(selector.__file__),Path(parents.__file__),Path(selector.local.__file__),Path(selector.local.prior.__file__),Path(old.__file__),Path(s.__file__),Path(e.__file__),Path(b.__file__),ROOT/'scripts/full_baseline_evaluate.py',ROOT/'second_zero/gap_spell.py',ROOT/'second_zero/gap_spell_dense.py',ROOT/'second_zero/gap_spell_model.py',ROOT/'second_zero/stacked_gap_owner.py',*scientific_sources()]
    inputs={**manifest['frozen_files'],**read(ROOT/'reports/research-push-20261030/rejected-full-artifact-freeze.json')['hashes'],'configs/development_best.json':sha256(ROOT/'configs/development_best.json'),'reports/development-best-20261030/manifest.json':best['manifest']['sha256']}
    for folder in [old.RUN,s.RUN]:
        for name in ['protocol.json','oof-seal.json']:
            path=folder/name;inputs[rel(path)]=sha256(path)
        seal=read(folder/'oof-seal.json');inputs.update(seal['outputs'])
    for which in ['0','1','full']:
        for folder in [s.RUN,e.RUN]:
            for name in ['freeze.json','calibration.json']:
                path=folder/f'models/{which}/{name}';inputs[rel(path)]=sha256(path);v=read(path)
                if 'model' in v:inputs[v['model']['path']]=v['model']['sha256']
        path=old.RUN/f'calibration/{which}.json';inputs[rel(path)]=sha256(path)
    p={'kind':'actual-full-pipeline-crossed-selector-v9','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':best['score'],'baseline_OOF':best['strict_OOF'],
       'validation':[split(m) for m in TRAIN],'producers':{p:producer_proof(p) for p in ['0','1']},'model':selector.MODEL,'arms':ARMS,'penalty_per_added_spell':selector.PENALTY,
       'parent_repair':'Every fitting and held example uses the complete actual .770846 baseline, not uniform-only owner/legacy-only status. Within each selector validation run, fit and held inputs use the SAME frozen TRAIN2 complete producer; neither fitting nor held match appears in any producer weight ancestry. Selector fits one unseen match and predicts the other, then swap. Union of primary/selector labels is exactly TRAIN3 excluding held. No fresh truncated/simplified parent.',
       'parent_architecture':'Uniform3+short, linear/learned interpolation, imputer,103-channel GRU, carrier, full legacy/context/temporal status plus endpoint models,175-quality graph/lattice, joint.25, old raw290 gap.5,747 credit blend and fixed.4/.01 decoder. Three richer complete challenger paths are the frozen v7 quarter/gap-calibrated and v8 region paths with the SAME producer. All existing saved label arrays must replay bitwise before features seal.',
       'features':'1298: sums of175 complete arc features for old/new intervals under actual baseline-final and richer-final13-channel evidence (700),26 complete-window confidence/count/source/quality contrasts,572 raw changed-span summaries. Actual complete baseline labels generate every edit. Every feature producer excludes the feature target and selector held match.',
       'proposals':'Existing complete graph/lattice/dense-gap arcs from actual models yield bounded inserts, reverse merges and native5-frame boundary shifts±125. Add minimal aligned-boundary disagreement windows from all3 complete challengers. Exact full incumbent alternative retained; baseline wins utility ties. Every edit replaces whole parent spells, all D/X fixed.',
       'target':'Exact change in TP averaged across5 official IoUs, with whole affected one-to-one overlap component. Also compute exact full-match F1 delta with both-half counts for checks, not fitting. Single HGB predicts TP gain. Score gain minus fixed.4*known change in predicted-spell count (targetF1.8/2); no learned estimate of held GT counts.',
       'calibration':'WPY only, actual SAME producer complete parents. Equal-half analytic affine TP-gain calibration, slope clipped[0,5]. Conservative arm subtracts nonnegative90th percentile of WPY overprediction residual; no coverage guarantee or F1 coefficient/threshold search.',
       'data':'Original corrupted100 only; fitting examples and held examples generated with identical full pipeline. Clean bypass returns incumbent exactly. No row-level split, extra independent-match claim or invented label. Coordinate-only or weaker status parents are not used.',
       'selection':'All16 half/condition views and all3 arms sealed before one current OOF audit. Strict overall/corrupted gain and nonnegative producer-pair fold deltas; report all4 physical matches. Highest eligible mean, baseline wins ties. If eligible fit selector on all4 cross-fitted full-parent matches, calibrate using actual full TRAIN4 WPY parents, test exactly one selected current full WQQ composition. No fallback.',
       'limits':'Adaptive reused development and previously inspected WQQ results inform design. No current WQQ fitting/calibration/tuning. Final primary producer has4 training matches versus2 in selector fitting features; this data-size difference remains and will be disclosed. Complete architecture, feature contract and within-validation producer size ARE matched. Selector fitting uses1 independent match per validation run; full selector uses4.',
       'clean_and_status':'Clean predictions bit-identical, D/X exact. No original CSV, checkpoint, .769357 or .770846 output overwritten; new model directory only. WR9/challenge references stay closed.',
       'sources':{rel(p):sha256(p) for p in sources},'inputs':inputs}
    write(RUN/'protocol.json',p);print('SELECTOR V9 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def pack_edits(edits):
    oldoff=np.r_[0,np.cumsum([len(v.old) for v in edits])];newoff=np.r_[0,np.cumsum([len(v.new) for v in edits])]
    return {'edit_meta':np.array([[v.start,v.stop,v.kind,v.source] for v in edits],np.int32).reshape(-1,4),'old_offsets':oldoff,'new_offsets':newoff,'old_arcs':np.concatenate([v.old for v in edits]).astype(np.int32) if edits else np.empty((0,3),np.int32),'new_arcs':np.concatenate([v.new for v in edits]).astype(np.int32) if edits else np.empty((0,3),np.int32)}
def unpack_edits(v):
    return [selector.Edit(*map(int,row),v['old_arcs'][a:b].astype(np.int64),v['new_arcs'][c:d].astype(np.int64)) for row,a,b,c,d in zip(v['edit_meta'],v['old_offsets'][:-1],v['old_offsets'][1:],v['new_offsets'][:-1],v['new_offsets'][1:])]


def case_path(which,match,half):return RUN/f'cases/producer{which}'/f'{match}-{half}-corrupted.npz'
def make_case(which,match,half):
    start=time.monotonic();p=parents.context(which,match,half);ee=selector.candidates(p['baseline'],p['alternatives'],p['pool'],p['pool_quality']);x,proxy,dn=selector.feature_matrix(p['raw'],p['baseline_signals'],p['rich_signals'],p['baseline'],ee,p['quality_model']);values={'features':x,'proxy':proxy,'delta_n':dn,'baseline':p['baseline'],**pack_edits(ee)};path=case_path(which,match,half);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**values)
    meta={'producer':str(which),'match':match,'half':half,'rows':len(ee),'features':1298,'cache':pin(path),'complete_baseline_replay_exact':p['replayed_baseline'],'complete_challenger_replays_exact':p['replayed_complete_challengers'],'parent_proof':producer_proof(str(which)),'kinds':{str(k):sum(v.kind==k for v in ee) for k in range(4)},'seconds':time.monotonic()-start};print('V9 FULL CASE',which,match,half,len(ee),'seconds',meta['seconds'],flush=True);return meta


def cache():
    verify();ALLOWED.clear();rows={}
    for m in TRAIN:
        for h in ['firstHalf','secondHalf']:rows[f'{m}-{h}']=make_case(producer(m),m,h)
    for which in ['0','1']:
        for h in ['firstHalf','secondHalf']:rows[f'WPY-producer{which}-{h}']=make_case(which,'J03WPY',h)
    assert not READS;write(RUN/'case-seal.json',{'rows':rows,'protocol_sha256':sha256(RUN/'protocol.json'),'reference_reads':READS.copy(),'all_saved_primary_baselines_and_challengers_replayed':True})


def load_case(which,match,half,seal=None):
    seal=read(RUN/'case-seal.json') if seal is None else seal;key=f'WPY-producer{which}-{half}' if match=='J03WPY' else f'{match}-{half}';meta=seal['rows'][key];assert meta['producer']==str(which) and meta['match']==match
    path=checked(meta['cache'])
    with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
    return v,unpack_edits(v),meta


def training_arrays(matches):
    pm=official_metrics();xx=[];yy=[];ww=[];rows=[];seal=read(RUN/'case-seal.json')
    for m in matches:
        which=producer(m);cases=[load_case(which,m,h,seal) for h in ['firstHalf','secondHalf']];ys=[truth(m,h) for h in ['firstHalf','secondHalf']];targets=selector.paired_targets(pm,ys,[v[0]['baseline'] for v in cases],[v[1] for v in cases])
        for (v,edits,meta),target in zip(cases,targets):
            valid=np.isfinite(target).all(1);xx.append(v['features'][valid]);yy.append(target[valid,0]);ww.append(np.full(int(valid.sum()),1/max(1,int(valid.sum()))));rows.append({'match':m,'half':meta['half'],'cache':meta['cache'],'producer':which,'producer_primary_matches':meta['parent_proof']['primary_label_matches'],'rows':int(valid.sum()),'positive_count_gain':int((target[valid,0]>0).sum()),'positive_exact_F1_delta':int((target[valid,1]>0).sum())})
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean();return x,y,w,rows


def fit(held):
    verify();which='full' if held=='full' else held;matches=TRAIN if held=='full' else split(held)['selector_fit_matches'];ALLOWED.update(matches);start=time.monotonic();x,y,w,rows=training_arrays(matches)
    for row in rows:
        assert row['match'] not in row['producer_primary_matches']
        if held!='full':assert held not in row['producer_primary_matches'] and row['producer']==producer(held)
    model=HistGradientBoostingRegressor(**selector.MODEL).fit(x,y,sample_weight=w);folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False)
    with (folder/'model.pkl').open('xb') as f:pickle.dump(model,f)
    write(folder/'freeze.json',{'selector_fit_matches':matches,'held_match':None if held=='full' else held,'validation_split':None if held=='full' else split(held),'shape':list(x.shape),'target':'mean five-IoU exact matched-count change','model':pin(folder/'model.pkl'),'rows':rows,'protocol_sha256':sha256(RUN/'protocol.json'),'case_seal':pin(RUN/'case-seal.json'),'reference_reads':READS.copy(),'seconds':time.monotonic()-start});ALLOWED.clear();print('V9 SELECTOR FIT',which,x.shape,'seconds',time.monotonic()-start,flush=True)


def fitted(held):
    f=read(RUN/f'models/{held}/freeze.json')
    with checked(f['model']).open('rb') as h:return pickle.load(h),f


def calibrate(held):
    verify();which='full' if held=='full' else producer(held);model,freeze=fitted(held);ALLOWED.clear();ALLOWED.add('J03WPY')
    if held=='full':
        rows={}
        for h in ['firstHalf','secondHalf']:rows[f'WPY-producerfull-{h}']=make_case('full','J03WPY',h)
        fullseal={'rows':rows};write(RUN/'full-calibration-case-seal.json',fullseal)
    else:fullseal=read(RUN/'case-seal.json')
    cases=[load_case(which,'J03WPY',h,fullseal) for h in ['firstHalf','secondHalf']];ys=[truth('J03WPY',h) for h in ['firstHalf','secondHalf']];targets=selector.paired_targets(official_metrics(),ys,[v[0]['baseline'] for v in cases],[v[1] for v in cases]);xx=[];yy=[];ww=[]
    for (v,_,_),target in zip(cases,targets):
        valid=np.isfinite(target).all(1);xx.append(model.predict(v['features'][valid]));yy.append(target[valid,0]);ww.append(np.full(int(valid.sum()),1/max(1,int(valid.sum()))))
    u=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.sum();mu=float(np.sum(w*u));my=float(np.sum(w*y));slope=float(np.clip(np.sum(w*(u-mu)*(y-my))/max(float(np.sum(w*(u-mu)**2)),1e-12),0,5));intercept=my-slope*mu;residual=slope*u+intercept-y;order=np.argsort(residual);risk=max(0.,float(residual[order[np.searchsorted(np.cumsum(w[order]),.9)]]))
    write(RUN/f'models/{held}/calibration.json',{'calibration_matches':['J03WPY'],'producer':which,'slope':slope,'intercept':intercept,'conservative_overprediction_q90':risk,'model':freeze['model'],'objective':'equal-half analytic matched-count affine fit; conservative q90 overprediction margin; no F1 search','reference_reads':READS.copy(),'case_inputs':[v[2]['cache'] for v in cases]});ALLOWED.clear();print('V9 CALIBRATION',held,'slope',slope,'intercept',intercept,'risk',risk,flush=True)


def predict_case(held,match,half,condition,arms,full=False):
    baseline=old.c.previous.incumbent(match,half,condition);out={'baseline':baseline.copy()}
    if condition=='clean':out.update({a:baseline.copy() for a in arms});return out,{'clean_bypass':True,'changed':{a:0 for a in arms}}
    which='full' if full else producer(match)
    if full:
        meta=make_case('full',match,half);v,edits,_=load_case('full',match,half,{'rows':{f'{match}-{half}':meta}})
    else:v,edits,meta=load_case(which,match,half)
    np.testing.assert_array_equal(v['baseline'],baseline);model,freeze=fitted(held);cal=read(RUN/f'models/{held}/calibration.json');assert cal['model']==freeze['model'];gain=cal['slope']*model.predict(v['features'])+cal['intercept'];chosen={}
    for arm in arms:
        utility=gain-selector.PENALTY*v['delta_n']-(cal['conservative_overprediction_q90'] if arm=='conservative' else 0.);selected=selector.choose(edits,utility);out[arm]=selector.apply(baseline,edits,selected);chosen[arm]={'edits':len(selected),'insert':sum(edits[i].kind==0 for i in selected),'merge':sum(edits[i].kind==1 for i in selected),'boundary':sum(edits[i].kind==2 for i in selected),'complete_path':sum(edits[i].kind==3 for i in selected)}
    return out,{'cases':len(edits),'selected':chosen,'changed':{a:int((out[a]!=baseline).sum()) for a in arms},'cache':meta['cache']}


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={}
    for held in TRAIN:
        freeze=read(RUN/f'models/{held}/freeze.json');assert freeze['validation_split']==split(held)
        for half in ['firstHalf','secondHalf']:
            for condition in ['clean','corrupted']:
                key=f'{held}-{half}-{condition}';v,meta=predict_case(held,held,half,condition,ARMS[1:]);path=RUN/f'oof/{held}'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);views[key]=meta;print('V9 OOF',key,meta,flush=True)
    assert not READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'primary_reference_reads':READS.copy(),'model_freezes':{m:pin(RUN/f'models/{m}/freeze.json') for m in TRAIN},'calibrations':{m:pin(RUN/f'models/{m}/calibration.json') for m in TRAIN},'case_seal':pin(RUN/'case-seal.json'),'protocol_sha256':sha256(RUN/'protocol.json')})


def validate():
    verify();seal=read(RUN/'oof-seal.json');assert len(seal['outputs'])==16 and not seal['primary_reference_reads'];checked(seal['case_seal']);loaded={}
    for held in TRAIN:
        freeze=read(checked(seal['model_freezes'][held]));cal=read(checked(seal['calibrations'][held]));checked(freeze['model']);assert freeze['validation_split']==split(held) and cal['calibration_matches']==['J03WPY'];assert freeze['selector_fit_matches']==split(held)['selector_fit_matches']
        for row in freeze['rows']:assert held not in row['producer_primary_matches'] and row['match'] not in row['producer_primary_matches'] and row['match'] in freeze['selector_fit_matches'];checked(row['cache'])
        for half in ['firstHalf','secondHalf']:
            for condition in ['clean','corrupted']:
                key=f'{held}-{half}-{condition}';path=RUN/f'oof/{held}'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
                with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
                np.testing.assert_array_equal(v['baseline'],old.c.previous.incumbent(held,half,condition))
                for a in ARMS:
                    parents.gs.check_lock(v[a],v['baseline'],v['baseline']<2)
                    if condition=='clean':np.testing.assert_array_equal(v[a],v['baseline'])
                loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One current OOF audit only')
    seal,loaded=validate();ALLOWED.update(TRAIN);pm=official_metrics();details={}
    for m in TRAIN:
        y=join_halves([truth(m,h) for h in ['firstHalf','secondHalf']]);details[m]={c:{a:score_full(pm,y,join_halves([loaded[f'{m}-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ARMS} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in ARMS};overall=avg(TRAIN,['clean','corrupted']);conditions={c:avg(TRAIN,[c]) for c in ['clean','corrupted']};folds={str(p):avg(TRAIN[:2] if p==0 else TRAIN[2:],['clean','corrupted']) for p in [0,1]};assert abs(overall['baseline']-.7099817195618008)<1e-12;valid=[a for a in ARMS[1:] if overall[a]>overall['baseline'] and conditions['corrupted'][a]>conditions['corrupted']['baseline'] and all(v[a]>=v['baseline'] for v in folds.values())];selected=max(['baseline']+valid,key=lambda a:overall[a]);result={'overall':overall,'conditions':conditions,'per_producer_pair':folds,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':details,'selected':selected,'passed':selected!='baseline','clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'oof-seal.json'),'reference_reads':READS.copy(),'protocol_sha256':sha256(RUN/'protocol.json'),'interpretation':'Four strict crossed selector validation runs; same complete TRAIN2 producer for each selector fitting/held pair. Adaptive reused development; no current full WQQ candidate.'};write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_producer_pair','per_match','selected','passed']}),flush=True)


def authorize():
    global FULL
    validate();a=read(RUN/'oof-audit.json');checked(a['prediction_seal']);assert a['passed'] and a['selected']!='baseline';FULL=True;return a['selected']


def full_run():
    selected=authorize();fit('full');calibrate('full');write(RUN/'full-selection.json',{'selected':selected,'OOF_audit':pin(RUN/'oof-audit.json'),'model_freeze':pin(RUN/'models/full/freeze.json'),'calibration':pin(RUN/'models/full/calibration.json'),'no_fallback':True});ALLOWED.clear();start=len(READS);outputs={};views={}
    for half in ['firstHalf','secondHalf']:
        for condition in ['clean','corrupted']:
            key=f'J03WQQ-{half}-{condition}';v,meta=predict_case('full','J03WQQ',half,condition,[selected],True);path=RUN/'full-predictions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);views[key]=meta;print('V9 FULL',key,meta,flush=True)
    assert not READS[start:];write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'selection':pin(RUN/'full-selection.json'),'prediction_reference_reads':READS[start:]})


def audit_full():
    if (RUN/'full-audit.json').exists():raise FileExistsError('One current full audit only')
    selected=authorize();seal=read(RUN/'full-seal.json');selection=read(checked(seal['selection']));checked(selection['model_freeze']);checked(selection['calibration']);assert len(seal['outputs'])==4 and not seal['prediction_reference_reads'];loaded={}
    for half in ['firstHalf','secondHalf']:
        for condition in ['clean','corrupted']:
            key=f'J03WQQ-{half}-{condition}';path=RUN/'full-predictions'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            np.testing.assert_array_equal(v['baseline'],old.c.previous.incumbent('J03WQQ',half,condition));parents.gs.check_lock(v[selected],v['baseline'],v['baseline']<2)
            if condition=='clean':np.testing.assert_array_equal(v[selected],v['baseline'])
            loaded[key]=v
    ALLOWED.clear();ALLOWED.add('J03WQQ');pm=official_metrics();y=join_halves([truth('J03WQQ',h) for h in ['firstHalf','secondHalf']]);conditions={c:{a:score_full(pm,y,join_halves([loaded[f'J03WQQ-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ['baseline',selected]} for c in ['clean','corrupted']};overall={a:float(np.mean([conditions[c][a]['spell_f1_mean'] for c in ['clean','corrupted']])) for a in ['baseline',selected]};assert abs(overall['baseline']-.7708462421456541)<1e-12;result={'overall':overall,'conditions':conditions,'selected':selected,'improved':overall[selected]>overall['baseline'],'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'full-seal.json'),'reference_reads':READS.copy(),'no_fallback':True,'interpretation':'One current selected composition on reused WQQ development, not private challenge accuracy.'};write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'conditions':{c:{a:v['spell_f1_mean'] for a,v in d.items()} for c,d in conditions.items()},'improved':result['improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=TRAIN);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(guard)
    if args.stage in ['fit','calibrate']:
        assert args.held is not None;globals()[args.stage](args.held)
    else:globals()[args.stage]()
