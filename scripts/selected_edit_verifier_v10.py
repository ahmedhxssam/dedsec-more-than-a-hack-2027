"""Frozen selected-action boundary verifier with exact complete-pipeline parents."""
from __future__ import annotations
import argparse, datetime, json, os, pickle, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from threadpoolctl import threadpool_limits
from second_zero import selected_edit_verifier as verifier
from scripts import full_pipeline_selector_v9 as prior

RUN=ROOT/'runs/selected-edit-verifier-v10';REPORT=ROOT/'reports/selected-edit-verifier-v10'
TRAIN=prior.TRAIN;HALVES=['firstHalf','secondHalf'];ARMS=['baseline','boundary_proposals','boundary_verified']
ALLOWED=set();FULL=False;HASH_ONLY=False;STAGE='prepare';READS=[]
read=prior.read;write=prior.write;rel=prior.rel;pin=prior.pin;checked=prior.checked
sha256=prior.sha256;selector=prior.selector


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.fsdecode(args[0])
    if any(s in path for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved assets closed')
    if 'J03WQQ' in path and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed until frozen full stage')
    if path.endswith('.truth.npy'):
        if Path(path).name.split('-')[0] not in ALLOWED:raise PermissionError('Reference outside declared role')
        READS.append({'stage':STAGE,'path':rel(path),'allowed':sorted(ALLOWED)})


def prepare():
    best=read(ROOT/'configs/development_best.json');assert best['score']['overall']==.7708462421456541
    p=read(prior.RUN/'protocol.json');inputs={**p['sources'],**p['inputs']}
    for path in [prior.RUN/'protocol.json',prior.RUN/'case-seal.json',ROOT/'reports/full-pipeline-selector-v9/artifact-freeze.json',ROOT/'reports/full-pipeline-selector-v9/verdict.json']:
        inputs[rel(path)]=sha256(path)
    artifacts=read(ROOT/'reports/full-pipeline-selector-v9/artifact-freeze.json')
    inputs.update(artifacts['hashes'])
    write(RUN/'protocol.json',{
        'kind':'selected-boundary-verifier-v10','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline':best['score'],'validation':[prior.split(m) for m in TRAIN],
        'proposer':'Frozen complete-pipeline quality delta + .01 owner unary delta from v9 case cache. Keep strictly positive scores; nonoverlapping weighted interval selection against exact baseline zero. No newly fit v9 regressor used as proposer or verifier feature.',
        'selection_distribution':'Select BEFORE reading this match truth. Verifier fitting and WPY calibration contain ONLY actions this exact proposer selected. Veto only, with no reselection or replacement after veto.',
        'boundary_contract':'kind2 only; exactly2 old and2 new complete H/A spells, unchanged endpoints and team sequence, one shifted internal boundary. No insert, merge or complete-challenger windows. All spell counts, D/X and clean outputs exact.',
        'verifier':{'class':'standardized Ridge','alpha':verifier.ALPHA,'features':verifier.FEATURE_NAMES,'target':'Exact individual change in one-to-one matched spell count averaged over all5 official IoUs, with pooled both-half counts. Count-neutral means its sign equals exact full-match F1-delta sign.'},
        'calibration':'WPY only, selected actions from the SAME actual primary producer as held/full inference. Equal-half analytic affine matching-gain calibration, slope clipped[0,5]; accept only calibrated expected gain>0. No grid, risk or F1 threshold tuning.',
        'ancestry':'Same complete TRAIN2 producer for fitting and held cases. Verifier fits paired other excluded match only; both excluded from all proposer primary weights. Final verifier fits four target-excluded matches. Frozen full TRAIN4 producer and WPY-full calibrator for full test.',
        'evaluation':'All16 OOF views with3arms sealed then audit once. Report unverified boundary control, but only predetermined boundary_verified is tested on WQQ. Exactly one full composition regardless of OOF outcome to complete requested research evaluation; no OOF selection or WQQ fallback. Promotion needs positive OOF mean, both primary-pair deltas nonnegative, and positive full overall/corrupted with exact clean.',
        'limits':'Four independent TRAIN matches; one verifier-fit match per outer validation,4 in final. TRAIN2 primary fitting parents versus TRAIN4 full producer remains. WPY is reused calibration, WQQ adaptively reused development, not untouched challenge accuracy. No new independently labeled data or guaranteed risk bound.',
        'sources':{rel(Path(__file__)):sha256(Path(__file__)),rel(Path(verifier.__file__)):sha256(Path(verifier.__file__))},'inputs':inputs})
    print('V10 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def source_case(which,match,half):
    path=prior.RUN/f'cases/producer{which}/{match}-{half}-corrupted.npz'
    p=read(RUN/'protocol.json');assert sha256(path)==p['inputs'][rel(path)]
    with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
    return v,prior.unpack_edits(v),pin(path)


def case_path(which,match,half):return RUN/f'selected/producer{which}/{match}-{half}.npz'


def make_case(which,match,half):
    v,edits,source=source_case(which,match,half)
    selected=verifier.select_boundaries(edits,v['proxy']);ee=[edits[i] for i in selected]
    x=verifier.features(v['features'],v['proxy'],edits,selected)
    out={'features':x,'proxy':v['proxy'][selected],'source_indices':selected,'baseline':v['baseline'],**prior.pack_edits(ee)}
    path=case_path(which,match,half);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**out)
    value={'cache':pin(path),'source_case':source,'producer':str(which),'match':match,'half':half,'selected':len(ee),'features':len(verifier.FEATURE_NAMES)}
    print('V10 SELECTED',which,match,half,len(ee),flush=True);return value


def select():
    verify();ALLOWED.clear();rows={}
    for m in TRAIN:
        for h in HALVES:rows[f'{m}-{h}']=make_case(prior.producer(m),m,h)
    for p in ['0','1']:
        for h in HALVES:rows[f'WPY-{p}-{h}']=make_case(p,'J03WPY',h)
    assert not READS;write(RUN/'selected-seal.json',{'rows':rows,'reference_reads':READS.copy(),'protocol':pin(RUN/'protocol.json')})


def load_case(which,match,half):
    if which=='full':
        path=RUN/('full-inference-selected-seal.json' if match=='J03WQQ' else 'full-selected-seal.json')
    else:path=RUN/'selected-seal.json'
    seal=read(path)
    key=f'WPY-{which}-{half}' if match=='J03WPY' else f'{match}-{half}';meta=seal['rows'][key]
    assert meta['producer']==str(which) and meta['match']==match
    with np.load(checked(meta['cache']),allow_pickle=False) as z:v={k:z[k] for k in z.files}
    return v,prior.unpack_edits(v),meta


def targets(match,which):
    cases=[load_case(which,match,h) for h in HALVES]
    ys=[prior.truth(match,h) for h in HALVES]
    tt=selector.paired_targets(prior.official_metrics(),ys,[v[0]['baseline'] for v in cases],[v[1] for v in cases])
    return cases,tt


def fit(which):
    verify();matches=TRAIN if which=='full' else prior.split(which)['selector_fit_matches'];ALLOWED.clear();ALLOWED.update(matches)
    xx=[];yy=[];ww=[];rows=[]
    for m in matches:
        p=prior.producer(m);proof=prior.producer_proof(p)
        assert m not in proof['primary_label_matches']
        if which!='full':assert which not in proof['primary_label_matches'] and p==prior.producer(which)
        cases,tt=targets(m,p)
        for (v,edits,meta),t in zip(cases,tt):
            good=np.isfinite(t).all(1);assert all(verifier.is_boundary(e) for e in edits)
            xx.append(v['features'][good]);yy.append(t[good,0]);ww.append(np.full(int(good.sum()),1/max(1,int(good.sum()))))
            rows.append({'match':m,'half':meta['half'],'selected':len(edits),'valid_targets':int(good.sum()),'positive':int((t[good,0]>0).sum()),'negative':int((t[good,0]<0).sum()),'target_sum':float(t[good,0].sum()),'cache':meta['cache'],'producer_primary_matches':proof['primary_label_matches']})
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean()
    model=verifier.GainVerifier().fit(x,y,w);folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False)
    with (folder/'model.pkl').open('xb') as f:pickle.dump(model,f)
    write(folder/'freeze.json',{'verifier_fit_matches':matches,'held_match':None if which=='full' else which,'shape':list(x.shape),'rows':rows,'model':pin(folder/'model.pkl'),'reference_reads':READS.copy(),'protocol':pin(RUN/'protocol.json')});ALLOWED.clear()
    print('V10 FIT',which,x.shape,flush=True)


def fitted(which):
    freeze=read(RUN/f'models/{which}/freeze.json')
    with checked(freeze['model']).open('rb') as f:model=pickle.load(f)
    return model,freeze


def calibrate(which):
    verify();p='full' if which=='full' else prior.producer(which);model,freeze=fitted(which);ALLOWED.clear();ALLOWED.add('J03WPY')
    cases,tt=targets('J03WPY',p);uu=[];yy=[];ww=[]
    for (v,_,_),t in zip(cases,tt):
        good=np.isfinite(t).all(1);uu.append(model.predict(v['features'][good]));yy.append(t[good,0]);ww.append(np.full(int(good.sum()),1/max(1,int(good.sum()))))
    cal=verifier.calibrate(np.concatenate(uu),np.concatenate(yy),np.concatenate(ww))
    cal.update({'calibration_matches':['J03WPY'],'producer':p,'model':freeze['model'],'selected_action_cases':[v[2]['cache'] for v in cases],'reference_reads':READS.copy()})
    write(RUN/f'models/{which}/calibration.json',cal);ALLOWED.clear();print('V10 CAL',which,{k:cal[k] for k in ['slope','intercept','rows']},flush=True)


def predict_case(which,match,half,condition,arms):
    baseline=prior.old.c.previous.incumbent(match,half,condition);out={'baseline':baseline.copy()}
    if condition=='clean':
        out.update({a:baseline.copy() for a in arms});return out,{'clean_bypass':True,'changed':{a:0 for a in arms}}
    p='full' if which=='full' else prior.producer(match);v,edits,meta=load_case(p,match,half)
    np.testing.assert_array_equal(v['baseline'],baseline);model,freeze=fitted(which);cal=read(RUN/f'models/{which}/calibration.json');assert cal['model']==freeze['model']
    gain=cal['slope']*model.predict(v['features'])+cal['intercept'];keep=verifier.accepted(model.predict(v['features']),cal)
    ids={'boundary_proposals':np.arange(len(edits)),'boundary_verified':keep}
    for a in arms:out[a]=verifier.apply(baseline,edits,ids[a])
    return out,{'selected_before_veto':len(edits),'accepted':len(keep),'vetoed':len(edits)-len(keep),'predicted_accepted_gain':float(gain[keep].sum()),'accepted_local_indices':keep.tolist(),'changed':{a:int((out[a]!=baseline).sum()) for a in arms},'cache':meta['cache']}


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={}
    for m in TRAIN:
        for h in HALVES:
            for c in ['clean','corrupted']:
                key=f'{m}-{h}-{c}';v,meta=predict_case(m,m,h,c,ARMS[1:]);path=RUN/f'oof/{m}/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:np.savez_compressed(f,**v)
                outputs[rel(path)]=sha256(path);views[key]=meta;print('V10 OOF',key,meta['changed'],flush=True)
    assert not READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'reference_reads':READS.copy(),'protocol':pin(RUN/'protocol.json'),'model_freezes':{m:pin(RUN/f'models/{m}/freeze.json') for m in TRAIN},'calibrations':{m:pin(RUN/f'models/{m}/calibration.json') for m in TRAIN},'selected_seal':pin(RUN/'selected-seal.json')})


def validate_oof():
    verify();seal=read(RUN/'oof-seal.json');checked(seal['protocol']);checked(seal['selected_seal']);assert len(seal['outputs'])==16 and not seal['reference_reads'];loaded={}
    for m in TRAIN:
        freeze=read(checked(seal['model_freezes'][m]));cal=read(checked(seal['calibrations'][m]));checked(freeze['model']);assert freeze['verifier_fit_matches']==prior.split(m)['selector_fit_matches'];assert cal['producer']==prior.producer(m) and cal['calibration_matches']==['J03WPY']
        for row in freeze['rows']:
            assert row['match'] in freeze['verifier_fit_matches'] and row['match'] not in row['producer_primary_matches'] and m not in row['producer_primary_matches'];checked(row['cache'])
        for h in HALVES:
            for c in ['clean','corrupted']:
                key=f'{m}-{h}-{c}';path=RUN/f'oof/{m}/{key}.npz';assert sha256(path)==seal['outputs'][rel(path)]
                with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
                np.testing.assert_array_equal(v['baseline'],prior.old.c.previous.incumbent(m,h,c))
                for a in ARMS:
                    selector.gs.check_lock(v[a],v['baseline'],v['baseline']<2)
                    assert np.array_equal(selector.gs.spells(v[a])[:,2],selector.gs.spells(v['baseline'])[:,2])
                    if c=='clean':np.testing.assert_array_equal(v[a],v['baseline'])
                loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One OOF audit only')
    seal,loaded=validate_oof();ALLOWED.update(TRAIN);pm=prior.official_metrics();details={}
    for m in TRAIN:
        y=prior.join_halves([prior.truth(m,h) for h in HALVES]);details[m]={c:{a:prior.score_full(pm,y,prior.join_halves([loaded[f'{m}-{h}-{c}'][a] for h in HALVES]))[0] for a in ARMS} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in ARMS}
    overall=avg(TRAIN,['clean','corrupted']);conditions={c:avg(TRAIN,[c]) for c in ['clean','corrupted']};pairs={str(i):avg(TRAIN[:2] if i==0 else TRAIN[2:],['clean','corrupted']) for i in [0,1]};a='boundary_verified';passed=overall[a]>overall['baseline'] and all(v[a]>=v['baseline'] for v in pairs.values())
    assert abs(overall['baseline']-.7099817195618008)<1e-12
    result={'overall':overall,'conditions':conditions,'per_producer_pair':pairs,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':details,'passed':passed,'predetermined_full_candidate':a,'full_test_regardless_of_OOF_predeclared':True,'clean_exact':True,'spell_counts_exact':True,'prediction_seal':pin(RUN/'oof-seal.json'),'reference_reads':READS.copy()}
    write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_producer_pair','per_match','passed']}),flush=True)


def full_run():
    global FULL
    validate_oof();audit=read(RUN/'oof-audit.json');checked(audit['prediction_seal']);fit('full')
    rows={}
    for h in HALVES:rows[f'WPY-full-{h}']=make_case('full','J03WPY',h)
    write(RUN/'full-selected-seal.json',{'rows':rows,'stage':'calibration_only','reference_reads':[]})
    calibrate('full');write(RUN/'full-selection.json',{'selected':'boundary_verified','OOF_audit':pin(RUN/'oof-audit.json'),'model_freeze':pin(RUN/'models/full/freeze.json'),'calibration':pin(RUN/'models/full/calibration.json'),'unverified_control_not_full_tested':True,'no_fallback':True})
    FULL=True;ALLOWED.clear();start=len(READS)
    for h in HALVES:rows[f'J03WQQ-{h}']=make_case('full','J03WQQ',h)
    # A new immutable stage seal includes calibration and inference selections.
    write(RUN/'full-inference-selected-seal.json',{'rows':rows,'reference_reads':READS[start:],'selection':pin(RUN/'full-selection.json')})
    outputs={};views={}
    for h in HALVES:
        for c in ['clean','corrupted']:
            key=f'J03WQQ-{h}-{c}';v,meta=predict_case('full','J03WQQ',h,c,['boundary_verified']);path=RUN/f'full-predictions/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:np.savez_compressed(f,**v)
            outputs[rel(path)]=sha256(path);views[key]=meta;print('V10 FULL',key,meta,flush=True)
    assert not READS[start:];write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'prediction_reference_reads':READS[start:],'selection':pin(RUN/'full-selection.json'),'selected_seal':pin(RUN/'full-inference-selected-seal.json')})


def audit_full():
    global FULL
    if (RUN/'full-audit.json').exists():raise FileExistsError('One full audit only')
    verify();FULL=True;ALLOWED.clear();seal=read(RUN/'full-seal.json');selection=read(checked(seal['selection']));checked(selection['model_freeze']);checked(selection['calibration']);checked(seal['selected_seal']);assert len(seal['outputs'])==4 and not seal['prediction_reference_reads'];loaded={};clean_frames=0
    for h in HALVES:
        for c in ['clean','corrupted']:
            key=f'J03WQQ-{h}-{c}';path=RUN/f'full-predictions/{key}.npz';assert sha256(path)==seal['outputs'][rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            np.testing.assert_array_equal(v['baseline'],prior.old.c.previous.incumbent('J03WQQ',h,c));selector.gs.check_lock(v['boundary_verified'],v['baseline'],v['baseline']<2)
            assert np.array_equal(selector.gs.spells(v['boundary_verified'])[:,2],selector.gs.spells(v['baseline'])[:,2])
            if c=='clean':np.testing.assert_array_equal(v['boundary_verified'],v['baseline']);clean_frames+=len(v['baseline'])
            loaded[key]=v
    ALLOWED.add('J03WQQ');pm=prior.official_metrics();y=prior.join_halves([prior.truth('J03WQQ',h) for h in HALVES]);arms=['baseline','boundary_verified'];conditions={c:{a:prior.score_full(pm,y,prior.join_halves([loaded[f'J03WQQ-{h}-{c}'][a] for h in HALVES]))[0] for a in arms} for c in ['clean','corrupted']};overall={a:float(np.mean([conditions[c][a]['spell_f1_mean'] for c in ['clean','corrupted']])) for a in arms};assert abs(overall['baseline']-.7708462421456541)<1e-12
    result={'overall':overall,'conditions':conditions,'improved':overall['boundary_verified']>overall['baseline'],'clean_exact':True,'clean_frames':clean_frames,'D_X_exact':True,'spell_counts_exact':True,'prediction_seal':pin(RUN/'full-seal.json'),'reference_reads':READS.copy(),'current_full_candidate_tests':1,'no_fallback':True,'limitation':'One reused WQQ development match; not untouched/private challenge.'}
    write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'conditions':{c:{a:v['spell_f1_mean'] for a,v in d.items()} for c,d in conditions.items()},'improved':result['improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','select','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=TRAIN);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);prior.torch.set_num_threads(2);sys.addaudithook(guard)
    if args.stage in ['fit','calibrate']:
        assert args.held is not None;globals()[args.stage](args.held)
    else:globals()[args.stage]()
