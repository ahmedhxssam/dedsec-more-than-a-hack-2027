"""Broader corrupted timeline, same stacked expert, observed-region NLL weight."""
from __future__ import annotations
import argparse,datetime,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from scripts import stacked_gap_recovery_v7 as s
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from scripts.full_baseline_evaluate import join_halves,score_full

c=s.common;b=s.b;e=s.e;gs=s.gs;OLD=s.OLD;TRAIN=s.TRAIN;FOLDS=s.FOLDS
RUN=ROOT/'runs/region-gap-recovery-v8';REPORT=ROOT/'reports/region-gap-recovery-v8';read=s.read;write=s.write;pin=s.pin;check=s.check;truth=s.truth


def prepare():
    p=read(s.RUN/'protocol.json');assert not (s.RUN/'oof-audit.json').exists()
    inputs={**p['inputs'],c.rel(s.RUN/'protocol.json'):sha256(s.RUN/'protocol.json'),c.rel(s.RUN/'oof-seal.json'):sha256(s.RUN/'oof-seal.json')}
    for f in ['0','1']:
        for name in ['freeze.json','calibration.json']:inputs[c.rel(s.RUN/f'models/{f}/{name}')]=sha256(s.RUN/f'models/{f}/{name}')
    write(RUN/'protocol.json',{'kind':'calibrated-observed-region-scope-v8','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':p['baseline'],'OOF_baseline':p['OOF_baseline'],
      'weights':'Reuse exactly v7 primary weights, features and gap temperature/alpha. No new primary fitting or grid. Added before reading any v7 OOF score.',
      'calibration':'WPY-only extra observed-region scalar alpha in[0,1] minimizing H/A NLL, gap alpha fixed from v7. Per-frame weight gapalpha*missing25+observedalpha*(1-missing25). Shared v7 temperature. Endpoints0/1 included. No F1 calibration.',
      'scope':'Clean exact incumbent bypass; all D/X fixed. Corrupted live timeline can change beyond old gap mask. Zero observed alpha reuses exact v7 gap-calibrated result. Otherwise original graph/lattice proposals plus dense raw-gap proposals and exact incumbent spell cover, feasible against all-live mutable mask. No quadratic dense grid across observed timeline.',
      'candidate':'One fixed region-calibrated composition against same .770846 incumbent. Score both halves together; strict OOF overall/corrupted gain and bothfold nonnegative. Compare v7/v8 eligible OOF results before exactly one full new WQQ model candidate; no fallback.',
      'limits':'Repeated adaptive development; .9 requires .994437 corrupted with clean fixed. H/A-only observed oracle was .878635 overall feasible, not a maximum. No private challenge claim. WR9/challenge closed.',
      'sources':{**p['sources'],c.rel(Path(__file__)):sha256(Path(__file__))},'inputs':inputs})
    print('V8 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    p=read(RUN/'protocol.json');c.HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:c.HASH_ONLY=False
    return p


def calibrate(which):
    verify();c.ALLOWED.add('J03WPY');model,freeze=s.fitted(which);cal=read(s.RUN/f'models/{which}/calibration.json');parts=[]
    for half,raw,p,canonical,dep in s.calibration_parent(which):
        z=s.predict_owner(model,s.owner.features(raw,canonical))/cal['temperature'];y=truth('J03WPY',half);support=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest');take=(y<2)&(p['predicted']<2);parts.append((expit(z[take]),y[take],expit(p['signals'][take,0]),support[take],dep))
    new=np.concatenate([v[0] for v in parts]);y=np.concatenate([v[1] for v in parts]);prior=np.concatenate([v[2] for v in parts]);support=np.concatenate([v[3] for v in parts])
    def loss(alpha):
        w=cal['alpha']*support+alpha*(1-support);p=np.clip((1-w)*prior+w*new,1e-8,1-1e-8);return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
    solution=float(minimize_scalar(loss,bounds=(0,1),method='bounded').x);alpha=min([0.,solution,1.],key=loss);write(RUN/f'calibration/{which}.json',{'observed_alpha':alpha,'gap_alpha':cal['alpha'],'temperature':cal['temperature'],'calibration_matches':['J03WPY'],'objective':'One observed-region scalar primary NLL, fixed gap weight/T; no F1 search','NLL_at_observed0':loss(0),'NLL_selected':loss(alpha),'model':freeze['model'],'gap_calibration':pin(s.RUN/f'models/{which}/calibration.json'),'parent_inputs':[v[4] for v in parts],'reference_reads':c.READS.copy()});c.ALLOWED.clear();print('V8 CAL',which,'observed alpha',alpha,'NLL',loss(0),loss(alpha),flush=True)


def prediction(which,key):
    match,half,condition=key.split('-');baseline=c.previous.incumbent(match,half,condition)
    if condition=='clean':return {'baseline':baseline.copy(),'region':baseline.copy()},{'changed':0,'clean_bypass':True}
    cal=read(RUN/f'calibration/{which}.json')
    if cal['observed_alpha']==0:
        out,meta=s.prediction(which,key,['calibrated']);return {'baseline':baseline.copy(),'region':out['calibrated']},{'changed':meta['changed']['calibrated'],'observed_alpha':0.,'exact_v7_scope':True}
    raw=b.raw_half(key);p=b.ceiling.current(match,half,condition);folder=OLD/'full-transfer/full_baseline' if which=='full' else OLD/f'baseline/fold{which}'
    with np.load(folder/(key+'.npz'),allow_pickle=False) as z:owners={f'owner_{i}':z[f'owner_{i}'] for i in range(3)}
    model,freeze=s.fitted(which);assert freeze['model']==cal['model'];new=expit(s.predict_owner(model,s.owner.features(raw,e.component(p['signals'],owners)))/cal['temperature']);support=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest');w=cal['gap_alpha']*support+cal['observed_alpha']*(1-support);signals=p['signals'].copy();signals[:,0]=logit(np.clip((1-w)*expit(signals[:,0])+w*new,1e-7,1-1e-7)).astype(np.float32);quality=b.original_quality(which);preliminary,arcs,_=s.decode_graph(signals,p['status_mask'],quality);dense=b.dense.proposals(preliminary,p['eligible_mask'],arcs);arcs=gs.pool(preliminary,preliminary<2,np.r_[arcs,dense,gs.spells(baseline)]);qo=b.old_quality(quality,signals,arcs);credit,_=b.fitted(which);qr=e.EvidenceIntervals(raw,e.component(signals,owners)).predict(credit,arcs);cc=read(e.RUN/f'models/{which}/calibration.json');pred=gs.decode(preliminary,preliminary<2,arcs,(1-cc['alpha'])*qo+cc['alpha']*qr,.4,signals[:,0],.01);gs.check_lock(pred,baseline,baseline<2)
    return {'baseline':baseline.copy(),'region':pred},{'changed':int((pred!=baseline).sum()),'observed_alpha':cal['observed_alpha'],'gap_alpha':cal['gap_alpha'],'pool':len(arcs),'D_X_exact':True}


def predict_oof():
    verify();c.ALLOWED.clear();outputs={};views={};start=time.monotonic()
    for f,(_,held) in enumerate(FOLDS):
        for key in b.keys(held):
            v,meta=prediction(str(f),key);path=RUN/f'oof/fold{f}'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[c.rel(path)]=sha256(path);views[key]=meta;print('V8 OOF',key,meta,'seconds',time.monotonic()-start,flush=True)
    assert not c.READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'primary_reference_reads':[],'protocol_sha256':sha256(RUN/'protocol.json'),'calibrations':{str(f):pin(RUN/f'calibration/{f}.json') for f in [0,1]}})


def validate():
    verify();s.validate();seal=read(RUN/'oof-seal.json');assert len(seal['outputs'])==16 and not seal['primary_reference_reads'];loaded={}
    for f,(_,held) in enumerate(FOLDS):
        cal=read(check(seal['calibrations'][str(f)]));check(cal['model']);check(cal['gap_calibration']);assert cal['calibration_matches']==['J03WPY']
        for key in b.keys(held):
            path=RUN/f'oof/fold{f}'/(key+'.npz');assert sha256(path)==seal['outputs'][c.rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            m,h,condition=key.split('-');np.testing.assert_array_equal(v['baseline'],c.previous.incumbent(m,h,condition));gs.check_lock(v['region'],v['baseline'],v['baseline']<2)
            if condition=='clean':np.testing.assert_array_equal(v['region'],v['baseline'])
            loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One OOF audit only')
    seal,loaded=validate();c.ALLOWED.update(TRAIN);pm=official_metrics();details={};arms=['baseline','region']
    for m in TRAIN:
        y=join_halves([truth(m,h) for h in ['firstHalf','secondHalf']]);details[m]={condition:{a:score_full(pm,y,join_halves([loaded[f'{m}-{h}-{condition}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in arms} for condition in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][condition][a]['spell_f1_mean'] for m in ms for condition in cs])) for a in arms};overall=avg(TRAIN,['clean','corrupted']);conditions={condition:avg(TRAIN,[condition]) for condition in ['clean','corrupted']};folds={str(f):avg(h,['clean','corrupted']) for f,(_,h) in enumerate(FOLDS)};assert abs(overall['baseline']-.7099817195618008)<1e-12;passed=overall['region']>overall['baseline'] and conditions['corrupted']['region']>conditions['corrupted']['baseline'] and all(v['region']>=v['baseline'] for v in folds.values());result={'overall':overall,'conditions':conditions,'per_fold':folds,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':details,'selected':'region' if passed else 'baseline','passed':passed,'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'oof-seal.json'),'reference_reads':c.READS.copy(),'interpretation':'One scope extension with primary weights reused, before v7 score. Repeated development; no WQQ candidate scored.'};write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_fold','selected','passed']}),flush=True)


def authorize():
    validate();a=read(RUN/'oof-audit.json');check(a['prediction_seal']);assert a['passed'] and a['selected']=='region';assert not (s.RUN/'full-audit.json').exists() and not (c.RUN/'full-audit.json').exists() and not (c.previous.RUN/'full-audit.json').exists();c.FULL=True


def full_run():
    authorize();s.fit('full');s.calibrate('full');calibrate('full');write(RUN/'full-selection.json',{'selected':'region','OOF_audit':pin(RUN/'oof-audit.json'),'no_fallback':True,'reused_primary_family':'v7'});outputs={};views={};c.ALLOWED.clear()
    for key in b.keys(['J03WQQ']):
        v,meta=prediction('full',key);path=RUN/'full-predictions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[c.rel(path)]=sha256(path);views[key]=meta;print('V8 FULL',key,meta,flush=True)
    write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'selection':pin(RUN/'full-selection.json'),'primary_reference_reads':[]})


def audit_full():
    if (RUN/'full-audit.json').exists():raise FileExistsError('One full audit only')
    authorize();seal=read(RUN/'full-seal.json');check(seal['selection']);assert len(seal['outputs'])==4 and not seal['primary_reference_reads'];loaded={}
    for key in b.keys(['J03WQQ']):
        path=RUN/'full-predictions'/(key+'.npz');assert sha256(path)==seal['outputs'][c.rel(path)]
        with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        m,h,condition=key.split('-');np.testing.assert_array_equal(v['baseline'],c.previous.incumbent(m,h,condition));gs.check_lock(v['region'],v['baseline'],v['baseline']<2)
        if condition=='clean':np.testing.assert_array_equal(v['region'],v['baseline'])
        loaded[key]=v
    c.ALLOWED.add('J03WQQ');pm=official_metrics();y=join_halves([truth('J03WQQ',h) for h in ['firstHalf','secondHalf']]);conditions={condition:{a:score_full(pm,y,join_halves([loaded[f'J03WQQ-{h}-{condition}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ['baseline','region']} for condition in ['clean','corrupted']};overall={a:float(np.mean([conditions[condition][a]['spell_f1_mean'] for condition in ['clean','corrupted']])) for a in ['baseline','region']};assert abs(overall['baseline']-.7708462421456541)<1e-12;result={'overall':overall,'conditions':conditions,'selected':'region','improved':overall['region']>overall['baseline'],'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'full-seal.json'),'reference_reads':c.READS.copy(),'no_fallback':True,'interpretation':'One reused full development match; no private challenge claim'};write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'improved':result['improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--fold',choices=['0','1']);args=p.parse_args();c.STAGE=args.stage;threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(c.guard)
    if args.stage=='calibrate':assert args.fold is not None;calibrate(args.fold)
    else:globals()[args.stage]()
