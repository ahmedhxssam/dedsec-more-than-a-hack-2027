"""Strict crossfit nonlinear gap ownership expert; exact current-best comparison."""
from __future__ import annotations
import argparse,datetime,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from second_zero import stacked_gap_owner as owner
from second_zero import gap_spell as gs
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from second_zero.full_baseline_sources import scientific_sources
from scripts import gap_spell_evidence as e
from scripts import observed_ball_recovery_v6 as common
from scripts.global_team_context_probe import owner as predict_owner
from scripts.vision_context_memory_owner import sample_spells
from scripts.aggressive_unseen_pipeline import decode_graph
from scripts.full_baseline_evaluate import join_halves,score_full

b=e.base;OLD=b.OLD;TRAIN=b.TRAIN;FOLDS=b.FOLDS;RUN=ROOT/'runs/stacked-gap-recovery-v7';REPORT=ROOT/'reports/stacked-gap-recovery-v7';ARMS=['baseline','fixed025','calibrated']
read=common.read;write=common.write;pin=common.pin;check=common.check;truth=common.truth


def prepare():
    pointer=read(ROOT/'configs/development_best.json');manifest=read(check(pointer['manifest']));prior=read(common.RUN/'oof-audit.json');assert not prior['passed']
    sources=[Path(__file__),Path(owner.__file__),Path(common.__file__),Path(common.previous.__file__),Path(e.__file__),Path(b.__file__),ROOT/'scripts/vision_context_memory_owner.py',ROOT/'scripts/full_baseline_evaluate.py',ROOT/'second_zero/gap_spell.py',ROOT/'second_zero/gap_spell_dense.py',ROOT/'second_zero/gap_spell_model.py',*scientific_sources()]
    write(RUN/'protocol.json',{'kind':'strict-crossfit-stacked-gap-owner-v7','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':pointer['score'],'OOF_baseline':pointer['strict_OOF'],'model':owner.MODEL,'arms':ARMS,
      'features':'425 per perspective: raw team/pitch motion290, raw observed ball/players98, uncertain bounded interpolation24, canonical crossfit uniform3/short/reliability/physical13. Sign all team-antisymmetric evidence; raw-only fields enforced. No final full-model logit in fitted features.',
      'training':'TRAIN2 outer weights; TRAIN4 full only if eligible. Clean/corruption100/corruption271, both halves. Target-excluded OTHER TRAIN1 canonical learned parents in outer fits, oppositeTRAIN2 parents in full fit. No outerheld weight ancestor. Native stride8 plus full-spell midpoint, half native-frame mass/half equal-spell mass, extra weight1+2*missing25, normalize every half/view to equal mass. Both team perspectives share one classifier.',
      'WPY':'Only analytic bounded NLL calibration: temperature[.25,4] on gap-live H/A, then scalar probability-blend alpha[0,1] minimizing same WPY NLL against actual .7694-family full final owner prior. Include endpoints alpha0/1. No F1 sweep, new thresholds or WQQ calibration.',
      'arms_definition':{'fixed025':'New temperature-calibrated expert, fixed.25*missing25 residual probability blend into actual prior final owner','calibrated':'Same with WPY NLL alpha instead of.25'},
      'composition':'Actual complete original full baseline plus old.5 raw-global expert and selected747 spell scorer are frozen. Update prior final gap owner, original175 geometry/lattice graph .55/.01/16, exact current-best fence outside original gap mask, expanded feasible pool, unchanged747 credit blend and fixed.4/.01 decoder. Zero new weight bypasses exactly to .770846 incumbent.',
      'lock':'Clean exact incumbent bypass. All D/X and outside original gap mask identical. Full-match scores pool both halves and equal conditions/five IoUs.',
      'selection':'Seal all16 views with all3 arms before one OOF audit. Strict overall/corrupted gain, nonnegative bothfold deltas. Highest eligible. V5 andV6 failed and received no WQQ model test. Exactly one final selected family/arm full WQQ test, no fallback.',
      'adaptivity':'Added after V5 broad edits regressed and V6 coordinate gain was negligible/mixed. Reused development throughout; no private challenge claim. .9 nearly saturates fixed-clean theoretical maximum. WR9/challenge closed.',
      'sources':{common.rel(p):sha256(p) for p in sources},'inputs':{**manifest['frozen_files'],common.rel(ROOT/'configs/development_best.json'):sha256(ROOT/'configs/development_best.json'),common.rel(ROOT/pointer['manifest']['path']):pointer['manifest']['sha256'],common.rel(common.RUN/'oof-audit.json'):sha256(common.RUN/'oof-audit.json')}})
    print('V7 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    p=read(RUN/'protocol.json');common.HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:common.HASH_ONLY=False
    return p


def fitted(which):
    f=read(RUN/f'models/{which}/freeze.json')
    with check(f['model']).open('rb') as h:return pickle.load(h),f


def fit(which):
    verify();matches=TRAIN if which=='full' else FOLDS[int(which)][0];mode='full' if which=='full' else 'oof';common.ALLOWED.update(matches);xx=[];yy=[];ww=[];rows=[];start=time.monotonic()
    for match in matches:
        for half in ['firstHalf','secondHalf']:
            y=truth(match,half);idx,w=sample_spells(y,8)
            for view in ['clean','100','271']:
                raw=b.raw(match,half,view);signals,ancestry=e.evidence(match,half,view,mode,raw);assert match not in ancestry['label_matches'] and set(ancestry['label_matches'])<=set(matches)
                x=owner.features(raw,signals);support=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest');weight=w*(1+2*support[idx]);weight/=weight.sum()
                for t in [0,1]:xx.append(x[idx,t]);yy.append((y[idx]==t).astype(np.int8));ww.append(weight)
                rows.append({'match':match,'half':half,'view':view,'samples':len(idx),'ancestor':ancestry});print('V7 FIT ROWS',which,match,half,view,len(idx),'seconds',time.monotonic()-start,flush=True);del x,raw,signals
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean();del xx,yy,ww
    model=HistGradientBoostingClassifier(**owner.MODEL).fit(x,y,sample_weight=w);folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False)
    with (folder/'model.pkl').open('xb') as f:pickle.dump(model,f)
    write(folder/'freeze.json',{'train_matches':matches,'calibration_matches':[],'model':pin(folder/'model.pkl'),'protocol_sha256':sha256(RUN/'protocol.json'),'shape':list(x.shape),'rows':rows,'reference_reads':common.READS.copy(),'seconds':time.monotonic()-start});common.ALLOWED.clear();print('V7 FIT COMPLETE',which,x.shape,time.monotonic()-start,flush=True)


def calibration_parent(which):
    folder=e.parent_wpy(which);rows=[]
    for half in ['firstHalf','secondHalf']:
        path=folder/f'J03WPY-{half}-corrupted.npz'
        with np.load(path,allow_pickle=False) as z:p={k:z[k] for k in z.files}
        raw=b.raw_half(f'J03WPY-{half}-corrupted');rows.append((half,raw,p,e.component(p['signals'],p),pin(path)))
    return rows


def calibrate(which):
    verify();common.ALLOWED.add('J03WPY');model,freeze=fitted(which);parts=[]
    for half,raw,p,signals,dep in calibration_parent(which):
        z=predict_owner(model,owner.features(raw,signals));y=truth('J03WPY',half);s=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest');take=(y<2)&(p['predicted']<2)&(s>0);parts.append((z[take],y[take],p['signals'][take,0],s[take],dep))
    z=np.concatenate([v[0] for v in parts]);y=np.concatenate([v[1] for v in parts]);prior=expit(np.concatenate([v[2] for v in parts]));support=np.concatenate([v[3] for v in parts]);nll=lambda p:float(np.mean(-y*np.log(np.clip(p,1e-8,1-1e-8))-(1-y)*np.log1p(-np.clip(p,1e-8,1-1e-8))))
    temperature=float(minimize_scalar(lambda t:nll(expit(z/t)),bounds=(.25,4),method='bounded').x);new=expit(z/temperature)
    loss=lambda a:nll((1-a*support)*prior+a*support*new);solution=float(minimize_scalar(loss,bounds=(0,1),method='bounded').x);alpha=min([0.,solution,1.],key=loss)
    write(RUN/f'models/{which}/calibration.json',{'calibration_matches':['J03WPY'],'temperature':temperature,'alpha':alpha,'NLL_original':loss(0),'NLL_selected':loss(alpha),'model':freeze['model'],'parent_inputs':[v[4] for v in parts],'reference_reads':common.READS.copy(),'objective':'Gap-live primary NLL only; temperature then one bounded scalar mixture, no F1 grid'});common.ALLOWED.clear();print('V7 CAL',which,'T',temperature,'alpha',alpha,'NLL',loss(0),loss(alpha),flush=True)


def prediction(which,key,arms):
    match,half,c=key.split('-');baseline=common.previous.incumbent(match,half,c);out={'baseline':baseline.copy()}
    if c=='clean':out.update({a:baseline.copy() for a in arms});return out,{'changed':{a:0 for a in arms},'clean_bypass':True}
    p=b.ceiling.current(match,half,c);raw=b.raw_half(key);folder=OLD/'full-transfer/full_baseline' if which=='full' else OLD/f'baseline/fold{which}'
    with np.load(folder/(key+'.npz'),allow_pickle=False) as z:owners={f'owner_{i}':z[f'owner_{i}'] for i in range(3)}
    canonical=e.component(p['signals'],owners);model,freeze=fitted(which);cal=read(RUN/f'models/{which}/calibration.json');assert cal['model']==freeze['model'];z=predict_owner(model,owner.features(raw,canonical))/cal['temperature'];support=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest');qmodel=b.original_quality(which);credit,_=b.fitted(which);creditcal=read(e.RUN/f'models/{which}/calibration.json');mask=p['eligible_mask']
    for arm in arms:
        weight=.25 if arm=='fixed025' else cal['alpha']
        if weight==0:out[arm]=baseline.copy();continue
        signals=p['signals'].copy();w=weight*support;signals[:,0]=logit(np.clip((1-w)*expit(signals[:,0])+w*expit(z),1e-7,1-1e-7)).astype(np.float32)
        preliminary,arcs,_=decode_graph(signals,p['status_mask'],qmodel);preliminary[~mask]=baseline[~mask];arcs=b.dense.proposals(preliminary,mask,arcs);qo=b.old_quality(qmodel,signals,arcs);qr=e.EvidenceIntervals(raw,e.component(signals,owners)).predict(credit,arcs);q=(1-creditcal['alpha'])*qo+creditcal['alpha']*qr;out[arm]=gs.decode(preliminary,mask,arcs,q,.4,signals[:,0],.01);gs.check_lock(out[arm],baseline,mask)
    return out,{'changed':{a:int((out[a]!=baseline).sum()) for a in arms},'alpha_WPY':cal['alpha'],'temperature_WPY':cal['temperature']}


def predict_oof():
    verify();common.ALLOWED.clear();outputs={};views={};start=time.monotonic()
    for f,(_,held) in enumerate(FOLDS):
        for key in b.keys(held):
            v,meta=prediction(str(f),key,ARMS[1:]);path=RUN/f'oof/fold{f}'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[common.rel(path)]=sha256(path);views[key]=meta;print('V7 OOF',key,meta,'seconds',time.monotonic()-start,flush=True)
    assert not common.READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'primary_reference_reads':[],'models':{str(f):{'freeze':pin(RUN/f'models/{f}/freeze.json'),'calibration':pin(RUN/f'models/{f}/calibration.json')} for f in [0,1]},'protocol_sha256':sha256(RUN/'protocol.json')})


def validate():
    verify();seal=read(RUN/'oof-seal.json');assert len(seal['outputs'])==16 and not seal['primary_reference_reads'];loaded={}
    for f,(train,held) in enumerate(FOLDS):
        freeze=read(check(seal['models'][str(f)]['freeze']));cal=read(check(seal['models'][str(f)]['calibration']));check(freeze['model']);assert freeze['train_matches']==train and cal['calibration_matches']==['J03WPY']
        for row in freeze['rows']:
            a=row['ancestor'];assert row['match'] not in a['label_matches'] and set(a['label_matches'])<=set(train);check(a['manifest'])
        for key in b.keys(held):
            path=RUN/f'oof/fold{f}'/(key+'.npz');assert sha256(path)==seal['outputs'][common.rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            m,h,c=key.split('-');np.testing.assert_array_equal(v['baseline'],common.previous.incumbent(m,h,c));p=b.ceiling.current(m,h,c)
            for arm in ARMS:
                gs.check_lock(v[arm],v['baseline'],p['eligible_mask'])
                if c=='clean':np.testing.assert_array_equal(v[arm],v['baseline'])
            loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One OOF audit only')
    seal,loaded=validate();common.ALLOWED.update(TRAIN);pm=official_metrics();details={}
    for m in TRAIN:
        y=join_halves([truth(m,h) for h in ['firstHalf','secondHalf']]);details[m]={c:{a:score_full(pm,y,join_halves([loaded[f'{m}-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ARMS} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in ARMS};overall=avg(TRAIN,['clean','corrupted']);conditions={c:avg(TRAIN,[c]) for c in ['clean','corrupted']};folds={str(f):avg(h,['clean','corrupted']) for f,(_,h) in enumerate(FOLDS)};assert abs(overall['baseline']-.7099817195618008)<1e-12;valid=[a for a in ARMS[1:] if overall[a]>overall['baseline'] and conditions['corrupted'][a]>conditions['corrupted']['baseline'] and all(v[a]>=v['baseline'] for v in folds.values())];selected=max(['baseline']+valid,key=lambda a:overall[a]);result={'overall':overall,'conditions':conditions,'per_fold':folds,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':details,'selected':selected,'passed':selected!='baseline','clean_exact':True,'D_X_outside_exact':True,'prediction_seal':pin(RUN/'oof-seal.json'),'reference_reads':common.READS.copy(),'interpretation':'Adaptive reused development; full incumbent validation; no WQQ candidate scored'};write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_fold','selected','passed']}),flush=True)


def authorize():
    validate();a=read(RUN/'oof-audit.json');check(a['prediction_seal']);assert a['passed'] and a['selected']!='baseline';assert not (common.RUN/'full-audit.json').exists() and not (common.previous.RUN/'full-audit.json').exists();common.FULL=True;return a['selected']


def full_run():
    selected=authorize();fit('full');calibrate('full');write(RUN/'full-selection.json',{'selected':selected,'OOF_audit':pin(RUN/'oof-audit.json'),'no_fallback':True});outputs={};views={};common.ALLOWED.clear()
    for key in b.keys(['J03WQQ']):
        v,meta=prediction('full',key,[selected]);path=RUN/'full-predictions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[common.rel(path)]=sha256(path);views[key]=meta;print('V7 FULL',key,meta,flush=True)
    write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'selection':pin(RUN/'full-selection.json'),'primary_reference_reads':[]})


def audit_full():
    if (RUN/'full-audit.json').exists():raise FileExistsError('One full audit only')
    selected=authorize();seal=read(RUN/'full-seal.json');check(seal['selection']);assert len(seal['outputs'])==4 and not seal['primary_reference_reads'];loaded={}
    for key in b.keys(['J03WQQ']):
        path=RUN/'full-predictions'/(key+'.npz');assert sha256(path)==seal['outputs'][common.rel(path)]
        with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        m,h,c=key.split('-');np.testing.assert_array_equal(v['baseline'],common.previous.incumbent(m,h,c));p=b.ceiling.current(m,h,c);gs.check_lock(v[selected],v['baseline'],p['eligible_mask'])
        if c=='clean':np.testing.assert_array_equal(v[selected],v['baseline'])
        loaded[key]=v
    common.ALLOWED.add('J03WQQ');pm=official_metrics();y=join_halves([truth('J03WQQ',h) for h in ['firstHalf','secondHalf']]);conditions={c:{a:score_full(pm,y,join_halves([loaded[f'J03WQQ-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ['baseline',selected]} for c in ['clean','corrupted']};overall={a:float(np.mean([conditions[c][a]['spell_f1_mean'] for c in ['clean','corrupted']])) for a in ['baseline',selected]};assert abs(overall['baseline']-.7708462421456541)<1e-12;result={'overall':overall,'conditions':conditions,'selected':selected,'improved':overall[selected]>overall['baseline'],'clean_exact':True,'D_X_outside_exact':True,'prediction_seal':pin(RUN/'full-seal.json'),'reference_reads':common.READS.copy(),'no_fallback':True,'interpretation':'One reused full development match; no private competition claim'};write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'conditions':{c:{a:s['spell_f1_mean'] for a,s in v.items()} for c,v in conditions.items()},'improved':result['improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--fold',choices=['0','1']);args=p.parse_args();common.STAGE=args.stage;threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(common.guard)
    if args.stage in ['fit','calibrate']:
        assert args.fold is not None;globals()[args.stage](args.fold)
    else:globals()[args.stage]()
