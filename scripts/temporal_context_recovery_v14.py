"""Temporal/event residual trained on exact excluded complete parents."""
from __future__ import annotations
import argparse,datetime,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from second_zero import temporal_gap_residual as model_api
from scripts import local_context_recovery_v13 as local
d=local.d;prior=d.prior;b=d.b;gs=d.gs
read=d.read;write=d.write;pin=d.pin;rel=d.rel;sha256=d.sha256
RUN=ROOT/'runs/temporal-context-recovery-v14';REPORT=ROOT/'reports/temporal-context-recovery-v14';TRAIN=d.TRAIN;HALVES=d.HALVES
ALPHAS=[0.,.1,.25,.5,.75,1.];STEPS_PER_MATCH=400;ALLOWED=set();FULL=False;HASH_ONLY=False;STAGE='prepare';READS=[]


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(s in p for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved assets closed')
    if 'J03WQQ' in p and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed')
    if p.endswith('.truth.npy'):
        if Path(p).name.split('-')[0] not in ALLOWED:raise PermissionError('Primary labels outside declared role')
        READS.append({'stage':STAGE,'path':rel(p),'allowed':sorted(ALLOWED)})
    if '/temporal-context-recovery-v14/training/' in p and p.endswith('.npz') and not HASH_ONLY:
        if Path(p).parent.name not in ALLOWED:raise PermissionError('Training cache outside current fit')


def checked(value):
    global HASH_ONLY
    saved=HASH_ONLY;HASH_ONLY=True
    try:return d.checked(value)
    finally:HASH_ONLY=saved


def prepare():
    p=read(d.RUN/'protocol.json');inputs={**p['sources'],**p['inputs']};af=ROOT/'reports/metric-decoder-v11/artifact-freeze.json';inputs[rel(af)]=sha256(af);inputs.update(read(af)['hashes'])
    sources=[Path(__file__),Path(model_api.__file__),Path(local.__file__),Path(local.local.__file__)]
    write(RUN/'protocol.json',{'kind':'temporal-full-parent-gap-residual-v14','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_best':.7713875141348288,
      'architecture':{'channels':307,'width':32,'dilations':[1,2,4,8,16,32],'kernel':3,'stride_native':5,'context_native_frames':635,'dropout':.1,'zero_initial_owner_residual':True,'symmetric_two_perspectives':True},
      'features':'Raw player/team/pitch290 +actual complete final13 +parent H/A onehots/missing/support4. Native5-frame smooth/downsample. Both A andH perspectives stored float16, dequantized float32 for BOTH fitting and inference. No clean-input teacher features at inference.',
      'supervision':'GT H/A owner BCE weighted sqrt125/duration clipped[.25,4] and1+2*missing25, .25 clean-teacher soft BCE at temperature2, .05 auxiliary H/A-change BCE with10positiveweight, .001 residual squared penalty. Clean teacher only as fitting target from SAME excluded frozen primary producer, never feature. Window centers half GT H/A-change events/half live native samples. Random team swap; symmetric inference.',
      'fit':{'steps_per_match':STEPS_PER_MATCH,'batch':8,'length_downsampled':128,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.01,'grad_clip':5.,'device':'cpu','threads':2,'seed':20261030,'checkpoint':'fixed final step, no validation selection','schedule':'cosine from1 to.1'},
      'device_profile':'Random-tensor30-step profile: CPU2 threads.00564sec/step vs local MPS.02423sec/step. Choose CPU for this small architecture, not based on scores.',
      'ancestry':[prior.split(m) for m in TRAIN],
      'calibration':'Each crossed model fits the paired other excluded match; same complete TRAIN2 feature producer excludes both. Full model fits all4 cross-fitted matches. WPY selects ONLY alpha[0,.1,.25,.5,.75,1] by pooled five-IoU SpellF1, alpha0 ties. Model weights frozen before calibration.',
      'composition':'Same fixed gap-only composition as v13: probability blend alpha*missing25 into prior, recompute original175 segment quality; canonical747 credit unchanged, penalty.4/frame.01. Existing expanded pool and exact .771388 parent cover. All outside-gap labels, D/X and clean exact. Alpha0 bypass copies parent.',
      'evaluation':'Seal each OOF view before audit; one selected full composition after full WPY alpha frozen, no WQQ reference fitting/calibration/fallback. Report pooled clean/corrupted/equal mean SpellF1 and all physical validation matches. No .8 claim until measured.',
      'limits':'Four primary matches; one versus4 residual fitting matches and TRAIN2 versusTRAIN4 producer-size shift. Reused WPY/WQQ development, no independent/private challenge claim. No CSV/export or reserved WR9/challenge access.',
      'sources':{rel(p):sha256(p) for p in sources},'inputs':inputs})
    print('V14 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def input_case(which,match,half):
    v,meta=d.load_case(which,match,half);raw=b.raw_half(f'{match}-{half}-corrupted');signals=local.signals(which,match,half);np.testing.assert_array_equal(signals[:,0],v['unary']);x,priorz,support=model_api.inputs(raw,signals,v['baseline']);x=x.astype(np.float16).astype(np.float32)
    return v,meta,signals,x,priorz,support


def cache():
    verify();rows={}
    for m in TRAIN:
        ALLOWED.clear();ALLOWED.add(m);which=prior.producer(m)
        for h in HALVES:
            start=time.monotonic();v,meta,signals,x,priorz,support=input_case(which,m,h);truth=prior.truth(m,h);clean=b.ceiling.current(m,h,'clean')['signals'][:,0];target=model_api.targets(truth,v['baseline'],support,clean);dest=RUN/f'training/{m}/{h}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:np.savez_compressed(f,x=x.astype(np.float16),prior=priorz,**target)
            proof=prior.producer_proof(which);assert m not in proof['primary_label_matches'];rows[f'{m}-{h}']={'cache':pin(dest),'match':m,'half':h,'producer':which,'primary_matches':proof['primary_label_matches'],'source':meta['cache'],'shape':list(x.shape),'clean_teacher_only_in_targets':True};print('V14 CACHE',m,h,x.shape,'seconds',time.monotonic()-start,flush=True)
    ALLOWED.clear();write(RUN/'training-seal.json',{'rows':rows,'reference_reads':READS.copy(),'protocol':pin(RUN/'protocol.json')})


def fit(which):
    verify();matches=TRAIN if which=='full' else prior.split(which)['selector_fit_matches'];ALLOWED.clear();ALLOWED.update(matches);seal=read(RUN/'training-seal.json');data=[];rows=[]
    for row in seal['rows'].values():
        if row['match'] not in matches:continue
        assert row['match'] not in row['primary_matches']
        if which!='full':assert which not in row['primary_matches'] and row['producer']==prior.producer(which)
        with np.load(checked(row['cache'])) as z:data.append({k:z[k].copy() for k in z.files})
        rows.append(row)
    torch.manual_seed(20261030);rng=np.random.default_rng(20261030);model=model_api.TemporalResidual();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);steps=STEPS_PER_MATCH*len(matches);history=[];start=time.monotonic();model.train()
    for step in range(steps):
        lr=.001*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        for g in optimizer.param_groups:g['lr']=lr
        v=model_api.batch(data,rng);optimizer.zero_grad(set_to_none=True);loss=model_api.loss(model,v)
        if not torch.isfinite(loss):raise ValueError('Nonfinite temporal fitting loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0 or step==steps-1:
            row={'step':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-start};history.append(row);print('V14 TRAIN',which,row,flush=True)
    folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False);torch.save(model.state_dict(),folder/'model.pt');write(folder/'freeze.json',{'residual_fit_matches':matches,'held':None if which=='full' else which,'steps':steps,'rows':rows,'model':pin(folder/'model.pt'),'history':history,'protocol':pin(RUN/'protocol.json'),'training_seal':pin(RUN/'training-seal.json'),'reference_reads':READS.copy(),'seconds':time.monotonic()-start});ALLOWED.clear()


def fitted(which):
    freeze=read(RUN/f'models/{which}/freeze.json');model=model_api.TemporalResidual();model.load_state_dict(torch.load(checked(freeze['model']),map_location='cpu',weights_only=True));return model,freeze


def owner_case(which,match,half,alphas):
    producer='full' if which=='full' else prior.producer(match) if match!='J03WPY' else prior.producer(which);v,meta,signals,x,priorz,support=input_case(producer,match,half);model,freeze=fitted(which);student=model_api.predict(model,x,priorz,len(v['baseline']));dest=RUN/f'owner/{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,student=student,support=support)
    out={'baseline':v['baseline']}
    for i,alpha in enumerate(alphas):out[f'alpha_{i}']=local.predict(v,signals,student,support,alpha,producer)
    result={'owner':pin(dest),'source':meta['cache'],'model':freeze['model'],'alphas':alphas};return out,result


def calibrate(which):
    verify();ALLOWED.clear();start=len(READS);cases=[];meta=[]
    for h in HALVES:
        v,m=owner_case(which,'J03WPY',h,ALPHAS);cases.append(v);meta.append(m)
    assert len(READS)==start;ALLOWED.add('J03WPY');y=prior.join_halves([prior.truth('J03WPY',h) for h in HALVES]);pm=prior.official_metrics();rows=[]
    for i,alpha in enumerate(ALPHAS):
        out=prior.join_halves([v[f'alpha_{i}'] for v in cases]);rows.append({'alpha':alpha,'score':d.spell_score(pm,y,out),'changed':sum(int((v[f'alpha_{i}']!=v['baseline']).sum()) for v in cases)})
    selected=max(enumerate(rows),key=lambda v:(v[1]['score']['spell_f1_mean'],-v[1]['changed'],-v[0]))[1];write(RUN/f'models/{which}/calibration.json',{'selected':selected,'rows':rows,'inputs':meta,'calibration_matches':['J03WPY'],'reference_reads':READS.copy(),'protocol':pin(RUN/'protocol.json')});ALLOWED.clear();print('V14 CALIBRATED',which,selected,flush=True)


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={}
    for m in TRAIN:
        alpha=read(RUN/f'models/{m}/calibration.json')['selected']['alpha']
        for h in HALVES:
            for c in ['clean','corrupted']:
                base=d.boundary_parent(prior.producer(m),m,h,c);meta={'alpha':alpha,'clean_bypass':c=='clean'}
                if c=='clean' or alpha==0:out=base.copy()
                else:v,cm=owner_case(m,m,h,[alpha]);out=v['alpha_0'];meta['owner']=cm
                key=f'{m}-{h}-{c}';dest=RUN/f'oof/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:np.savez_compressed(f,baseline_07714=base,temporal_context=out)
                outputs[rel(dest)]=sha256(dest);meta['changed']=int((out!=base).sum());views[key]=meta
    assert not READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'reference_reads':[],'protocol':pin(RUN/'protocol.json'),'models':{m:{'freeze':pin(RUN/f'models/{m}/freeze.json'),'calibration':pin(RUN/f'models/{m}/calibration.json')} for m in TRAIN}})


def audit(full=False):
    global FULL
    verify();FULL=full;name='full' if full else 'oof';dest=RUN/f'{name}-audit.json';assert not dest.exists();seal=read(RUN/f'{name}-seal.json');checked(seal['protocol']);assert not seal['reference_reads'];matches=['J03WQQ'] if full else TRAIN;arms=['baseline_07714','temporal_context'];loaded={}
    for path,h in seal['outputs'].items():
        assert sha256(ROOT/path)==h
        with np.load(ROOT/path) as z:v={k:z[k] for k in z.files}
        key=Path(path).stem
        for a in arms:
            gs.check_lock(v[a],v['baseline_07714'],v['baseline_07714']<2)
            if key.endswith('-clean'):np.testing.assert_array_equal(v[a],v['baseline_07714'])
        loaded[key]=v
    if not full:
        for held,meta in seal['models'].items():
            f=read(checked(meta['freeze']));checked(f['model']);checked(meta['calibration']);assert f['residual_fit_matches']==prior.split(held)['selector_fit_matches']
            for row in f['rows']:assert held not in row['primary_matches'] and row['match'] not in row['primary_matches'];checked(row['cache'])
    else:
        selection=read(checked(seal['selection']));f=read(checked(selection['model']));checked(f['model']);checked(selection['calibration']);assert f['residual_fit_matches']==TRAIN
    ALLOWED.clear();ALLOWED.update(matches);pm=prior.official_metrics();details={}
    for m in matches:
        y=prior.join_halves([prior.truth(m,h) for h in HALVES]);details[m]={c:{a:prior.score_full(pm,y,prior.join_halves([loaded[f'{m}-{h}-{c}'][a] for h in HALVES]))[0] for a in arms} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in arms};overall=avg(matches,['clean','corrupted']);conditions={c:avg(matches,[c]) for c in ['clean','corrupted']};pairs={} if full else {str(i):avg(TRAIN[:2] if i==0 else TRAIN[2:],['clean','corrupted']) for i in [0,1]};passed=overall['temporal_context']>overall['baseline_07714'] and all(v['temporal_context']>=v['baseline_07714'] for v in pairs.values());result={'overall':overall,'conditions':conditions,'per_match':{m:avg([m],['clean','corrupted']) for m in matches},'per_producer_pair':pairs,'details':details,'passed':passed,'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/f'{name}-seal.json'),'reference_reads':READS.copy(),'limitation':'One WPY-calibrated neural residual, adaptively reused development; no independent/private challenge claim.'};write(dest,result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_match','per_producer_pair','passed']}),flush=True)


def full_run():
    global FULL
    verify();a=read(RUN/'oof-audit.json');checked(a['prediction_seal']);fit('full');calibrate('full');cal=read(RUN/'models/full/calibration.json');write(RUN/'full-selection.json',{'model':pin(RUN/'models/full/freeze.json'),'calibration':pin(RUN/'models/full/calibration.json'),'OOF_audit':pin(RUN/'oof-audit.json'),'no_fallback':True});alpha=cal['selected']['alpha'];FULL=True;ALLOWED.clear();start=len(READS);outputs={};views={}
    for h in HALVES:
        for c in ['clean','corrupted']:
            base=d.boundary_parent('full','J03WQQ',h,c);meta={'alpha':alpha,'clean_bypass':c=='clean'}
            if c=='clean' or alpha==0:out=base.copy()
            else:v,cm=owner_case('full','J03WQQ',h,[alpha]);out=v['alpha_0'];meta['owner']=cm
            key=f'J03WQQ-{h}-{c}';dest=RUN/f'full-predictions/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:np.savez_compressed(f,baseline_07714=base,temporal_context=out)
            outputs[rel(dest)]=sha256(dest);meta['changed']=int((out!=base).sum());views[key]=meta;print('V14 FULL',key,meta,flush=True)
    assert not READS[start:];write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'reference_reads':[],'selection':pin(RUN/'full-selection.json'),'protocol':pin(RUN/'protocol.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=TRAIN+['full']);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);torch.set_num_threads(2);sys.addaudithook(guard)
    if args.stage in ['fit','calibrate']:assert args.held;globals()[args.stage](args.held)
    elif args.stage.startswith('audit_'):audit(args.stage=='audit_full')
    else:globals()[args.stage]()
