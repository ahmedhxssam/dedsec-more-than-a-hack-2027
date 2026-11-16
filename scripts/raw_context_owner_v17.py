"""Three-match outer fits of a truly raw-only temporal ownership student."""
import argparse,datetime,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import masked_gap_context_v16 as composition
from second_zero import raw_context_owner as api
r=composition.r;BASE_RUN=composition.BASE_RUN
RUN=ROOT/'runs/raw-context-owner-v17';REPORT=ROOT/'reports/raw-context-owner-v17';VIEWS=['100','271'];ORIGINAL_GUARD=r.guard


def split(held):
    fit=api.fitting_matches(r.TRAIN,held)
    return {'held':held,'selector_fit_matches':fit,'producer':'raw-only','producer_primary_matches':[],'all_primary_label_ancestors':fit,'same_actual_producer_for_fit_and_held':'Identical deterministic raw feature API; no learned feature weights.'}


def guard(event,args):
    ORIGINAL_GUARD(event,args)
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if str(RUN/'training') in p and p.endswith('.npz') and not r.HASH_ONLY:
            if Path(p).parent.name not in r.ALLOWED:raise PermissionError('Raw training cache outside current fit')


def cache():
    r.verify();rows={}
    for m in r.TRAIN:
        r.ALLOWED.clear();r.ALLOWED.add(m)
        for h in r.HALVES:
            truth=r.prior.truth(m,h)
            for view in VIEWS:
                start=time.monotonic();raw=r.b.raw(m,h,view);x,prior,support=api.inputs(raw);target=r.model_api.targets(truth,truth,support,np.zeros(len(truth),np.float32));target['weight']*=.25+.75*(support[::5]>0)
                dest=RUN/f'training/{m}/{h}-{view}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:np.savez_compressed(f,x=x.astype(np.float16),prior=prior,**target)
                rows[f'{m}-{h}-{view}']={'cache':r.pin(dest),'match':m,'half':h,'view':view,'producer':'raw-only','primary_matches':[],'shape':list(x.shape),'raw_only_no_learned_features':True,'no_clean_teacher':True};print('V17 CACHE',m,h,view,x.shape,'seconds',time.monotonic()-start,flush=True)
    r.ALLOWED.clear();r.write(RUN/'training-seal.json',{'rows':rows,'reference_reads':r.READS.copy(),'protocol':r.pin(RUN/'protocol.json')})


def fit(which):
    r.verify();matches=r.TRAIN if which=='full' else split(which)['selector_fit_matches'];r.ALLOWED.clear();r.ALLOWED.update(matches);seal=r.read(RUN/'training-seal.json');data=[];rows=[]
    for row in seal['rows'].values():
        if row['match'] not in matches:continue
        assert row['primary_matches']==[] and row['raw_only_no_learned_features']
        with np.load(r.checked(row['cache'])) as z:data.append({k:z[k].copy() for k in z.files})
        rows.append(row)
    r.torch.manual_seed(20261030);rng=np.random.default_rng(20261030);model=api.RawContextTemporal();optimizer=r.torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);steps=2000*len(matches);history=[];start=time.monotonic()
    for step in range(steps):
        for g in optimizer.param_groups:g['lr']=.001*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        v=api.batch(data,rng);optimizer.zero_grad(set_to_none=True);loss=api.loss(model,v)
        if not r.torch.isfinite(loss):raise ValueError('Nonfinite raw fitting loss')
        loss.backward();r.torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%200==0 or step==steps-1:
            value={'step':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-start};history.append(value);print('V17 TRAIN',which,value,flush=True)
    folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False);r.torch.save(model.state_dict(),folder/'model.pt');r.write(folder/'freeze.json',{'residual_fit_matches':matches,'held':None if which=='full' else which,'steps':steps,'rows':rows,'raw_only':True,'learned_feature_ancestors':[],'clean_teacher_targets_used':False,'model':r.pin(folder/'model.pt'),'history':history,'protocol':r.pin(RUN/'protocol.json'),'training_seal':r.pin(RUN/'training-seal.json'),'reference_reads':r.READS.copy(),'seconds':time.monotonic()-start});r.ALLOWED.clear()


def infer_context(which,match,half):
    producer='full' if which=='full' else r.prior.producer(match) if match!='J03WPY' else r.prior.producer(which);v,meta=r.d.load_case(producer,match,half);raw=r.b.raw_half(f'{match}-{half}-corrupted');x,prior,support=api.inputs(raw);signals=r.local.signals(producer,match,half);model,freeze=r.fitted(which);student=api.predict(model,x,prior,len(v['baseline']))
    oldcal=r.read(BASE_RUN/f'models/{which}/calibration.json');oldalpha=oldcal['selected']['alpha'];oldowner=None
    if oldalpha:
        oldpath=BASE_RUN/f'owner/{which}/{match}-{half}.npz'
        with np.load(oldpath) as z:oldstudent=z['student']
        oldowner=r.pin(oldpath);oldlogit=r.local.local.blend(v['unary'],oldstudent,support,oldalpha)
    else:oldstudent=v['unary'];oldlogit=v['unary']
    base=r.local.predict(v,signals,oldstudent,support,oldalpha,producer) if match=='J03WPY' else composition.parent(producer,match,half)
    composed={**v,'baseline':base,'unary':oldlogit}
    dest=RUN/f'owner/{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,student=student,support=support)
    return composed,signals,student,support,producer,{'owner':r.pin(dest),'source':meta['cache'],'model':freeze['model'],'parent_owner':oldowner,'parent_calibration':r.pin(BASE_RUN/f'models/{which}/calibration.json'),'raw_student_no_learned_features':True}


def owner_case(which,match,half,alphas):
    v,signals,student,support,producer,meta=infer_context(which,match,half);temperature=r.read(RUN/f'models/{which}/calibration.json')['selected']['temperature'];out={'baseline':v['baseline']}
    for i,alpha in enumerate(alphas):out[f'alpha_{i}']=r.local.predict(v,signals,student/temperature,support,alpha,producer)
    return out,{**meta,'alphas':alphas,'temperature':temperature}


def calibrate(which):
    r.verify();r.ALLOWED.clear();start=len(r.READS);cases=[infer_context(which,'J03WPY',h) for h in r.HALVES];assert len(r.READS)==start;r.ALLOWED.add('J03WPY');y=r.prior.join_halves([r.prior.truth('J03WPY',h) for h in r.HALVES]);pm=r.prior.official_metrics();rows=[]
    grid=[(0.,1.)]+[(a,t) for t in [.5,1.,2.] for a in [.1,.25,.5,.75,1.]]
    for alpha,temperature in grid:
        predictions=[r.local.predict(v,x,z/temperature,support,alpha,producer) for v,x,z,support,producer,_ in cases];out=r.prior.join_halves(predictions);score=r.d.spell_score(pm,y,out);changed=sum(int((p!=c[0]['baseline']).sum()) for p,c in zip(predictions,cases));rows.append({'alpha':alpha,'temperature':temperature,'score':score,'changed':changed})
    selected=max(enumerate(rows),key=lambda v:(v[1]['score']['spell_f1_mean'],-v[1]['changed'],-v[0]))[1];r.write(RUN/f'models/{which}/calibration.json',{'selected':selected,'rows':rows,'inputs':[c[-1] for c in cases],'calibration_matches':['J03WPY'],'reference_reads':r.READS.copy(),'protocol':r.pin(RUN/'protocol.json')});r.ALLOWED.clear();print('V17 CALIBRATED',which,selected,flush=True)


r.RUN=RUN;r.REPORT=REPORT;r.guard=guard;r.cache=cache;r.fit=fit;r.calibrate=calibrate;r.owner_case=owner_case;r.prior.split=split
r.model_api.TemporalResidual=api.RawContextTemporal;r.model_api.batch=api.batch;r.model_api.loss=api.loss;r.model_api.predict=api.predict


def prepare():
    p=r.read(BASE_RUN/'protocol.json');af=ROOT/'reports/temporal-context-recovery-v14/artifact-freeze.json';inputs={**p['inputs'],**p['sources'],r.rel(af):r.sha256(af)};inputs.update(r.read(af)['hashes'])
    p={**p,'kind':'raw-only-three-match-temporal-v17','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_research_best':.773631491936103,
       'architecture':{**p['architecture'],'channels':414,'raw_channels':412,'missing_channels':2,'absolute_owner':True},
       'features':'414 truly raw-only:290 observed player motion/shape,98 observed ball/player geometry+missing flags,24 bounded linear ball interpolation geometry,raw ball missing andmissing25 support. No frozen logit, decoded label, status, time/match ID or learned feature enters the student. Both team perspectives, native5 average, matching FP16 quantization in train/inference.',
       'supervision':'GT owner BCE with inverse-sqrt spell-duration weights and extra missing25 mass; outside-gap weights further multiplied.25. Auxiliary true control-change BCE .05 withpositive10, absolute-logit square penalty.001. ZERO clean-teacher/distillation term. No learned target or feature ancestry.',
       'fit':{**p['fit'],'steps_per_match':2000,'outer_steps':6000,'full_steps':8000,'outer_matches':3,'full_matches':4,'views':VIEWS},
       'ancestry':[split(h) for h in r.TRAIN],
       'calibration':'Each outer student fits ALL other3 TRAIN4 matches, only raw corrupted100/271 views. Full fits4. WPY-only alpha[0,.1,.25,.5,.75,1] andtemperature[.5,1,2] calibration on official pooled SpellF1; alpha0 baseline once. Model weights frozen first, no checkpoint sweep. More corruption views do not create independent matches. No WPY gradient fitting.',
       'composition':'Compose on frozen v14 .773631492 as in v16, but the student is raw-only and trains on3 outer matches instead of1. Old v14 parent for each held match excludes held across all ancestors. Alpha0 copies the exact v14 labels. Only one selected fullalpha is scored. Legacybaseline_07714 field denotes v14 composed parent, not its old numeric name.',
       'limits':'Primary student size shift3→4 instead of1→4; frozen composition parent still hasTRAIN2→4 feature ancestry andadaptive WPY/WQQ exposure. This is reused development, not independent/private challenge accuracy. All clean/outside-gap/DX exact and all previous candidates intact.',
       'sources':{**p['sources'],r.rel(Path(__file__)):r.sha256(Path(__file__)),r.rel(Path(api.__file__)):r.sha256(Path(api.__file__)),r.rel(Path(api.context.__file__)):r.sha256(Path(api.context.__file__)),r.rel(Path(api.trajectory_features.__code__.co_filename)):r.sha256(Path(api.trajectory_features.__code__.co_filename)),r.rel(Path(api.interpolated_features.__code__.co_filename)):r.sha256(Path(api.interpolated_features.__code__.co_filename)),r.rel(Path(composition.__file__)):r.sha256(Path(composition.__file__)),r.rel(Path(composition.masked.__file__)):r.sha256(Path(composition.masked.__file__))},'inputs':inputs}
    r.write(RUN/'protocol.json',p);print('V17 PROTOCOL',r.sha256(RUN/'protocol.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=r.TRAIN+['full']);args=p.parse_args();r.threadpool_limits(2);r.torch.set_num_threads(2);r.STAGE=args.stage
    if args.stage=='prepare':prepare()
    else:
        sys.addaudithook(r.guard)
        if args.stage in ['fit','calibrate']:assert args.held;getattr(r,args.stage)(args.held)
        elif args.stage.startswith('audit_'):r.audit(args.stage=='audit_full')
        else:getattr(r,args.stage)()
