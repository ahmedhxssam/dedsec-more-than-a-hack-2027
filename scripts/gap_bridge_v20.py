"""Raw-only phase-augmented gap memory; nested full-match calibration."""
import argparse,datetime,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import nested_context_calibration_v18 as nested
from second_zero import gap_bridge_training as bridge
from second_zero.gap_change_evidence import endpoint_credit
import numpy as np
from scipy.special import expit,logit
r=nested.r;d=r.d;gs=r.gs;api=bridge
BASE_RUN=nested.RUN;RAW_RUN=nested.BASE_RUN
RUN=ROOT/'runs/gap-bridge-v20';REPORT=ROOT/'reports/gap-bridge-v20'
FITS={**{m:[v for v in r.TRAIN if v!=m] for m in r.TRAIN},'full':r.TRAIN,**nested.FITS}
PARAMS=[{'owner_alpha':0.,'beta':0.,'gap_alpha':0.,'observed_alpha':0.,'temperature':1.}]+[{'owner_alpha':a,'beta':b,'gap_alpha':max(a,b),'observed_alpha':0.,'temperature':t} for a in [0.,.25,.5,.75,1.] for b in [0.,.025,.05] for t in [1.,2.] if a or b if a or t==1.]
ORIGINAL_GUARD=r.guard


def guard(event,args):
    ORIGINAL_GUARD(event,args)
    if event=='open' and isinstance(args[0],(str,bytes,Path)):
        p=Path(args[0])
        if str(RUN/'training') in str(p) and p.suffix=='.npz' and not r.HASH_ONLY:
            if p.parent.name not in r.ALLOWED:raise PermissionError('Bridge training cache outside current fit')


def cache():
    r.verify();rows={}
    for m in r.TRAIN:
        r.ALLOWED.clear();r.ALLOWED.add(m)
        for h in r.HALVES:
            clean=r.b.raw(m,h,'clean');truth=r.prior.truth(m,h)
            for phase,seed in [(0,731),(275,991)]:
                raw=bridge.phase_corruption(clean,phase,seed);x,prior,support=bridge.inputs(raw)
                target=r.model_api.targets(truth,truth,support,np.zeros(len(truth),np.float32));target['weight']*=.25+.75*(support[::5]>0)
                dest=RUN/f'training/{m}/{h}-phase{phase}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:np.savez_compressed(f,x=x.astype(np.float16),prior=prior,**target)
                rows[f'{m}-{h}-phase{phase}']={'cache':r.pin(dest),'match':m,'half':h,'view':f'phase{phase}','producer':'raw-only','primary_matches':[],'shape':list(x.shape),'raw_only_no_learned_features':True,'no_clean_teacher':True};print('V20 CACHE',m,h,phase,flush=True)
    old=r.read(RAW_RUN/'training-seal.json');rows.update(old['rows']);r.ALLOWED.clear();r.write(RUN/'training-seal.json',{'rows':rows,'reference_reads':r.READS.copy(),'protocol':r.pin(RUN/'protocol.json')})


def fit(which):
    r.verify();matches=FITS[which];r.ALLOWED.clear();r.ALLOWED.update(matches);seal=r.read(RUN/'training-seal.json');data=[];rows=[]
    if which=='full' and (RUN/'models/full/freeze.json').exists():
        f=r.read(RUN/'models/full/freeze.json');r.checked(f['model']);assert f['residual_fit_matches']==r.TRAIN;r.ALLOWED.clear();return
    for row in seal['rows'].values():
        if row['match'] not in matches:continue
        assert not row['primary_matches']
        with np.load(r.checked(row['cache'])) as z:data.append({k:z[k].copy() for k in z.files})
        rows.append(row)
    r.torch.manual_seed(20261030);rng=np.random.default_rng(20261030);model=bridge.GapBridge();opt=r.torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);steps=2000*len(matches);history=[];start=time.monotonic()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:g['lr']=.001*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        v=bridge.batch(data,rng);opt.zero_grad(set_to_none=True);loss=bridge.loss(model,v)
        if not r.torch.isfinite(loss):raise ValueError('Nonfinite bridge loss')
        loss.backward();r.torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step%400==0 or step==steps-1:
            value={'step':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-start};history.append(value);print('V20 TRAIN',which,value,flush=True)
    folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False);r.torch.save(model.state_dict(),folder/'model.pt');r.write(folder/'freeze.json',{'residual_fit_matches':matches,'held':None if which=='full' else which,'steps':steps,'rows':rows,'raw_only':True,'learned_feature_ancestors':[],'model':r.pin(folder/'model.pt'),'history':history,'reference_reads':r.READS.copy(),'training_seal':r.pin(RUN/'training-seal.json'),'protocol':r.pin(RUN/'protocol.json')});r.ALLOWED.clear()


def infer(which,producer,match,half):
    v,meta=nested.context(producer,match,half);x,prior,support=bridge.inputs(r.b.raw_half(f'{match}-{half}-corrupted'));model,freeze=r.fitted(which);student,boundary=bridge.predict_both(model,x,prior,len(v['baseline']));signals=r.local.signals(producer,match,half)
    dest=RUN/f'owner/{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,student=student,boundary=boundary,support=support)
    return v,signals,(student,boundary),support,producer,{'owner':r.pin(dest),'model':freeze['model'],'source':meta['cache'],'fit_matches':freeze['residual_fit_matches'],'primary_label_matches':r.prior.producer_proof(producer)['primary_label_matches']}


def compose(v,signals,student,support,cfg,producer):
    a,b,t=cfg['owner_alpha'],cfg['beta'],cfg['temperature']
    if not(a or b):return v['baseline'].copy()
    student,boundary=student;weight=a*support;z=logit(np.clip((1-weight)*expit(v['unary'])+weight*expit(student/t),1e-7,1-1e-7)).astype(np.float32)
    take=v['gap_arc_mask'];arcs=v['arcs'][take];x=signals.copy();x[:,0]=z;qo=r.b.old_quality(r.b.original_quality(producer),x,arcs);alpha=r.read(r.d.e.RUN/f'models/{producer}/calibration.json')['alpha'];q=(1-alpha)*qo+alpha*v['credit'][take]
    q+=endpoint_credit(arcs,v['baseline'],support,boundary,b,1.)
    out=d.fast.Graph(z,v['baseline'],arcs).decode(q,.4,.01);gs.check_lock(out,v['baseline'],v['gap_frame_mask']);return out


def prepare():
    p=r.read(BASE_RUN/'protocol.json');af=ROOT/'reports/nested-context-calibration-v18/artifact-freeze.json';inputs={**p['inputs'],**p['sources'],r.rel(af):r.sha256(af),**r.read(af)['hashes']}
    p={**p,'kind':'phase-augmented-bidirectional-gap-bridge-v20','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parameter_grid':PARAMS,
       'architecture':{'raw_channels':414,'projected':64,'bidirectional_GRU_hidden':64,'layers':2,'dropout':.1,'native_stride':5,'training_window':256,'scored_center':128,'inference':'same256-sample windows, zero-padded half edges, central128 only; two-perspective antisymmetric owner and symmetric change'},
       'training':'All other3 physical TRAIN4 for outer models,4 full,2 complementary inner fits. Actual raw100/271 plus new independently shifted holes at phase0/275 each375frames; lengths25..200, smooth.5m noise and100frame player holes, seed731/991. Phase independent of labels, no clean coordinates survive hidden gaps in input. Same raw414 API. No learned ancestor, ID/time/event field, clean teacher or WPY gradient. No augmentation of WQQ at inference.',
       'fit':{'steps_per_match':2000,'batch':8,'length':256,'central_loss':128,'optimizer':'AdamW','lr':.001,'wd':.01,'schedule':'cosine1to.1','seed':20261030,'final_checkpoint_only':True,'cpu_threads':2},
       'loss':'Duration/gap-weighted GT owner BCE +.1 Gaussian H/A change BCE posweight10 +.001 absolute owner logit square. Only central128 samples contribute to either task.',
       'composition':'Gap-only probability blend with raw student temperature1/2, recompute175 segment quality, canonical747 unchanged; beta0/.025/.05 change-logit-minus1 start credit; graph .4/.01 fixed. At allzero exact .770846 primary baseline. Clean, alloutsidegap, D/X identical. Preserved research best .773631 independent.',
       'calibration':'Same excluded nested2→outer3 policy as v18. Each outer h uses paired other game plus WPY, both primary2 and raw2 exclude h and calibrationgame. Full mean over4 cross-fitted games+full WPY. Pooled actual full-match five-IoU SpellF1; weights frozen and allcal candidate outputs sealed before references. One full candidate, no WQQ parameter search/fallback.',
       'limits':'Four training games, repeated development; raw2 versus full4 calibration model size shift remains. No private challenge claim, no WR9/challenge or CSV/export.',
       'sources':{**p['sources'],r.rel(Path(__file__)):r.sha256(Path(__file__)),r.rel(Path(bridge.__file__)):r.sha256(Path(bridge.__file__)),r.rel(Path(endpoint_credit.__code__.co_filename)):r.sha256(Path(endpoint_credit.__code__.co_filename))},'inputs':inputs}
    r.write(RUN/'protocol.json',p);print('V20 PROTOCOL',r.sha256(RUN/'protocol.json'),flush=True)


nested.RUN=RUN;nested.REPORT=REPORT;nested.PARAMS=PARAMS;nested.infer=infer;nested.compose=compose
r.RUN=RUN;r.REPORT=REPORT;r.fit=fit;r.guard=guard;r.model_api.TemporalResidual=bridge.GapBridge


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','fit','cache_cal','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=list(FITS));args=p.parse_args();r.threadpool_limits(2);r.torch.set_num_threads(2);r.STAGE=args.stage
    if args.stage=='prepare':prepare()
    else:
        sys.addaudithook(r.guard)
        if args.stage=='cache':cache()
        elif args.stage=='fit':assert args.held;fit(args.held)
        elif args.stage=='cache_cal':nested.cache_cal()
        elif args.stage=='calibrate':assert args.held;nested.calibrate(args.held)
        elif args.stage.startswith('audit_'):r.audit(args.stage=='audit_full')
        else:getattr(r,args.stage)()
