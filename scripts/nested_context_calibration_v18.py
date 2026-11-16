"""Nested multi-match regional calibration, reusing raw17 body weights."""
import argparse,datetime,itertools,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from scipy.ndimage import uniform_filter1d
from scripts import raw_context_owner_v17 as raw
r=raw.r;api=raw.api;d=r.d;gs=r.gs;BASE_RUN=raw.RUN
RUN=ROOT/'runs/nested-context-calibration-v18';REPORT=ROOT/'reports/nested-context-calibration-v18'
FITS={'inner0':r.TRAIN[2:],'inner1':r.TRAIN[:2]}
PARAMS=[{'gap_alpha':0.,'observed_alpha':0.,'temperature':1.}]+[{'gap_alpha':g,'observed_alpha':o,'temperature':t} for g,o,t in itertools.product([0.,.25,.5,.75,1.],[0.,.1,.25],[.5,1.,2.]) if g or o]


def primary_parent(which,match,half,condition='corrupted'):
    if condition=='clean':return r.prior.old.c.previous.incumbent(match,half,condition)
    return d.load_case(which,match,half)[0]['baseline_07708'].copy()


def context(which,match,half):
    v,meta=d.load_case(which,match,half);base=v['baseline_07708'];out={**v,'baseline':base,'gap_arc_mask':gs.feasible(v['arcs'],base,v['gap_frame_mask'])}
    return out,meta


def fit(which):
    r.verify()
    if which=='full':
        f=r.read(RUN/'models/full/freeze.json');r.checked(f['model']);assert f['residual_fit_matches']==r.TRAIN;return
    assert which in FITS;matches=FITS[which];r.ALLOWED.clear();r.ALLOWED.update(matches);seal=r.read(BASE_RUN/'training-seal.json');data=[];rows=[]
    for row in seal['rows'].values():
        if row['match'] not in matches:continue
        assert row['primary_matches']==[]
        with np.load(r.checked(row['cache'])) as z:data.append({k:z[k].copy() for k in z.files})
        rows.append(row)
    r.torch.manual_seed(20261030);rng=np.random.default_rng(20261030);model=api.RawContextTemporal();opt=r.torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);steps=2000*len(matches);start=time.monotonic();history=[]
    for step in range(steps):
        for g in opt.param_groups:g['lr']=.001*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        v=api.batch(data,rng);opt.zero_grad(set_to_none=True);loss=api.loss(model,v)
        if not r.torch.isfinite(loss):raise ValueError('Nonfinite inner loss')
        loss.backward();r.torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step%400==0 or step==steps-1:
            value={'step':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-start};history.append(value);print('V18 INNER',which,value,flush=True)
    folder=RUN/f'models/{which}';folder.mkdir(parents=True,exist_ok=False);r.torch.save(model.state_dict(),folder/'model.pt');r.write(folder/'freeze.json',{'residual_fit_matches':matches,'raw_only':True,'steps':steps,'rows':rows,'model':r.pin(folder/'model.pt'),'history':history,'reference_reads':r.READS.copy(),'protocol':r.pin(RUN/'protocol.json')});r.ALLOWED.clear()


def infer(which,producer,match,half):
    v,meta=context(producer,match,half);signals=r.local.signals(producer,match,half);support=uniform_filter1d((~np.isfinite(r.b.raw_half(f'{match}-{half}-corrupted')['ball']).all(1)).astype(np.float64),25,mode='nearest');freeze=r.read(RUN/f'models/{which}/freeze.json')
    cached=BASE_RUN/f'owner/{which}/{match}-{half}.npz'
    if which in r.TRAIN+['full'] and cached.exists():
        with np.load(cached) as z:student=z['student'].copy()
        receipt={'reused_owner':r.pin(cached),'model':freeze['model']}
    else:
        model,_=r.fitted(which);x,prior,support=api.inputs(r.b.raw_half(f'{match}-{half}-corrupted'));student=api.predict(model,x,prior,len(v['baseline']));dest=RUN/f'owner/{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:np.savez_compressed(f,student=student,support=support)
        receipt={'owner':r.pin(dest),'model':freeze['model']}
    return v,signals,student,support,producer,{**receipt,'source':meta['cache'],'fit_matches':freeze['residual_fit_matches'],'primary_label_matches':r.prior.producer_proof(producer)['primary_label_matches']}


def compose(v,signals,student,support,cfg,producer):
    g,o,t=cfg['gap_alpha'],cfg['observed_alpha'],cfg['temperature']
    if not(g or o):return v['baseline'].copy()
    weight=g*support+o*(1-support);z=logit(np.clip((1-weight)*expit(v['unary'])+weight*expit(student/t),1e-7,1-1e-7)).astype(np.float32);x=signals.copy();x[:,0]=z;take=np.ones(len(v['arcs']),bool) if o else v['gap_arc_mask'];arcs=v['arcs'][take];qo=r.b.old_quality(r.b.original_quality(producer),x,arcs);a=r.read(r.d.e.RUN/f'models/{producer}/calibration.json')['alpha'];q=(1-a)*qo+a*v['credit'][take]
    out=d.fast.Graph(z,v['baseline'],arcs).decode(q,.4,.01);gs.check_lock(out,v['baseline'],v['baseline']<2 if o else v['gap_frame_mask']);return out


def make_cal_case(which,producer,match,half):
    v,x,z,s,p,meta=infer(which,producer,match,half);values={'baseline':v['baseline']}
    for i,cfg in enumerate(PARAMS):values[f'param_{i}']=compose(v,x,z,s,cfg,p)
    dest=RUN/f'calibration-cases/{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,**values)
    print('V18 CAL CASE',which,match,half,len(PARAMS),flush=True);return {'cache':r.pin(dest),'student':meta,'producer':producer,'match':match,'half':half}


def cache_cal():
    r.verify();r.ALLOWED.clear();rows={}
    for m in r.TRAIN:
        p=r.prior.producer(m)
        for h in r.HALVES:rows[f'{m}-{h}']=make_cal_case('inner'+p,p,m,h)
    for p in ['0','1','full']:
        for h in r.HALVES:rows[f'WPY-{p}-{h}']=make_cal_case('full' if p=='full' else 'inner'+p,p,'J03WPY',h)
    assert not r.READS;r.write(RUN/'calibration-case-seal.json',{'rows':rows,'reference_reads':[],'protocol':r.pin(RUN/'protocol.json')})


def calibrate(which):
    r.verify();seal=r.read(RUN/'calibration-case-seal.json')
    if which=='full':cohorts=[(m,r.prior.producer(m)) for m in r.TRAIN]+[('J03WPY','full')]
    else:
        p=r.prior.producer(which);group=r.TRAIN[:2] if p=='0' else r.TRAIN[2:];mate=next(m for m in group if m!=which);cohorts=[(mate,p),('J03WPY',p)]
    r.ALLOWED.clear();r.ALLOWED.update(m for m,_ in cohorts);pm=r.prior.official_metrics();cases=[]
    for m,p in cohorts:
        halves=[];metas=[]
        for h in r.HALVES:
            key=f'WPY-{p}-{h}' if m=='J03WPY' else f'{m}-{h}';meta=seal['rows'][key]
            if which!='full':
                assert which not in meta['student']['fit_matches'] and m not in meta['student']['fit_matches']
                assert which not in meta['student']['primary_label_matches'] and m not in meta['student']['primary_label_matches']
            with np.load(r.checked(meta['cache'])) as z:halves.append({k:z[k] for k in z.files})
            metas.append(meta)
        y=r.prior.join_halves([r.prior.truth(m,h) for h in r.HALVES]);cases.append((m,y,halves,metas))
    rows=[]
    for i,cfg in enumerate(PARAMS):
        scores={};changed=0
        for m,y,halves,_ in cases:
            out=r.prior.join_halves([v[f'param_{i}'] for v in halves]);scores[m]=d.spell_score(pm,y,out);changed+=sum(int((v[f'param_{i}']!=v['baseline']).sum()) for v in halves)
        rows.append({**cfg,'alpha':max(cfg['gap_alpha'],cfg['observed_alpha']),'mean_spell_f1':float(np.mean([v['spell_f1_mean'] for v in scores.values()])),'per_match':scores,'changed':changed})
    selected=max(enumerate(rows),key=lambda v:(v[1]['mean_spell_f1'],-v[1]['changed'],-v[0]))[1];r.write(RUN/f'models/{which}/calibration.json',{'selected':selected,'rows':rows,'calibration_matches':[m for m,_ in cohorts],'nested_held_excluded':None if which=='full' else which,'inputs':[meta['cache'] for c in cases for meta in c[3]],'reference_reads':r.READS.copy(),'protocol':r.pin(RUN/'protocol.json')});r.ALLOWED.clear();print('V18 CALIBRATED',which,selected,flush=True)


def owner_case(which,match,half,alphas):
    p='full' if which=='full' else r.prior.producer(match);v,x,z,s,p,meta=infer(which,p,match,half);cfg=r.read(RUN/f'models/{which}/calibration.json')['selected'];out={'baseline':v['baseline']}
    for i,_ in enumerate(alphas):out[f'alpha_{i}']=compose(v,x,z,s,cfg,p)
    return out,{**meta,'config':cfg,'activity_alpha_is_legacy_dispatch_only':True}


r.RUN=RUN;r.REPORT=REPORT;r.fit=fit;r.calibrate=calibrate;r.owner_case=owner_case;r.d.boundary_parent=primary_parent


def prepare():
    p=r.read(BASE_RUN/'protocol.json');af=ROOT/'reports/raw-context-owner-v17/artifact-freeze.json';inputs={**p['inputs'],**p['sources'],r.rel(af):r.sha256(af)};inputs.update(r.read(af)['hashes'])
    p={**p,'kind':'nested-multi-match-regional-calibration-v18','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_research_best':.773631491936103,
       'weights':'Reuse exact frozen raw17 outer3/full4 student weights. Fit TWO new inner raw-only students on complementary TRAIN2 groups, same4000steps/architecture/objective/views. No WQQ or WPY gradient fitting.',
       'calibration':'Outer held h: its paired other match plus WPY, each from inner raw2 and complete primary2 excluding BOTH h and paired calibration match. Apply chosen parameters to raw3/primary2 held output. Full: equal-physical-match mean over4 cross-fitted raw2/primary2 matches plus raw4/primary4 WPY. Each match pools both halves before official SpellF1. No same-game half validation or WQQ parameter search.',
       'parameter_grid':PARAMS,'composition':'Regional probability blend by gapalpha*missing25+observedalpha*(1-missing25), temperature on raw student only; recompute175 quality, canonical747 credit unchanged,.4/.01 fixed. Observedalpha0 retains oldgap locks; nonzero permits all corrupted live frames using original graph/gap pool. All D/X and clean exact. Exact primary-only .770846 baseline at zero; no neural14 parent whose inner calibration ancestry could include outerheld. Legacy baseline_07714 array field denotes actual primary .770846, explicitly.',
       'dispatch':'Selected alpha field is max(gapalpha,observedalpha) only to activate inherited runner; actual regional params/temperature live in selected config. One frozen full composition, no fallback. Compare against preserved .773631 research best.',
       'limits':'Inner2→outer3/full4 student size shift remains, full calibration mixes4 primary2 views and full WPY; five physical calibration matches, reused adaptively. Strict current outer exclusion, no private/untouched challenge claim.',
       'sources':{**p['sources'],r.rel(Path(__file__)):r.sha256(Path(__file__))},'inputs':inputs}
    r.write(RUN/'protocol.json',p)
    for name in r.TRAIN+['full']:
        old=BASE_RUN/f'models/{name}/freeze.json';freeze={**r.read(old),'reused_fit':r.pin(old),'policy_protocol':r.pin(RUN/'protocol.json')};r.write(RUN/f'models/{name}/freeze.json',freeze)
    print('V18 PROTOCOL',r.sha256(RUN/'protocol.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit_inner','cache_cal','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=r.TRAIN+['full']);p.add_argument('--inner',choices=list(FITS));args=p.parse_args();r.threadpool_limits(2);r.torch.set_num_threads(2);r.STAGE=args.stage
    if args.stage=='prepare':prepare()
    else:
        sys.addaudithook(r.guard)
        if args.stage=='fit_inner':assert args.inner;fit(args.inner)
        elif args.stage=='cache_cal':cache_cal()
        elif args.stage=='calibrate':assert args.held;calibrate(args.held)
        elif args.stage.startswith('audit_'):r.audit(args.stage=='audit_full')
        else:getattr(r,args.stage)()
