"""WPY full-match F1 calibration of frozen interval evidence; clean bypass."""
from __future__ import annotations
import argparse,datetime,itertools,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from threadpoolctl import threadpool_limits
from second_zero import fast_spell_graph as fast
from scripts import selected_edit_verifier_v10 as previous

prior=previous.prior;b=prior.b;e=prior.e;gs=prior.selector.gs
read=prior.read;write=prior.write;pin=prior.pin;rel=prior.rel;checked=prior.checked;sha256=prior.sha256
TRAIN=prior.TRAIN;HALVES=previous.HALVES;RUN=ROOT/'runs/metric-decoder-v11';REPORT=ROOT/'reports/metric-decoder-v11'
FULL=False;HASH_ONLY=False;ALLOWED=set();READS=[];STAGE='prepare'


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(v in p for v in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved assets closed')
    if 'J03WQQ' in p and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed')
    if p.endswith('.truth.npy'):
        if Path(p).name.split('-')[0] not in ALLOWED:raise PermissionError('Reference outside declared role')
        READS.append({'stage':STAGE,'path':rel(p),'allowed':sorted(ALLOWED)})


def prepare():
    p=read(previous.RUN/'protocol.json');inputs={**p['inputs'],**p['sources']}
    for folder in [ROOT/'reports/selected-edit-verifier-v10']:
        af=folder/'artifact-freeze.json';inputs[rel(af)]=sha256(af);inputs.update(read(af)['hashes'])
    fast.library()
    sources=[Path(__file__),Path(fast.__file__),fast.BUILD/'graph.c',fast.BUILD/'graph.dylib']
    write(RUN/'protocol.json',{'kind':'WPY-full-match-F1-decoder-calibration-v11','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'preserved_baselines':[.7708462421456541,.7713875141348288],
       'primary_weights':'All frozen complete actual primary, quality175 and747 credit weights; no new primary model fit. Strict TRAIN2 OOF and TRAIN4 full parents, already bound ancestry.',
       'calibration':'WPY ONLY, both halves pooled, official mean five-IoU SpellF1. Pick exactly one decoder setting per producer, baseline wins score ties. Explicit F1 calibration replaces prior MSE blend calibration. No WQQ references.',
       'grid':{'alpha':[0.,.1,.25,.5,1.,'existing_MSE_alpha'],'penalty':[.3,.4,.5,.6],'frame_weight':[0.,.001,.003,.01,.03],'scope':['gap','all_live']},
       'pool':'Original graph + expanded gap proposals + exact .771388 boundary cover. Gap scope locks outside old missing25 mask to boundary parent; all_live scope fixes D/X only. No new GT or model-generated sampling boundaries.',
       'metric':'Both halves joined by X; mean IoUs .3/.4/.5/.6/.7 and equal clean/corrupted. Fast interval score must match official evaluate.',
       'selection':'No OOF hyperparameter choice: independently WPY-calibrated setting per producer. Seal all OOF outputs then audit. One predetermined calibrated full composition receives one reused WQQ audit after full WPY calibration, even if OOF mixed, no fallback. Report gate and adaptive limits honestly.',
       'ties':'Prefer exact .771388-equivalent boundary parent, then fewer changed frames, then earlier grid index. No random tie or WQQ selection.',
       'clean':'Exact incumbent bypass without feature/model load. Preserve both prior candidates and original CSV/checkpoints. No WR9/challenge references or export.',
       'limits':'WPY reused and finite grid can overfit calibration; TRAIN4 OOF remains honest held primary ancestry but adaptively reused. Final primary TRAIN4 versus OOF TRAIN2 shift. WQQ reused development, not independent test.',
       'sources':{rel(x):sha256(x) for x in sources},'inputs':inputs})
    print('V11 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def boundary_parent(which,match,half,condition='corrupted'):
    if condition=='clean':return prior.old.c.previous.incumbent(match,half,condition)
    if match=='J03WQQ':
        with np.load(previous.RUN/f'full-predictions/{match}-{half}-corrupted.npz') as z:return z['boundary_verified'].copy()
    v,ee,_=previous.load_case(which,match,half)
    return previous.verifier.apply(v['baseline'],ee,np.arange(len(ee)))


def index_rows(arcs,query):
    dtype=np.dtype([('s','<i8'),('e','<i8'),('t','<i8')]);a=np.ascontiguousarray(arcs).view(dtype).ravel();q=np.ascontiguousarray(query).view(dtype).ravel()
    ids=np.searchsorted(a,q);np.testing.assert_array_equal(arcs[ids],query);return ids


def make_case(which,match,half):
    start=time.monotonic();key=f'{match}-{half}-corrupted';raw=b.raw_half(key)
    if match=='J03WPY':
        path=e.parent_wpy(which)/(key+'.npz')
        with np.load(path) as z:p={k:z[k] for k in z.files};owners=p
    else:
        p=b.ceiling.current(match,half,'corrupted');folder=prior.parents.OLD/'full-transfer/full_baseline' if which=='full' else prior.parents.OLD/f'baseline/fold{which}';path=folder/(key+'.npz')
        with np.load(path) as z:owners={f'owner_{i}':z[f'owner_{i}'] for i in range(3)}
    original=b.dense.proposals(p['predicted'],p['eligible_mask'],p['segments']);base=boundary_parent(which,match,half)
    arcs=gs.pool(base,base<2,np.r_[original,p['segments'],gs.spells(base)]);mask=gs.feasible(arcs,base,p['eligible_mask'])
    quality=b.original_quality(which);credit,_=b.fitted(which);qo=b.old_quality(quality,p['signals'],arcs);qr=e.EvidenceIntervals(raw,e.component(p['signals'],owners)).predict(credit,arcs)
    alpha=read(e.RUN/f'models/{which}/calibration.json')['alpha'];ids=index_rows(arcs,original);replay=fast.Graph(p['signals'][:,0],p['predicted'],original).decode((1-alpha)*qo[ids]+alpha*qr[ids],.4,.01)
    if match=='J03WPY':
        v,_,_=previous.load_case(which,match,half);expected=v['baseline']
    else:expected=prior.old.c.previous.incumbent(match,half,'corrupted')
    np.testing.assert_array_equal(replay,expected)
    values={'baseline':base,'baseline_07708':expected,'arcs':arcs,'gap_arc_mask':mask,'gap_frame_mask':p['eligible_mask'],'unary':p['signals'][:,0],'quality':qo,'credit':qr}
    dest=RUN/f'cases/producer{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,**values)
    meta={'cache':pin(dest),'primary_parent':pin(path),'producer':which,'match':match,'half':half,'arcs':len(arcs),'gap_arcs':int(mask.sum()),'baseline_replay_exact':True,'primary_proof':prior.producer_proof(which),'seconds':time.monotonic()-start}
    print('V11 CASE',which,match,half,len(arcs),'seconds',meta['seconds'],flush=True);return meta


def cache():
    verify();ALLOWED.clear();rows={}
    for m in TRAIN:
        for h in HALVES:rows[f'{m}-{h}']=make_case(prior.producer(m),m,h)
    for which in ['0','1','full']:
        for h in HALVES:rows[f'WPY-{which}-{h}']=make_case(which,'J03WPY',h)
    assert not READS;write(RUN/'case-seal.json',{'rows':rows,'reference_reads':[],'protocol':pin(RUN/'protocol.json')})


def load_case(which,match,half):
    seal=read(RUN/'full-case-seal.json' if match=='J03WQQ' else RUN/'case-seal.json');key=f'WPY-{which}-{half}' if match=='J03WPY' else f'{match}-{half}';meta=seal['rows'][key];assert meta['producer']==which
    with np.load(checked(meta['cache'])) as z:v={k:z[k] for k in z.files}
    return v,meta


def grid(which):
    p=read(RUN/'protocol.json')['grid'];alpha=read(e.RUN/f'models/{which}/calibration.json')['alpha']
    return [{'scope':s,'alpha':a,'penalty':d,'frame_weight':w} for s,a,d,w in itertools.product(p['scope'],sorted(set([v for v in p['alpha'] if isinstance(v,(int,float))]+[alpha])),p['penalty'],p['frame_weight'])]


def decode(v,cfg):
    if cfg['scope']=='baseline':return v['baseline'].copy()
    take=v['gap_arc_mask'] if cfg['scope']=='gap' else np.ones(len(v['arcs']),bool);q=(1-cfg['alpha'])*v['quality'][take]+cfg['alpha']*v['credit'][take]
    out=fast.Graph(v['unary'],v['baseline'],v['arcs'][take]).decode(q,cfg['penalty'],cfg['frame_weight'])
    gs.check_lock(out,v['baseline'],v['gap_frame_mask'] if cfg['scope']=='gap' else v['baseline']<2);return out


def spell_score(pm,y,p):
    runs=pm.valid_runs(y);refs=pm.possession_spells(y,runs);preds=pm.possession_spells(pm.fill_abstentions(p,runs),runs);overlap=pm.spell_overlaps(refs,preds)
    tp=np.array([pm.match_spells(overlap,len(refs),len(preds),t)[0] for t in pm.SPELL_IOUS])
    return {'spell_f1_mean':float(2*np.mean(tp)/(len(refs)+len(preds))) if len(refs)+len(preds) else 0.,'n_pred_spells':len(preds),'n_gt_spells':len(refs),'mean_TP':float(tp.mean())}


def calibrate(which):
    verify();ALLOWED.clear();ALLOWED.add('J03WPY');cases=[load_case(which,'J03WPY',h) for h in HALVES];y=prior.join_halves([prior.truth('J03WPY',h) for h in HALVES]);pm=prior.official_metrics();baseline=prior.join_halves([v[0]['baseline'] for v in cases]);rows=[];selected=None;best=None
    for i,cfg in enumerate([{'scope':'baseline'}]+grid(which)):
        out=prior.join_halves([decode(v,cfg) for v,_ in cases]);score=spell_score(pm,y,out);changed=int((out!=baseline).sum());rank=(score['spell_f1_mean'],-changed,-i);rows.append({'config':cfg,'score':score,'changed':changed})
        if best is None or rank>best:best=rank;selected=rows[-1]
        if i%40==0:print('V11 CAL PROGRESS',which,i,'best',best[0],flush=True)
    assert abs(spell_score(pm,y,baseline)['spell_f1_mean']-pm.evaluate(y,baseline,bootstrap_n=0)['spell_f1_mean'])<1e-12
    write(RUN/f'calibration/{which}.json',{'selected':selected,'rows':rows,'calibration_matches':['J03WPY'],'primary_proof':prior.producer_proof(which),'inputs':[m['cache'] for _,m in cases],'reference_reads':READS.copy(),'protocol':pin(RUN/'protocol.json')});ALLOWED.clear();print('V11 CALIBRATED',which,selected,flush=True)


def predict_case(which,match,half,condition):
    base=boundary_parent(which,match,half,condition);old=prior.old.c.previous.incumbent(match,half,condition)
    if condition=='clean':return {'baseline_07708':old,'baseline_07714':base,'metric_calibrated':base.copy()},{'clean_bypass':True,'changed':0}
    v,meta=load_case(which,match,half);np.testing.assert_array_equal(base,v['baseline']);cfg=read(RUN/f'calibration/{which}.json')['selected']['config'];out=decode(v,cfg)
    return {'baseline_07708':old,'baseline_07714':base,'metric_calibrated':out},{'changed':int((out!=base).sum()),'config':cfg,'cache':meta['cache']}


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={}
    for m in TRAIN:
        for h in HALVES:
            for c in ['clean','corrupted']:
                key=f'{m}-{h}-{c}';v,meta=predict_case(prior.producer(m),m,h,c);dest=RUN/f'oof/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:np.savez_compressed(f,**v)
                outputs[rel(dest)]=sha256(dest);views[key]=meta
    assert not READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'reference_reads':[],'calibrations':{w:pin(RUN/f'calibration/{w}.json') for w in ['0','1']},'cases':pin(RUN/'case-seal.json'),'protocol':pin(RUN/'protocol.json')})


def audit(full=False):
    global FULL
    verify();FULL=full;name='full' if full else 'oof';dest=RUN/f'{name}-audit.json';assert not dest.exists();seal=read(RUN/f'{name}-seal.json');checked(seal['protocol']);checked(seal['cases']);assert not seal['reference_reads'];matches=['J03WQQ'] if full else TRAIN;arms=['baseline_07708','baseline_07714','metric_calibrated'];loaded={}
    for path,h in seal['outputs'].items():
        assert sha256(ROOT/path)==h
        with np.load(ROOT/path) as z:v={k:z[k] for k in z.files}
        key=Path(path).stem;condition=key.rsplit('-',1)[1]
        for a in arms:
            gs.check_lock(v[a],v['baseline_07714'],v['baseline_07714']<2)
            if condition=='clean':np.testing.assert_array_equal(v[a],v['baseline_07714'])
        loaded[key]=v
    ALLOWED.clear();ALLOWED.update(matches);pm=prior.official_metrics();details={}
    for m in matches:
        y=prior.join_halves([prior.truth(m,h) for h in HALVES]);details[m]={c:{a:prior.score_full(pm,y,prior.join_halves([loaded[f'{m}-{h}-{c}'][a] for h in HALVES]))[0] for a in arms} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in arms};overall=avg(matches,['clean','corrupted']);conditions={c:avg(matches,[c]) for c in ['clean','corrupted']};pairs={} if full else {str(i):avg(TRAIN[:2] if i==0 else TRAIN[2:],['clean','corrupted']) for i in [0,1]};passed=overall['metric_calibrated']>overall['baseline_07714'] and all(v['metric_calibrated']>=v['baseline_07714'] for v in pairs.values())
    result={'overall':overall,'conditions':conditions,'per_match':{m:avg([m],['clean','corrupted']) for m in matches},'per_producer_pair':pairs,'details':details,'passed':passed,'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/f'{name}-seal.json'),'reference_reads':READS.copy(),'limitation':'Adaptively reused development; calibration only WPY, one full selected composition.'};write(dest,result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_match','per_producer_pair','passed']}),flush=True)


def full_run():
    global FULL
    verify();a=read(RUN/'oof-audit.json');checked(a['prediction_seal']);cfg=read(RUN/'calibration/full.json');checked(cfg['protocol']);write(RUN/'full-selection.json',{'calibration':pin(RUN/'calibration/full.json'),'OOF_audit':pin(RUN/'oof-audit.json'),'selected':cfg['selected'],'no_fallback':True});FULL=True;ALLOWED.clear();rows={}
    for h in HALVES:rows[f'J03WQQ-{h}']=make_case('full','J03WQQ',h)
    write(RUN/'full-case-seal.json',{'rows':rows,'reference_reads':[]});outputs={};views={}
    for h in HALVES:
        for c in ['clean','corrupted']:
            key=f'J03WQQ-{h}-{c}';v,meta=predict_case('full','J03WQQ',h,c);dest=RUN/f'full-predictions/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:np.savez_compressed(f,**v)
            outputs[rel(dest)]=sha256(dest);views[key]=meta;print('V11 FULL',key,meta,flush=True)
    assert not READS;write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'reference_reads':[],'cases':pin(RUN/'full-case-seal.json'),'selection':pin(RUN/'full-selection.json'),'protocol':pin(RUN/'protocol.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--producer',choices=['0','1','full']);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);prior.torch.set_num_threads(2);sys.addaudithook(guard)
    if args.stage=='calibrate':assert args.producer;calibrate(args.producer)
    elif args.stage.startswith('audit_'):audit(args.stage=='audit_full')
    else:globals()[args.stage]()
