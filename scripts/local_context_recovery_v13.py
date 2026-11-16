"""Unlabeled match-local ownership adaptation; WPY-only blend calibration."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits
from second_zero import local_context_adaptation as local
from scripts import metric_decoder_v11 as d
prior=d.prior;b=d.b;e=d.e;gs=d.gs
read=d.read;write=d.write;pin=d.pin;rel=d.rel;checked=d.checked;sha256=d.sha256
RUN=ROOT/'runs/local-context-recovery-v13';REPORT=ROOT/'reports/local-context-recovery-v13';TRAIN=d.TRAIN;HALVES=d.HALVES
ALPHAS=[0.,.1,.25,.5,.75,1.];FULL=False;HASH_ONLY=False;ALLOWED=set();READS=[];STAGE='prepare'


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(s in p for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved assets closed')
    if 'J03WQQ' in p and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed before frozen selection')
    if p.endswith('.truth.npy'):
        if Path(p).name.split('-')[0] not in ALLOWED:raise PermissionError('Primary labels closed during local adaptation')
        READS.append({'stage':STAGE,'path':rel(p),'allowed':sorted(ALLOWED)})


def prepare():
    p=read(d.RUN/'protocol.json');inputs={**p['sources'],**p['inputs']}
    for file in [d.RUN/'case-seal.json',ROOT/'reports/metric-decoder-v11/artifact-freeze.json']:
        inputs[rel(file)]=sha256(file)
    inputs.update(read(ROOT/'reports/metric-decoder-v11/artifact-freeze.json')['hashes'])
    sources=[Path(__file__),Path(local.__file__),Path(local.context.__file__)]
    write(RUN/'protocol.json',{'kind':'unlabeled-local-context-gap-recovery-v13','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_best':.7713875141348288,
       'adaptation':'Fit separately on each input half using only corrupted raw coordinates + frozen complete teacher logit. Pseudo control anchors require observed ball, nearest player<2m, opponent margin>2m, speed<12m/s, |teacher|>=1.5 and teacher agreement. Native stride5. No originally missing ball or any primary reference is a pseudo target.',
       'student':'Only raw290 player/team/pitch flow/shape context. Both symmetric team perspectives share one HGB. No ball, model logit, time, match ID or reference enters student features. Frozen model settings.',
       'model':local.MODEL,'pseudo_validation':'Every fifth500-frame block held;501-frame buffer excludes neighboring training anchors. NLL must beat balanced chance log2. Same fixed model then fits all observed pseudo anchors; no hyperparameter selection. At least200 anchors,100 training and50 held. No primary GT.',
       'composition':'Blend local and frozen teacher probabilities by alpha*raw-missing25 support. Existing .771388 parent/status/old gap mask and expanded proposals fixed. Recompute original175 quality using updated FULL final evidence; canonical747 credit is unchanged by x0 update because component replaces x0 with frozen uniform3. Fixed.4 segment penalty/.01 frame weight. Outside original gap mask, D/X and clean exact. Alpha0 bypass returns exact boundary parent.',
       'alpha_grid':ALPHAS,'calibration':'WPY-only scalar alpha maximizing pooled full-match five-IoU SpellF1; alpha0 wins ties. Fit local pseudo models and seal all WPY/OOF alpha arrays before any current calibration/audit reference read. No WQQ fit/calibration reference access.',
       'selection':'Independent WPY calibration per frozen producer; all OOF predictions sealed before audit. One full producer WPY alpha frozen before WQQ adaptation/inference. Only that alpha evaluated on WQQ, no fallback. Alpha0 requires no local WQQ adaptation.',
       'interpretation':'Transductive unlabeled adaptation on test input coordinates, deployable with same input, not supervised fitting to held truth. Primary teachers are strictly held-match excluded. Pseudo block NLL is a diagnostic, not true possession accuracy. Reused WPY/WQQ, primary TRAIN2→4 shift and only4 physical OOF matches limit claims.',
       'sources':{rel(x):sha256(x) for x in sources},'inputs':inputs})
    print('V13 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def signals(which,match,half):
    if match=='J03WPY':
        with np.load(e.parent_wpy(which)/f'{match}-{half}-corrupted.npz') as z:return z['signals'].copy()
    return b.ceiling.current(match,half,'corrupted')['signals']


def adapt(which,match,half):
    start=time.monotonic();v,meta=d.load_case(which,match,half);raw=b.raw_half(f'{match}-{half}-corrupted');pred,model,info=local.fit_predict(raw,v['unary'],v['baseline']);folder=RUN/f'adaptation/producer{which}/{match}-{half}';folder.mkdir(parents=True,exist_ok=False);model_pin=None
    if model is not None:
        with (folder/'model.pkl').open('xb') as f:pickle.dump(model,f)
        model_pin=pin(folder/'model.pkl')
    support=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest')
    with (folder/'local.npz').open('xb') as f:np.savez_compressed(f,local=pred,support=support)
    result={'producer':which,'match':match,'half':half,'model':model_pin,'local':pin(folder/'local.npz'),'pseudo':info,'source':meta['cache'],'primary_proof':prior.producer_proof(which),'primary_reference_reads':READS.copy(),'seconds':time.monotonic()-start};write(folder/'freeze.json',result);print('V13 LOCAL',which,match,half,info,'seconds',result['seconds'],flush=True);return v,signals(which,match,half),result


def predict(v,x,local_pred,support,alpha,which,adapted=True):
    if alpha==0 or not adapted:return v['baseline'].copy()
    z=local.blend(v['unary'],local_pred,support,alpha);xx=x.copy();xx[:,0]=z;take=v['gap_arc_mask'];arcs=v['arcs'][take];qo=b.old_quality(b.original_quality(which),xx,arcs);oldalpha=read(e.RUN/f'models/{which}/calibration.json')['alpha'];q=(1-oldalpha)*qo+oldalpha*v['credit'][take]
    out=d.fast.Graph(z,v['baseline'],arcs).decode(q,.4,.01);gs.check_lock(out,v['baseline'],v['gap_frame_mask']);return out


def make_case(which,match,half,alphas):
    before=len(READS);v,x,meta=adapt(which,match,half)
    with np.load(checked(meta['local'])) as z:lp=z['local'];support=z['support']
    values={'baseline':v['baseline']}
    for i,alpha in enumerate(alphas):values[f'alpha_{i}']=predict(v,x,lp,support,alpha,which,meta['pseudo']['accepted'])
    assert len(READS)==before;dest=RUN/f'cases/producer{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,**values)
    return {'cache':pin(dest),'adaptation':meta,'alphas':alphas}


def cache():
    verify();ALLOWED.clear();rows={}
    for m in TRAIN:
        for h in HALVES:rows[f'{m}-{h}']=make_case(prior.producer(m),m,h,ALPHAS)
    for which in ['0','1','full']:
        for h in HALVES:rows[f'WPY-{which}-{h}']=make_case(which,'J03WPY',h,ALPHAS)
    assert not READS;write(RUN/'case-seal.json',{'rows':rows,'reference_reads':[],'protocol':pin(RUN/'protocol.json')})


def load_case(which,match,half,full=False):
    seal=read(RUN/('full-case-seal.json' if full else 'case-seal.json'));key=f'WPY-{which}-{half}' if match=='J03WPY' else f'{match}-{half}';meta=seal['rows'][key]
    with np.load(checked(meta['cache'])) as z:v={k:z[k] for k in z.files}
    return v,meta


def calibrate(which):
    verify();ALLOWED.clear();ALLOWED.add('J03WPY');cases=[load_case(which,'J03WPY',h)[0] for h in HALVES];y=prior.join_halves([prior.truth('J03WPY',h) for h in HALVES]);pm=prior.official_metrics();rows=[]
    for i,alpha in enumerate(ALPHAS):
        out=prior.join_halves([v[f'alpha_{i}'] for v in cases]);rows.append({'alpha':alpha,'score':d.spell_score(pm,y,out),'changed':sum(int((v[f'alpha_{i}']!=v['baseline']).sum()) for v in cases)})
    selected=max(enumerate(rows),key=lambda v:(v[1]['score']['spell_f1_mean'],-v[1]['changed'],-v[0]))[1];write(RUN/f'calibration/{which}.json',{'selected':selected,'rows':rows,'calibration_matches':['J03WPY'],'reference_reads':READS.copy(),'case_seal':pin(RUN/'case-seal.json'),'protocol':pin(RUN/'protocol.json')});ALLOWED.clear();print('V13 CALIBRATED',which,selected,flush=True)


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={}
    for m in TRAIN:
        which=prior.producer(m);alpha=read(RUN/f'calibration/{which}.json')['selected']['alpha'];idx=ALPHAS.index(alpha)
        for h in HALVES:
            for c in ['clean','corrupted']:
                key=f'{m}-{h}-{c}';base=d.boundary_parent(which,m,h,c)
                if c=='clean':out=base.copy();meta={'clean_bypass':True}
                else:v,cm=load_case(which,m,h);np.testing.assert_array_equal(v['baseline'],base);out=v[f'alpha_{idx}'];meta={'alpha':alpha,'case':cm['cache'],'changed':int((out!=base).sum())}
                dest=RUN/f'oof/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:np.savez_compressed(f,baseline_07714=base,local_context=out)
                outputs[rel(dest)]=sha256(dest);views[key]=meta
    assert not READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'reference_reads':[],'protocol':pin(RUN/'protocol.json'),'cases':pin(RUN/'case-seal.json'),'calibrations':{w:pin(RUN/f'calibration/{w}.json') for w in ['0','1']}})


def audit(full=False):
    global FULL
    verify();FULL=full;name='full' if full else 'oof';dest=RUN/f'{name}-audit.json';assert not dest.exists();seal=read(RUN/f'{name}-seal.json');checked(seal['protocol']);assert not seal['reference_reads'];matches=['J03WQQ'] if full else TRAIN;loaded={};arms=['baseline_07714','local_context']
    for path,h in seal['outputs'].items():
        assert sha256(ROOT/path)==h
        with np.load(ROOT/path) as z:v={k:z[k] for k in z.files}
        key=Path(path).stem
        for a in arms:
            gs.check_lock(v[a],v['baseline_07714'],v['baseline_07714']<2)
            if key.endswith('-clean'):np.testing.assert_array_equal(v[a],v['baseline_07714'])
        loaded[key]=v
    ALLOWED.clear();ALLOWED.update(matches);pm=prior.official_metrics();details={}
    for m in matches:
        y=prior.join_halves([prior.truth(m,h) for h in HALVES]);details[m]={c:{a:prior.score_full(pm,y,prior.join_halves([loaded[f'{m}-{h}-{c}'][a] for h in HALVES]))[0] for a in arms} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in arms};overall=avg(matches,['clean','corrupted']);conditions={c:avg(matches,[c]) for c in ['clean','corrupted']};pairs={} if full else {str(i):avg(TRAIN[:2] if i==0 else TRAIN[2:],['clean','corrupted']) for i in [0,1]};passed=overall['local_context']>overall['baseline_07714'] and all(v['local_context']>=v['baseline_07714'] for v in pairs.values());result={'overall':overall,'conditions':conditions,'per_match':{m:avg([m],['clean','corrupted']) for m in matches},'per_producer_pair':pairs,'details':details,'passed':passed,'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/f'{name}-seal.json'),'reference_reads':READS.copy(),'limitation':'Unlabeled transductive adaptation; primary labels only WPY calibration and sealed audit. Reused development, no independent/private test claim.'};write(dest,result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_match','per_producer_pair','passed']}),flush=True)


def full_run():
    global FULL
    verify();a=read(RUN/'oof-audit.json');checked(a['prediction_seal']);cal=read(RUN/'calibration/full.json');write(RUN/'full-selection.json',{'OOF_audit':pin(RUN/'oof-audit.json'),'calibration':pin(RUN/'calibration/full.json'),'selected':cal['selected'],'no_fallback':True});alpha=cal['selected']['alpha'];FULL=True;ALLOWED.clear();outputs={};views={};rows={}
    if alpha:
        for h in HALVES:rows[f'J03WQQ-{h}']=make_case('full','J03WQQ',h,[alpha])
        assert not READS;write(RUN/'full-case-seal.json',{'rows':rows,'reference_reads':[],'selection':pin(RUN/'full-selection.json')})
    for h in HALVES:
        for c in ['clean','corrupted']:
            base=d.boundary_parent('full','J03WQQ',h,c);meta={'alpha':alpha,'clean_bypass':c=='clean'}
            if c=='clean' or not alpha:out=base.copy()
            else:v,cm=load_case('full','J03WQQ',h,True);out=v['alpha_0'];meta['case']=cm['cache']
            meta['changed']=int((out!=base).sum());key=f'J03WQQ-{h}-{c}';dest=RUN/f'full-predictions/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:np.savez_compressed(f,baseline_07714=base,local_context=out)
            outputs[rel(dest)]=sha256(dest);views[key]=meta;print('V13 FULL',key,meta,flush=True)
    assert not READS;write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'reference_reads':[],'selection':pin(RUN/'full-selection.json'),'protocol':pin(RUN/'protocol.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--producer',choices=['0','1','full']);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);prior.torch.set_num_threads(2);sys.addaudithook(guard)
    if args.stage=='calibrate':assert args.producer;calibrate(args.producer)
    elif args.stage.startswith('audit_'):audit(args.stage=='audit_full')
    else:globals()[args.stage]()
