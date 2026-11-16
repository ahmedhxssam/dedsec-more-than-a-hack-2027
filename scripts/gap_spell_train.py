"""Raw-only gap spell scorer, fixed controls, sealed OOF and one full transfer."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from second_zero import gap_spell as gs,gap_spell_dense as dense,gap_spell_model as model_api
from second_zero.full_baseline_pipeline import ExplicitRawPipeline,gap_candidate,raw_half,FOLDS,keys
from second_zero.full_baseline_sources import scientific_sources
from scripts.global_team_context_probe import owner as global_owner
from scripts.full_baseline_evaluate import join_halves,score_full
from scripts import gap_spell_ceiling as ceiling
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts.train_contextual import corruption

RUN=ceiling.RUN;REPORT=ceiling.REPORT;OLD=ceiling.OLD;TRAIN=ceiling.TRAIN
STAGE='prepare';ALLOWED=set();READS=[];FULL_ALLOWED=False;HASH_ONLY=False
ARMS=['baseline','proposal_control','mixed','raw_credit']


def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as h:json.dump(v,h,indent=2,allow_nan=False);h.write('\n')
def pin(p):return {'path':rel(p),'sha256':sha256(p)}
def check(v):
    p=ROOT/v['path']
    if sha256(p)!=v['sha256']:raise ValueError('Changed artifact: '+v['path'])
    return p
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.fsdecode(args[0])
    if any(s in path for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved asset closed')
    if 'J03WQQ' in path and not FULL_ALLOWED and not HASH_ONLY:raise PermissionError('WQQ closed before frozen OOF selection')
    if path.endswith('.truth.npy'):
        match=Path(path).name.split('-')[0]
        if match not in ALLOWED:raise PermissionError('Reference outside current fitting/audit role: '+path)
        READS.append({'stage':STAGE,'path':rel(path),'allowed':sorted(ALLOWED)})
    if '/train-cache/' in path and path.endswith('.npy'):
        match=Path(path).parent.name
        if match not in ALLOWED:raise PermissionError('Training cache outside current fit matches')


def raw(match,half,view):
    with np.load(ROOT/f'private/improvement/external/prepared/{match}-{half}.npz',allow_pickle=False) as z:r={k:z[k] for k in ['ball','players','team','pitch']}
    return r if view=='clean' else corruption(r,int(view))
def truth(match,half):return np.load(ROOT/f'private/improvement/external/prepared/{match}-{half}.truth.npy')


def prepare():
    headroom=read(REPORT/'expanded-ceiling.json')
    if not headroom['sufficient_headroom']:raise PermissionError('Ceiling did not authorize scorer fit')
    old=read(RUN/'ceiling-protocol.json')
    files=[Path(__file__),Path(model_api.__file__),Path(dense.__file__),Path(gs.__file__),*scientific_sources(),ROOT/'scripts/gap_spell_ceiling.py',ROOT/'scripts/full_baseline_evaluate.py']
    inputs={**old['inputs'],rel(REPORT/'expanded-ceiling.json'):sha256(REPORT/'expanded-ceiling.json'),rel(RUN/'expanded-ceiling-protocol.json'):sha256(RUN/'expanded-ceiling-protocol.json')}
    for p in [OLD/'producer-protocol.json',OLD/'candidate-ancestry.json',OLD/'OOF-audit.json',OLD/'full-transfer/protocol.json',OLD/'full-reference-spec.json',OLD/'experiment-recipe.json',OLD/'specs/fold0.json',OLD/'specs/fold1.json']:
        inputs[rel(p)]=sha256(p)
    write(RUN/'scorer-protocol.json',{'kind':'raw-gap-spell-scorer-v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline':.769357376463475,'baseline_id':'gap_recovery from full-baseline-gap-v1','folds':{str(i):{'train':t,'held':h} for i,(t,h) in enumerate(FOLDS)},
        'model':model_api.MODEL,'features':'Raw314=anonymous player/team/pitch290 + bounded observed-ball interpolation24. Interval means, selected standard deviations, before/after5/25/75, native first/last5, duration/missingness. No learned logits, quality, predictions, teacher or match identifiers enter fitted features.',
        'training':'TRAIN2 only per fold; TRAIN4 full only if selected. Each match two halves x clean/corruption100/corruption271. Fitting-only GT jitter ±25/10/5/0 plus raw fixed-duration grid; uniform fixed-seed cap20000 per view, no target-dependent retention. Equal view weights. No learned feature ancestors.',
        'target':'Original expected independent best same-team IoU matching credit over .3/.4/.5/.6/.7; one HGB model, no model grid or checkpoint selection.',
        'calibration':'WPY corrupted views only, actual samefold full .7694-family parent; scalar alpha minimizing squared proposal-credit residual via analytic least squares, clipped[0,1]. No WPY SpellF1 search.',
        'pool':'Frozen expanded-ceiling proposal generator, stride5/duration<=250, feasible exact whole-spell locks, exact parent cover.',
        'arms':{'proposal_control':'Original175 quality, penalty.55/frame.01, denser feasible pool only','mixed':'Same control plus q=(1-alpha)*old_quality+alpha*raw_credit, WPYalpha only','raw_credit':'Raw interval credit only; independent-credit F1 target.8 gives penalty.4, frame_weight0; no tuning'},
        'selection':'All64 OOF label arrays sealed before one audit. Best official full-match SpellF1 with strict overall/corrupted improvement and nonnegative both-fold deltas; baseline wins ties. Report matched control. One selected full WQQ transfer only, no fallback.',
        'clean_and_status_lock':'Exact parent outside supplied raw missing25 live mask; clean bypass, all D/X exact. Parent .7694 never overwritten.',
        'adaptivity':'TRAIN4/WPY/WQQ have prior development exposure, including descriptive ceiling; no untouched-test claim. WQQ GT never enters fitting or calibration.',
        'sources':{rel(p):sha256(p) for p in files},'inputs':inputs})
    print('SCORER PROTOCOL',sha256(RUN/'scorer-protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'scorer-protocol.json')
    # Streaming preservation hashes of the already-scored parent are not
    # feature/model input reads. References remain blocked by ALLOWED below.
    HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():
            if sha256(ROOT/path)!=h:raise ValueError('Frozen source/input changed: '+path)
    finally:HASH_ONLY=False
    return p


def cache():
    verify();pm=official_metrics();manifest={}
    for mi,match in enumerate(TRAIN):
        ALLOWED.clear();ALLOWED.add(match)
        for hi,half in enumerate(['firstHalf','secondHalf']):
            y=truth(match,half)
            for vi,view in enumerate(['clean','100','271']):
                before=time.monotonic();r=raw(match,half,view);a=model_api.training_proposals(y,r,pm,20261030+mi*100+hi*10+vi);pack=model_api.RawIntervals(r);x=pack.features(a);target=gs.credit(y,a,pm);folder=RUN/'train-cache'/match;folder.mkdir(parents=True,exist_ok=True);stem=folder/f'{half}-{view}'
                np.save(str(stem)+'-x.npy',x);np.save(str(stem)+'-y.npy',target.astype(np.float32))
                manifest[str(stem.relative_to(ROOT))]={'match':match,'view':view,'rows':len(a),'features':x.shape[1],'x':pin(Path(str(stem)+'-x.npy')),'y':pin(Path(str(stem)+'-y.npy')),'reference':pin(ROOT/f'private/improvement/external/prepared/{match}-{half}.truth.npy'),'raw_observations':pin(ROOT/f'private/improvement/external/prepared/{match}-{half}.npz'),'seconds':time.monotonic()-before}
                print('RAW INTERVAL CACHE',match,half,view,x.shape,'seconds',time.monotonic()-before,flush=True);del x,target,pack,r
    ALLOWED.clear();write(RUN/'training-cache-seal.json',{'protocol_sha256':sha256(RUN/'scorer-protocol.json'),'rows':manifest,'reference_reads':READS})


def fitted(which):
    folder=RUN/'models'/str(which);f=read(folder/'freeze.json')
    with check(f['model']).open('rb') as h:return pickle.load(h),f


def fit(which):
    verify();matches=TRAIN if which=='full' else FOLDS[int(which)][0];ALLOWED.update(matches);cache_seal=read(RUN/'training-cache-seal.json');xx=[];yy=[];ww=[];inputs={};start=time.monotonic()
    for row in cache_seal['rows'].values():
        if row['match'] not in matches:continue
        x=np.load(check(row['x']));y=np.load(check(row['y']));xx.append(x);yy.append(y);ww.append(np.full(len(y),1/len(y)));inputs[row['x']['path']]=row['x']['sha256'];inputs[row['y']['path']]=row['y']['sha256']
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean();del xx,yy,ww
    model=HistGradientBoostingRegressor(**model_api.MODEL).fit(x,y,sample_weight=w);folder=RUN/'models'/str(which);folder.mkdir(parents=True,exist_ok=False)
    with (folder/'model.pkl').open('xb') as h:pickle.dump(model,h)
    freeze={'train_matches':matches,'calibration_matches':[],'raw_only_no_learned_ancestors':True,'shape':list(x.shape),'model':pin(folder/'model.pkl'),'protocol_sha256':sha256(RUN/'scorer-protocol.json'),'training_cache_seal':pin(RUN/'training-cache-seal.json'),'training_inputs':inputs,'seconds':time.monotonic()-start,'reference_reads':READS.copy()};write(folder/'freeze.json',freeze);ALLOWED.clear();print('RAW SPELL FIT',which,x.shape,freeze['seconds'],flush=True)


def parent_wpy(which):
    folder=RUN/'calibration-parent'/str(which)
    if (folder/'seal.json').exists():
        s=read(folder/'seal.json')
        for path,h in s['outputs'].items():
            if sha256(ROOT/path)!=h:raise ValueError('WPY parent changed')
        return folder
    spec=read(OLD/'full-reference-spec.json') if which=='full' else read(OLD/f'specs/fold{which}.json');pipeline=ExplicitRawPipeline(spec)
    item=read(OLD/'full-transfer/protocol.json')['expert'] if which=='full' else read(OLD/'experiment-recipe.json')['expert_models'][str(which)]
    with check(item).open('rb') as h:expert=pickle.load(h)
    outputs={}
    for half in ['firstHalf','secondHalf']:
        key=f'J03WPY-{half}-corrupted';r=raw_half(key);base=pipeline.predict(r);pack=model_api.RawIntervals(r);z=global_owner(expert,pack.x[:,:,:290]);g=gap_candidate(r,base,z,pipeline.quality);values={**base,**g,'global_logit':z};folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.npz');np.savez_compressed(path,**values);outputs[rel(path)]=sha256(path);print('WPY FULL PARENT',which,half,flush=True)
    write(folder/'seal.json',{'outputs':outputs,'spec':pin(OLD/'full-reference-spec.json' if which=='full' else OLD/f'specs/fold{which}.json'),'expert':item,'truth_reads':[]})
    return folder


def original_quality(which):
    spec=read(OLD/'full-reference-spec.json' if which=='full' else OLD/f'specs/fold{which}.json')
    with check(spec['quality']).open('rb') as h:return pickle.load(h)


def old_quality(model,x,arcs):
    out=np.empty(len(arcs))
    for s in range(0,len(arcs),8192):out[s:s+8192]=model.predict(graph.segment_features(x,arcs[s:s+8192]))
    return np.clip(out,0,1)


def calibrate(which):
    verify();model,freeze=fitted(which);folder=parent_wpy(which);qmodel=original_quality(which);pm=official_metrics();ALLOWED.clear();ALLOWED.add('J03WPY');numerator=0.;denominator=0.;views={};dependencies={}
    for half in ['firstHalf','secondHalf']:
        path=folder/f'J03WPY-{half}-corrupted.npz'
        with np.load(path,allow_pickle=False) as z:parent={k:z[k] for k in ['predicted','eligible_mask','segments','signals']}
        r=raw_half(f'J03WPY-{half}-corrupted');pack=model_api.RawIntervals(r);arcs=dense.proposals(parent['predicted'],parent['eligible_mask'],parent['segments']);qr=pack.predict(model,arcs);qo=old_quality(qmodel,parent['signals'],arcs);target=gs.credit(truth('J03WPY',half),arcs,pm);d=qr-qo
        numerator+=float(np.mean(d*(target-qo)));denominator+=float(np.mean(d*d));dependencies[rel(path)]=sha256(path);views[half]={'proposals':len(arcs),'raw_MSE':float(np.mean((qr-target)**2)),'old_MSE':float(np.mean((qo-target)**2))}
    alpha=float(np.clip(numerator/max(denominator,1e-12),0,1));write(RUN/'models'/str(which)/'calibration.json',{'calibration_matches':['J03WPY'],'alpha':alpha,'model':freeze['model'],'dependencies':dependencies,'views':views,'objective':'equal-view squared independent IoU-credit error; analytic convex blend; no SpellF1 selection','reference_reads':READS.copy()});ALLOWED.clear();print('RAW SPELL ALPHA',which,alpha,flush=True)


def prediction(which,match,half,condition,arms):
    parent=ceiling.current(match,half,condition);p=parent['predicted'];mask=parent['eligible_mask'];out={'baseline':p.copy(),'eligible_mask':mask}
    if not mask.any():
        out.update({a:p.copy() for a in arms});return out
    r=raw_half(f'{match}-{half}-{condition}');arcs=dense.proposals(p,mask,parent['segments']);qmodel=original_quality(which);qo=old_quality(qmodel,parent['signals'],arcs)
    if any(a!='proposal_control' for a in arms):
        model,freeze=fitted(which);cal=read(RUN/'models'/str(which)/'calibration.json');assert cal['model']==freeze['model'];pack=model_api.RawIntervals(r);qr=pack.predict(model,arcs);alpha=cal['alpha']
    for arm in arms:
        if arm=='proposal_control':q=qo;penalty=.55;fw=.01
        elif arm=='mixed':q=(1-alpha)*qo+alpha*qr;penalty=.55;fw=.01
        elif arm=='raw_credit':q=qr;penalty=.4;fw=0.
        else:raise ValueError(arm)
        out[arm]=gs.decode(p,mask,arcs,q,penalty,parent['signals'][:,0],fw)
        gs.check_lock(out[arm],p,mask)
    return out


def predict_oof():
    verify();ALLOWED.clear();outputs={};details={}
    for f,(_,held) in enumerate(FOLDS):
        for match in held:
            for half in ['firstHalf','secondHalf']:
                for condition in ['clean','corrupted']:
                    key=f'{match}-{half}-{condition}';before=time.monotonic();values=prediction(str(f),match,half,condition,ARMS[1:]);path=RUN/f'oof/fold{f}'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values);outputs[rel(path)]=sha256(path);details[key]={a:int((values[a]!=values['baseline']).sum()) for a in ARMS[1:]};print('OOF SPELL PREDICTED',key,details[key],'seconds',time.monotonic()-before,flush=True)
    write(RUN/'oof-prediction-seal.json',{'protocol_sha256':sha256(RUN/'scorer-protocol.json'),'outputs':outputs,'label_arrays':ARMS,'views':details,'held_truth_reads':[],'models':{str(f):{'freeze':pin(RUN/f'models/{f}/freeze.json'),'calibration':pin(RUN/f'models/{f}/calibration.json')} for f in [0,1]}})


def validated_oof():
    verify();seal=read(RUN/'oof-prediction-seal.json');assert len(seal['outputs'])==16 and seal['label_arrays']==ARMS and not seal['held_truth_reads'];loaded={}
    for f,(_,held) in enumerate(FOLDS):
        model=seal['models'][str(f)];freeze=read(check(model['freeze']));cal=read(check(model['calibration']));check(freeze['training_cache_seal']);assert freeze['train_matches']==FOLDS[f][0] and freeze['raw_only_no_learned_ancestors'];assert cal['calibration_matches']==['J03WPY'] and cal['model']==freeze['model'] and 0<=cal['alpha']<=1;check(freeze['model'])
        for key in keys(held):
            path=RUN/f'oof/fold{f}'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            match,half,condition=key.split('-');old=ceiling.current(match,half,condition);np.testing.assert_array_equal(v['baseline'],old['predicted']);np.testing.assert_array_equal(v['eligible_mask'],old['eligible_mask'])
            for a in ARMS:
                gs.check_lock(v[a],v['baseline'],v['eligible_mask'])
                if condition=='clean':np.testing.assert_array_equal(v[a],v['baseline'])
            loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One OOF audit only')
    seal,loaded=validated_oof();pm=official_metrics();ALLOWED.update(TRAIN);rows={}
    for match in TRAIN:
        y=join_halves([truth(match,h) for h in ['firstHalf','secondHalf']]);rows[match]={}
        for c in ['clean','corrupted']:
            rows[match][c]={a:score_full(pm,y,join_halves([loaded[f'{match}-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ARMS}
    avg=lambda ms,cs:{a:float(np.mean([rows[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in ARMS}
    overall=avg(TRAIN,['clean','corrupted']);conditions={c:avg(TRAIN,[c]) for c in ['clean','corrupted']};folds={str(f):avg(h,['clean','corrupted']) for f,(_,h) in enumerate(FOLDS)}
    valid=[a for a in ARMS[1:] if overall[a]>overall['baseline'] and conditions['corrupted'][a]>conditions['corrupted']['baseline'] and all(v[a]>=v['baseline'] for v in folds.values())]
    selected=max(['baseline']+valid,key=lambda a:overall[a]);result={'overall':overall,'conditions':conditions,'per_fold':folds,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':rows,'selected':selected,'passed':selected!='baseline','prediction_seal':pin(RUN/'oof-prediction-seal.json'),'protocol_sha256':sha256(RUN/'scorer-protocol.json'),'clean_exact':True,'D_X_outside_exact':True,'reference_reads':READS,'interpretation':'Repeated development; all64 label arrays sealed; baseline/control included; no WQQ candidate scored.'};write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_fold','selected','passed']}),flush=True)


def authorize_full():
    global FULL_ALLOWED
    verify();a=read(RUN/'oof-audit.json');check(a['prediction_seal']);validated_oof()
    if not a['passed'] or a['selected']=='baseline':raise PermissionError('OOF gate did not authorize full transfer')
    if a['overall'][a['selected']]<=a['overall']['baseline']:raise PermissionError('No strict OOF improvement')
    FULL_ALLOWED=True;return a['selected']


def fit_full():
    selected=authorize_full()
    if selected!='proposal_control':fit('full');calibrate('full')
    write(RUN/'full-selection.json',{'selected':selected,'OOF_audit':pin(RUN/'oof-audit.json'),'protocol':pin(RUN/'scorer-protocol.json'),'no_fallback':True})


def predict_full():
    selected=authorize_full();selection=read(RUN/'full-selection.json');assert selection['selected']==selected;ALLOWED.clear();outputs={}
    for half in ['firstHalf','secondHalf']:
        for c in ['clean','corrupted']:
            key=f'J03WQQ-{half}-{c}';v=prediction('full','J03WQQ',half,c,[selected]);path=RUN/'full-predictions'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);print('FULL SPELL PREDICTED',key,'changed',int((v[selected]!=v['baseline']).sum()),flush=True)
    write(RUN/'full-prediction-seal.json',{'outputs':outputs,'selection':pin(RUN/'full-selection.json'),'label_arrays':['baseline',selected],'truth_reads':[]})


def audit_full():
    if (RUN/'full-audit.json').exists():raise FileExistsError('One full transfer audit only')
    selected=authorize_full();seal=read(RUN/'full-prediction-seal.json');check(seal['selection']);assert len(seal['outputs'])==4 and not seal['truth_reads'];loaded={}
    for half in ['firstHalf','secondHalf']:
        for c in ['clean','corrupted']:
            key=f'J03WQQ-{half}-{c}';path=RUN/'full-predictions'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            parent=ceiling.current('J03WQQ',half,c);np.testing.assert_array_equal(v['baseline'],parent['predicted']);np.testing.assert_array_equal(v['eligible_mask'],parent['eligible_mask']);gs.check_lock(v[selected],v['baseline'],v['eligible_mask'])
            if c=='clean':np.testing.assert_array_equal(v[selected],v['baseline'])
            loaded[key]=v
    ALLOWED.clear();ALLOWED.add('J03WQQ');pm=official_metrics();y=join_halves([truth('J03WQQ',h) for h in ['firstHalf','secondHalf']]);conditions={}
    for c in ['clean','corrupted']:conditions[c]={a:score_full(pm,y,join_halves([loaded[f'J03WQQ-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ['baseline',selected]}
    overall={a:float(np.mean([conditions[c][a]['spell_f1_mean'] for c in ['clean','corrupted']])) for a in ['baseline',selected]};assert abs(overall['baseline']-.769357376463475)<1e-12
    result={'selected':selected,'conditions':conditions,'overall':overall,'improved':overall[selected]>overall['baseline'],'clean_exact':True,'D_X_outside_exact':True,'prediction_seal':pin(RUN/'full-prediction-seal.json'),'reference_reads':READS,'no_fallback':True,'interpretation':'One adaptively reused WQQ development match, no private challenge claim.'};write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'conditions':{c:{a:r['spell_f1_mean'] for a,r in v.items()} for c,v in conditions.items()},'improved':result['improved']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cache','fit','calibrate','predict_oof','audit_oof','fit_full','predict_full','audit_full']);parser.add_argument('--fold',choices=['0','1']);args=parser.parse_args();STAGE=args.stage
    threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(guard)
    if args.stage in ['fit','calibrate']:
        if args.fold is None:raise ValueError('Fold required')
        globals()[args.stage](args.fold)
    else:globals()[args.stage]()
