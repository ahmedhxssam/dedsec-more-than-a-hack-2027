"""Self-supervised observed-coordinate adaptation; exact complete incumbent test."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from second_zero import observed_ball_adaptation as adaptation
from second_zero import gap_spell as gs
from second_zero.full_baseline_pipeline import ExplicitRawPipeline,gap_candidate
from second_zero.global_team_context import features as global_features
from second_zero.full_baseline_sources import scientific_sources
from scripts.global_team_context_probe import owner as global_owner
from scripts import gap_spell_evidence as e
from scripts import corrupted_recovery_v5 as previous
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from scripts.full_baseline_evaluate import join_halves,score_full

b=e.base;OLD=b.OLD;INC=ROOT/'runs/gap-spell-v4-f1';RUN=ROOT/'runs/observed-ball-recovery-v6';REPORT=ROOT/'reports/observed-ball-recovery-v6'
TRAIN=b.TRAIN;FOLDS=b.FOLDS;ALLOWED=set();FULL=False;HASH_ONLY=False;READS=[];STAGE='prepare'


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def pin(p):return {'path':rel(p),'sha256':sha256(p)}
def check(v):
    p=ROOT/v['path'];assert sha256(p)==v['sha256'];return p
def truth(m,h):return np.load(ROOT/f'private/improvement/external/prepared/{m}-{h}.truth.npy')
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.fsdecode(args[0])
    if any(s in path for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved asset closed')
    if 'J03WQQ' in path and not FULL and not HASH_ONLY:raise PermissionError('WQQ closed before fixed OOF selection')
    if path.endswith('.truth.npy'):
        if Path(path).name.split('-')[0] not in ALLOWED:raise PermissionError('Possession/status references closed during coordinate adaptation')
        READS.append({'stage':STAGE,'path':rel(path),'allowed':sorted(ALLOWED)})


def prepare():
    pointer=read(ROOT/'configs/development_best.json');manifest=read(check(pointer['manifest']));prior=read(previous.RUN/'oof-audit.json');assert not prior['passed']
    sources=[Path(__file__),Path(adaptation.__file__),Path(adaptation.imputer.__file__),Path(previous.__file__),Path(e.__file__),Path(b.__file__),ROOT/'second_zero/gap_spell.py',ROOT/'second_zero/gap_spell_dense.py',ROOT/'second_zero/gap_spell_model.py',ROOT/'scripts/full_baseline_evaluate.py',*scientific_sources()]
    write(RUN/'protocol.json',{'kind':'observed-coordinate-self-supervised-corrupted-recovery-v6','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':pointer['score'],'OOF_baseline':pointer['strict_OOF'],
      'arms':['baseline','adaptive'],'model':adaptation.MODEL,
      'motivation':'V5 complete edit ranker failed both held folds. Test improvement in input geometry instead of another possession-label continuation or threshold sweep.',
      'raw_coordinate_adaptation':'Each corrupted half independently. Three deterministic pseudo-occlusion views. Targets only coordinates actually observed in the corrupted input, every4 frames, no possession/status labels. Gap lengths25/50/100/150/200 inside observed runs, block2500 with250-frame margins. No originally missing coordinate can be a target.',
      'calibration':'Two complementary blocked coordinate fits per axis. All pseudo views of a frame remain in the same temporal block fold. Original/local residuals clipped±1.25 (25m); analytic convex alpha in[0,1] minimizes cross-block coordinate MSE. No F1 search, WPY fitting, match-ID feature or target labels. Fit final local residuals on all visible pseudo targets only if alpha>0.',
      'pipeline':'All original full baseline weights, temperatures, status, owner compositions, raw-global.5 gap mixture,747-credit calibrated mixture, expanded proposals and fixed.4/.01 decoder retained. Substitute only original imputer with per-view coordinate-CV blend. Corrected geometry affects learned interpolation owner, GRU and physical proposals through the unchanged full raw prediction method. Raw missing flags remain missing. All learned ownership/status weights unchanged.',
      'lock':'Clean exact .770846 incumbent bypass. Corrupted D/X must replay exactly. Corrupted H/A can change outside old gap fence due to complete full-model geometric context.',
      'baseline_verification':'Before each fold adaptive inference, perform one complete original-imputer raw replay through the entire .770846 recipe and compare all discrete baseline arrays and candidate labels bitwise. Alpha0 directly copies incumbent. All other baseline arrays are immutable pinned incumbent outputs.',
      'role':'Original primary weights TRAIN2 per fold/TRAIN4 full; local coordinate weights are transductive per-half raw-input adaptation without primary-label access, explicitly allowed and reported. No hidden clean counterpart positions enter an adaptation function.',
      'selection':'All16 paired views sealed before one OOF full-match audit. Require strict overall/corrupted improvement and nonnegative bothfold deltas. One selected new WQQ family test only; V5 failed and received no full test. No unselected fallback.',
      'limits':'Reused development, no private challenge accuracy claim. Coordinate CV improvement need not improve ownership SpellF1. .9 overall requires.994437 corrupted with fixed clean; no promise of that target.',
      'sources':{rel(p):sha256(p) for p in sources},'inputs':{**manifest['frozen_files'],rel(ROOT/'configs/development_best.json'):sha256(ROOT/'configs/development_best.json'),rel(ROOT/pointer['manifest']['path']):pointer['manifest']['sha256'],rel(previous.RUN/'oof-audit.json'):sha256(previous.RUN/'oof-audit.json')}})
    print('V6 PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    global HASH_ONLY
    p=read(RUN/'protocol.json');HASH_ONLY=True
    try:
        for path,h in {**p['sources'],**p['inputs']}.items():assert sha256(ROOT/path)==h,path
    finally:HASH_ONLY=False
    return p


def load(which):
    spec=read(OLD/'full-reference-spec.json' if which=='full' else OLD/f'specs/fold{which}.json');pipeline=ExplicitRawPipeline(spec)
    item=read(OLD/'full-transfer/protocol.json')['expert'] if which=='full' else read(OLD/'experiment-recipe.json')['expert_models'][str(which)]
    with check(item).open('rb') as f:expert=pickle.load(f)
    model,freeze=b.fitted(which);cal=read(e.RUN/f'models/{which}/calibration.json');assert cal['model']==freeze['model']
    return pipeline,expert,model,cal


def composed(raw,pipeline,expert,credit,cal):
    base=pipeline.predict(raw);z=global_owner(expert,global_features(raw['players'],raw['team'],raw['pitch']));g=gap_candidate(raw,base,z,pipeline.quality);parent={**base,**g};arcs=b.dense.proposals(g['predicted'],g['eligible_mask'],g['segments']);qo=b.old_quality(pipeline.quality,g['signals'],arcs);qr=e.EvidenceIntervals(raw,e.component(g['signals'],base)).predict(credit,arcs);q=(1-cal['alpha'])*qo+cal['alpha']*qr
    predicted=gs.decode(g['predicted'],g['eligible_mask'],arcs,q,.4,g['signals'][:,0],.01)
    return predicted,base


def replay(which,key,raw,models):
    predicted,base=composed(raw,*models);m,h,c=key.split('-');expected=previous.incumbent(m,h,c);np.testing.assert_array_equal(predicted,expected)
    folder=OLD/'full-transfer/full_baseline' if which=='full' else OLD/f'baseline/fold{which}'
    with np.load(folder/(key+'.npz'),allow_pickle=False) as z:
        for name in ['predicted','status_mask','sprint_predicted']:np.testing.assert_array_equal(base[name],z[name])
    return {'key':key,'current_best_exact':True,'original_discrete_fields_exact':True,'frames':len(predicted),'sources':pin(folder/(key+'.npz'))}


def prediction(which,key,models,do_replay=False):
    m,h,c=key.split('-');baseline=previous.incumbent(m,h,c)
    if c=='clean':return {'baseline':baseline.copy(),'adaptive':baseline.copy()},{'clean_exact':True,'changed':0}
    raw=b.raw_half(key);pipeline,expert,credit,cal=models;original=pipeline.imputers
    proof=replay(which,key,raw,models) if do_replay else None
    local,stats=adaptation.adapt(raw,original);path=RUN/f'adaptation/{which}'/(key+'.pkl');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:pickle.dump(local,f)
    try:
        pipeline.imputers=local
        predicted=baseline.copy() if stats['alpha']==0 else composed(raw,pipeline,expert,credit,cal)[0]
    finally:pipeline.imputers=original
    fixed=baseline>=2;np.testing.assert_array_equal(predicted[fixed],baseline[fixed]);assert np.array_equal(predicted>=2,baseline>=2)
    return {'baseline':baseline.copy(),'adaptive':predicted},{'adaptation':stats,'local_model':pin(path),'replay':proof,'changed':int((predicted!=baseline).sum()),'D_X_exact':True}


def predict_oof():
    verify();ALLOWED.clear();outputs={};views={};start=time.monotonic()
    for fold,(_,held) in enumerate(FOLDS):
        models=load(str(fold));replayed=False
        for key in b.keys(held):
            do_replay=not replayed and key.endswith('-corrupted');v,meta=prediction(str(fold),key,models,do_replay);replayed|=do_replay;path=RUN/f'oof/fold{fold}'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);views[key]=meta;print('V6 OOF',key,'changed',meta['changed'],'adaptation',meta.get('adaptation',{}).get('alpha'),'seconds',time.monotonic()-start,flush=True)
    assert not READS;write(RUN/'oof-seal.json',{'outputs':outputs,'views':views,'protocol_sha256':sha256(RUN/'protocol.json'),'primary_reference_reads':READS.copy(),'role':'Held raw observed-coordinate adaptation only; no held primary labels'})


def validate():
    verify();seal=read(RUN/'oof-seal.json');assert len(seal['outputs'])==16 and not seal['primary_reference_reads'];loaded={}
    for fold,(_,held) in enumerate(FOLDS):
        assert sum(v.get('replay') is not None for k,v in seal['views'].items() if k.split('-')[0] in held)==1
        for key in b.keys(held):
            path=RUN/f'oof/fold{fold}'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            m,h,c=key.split('-');np.testing.assert_array_equal(v['baseline'],previous.incumbent(m,h,c));fixed=v['baseline']>=2;np.testing.assert_array_equal(v['adaptive'][fixed],v['baseline'][fixed]);assert np.array_equal(v['adaptive']>=2,fixed)
            if c=='clean':np.testing.assert_array_equal(v['adaptive'],v['baseline'])
            else:
                meta=seal['views'][key];check(meta['local_model']);a=meta['adaptation'];assert a['input_only_observed_targets'] and a.get('original_missing_targets',0)==0
            loaded[key]=v
    return seal,loaded


def audit_oof():
    if (RUN/'oof-audit.json').exists():raise FileExistsError('One OOF audit only')
    seal,loaded=validate();ALLOWED.update(TRAIN);pm=official_metrics();details={};arms=['baseline','adaptive']
    for m in TRAIN:
        y=join_halves([truth(m,h) for h in ['firstHalf','secondHalf']]);details[m]={c:{a:score_full(pm,y,join_halves([loaded[f'{m}-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in arms} for c in ['clean','corrupted']}
    avg=lambda ms,cs:{a:float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in arms};overall=avg(TRAIN,['clean','corrupted']);conditions={c:avg(TRAIN,[c]) for c in ['clean','corrupted']};folds={str(f):avg(h,['clean','corrupted']) for f,(_,h) in enumerate(FOLDS)};assert abs(overall['baseline']-.7099817195618008)<1e-12
    passed=overall['adaptive']>overall['baseline'] and conditions['corrupted']['adaptive']>conditions['corrupted']['baseline'] and all(v['adaptive']>=v['baseline'] for v in folds.values());result={'overall':overall,'conditions':conditions,'per_fold':folds,'per_match':{m:avg([m],['clean','corrupted']) for m in TRAIN},'details':details,'selected':'adaptive' if passed else 'baseline','passed':passed,'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'oof-seal.json'),'reference_reads':READS.copy(),'interpretation':'Repeated development. Coordinate adaptation uses only held raw observed coordinates; no held primary-label fitting. No WQQ candidate scored.'};write(RUN/'oof-audit.json',result);print(json.dumps({k:result[k] for k in ['overall','conditions','per_fold','selected','passed']}),flush=True)


def authorize():
    global FULL
    validate();a=read(RUN/'oof-audit.json');check(a['prediction_seal']);assert a['passed'] and a['selected']=='adaptive';assert not (previous.RUN/'full-audit.json').exists();FULL=True


def predict_full():
    authorize();ALLOWED.clear();models=load('full');outputs={};views={};replayed=False;write(RUN/'full-selection.json',{'selected':'adaptive','OOF_audit':pin(RUN/'oof-audit.json'),'no_fallback':True,'raw_coordinate_adaptation_only':True})
    for key in b.keys(['J03WQQ']):
        do_replay=not replayed and key.endswith('-corrupted');v,meta=prediction('full',key,models,do_replay);replayed|=do_replay;path=RUN/'full-predictions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**v);outputs[rel(path)]=sha256(path);views[key]=meta;print('V6 FULL',key,'changed',meta['changed'],'adaptation',meta.get('adaptation',{}).get('alpha'),flush=True)
    assert not READS;write(RUN/'full-seal.json',{'outputs':outputs,'views':views,'selection':pin(RUN/'full-selection.json'),'primary_reference_reads':READS.copy()})


def audit_full():
    if (RUN/'full-audit.json').exists():raise FileExistsError('One full audit only')
    authorize();seal=read(RUN/'full-seal.json');check(seal['selection']);assert len(seal['outputs'])==4 and not seal['primary_reference_reads'];loaded={}
    for key in b.keys(['J03WQQ']):
        path=RUN/'full-predictions'/(key+'.npz');assert sha256(path)==seal['outputs'][rel(path)]
        with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        m,h,c=key.split('-');np.testing.assert_array_equal(v['baseline'],previous.incumbent(m,h,c));fixed=v['baseline']>=2;np.testing.assert_array_equal(v['adaptive'][fixed],v['baseline'][fixed]);assert np.array_equal(v['adaptive']>=2,fixed)
        if c=='clean':np.testing.assert_array_equal(v['adaptive'],v['baseline'])
        loaded[key]=v
    ALLOWED.add('J03WQQ');pm=official_metrics();y=join_halves([truth('J03WQQ',h) for h in ['firstHalf','secondHalf']]);conditions={c:{a:score_full(pm,y,join_halves([loaded[f'J03WQQ-{h}-{c}'][a] for h in ['firstHalf','secondHalf']]))[0] for a in ['baseline','adaptive']} for c in ['clean','corrupted']};overall={a:float(np.mean([conditions[c][a]['spell_f1_mean'] for c in ['clean','corrupted']])) for a in ['baseline','adaptive']};assert abs(overall['baseline']-.7708462421456541)<1e-12
    result={'overall':overall,'conditions':conditions,'selected':'adaptive','improved':overall['adaptive']>overall['baseline'],'clean_exact':True,'D_X_exact':True,'prediction_seal':pin(RUN/'full-seal.json'),'reference_reads':READS.copy(),'interpretation':'Single reused full development match. Raw-coordinate adaptation without held primary supervision; not private challenge accuracy.'};write(RUN/'full-audit.json',result);print(json.dumps({'overall':overall,'improved':result['improved']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict_oof','audit_oof','predict_full','audit_full']);args=p.parse_args();STAGE=args.stage;threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(guard);globals()[args.stage]()
