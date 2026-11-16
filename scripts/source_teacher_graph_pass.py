"""One TRAIN5 paired source-teacher graph pass; sealed OOF before scoring."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from second_zero.common import sha256,json_write
from second_zero.source_control_teacher import SourceControlTeacher,TeacherConfig
from second_zero import source_teacher_graph as adapter
from scripts import anchor_export_reduced_crossfit as cf
from scripts import joint_controller_ball_crossfit as joint
from scripts.sprint_evaluate import score
from scripts.anchor_refine_audit import event_change
from second_zero.evaluate import official_metrics

RUN=ROOT/'runs/aggressive-20261029/source-teacher-graph-v1'
TEACHER=ROOT/'runs/aggressive-20261029/source-teacher-v1'
BASELINE=ROOT/'runs/joint-controller-ball/base-transfer-v1'
PRIMARY5=['J03WMX','J03WN1','J03WOH','J03WOY','J03WPY']
ARMS=('baseline','common52','teacher92')
READS=[];ALLOWED=[];INPUTS={}
MODEL=dict(max_iter=180,max_leaf_nodes=15,max_depth=5,min_samples_leaf=40,learning_rate=.05,l2_regularization=5.,random_state=20261024,early_stopping=False)


def guard(event,args):
    if event!='open'or not isinstance(args[0],(str,bytes,os.PathLike)):return
    text=os.fsdecode(args[0])
    if any(v in text for v in ('J03WQQ','J03WR9','/evaluation/gt/','/runs/main/')):raise PermissionError('WQQ/WR9/challenge closed throughout pass')
    if text.endswith('.truth.npy'):
        if not any(Path(text).name.startswith(m+'-')for m in ALLOWED):raise PermissionError('Held truth forbidden until paired seal: '+text)
        READS.append(text)


def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def pin(p,expected=None):
    p=Path(p).resolve();digest=sha256(p)
    if expected is not None and digest!=expected:raise ValueError('Pinned artifact changed: '+rel(p))
    INPUTS[rel(p)]=digest;return p

def meta(p,expected=None):return json.loads(pin(p,expected).read_text())
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def code():
    return [Path(__file__),Path(adapter.__file__),Path(cf.__file__),Path(cf.graph.__file__),Path(cf.lattice.__file__),
        Path(joint.__file__),ROOT/'second_zero/joint_controller_ball.py',ROOT/'second_zero/source_control_teacher.py',
        ROOT/'second_zero/aggressive_controller.py',ROOT/'scripts/sprint_evaluate.py',ROOT/'scripts/anchor_refine_audit.py',
        ROOT/'vendor/mtah/evaluation/possession_metrics.py',ROOT/'scripts/train_contextual.py']

def validate():
    p=meta(RUN/'protocol.json')
    for path,digest in p['source_sha256'].items():pin(ROOT/path,digest)
    for path,digest in p['ancestor_sha256'].items():pin(ROOT/path,digest)
    return p


def prepare(args):
    RUN.mkdir(parents=True,exist_ok=False)
    teacher=meta(TEACHER/'protocol.json')
    assert teacher['source_train_matches']==list(joint_train16()) and teacher['recipe']['steps']==600
    ancestors={};folds={}
    for fold,(train,held)in enumerate(cf.FOLDS):
        folder=BASELINE/f'fold{fold}';f=meta(folder/'freeze.json');inferred=meta(folder/'inference.json')
        assert f['train_matches']==train and f['heldout_matches']==held and not inferred['heldout_truth_read']
        assert (f['weight'],f['scope'],f['graph_penalty'],f['interval_weight'])==(.5,'all',.45,0.)
        fitted=meta(Path(f['model_path'])/'freeze-fit.json');assert set(fitted['train_matches'])==set(train)
        pin(Path(f['model_path'])/'model.pt',fitted['model_sha256'])
        protocol=meta(Path(f['model_path']).parent/'protocol.json',fitted['protocol_sha256']);assert protocol['train_matches']==train
        for path,digest in protocol['sources'].items():pin(ROOT/path,digest)
        quality=cf.ROOT/f'runs/anchor-refinement/reduced-crossfit-v1/fold{fold}'
        qp=meta(quality/'protocol.json');qf=meta(quality/'freeze.json')
        assert qp['train_matches']==train and qp['heldout_matches']==held
        pin(quality/'model.pkl',qf['model_sha256']);assert qf['model_sha256']==f['base_quality_sha256']
        for variant in ('interp_learned_swap','long'):
            folder_model=cf.SPRINT/f'model/{variant}-{fold}';mp=meta(folder_model/'protocol.json');assert mp['train_matches']==train
            pin(folder_model/'model.pt');meta(folder_model/'calibration.json')
        for folder_model,filename,train_key in [(cf.SPRINT/f'ball-imputer-fold{fold}','models.pkl','train_matches'),(cf.SPRINT/f'status-crossfit-v2/fold{fold}','model.pkl','training_matches')]:
            mp=meta(folder_model/('protocol.json'if train_key=='train_matches'else'freeze.json'));assert mp[train_key]==train
            pin(folder_model/filename)
        pin(ROOT/f'runs/overnight-boundary-v1/crossfit/fold{fold}-models.pkl')
        legacy=meta(ROOT/'runs/overnight-boundary-v1/crossfit_manifest.json')
        assert legacy['whole_match_disjoint']
        assert all(legacy['records'][key]['model_train_matches']==train for key in cf.graph.keys(held))
        # Existing receipts audit the legacy same-fold ancestry; never load a full-fit model.
        meta(quality/'completion.json')
        folds[str(fold)]={'ancestor_train':train,'fit':train+['J03WPY'],'held':held,'baseline_freeze':rel(BASELINE/f'fold{fold}/freeze.json'),'baseline_inference':rel(BASELINE/f'fold{fold}/inference.json')}
    ancestors=dict(INPUTS)
    p={'kind':'source-teacher-TRAIN5-matched-new-arcs-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'folds':folds,'primary_training_matches':PRIMARY5,'WPY_permanently_retired_from_validation_for_this_lineage':True,
       'teacher_protocol_sha256':sha256(TEACHER/'protocol.json'),'teacher_final_step':600,'teacher_source_only':True,
       'arms':list(ARMS),'new_head_parameters':MODEL,'new_arc_row_limit_per_view':20000,'sample_seed':20261105,
       'old_graph':'Same-fold retained joint blend weight .5/all, frozen WPY temperature, TRAIN2 quality. Old segments/quality/unary/status/penalty .45/frame .01 preserved exactly; duplicates retain old score.',
       'training_features':'All learned ancestors exclude held side. TRAIN2 ancestors may be reused on their own fit side and WPY. These fitting features are in-sample; no opposite-fold training cache is reused.',
       'proposals':'DE team logodds smoothed3, cost3 path with both tie conventions; <=250-frame arcs, +/-2 variants. At most2 PP maxima/team/retained graph spell, separation3, include [peak,peak+1). 16-node <=250 both-team connections. Background arcs follow retained baseline spells, start/end and32neighbor <=1250 connections. All old arcs remain.',
       'feature_contract':{'common52':'old quality1+duration3+physical6 mean/std and before/after9,31,125; first3 physical channels candidate-signed','teacher40':'8 channels mean/std before/after9 (32), plus8 +/-2 boundary summaries; first3 signed; no hidden embeddings/IDs/time/match'},
       'teacher_frame_channels':list(adapter.CHANNELS),'teacher_inference':'FP32, chunk2000, context200, global roster, input-only masks; no status or ownership inputs; no-node neutral zeros and explicit availability.',
       'sampler':'20000/view provenance x duration strata, inverse inclusion weights; same indices and targets for both arms, no credit-dependent row sampling.',
       'selection':'Overall equal-view official SpellF1 first, macroF1 exact ties only. Both folds and both arms sealed before any held scoring. One paired audit; no WQQ/WR9/final/full fit.',
       'budget':'One bounded Mac pass; source teacher profile <20min; checkpoint on projected runtime beyond remaining roughly60min budget.',
       'source_sha256':{rel(p):sha256(p)for p in code()},'ancestor_sha256':ancestors,
       'WQQ_access':False,'WR9_access':False,'held_truth_opened':False,'metrics_computed':False,
       'adaptive_development':'Four repeatedly used physical matches, paired clean/corrupt halves correlated. No independent test claim.'}
    write(RUN/'protocol.json',p)
    for name in ('teacher','cache','heads','predictions'):(RUN/name).mkdir()
    print('FROZEN TRAIN5',sha256(RUN/'protocol.json'),flush=True)


def joint_train16():
    from scripts.source_control_teacher import TRAIN
    return TRAIN


def extract(args):
    protocol=validate();freeze=meta(TEACHER/'freeze.json');pin(TEACHER/'model.pt',freeze['model_sha256'])
    assert freeze['protocol_sha256']==protocol['teacher_protocol_sha256'] and freeze['step']==600
    ck=torch.load(TEACHER/'model.pt',weights_only=True,map_location='cpu');model=SourceControlTeacher(TeacherConfig(**ck['architecture'])).to(args.device);model.load_state_dict(ck['state_dict']);model.eval()
    started=time.monotonic();outputs={};timings={}
    for key in cf.graph.keys(PRIMARY5):
        path=RUN/'teacher'/(key+'.npz')
        if path.exists():raise FileExistsError(path)
        before=time.monotonic();raw=cf.raw_data(key);values=adapter.infer(model,raw,device=args.device)
        np.savez_compressed(path,**values);outputs[rel(path)]=sha256(path);timings[key]=time.monotonic()-before
        print('TEACHER FEATURES',key,round(timings[key],2),flush=True)
    write(RUN/'teacher-freeze.json',{'model_sha256':freeze['model_sha256'],'teacher_freeze_sha256':sha256(TEACHER/'freeze.json'),'protocol_sha256':sha256(RUN/'protocol.json'),'outputs':outputs,'timings':timings,'elapsed_seconds':time.monotonic()-started,'input_sha256':dict(cf.HASHES),'reference_reads':READS.copy(),'source_only':True})


def quality_predict(model,x,segments):
    return np.concatenate([np.clip(model.predict(cf.graph.segment_features(x,segments[i:i+20000])),0,1)for i in range(0,len(segments),20000)])if len(segments)else np.zeros(0)


def cache(args):
    protocol=validate();tf=meta(RUN/'teacher-freeze.json');assert tf['protocol_sha256']==sha256(RUN/'protocol.json')
    started=time.monotonic();timings={};processed=0
    for fold,(train,held)in enumerate(cf.FOLDS):
        folder=RUN/'cache'/f'fold{fold}';folder.mkdir(exist_ok=True)
        declared=protocol['folds'][str(fold)];f=meta(ROOT/declared['baseline_freeze']);inferred=meta(ROOT/declared['baseline_inference'])
        engine=cf.FoldEvidence(fold,train,held)
        with pin(joint.PRIOR/f'fold{fold}/model.pkl',f['base_quality_sha256']).open('rb')as h:quality=pickle.load(h)
        model=joint.load_model(Path(f['model_path']),train,meta(BASELINE/'recipe.json')).to('cpu')
        # A held view then a training view give a useful fresh preparation profile.
        held_keys=cf.graph.keys(held);train_keys=cf.graph.keys(declared['fit']);keys=[held_keys[0],train_keys[0]]+held_keys[1:]+train_keys[1:]
        evidence=ROOT/f'runs/joint-controller-ball/transfer-inputs-v1/fold{fold}';ef=meta(evidence/'freeze.json')
        for key in keys:
            path=folder/(key+'.npz')
            if path.exists():continue
            before=time.monotonic();is_held=key.split('-')[0]in held
            if is_held:
                with np.load(pin(evidence/(key+'.npz'),ef['outputs'][key+'.npz']))as z:x=z['x'];base=z['base']
                neural_path=BASELINE/f'fold{fold}/raw'/(key+'.npz')
                with np.load(pin(neural_path,inferred['raw_outputs'][key+'.npz']))as z:neural=z['owner']
            else:
                x,base,_=engine.signals(key)
                neural_path=BASELINE/f'fold{fold}/raw'/(key+'.npz')
                if neural_path.exists():
                    assert key.startswith('J03WPY-')
                    with np.load(pin(neural_path,inferred['raw_outputs'][key+'.npz']))as z:neural=z['owner']
                else:neural=joint.infer_raw(model,cf.raw_data(key))['owner']
            nx=joint.blend_evidence(x,neural,None,f['temperature'],.5,'all');segments=cf.pool(nx,base)
            q=quality_predict(quality,nx,segments);baseline=cf.graph_prediction(nx,base,segments,q,.45)
            if is_held:
                target=BASELINE/f'fold{fold}/candidate'/(key+'.npz')
                with np.load(pin(target,inferred['outputs']['candidate/'+key+'.npz']))as z:
                    np.testing.assert_array_equal(baseline,z['predicted']);np.testing.assert_array_equal(base,z['status_mask'])
            teacher_path=RUN/'teacher'/(key+'.npz')
            with np.load(pin(teacher_path,tf['outputs'][rel(teacher_path)]))as z:teacher={k:z[k]for k in z.files}
            added,origin=adapter.proposals(base,baseline,segments,teacher)
            added_q=quality_predict(quality,nx,added);common,treatment=adapter.features(nx,added,added_q,teacher)
            np.savez_compressed(path,x=nx,base=base,baseline=baseline,old_segments=segments,old_quality=q,new_segments=added,origin=origin,common52=common,teacher92=treatment)
            timings[key]={'seconds':time.monotonic()-before,'held':is_held,'old_arcs':len(segments),'new_arcs':len(added),'baseline_exact_replay':is_held}
            write(path.with_suffix('.json'),{'fold':fold,'key':key,'protocol_sha256':sha256(RUN/'protocol.json'),'teacher_freeze_sha256':sha256(RUN/'teacher-freeze.json'),'cache_sha256':sha256(path),'timing':timings[key],'input_sha256':{**INPUTS,**cf.HASHES},'reference_reads':READS.copy(),'held_matches':held,'ancestor_train_matches':train})
            print('GRAPH CACHE',fold,key,timings[key],flush=True);processed+=1
            if args.max_views and processed>=args.max_views:
                write(RUN/'cache-profile.json',{'timings':timings,'elapsed_seconds':time.monotonic()-started,'remaining_views':40-sum(1 for _ in (RUN/'cache').glob('fold*/*.npz')),'metrics_computed':False,'reference_reads':READS.copy()});return
    write(RUN/'cache-freeze.json',{'protocol_sha256':sha256(RUN/'protocol.json'),'outputs':{rel(p):sha256(p)for p in (RUN/'cache').glob('fold*/*.npz')},'receipts':{rel(p):sha256(p)for p in (RUN/'cache').glob('fold*/*.json')},'elapsed_seconds':time.monotonic()-started,'reference_reads':READS.copy(),'held_baselines_exact_replayed':True})


def fit_predict(args):
    protocol=validate();frozen=meta(RUN/'cache-freeze.json');assert len(frozen['outputs'])==40
    started=time.monotonic();rng=np.random.default_rng(protocol['sample_seed']);outputs={};fitting={}
    for fold,(train,held)in enumerate(cf.FOLDS):
        ALLOWED[:]=protocol['folds'][str(fold)]['fit'];xx={arm:[]for arm in ARMS[1:]};yy=[];weights=[];rows={}
        for key in cf.graph.keys(ALLOWED):
            path=RUN/'cache'/f'fold{fold}'/(key+'.npz')
            with np.load(pin(path,frozen['outputs'][rel(path)]))as z:
                indices,w=adapter.sample_indices(z['new_segments'],z['origin'],rng)
                y=cf.graph.quality_targets(cf.truth(key),z['new_segments'][indices])[0]
                for arm in ARMS[1:]:xx[arm].append(z[arm][indices])
                yy.append(y);weights.append(w);rows[key]={'all':len(z['new_segments']),'sampled':len(indices),'index_sha256':__import__('hashlib').sha256(indices.tobytes()).hexdigest()}
        yy=np.concatenate(yy);weights=np.concatenate(weights);heads={}
        for arm in ARMS[1:]:
            heads[arm]=HistGradientBoostingRegressor(**MODEL).fit(np.concatenate(xx[arm]),yy,sample_weight=weights)
            target=RUN/'heads'/f'fold{fold}-{arm}.pkl'
            with target.open('xb')as h:pickle.dump(heads[arm],h)
        fitting[str(fold)]={'rows':rows,'fit_matches':list(ALLOWED),'held':held,'heads':{arm:sha256(RUN/'heads'/f'fold{fold}-{arm}.pkl')for arm in ARMS[1:]},'reference_reads':READS.copy()}
        ALLOWED[:]=[]
        for key in cf.graph.keys(held):
            path=RUN/'cache'/f'fold{fold}'/(key+'.npz')
            with np.load(pin(path,frozen['outputs'][rel(path)]))as z:
                old=z['old_segments'];new=z['new_segments'];pool=np.r_[old,new];predictions={'baseline':z['baseline']}
                for arm,head in heads.items():
                    q=np.r_[z['old_quality'],np.clip(head.predict(z[arm]),0,1)]
                    predictions[arm]=cf.graph_prediction(z['x'],z['base'],pool,q,.45)
                for pred in predictions.values():np.testing.assert_array_equal(pred[z['base']>=2],z['base'][z['base']>=2])
                target=RUN/'predictions'/(key+'.npz');np.savez_compressed(target,**predictions,status_mask=z['base']);outputs[rel(target)]=sha256(target)
            print('PAIRED PREDICTED',fold,key,flush=True)
    write(RUN/'prediction-seal.json',{'kind':'paired-TRAIN5-OOF-seal','protocol_sha256':sha256(RUN/'protocol.json'),'cache_freeze_sha256':sha256(RUN/'cache-freeze.json'),'teacher_freeze_sha256':sha256(RUN/'teacher-freeze.json'),'outputs':outputs,'fitting':fitting,'held_truth_opened':False,'reference_reads':READS.copy(),'elapsed_seconds':time.monotonic()-started})


def audit(args):
    protocol=validate();seal=meta(RUN/'prediction-seal.json');assert seal['protocol_sha256']==sha256(RUN/'protocol.json')and len(seal['outputs'])==16
    for path,digest in seal['outputs'].items():pin(ROOT/path,digest)
    ALLOWED[:]=[m for _,held in cf.FOLDS for m in held];pm=official_metrics();views={}
    for fold,(_,held)in enumerate(cf.FOLDS):
        for key in cf.graph.keys(held):
            with np.load(RUN/'predictions'/(key+'.npz'))as z:pred={arm:z[arm]for arm in ARMS}
            y=cf.truth(key);scored={arm:score(pm,y,pred[arm],True)for arm in ARMS}
            with np.load(RUN/'cache'/f'fold{fold}'/(key+'.npz'))as z:old=z['old_segments'];added=z['new_segments']
            refs=pm.possession_spells(y,pm.valid_runs(y));short=np.array([e-s<=125 for s,e,t in refs],bool);very_short=np.array([e-s<=25 for s,e,t in refs],bool)
            coverage={}
            for label,pool in [('old',old),('augmented',np.r_[old,added])]:
                best=np.zeros(len(refs))
                for iou,gi,pi in pm.spell_overlaps(refs,pool.tolist()):best[gi]=max(best[gi],iou)
                coverage[label]={str(t):{'short':int(((best>=t)&short).sum()),'very_short25':int(((best>=t)&very_short).sum())}for t in pm.SPELL_IOUS}
            views[key]={'metrics':{a:r['metrics']for a,r in scored.items()},'event_change':{arm:event_change(scored['baseline'],scored[arm])for arm in ARMS[1:]},'candidate_coverage':coverage,'short_references':int(short.sum()),'very_short25_references':int(very_short.sum()),'old_arcs':len(old),'new_arcs':len(added),'D_X_unchanged':True}
    def means(items):
        return {a:{m:float(np.mean([r['metrics'][a][m]for r in items]))for m in ('spell_f1_mean','macro_f1')}for a in ARMS}
    groups={name:means([v for k,v in views.items()if name=='overall'or k.endswith(name)])for name in ('overall','clean','corrupted')}
    matches={m:means([v for k,v in views.items()if k.startswith(m+'-')])for m in ALLOWED}
    rank=lambda a:(groups['overall'][a]['spell_f1_mean'],groups['overall'][a]['macro_f1'])
    selected=max(ARMS,key=rank);decision='keep'if selected!='baseline'else'reject'
    report={'kind':'paired-source-teacher-TRAIN5-OOF-audit','protocol_sha256':sha256(RUN/'protocol.json'),'prediction_seal_sha256':sha256(RUN/'prediction-seal.json'),'groups':groups,'per_match':matches,'selected':selected,'decision':decision,'teacher_vs_common52_delta':groups['overall']['teacher92']['spell_f1_mean']-groups['overall']['common52']['spell_f1_mean'],'views':views,'held_reference_reads':READS,'WQQ_access':False,'WR9_access':False,'submission_changed':False,'limitation':protocol['adaptive_development'],'interpretation':'Teacher92 vs common52 isolates source feature bundle with matched fitting rows/candidates. Either vs baseline also includes proposal and WPY-training effects. Keep means eligible for a separately authorized future step; this pass does not promote externally.'}
    write(RUN/'OOF-audit.json',report);print('PAIRED OOF',json.dumps({'groups':groups,'selected':selected,'decision':decision}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','extract','cache','fit_predict','audit']);parser.add_argument('--device',choices=['cpu','mps'],default='mps');parser.add_argument('--max-views',type=int,default=0);args=parser.parse_args()
    sys.addaudithook(guard);torch.set_num_threads(2);threadpool_limits(2);globals()[args.stage](args)
