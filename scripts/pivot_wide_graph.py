"""Prospective integration of split-safe wide context into the teacher92 graph.

No fitting or reference reads. Two inherited small blend choices are declared
before predictions; a zero-strength replay verifies the actual strongest graph.
"""
from __future__ import annotations
import argparse,datetime,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from second_zero.common import sha256
from scripts import source_teacher_graph_pass as source
from scripts import anchor_export_reduced_crossfit as cf
from second_zero import source_teacher_graph as adapter

RUN=ROOT/'runs/context-pivot-20261029/wide-graph'
REPORT=ROOT/'reports/context-pivot-20261029/wide-graph'
WIDE=ROOT/'runs/aggressive-20261029/crossfit'
CHOICES={'wide25':.25,'wide50':.5}
INPUTS={}


def guard(event,args):
    if event!='open'or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(s in p for s in ('J03WQQ','J03WR9','/evaluation/gt/'))or p.endswith('.truth.npy'):
        raise PermissionError('Producer cannot read held/competition references')
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def pin(p,expected=None):
    p=Path(p);digest=sha256(p)
    if expected is not None and digest!=expected:raise ValueError('Changed input:'+rel(p))
    INPUTS[rel(p)]=digest;return p
def read(p):return json.loads(pin(p).read_text())
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as h:json.dump(value,h,indent=2);h.write('\n')
def sources():return [Path(__file__),Path(source.__file__),Path(cf.__file__),Path(cf.graph.__file__),Path(cf.lattice.__file__),Path(adapter.__file__)]


def prepare():
    RUN.mkdir(parents=True,exist_ok=False);REPORT.mkdir(parents=True,exist_ok=False)
    source.validate();frozen=read(source.RUN/'prediction-seal.json');wide=read(WIDE/'protocol.json')
    cal_protocol=read(WIDE/'fold-specific-temperature/protocol.json');assert cal_protocol['WQQ_access']is False
    for fold,(train,held)in enumerate(cf.FOLDS):
        fit=read(WIDE/f'fold{fold}/fit-freeze.json');assert fit['train_matches']==train and fit['heldout_matches']==held and fit['random_initialization']
        pin(WIDE/f'fold{fold}/model.pt',fit['model_sha256'])
        calibration=read(WIDE/f'fold-specific-temperature/fold{fold}/calibration-freeze.json')
        assert calibration['model_sha256']==fit['model_sha256']and not calibration['held_side_truth_read']
        predictions=read(WIDE/f'fold{fold}/prediction-freeze.json')
        for filename,digest in predictions['outputs'].items():pin(WIDE/f'fold{fold}/predictions'/filename,digest)
    for path,digest in frozen['outputs'].items():pin(ROOT/path,digest)
    p={'kind':'frozen-wide-context-to-strong-graph-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'folds':[{'train':t,'held':h,'existing_head_fit':t+['J03WPY']}for t,h in cf.FOLDS],
        'candidates':CHOICES,'fitting':'None. Existing random-initialized2400-step fold models and frozen fold-specific WPY NLL temperatures.',
        'inputs':'teacher92 retained joint-mixture x; source-only teacher outputs; existing wide native122-channel geometry model.',
        'integration':'Probability blend x[:,0] with wide output/Tfold at fixed25%or50%; all ancillary evidence/status fixed. Recompute old proposals, old quality, source additions and teacher92 qualities consistently.',
        'zero_strength':'Must exactly replay all16 saved teacher92 outputs before sealing.',
        'selection':'Full-match official SpellF1, equal physical-match/clean-corrupt view mean. Macro exact ties only, unchanged baseline included. Cross-family winner before any WQQ audit; no fallback after its audit.',
        'sources':{rel(p):sha256(p)for p in sources()},'inputs_sha256':INPUTS.copy(),
        'WPY':'Already on fitting side, no new references read. Temperatures inherited from fold-only predictions.',
        'WQQ_access':False,'WR9_access':False,'held_truth_access':False,
        'limitations':'Previously used development matches, adaptive architecture history. Tests incremental existing wide evidence in current graph; no new training.'}
    write(RUN/'protocol.json',p);print('WIDE GRAPH FROZEN',sha256(RUN/'protocol.json'),flush=True)


def check():
    p=read(RUN/'protocol.json')
    for path,digest in p['sources'].items():pin(ROOT/path,digest)
    for path,digest in p['inputs_sha256'].items():pin(ROOT/path,digest)
    return p


def decode(x,base,quality,head,teacher):
    old=cf.pool(x,base);q=source.quality_predict(quality,x,old);baseline=cf.graph_prediction(x,base,old,q,.45)
    added,_=adapter.proposals(base,baseline,old,teacher)
    aq=source.quality_predict(quality,x,added);_,features=adapter.features(x,added,aq,teacher)
    pool=np.r_[old,added];q=np.r_[q,np.clip(head.predict(features),0,1)]
    return cf.graph_prediction(x,base,pool,q,.45)


def predict():
    protocol=check();cache=read(source.RUN/'cache-freeze.json');tf=read(source.RUN/'teacher-freeze.json')
    seals={};started=time.monotonic()
    for fold,(train,held)in enumerate(cf.FOLDS):
        cal=read(WIDE/f'fold-specific-temperature/fold{fold}/calibration-freeze.json');temperature=cal['temperature']
        with pin(source.joint.PRIOR/f'fold{fold}/model.pkl').open('rb')as h:quality=pickle.load(h)
        with pin(source.RUN/'heads'/f'fold{fold}-teacher92.pkl').open('rb')as h:head=pickle.load(h)
        for key in cf.graph.keys(held):
            before=time.monotonic();path=source.RUN/'cache'/f'fold{fold}'/(key+'.npz')
            with np.load(pin(path,cache['outputs'][rel(path)]))as z:x=z['x'];base=z['base']
            path=source.RUN/'teacher'/(key+'.npz')
            with np.load(pin(path,tf['outputs'][rel(path)]))as z:teacher={k:z[k]for k in z.files}
            with np.load(pin(WIDE/f'fold{fold}/predictions'/(key+'.npz')))as z:wide=z['raw_logit']/temperature
            zero=decode(x,base,quality,head,teacher)
            with np.load(source.RUN/'predictions'/(key+'.npz'))as z:np.testing.assert_array_equal(zero,z['teacher92'])
            for name,weight in CHOICES.items():
                nx=x.copy();nx[:,0]=logit(np.clip((1-weight)*expit(x[:,0])+weight*expit(wide),1e-7,1-1e-7))
                predicted=decode(nx,base,quality,head,teacher);np.testing.assert_array_equal(predicted[base>=2],base[base>=2])
                dest=RUN/name/f'fold{fold}/predictions'/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,predicted=predicted,logit=nx[:,0]);seals[rel(dest)]=sha256(dest)
            print('WIDE GRAPH PREDICTED',fold,key,round(time.monotonic()-before,2),'zero replay exact',flush=True)
    write(RUN/'producer-seal.json',{'protocol_sha256':sha256(RUN/'protocol.json'),'outputs':seals,'all16_zero_replays_exact':True,'held_truth_reads':[],
        'input_sha256':INPUTS,'seconds':time.monotonic()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','predict']);args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);sys.addaudithook(guard);globals()[args.stage]()
