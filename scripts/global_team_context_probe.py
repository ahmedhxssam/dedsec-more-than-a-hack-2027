"""Fixed raw-team context expert, sealed TRAIN-only crossfit before audit."""
from __future__ import annotations
import argparse,datetime,json,os,pickle,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from second_zero import global_team_context as context
from second_zero.common import sha256
from scripts import anchor_export_reduced_crossfit as cf
from scripts import source_teacher_graph_pass as source
from scripts import pivot_combination_inputs as combine
from scripts import pivot_wide_graph as wide
from scripts.train_contextual import corruption
from scripts.vision_context_memory_owner import sample_spells

RUN=ROOT/'runs/context-pivot-20261029/global-team-context-v1'
REPORT=ROOT/'reports/context-pivot-20261029/global-context'
MODEL=dict(max_iter=180,max_leaf_nodes=15,max_depth=5,min_samples_leaf=80,
    learning_rate=.05,l2_regularization=10.,random_state=20261029,early_stopping=False)
ALLOWED=[];READS=[];INPUTS={}


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(s in p for s in ('J03WQQ','J03WR9','/evaluation/gt/','/runs/main/')):raise PermissionError('Reserved inputs closed')
    if p.endswith('.truth.npy'):
        if not any(Path(p).name.startswith(m+'-') for m in ALLOWED):raise PermissionError('Truth outside fitting phase:'+p)
        READS.append(p)


def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def pin(p,expected=None):
    p=Path(p);digest=sha256(p)
    if expected is not None and digest!=expected:raise ValueError('Changed input:'+rel(p))
    INPUTS[rel(p)]=digest;return p
def read(p):return json.loads(pin(p).read_text())
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\n')
def codes():
    return [Path(__file__),Path(context.__file__),Path(cf.__file__),Path(source.__file__),
        Path(combine.__file__),Path(wide.__file__),Path(cf.graph.__file__),Path(cf.lattice.__file__),
        ROOT/'second_zero/source_teacher_graph.py',ROOT/'scripts/train_contextual.py',
        ROOT/'scripts/vision_context_memory_owner.py']


def raw(key):
    name,condition=key.rsplit('-',1)
    with np.load(pin(cf.DATA/(name+'.npz'))) as z:value={k:z[k] for k in ('ball','players','team','pitch')}
    return corruption(value,100) if condition=='corrupted' else value


def prepare():
    RUN.mkdir(parents=True,exist_ok=False)
    source.validate();wp=read(wide.RUN/'protocol.json');wf=read(wide.RUN/'producer-seal.json')
    assert wp['candidates']['wide50']==.5
    for path,digest in wf['outputs'].items():
        if '/wide50/' in path:pin(ROOT/path,digest)
    for fold,(train,held) in enumerate(cf.FOLDS):
        for path in [source.joint.PRIOR/f'fold{fold}/model.pkl',source.RUN/'heads'/f'fold{fold}-teacher92.pkl']:pin(path)
    for manifest in ['cache-freeze.json','teacher-freeze.json']:
        frozen=read(source.RUN/manifest)
        for path,digest in frozen['outputs'].items():pin(ROOT/path,digest)
    protocol={'kind':'global-team-context-expert-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'folds':[{'fit':train+['J03WPY'],'held':held} for train,held in cf.FOLDS],
        'feature_shape':[2,context.CHANNELS],'feature_names':context.names(),'model':MODEL,
        'sampling':'Native live frames stride8 plus full-spell midpoint; shared samples across conditions and perspectives.',
        'weights':'Half native-frame mass and half equal-full-spell mass. Mean spell length from all fitting matches/halves; normalize final row weights to mean1.',
        'target':'Binary own-perspective team on H/A frames; shared classifier. Output half(A logit minus H logit), each clipped probability1e-6..1-1e-6.',
        'temperature':1.,'candidates':{'global025':{'weight':.25,'scope':'all'},'global_missing025':{'weight':.25,'scope':'missing25'}},
        'parent':'Exact teacher92+wide50. Recompute graph old/new proposals and quality after probability blend, penalty.45/frame.01. Parent alias all16 replays exact.',
        'missing25':'uniform_filter1d(ball-missing float64,25,mode=nearest), fixed existing joint gate.',
        'selection':'No fitting-side model selection/calibration. Both folds/all16 candidates sealed before full-match official SpellF1 audit by root. Baseline included, noWQQ fallback.',
        'feature_ancestry':'No learned ancestors: observed positions/team/pitch, fixed corruption100, fixed physical scales. Partial-camera and anonymous adjacent-slot movement limitations disclosed.',
        'profile_disclosure':'TRAIN5 truth read only for descriptive sample counts before protocol; no fold held scores. Fit strictly TRAIN2+WPY; inference zero truth reads.',
        'WPY':'Permanently fitting-side. No held calibration or parameter selection.',
        'sources':{rel(p):sha256(p) for p in codes()},'inputs_sha256':INPUTS.copy(),
        'WQQ_access':False,'WR9_access':False,'held_truth_access':False}
    write(RUN/'protocol.json',protocol)
    print('GLOBAL HGB PROTOCOL',sha256(RUN/'protocol.json'),flush=True)


def verify():
    p=read(RUN/'protocol.json')
    for path,digest in p['sources'].items():pin(ROOT/path,digest)
    for path,digest in p['inputs_sha256'].items():pin(ROOT/path,digest)
    return p


def fit():
    verify();started=time.monotonic();receipts={}
    for fold,(train,held) in enumerate(cf.FOLDS):
        ALLOWED[:]=train+['J03WPY'];ys={};xx=[];yy=[];ww=[];counts={}
        for match in ALLOWED:
            for half in ('firstHalf','secondHalf'):
                name=match+'-'+half;ys[name]=np.load(pin(cf.DATA/(name+'.truth.npy')))
        live=sum(int((y<2).sum()) for y in ys.values())
        spells=sum(int(((y<2)&np.r_[True,y[1:]!=y[:-1]]).sum()) for y in ys.values())
        mass=live/max(spells,1)
        for key in cf.graph.keys(ALLOWED):
            before=time.monotonic();a=raw(key);f=context.features(a['players'],a['team'],a['pitch'])
            name=key.rsplit('-',1)[0];y=ys[name];idx,w=sample_spells(y,8,mass)
            for t in (0,1):xx.append(f[idx,t]);yy.append((y[idx]==t).astype(np.int8));ww.append(w)
            counts[key]={'sampled_frames':len(idx),'seconds':time.monotonic()-before}
            del a,f
            print('GLOBAL PREPARED',fold,key,len(idx),round(counts[key]['seconds'],2),flush=True)
        x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean();del xx,yy,ww
        before=time.monotonic();model=HistGradientBoostingClassifier(**MODEL).fit(x,y,sample_weight=w)
        dest=RUN/f'fold{fold}'/'model.pkl';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as h:pickle.dump(model,h)
        receipt={'fit':ALLOWED.copy(),'held':held,'shape':list(x.shape),'live_frames':live,'live_spells':spells,
            'spell_mass':mass,'model_sha256':sha256(dest),'fit_seconds':time.monotonic()-before,
            'views':counts,'reference_reads':[p for p in READS if any(m in p for m in ALLOWED)]}
        write(RUN/f'fold{fold}'/'fit-freeze.json',receipt);receipts[str(fold)]=receipt
        print('GLOBAL FIT',fold,receipt['shape'],receipt['fit_seconds'],flush=True)
        del model,x,y,w,ys
    ALLOWED.clear()
    write(RUN/'fit-seal.json',{'protocol_sha256':sha256(RUN/'protocol.json'),'folds':receipts,
        'seconds':time.monotonic()-started,'inputs_sha256':INPUTS,'reference_reads':READS,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2})


def owner(model,features,chunk=16384):
    result=[]
    for start in range(0,len(features),chunk):
        part=features[start:start+chunk];z=[]
        for t in (0,1):z.append(logit(np.clip(model.predict_proba(part[:,t])[:,1],1e-6,1-1e-6)))
        result.append((z[1]-z[0])*.5)
    return np.concatenate(result).astype(np.float32)


def predict():
    protocol=verify();fitseal=read(RUN/'fit-seal.json');ALLOWED.clear();before_reads=len(READS)
    outputs={};started=time.monotonic();views={}
    for fold,(train,held) in enumerate(cf.FOLDS):
        with pin(RUN/f'fold{fold}'/'model.pkl',fitseal['folds'][str(fold)]['model_sha256']).open('rb') as h:model=pickle.load(h)
        heads=combine.load_heads(fold)
        for key in cf.graph.keys(held):
            before=time.monotonic();a=raw(key);f=context.features(a['players'],a['team'],a['pitch']);z=owner(model,f);del f
            baseline=combine.parent_view(fold,key,heads=heads)
            for name,choice in [('wide_parent',None),*protocol['candidates'].items()]:
                if choice is None:parent=baseline
                else:
                    weight=choice['weight']
                    if choice['scope']=='missing25':weight=weight*uniform_filter1d((~np.isfinite(a['ball']).all(1)).astype(np.float64),25,mode='nearest')
                    parent=combine.parent_view(fold,key,extra_owner=z,weight=weight,heads=heads)
                np.testing.assert_array_equal(parent['predicted'][baseline['base']>=2],baseline['base'][baseline['base']>=2])
                dest=RUN/name/f'fold{fold}'/'predictions'/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,predicted=parent['predicted'],logit=parent['x'][:,0],global_logit=z)
                outputs[rel(dest)]=sha256(dest)
            views[key]={'seconds':time.monotonic()-before,'parent_exact':True,'global_logit_std':float(z.std())}
            print('GLOBAL PREDICTED',fold,key,round(views[key]['seconds'],2),'parent replay exact',flush=True)
    assert len(READS)==before_reads and len(outputs)==48
    write(RUN/'producer-seal.json',{'protocol_sha256':sha256(RUN/'protocol.json'),'outputs':outputs,
        'all16_parent_replays_exact':True,'held_truth_reads':[],'seconds':time.monotonic()-started,'views':views,
        'input_sha256':INPUTS,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','predict']);args=parser.parse_args()
    threadpool_limits(2);sys.addaudithook(guard);globals()[args.stage]()
