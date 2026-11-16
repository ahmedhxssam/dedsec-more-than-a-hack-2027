"""OOF segment-IoU quality learning and a full-coverage possession graph decoder.

Candidate intervals come only from global decoder paths. Inference features
contain source/model evidence and relative durations, never reference intervals.
"""
from __future__ import annotations
import sys,os,json,pickle,time,itertools,argparse
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from second_zero.sprint_decoder import decode,adaptive_decode,sided_mean
from second_zero.common import json_write,sha256
from second_zero.evaluate import official_metrics
from scripts.sprint_decoder_search import metric
from scripts.sprint_decoder_boundary import signals as base_signals,truth,keys,MATCHES
from scipy.ndimage import uniform_filter1d
from scripts.train_contextual import corruption
from scripts.sprint_ball_imputer import predict as impute_ball
from second_zero.sprint_model_interpolation import interpolated_features
RUN=ROOT/'runs/two-hour-sprint/decoder/segment-quality-geometry-v2';OUT=ROOT/'reports/two-hour-sprint/decoder/segment-quality-geometry-v2'
COSTS=[.5,1.5,3,6,9,12,18,24,36]


IMPUTERS={}
def physical_geometry(key):
 cache=ROOT/'runs/two-hour-sprint/decoder/physical-geometry-cache';cache.mkdir(parents=True,exist_ok=True);path=cache/(key+'.npy')
 if path.exists():return np.load(path)
 name,condition=key.rsplit('-',1);match=name.split('-')[0]
 with np.load(ROOT/'private/improvement/external/prepared'/(name+'.npz')) as z:a={k:z[k] for k in ['ball','players','team','pitch']}
 if condition=='corrupted':a=corruption(a,100)
 fold=('0' if match in MATCHES[:2] else '1') if match in MATCHES else 'full'
 imputer=ROOT/'runs/two-hour-sprint'/('ball-imputer-v1' if fold=='full' else 'ball-imputer-fold'+fold)
 if fold not in IMPUTERS:
  with (imputer/'models.pkl').open('rb') as f:IMPUTERS[fold]=pickle.load(f)
 filled,eligible=impute_ball(a,IMPUTERS[fold]);linear=interpolated_features(a['ball'],a['players'],a['team'],a['pitch']);learned=interpolated_features(filled,a['players'],a['team'],a['pitch'])
 raw=np.load(ROOT/'runs/ownership-temporal-v1/features'/(key+'.npy'),mmap_mode='r')
 unaries=[];confidences=[]
 for features,observed in [(raw,True),(linear,False),(learned,False)]:
  d=features[:,:,0]*20;rv=features[:,:,3]*30
  valid=features[:,:,49 if observed else 18]<.5
  both=valid.all(1);confidence=np.where(both,np.exp(-np.min(d,axis=1)/10),0)
  z=np.clip((d[:,0]-d[:,1])/3+.25*(rv[:,0]-rv[:,1])/10,-4,4)*confidence
  z[~both]=0;unaries.append(z);confidences.append(confidence)
 result=np.c_[*unaries,*confidences].astype(np.float32);assert np.isfinite(result).all();np.save(path,result);return result

def signals(key):
 return np.c_[base_signals(key),physical_geometry(key)].astype(np.float32)

def get_base(key):
 m=key.split('-')[0]
 path=ROOT/('runs/overnight-boundary-v1/crossfit' if m in MATCHES else 'runs/overnight-boundary-v1')/(key+'.npz')
 with np.load(path) as a:return a['predicted']

def proposals(z,base,seen,geometry=None):
 intervals=set();live=base<2;bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(z)]
 paths=[decode(z,base,cost=c) for c in COSTS]
 paths += [adaptive_decode(z,base,seen,cost=c,width=31,discount=.5) for c in (6,12,18)]
 for path in paths:
  for lo,hi in zip(bounds[:-1],bounds[1:]):
   if not live[lo]:continue
   edges=np.r_[lo,np.flatnonzero(path[lo+1:hi]!=path[lo:hi-1])+lo+1,hi]
   for s,e in zip(edges[:-1],edges[1:]):intervals.add((int(s),int(e),int(path[s])))
 shifted=set()
 for s,e,t in intervals:
  island=np.searchsorted(bounds,s,side='right')-1;lo,hi=int(bounds[island]),int(bounds[island+1])
  for ds,de in itertools.product((-16,-8,0,8,16),repeat=2):
   ss=max(lo,min(hi,s+ds));ee=max(lo,min(hi,e+de))
   if ee>ss:shifted.add((ss,ee,t))
 if geometry is not None:
  geometric=set();primary=decode(z,base,cost=12)
  for j in range(3):
   for width in (3,11):
    signal=uniform_filter1d(geometry[:,j],width,mode='nearest')
    for cost in (3,6):
     path=decode(signal,base,cost=cost);other=decode(-signal,base,cost=cost);other[base<2]=1-other[base<2]
     # Include both tie conventions: geometry can be exactly neutral in gaps.
     # This makes the set, rather than a floating tie-break, team-equivariant.
     for candidate in (path,other):
      for lo,hi in zip(bounds[:-1],bounds[1:]):
       if not live[lo]:continue
       edges=np.r_[lo,np.flatnonzero(candidate[lo+1:hi]!=candidate[lo:hi-1])+lo+1,hi]
       for ss,ee in zip(edges[:-1],edges[1:]):geometric.add((int(ss),int(ee),int(candidate[ss])))
  shifted.update(geometric)
  # Give a short physically suggested opposite-team arc compatible surrounding
  # arcs inside a primary segment, so recovery need not adopt a whole geometry path.
  edges=np.r_[0,np.flatnonzero(primary[1:]!=primary[:-1])+1,len(z)]
  for ss,ee,t in geometric:
   if ee-ss>250:continue
   i=np.searchsorted(edges,ss,side='right')-1;lo,hi=int(edges[i]),int(edges[i+1]);old=int(primary[ss])
   if old>=2 or old==t or ee>hi:continue
   if ss>lo:shifted.add((lo,ss,old))
   if ee<hi:shifted.add((ee,hi,old))
 return np.array(sorted(shifted),dtype=np.int64)

def segment_features(x,segments):
 starts,ends,teams=segments.T;duration=ends-starts;sign=2*teams-1;n=len(x)
 # Team-symmetric point evidence and invariant reliability channels.
 values=np.c_[x[:,0],x[:,1],np.abs(x[:,0]),np.abs(x[:,1]),(np.sign(x[:,0])+1)/2,(np.sign(x[:,1])+1)/2,x[:,2],x[:,4],x[:,5],x[:,6]]
 prefix=np.vstack([np.zeros(values.shape[1]),np.cumsum(values,axis=0)]);squared=np.vstack([np.zeros(values.shape[1]),np.cumsum(values**2,axis=0)])
 mean=(prefix[ends]-prefix[starts])/duration[:,None];var=np.maximum(0,(squared[ends]-squared[starts])/duration[:,None]-mean**2)
 mean[:,:2]*=sign[:,None];mean[:,4:6]=np.where(teams[:,None]==1,mean[:,4:6],1-mean[:,4:6])
 feats=[np.log1p(duration)[:,None],np.sqrt(duration[:,None])/25,duration[:,None]/250,mean,np.sqrt(var)]
 for width in (3,9,31,63,125):
  lo=np.maximum(starts-width,0);hi=np.minimum(ends+width,n)
  before=(prefix[starts]-prefix[lo])/np.maximum(1,starts-lo)[:,None]
  after=(prefix[hi]-prefix[ends])/np.maximum(1,hi-ends)[:,None]
  before[starts==lo,4:6]=.5;after[ends==hi,4:6]=.5
  for a in (before,after):
   a[:,:2]*=sign[:,None];a[:,4:6]=np.where(teams[:,None]==1,a[:,4:6],1-a[:,4:6])
  feats.extend([before,after])
 # Local endpoint evidence; no absolute start/stop/frame or match identifier.
 feats.extend([x[starts,:2]*sign[:,None],x[ends-1,:2]*sign[:,None]])
 if x.shape[1]>7:
  g=x[:,7:];gp=np.vstack([np.zeros(g.shape[1]),np.cumsum(g,axis=0)]);g2=np.vstack([np.zeros(g.shape[1]),np.cumsum(g*g,axis=0)])
  gm=(gp[ends]-gp[starts])/duration[:,None];gv=np.maximum(0,(g2[ends]-g2[starts])/duration[:,None]-gm*gm);gm[:,:3]*=sign[:,None]
  feats.extend([gm,np.sqrt(gv)])
  for width in (9,31,125):
   lo=np.maximum(starts-width,0);hi=np.minimum(ends+width,n)
   before=(gp[starts]-gp[lo])/np.maximum(1,starts-lo)[:,None];after=(gp[hi]-gp[ends])/np.maximum(1,hi-ends)[:,None]
   before[:,:3]*=sign[:,None];after[:,:3]*=sign[:,None];feats.extend([before,after])
 return np.concatenate(feats,axis=1).astype(np.float32)

def quality_targets(y,segments):
 pm=official_metrics();gs=pm.possession_spells(y,pm.valid_runs(y));best=np.zeros(len(segments))
 for iou,gi,pi in pm.spell_overlaps(gs,segments.tolist()):best[pi]=max(best[pi],iou)
 return np.mean(best[:,None]>=np.array([.3,.4,.5,.6,.7]),axis=1),best

def train():
 if RUN.exists():raise FileExistsError(RUN)
 RUN.mkdir(parents=True);OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();xx=[];yy=[];counts={}
 plan={'training_matches':MATCHES,'calibration_match':'J03WPY','evaluation_match':'J03WQQ','physical_proposals':'observed/linear/OOFlearned geometry; signed nearest-team distance + relative velocity, boundedconfidence; smooth3/11frames and costs3/6; compatible surrounding arcs for≤250f opposite-team candidates', 'physical_features':'team-relative geometry and invariant confidence; interval mean/variance and before/after9,31,125frames', 'candidate_costs':COSTS,'candidate_adaptive_costs':[6,12,18],'endpoint_offsets_frames':[-16,-8,0,8,16],
 'features':'Relative duration and signed model evidence plus visibility, dispersion, and before/after context at3,9,31,63,125frames; no absolute time/match/team identifiers.',
 'target':'Mean credit for bestsame-team GT IoU overofficial thresholds. This is expected per-proposal credit, not groundtruth at inference; greedy one-to-one conflicts are not modeled in regressor.',
 'model':{'max_iter':180,'max_leaf_nodes':15,'max_depth':5,'min_samples_leaf':40,'learning_rate':.05,'l2_regularization':5.,'random_state':20261024,'early_stopping':False},
 'coverage':'Graph pathscover every fixed liveframe; D/X unchanged; neighboringsegments mustalternate teams. Never abstain.',
 'source_sha256':sha256(__file__)}
 json_write(RUN/'protocol.json',plan)
 for key in keys(MATCHES):
  raw=signals(key);base=get_base(key);y=truth(key)
  for primary in (0,1):
   x=raw.copy()
   if primary==1:x[:,[0,1]]=x[:,[1,0]]
   segments=proposals(x[:,0],base,x[:,4],x[:,7:10]);features=segment_features(x,segments);target,iou=quality_targets(y,segments);xx.append(features);yy.append(target)
   counts[key+'-primary'+str(primary)]={'proposals':len(segments),'positive_credit':int((target>0).sum()),'mean_credit':float(target.mean())}
  print('features',key,flush=True)
 x=np.concatenate(xx);y=np.concatenate(yy);model=HistGradientBoostingRegressor(**plan['model']).fit(x,y)
 with (RUN/'model.pkl').open('wb') as f:pickle.dump(model,f)
 json_write(RUN/'training.json',{'shape':list(x.shape),'seconds':time.monotonic()-start,'counts':counts,'public_final_labels_used':False});print('trained',x.shape,'seconds',time.monotonic()-start,flush=True)

def get_model():
 with (RUN/'model.pkl').open('rb') as f:return pickle.load(f)

def graph_decode(z,base,segments,quality,penalty,frame_weight):
 # Exact DAG DP with alternating labels; every proposed arc is an actual
 # candidate segment and every candidate full path supplies a feasible cover.
 out=base.copy();live=base<2;bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(z)];prefix=np.r_[0,np.cumsum(z)]
 for lo,hi in zip(bounds[:-1],bounds[1:]):
  if not live[lo]:continue
  take=np.flatnonzero((segments[:,0]>=lo)&(segments[:,1]<=hi));seg=segments[take];q=quality[take]
  edges=np.unique(seg[:,:2]);lookup={int(v):i for i,v in enumerate(edges)};score=np.full((len(edges),2),-np.inf);back=np.full((len(edges),2),-1,dtype=np.int64);score[lookup[lo]]=0
  order=np.lexsort((seg[:,0],seg[:,1]))
  for j in order:
   s,e,t=seg[j];a,b=lookup[int(s)],lookup[int(e)];gain=q[j]-penalty+frame_weight*(2*t-1)*(prefix[e]-prefix[s]);value=score[a,1-t]+gain
   if value>score[b,t]:score[b,t]=value;back[b,t]=j
  end=hi;team=int(score[lookup[hi],1]>score[lookup[hi],0])
  if not np.isfinite(score[lookup[hi],team]):raise AssertionError('No full-cover graph path')
  while end>lo:
   j=back[lookup[end],team]
   if j<0:raise AssertionError('Missing graph predecessor')
   s,e,t=seg[j];out[s:e]=t;end=int(s);team=1-int(t)
 assert np.array_equal(out[base>=2],base[base>=2]);return out

def cached(key,model,folder=None,status=None,tag='base'):
 dest=RUN/'proposal-cache'/tag;dest.mkdir(parents=True,exist_ok=True);path=dest/(key+'.npz')
 if path.exists():
  with np.load(path) as a:return {k:a[k] for k in a.files}
 x=signals(key);base=get_base(key)
 if folder:
  with np.load(Path(folder)/(key+'.npz')) as a:x[:,0]=a['logit'];base=a['predicted']
 if status:
  with np.load(Path(status)/(key+'.npz')) as a:base=a['predicted']
 segments=proposals(x[:,0],base,x[:,4],x[:,7:10]);features=segment_features(x,segments);q=np.clip(model.predict(features),0,1)
 data={'logit':x[:,0],'base':base,'segments':segments,'quality':q};np.savez_compressed(path,**data);return data

def calibrate():
 pm=official_metrics();model=get_model();cache={key:cached(key,model) for key in keys(['J03WPY'])};ys={key:truth(key) for key in cache};rows=[]
 for penalty,fw in itertools.product((.15,.25,.35,.45,.55),(.0,.001,.003,.01,.03)):
  view={k:metric(pm,ys[k],graph_decode(a['logit'],a['base'],a['segments'],a['quality'],penalty,fw)) for k,a in cache.items()}
  rows.append({'penalty':penalty,'frame_weight':fw,'overall':{m:float(np.mean([r[m] for r in view.values()])) for m in ['spell','macro']},'views':view})
 baseline={k:metric(pm,ys[k],decode(a['logit'],a['base'],cost=6)) for k,a in cache.items()};base_macro=np.mean([r['macro'] for r in baseline.values()]);eligible=[r for r in rows if r['overall']['macro']>=base_macro-.005]
 selected=max(eligible or rows,key=lambda r:(r['overall']['spell'],r['overall']['macro']))
 json_write(OUT/'WPY_calibration.json',{'rows':rows,'baseline':baseline,'macro_guard':.005,'selected':selected});json_write(RUN/'freeze.json',{'penalty':selected['penalty'],'frame_weight':selected['frame_weight'],'model_sha256':sha256(RUN/'model.pkl'),'source_sha256':sha256(__file__),'WPY_selected':True,'public_final_labels_used':False});print('selected',selected,flush=True)

def evaluate(folder=None,status=None,tag='base'):
 pm=official_metrics();model=get_model();cfg=json.loads((RUN/'freeze.json').read_text());rows={};out=RUN/('evaluated-'+tag);out.mkdir(exist_ok=True)
 for key in keys(['J03WQQ']):
  a=cached(key,model,folder,status,tag);pred=graph_decode(a['logit'],a['base'],a['segments'],a['quality'],cfg['penalty'],cfg['frame_weight']);np.savez_compressed(out/(key+'.npz'),predicted=pred,logit=a['logit']);rows[key]=metric(pm,truth(key),pred)
 result={'rows':rows,'overall':{m:float(np.mean([r[m] for r in rows.values()])) for m in ['spell','macro']},'config':cfg,'folder':folder,'status':status,'public_final_labels_used':False};json_write(OUT/('WQQ-'+tag+'.json'),result);print(result['overall'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','calibrate','evaluate']);p.add_argument('--input');p.add_argument('--status');p.add_argument('--tag',default='base');a=p.parse_args()
 if a.stage=='train':train()
 elif a.stage=='calibrate':calibrate()
 else:evaluate(a.input,a.status,a.tag)
