"""Learn directional H/A boundary costs from OOF evidence; select on WPY only.

This is a conditional ownership experiment. Every live/dead mask is fixed,
and no evaluation references enter feature construction or prediction.
"""
from __future__ import annotations
import os,sys,json,time,pickle,argparse,itertools
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from scipy.special import expit,logit
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from second_zero.sprint_decoder import sided_mean,segment_decode
from second_zero.evaluate import official_metrics
from second_zero.common import sha256,json_write
from scripts.sprint_decoder_search import metric
RUN=ROOT/'runs/two-hour-sprint/decoder/boundary-v1'
OUT=ROOT/'reports/two-hour-sprint/decoder/boundary-v1'
DATA=ROOT/'private/improvement/external/prepared'
MATCHES=['J03WMX','J03WN1','J03WOH','J03WOY'];HALVES=['firstHalf','secondHalf'];VIEWS=['clean','corrupted'];SEEDS=[20261023,20261024,20261025]
WIDTHS=[3,9,25,63,125]
P=json.loads((ROOT/'runs/oof-uniform-ensemble-v1/protocol.json').read_text())

def keys(matches):return [f'{m}-{h}-{v}' for m,h,v in itertools.product(matches,HALVES,VIEWS)]
def truth(key):return np.load(DATA/('-'.join(key.split('-')[:2])+'.truth.npy'))
def memberfolder(seed,short=False):
 return ROOT/('runs/extended-short-v2' if short else 'runs/ownership-temporal-v1') if seed==20261023 else ROOT/(f'runs/next-evidence-seeds/{seed}-'+('short25' if short else 'uniform'))

def signals(key):
 m=key.split('-')[0];columns=[]
 for arm in ['uniform'+str(s) for s in SEEDS]+['short25']:
  if m in MATCHES:
   fold=0 if m in MATCHES[:2] else 1;folder=ROOT/P['models'][str(fold)][arm]
  else:
   seed=20261023 if arm=='short25' else int(arm[len('uniform'):]);folder=memberfolder(seed,arm=='short25')
  path=folder/(key+'.npz');temperature=json.loads((folder/'freeze.json').read_text())['temperature']
  if path.exists():
   with np.load(path) as a:z=a['logit'] if 'logit' in a.files else a['raw_logit']/temperature
  else:z=np.load(folder/(key+'.raw.npy'))/temperature
  columns.append(z)
 zs=np.array(columns).T;ensemble=logit(np.clip(np.mean(expit(zs[:,:3]),axis=1),1e-10,1-1e-10))
 obs=np.load(ROOT/'runs/live-dead-temporal-v1/features'/(key+'.npy'),mmap_mode='r')
 return np.c_[ensemble,zs[:,3],zs[:,:3].std(1),np.mean(np.sign(zs[:,:3]),axis=1),obs[:,36],obs[:,38],obs[:,39]]

def features(x,team,points=None):
 s=1 if team else -1;parts=[x[:,0]*s,x[:,1]*s,x[:,2],x[:,3]*s,x[:,4],x[:,5],x[:,6]]
 for width in WIDTHS:
  for col in (0,1):
   a,b=sided_mean(x[:,col],width);parts.extend([a*s,b*s,(b-a)*s,a*b])
  for col in (2,4):
   a,b=sided_mean(x[:,col],width);parts.extend([a,b])
 result=np.array(parts,dtype=np.float32).T
 return result if points is None else result[points]

def targets(y,radius=3):
 valid=np.r_[False,(y[1:]<2)&(y[:-1]<2)];targets=np.zeros((len(y),2),bool)
 edges=np.flatnonzero(valid & np.r_[False,y[1:]!=y[:-1]])
 for edge in edges:targets[max(1,edge-radius):min(len(y),edge+radius+1),y[edge]]=True
 targets[~valid]=False
 return targets,valid

def train():
 if RUN.exists():raise FileExistsError(RUN)
 RUN.mkdir(parents=True);OUT.mkdir(parents=True,exist_ok=True)
 plan={'source_sha256':sha256(__file__),'train_matches':MATCHES,'calibration_match':'J03WPY','evaluation_match':'J03WQQ','training':'Directional boundary target±3f, every8th eligible edge plus everypositive, negative selection weight8 andpositive1; team-equivariant feature orientations. Owner models are opposite-fold trained. OOF ownership logits are reused method-development evidence.',
 'features':{'widths':WIDTHS,'columns':'ensemble logit,short25 logit,uniform logitstd,meanuniformsign,ballseen,previousgap,nextgap; past/future ownership means and difference/product; availabilitymeans'},
 'model':{'max_iter':160,'max_leaf_nodes':15,'max_depth':5,'learning_rate':.06,'min_samples_leaf':80,'l2_regularization':10.,'random_state':20261024,'early_stopping':False},
 'posterior_calibration':'Unweighted Platt sigmoid fit on allWPY eligible edges bydestination; decoderparameters subsequently selectedsameWPY; noWQQ orpublic orfinalused'},
 json_write(RUN/'protocol.json',plan)
 xx=[];yy=[];ww=[];stats={};start=time.monotonic()
 for key in keys(MATCHES):
  x=signals(key);y=truth(key);target,valid=targets(y)
  for team in (0,1):
   points=np.union1d(np.flatnonzero(valid & (np.arange(len(y))%8==0)),np.flatnonzero(target[:,team]))
   labels=target[points,team];xx.append(features(x,team,points));yy.append(labels);ww.append(np.where(labels,1.,8.))
  stats[key]={'frames':len(y),'boundary_band':int(target.sum()),'live_edges':int(valid.sum())}
  print('trainfeatures',key,flush=True)
 x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);del xx,yy,ww
 print('fit',x.shape,'seconds',time.monotonic()-start,flush=True)
 model=HistGradientBoostingClassifier(**plan[0]['model'] if isinstance(plan,tuple) else plan['model']).fit(x,y,sample_weight=w)
 with (RUN/'model.pkl').open('wb') as f:pickle.dump(model,f)
 json_write(RUN/'training.json',{'seconds':time.monotonic()-start,'shape':list(x.shape),'label_fraction':float(y.mean()),'counts':stats,'public_final_labels_used':False})

def getmodel():
 with (RUN/'model.pkl').open('rb') as f:return pickle.load(f)

def infer_scores(model,key):
 path=RUN/'scores'/(key+'.npz');path.parent.mkdir(exist_ok=True)
 if path.exists():
  with np.load(path) as a:return a['score'],a['logit']
 x=signals(key);q=np.stack([model.predict_proba(features(x,t))[:,1] for t in (0,1)],axis=1)
 np.savez_compressed(path,score=q,logit=x[:,0]);return q,x[:,0]

def costs_from_scores(q,offset,weight,calibration):
 q=expit(calibration['slope']*logit(np.clip(q,1e-8,1-1e-8))+calibration['intercept'])
 return np.clip(offset-weight*logit(np.clip(q,1e-8,1-1e-8)),.05,50)

def decode(z,base,costs):
 out=np.asarray(base).copy();live=out<2;bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
 for lo,hi in zip(bounds[:-1],bounds[1:]):
  if live[lo]:out[lo:hi]=segment_decode(z[lo:hi],costs[lo:hi])
 return out

def calibrate():
 model=getmodel();xx=[];yy=[];cache={};start=time.monotonic();pm=official_metrics()
 for key in keys(['J03WPY']):
  q,z=infer_scores(model,key);y=truth(key);target,valid=targets(y);xx.append(logit(np.clip(q[valid],1e-8,1-1e-8)).ravel());yy.append(target[valid].ravel())
  with np.load(ROOT/'runs/overnight-boundary-v1'/(key+'.npz')) as a:base=a['predicted']
  cache[key]=(q,z,y,base)
 xx=np.concatenate(xx);yy=np.concatenate(yy);fit=LogisticRegression(C=100000.,max_iter=300).fit(xx[:,None],yy)
 calibration={'slope':float(fit.coef_[0,0]),'intercept':float(fit.intercept_[0]),'frames':len(yy),'positive_prevalence':float(yy.mean()),'match':'J03WPY','public_final_labels_used':False}
 json_write(RUN/'calibration.json',calibration)
 rows=[]
 for offset,weight in itertools.product((-2.,0.,2.,4.,6.),(.25,.5,1.,2.)):
  view={k:metric(pm,y,decode(z,b,costs_from_scores(q,offset,weight,calibration))) for k,(q,z,y,b) in cache.items()}
  rows.append({'offset':offset,'weight':weight,'overall':{m:float(np.mean([v[m] for v in view.values()])) for m in ('spell','macro')},'views':view})
 rows.sort(key=lambda r:(r['overall']['spell'],r['overall']['macro']),reverse=True);selected=rows[0]
 baseline={k:metric(pm,y,decode(z,b,np.full((len(z),2),6.))) for k,(q,z,y,b) in cache.items()}
 freeze={'offset':selected['offset'],'weight':selected['weight'],'calibration':calibration,'model_sha256':sha256(RUN/'model.pkl'),'source_sha256':sha256(__file__),'WPY_selected':True,'WQQ_public_final_accessed':False}
 json_write(RUN/'freeze.json',freeze);json_write(OUT/'calibration_results.json',{'rows':rows,'baseline':baseline,'selected':selected,'seconds':time.monotonic()-start});print('SELECTED',selected['offset'],selected['weight'],selected['overall'],flush=True)

def evaluate():
 model=getmodel();freeze=json.loads((RUN/'freeze.json').read_text());pm=official_metrics();rows={}
 for key in keys(['J03WQQ']):
  q,z=infer_scores(model,key)
  with np.load(ROOT/'runs/fixed-ensemble-v1/A'/(key+'.npz')) as a:
   base=a['predicted'];assert np.max(np.abs(a['logit']-z))<.01;z=a['logit']
  costs=costs_from_scores(q,freeze['offset'],freeze['weight'],freeze['calibration']);pred=decode(z,base,costs)
  np.savez_compressed(RUN/(key+'.npz'),predicted=pred,logit=z,switch_cost=costs,boundary_score=q)
  y=truth(key);rows[key]=metric(pm,y,pred)
 result={'rows':rows,'overall':{m:float(np.mean([v[m] for v in rows.values()])) for m in ('spell','macro')},'configuration':freeze,'selection_on_WPY_only':True,'public_final_labels_used':False}
 json_write(OUT/'WQQ_results.json',result);print('WQQ',result['overall'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','calibrate','evaluate','all']);a=p.parse_args()
 for stage,function in [('train',train),('calibrate',calibrate),('evaluate',evaluate)]:
  if a.stage in (stage,'all'):function()
