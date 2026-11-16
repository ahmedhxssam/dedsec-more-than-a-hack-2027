"""Connect evidence-derived boundary nodes into a denser full-coverage graph."""
import sys,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
import scripts.sprint_decoder_segments_geometry_v2 as graph
from second_zero.common import sha256,json_write
from second_zero.evaluate import official_metrics
RUN=ROOT/'runs/two-hour-sprint/decoder/segment-lattice-v1';OUT=ROOT/'reports/two-hour-sprint/decoder/segment-lattice-v1'
OWNER=ROOT/'runs/two-hour-sprint/carrier-mixture-v1';STATUS=ROOT/'runs/two-hour-sprint/status-mixture-v1/reliability'

def proposals(z,base,pool,neighbors):
 if not neighbors:return pool
 result=set(map(tuple,pool.tolist()));live=base<2;islands=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)];nodes=np.unique(pool[:,:2]);primary=graph.decode(z,base,cost=12);cuts=np.r_[0,np.flatnonzero(primary[1:]!=primary[:-1])+1,len(base)]
 for lo,hi in zip(islands[:-1],islands[1:]):
  if not live[lo]:continue
  local=nodes[(nodes>=lo)&(nodes<=hi)]
  for i,s in enumerate(local[:-1]):
   for e in local[i+1:i+1+neighbors]:
    if e-s>250:break
    result.add((int(s),int(e),0));result.add((int(s),int(e),1))
 # Background arcs join multiple proposed reversals in one primary possession.
 # This closes connectivity gaps without inventing new boundary locations.
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  t=int(primary[lo])
  if t>=2:continue
  local=nodes[(nodes>=lo)&(nodes<=hi)]
  for i,s in enumerate(local[:-1]):
   result.add((int(lo),int(local[i+1]),t));result.add((int(s),int(hi),t))
   for e in local[i+1:i+1+max(32,neighbors)]:
    if e-s>1250:break
    result.add((int(s),int(e),t))
 return np.array(sorted(result),dtype=np.int64)

def prepare(key,model,status,neighbors,tag):
 cache=RUN/'cache'/tag/str(neighbors);cache.mkdir(parents=True,exist_ok=True);path=cache/(key+'.npz')
 if path.exists():
  with np.load(path) as a:return {k:a[k] for k in a.files}
 x=graph.signals(key)
 with np.load(OWNER/(key+'.npz')) as a:x[:,0]=a['logit']
 with np.load(status/(key+'.npz')) as a:base=a['predicted']
 pool=proposals(x[:,0],base,graph.proposals(x[:,0],base,x[:,4],x[:,7:10]),neighbors);q=np.clip(model.predict(graph.segment_features(x,pool)),0,1)
 data={'logit':x[:,0],'base':base,'segments':pool,'quality':q};np.savez_compressed(path,**data);print('prepared',key,neighbors,len(pool),flush=True);return data

def main():
 RUN.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=True);model=graph.get_model();pm=official_metrics();rows=[]
 for neighbors in [0,8,16]:
  views={}
  for key in graph.keys(['J03WPY']):
   a=prepare(key,model,STATUS,neighbors,'WPY');p=graph.graph_decode(a['logit'],a['base'],a['segments'],a['quality'],.55,.01);views[key]=graph.metric(pm,graph.truth(key),p)
  rows.append({'neighbors':neighbors,'overall':{m:float(np.mean([r[m] for r in views.values()])) for m in ['spell','macro']},'views':views});print('WPY mode',neighbors,rows[-1]['overall'],flush=True)
 best=max(rows,key=lambda r:(r['overall']['spell'],r['overall']['macro']));json_write(OUT/'WPY.json',{'rows':rows,'selected':best});json_write(RUN/'freeze.json',{'neighbors':best['neighbors'],'penalty':.55,'frame_weight':.01,'source_sha256':sha256(__file__),'model_sha256':sha256(graph.RUN/'model.pkl'),'selection':'WPY only','boundaries':'Only existing predicted path nodes; short arcs<=250frames, primary-background arcs<=1250frames and32neighbor nodes','no_abstention':True});print('Selected',best['neighbors'],flush=True)
 for tag,status in [('reliability',STATUS),('joint-carrier',ROOT/'runs/two-hour-sprint/joint-carrier-v2')]:
  dest=RUN/tag;dest.mkdir();views={}
  for key in graph.keys(['J03WQQ']):
   a=prepare(key,model,status,best['neighbors'],tag);p=graph.graph_decode(a['logit'],a['base'],a['segments'],a['quality'],.55,.01);np.savez_compressed(dest/(key+'.npz'),predicted=p,logit=a['logit'],status_mask=a['base']);views[key]=graph.metric(pm,graph.truth(key),p)
  result={'overall':{m:float(np.mean([r[m] for r in views.values()])) for m in ['spell','macro']},'views':views};json_write(OUT/('WQQ-'+tag+'.json'),result);print('WQQ',tag,result['overall'],flush=True)
if __name__=='__main__':main()
