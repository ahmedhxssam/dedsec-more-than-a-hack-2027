"""Bounded global decoder search on reused external development predictions."""
from __future__ import annotations
import sys,json,time,itertools,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from second_zero.sprint_decoder import decode,segment_decode
from second_zero.contextual import binary_decode
from second_zero.evidence_sequence import dynamic_decode
from second_zero.evaluate import official_metrics
from second_zero.common import sha256,json_write
OUT=ROOT/'reports/two-hour-sprint/decoder'
DATA=ROOT/'private/improvement/external/prepared'
FOLDS=[['J03WMX','J03WN1'],['J03WOH','J03WOY']]
HALVES=['firstHalf','secondHalf'];VIEWS=['clean','corrupted']

def verify():
 rng=np.random.default_rng(171);checks=[]
 for n in (0,1,2,10,100,1000):
  for cost in (0,1.5,6,24):
   z=rng.normal(0,3,n);c=np.full((n,2),cost,dtype=float)
   assert np.array_equal(segment_decode(z,c),binary_decode(z,cost))
   if n:c/=1+rng.random((n,2))
   assert np.array_equal(segment_decode(z,c),dynamic_decode(z,c))
   checks.append([n,cost])
 # Brute force confirms minimum-duration DP for short sequences.
 for n in range(2,11):
  z=rng.normal(size=n);c=np.full((n,2),1.2)
  for duration in range(2,min(5,n)+1):
   result=segment_decode(z,c,duration)
   def score(a):return float(np.dot(a,z)-1.2*np.count_nonzero(a[1:]!=a[:-1]))
   best=-np.inf
   for labels in itertools.product((0,1),repeat=n):
    a=np.array(labels);bounds=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,n]
    if min(np.diff(bounds))>=duration:best=max(best,score(a))
   assert abs(score(result)-best)<1e-8,(n,duration,result,best,score(result))
 json_write(OUT/'verification.json',{'exact_baseline_checks':checks,'semimarkov_bruteforce':True})

def configs():
 rows=[]
 for cost in (0,1,2,3,4.5,6,9,12,18,24,36):rows.append(dict(cost=cost))
 for kind in ('logit','probability','median'):
  for width in (3,7,15,31,61):
   for cost in (3,6,12):rows.append(dict(cost=cost,kind=kind,width=width))
 for cap in (.5,1,2,4):
  for cost in (1,3,6,12):rows.append(dict(cost=cost,cap=cap))
 for duration in (3,6,12,20,32):
  for cost in (1,3,6,12):rows.append(dict(cost=cost,duration=duration))
 return rows

def metric(pm,y,p):
 runs=pm.valid_runs(y);filled=pm.fill_abstentions(p,runs);sp=pm.spell_scores(y,filled,runs)
 return {'spell':sp['mean_f1'],'macro':pm.macro_f1(pm.scored_confusion(pm.confusion(y,p))),
         'n_pred':sp['n_pred'],'n_gt':sp['n_gt']}

def load(mode,arm):
 rows=[];manifest={}
 if mode=='oof':
  p=json.loads((ROOT/'runs/oof-uniform-ensemble-v1/protocol.json').read_text())
  sources=[]
  for fold,matches in enumerate(FOLDS):
   folder=ROOT/p['models'][str(fold)][arm] if arm!='ensemble' else ROOT/'runs/oof-uniform-ensemble-v1/ensemble'
   for m,h,v in itertools.product(matches,HALVES,VIEWS):sources.append((f'{m}-{h}-{v}',folder))
 else:
  folder=ROOT/'runs/fixed-ensemble-v1'/arm
  sources=[(f'J03WQQ-{h}-{v}',folder) for h,v in itertools.product(HALVES,VIEWS)]
 for key,folder in sources:
  path=folder/(key+'.npz');truth=DATA/('-'.join(key.split('-')[:2])+'.truth.npy')
  with np.load(path) as z:
   logit=z['logit'] if 'logit' in z.files else z['raw_logit']/json.loads((folder/'freeze.json').read_text())['temperature']
   if 'predicted' in z.files:base=z['predicted']
   else:
    with np.load(ROOT/'runs/overnight-boundary-v1/crossfit'/(key+'.npz')) as b:base=b['predicted']
  rows.append((key,logit,base,np.load(truth)))
  manifest[str(path.relative_to(ROOT))]=sha256(path);manifest[str(truth.relative_to(ROOT))]=sha256(truth)
 return rows,manifest

def run(mode,arm):
 start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);verify();allconfigs=configs()
 dest=OUT/f'{mode}-{arm}.json'
 if dest.exists():raise FileExistsError(dest)
 data,manifest=load(mode,arm);pm=official_metrics()
 json_write(OUT/f'{mode}-{arm}-plan.json',{'configurations':allconfigs,'input_hashes':manifest,'source_sha256':sha256(__file__),
  'decoder_sha256':sha256(ROOT/'second_zero/sprint_decoder.py'),'scope':'Reused external development selection. No public or reserved-final labels.'})
 rows=[]
 for i,cfg in enumerate(allconfigs):
  view={key:metric(pm,y,decode(z,b,**cfg)) for key,z,b,y in data}
  overall={m:float(np.mean([r[m] for r in view.values()])) for m in ('spell','macro')}
  bycondition={v:{m:float(np.mean([r[m] for k,r in view.items() if k.endswith(v)])) for m in ('spell','macro')} for v in VIEWS}
  rows.append({'config':cfg,'overall':overall,'conditions':bycondition,'views':view})
  if i%10==0:print(mode,arm,i+1,'/',len(allconfigs),cfg,overall,'seconds',round(time.monotonic()-start,2),flush=True)
 rows.sort(key=lambda r:(r['overall']['spell'],r['overall']['macro']),reverse=True)
 json_write(dest,{'mode':mode,'arm':arm,'rows':rows,'seconds':time.monotonic()-start,'input_hashes':manifest,
  'public_labels_read':False,'reserved_final_labels_read':False,'interpretation':'Development parameter search; not held-out final accuracy.'})
 print('TOP',mode,arm,[(r['config'],r['overall']) for r in rows[:10]],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['oof','wqq'],default='oof');p.add_argument('--arm',default='short25');a=p.parse_args();run(a.mode,a.arm)
