"""Fit a raw-motion temporal ROI scorer, with WPY-only selection.

All trained upstream owner/imputation/status components exclude the training
match used to construct an example. The existing quality head is explicitly
in-sample on these four training matches; no held-out claim is made for them.
"""
from __future__ import annotations
import os,sys,json,time,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from scripts.anchor_matched_quality import guard,write,array,TRAIN,READS,HASHES,evidence,pool
from scripts import sprint_decoder_segments_geometry_v2 as graph
from second_zero.vision_interval import motion_rois,IntervalMotion
from second_zero.evaluate import official_metrics
from second_zero.common import sha256


def record(key,quality,rng=None):
    if key.split('-')[0] in TRAIN:
        x=graph.signals(key);fold=0 if key.split('-')[0] in TRAIN[:2] else 1
        a=array(f'runs/two-hour-sprint/model/interp_learned_swap-{fold}',key)
        b=array(f'runs/two-hour-sprint/model/long-{fold}',key)
        x[:,0]=logit(np.clip(.5*expit(a)+.5*expit(b),1e-7,1-1e-7))
        base=array('runs/two-hour-sprint/status-crossfit-v2/cost50',key,'predicted')
    else:x,base=evidence(key)
    segments=pool(x,base)
    if rng is not None and len(segments)>12000:
        # Label-independent sample: retain an equal number of short and longer
        # candidates, so the training allocation does not scale with duration.
        short=segments[:,1]-segments[:,0]<=125;chosen=[]
        for mask in (short,~short):
            ids=np.flatnonzero(mask);chosen.extend(rng.choice(ids,min(6000,len(ids)),replace=False))
        segments=segments[np.sort(chosen)]
    features=graph.segment_features(x,segments)
    row={'key':key,'logit':x[:,0],'base':base,'segments':segments,'features':features,
         'quality':np.clip(quality.predict(features),0,1).astype(np.float32),
         'native':np.load(ROOT/'runs/ownership-temporal-v1/features'/f'{key}.npy',mmap_mode='r')}
    if rng is not None:row['target']=graph.quality_targets(graph.truth(key),segments)[0].astype(np.float32)
    return row


def infer(model,row,mean,std):
    model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(row['segments']),256):
            stop=start+256
            motion=torch.from_numpy(motion_rois(row['native'],row['segments'][start:stop]))
            summary=torch.from_numpy(np.clip((row['features'][start:stop]-mean)/std,-8,8))
            q=torch.from_numpy(row['quality'][start:stop])
            values.append(model(motion,summary,q).numpy())
    return np.concatenate(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=1200);p.add_argument('--independent',action='store_true');args=p.parse_args()
    sys.addaudithook(guard);torch.set_num_threads(2);threadpool_limits(2);torch.manual_seed(20261029)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(20261029)
    sources=out/'sources';sources.mkdir()
    for source in [Path(__file__),ROOT/'second_zero/vision_interval.py']:shutil.copyfile(source,sources/source.name)
    write(out/'protocol.json',{'training_matches':TRAIN,'calibration_match':'J03WPY','evaluation_match':'J03WQQ',
       'hypothesis':'Raw motion within candidate intervals and both flanks can identify short possession episodes hidden by aggregated owner probabilities.',
       'architecture':'Temporal ROI conv head;48 equally spaced interior samples+16eachflank; shared team-relative196 raw motion channels;175 existing quality features. Learned bounded logit correction to original interval quality.',
       'training':'12000 label-independent arcs perview, half<=125frames, half longer; raw motion mmap; MSE expectedIoU-credit target; no new feature cache. AdamW .001 weightdecay.001 batch64.',
       'split':'Modern learnedfill/long owners, status, geometry use oppositefold for training. Existing quality head is in-sample on train4. WPY selection only; WQQ forbidden until frozen predictions. No reserved/public reads.',
       'selection':'Checkpoints400/800/1200; qualityblend.25/.5/1; fixedwinner status/graphpenalty.55/frameweight.01; macroguard.002. No minimumduration.',
       'independent_quality':args.independent,
       'independent_interpretation':'If enabled, predicts quality directly from motion+summary without old quality in its forward pass; blending with old quality occurs only at inference. This removes the bounded residual lock on confidently wrong old proposals.',
       'steps':args.steps,'source_sha256':sha256(__file__)})
    quality=graph.get_model();rows=[];started=time.monotonic()
    for key in graph.keys(TRAIN):
        row=record(key,quality,rng);rows.append(row);print('interval prepared',key,len(row['segments']),flush=True)
    count=sum(len(r['features']) for r in rows)
    mean=sum(r['features'].sum(0,dtype=np.float64) for r in rows)/count
    std=np.sqrt(np.maximum(sum(np.square(r['features'],dtype=np.float64).sum(0) for r in rows)/count-mean**2,.0025))
    mean=mean.astype(np.float32);std=std.astype(np.float32)
    np.savez_compressed(out/'normalizer.npz',mean=mean,std=std)
    for r in rows:r['normalized']=np.clip((r['features']-mean)/std,-8,8);del r['features']
    cal={k:record(k,quality) for k in graph.keys(['J03WPY'])};truth={k:graph.truth(k) for k in cal};pm=official_metrics()
    def measure(prediction,step,weight):
        views={}
        for key,r in cal.items():
            q=r['quality'] if weight==0 else (1-weight)*r['quality']+weight*prediction[key]
            pred=graph.graph_decode(r['logit'],r['base'],r['segments'],q,.55,.01)
            views[key]=graph.metric(pm,truth[key],pred)
        result={'step':step,'weight':weight,'overall':{m:float(np.mean([v[m] for v in views.values()])) for m in ['spell','macro']},'views':views}
        print('interval WPY',result,flush=True);return result
    model=IntervalMotion(independent=args.independent);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    baseline=measure(None,0,0);scores=[baseline];best=baseline;history=[]
    for step in range(1,args.steps+1):
        r=rows[int(rng.integers(len(rows)))];ids=rng.integers(0,len(r['segments']),64)
        motion=torch.from_numpy(motion_rois(r['native'],r['segments'][ids]))
        summary=torch.from_numpy(r['normalized'][ids]);q=torch.from_numpy(r['quality'][ids]);target=torch.from_numpy(r['target'][ids])
        model.train();pred=model(motion,summary,q);loss=(pred-target).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        if step%100==0:
            hist={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-started};history.append(hist);print('interval train',hist,flush=True)
        if step in (args.steps//3,2*args.steps//3,args.steps):
            predictions={k:infer(model,r,mean,std) for k,r in cal.items()}
            for weight in (.25,.5,1.):
                score=measure(predictions,step,weight);scores.append(score)
                if score['overall']['macro']>=baseline['overall']['macro']-.002 and (score['overall']['spell'],score['overall']['macro'])>(best['overall']['spell'],best['overall']['macro']):
                    best=score;torch.save(model.state_dict(),out/'model.pt')
            with (out/'progress.json').open('w') as f:json.dump({'selected':best,'rows':scores,'history':history},f,indent=2)
    if best['weight']==0:torch.save(model.state_dict(),out/'model.pt')
    write(out/'freeze.json',{'selected':best,'WPY_rows':scores,'model_sha256':sha256(out/'model.pt'),'normalizer_sha256':sha256(out/'normalizer.npz'),'WQQ_truth_read':False})
    write(out/'training.json',{'history':history,'input_sha256':HASHES,'reference_reads':sorted(READS),'seconds':time.monotonic()-started})
    del rows,cal
    if best['weight']:
        model.load_state_dict(torch.load(out/'model.pt',weights_only=True));dest=out/'predictions';dest.mkdir()
        for key in graph.keys(['J03WQQ']):
            r=record(key,quality);q=infer(model,r,mean,std);q=(1-best['weight'])*r['quality']+best['weight']*q
            pred=graph.graph_decode(r['logit'],r['base'],r['segments'],q,.55,.01)
            np.savez_compressed(dest/f'{key}.npz',predicted=pred,logit=r['logit'],status_mask=r['base'])
        write(out/'inference.json',{'outputs':{p.name:sha256(p) for p in dest.glob('*.npz')},'WQQ_truth_read':False})
    print('interval selected',best,flush=True)

if __name__=='__main__':main()
