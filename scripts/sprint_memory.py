"""Low-rate bidirectional control memory, trained on OOF ensemble residuals.

Geometry and ensemble outputs are observations. No provider label enters inference.
Training excludes WPY/WQQ/final; WPY selects checkpoint and temperature, WQQ is dev.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
import torch
from torch import nn
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from second_zero.common import json_write,sha256
from scripts.sprint_stack import inputs,truth,keys,TRAIN,guard,timeguard
from scripts import sprint_stack as stack_source
from scripts.evaluate_temporal_ownership import fixed_state_owner
from second_zero.evaluate import official_metrics

STRIDE=5
CONTEXT=64
EXTRA_GEOMETRY_CHANNELS=0
EXTRA_GEOMETRY_PROTOCOL=None
WARMSTART=None
LEARNING_RATE=.001

class ControlMemory(nn.Module):
    def __init__(self,channels=102,width=32):
        super().__init__()
        self.input=nn.Linear(channels,width)
        self.memory=nn.GRU(width,width,batch_first=True,bidirectional=True)
        self.output=nn.Linear(2*width,1)
        self.dropout=nn.Dropout(.1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        b,n,t,c=x.shape
        z=x.permute(0,2,1,3).reshape(b*t,n,c)
        z=self.dropout(torch.nn.functional.gelu(self.input(z)))
        z,_=self.memory(z)
        z=self.output(z).reshape(b,t,n)
        return 6*torch.tanh((z[:,1]-z[:,0])/6)

def pack(x,z,y=None):
    n=len(x);pad=(-n)%STRIDE
    values=np.concatenate([np.asarray(x),np.stack([-z,z],axis=1)],axis=2).astype(np.float32)
    values=np.pad(values,((0,pad),(0,0),(0,0)),mode='edge')
    values=values.reshape(-1,STRIDE,2,values.shape[-1]).mean(axis=1)
    base=logit(np.clip(expit(z[:,:3]).mean(axis=1),1e-6,1-1e-6)).astype(np.float32)
    base_coarse=np.pad(base,(0,pad),mode='edge').reshape(-1,STRIDE).mean(axis=1)
    row={'features':values,'base':base,'coarse_base':base_coarse,'frames':n}
    if y is not None:
        valid=y<2; yp=np.pad(np.where(valid,y,0),(0,pad));vp=np.pad(valid,(0,pad))
        weight=vp.reshape(-1,STRIDE).mean(axis=1)
        target=np.divide(yp.reshape(-1,STRIDE).sum(axis=1),vp.reshape(-1,STRIDE).sum(axis=1),out=np.zeros(len(weight)),where=weight>0)
        row.update(target=target.astype(np.float32),weight=weight.astype(np.float32),truth=y)
    return row

def infer(model,row):
    model.eval(); x=row['features']; out=np.empty(len(x),np.float32)
    with torch.inference_mode():
        for s in range(0,len(x),512):
            e=min(len(x),s+512);lo=max(0,s-CONTEXT);hi=min(len(x),e+CONTEXT)
            tensor=torch.from_numpy(x[lo:hi][None].copy())
            out[s:e]=model(tensor)[0,s-lo:e-lo].numpy()
    native=np.interp(np.arange(row['frames']),np.arange(len(x))*STRIDE+2,out)
    return (row['base']+native).astype(np.float32)

def quality(model,records):
    zz=[];yy=[]
    for row in records.values():
        z=infer(model,row);valid=row['truth']<2;zz.append(z[valid]);yy.append(row['truth'][valid])
    z=np.concatenate(zz);y=np.concatenate(yy)
    return float(np.mean(np.logaddexp(0,z)-y*z))

def train(folder,steps,seed,width=32):
    folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    extra_models=([stack_source.EXTRAS] if stack_source.EXTRAS else [])+stack_source.EXTRAS_ADDITIONAL
    channels=102+len(extra_models)+EXTRA_GEOMETRY_CHANNELS
    protocol={'architecture':'shared bidirectionalGRU residual to 3uniform probability ensemble',
       'channels':channels,'width':width,'extra_models':stack_source.EXTRAS,
       'extra_geometry':EXTRA_GEOMETRY_PROTOCOL,
       'extra_model_list':extra_models,
       'training_matches':TRAIN,'base_predictions':'whole-match-out-of-fold','calibration':'WPY checkpointNLL and scalar temperature',
       'development':'WQQ repeatedly used, never training','stride':STRIDE,'context':CONTEXT,'steps':steps,
       'seed':seed,'batch':8,'target_coarse_frames':192,'learning_rate':LEARNING_RATE,'warmstart':str(WARMSTART) if WARMSTART else None,'source_sha256':sha256(Path(__file__))}
    json_write(folder/'protocol.json',protocol);(folder/'source.py').write_text(Path(__file__).read_text())
    records=[]
    for key in keys(TRAIN):
        x,z=inputs(key);records.append(pack(x,z,truth(key)));print('MEMORY prepared',key,flush=True)
    calibration={}
    for key in keys(['J03WPY']):
        x,z=inputs(key);calibration[key]=pack(x,z,truth(key))
    model=ControlMemory(channels=channels,width=width)
    if WARMSTART:
        previous=torch.load(Path(WARMSTART)/'model.pt',weights_only=True);current=model.state_dict()
        for key in current:
            if key!='input.weight':current[key]=previous[key]
        before=previous['input.weight'];current['input.weight'].zero_()
        expert_count=4+len(extra_models);old_geometry=before.shape[1]-expert_count
        current['input.weight'][:,:old_geometry]=before[:,:old_geometry]
        current['input.weight'][:,-expert_count:]=before[:,-expert_count:]
        model.load_state_dict(current)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    best=quality(model,calibration);torch.save(model.state_dict(),folder/'model.pt');history=[{'step':0,'WPY_NLL':best}]
    width=192;started=time.perf_counter()
    for step in range(1,steps+1):
        timeguard();xx=[];yy=[];ww=[];bb=[]
        for _ in range(8):
            row=records[int(rng.integers(len(records)))];n=len(row['features'])
            s=int(rng.integers(CONTEXT,n-width-CONTEXT));e=s+width
            xx.append(row['features'][s-CONTEXT:e+CONTEXT]);yy.append(row['target'][s:e]);ww.append(row['weight'][s:e]);bb.append(row['coarse_base'][s:e])
        model.train();x=torch.from_numpy(np.stack(xx));y=torch.from_numpy(np.stack(yy));w=torch.from_numpy(np.stack(ww));base=torch.from_numpy(np.stack(bb))
        z=base+model(x)[:,CONTEXT:CONTEXT+width]
        error=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')
        loss=(error*w).sum()/w.sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:print('MEMORY step',step,'loss',float(loss.detach()),'seconds',time.perf_counter()-started,flush=True)
        if step%300==0 or step==steps:
            score=quality(model,calibration);row={'step':step,'WPY_NLL':score,'seconds':time.perf_counter()-started};history.append(row)
            if score<best:best=score;torch.save(model.state_dict(),folder/'model.pt');row['selected']=True
            json_write(folder/'training.json',{'history':history,'best_WPY_NLL':best,'seconds':time.perf_counter()-started})
            print('MEMORY CHECKPOINT',json.dumps(row),flush=True)
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True));zz=[];yy=[]
    for key,row in calibration.items():
        raw=infer(model,row);np.savez_compressed(folder/(key+'.npz'),raw_logit=raw);v=row['truth']<2;zz.append(raw[v]);yy.append(row['truth'][v])
    z=np.concatenate(zz);y=np.concatenate(yy)
    optt=minimize_scalar(lambda lt:np.mean(np.logaddexp(0,z/np.exp(lt))-y*z/np.exp(lt)),bounds=(-2.3,2.3),method='bounded')
    temp=float(np.exp(optt.x));json_write(folder/'freeze.json',{'temperature':temp,'model_sha256':sha256(folder/'model.pt'),'protocol_sha256':sha256(folder/'protocol.json'),'best_WPY_NLL':best})
    for key in keys(['J03WQQ']):
        timeguard();x,z=inputs(key);row=pack(x,z);raw=infer(model,row)
        np.savez_compressed(folder/(key+'.npz'),raw_logit=raw,logit=raw/temp)
        print('MEMORY predicted',key,flush=True)

def evaluate(folder):
    from scripts.sprint_stack import evaluate as shared_evaluate,REPORT
    REPORT.mkdir(parents=True,exist_ok=True);shared_evaluate(folder)

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=1500);p.add_argument('--seed',type=int,default=20261024);p.add_argument('--width',type=int,default=32)
    p.add_argument('--extra-full');p.add_argument('--extra-fold0');p.add_argument('--extra-fold1');p.add_argument('--extra-config',type=Path);a=p.parse_args()
    if a.extra_full or a.extra_fold0 or a.extra_fold1:
        if not(a.extra_full and a.extra_fold0 and a.extra_fold1):raise ValueError('Need full and both OOF extra models')
        stack_source.EXTRAS={'full':a.extra_full,'0':a.extra_fold0,'1':a.extra_fold1}
    if a.extra_config:
        stack_source.EXTRAS_ADDITIONAL=json.loads(a.extra_config.read_text())['models']
        for mapping in stack_source.EXTRAS_ADDITIONAL:
            if set(mapping)!={'full','0','1'}:raise ValueError('Additional experts need disjointfolds and full model')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(guard)
    folder=ROOT/'runs/two-hour-sprint/memory'/a.name;train(folder,a.steps,a.seed,a.width);evaluate(folder)

if __name__=='__main__':main()
