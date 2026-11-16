"""Absolute gap reconstruction: hide all in-gap ownership shortcuts."""
import numpy as np
import torch
from second_zero import temporal_gap_residual as temporal
from second_zero.temporal_ownership import trajectory_features
from second_zero.sprint_model_interpolation import interpolated_features
from scipy.ndimage import uniform_filter1d

ORIGINAL_BATCH=temporal.batch
ORIGINAL_PREDICT=temporal.predict
ORIGINAL_TEMPORAL=temporal.TemporalResidual
CHANNELS=429
_SOURCE=None
_DATA=None


class RawGeometryTemporal(ORIGINAL_TEMPORAL):
    def __init__(self):
        super().__init__()
        self.input=torch.nn.Conv1d(CHANNELS,temporal.WIDTH,1)


def raw_geometry(raw):
    observed=trajectory_features(raw['ball'],raw['players'],raw['team'],raw['pitch'])
    linear=interpolated_features(raw['ball'],raw['players'],raw['team'],raw['pitch'])
    raw_features=np.concatenate([observed,linear],axis=-1)
    smoothed=uniform_filter1d(np.clip(raw_features,-5,5),5,axis=0,mode='nearest')[::5]
    result=smoothed[:,[1,0]].transpose(1,2,0).copy()
    if result.shape[1]!=122 or not np.isfinite(result).all():raise ValueError('Invalid raw geometry features')
    return result


def mask_features(x):
    out=np.asarray(x).copy();mask=out[:,306,:]>0
    out[:,290:305,:]*=(~mask)[:,None,:]
    return out


def training_data(data):
    result=[]
    for d in data:
        v={**d};gap=d['x'][0,306]>0;v['weight']=d['weight']*gap;v['prior']=np.zeros_like(d['prior']);events=d['events'];inside=(events>=0)&(events<len(gap));events=events[inside];v['events']=events[v['weight'][events]>0]
        if not (v['weight']>0).any():raise ValueError('No eligible gap training frames')
        result.append(v)
    return result


def batch(data,rng,batch_size=8,length=128):
    global _SOURCE,_DATA
    if _SOURCE is not data:_SOURCE=data;_DATA=training_data(data)
    values=ORIGINAL_BATCH(_DATA,rng,batch_size,length);gap=values['x'][:,306,:]>0
    values['x'][:,290:305,:]*=(~gap)[:,None,:]
    return values


def predict(model,x,prior,n,device='cpu'):
    return ORIGINAL_PREDICT(model,mask_features(x),np.zeros_like(prior),n,device)
