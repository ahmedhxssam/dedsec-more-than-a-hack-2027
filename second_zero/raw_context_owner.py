"""Raw-only temporal ownership: no learned feature or teacher ancestors."""
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from second_zero import global_team_context as context
from second_zero.temporal_ownership import trajectory_features
from second_zero.sprint_model_interpolation import interpolated_features
from second_zero import masked_gap_training as original

CHANNELS=414
batch=original.ORIGINAL_BATCH


def fitting_matches(train,held):
    if held not in train:raise ValueError('Held match outside primary cohort')
    return [m for m in train if m!=held]


class RawContextTemporal(original.ORIGINAL_TEMPORAL):
    def __init__(self):
        super().__init__();self.input=torch.nn.Conv1d(CHANNELS,32,1)


def inputs(raw):
    if set(raw)!={'ball','players','team','pitch'}:raise ValueError('Raw observations only')
    g=context.features(raw['players'],raw['team'],raw['pitch']);o=trajectory_features(raw['ball'],raw['players'],raw['team'],raw['pitch']);i=interpolated_features(raw['ball'],raw['players'],raw['team'],raw['pitch'])
    missing=(~np.isfinite(raw['ball']).all(1)).astype(np.float32);support=uniform_filter1d(missing.astype(np.float64),25,mode='nearest').astype(np.float32)
    flag=np.repeat(np.c_[missing,support][:,None],2,axis=1);features=np.concatenate([g,o,i,flag],axis=-1);features=uniform_filter1d(np.clip(features,-5,5),5,axis=0,mode='nearest')[::5];x=features[:,[1,0]].transpose(1,2,0).copy().astype(np.float16).astype(np.float32)
    if x.shape[1]!=CHANNELS or not np.isfinite(x).all():raise ValueError('Invalid raw temporal features')
    return x,np.zeros(x.shape[-1],np.float32),support


def loss(model,v):
    z,b,absolute=model(v['x'],v['prior']);w=v['weight'];den=w.sum().clamp_min(1.)
    owner=(torch.nn.functional.binary_cross_entropy_with_logits(z,v['y'],reduction='none')*w).sum()/den
    event=(torch.nn.functional.binary_cross_entropy_with_logits(b,v['boundary'],reduction='none',pos_weight=torch.tensor(10.,device=b.device))*(w>0)).sum()/(w>0).sum().clamp_min(1.)
    return owner+.05*event+.001*(absolute.square()*w).sum()/den


def predict(model,x,prior,n,device='cpu'):
    return original.ORIGINAL_PREDICT(model,x,np.zeros_like(prior),n,device)
