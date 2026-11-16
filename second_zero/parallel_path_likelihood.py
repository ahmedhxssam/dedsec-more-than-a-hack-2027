"""Exact three-state conditional likelihood via associative log-matrix products."""
import numpy as np
import torch
from torch import nn


def multiply(left, right):
    values = left[..., :, :, None] + right[..., None, :, :]
    # Identity padding has unreachable entries. Avoid differentiating an
    # all-minus-infinity logsumexp; those entries have exactly zero gradient.
    reachable = torch.isfinite(values).any(-2)
    safe = torch.where(reachable[..., None, :], values, torch.zeros_like(values))
    result = torch.logsumexp(safe, -2)
    return torch.where(reachable, result, torch.full_like(result, -torch.inf))


def path_nll(logp, change, truth, mask):
    active = mask.bool()
    if not active.any():
        return logp.sum()*0
    b, t, c = logp.shape
    if c != 3 or change.shape != (b,t) or truth.shape != (b,t) or active.shape != (b,t):
        raise ValueError('Aligned three-state sequences required')
    eye = torch.eye(3, dtype=torch.bool, device=logp.device)
    identity = torch.where(eye, logp.new_tensor(0.), logp.new_tensor(-torch.inf))
    previous = torch.cat([torch.zeros((b,1),dtype=torch.bool,device=logp.device),active[:,:-1]],1)
    cont = active & previous
    stay = -nn.functional.softplus(change)
    switch = -nn.functional.softplus(-change)-np.log(2.)
    transition = torch.where(eye[None,None],stay[...,None,None],switch[...,None,None])
    # At a reset, every incoming row is the new emission. Its matrix product
    # carries forward the preceding segment's partition, multiplying independent
    # segment partitions rather than connecting states across ungraded frames.
    matrix = logp[:,:,None,:]+torch.where(cont[...,None,None],transition,torch.zeros_like(transition))
    matrix = torch.where(active[...,None,None],matrix,identity[None,None])
    while matrix.shape[1] > 1:
        if matrix.shape[1] % 2:
            matrix = torch.cat([matrix,identity[None,None].expand(b,1,3,3)],1)
        matrix = multiply(matrix[:,0::2],matrix[:,1::2])
    partition = torch.logsumexp(matrix[:,0].reshape(b,-1),1)-np.log(3.)
    y = truth.long().clamp(0,2)
    emission_score = (logp.gather(2,y[...,None]).squeeze(-1)*active).sum(1)
    prev_y = torch.cat([y[:,:1],y[:,:-1]],1)
    transition_score = (torch.where(y==prev_y,stay,switch)*cont).sum(1)
    return (partition-emission_score-transition_score).sum()/active.sum().clamp_min(1)
