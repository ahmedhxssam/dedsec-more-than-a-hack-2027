"""Equivalent native architecture with explicit depthwise shifts and tree CRF."""
import numpy as np
import torch
from torch import nn
from second_zero import native_joint_recovery as original
from second_zero import joint_corrupted_recovery as old
from second_zero.parallel_path_likelihood import path_nll as parallel_path_nll

DILATIONS, RADIUS = original.DILATIONS, original.RADIUS
LENGTH, MARGIN, CENTRAL, CHANNELS = original.LENGTH, original.MARGIN, original.CENTRAL, original.CHANNELS
steps_per_match = original.steps_per_match
inputs, targets, joined, batch, predict, compose = original.inputs, original.targets, original.joined, original.batch, original.predict, original.compose


class ShiftDepthwise(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        if not (convolution.groups == convolution.in_channels == convolution.out_channels and
                convolution.kernel_size == (5,) and convolution.stride == (1,) and
                convolution.padding == (2*convolution.dilation[0],)):
            raise ValueError('Exact registered five-tap depthwise convolution required')
        self.weight = convolution.weight
        self.bias = convolution.bias
        self.dilation = convolution.dilation[0]

    def forward(self, x):
        n = x.shape[-1]
        padded = nn.functional.pad(x, (2*self.dilation, 2*self.dilation))
        out = padded[:, :, :n]*self.weight[:,0,0][None,:,None]
        for tap in range(1,5):
            a = tap*self.dilation
            out = out + padded[:, :, a:a+n]*self.weight[:,0,tap][None,:,None]
        return out + self.bias[None,:,None]


class JointRecovery(original.JointRecovery):
    def __init__(self):
        super().__init__()
        for block in self.blocks:
            block[0] = ShiftDepthwise(block[0])


def loss(model, v):
    owner, dead, change = model(v['x'])
    logp = old.emissions(owner, dead)
    w = v['weight']
    known = (w > 0) & (v['y'] < 3)
    y = v['y'].clamp(0, 2).long()
    error = -logp.gather(2, y[..., None]).squeeze(-1)
    frame = .5 * (error * w).sum() / w.sum().clamp_min(1.) + .5 * error[known].mean()
    mask = known.clone()
    mask[:, 0] = False
    mask[:, 1:] &= v['y'][:, :-1] < 3
    boundary = nn.functional.binary_cross_entropy_with_logits(change, v['edge'], reduction='none', pos_weight=change.new_tensor(25.))
    boundary = boundary[mask].mean() if mask.any() else boundary.sum() * 0
    # The full 25Hz frame and edge tasks retain short spells. A 5Hz structured
    # regularizer controls long paths without reducing the input/output rate.
    structured = parallel_path_nll(logp[:, ::5], change[:, ::5] - np.log(5.), y[:, ::5], known[:, ::5])
    return frame + .1 * boundary + .1 * structured + .0001 * (owner.square() + dead.square())[known].mean()

