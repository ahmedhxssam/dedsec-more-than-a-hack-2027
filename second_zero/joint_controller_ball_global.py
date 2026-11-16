"""Add the complete controller posterior to ball reconstruction.

The initial model's usual bounded-gap path aggregates each team's attended
players equally. This version also carries cross-team probability mass, the
global controller-weighted embedding and its observed spatial center. Adding
zero-initialized input columns preserves the warmstart's function.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .joint_controller_ball import JointControllerBall, control_probabilities


class GlobalJointControllerBall(nn.Module):
    def __init__(self,width=32,global_context=True):
        super().__init__();self.width=width;self.global_context=global_context
        self.network=JointControllerBall(width=width,feedback=True)
        original=self.network.ball_context.input
        expanded=nn.Conv1d(original.in_channels+width+4,original.out_channels,1)
        with torch.no_grad():
            expanded.weight.zero_();expanded.weight[:,:original.in_channels]=original.weight
            expanded.bias.copy_(original.bias)
        self.original_channels=original.in_channels
        self.network.ball_context.input=expanded
        self._values=None;self._controller=None;self.last_result=None
        self.network.actor.register_forward_hook(self._capture_controller)
        self.network.ball_context.register_forward_pre_hook(self._supply_context)

    def load_warmstart(self,state):
        own=self.network.state_dict()
        with torch.no_grad():
            for name,value in state.items():
                if name=='ball_context.input.weight':
                    own[name].zero_();own[name][:,:value.shape[1]].copy_(value)
                else:own[name].copy_(value)
        self.network.load_state_dict(own)

    def _capture_controller(self,module,inputs,output):
        self._controller=(inputs[0],output[...,0])

    def _supply_context(self,module,inputs):
        original=inputs[0];h,scores=self._controller
        values=self._values;x=values['player'];b,n,teams,p,_=x.shape
        if self.global_context:
            present=x[...,-1]>.5
            active=F.max_pool1d(present.permute(0,2,3,1).reshape(b*2*p,1,n).float(),
                251,stride=1,padding=125).reshape(b,2,p,n).permute(0,3,1,2)>.5
            prob,_=control_probabilities(scores,active)
            global_embedding=(prob[...,None]*h).sum((2,3))
            observed=prob*present.to(h.dtype);mass=observed.sum((2,3))
            center=(observed[...,None]*values['position']).sum((2,3))/mass[...,None].clamp_min(1e-8)
            # The original feature's last four channels are reference XY and
            # measured/bounded flags. No label or hidden coordinate enters here.
            delta=(center-original[...,-4:-2])*(mass>0)[...,None]
            entropy=-(prob*prob.clamp_min(1e-8).log()).sum((2,3))/active.sum((2,3)).clamp_min(2).to(h.dtype).log()
            extra=torch.cat([global_embedding,delta,mass[...,None],entropy[...,None]],-1)
        else:
            extra=original.new_zeros((b,n,self.width+4))
        return (torch.cat([original,extra],-1),)

    def forward(self,values):
        self._values=values
        try:
            result=self.network(values);self.last_result=result;return result
        finally:
            self._values=None;self._controller=None
