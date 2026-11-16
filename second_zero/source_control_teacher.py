"""Source-only conditional control identity and documented possession-start teacher.

The raw-tracking and Dynamic Event actor conventions have separate heads.
Unknown actors are never transit/negative labels. There is no primary ownership
head, memory variable, or Sportec label input. Native temporal features retain
short impulses before combining with coarse trajectory/social context.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Mapping
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from .aggressive_controller import CHANNELS,PlayerSocial,ResetTrackBlock


@dataclass(frozen=True)
class TeacherConfig:
    channels:int=CHANNELS
    width:int=64
    dropout:float=.1


class SourceControlTeacher(nn.Module):
    def __init__(self,config:TeacherConfig|None=None):
        super().__init__();self.config=config or TeacherConfig();c=self.config
        if c.width%4 or c.width<4 or c.channels!=CHANNELS:raise ValueError('Use38 channels and width divisible by4')
        self.architecture=asdict(c);self.required_context=200;self.radius=200;self.coarse_factor=5;self.factor=5
        self.input=nn.Linear(c.channels,c.width)
        self.native=nn.ModuleList(ResetTrackBlock(c.width,d,c.dropout)for d in(1,2,4))
        self.before=nn.ModuleList(ResetTrackBlock(c.width,d,c.dropout)for d in(1,2,4))
        self.social=PlayerSocial(c.width)
        self.after=nn.ModuleList(ResetTrackBlock(c.width,d,c.dropout)for d in(8,16))
        self.fusion=nn.Linear(2*c.width,c.width)
        self.raw_actor_head=nn.Linear(c.width,1)
        self.controlled_actor_head=nn.Linear(c.width,1)
        self.pp_start_head=nn.Linear(c.width,1)
        self.controlled_actor_head.load_state_dict(self.raw_actor_head.state_dict())

    def forward(self,values:Mapping[str,Tensor],core_start:int=0,core_stop:int|None=None):
        if set(values)-{'player','position','position_available','team','observed','track_jumps','ball'}:
            raise ValueError('Teacher inputs must contain observations only')
        x=values['player'];b,n,p,c=x.shape;w=self.config.width
        if not n or not p or c!=self.config.channels:raise ValueError('Nonempty[B,N,P,38] required')
        teams=values['team'].long()
        if teams.shape!=(b,p)or not((teams>=-1)&(teams<=1)).all():raise ValueError('Fixed[B,P] team codes required')
        active=values['position_available'].bool()&(teams>=0)[:,None]
        if active.shape!=(b,n,p)or values['position'].shape!=(b,n,p,2):raise ValueError('Observation shapes differ')
        stop=n if core_stop is None else core_stop
        if not 0<=core_start<stop<=n:raise ValueError('Invalid core window')
        jumps=values.get('track_jumps',x[...,-1]>.5).bool().clone()
        jumps[:,1:]|=active[:,1:]!=active[:,:-1]
        stem=F.gelu(self.input(x.contiguous().reshape(b*n*p,c))).reshape(b,n,p,w)
        stem=stem.masked_fill(~active[...,None],0)
        native=stem.permute(0,2,1,3).contiguous().reshape(b*p,n,w)
        native_break=jumps.permute(0,2,1).contiguous().reshape(b*p,n)
        for block in self.native:native=block(native,native_break)
        native=native.reshape(b,p,n,w).permute(0,2,1,3).contiguous()
        coarse=F.avg_pool1d(stem.permute(0,2,3,1).contiguous().reshape(b*p,w,n),5,5,2)
        length=coarse.shape[-1]
        coarse_break=F.max_pool1d(native_break[:,None].float(),5,5,2)[:,0]
        coarse=coarse.transpose(1,2).contiguous()
        for block in self.before:coarse=block(coarse,coarse_break)
        av=active.float().permute(0,2,1).contiguous().reshape(b*p,1,n)
        avc=F.avg_pool1d(av,5,5,2)
        xy=values['position'].permute(0,2,3,1).contiguous().reshape(b*p,2,n)*av
        xy=(F.avg_pool1d(xy,5,5,2)/avc.clamp_min(1e-8)).reshape(b,p,2,length).permute(0,3,1,2).contiguous()
        available=(avc[:,0]>0).reshape(b,p,length).permute(0,2,1).contiguous()
        coarse=coarse.reshape(b,p,length,w).permute(0,2,1,3).contiguous()
        coarse=self.social(coarse,xy,available,available,teams)
        coarse=coarse.permute(0,2,1,3).contiguous().reshape(b*p,length,w)
        for block in self.after:coarse=block(coarse,coarse_break)
        coarse=F.interpolate(coarse.transpose(1,2),scale_factor=5.,mode='linear',align_corners=False)[...,:n]
        coarse=coarse.reshape(b,p,w,n).permute(0,3,1,2).contiguous()
        h=F.gelu(self.fusion(torch.cat((native,coarse),-1).reshape(b*n*p,2*w))).reshape(b,n,p,w)
        h=h[:,core_start:stop];a=active[:,core_start:stop];t=stop-core_start
        flat=h.contiguous().reshape(b*t*p,w)
        result={name:head(flat).reshape(b,t,p).masked_fill(~a,-1e4)for name,head in(
            ('raw_actor_logits',self.raw_actor_head),('controlled_actor_logits',self.controlled_actor_head),
            ('pp_start_logits',self.pp_start_head))}
        result['active']=a
        return result


def actor_loss(logits:Tensor,labels:Tensor,active:Tensor):
    if logits.shape!=active.shape or labels.shape!=logits.shape[:2]:raise ValueError('Actor target shapes differ')
    if ((labels<-1)|(labels>=logits.shape[-1])).any():raise ValueError('Actor targets must be−1 or valid player slots')
    index=labels.clamp_min(0).long();known=(labels>=0)&active.gather(2,index[...,None])[...,0]
    logp=F.log_softmax(logits.masked_fill(~active,-1e4),-1)
    loss=-logp.gather(2,index[...,None])[...,0]
    return torch.where(known,loss,0).sum()/known.sum().clamp_min(1),known.sum()


def conditional_actor_probabilities(logits:Tensor,active:Tensor):
    mass=logits.masked_fill(~active,-1e4).softmax(-1)*active.to(logits.dtype)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-8)
