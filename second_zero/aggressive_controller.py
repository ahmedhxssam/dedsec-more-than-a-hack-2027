"""Controller-aware transfer with separate provider and target team heads.

Source player IDs supervise the controller distribution, never a nearest-player
proxy. The source's team-control convention and Sportec possession convention
have separate readouts. Unknown controllers do not become transit labels.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .aggressive_sequence import MultiscaleBlock

CHANNELS=38
FEATURE_CONTEXT=251


def _bounded(values,maximum=250):
    values=np.asarray(values,np.float32);n=len(values)
    if values.ndim==2:values=values[:,None]
    out=values.copy();seen=np.isfinite(values).all(-1);ix=np.arange(n)[:,None]
    previous=np.maximum.accumulate(np.where(seen,ix,-n),axis=0)
    following=np.minimum.accumulate(np.where(seen,ix,2*n)[::-1],axis=0)[::-1]
    eligible=(~seen)&(previous>=0)&(following<n)&(following-previous<=maximum)
    for j in range(values.shape[1]):
        known=np.flatnonzero(seen[:,j])
        if len(known)>=2:
            for a in range(2):out[eligible[:,j],j,a]=np.interp(np.flatnonzero(eligible[:,j]),known,values[known,j,a])
    age=np.clip(np.stack([ix-previous,following-ix],-1),0,maximum)/maximum
    return out,seen,age.astype(np.float32)


def _velocity(x,maximum):
    v=np.full_like(x,np.nan)
    v[1:-1]=(x[2:]-x[:-2])*12.5
    valid=np.isfinite(v).all(-1)&(np.linalg.norm(v,axis=-1)<=maximum)
    v[~valid]=np.nan
    return v,valid


def coordinate_age(players,maximum=250):
    """Time since/until a provided coordinate changes; not a freshness verdict."""
    players=np.asarray(players);n,p=players.shape[:2];finite=np.isfinite(players).all(-1)
    changed=np.zeros((n,p),bool);changed[0]=finite[0]
    displacement=np.linalg.norm(players[1:]-players[:-1],axis=-1)
    changed[1:]=finite[1:]&((~finite[:-1])|(displacement>1e-4))
    ix=np.arange(n)[:,None]
    previous=np.maximum.accumulate(np.where(changed,ix,-maximum),axis=0)
    following=np.minimum.accumulate(np.where(changed,ix,n+maximum)[::-1],axis=0)[::-1]
    age=np.clip(np.stack([ix-previous,following-ix],-1),0,maximum)/maximum
    jump=np.zeros((n,p),bool);jump[1:]=finite[1:]&finite[:-1]&(displacement>2.)
    return np.concatenate([age,jump[...,None]],-1).astype(np.float32)


def held_coordinates(raw,slot,start,stop):
    """Simulate a finite but stale slot, preserving the original missing mask."""
    out=dict(raw);players=np.asarray(raw['players']);finite=np.isfinite(players[:,slot]).all(-1)
    start=max(0,start);stop=min(len(players),stop);known=np.flatnonzero(finite[start:stop])+start
    if not len(known):return out
    out['players']=players.copy();out['players'][known,slot]=players[known[0],slot]
    return out


def features(raw):
    ball=np.asarray(raw['ball'],np.float32);players=np.asarray(raw['players'],np.float32);pitch=np.asarray(raw['pitch'])
    n,p=players.shape[:2]
    eb,bs,ba=_bounded(ball);eb=eb[:,0];bs=bs[:,0];ba=ba[:,0]
    ep,ps,pa=_bounded(players)
    bv,bvv=_velocity(ball,60.);pv,pvv=_velocity(players,15.)
    ev,evv=_velocity(ep,15.);ebv,ebvv=_velocity(eb,60.)
    rel=players-ball[:,None];dist=np.linalg.norm(rel,axis=-1);rv=pv-bv[:,None]
    erel=ep-eb[:,None];edist=np.linalg.norm(erel,axis=-1)
    speed=np.linalg.norm(pv,axis=-1);bspeed=np.linalg.norm(bv,axis=-1);ebspeed=np.linalg.norm(ebv,axis=-1)
    repeat=lambda v:np.broadcast_to(v[:,None],(n,p))
    exact=np.asarray(raw.get('source_exact',np.ones(n,bool)))
    bounded=np.isfinite(ep).all(-1);ballbounded=np.isfinite(eb).all(-1)
    co=np.sum(ev*ebv[:,None],-1)/np.maximum(np.linalg.norm(ev,axis=-1)*ebspeed[:,None],.1)
    values=np.concatenate([
        (players-pitch/2)/50,pv/12,speed[...,None]/12,rel/20,dist[...,None]/20,
        rv/30,np.linalg.norm(rv,axis=-1)[...,None]/30,
        (np.sum(rel*rv,-1)/np.maximum(dist,.1))[...,None]/30,
        (ep-pitch/2)/50,erel/20,edist[...,None]/20,ev/12,co[...,None],
        np.stack([ps,bounded,pvv,np.isfinite(dist),np.isfinite(edist),repeat(exact)],-1),
        pa,np.broadcast_to(ba[:,None],(n,p,2)),
        np.stack([repeat(ebspeed/40),repeat(bs),repeat(ballbounded),repeat(bvv),repeat(bspeed/40)],-1),
        coordinate_age(players)
    ],-1)
    assert values.shape==(n,p,CHANNELS),values.shape
    return {'player':np.clip(np.nan_to_num(values,nan=0,posinf=0,neginf=0),-5,5).astype(np.float32),
            'position':np.nan_to_num((ep-pitch/2)/50,nan=0).astype(np.float32),
            'observed':ps,'position_available':bounded,'team':np.asarray(raw['team'],np.int64)}


def crop_features(raw,start,stop):
    n=len(raw['ball']);lo=max(0,start-FEATURE_CONTEXT);hi=min(n,stop+FEATURE_CONTEXT)
    local={k:(v[lo:hi] if k in ('ball','players','source_exact') else v) for k,v in raw.items()}
    values=features(local);a=max(0,start)-lo;b=min(n,stop)-lo
    return {k:(v if k=='team' else np.pad(v[a:b],((max(0,-start),max(0,stop-n)),)+((0,0),)*(v.ndim-1))) for k,v in values.items()}


def pack_values(values,minimum=28,maximum=None):
    """Remove slots unavailable throughout the entire model context, remap labels.

    The network masks these slots at every social/pooling operation. Packing
    therefore preserves its function while avoiding computation on bench slots.
    """
    cols=np.flatnonzero(np.asarray(values['position_available']).any(0))
    if maximum is not None and len(cols)>maximum:raise ValueError('Active slot union exceeds audited limit')
    size=max(minimum,len(cols));inverse=np.full(len(values['team']),-1,np.int64);inverse[cols]=np.arange(len(cols))
    out={}
    for k,v in values.items():
        if k=='team':out[k]=np.pad(v[cols],(0,size-len(cols)),constant_values=-1)
        else:out[k]=np.pad(v[:,cols],((0,0),(0,size-len(cols)))+((0,0),)*(v.ndim-2))
    return out,inverse


class PlayerSocial(nn.Module):
    def __init__(self,width=64,heads=4):
        super().__init__();self.width=width;self.heads=heads
        self.qkv=nn.Linear(width,3*width);self.output=nn.Linear(width,width);self.norm=nn.LayerNorm(width)
        self.distance_scale=nn.Parameter(torch.zeros(heads));self.same_team=nn.Parameter(torch.zeros(heads))

    def forward(self,h,position,active,available,team):
        b,n,p,w=h.shape;tokens=h.reshape(b*n,p,w);qkv=self.qkv(tokens).reshape(b*n,p,3,self.heads,w//self.heads)
        q,k,v=[qkv[:,:,j].transpose(1,2) for j in range(3)]
        xy=position.reshape(b*n,p,2);known=available.reshape(b*n,p)
        distance=torch.linalg.vector_norm(xy[:,:,None]-xy[:,None],dim=-1)
        distance=distance*(known[:,:,None]&known[:,None]).to(distance.dtype)
        same=(team[:,:,None]==team[:,None]).to(distance.dtype)[:,None].expand(-1,n,-1,-1).reshape(b*n,p,p)
        score=q@k.transpose(-2,-1)/(w//self.heads)**.5
        score=score-F.softplus(self.distance_scale)[None,:,None,None]*distance[:,None]
        score=score+self.same_team[None,:,None,None]*same[:,None]
        mask=active.reshape(b*n,p)
        attention=score.masked_fill(~mask[:,None,None],-1e4).softmax(-1)*mask[:,None,None]
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        result=(attention@v).transpose(1,2).reshape(b*n,p,w)
        return self.norm(tokens+self.output(result)).reshape(b,n,p,w)


class ResetTrackBlock(nn.Module):
    """Dilated per-slot messages stop at an observed jump or availability break."""
    def __init__(self,width,dilation,dropout):
        super().__init__();self.width=width;self.dilation=dilation
        self.transforms=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(3))
        self.bias=nn.Parameter(torch.zeros(width));self.norm=nn.LayerNorm(width);self.dropout=nn.Dropout(dropout)

    def linear(self,layer,h):
        return layer(h.reshape(-1,self.width)).reshape_as(h)

    def forward(self,h,breaks):
        d,n=self.dilation,h.shape[1];v=self.linear(self.transforms[1],h)+self.bias
        if n>d:
            cumulative=breaks.cumsum(1)
            mask=(cumulative[:,d:]-cumulative[:,:-d]==0).to(h.dtype)[...,None]
            left=self.linear(self.transforms[0],h[:,:-d])*mask
            right=self.linear(self.transforms[2],h[:,d:])*mask
            v=v+F.pad(left,(0,0,d,0))+F.pad(right,(0,0,0,d))
        return h+.5*self.dropout(F.gelu(self.norm(v)))


class ActorController(nn.Module):
    def __init__(self,channels=CHANNELS,width=64,team_width=96,dropout=.1):
        super().__init__();self.width=width
        self.architecture=dict(channels=channels,width=width,team_width=team_width,dropout=dropout)
        self.before_dilations=(1,2,4);self.after_dilations=(8,16);self.team_dilations=(1,2,4,8,16,32)
        self.factor=5
        # Coarseplayer context155f +pool/linearupsampling<10f +native63f.
        # Rounded to the pooling period so chunk starts share a global phase.
        self.radius=230
        self.input=nn.Linear(channels,width)
        self.before=nn.ModuleList(ResetTrackBlock(width,d,dropout) for d in self.before_dilations)
        self.social=PlayerSocial(width)
        self.after=nn.ModuleList(ResetTrackBlock(width,d,dropout) for d in self.after_dilations)
        self.fusion=nn.Linear(2*width,width)
        self.controller=nn.Linear(width,1)
        self.transit=nn.Linear(width,1);nn.init.constant_(self.transit.bias,-3.)
        self.team_input=nn.Conv1d(2*(3*width+2),team_width,1)
        self.team_blocks=nn.ModuleList(MultiscaleBlock(team_width,d,dropout) for d in self.team_dilations)
        self.target_owner=nn.Conv1d(team_width,1,1)
        self.source_group=nn.Conv1d(team_width,1,1)

    def forward(self,values):
        x=values['player'];b,n,p,c=x.shape
        # Current measured or bounded-interpolated positions retain the slot;
        # a completely unavailable player never receives a fabricated positive.
        active=values['position_available'].bool()
        # Flatten Linear's input explicitly: MPS2.9 otherwise propagates a
        # channels-last gradient marker from the following4D permutation into
        # LinearBackward's3D workspace.
        stem=F.gelu(self.input(x.reshape(b*n*p,c))).reshape(b,n,p,self.width)
        h=stem.permute(0,2,3,1).contiguous().reshape(b*p,self.width,n)
        h=F.avg_pool1d(h,5,5,2)
        coarse=h.shape[-1]
        native_break=x[...,-1]>.5
        native_break=native_break.clone()
        native_break[:,1:]|=active[:,1:]!=active[:,:-1]
        coarse_break=F.max_pool1d(native_break.permute(0,2,1).reshape(b*p,1,n).float(),5,5,2)[:,0]
        h=h.transpose(1,2)
        for block in self.before:h=block(h,coarse_break)
        h=h.reshape(b,p,coarse,self.width).permute(0,2,1,3).contiguous()
        av=values['position_available'].float().permute(0,2,1).reshape(b*p,1,n)
        avc=F.avg_pool1d(av,5,5,2)
        xy=values['position'].permute(0,2,3,1).reshape(b*p,2,n)*av
        xy=(F.avg_pool1d(xy,5,5,2)/avc.clamp_min(1e-8)).reshape(b,p,2,coarse).permute(0,3,1,2)
        available=(avc[:,0]>0).reshape(b,p,coarse).permute(0,2,1)
        h=self.social(h,xy,available,available,values['team'])
        h=h.permute(0,2,1,3).contiguous().reshape(b*p,coarse,self.width)
        for block in self.after:h=block(h,coarse_break)
        h=F.interpolate(h.transpose(1,2),scale_factor=5.,mode='linear',align_corners=False)[...,:n]
        h=h.reshape(b,p,self.width,n).permute(0,3,1,2).contiguous()
        joined=torch.cat([stem,h],-1).reshape(b*n*p,2*self.width)
        h=F.gelu(self.fusion(joined)).reshape(b,n,p,self.width)
        score=self.controller(h.reshape(b*n*p,self.width)).reshape(b,n,p).masked_fill(~active,-1e4)
        actor_logp=score.log_softmax(-1)
        team=values['team'];pool=[];transit=[]
        for t in (0,1):
            mask=(team[:,None]==t)&active
            mass=mask.sum(2).clamp_min(1)[...,None]
            mean=(h*mask[...,None]).sum(2)/mass
            maximum=h.masked_fill(~mask[...,None],-1e4).amax(2)
            maximum=torch.where(mask.any(2)[...,None],maximum,torch.zeros_like(maximum))
            attention=score.masked_fill(~mask,-1e4).softmax(2)*mask
            attention=attention/attention.sum(2,keepdim=True).clamp_min(1e-8)
            attended=(h*attention[...,None]).sum(2)
            controller_mass=score.masked_fill(~mask,-1e4).logsumexp(2)
            tr=self.transit(mean)[...,0];transit.append(torch.logaddexp(controller_mass,tr))
            pool.append(torch.cat([mean,maximum,attended,(mask.sum(2)/11)[...,None],torch.sigmoid(controller_mass-tr)[...,None]],-1))
        pool=torch.stack(pool,1);paired=torch.cat([pool,pool.flip(1)],-1).permute(0,1,3,2).reshape(b*2,-1,n)
        z=F.gelu(self.team_input(paired))
        for block in self.team_blocks:z=block(z)
        target=self.target_owner(z).reshape(b,2,n);source=self.source_group(z).reshape(b,2,n)
        controller=transit[1]-transit[0]
        return {'owner':target[:,1]-target[:,0]+.5*controller,
                'source_group':source[:,1]-source[:,0]+controller,
                'actor_logp':actor_logp,'active':active,'controller_owner':controller}


def actor_loss(logp,labels,active):
    index=labels.clamp(0,logp.shape[-1]-1).long()
    mask=(labels>=0)&active.gather(2,index[...,None])[...,0]
    loss=-logp.gather(2,index[...,None])[...,0]
    return (loss*mask).sum()/mask.sum().clamp_min(1),mask.sum()
