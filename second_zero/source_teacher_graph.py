"""Observation-only source teacher summaries and team-relative graph additions."""
from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from .aggressive_controller import crop_features
from .source_control_teacher import conditional_actor_probabilities
from .sprint_decoder import decode

CHANNELS=('DE_team_logodds','raw_team_logodds','PP_team_contrast','DE_entropy','raw_entropy','head_TV','maximum_PP','any_node')


def summarize(output,teams):
    active=output['active'];prob=[conditional_actor_probabilities(output[k].float(),active)for k in ('controlled_actor_logits','raw_actor_logits')]
    available=active.any(-1);count=active.sum(-1).clamp_min(2).float()
    mass=[torch.stack([(p*(teams[:,None]==t)).sum(-1)for t in (0,1)],-1)for p in prob]
    odds=[(m[...,1].clamp_min(1e-6).log()-m[...,0].clamp_min(1e-6).log()).clamp(-12,12)for m in mass]
    entropy=[-(p*p.clamp_min(1e-8).log()).sum(-1)/count.log()for p in prob]
    pp=torch.sigmoid(output['pp_start_logits'].float())
    starts=torch.stack([(prob[0]*pp*(teams[:,None]==t)).sum(-1)for t in (0,1)],-1)
    channels=torch.stack([odds[0],odds[1],starts[...,1]-starts[...,0],*entropy,(prob[0]-prob[1]).abs().sum(-1)/2,starts.max(-1).values,available.float()],-1)
    channels=torch.where(available[...,None],channels,0)
    team_available=torch.stack([(active&(teams[:,None]==t)).any(-1)for t in (0,1)],-1)
    return {k:v[0].cpu().numpy()for k,v in {'channels':channels,'PP_team':starts,'team_available':team_available}.items()}


@torch.no_grad()
def infer(model,raw,device='mps',chunk=2000):
    """Fixed original full-half roster and phase; no labels/status input."""
    if chunk%5:raise ValueError('Chunk must preserve phase5')
    model.eval();n=len(raw['ball']);out={k:[]for k in ('channels','PP_team','team_available')}
    for start in range(0,n,chunk):
        stop=min(n,start+chunk);left=start-200;right=((stop+4)//5)*5+200
        values=crop_features(raw,left,right)
        values={k:torch.from_numpy(np.ascontiguousarray(v))[None].to(device)for k,v in values.items()}
        output=model(values,core_start=200,core_stop=200+stop-start)
        result=summarize(output,values['team'])
        for k,v in result.items():out[k].append(v)
    return {k:np.concatenate(v)for k,v in out.items()}


def statistics(values,segments,widths,signed):
    s,e,t=segments.T;d=e-s;sign=(2*t-1)[:,None];n=len(values)
    prefix=np.vstack([np.zeros(values.shape[1]),np.cumsum(values.astype(np.float64),0)])
    squared=np.vstack([np.zeros(values.shape[1]),np.cumsum(values.astype(np.float64)**2,0)])
    mean=(prefix[e]-prefix[s])/d[:,None];std=np.sqrt(np.maximum(0,(squared[e]-squared[s])/d[:,None]-mean**2))
    mean[:,:signed]*=sign;result=[mean,std]
    for width in widths:
        lo=np.maximum(s-width,0);hi=np.minimum(e+width,n)
        before=(prefix[s]-prefix[lo])/np.maximum(1,s-lo)[:,None];after=(prefix[hi]-prefix[e])/np.maximum(1,hi-e)[:,None]
        before[:,:signed]*=sign;after[:,:signed]*=sign;result.extend([before,after])
    return np.concatenate(result,1)


def features(x,segments,old_quality,teacher):
    s,e,t=segments.T;d=e-s;n=len(x);sign=2*t-1
    common=np.c_[old_quality,np.log1p(d),np.sqrt(d)/25,d/250,statistics(x[:,7:13],segments,(9,31,125),3)]
    channels=teacher['channels'];pp=teacher['PP_team']
    # Boundary windows use input predictions only, including unavailable zeros.
    def window(values,index):
        prefix=np.vstack([np.zeros(values.shape[1]),np.cumsum(values.astype(np.float64),0)])
        lo=np.maximum(index-2,0);hi=np.minimum(index+3,n)
        return (prefix[hi]-prefix[lo])/np.maximum(1,hi-lo)[:,None]
    entry=window(pp,s);exit=window(pp,e-1)
    before=window(channels,np.maximum(0,s-1));after=window(channels,np.minimum(n-1,e))
    uncertainty=window(channels[:,3:5],s).mean(1);uncertainty_exit=window(channels[:,3:5],e-1).mean(1)
    boundaries=np.c_[entry[np.arange(len(t)),t],exit[np.arange(len(t)),1-t],before[:,0]*sign,after[:,0]*sign,before[:,1]*sign,after[:,1]*sign,uncertainty,uncertainty_exit]
    block=np.c_[statistics(channels,segments,(9,),3),boundaries]
    assert common.shape==(len(segments),52) and block.shape==(len(segments),40)
    assert np.isfinite(common).all() and np.isfinite(block).all()
    return common.astype(np.float32),np.c_[common,block].astype(np.float32)


def proposals(base,baseline,old,teacher):
    """Preserve old arcs; bounded source shorts and baseline-compatible backgrounds."""
    original=set(map(tuple,old.tolist()));result=set(original);origin={};nodes=set()
    def add(s,e,t,kind):
        if e>s:
            arc=(int(s),int(e),int(t));result.add(arc)
            if arc not in original:origin.setdefault(arc,kind)
    def short(s,e,t,lo,hi):
        for ds in (-2,0,2):
            for de in (-2,0,2):
                a=max(lo,s+ds);b=min(hi,e+de)
                if 0<b-a<=250:add(a,b,t,0);nodes.update((int(a),int(b)))
    live=base<2;islands=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
    z=uniform_filter1d(teacher['channels'][:,0],3,mode='nearest');path=decode(z,base,cost=3.)
    reverse=decode(-z,base,cost=3.);reverse[live]=1-reverse[live]
    cuts=np.r_[0,np.flatnonzero(baseline[1:]!=baseline[:-1])+1,len(base)]
    for lo,hi in zip(islands[:-1],islands[1:]):
        if not live[lo]:continue
        for candidate in (path,reverse):
            edges=np.r_[lo,np.flatnonzero(candidate[lo+1:hi]!=candidate[lo:hi-1])+lo+1,hi]
            for s,e in zip(edges[:-1],edges[1:]):
                if e-s<=250 and teacher['team_available'][s:e,int(candidate[s])].any():short(s,e,int(candidate[s]),lo,hi)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if baseline[lo]>=2:continue
        for t in (0,1):
            score=teacher['PP_team'][lo:hi,t];valid=teacher['team_available'][lo:hi,t]
            local=np.flatnonzero(valid&(score==maximum_filter1d(score,3,mode='nearest')))
            selected=[]
            for j in local[np.argsort(-score[local],kind='stable')]:
                if all(abs(int(j)-k)>=3 for k in selected):
                    selected.append(int(j));short(lo+j,lo+j+1,t,lo,hi)
                    if len(selected)==2:break
    all_nodes=np.unique(np.r_[old[:,:2].ravel(),list(nodes)])
    for lo,hi in zip(islands[:-1],islands[1:]):
        if not live[lo]:continue
        local=all_nodes[(all_nodes>=lo)&(all_nodes<=hi)]
        for i,s in enumerate(local[:-1]):
            for e in local[i+1:i+17]:
                if e-s>250:break
                for t in (0,1):add(s,e,t,0)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        t=int(baseline[lo])
        if t>=2:continue
        local=np.unique(np.r_[lo,all_nodes[(all_nodes>=lo)&(all_nodes<=hi)],hi])
        for i,s in enumerate(local[:-1]):
            add(lo,local[i+1],t,1);add(s,hi,t,1)
            for e in local[i+1:i+33]:
                if e-s>1250:break
                add(s,e,t,1)
    new=sorted(result-original);segments=np.asarray(new,np.int64).reshape(-1,3)
    return segments,np.array([origin[a]for a in new],np.int8)


def sample_indices(segments,origins,rng,limit=20000):
    """Provenance/duration stratification; independent of target credit."""
    if len(segments)<=limit:return np.arange(len(segments)),np.ones(len(segments))
    strata=(segments[:,1]-segments[:,0]>125).astype(int)*2+origins
    groups=[np.flatnonzero(strata==i)for i in range(4)];allocation=[min(len(g),limit//4)for g in groups]
    remaining=limit-sum(allocation)
    for i,g in enumerate(groups):
        extra=min(remaining,len(g)-allocation[i]);allocation[i]+=extra;remaining-=extra
    indices=[];weights=[]
    for g,count in zip(groups,allocation):
        if count:indices.extend(rng.choice(g,count,replace=False));weights.extend([len(g)/count]*count)
    order=np.argsort(indices);return np.asarray(indices)[order],np.asarray(weights)[order]
