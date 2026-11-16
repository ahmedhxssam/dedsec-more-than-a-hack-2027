"""Gap-local segment utilities; raw observations and fixed parent labels only."""
from __future__ import annotations
import numpy as np
from scripts import sprint_decoder_segments_geometry_v2 as graph


def spells(predicted):
    p=np.asarray(predicted);edges=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1,len(p)]
    return np.asarray([(s,e,int(p[s])) for s,e in zip(edges[:-1],edges[1:]) if p[s]<2],np.int64).reshape(-1,3)


def feasible(arcs,parent,eligible):
    """A complete spell can exist without changing any locked frame."""
    arcs=np.asarray(arcs,np.int64).reshape(-1,3);p=np.asarray(parent);mask=np.asarray(eligible,bool);n=len(p)
    if mask.shape!=p.shape or np.any(mask&(p>=2)):raise ValueError('Only parent live frames may be mutable')
    result=np.zeros(len(arcs),bool)
    for team in (0,1):
        ix=np.flatnonzero(arcs[:,2]==team)
        if not len(ix):continue
        a,b=arcs[ix,:2].T
        if np.any((a<0)|(b>n)|(a>=b)):raise ValueError('Invalid segment bounds')
        blocked=np.r_[0,np.cumsum((~mask)&(p!=team))]
        left=(a==0)|mask[np.maximum(a-1,0)]|(p[np.maximum(a-1,0)]!=team)
        right=(b==n)|mask[np.minimum(b,n-1)]|(p[np.minimum(b,n-1)]!=team)
        result[ix]=(blocked[b]-blocked[a]==0)&left&right
    return result


def pool(parent,eligible,arcs):
    p=np.asarray(parent);a=np.unique(np.r_[np.asarray(arcs,np.int64).reshape(-1,3),spells(p)],axis=0)
    a=a[feasible(a,p,eligible)]
    return a


def check_lock(predicted,parent,eligible):
    p=np.asarray(predicted);b=np.asarray(parent);m=np.asarray(eligible,bool)
    if p.shape!=b.shape or not np.isin(p,[0,1,2,3]).all():raise ValueError('Invalid prediction')
    if not np.array_equal(p[~m],b[~m]) or not np.array_equal(p>=2,b>=2):raise ValueError('Gap lock violated')


def decode(parent,eligible,arcs,quality,penalty,unary=None,frame_weight=0.):
    z=np.zeros(len(parent),np.float32) if unary is None else np.asarray(unary)
    predicted=graph.graph_decode(z,np.asarray(parent),arcs,np.asarray(quality),penalty,frame_weight)
    check_lock(predicted,parent,eligible)
    return predicted


def credit(reference,arcs,pm):
    """Independent best-reference credit: upper-bounds greedy matched credit."""
    arcs=np.asarray(arcs,np.int64).reshape(-1,3);best=np.zeros(len(arcs),np.float64)
    if not len(arcs):return best
    a,b,t=arcs.T
    for s,e,team in pm.possession_spells(reference,pm.valid_runs(reference)):
        ix=np.flatnonzero((t==team)&(a<e)&(b>s))
        if len(ix):best[ix]=np.maximum(best[ix],(np.minimum(b[ix],e)-np.maximum(a[ix],s))/(np.maximum(b[ix],e)-np.minimum(a[ix],s)))
    return np.mean(best[:,None]>=np.asarray(pm.SPELL_IOUS)[None],axis=1)


def fractional_oracle(halves,pm,max_iter=30):
    """Exact independent-credit ratio DP; official scoring supplies a lower bound.

    For a path with P possession spells, mean official SpellF1 is at most
    2*sum(best-IoU credits)/(G+P). Dinkelbach solves that additive graph ratio.
    It is an upper bound because multiple proposed spells can claim one GT.
    Its decoded path is feasible, and its official score is an achievable
    reference-assisted lower bound, never a learned model score.
    """
    nref=sum(len(pm.possession_spells(h['truth'],pm.valid_runs(h['truth']))) for h in halves)
    work=[]
    for h in halves:
        a=pool(h['predicted'],h['eligible_mask'],h['segments']);q=credit(h['truth'],a,pm)
        work.append((h,a,q,{tuple(v):float(c) for v,c in zip(a.tolist(),q)}))
    ratio=0.;history=[];paths=[]
    for iteration in range(max_iter):
        numerator=0.;count=0;paths=[]
        for h,a,q,lookup in work:
            pred=decode(h['predicted'],h['eligible_mask'],a,q,ratio)
            chosen=spells(pred);count+=len(chosen);numerator+=sum(lookup[tuple(row)] for row in chosen.tolist());paths.append(pred)
        new=numerator/(nref+count);residual=numerator-ratio*(nref+count)
        history.append({'iteration':iteration,'ratio':float(ratio),'next_ratio':float(new),'residual':float(residual),'predicted_spells':count})
        if abs(residual)<1e-8:break
        ratio=new
    else:raise RuntimeError('Fractional oracle failed to converge')
    join=lambda values:np.r_[values[0],np.int8(3),values[1]]
    metric=pm.evaluate(join([h['truth'] for h in halves]),join(paths),bootstrap_n=0)
    upper=min(1.,2*new)
    if metric['spell_f1_mean']>upper+1e-8:raise AssertionError('Invalid oracle bound')
    return {'official_feasible_spell_f1':float(metric['spell_f1_mean']),'independent_credit_upper_bound':float(upper),'five_iou':metric['spell_f1'],'n_gt_spells':nref,'n_pred_spells':metric['n_pred_spells'],'history':history,'pool_sizes':[len(a) for _,a,_,_ in work]},paths
