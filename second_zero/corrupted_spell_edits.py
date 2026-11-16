"""Corrupted-only complete possession edits and exact full-match utility targets."""
import numpy as np
from second_zero import path_rank_edits as prior
from second_zero.gap_spell_model import RawIntervals
from scripts import sprint_decoder_segments_geometry_v2 as graph


def alternatives(row):
    s,e,t,a,b,kind,_,_=map(int,row)
    if kind==2:
        return np.array([(s,a,t),(a,e,1-t)]),np.array([(s,b,t),(b,e,1-t)])
    return prior.alternatives(row)


def candidates(predicted,pool,quality):
    """Bounded insert/merge candidates plus native five-frame boundary shifts."""
    rows=prior.candidates(predicted,pool,quality,limit=24).tolist()
    spells=prior.runs(predicted)
    for i in range(len(spells)-1):
        left,right=spells[i:i+2];s,a,t=map(int,left);_,e,other=map(int,right)
        if a!=right[0] or t==other:continue
        low=max(s+1,a-125);high=min(e-1,a+125)
        points=np.unique(np.r_[np.arange(low,high+1,5),low,high,a-1,a+1])
        rows.extend((s,e,t,a,int(b),2,i,2) for b in points if s<b<e and b!=a)
    return np.asarray(rows,np.int64).reshape(-1,8)


def apply(predicted,rows,selected):
    out=np.asarray(predicted).copy()
    for i in selected:
        _,new=alternatives(rows[i])
        for s,e,t in new:out[s:e]=t
    fixed=np.asarray(predicted)>=2
    if not np.array_equal(out[fixed],np.asarray(predicted)[fixed]):raise ValueError('Status lock violated')
    return out


class ExactContext(prior.ExactDeltaContext):
    def target(self,row,scale=1000.):
        old,new=alternatives(row);s,e=map(int,row[:2])
        if np.any(self.truth[s:e]==3):return np.nan
        removed=[self.old_lookup.get(tuple(v),-1) for v in old.tolist()]
        if any(i<0 for i in removed):return np.nan
        gt,ps,old_tp=self.component(s,e,removed)
        replacement=[self.old[i] for i in ps if i not in removed]+[tuple(v) for v in new.tolist()]
        replacement.sort(key=lambda v:(v[0],v[1],v[2]))
        overlaps=self.pm.spell_overlaps(gt,replacement)
        tp=np.array([self.pm.match_spells(overlaps,len(gt),len(replacement),t)[0] for t in self.pm.SPELL_IOUS],int)
        denominator=float(self.n_gt+self.n_pred+len(new)-len(old))
        new_f1=2*(self.tp+tp-old_tp)/denominator if denominator else np.zeros(len(self.pm.SPELL_IOUS))
        return float(scale*np.mean(new_f1-self.old_f1))


def paired_targets(pm,truths,parents,rows):
    contexts=[ExactContext(pm,y,p) for y,p in zip(truths,parents)]
    counts=(sum(c.n_gt for c in contexts),sum(c.n_pred for c in contexts),sum((c.tp for c in contexts),np.zeros(len(pm.SPELL_IOUS),int)))
    result=[]
    for y,p,rs in zip(truths,parents,rows):
        context=ExactContext(pm,y,p,counts)
        result.append(np.array([context.target(row) for row in rs],np.float32))
    return result


def features(raw,signals,parent,rows,quality_model):
    """Canonical crossfit evidence; complete old/new arcs, raw changed span."""
    if not len(rows):return np.empty((0,1638),np.float32),np.empty(0)
    padded=np.zeros((len(rows),6,3),np.int64);active=np.zeros((len(rows),6),bool)
    for i,row in enumerate(rows):
        old,new=alternatives(row);padded[i,:len(old)]=old;padded[i,3:3+len(new)]=new;active[i,:len(old)]=True;active[i,3:3+len(new)]=True
    unique,inverse=np.unique(padded[active],axis=0,return_inverse=True)
    af=graph.segment_features(signals,unique).astype(np.float32);q=np.clip(quality_model.predict(af),0,1)
    expanded=np.zeros((len(rows),6,175),np.float32);expanded[active]=af[inverse]
    qualities=np.zeros((len(rows),6),np.float32);qualities[active]=q[inverse]
    s,e,t,a,b,kind,_,_=rows.T
    lo=np.minimum(a,b);hi=np.maximum(a,b)
    newteam=np.where(kind==0,1-t,np.where(kind==1,t,np.where(b>a,t,1-t)))
    rf=RawIntervals(raw).features(np.c_[lo,hi,newteam])
    prefix=np.r_[0,np.cumsum(signals[:,0],dtype=np.float64)]
    unary=2*(2*newteam-1)*(prefix[hi]-prefix[lo])
    nold=active[:,:3].sum(1);nnew=active[:,3:].sum(1);delta=nnew-nold
    oldq=qualities[:,:3].sum(1);newq=qualities[:,3:].sum(1)
    proxy=newq-oldq-.4*delta+.01*unary
    extras=np.c_[kind==0,kind==1,kind==2,np.log1p(len(prior.runs(parent)))*np.ones(len(rows)),
                 np.log1p(e-s),np.log1p(hi-lo),(hi-lo)/(e-s),np.clip(unary/50,-20,20),
                 nold,nnew,delta,oldq,newq,newq-oldq,proxy,np.sign(b-a)]
    out=np.c_[expanded.reshape(len(rows),-1),rf,extras].astype(np.float32)
    if out.shape[1]!=1638 or not np.isfinite(out).all():raise ValueError('Invalid edit features')
    return out,proxy
