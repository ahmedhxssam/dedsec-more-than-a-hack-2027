"""Complete local possession alternatives and exact one-to-one metric targets.

Inference candidates use predicted intervals only. Reference components exist
only while making fitting-side targets or explicit diagnostics.
"""
from __future__ import annotations
import numpy as np


def runs(predicted):
    p=np.asarray(predicted);edges=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1,len(p)]
    return np.array([(s,e,int(p[s]))for s,e in zip(edges[:-1],edges[1:])if p[s]<2],np.int64).reshape(-1,3)


def candidates(predicted,pool,quality,limit=24):
    """At most six evidence-ranked internal opposite arcs per duration bin.

    Return N×8 rows: edit start/stop, outerteam, split start/stop, kind,
    first baseline spell index, count of replaced baseline spells.
    kind0 is H→H-A-H, kind1 is H-A-H→H. There is no duration floor.
    """
    spells=runs(predicted);rows=[];p=np.asarray(pool);q=np.asarray(quality)
    if not len(spells):return np.empty((0,8),np.int64)
    owner=np.searchsorted(spells[:,0],p[:,0],side='right')-1
    valid=owner>=0;ids=np.flatnonzero(valid);o=owner[valid];s,e,t=spells[o].T
    take=(p[ids,2]!=t)&(p[ids,0]>s)&(p[ids,1]<e)&(p[ids,1]-p[ids,0]<=250)
    ids=ids[take];o=o[take]
    # Predicted D/X holes cannot have generated arcs; still verify containment.
    for i in np.unique(o):
        choices=ids[o==i];duration=p[choices,1]-p[choices,0];selected=[]
        for lo,hi in ((1,25),(26,75),(76,125),(126,250)):
            group=choices[(duration>=lo)&(duration<=hi)]
            group=group[np.argsort(-q[group],kind='stable')]
            count=0
            for j in group:
                a,b,_=p[j]
                # Do not exhaust the quota on near-identical shifted copies.
                if any(abs(int(a)-aa)<3 and abs(int(b)-bb)<3 for aa,bb in selected):continue
                selected.append((int(a),int(b)));count+=1
                if count>=limit//4:break
        s,e,t=spells[i]
        rows.extend((s,e,t,a,b,0,i,1)for a,b in selected)
    for i in range(len(spells)-2):
        a,b,c=spells[i:i+3]
        if a[1]==b[0]and b[1]==c[0]and a[2]==c[2]and a[2]!=b[2]and b[1]-b[0]<=250:
            rows.append((a[0],c[1],a[2],b[0],b[1],1,i,3))
    return np.asarray(rows,np.int64).reshape(-1,8)


def alternatives(row):
    s,e,t,a,b,kind,_,_=map(int,row)
    if kind==0:
        return np.array([(s,e,t)],np.int64),np.array([(s,a,t),(a,b,1-t),(b,e,t)],np.int64)
    return np.array([(s,a,t),(a,b,1-t),(b,e,t)],np.int64),np.array([(s,e,t)],np.int64)


def select_edits(rows,utility):
    """Maximum-utility nonoverlapping edit windows; keep baseline at zero ties."""
    rows=np.asarray(rows);u=np.asarray(utility,dtype=float)
    if not len(rows):return np.empty(0,np.int64)
    order=np.lexsort((np.arange(len(rows)),rows[:,0],rows[:,1]));r=rows[order];score=u[order]
    previous=np.searchsorted(r[:,1],r[:,0],side='right')-1
    # Zero-length edits are never generated, so prior cannot include current.
    dp=np.zeros(len(rows)+1);take=np.zeros(len(rows),bool)
    for i in range(len(rows)):
        value=dp[previous[i]+1]+score[i]
        if np.isfinite(value)and value>dp[i]:dp[i+1]=value;take[i]=True
        else:dp[i+1]=dp[i]
    selected=[];i=len(rows)-1
    while i>=0:
        if take[i]and dp[i+1]>dp[i]:selected.append(int(order[i]));i=int(previous[i])
        else:i-=1
    return np.array(selected[::-1],np.int64)


def apply(predicted,rows,selected):
    out=np.asarray(predicted).copy()
    for i in selected:
        _,new=alternatives(rows[i])
        for s,e,t in new:out[s:e]=t
    fixed=np.asarray(predicted)>=2
    np.testing.assert_array_equal(out[fixed],np.asarray(predicted)[fixed])
    return out


class ExactDeltaContext:
    """Exact global greedy TP deltas using the entire affected overlap component."""
    def __init__(self,pm,truth,predicted,global_counts=None):
        self.pm=pm;self.truth=np.asarray(truth);self.predicted=np.asarray(predicted)
        if np.any((self.truth!=3)&(self.predicted==3)):
            raise ValueError('Edit target requires no abstentions inside graded runs')
        valid=pm.valid_runs(self.truth)
        self.gt=pm.possession_spells(self.truth,valid);self.old=pm.possession_spells(self.predicted,valid)
        overlap=pm.spell_overlaps(self.gt,self.old)
        self.g_to_p=[set()for _ in self.gt];self.p_to_g=[set()for _ in self.old]
        for _,g,p in overlap:self.g_to_p[g].add(p);self.p_to_g[p].add(g)
        self.tp=np.array([pm.match_spells(overlap,len(self.gt),len(self.old),t)[0]for t in pm.SPELL_IOUS],int)
        self.n_gt=len(self.gt);self.n_pred=len(self.old)
        if global_counts is not None:self.n_gt,self.n_pred,self.tp=global_counts
        self.old_lookup={tuple(v):i for i,v in enumerate(self.old)};self.cache={}
        denominator=float(self.n_gt+self.n_pred)
        self.old_f1=2*self.tp/denominator if denominator else np.zeros(len(pm.SPELL_IOUS))

    def component(self,s,e,removed):
        key=(int(s),int(e),tuple(removed))
        if key in self.cache:return self.cache[key]
        gs={i for i,(a,b,_)in enumerate(self.gt)if a<e and b>s};ps=set(removed)
        todo_g=list(gs);todo_p=list(ps)
        while todo_g or todo_p:
            while todo_g:
                g=todo_g.pop()
                for p in self.g_to_p[g]-ps:ps.add(p);todo_p.append(p)
            while todo_p:
                p=todo_p.pop()
                for g in self.p_to_g[p]-gs:gs.add(g);todo_g.append(g)
        gs=sorted(gs);ps=sorted(ps);gt=[self.gt[i]for i in gs];pred=[self.old[i]for i in ps]
        overlaps=self.pm.spell_overlaps(gt,pred)
        tp=np.array([self.pm.match_spells(overlaps,len(gt),len(pred),t)[0]for t in self.pm.SPELL_IOUS],int)
        result=(gt,ps,tp);self.cache[key]=result;return result

    def target(self,row,scale=1000.):
        old,new=alternatives(row);s,e=map(int,row[:2])
        # Reference unknowns never become synthetic labels; skip target only.
        if np.any(self.truth[s:e]==3):return np.nan
        removed=[self.old_lookup.get(tuple(v),-1)for v in old.tolist()]
        if any(i<0 for i in removed):return np.nan
        gt,ps,old_tp=self.component(s,e,removed)
        replacement=[self.old[i]for i in ps if i not in removed]+[tuple(v)for v in new.tolist()]
        replacement.sort(key=lambda v:(v[0],v[1],v[2]))
        overlaps=self.pm.spell_overlaps(gt,replacement)
        tp=np.array([self.pm.match_spells(overlaps,len(gt),len(replacement),t)[0]for t in self.pm.SPELL_IOUS],int)
        delta_n=len(new)-len(old)
        denominator=float(self.n_gt+self.n_pred+delta_n)
        new_f1=2*(self.tp+tp-old_tp)/denominator if denominator else np.zeros(len(self.pm.SPELL_IOUS))
        return float(scale*np.mean(new_f1-self.old_f1))
