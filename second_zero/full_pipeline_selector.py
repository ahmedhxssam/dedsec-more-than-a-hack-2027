"""Complete-path edit candidates, matching-count targets and full-parent features."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from second_zero import corrupted_spell_edits as local
from second_zero import gap_spell as gs
from second_zero.gap_spell_model import RawIntervals
from scripts import sprint_decoder_segments_geometry_v2 as graph

MODEL=dict(max_iter=180,max_leaf_nodes=7,max_depth=3,min_samples_leaf=80,learning_rate=.04,l2_regularization=20.,random_state=20261030,early_stopping=False)
PENALTY=.4


@dataclass
class Edit:
    start:int
    stop:int
    kind:int
    source:int
    old:np.ndarray
    new:np.ndarray


def path_arcs(p,start,stop):
    a=gs.spells(np.asarray(p)[start:stop]).copy();a[:,:2]+=start;return a


def candidates(parent,alternates,pool,quality):
    p=np.asarray(parent);items={}
    def add(edit):
        if np.array_equal(edit.old,edit.new):return
        key=(edit.start,edit.stop,edit.new.tobytes())
        if key in items:items[key].source|=edit.source
        else:items[key]=edit
    for row in local.candidates(p,pool,quality):
        old,new=local.alternatives(row);add(Edit(int(row[0]),int(row[1]),int(row[5]),0,old,new))
    edges=np.r_[0,np.flatnonzero((p[1:]<2)!=(p[:-1]<2))+1,len(p)]
    for source,q in enumerate(alternates,1):
        gs.check_lock(q,p,p<2)
        for lo,hi in zip(edges[:-1],edges[1:]):
            if p[lo]>=2:continue
            at=np.arange(lo+1,hi);aligned=at[(p[at]!=p[at-1])&(q[at]!=q[at-1])&(p[at]==q[at])&(p[at-1]==q[at-1])];anchors=np.r_[lo,aligned,hi]
            for s,e in zip(anchors[:-1],anchors[1:]):
                if np.array_equal(p[s:e],q[s:e]):continue
                add(Edit(int(s),int(e),3,1<<(source-1),path_arcs(p,s,e),path_arcs(q,s,e)))
    return list(items.values())


def apply(parent,edits,selected):
    out=np.asarray(parent).copy()
    for i in selected:
        for s,e,t in edits[int(i)].new:out[s:e]=t
    gs.check_lock(out,parent,np.asarray(parent)<2)
    return out


def choose(edits,utility):
    rows=np.array([[e.start,e.stop] for e in edits],np.int64).reshape(-1,2)
    return local.prior.select_edits(rows,utility)


class TargetContext(local.prior.ExactDeltaContext):
    def target(self,edit):
        if np.any(self.truth[edit.start:edit.stop]==3):return np.nan,np.nan
        removed=[self.old_lookup.get(tuple(v),-1) for v in edit.old.tolist()]
        if any(i<0 for i in removed):raise ValueError('Candidate does not remove complete actual parent spells')
        gt,ps,old_tp=self.component(edit.start,edit.stop,removed)
        replacement=[self.old[i] for i in ps if i not in removed]+[tuple(v) for v in edit.new.tolist()];replacement.sort(key=lambda v:(v[0],v[1],v[2]));overlaps=self.pm.spell_overlaps(gt,replacement)
        tp=np.array([self.pm.match_spells(overlaps,len(gt),len(replacement),t)[0] for t in self.pm.SPELL_IOUS]);delta=tp-old_tp;dn=len(edit.new)-len(edit.old);denominator=self.n_gt+self.n_pred+dn
        after=2*(self.tp+delta)/denominator if denominator else np.zeros(len(self.pm.SPELL_IOUS))
        return float(np.mean(delta)),float(1000*np.mean(after-self.old_f1))


def paired_targets(pm,ys,parents,edits):
    contexts=[TargetContext(pm,y,p) for y,p in zip(ys,parents)];counts=(sum(c.n_gt for c in contexts),sum(c.n_pred for c in contexts),sum((c.tp for c in contexts),np.zeros(len(pm.SPELL_IOUS),int)));result=[]
    for y,p,ee in zip(ys,parents,edits):
        c=TargetContext(pm,y,p,counts);result.append(np.array([c.target(v) for v in ee],np.float32).reshape(-1,2))
    return result


def feature_matrix(raw,baseline_signals,rich_signals,parent,edits,qmodel):
    if not edits:return np.empty((0,1298),np.float32),np.empty(0),np.empty(0)
    # Reuse the exact complete model's final evidence in both fitting and
    # prediction. No simplified owner or legacy-only status parent is built.
    all_arcs=np.unique(np.concatenate([a for edit in edits for a in [edit.old,edit.new]]),axis=0);lookup={tuple(v):i for i,v in enumerate(all_arcs.tolist())}
    bf=graph.segment_features(baseline_signals,all_arcs).astype(np.float32);nf=graph.segment_features(rich_signals,all_arcs).astype(np.float32);bq=np.clip(qmodel.predict(bf),0,1);nq=np.clip(qmodel.predict(nf),0,1)
    changed=[];rows=[];proxy=[];delta=[];p0=expit(baseline_signals[:,0]);p1=expit(rich_signals[:,0]);missing=~np.isfinite(raw['ball']).all(1)
    for edit in edits:
        oi=np.array([lookup[tuple(v)] for v in edit.old.tolist()]);ni=np.array([lookup[tuple(v)] for v in edit.new.tolist()]);s,e=edit.start,edit.stop;old=np.asarray(parent[s:e]);new=old.copy()
        for a,b,t in edit.new:new[a-s:b-s]=t
        ix=np.flatnonzero(old!=new)
        if not len(ix):raise ValueError('Empty edit')
        a,b=s+int(ix[0]),s+int(ix[-1])+1;team=int(parent[s]);changed.append([a,b,team]);dn=len(ni)-len(oi);delta.append(dn)
        probability=[np.where(v==1,p[s:e],1-p[s:e]) for p in [p0,p1] for v in [old,new]]
        stats=[]
        for v in probability:stats.extend([float(v.mean()),float(v.min()),float(np.mean(np.log(np.clip(v,1e-7,1))))])
        unary=float(np.sum((2*new-1)*baseline_signals[s:e,0]-(2*old-1)*baseline_signals[s:e,0]));quality_delta=float(bq[ni].sum()-bq[oi].sum());proxy.append(quality_delta-PENALTY*dn+.01*unary)
        extra=[edit.kind,edit.source,np.log1p(e-s),np.log1p(b-a),len(oi),len(ni),dn,len(ix)/(e-s),float(missing[a:b].mean()),float(np.mean(np.abs(p0[a:b]-p1[a:b]))),np.log1p(len(gs.spells(parent))),quality_delta,float(nq[ni].sum()-nq[oi].sum()),np.clip(unary/50,-20,20),*stats]
        rows.append(np.r_[bf[oi].sum(0),bf[ni].sum(0),nf[oi].sum(0),nf[ni].sum(0),extra])
    raw_features=RawIntervals(raw).features(np.array(changed));x=np.c_[np.asarray(rows,np.float32),raw_features].astype(np.float32)
    if x.shape[1]!=1298 or not np.isfinite(x).all():raise ValueError('Invalid complete-pipeline selector features: '+str(x.shape))
    return x,np.asarray(proxy,np.float32),np.asarray(delta,np.int16)
