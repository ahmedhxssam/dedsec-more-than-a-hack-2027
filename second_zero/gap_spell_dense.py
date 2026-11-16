"""Observation-only dense gap proposals with exact outside-frame preservation."""
import numpy as np
from second_zero import gap_spell as gs


def proposals(parent,eligible,existing,stride=5,maximum_duration=250):
    p=np.asarray(parent);m=np.asarray(eligible,bool);n=len(p);base=gs.spells(p)
    edges=np.r_[0,np.flatnonzero(m[1:]!=m[:-1])+1,n];extra=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        if not m[lo]:continue
        inside=base[(base[:,0]<hi)&(base[:,1]>lo)]
        points=np.unique(np.r_[np.arange(lo,hi+1,stride),hi,np.clip(inside[:,:2].ravel(),lo,hi)])
        a,b=np.meshgrid(points,points,indexing='ij');take=(b>a)&(b-a<=maximum_duration)
        a=a[take];b=b[take]
        extra.extend([np.c_[a,b,np.full(len(a),t)] for t in (0,1)])
        for s,e,t in inside:
            q=points[(points>s)&(points<e)]
            if len(q):extra.extend([np.c_[np.full(len(q),s),q,np.full(len(q),t)],np.c_[q,np.full(len(q),e),np.full(len(q),t)]])
    combined=np.r_[np.asarray(existing,np.int64).reshape(-1,3),*extra] if extra else existing
    return gs.pool(p,m,combined)
