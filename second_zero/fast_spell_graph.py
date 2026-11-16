"""Exact original alternating interval DAG with a local C execution kernel."""
import ctypes
import subprocess
from pathlib import Path
import numpy as np

SOURCE = r'''
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
int graph(long n,long m,const int8_t*base,const int64_t*arcs,
          const double*gain,int8_t*out) {
 double*dp=malloc((n+1)*2*sizeof(double));
 int64_t*back=malloc((n+1)*2*sizeof(int64_t));
 if(!dp || !back){free(dp);free(back);return 1;}
 for(long i=0;i<=n;i++){dp[2*i]=dp[2*i+1]=-INFINITY;back[2*i]=back[2*i+1]=-1;}
 for(long i=0;i<n;i++){out[i]=base[i];if(base[i]<2 && (i==0 || base[i-1]>=2)){dp[2*i]=dp[2*i+1]=0;}}
 for(long j=0;j<m;j++){
  long s=arcs[3*j],e=arcs[3*j+1],t=arcs[3*j+2];
  double v=dp[2*s+1-t]+gain[j];
  if(v>dp[2*e+t]){dp[2*e+t]=v;back[2*e+t]=j;}
 }
 for(long hi=1;hi<=n;hi++)if(base[hi-1]<2 && (hi==n || base[hi]>=2)){
  long lo=hi-1;while(lo>0 && base[lo-1]<2)lo--;
  int t=dp[2*hi+1]>dp[2*hi];long end=hi;
  if(!isfinite(dp[2*hi+t])){free(dp);free(back);return 2;}
  while(end>lo){long j=back[2*end+t];if(j<0){free(dp);free(back);return 3;}
   long s=arcs[3*j];for(long k=s;k<end;k++)out[k]=t;end=s;t=1-t;
  }
 }
 free(dp);free(back);return 0;
}
'''
_LIB = None
BUILD = Path(__file__).resolve().parents[1]/'runs/metric-decoder-v11/kernel'


def library():
    global _LIB
    if _LIB is None:
        BUILD.mkdir(parents=True,exist_ok=True)
        src,lib=BUILD/'graph.c',BUILD/'graph.dylib'
        if src.exists() and src.read_text()!=SOURCE:raise ValueError('Frozen graph source changed')
        if not src.exists():src.write_text(SOURCE)
        if not lib.exists():subprocess.run(['clang','-O3','-dynamiclib',str(src),'-o',str(lib)],check=True)
        _LIB=ctypes.CDLL(str(lib));_LIB.graph.argtypes=[ctypes.c_long,ctypes.c_long,*([ctypes.c_void_p]*4)];_LIB.graph.restype=ctypes.c_int
    return _LIB


class Graph:
    def __init__(self,unary,parent,arcs):
        p=np.asarray(parent);a=np.asarray(arcs,np.int64).reshape(-1,3)
        if p.ndim!=1 or not np.isin(p,[0,1,2,3]).all() or np.asarray(unary).shape!=p.shape:raise ValueError('Invalid timeline')
        if len(a) and np.any((a[:,0]<0)|(a[:,1]>len(p))|(a[:,0]>=a[:,1])|(a[:,2]<0)|(a[:,2]>1)):raise ValueError('Invalid arc')
        blocked=np.r_[0,np.cumsum(p>=2)]
        if np.any(blocked[a[:,1]]!=blocked[a[:,0]]):raise ValueError('Arc crosses fixed status')
        self.order=np.lexsort((a[:,0],a[:,1]));self.arcs=np.ascontiguousarray(a[self.order]);self.parent=np.ascontiguousarray(p,dtype=np.int8)
        # Preserve the original prefix/difference dtype and operation order.
        prefix=np.r_[0,np.cumsum(unary)]
        self.unary=(2*a[:,2]-1)*(prefix[a[:,1]]-prefix[a[:,0]])

    def decode(self,quality,penalty,frame_weight):
        q=np.asarray(quality)
        if q.shape!=(len(self.arcs),) or not np.isfinite(q).all() or not np.isfinite(self.unary).all():raise ValueError('Invalid graph scores')
        gain=np.ascontiguousarray((q-penalty+frame_weight*self.unary)[self.order],dtype=np.float64);out=np.empty_like(self.parent)
        result=library().graph(len(out),len(gain),self.parent.ctypes.data,self.arcs.ctypes.data,gain.ctypes.data,out.ctypes.data)
        if result:raise ValueError('No full graph cover or allocation failed: '+str(result))
        return out
