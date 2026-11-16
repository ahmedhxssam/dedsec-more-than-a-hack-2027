"""Global ownership decoding experiments; no label access in this module.

Inputs are frozen ownership logits, live/dead masks and optional destination
contact scores. C implements the exact DP and an optional duration constraint.
"""
from __future__ import annotations
import ctypes
import subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.special import expit, logit as inverse_sigmoid

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'runs/two-hour-sprint/decoder'
SOURCE=r'''
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
void decode(long n, const double*x, const double*c, int duration, int8_t*out) {
 if(n<=0)return;
 if(duration<=1){
  int8_t*back=calloc(n*2,1); double a=0.,b=x[0];
  for(long i=1;i<n;i++){
   double aa=a,bb=b;back[2*i+1]=1;
   if(b-c[2*i]>a){aa=b-c[2*i];back[2*i]=1;}
   if(a-c[2*i+1]>b){bb=a-c[2*i+1];back[2*i+1]=0;}
   a=aa;b=bb+x[i];double m=fmax(a,b);a-=m;b-=m;
  }
  out[n-1]=b>a;for(long i=n-1;i>0;i--)out[i-1]=back[2*i+out[i]];free(back);return;
 }
 if(n<duration){double s=0;for(long i=0;i<n;i++)s+=x[i];for(long i=0;i<n;i++)out[i]=s>0;return;}
 double *p=calloc(n+1,sizeof(double)),*dp=malloc((n+1)*2*sizeof(double));
 long *back=calloc((n+1)*2,sizeof(long));
 for(long i=0;i<=n;i++){dp[2*i]=dp[2*i+1]=-INFINITY;if(i>0)p[i]=p[i-1]+x[i-1];}
 double b0=0,b1=0;long s0=0,s1=0;
 for(long i=duration;i<=n;i++){
  long j=i-duration;
  if(j>=duration){
   double v0=dp[2*j+1]-c[2*j],v1=dp[2*j]-p[j]-c[2*j+1];
   if(v0>b0){b0=v0;s0=j;}if(v1>b1){b1=v1;s1=j;}
  }
  dp[2*i]=b0;dp[2*i+1]=b1+p[i];back[2*i]=s0;back[2*i+1]=s1;
 }
 int label=dp[2*n+1]>dp[2*n];long end=n;
 while(end>0){long start=back[2*end+label];for(long i=start;i<end;i++)out[i]=label;label=1-label;end=start;}
 free(p);free(dp);free(back);
}
'''
_LIB=None

def library():
 global _LIB
 if _LIB is None:
  BUILD.mkdir(parents=True,exist_ok=True)
  src=BUILD/'decoder.c';lib=BUILD/'decoder.dylib'
  if not src.exists() or src.read_text()!=SOURCE or not lib.exists():
   src.write_text(SOURCE)
   subprocess.run(['clang','-O3','-dynamiclib',str(src),'-o',str(lib)],check=True)
  _LIB=ctypes.CDLL(str(lib));_LIB.decode.argtypes=[ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
 return _LIB

def segment_decode(z,costs,duration=1):
 z=np.ascontiguousarray(z,dtype=np.float64);costs=np.ascontiguousarray(costs,dtype=np.float64)
 if costs.shape!=(len(z),2) or not np.isfinite(z).all() or not np.isfinite(costs).all() or (costs<0).any():raise ValueError('Invalid inputs')
 out=np.empty(len(z),np.int8);library().decode(len(z),z.ctypes.data,costs.ctypes.data,duration,out.ctypes.data);return out

def transformed_logits(z,kind='none',width=1,cap=0):
 z=np.asarray(z,dtype=np.float64)
 if cap:z=np.clip(z,-cap,cap)
 if width>1:
  if kind=='logit':z=uniform_filter1d(z,width,mode='nearest')
  elif kind=='probability':z=inverse_sigmoid(np.clip(uniform_filter1d(expit(z),width,mode='nearest'),1e-12,1-1e-12))
  elif kind=='median':z=median_filter(z,size=width,mode='nearest')
  else:raise ValueError(kind)
 return z

def decode(z,baseline,cost=6.,kind='none',width=1,cap=0,duration=1,touch=None,touch_weight=1.):
 z=np.asarray(z);base=np.asarray(baseline);out=base.copy();live=base<2
 if z.shape!=base.shape or not np.isin(base,[0,1,2,3]).all():raise ValueError('Invalid fixed mask')
 boundaries=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
 for lo,hi in zip(boundaries[:-1],boundaries[1:]):
  if not live[lo]:continue
  zz=transformed_logits(z[lo:hi],kind,width,cap)
  costs=np.full((hi-lo,2),cost,dtype=np.float64)
  if touch is not None:costs/=1+touch_weight*np.clip(touch[lo:hi],0,1)
  out[lo:hi]=segment_decode(zz,costs,duration)
 return out

def sided_mean(x,width):
 """Past excludes current frame; future includes it; edges truncate context."""
 x=np.asarray(x,dtype=np.float64);n=len(x);p=np.r_[0,np.cumsum(x)]
 ix=np.arange(n);lo=np.maximum(0,ix-width);hi=np.minimum(n,ix+width)
 before=(p[ix]-p[lo])/np.maximum(1,ix-lo)
 after=(p[hi]-p[ix])/np.maximum(1,hi-ix)
 if n:before[0]=x[0]
 return before,after

def reliability_decode(z,baseline,reliability,*,cost=6.,mode='unary',factor=.5,width=25,cap=0,duration=1,touch=None):
 """Adapt confidence/persistence to input availability, without reference labels."""
 z=np.asarray(z);base=np.asarray(baseline);seen=np.asarray(reliability,dtype=np.float64)
 if z.shape!=base.shape or seen.shape!=base.shape:raise ValueError('Unaligned inputs')
 out=base.copy();live=base<2;bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
 for lo,hi in zip(bounds[:-1],bounds[1:]):
  if not live[lo]:continue
  zz=z[lo:hi].astype(np.float64).copy();ss=seen[lo:hi];cc=np.full((hi-lo,2),cost,dtype=np.float64)
  if cap:zz=np.clip(zz,-cap,cap)
  if touch is not None:cc/=1+np.clip(touch[lo:hi],0,1)
  if mode=='unary':zz*=factor+(1-factor)*ss
  elif mode=='unary_window':
   rr=uniform_filter1d(ss,width,mode='nearest');zz*=factor+(1-factor)*rr
  elif mode=='cost':
   rr=uniform_filter1d(ss,width,mode='nearest');cc*=1+factor*(1-rr[:,None])
  elif mode in ('interpolate','hold','bridge'):
   available=np.flatnonzero(ss>.5)
   if len(available):
    ix=np.arange(hi-lo);left=np.maximum.accumulate(np.where(ss>.5,ix,-1));right=np.minimum.accumulate(np.where(ss>.5,ix,len(ix))[::-1])[::-1]
    valid=(left>=0)&(right<len(ix))&(ss<.5)
    left=np.maximum(left,0);right=np.minimum(right,len(ix)-1)
    if mode=='interpolate':addition=np.interp(ix,available,zz[available])
    elif mode=='hold':addition=zz[left]
    else:
     addition=.5*(zz[left]+zz[right]);valid &= zz[left]*zz[right]>0
    valid &= right-left<=width
    zz[valid]=(1-factor)*zz[valid]+factor*addition[valid]
  elif mode in ('direction','direction_seen','destination'):
   old,new=sided_mean(zz,width)
   if mode=='destination':support=np.c_[np.maximum(-new,0),np.maximum(new,0)];support=np.tanh(support/2)
   else:
    support=np.c_[np.maximum(old,0)*np.maximum(-new,0),np.maximum(-old,0)*np.maximum(new,0)]
    support=np.tanh(np.sqrt(support)/2)
   if mode=='direction_seen':support*=uniform_filter1d(ss,width,mode='nearest')[:,None]
   cc*=1-factor*support
  else:raise ValueError(mode)
  out[lo:hi]=segment_decode(zz,cc,duration)
 return out

SOURCE += r'''
static inline double lae(double a,double b){double m=fmax(a,b);return m+log(exp(a-m)+exp(b-m));}
void posterior(long n,const double*x,const double*c,double*out){
 if(n<=0)return;
 double*al=malloc(n*2*sizeof(double));al[0]=0;al[1]=x[0];
 for(long i=1;i<n;i++){
  double a=lae(al[2*i-2],al[2*i-1]-c[2*i]),b=lae(al[2*i-1],al[2*i-2]-c[2*i+1])+x[i];
  double m=fmax(a,b);al[2*i]=a-m;al[2*i+1]=b-m;
 }
 double a=0,b=0;out[n-1]=al[2*n-1]-al[2*n-2];
 for(long i=n-2;i>=0;i--){
  double aa=lae(a,b+x[i+1]-c[2*i+3]),bb=lae(a-c[2*i+2],b+x[i+1]);
  double m=fmax(aa,bb);a=aa-m;b=bb-m;out[i]=al[2*i+1]+b-al[2*i]-a;
 }
 free(al);
}
'''

def marginal_logits(z,costs):
 z=np.ascontiguousarray(z,dtype=np.float64);costs=np.ascontiguousarray(costs,dtype=np.float64)
 if costs.shape!=(len(z),2):raise ValueError('Mismatched posterior inputs')
 out=np.empty(len(z));lib=library();lib.posterior.argtypes=[ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
 lib.posterior(len(z),z.ctypes.data,costs.ctypes.data,out.ctypes.data);return out

def posterior_decode(z,baseline,*,cost=6.,temperature=1.,postcost=0.,reliability=None,missing_weight=1.,touch=None):
 z=np.asarray(z);base=np.asarray(baseline);out=base.copy();live=base<2
 bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
 for lo,hi in zip(bounds[:-1],bounds[1:]):
  if not live[lo]:continue
  zz=z[lo:hi].astype(np.float64)/temperature;cc=np.full((hi-lo,2),cost,dtype=np.float64)
  if touch is not None:cc/=1+np.clip(touch[lo:hi],0,1)
  if reliability is not None:zz*=missing_weight+(1-missing_weight)*reliability[lo:hi]
  marg=marginal_logits(zz,cc)
  out[lo:hi]=segment_decode(marg,np.full((hi-lo,2),postcost)) if postcost else marg>0
 return out

def adaptive_decode(z,baseline,reliability,*,cost=6.,width=31,discount=.9,missing=0.,reliability_width=63,scale=2.,cap=0.,unary_missing=1.,duration=1,touch=None,tie_break=0.):
 """Combine destination confidence with missingness-aware persistence costs."""
 z=np.asarray(z);base=np.asarray(baseline);seen=np.asarray(reliability,dtype=float);out=base.copy();live=base<2
 bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
 for lo,hi in zip(bounds[:-1],bounds[1:]):
  if not live[lo]:continue
  zz=z[lo:hi].astype(float).copy();ss=seen[lo:hi]
  _,new=sided_mean(zz,width);support=np.tanh(np.c_[np.maximum(-new,0),np.maximum(new,0)]/scale)
  cc=cost*(1-discount*support)
  cc*=1+missing*(1-uniform_filter1d(ss,reliability_width,mode='nearest')[:,None])
  if touch is not None:cc/=1+np.clip(touch[lo:hi],0,1)
  if cap:zz=np.clip(zz,-cap,cap)+tie_break*zz
  zz*=unary_missing+(1-unary_missing)*ss
  out[lo:hi]=segment_decode(zz,cc,duration)
 return out

def apply_config(baseline,logit,seen,config,touch=None):
 """Stable adapter for sprint export, preserving every fixed nonlive frame.

config requires family in plain/reliability/adaptive/posterior; remaining keys
are that family's global parameters. Touch is optional existing n-by2 support.
"""
 params=dict(config);family=params.pop('family')
 if family=='plain':result=decode(logit,baseline,touch=touch,**params)
 elif family=='reliability':result=reliability_decode(logit,baseline,seen,touch=touch,**params)
 elif family=='adaptive':result=adaptive_decode(logit,baseline,seen,touch=touch,**params)
 elif family=='posterior':result=posterior_decode(logit,baseline,reliability=seen,touch=touch,**params)
 else:raise ValueError('Unknown decoder family: '+family)
 nonlive=np.asarray(baseline)>=2
 assert np.array_equal(result[nonlive],np.asarray(baseline)[nonlive])
 return result

def anchor_decode(z,baseline,reliability,*,cost=6.,width=251,confidence=2.,strength=.5,mode='same_missing',touch=None):
 """Use agreement of reliable control anchors to regularize uncertain gaps."""
 z=np.asarray(z);base=np.asarray(baseline);seen=np.asarray(reliability,dtype=float);out=base.copy();live=base<2
 bounds=np.r_[0,np.flatnonzero(live[1:]!=live[:-1])+1,len(base)]
 for lo,hi in zip(bounds[:-1],bounds[1:]):
  if not live[lo]:continue
  zz=z[lo:hi].astype(float).copy();ss=seen[lo:hi];n=hi-lo;ix=np.arange(n)
  anchor=(ss>.5)&(np.abs(zz)>=confidence)
  left=np.maximum.accumulate(np.where(anchor,ix,-1));right=np.minimum.accumulate(np.where(anchor,ix,n)[::-1])[::-1]
  valid=(left>=0)&(right<n)&(right-left<=width)&~anchor
  left=np.maximum(left,0);right=np.minimum(right,n-1)
  if mode.startswith('same'):valid &= zz[left]*zz[right]>0
  if mode.endswith('missing'):valid &= ss<.5
  if mode.startswith('hold'):prior=zz[left]
  elif mode.startswith('linear'):
   weight=(ix-left)/np.maximum(1,right-left);prior=(1-weight)*zz[left]+weight*zz[right]
  else:prior=.5*(zz[left]+zz[right])
  zz[valid]=(1-strength)*zz[valid]+strength*prior[valid]
  cc=np.full((n,2),cost,dtype=float)
  if touch is not None:cc/=1+np.clip(touch[lo:hi],0,1)
  out[lo:hi]=segment_decode(zz,cc)
 return out
