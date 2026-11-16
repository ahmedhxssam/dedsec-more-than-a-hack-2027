
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
