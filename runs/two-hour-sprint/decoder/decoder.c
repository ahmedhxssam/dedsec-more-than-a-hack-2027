
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
