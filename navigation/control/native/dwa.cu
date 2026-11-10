// Bounded double-precision DWA integration/scoring on resident map layers.
// Final finite-horizon swept geometry is independently certified.
#include <cuda_runtime.h>
#include <cmath>

namespace {
struct Pose {double x,y,a;};struct Vel {double v,w;};
__device__ double clip(double x,double a,double b){return fmin(b,fmax(a,x));}
__device__ double wrap(double a){return atan2(sin(a),cos(a));}
__device__ Vel response(Vel v,Vel t,double a,double aw,double dt){
    // Match velocity_step: opposite directions must pass through a full stop.
    if(v.v*t.v<0)t={0.,0.};
    double f=fmin(1.,fmin(a*dt/fmax(fabs(t.v-v.v),1e-15),aw*dt/fmax(fabs(t.w-v.w),1e-15)));
    return {v.v+f*(t.v-v.v),v.w+f*(t.w-v.w)};
}
__device__ Pose move(Pose p,Vel v,double dt){double z=v.w*dt,half=z/2;
    double length=v.v*dt*(fabs(half)<1e-12?1.:sin(half)/half);
    return {p.x+length*cos(p.a+half),p.y+length*sin(p.a+half),p.a+z};}
__device__ void save(double* p,int i,double t,Pose x){p[4*i]=t;p[4*i+1]=x.x;p[4*i+2]=x.y;p[4*i+3]=x.a;}
__device__ Pose load(const double* p,int i){return {p[4*i+1],p[4*i+2],p[4*i+3]};}
__device__ bool coarse(Pose p,const signed char* grid,int h,int w,double ox,double oy,double res,const double* fp,int nv){
    double c=cos(p.a),s=sin(p.a);for(int k=0;k<nv;++k){int x=floor((p.x+c*fp[2*k]-s*fp[2*k+1]-ox)/res);
        int y=floor((p.y+s*fp[2*k]+c*fp[2*k+1]-oy)/res);
        if(x<0||y<0||x>=w||y>=h||grid[y*w+x]!=0)return false;}return true;}
__global__ void evaluate(const double* targets,int count,const double* path,const double* arc,int paths,
    const double* q,const signed char* grid,const float* clearance,int h,int w,const double* fp,int vertices,
    const double* people,int people_count,const int* pairs,int pair_count,const double* social,
    double* poses,double* scores,int n){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;
    // q: start3, initial2, acceleration2, horizon, sample_dt, origin2,
    // resolution, progress, previous2, weights5, people_tau.
    Pose start{q[0],q[1],q[2]},p=start;Vel v{q[3],q[4]},target{targets[2*j],targets[2*j+1]};
    double* prediction=poses+j*(n+1)*4;
    double step=q[7]/n;save(prediction,0,0,p);
    for(int i=0;i<n;++i){
        Vel next=response(v,target,q[5],q[6],step);p=move(p,{(v.v+next.v)/2,(v.w+next.w)/2},step);v=next;save(prediction,i+1,(i+1)*step,p);}
    for(int i=0;i<=n;++i)if(!coarse(load(prediction,i),grid,h,w,q[9],q[10],q[11],fp,vertices)){scores[j]=INFINITY;return;}
    double sum=0,end_offset=0,end_progress=0;
    const bool rotating=arc[paths-1]<=1e-7;
    for(int i=0;i<=n;++i){Pose x=load(prediction,i);double best=INFINITY,progress=0;
        if(rotating)best=hypot(x.x-path[0],x.y-path[1]);
        for(int a=0;a<paths-1;++a){double dx=path[3*(a+1)]-path[3*a],dy=path[3*(a+1)+1]-path[3*a+1];
            double length=arc[a+1]-arc[a];if(length<=1e-7)continue;
            double f=clip(((x.x-path[3*a])*dx+(x.y-path[3*a+1])*dy)/(length*length),0.,1.);
            double d=hypot(x.x-path[3*a]-f*dx,x.y-path[3*a+1]-f*dy);
            if(d<best){best=d;progress=arc[a]+f*length;}}
        sum+=best;if(i==n){end_offset=best;end_progress=progress;}}
    Pose end=load(prediction,n);double yaw=rotating?path[3*(paths-1)+2]:path[2];
    for(int a=0;!rotating&&a<paths-1;++a){double next=yaw+wrap(path[3*(a+1)+2]-yaw);
        if(end_progress<=arc[a+1]){double f=clip((end_progress-arc[a])/(arc[a+1]-arc[a]),0.,1.);yaw+=f*(next-yaw);break;}yaw=next;}
    double score=q[15]*(sum/(n+1)+end_offset)-q[16]*(end_progress-q[12])+q[17]*fabs(wrap(yaw-end.a))+
        q[18]*hypot(target.v-q[13],target.w-q[14]);
    if(clearance){int x=floor((end.x-q[9])/q[11]),y=floor((end.y-q[10])/q[11]);
        if(x>=0&&y>=0&&x<w&&y<h&&isfinite(clearance[y*w+x]))score+=q[19]/fmax(.1,double(clearance[y*w+x]));}
    for(int i=0;i<n&&people_count;++i){Pose a=load(prediction,i),b=load(prediction,i+1);
        double angle=b.a-a.a,dx=b.x-a.x,dy=b.y-a.y;
        double x=(a.x+b.x)/2+dy*tan(angle/4)/2,y=(a.y+b.y)/2-dx*tan(angle/4)/2;
        double tau=q[20]+(i+.5)*step,value=0;
        for(int t=0;t<people_count;++t){const double* p=people+4*t;double vx=p[2],vy=p[3],speed=hypot(vx,vy);
            bool moving=speed>.15;if(!moving){vx=1;vy=0;speed=1;}vx/=speed;vy/=speed;
            double ex=x-p[0]-tau*p[2],ey=y-p[1]-tau*p[3],along=ex*vx+ey*vy,across=ex*vy-ey*vx;
            double front=!moving?social[2]:along>=0?social[0]:social[1];
            value+=social[3]*exp(-.5*(along*along/(front*front)+across*across/(social[2]*social[2])));}
        for(int t=0;t<pair_count;++t){const double* a=people+4*pairs[2*t];const double* b=people+4*pairs[2*t+1];
            double ax=a[0]+tau*a[2],ay=a[1]+tau*a[3],dx=b[0]+tau*b[2]-ax,dy=b[1]+tau*b[3]-ay;
            double f=clip(((x-ax)*dx+(y-ay)*dy)/fmax(dx*dx+dy*dy,1e-9),0.,1.);
            double d=hypot(x-ax-f*dx,y-ay-f*dy);value+=social[4]*exp(-.5*d*d/(social[5]*social[5]));}
        score+=value*step;}
    scores[j]=score;
}
}
extern "C" int sn_dwa_predict(const double* targets,int count,const double* path,const double* arc,int paths,
    const double* parameters,const signed char* grid,const float* clearance,int height,int width,const double* footprint,int vertices,
    const double* people,int people_count,const int* pairs,int pair_count,const double* social,
    double* poses,double* scores,int steps,void* stream){
    if(count<1||count>64||paths<2||steps<2||steps>128||vertices<3||vertices>16)return -1;
    evaluate<<<(count+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(targets,count,path,arc,paths,parameters,grid,clearance,height,width,footprint,vertices,
        people,people_count,pairs,pair_count,social,poses,scores,steps);
    return int(cudaGetLastError());
}
extern "C" int sn_dwa_supports_rotation(){return 1;}
