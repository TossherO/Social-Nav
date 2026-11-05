// One cell per thread. All positive-area overlaps of the collision column's
// XY zonotope must have measured static FREE, compatible ground and old time.
struct SupportPolicy { double v[30]; int h,w,ph,pw,dx,dy; };
__global__ void support_cells(const int8_t* raw,const double* seen,const double* prior,
    const double* previous,double* hits,double* out,uint8_t* allowed,double* floors,SupportPolicy p) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,n=p.h*p.w;if(i>=n)return;
    const double* v=p.v;const double* t=v+18;const double* plane=v+15;
    int x=i%p.w,y=i/p.w,ox=x-p.dx,oy=y-p.dy;
    for(int k=0;k<2;++k)hits[k*n+i]=previous && ox>=0 && ox<p.w && oy>=0 && oy<p.h?previous[k*n+oy*p.w+ox]:-INFINITY;
    out[i]=raw[i];out[n+i]=raw[i]==0?1:raw[i]==100?2:0;out[2*n+i]=raw[i]==-1?NAN:seen[i];allowed[i]=0;floors[i]=NAN;
    double X=(x+.5)*v[0]+v[2],Y=(y+.5)*v[0]+v[3];
    double rx=fabs(X-v[6])+v[0]/2,ry=fabs(Y-v[7])+v[0]/2;
    if(rx*rx+ry*ry>v[8]*v[8])return;
    double Z=X*plane[0]+Y*plane[1]+plane[2];
    int px=floor((t[0]*X+t[1]*Y+t[2]*Z+t[3]-v[4])/v[1]);
    int py=floor((t[4]*X+t[5]*Y+t[6]*Z+t[7]-v[5])/v[1]);
    if(px<0 || px>=p.pw || py<0 || py>=p.ph)return;
    int pn=p.ph*p.pw,pi=py*p.pw+px;
    if(prior[pi]!=0 || !prior[2*pn+pi] || !isfinite(prior[pn+pi]))return;
    Z=(prior[pn+pi]-X*t[8]-Y*t[9]-t[11])/t[10];floors[i]=Z;
    double mid=Z+(v[10]+v[11])/2;
    double cx=t[0]*X+t[1]*Y+t[2]*mid+t[3],cy=t[4]*X+t[5]*Y+t[6]*mid+t[7];
    double ax=t[0],ay=t[1],bx=t[4],by=t[5],det=ax*by-ay*bx;
    double gx[3]={ax*v[0]/2,ay*v[0]/2,t[2]*(v[11]-v[10])/2};
    double gy[3]={bx*v[0]/2,by*v[0]/2,t[6]*(v[11]-v[10])/2};
    double ex=fabs(gx[0])+fabs(gx[1])+fabs(gx[2]),ey=fabs(gy[0])+fabs(gy[1])+fabs(gy[2]);
    int lx=floor((cx-ex-v[4])/v[1]+1e-9),hx=floor((cx+ex-v[4])/v[1]-1e-9);
    int ly=floor((cy-ey-v[5])/v[1]+1e-9),hy=floor((cy+ey-v[5])/v[1]-1e-9);
    double groundx=t[0]*X+t[1]*Y+t[2]*Z+t[3],groundy=t[4]*X+t[5]*Y+t[6]*Z+t[7];
    double oldest=INFINITY;
    for(int yy=ly;yy<=hy;++yy)for(int xx=lx;xx<=hx;++xx) {
        double u=(xx+.5)*v[1]+v[4],w=(yy+.5)*v[1]+v[5],dx=u-cx,dy=w-cy;
        bool overlap=fabs(dx)<ex+v[1]/2-1e-10 && fabs(dy)<ey+v[1]/2-1e-10;
        for(int k=0;k<3 && overlap;++k) {
            double norm=hypot(gx[k],gy[k]);if(norm<=1e-10)continue;
            double nx=-gy[k]/norm,ny=gx[k]/norm;
            double radius=v[1]/2*(fabs(nx)+fabs(ny));
            for(int j=0;j<3;++j)radius+=fabs(gx[j]*nx+gy[j]*ny);
            overlap=fabs(dx*nx+dy*ny)<radius-1e-10;
        }
        if(!overlap)continue;
        if(xx<0 || xx>=p.pw || yy<0 || yy>=p.ph)return;
        int j=yy*p.pw+xx;
        double localx=X+(by*(u-groundx)-ay*(w-groundy))/det;
        double localy=Y+(-bx*(u-groundx)+ax*(w-groundy))/det;
        double mapped=t[8]*localx+t[9]*localy+t[10]*Z+t[11];
        if(prior[j]!=0 || !prior[2*pn+j] || !isfinite(prior[pn+j]) ||
           fabs(prior[pn+j]-mapped)>v[12] || !isfinite(prior[3*pn+j]) || prior[3*pn+j]>v[9])return;
        oldest=fmin(oldest,prior[3*pn+j]);
    }
    if(!isfinite(oldest))return;
    allowed[i]=1;
    if(raw[i]==-1) {out[i]=0;out[n+i]=3;out[2*n+i]=oldest;}
}
__global__ void support_hits(const double* xyz,const uint8_t* moving,int count,const double* floors,double* hits,SupportPolicy p) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;
    double x=xyz[3*j],y=xyz[3*j+1];
    int xx=floor((x-p.v[2])/p.v[0]),yy=floor((y-p.v[3])/p.v[0]);
    if(xx<0 || xx>=p.w || yy<0 || yy>=p.h)return;
    double height=xyz[3*j+2]-floors[yy*p.w+xx];
    if(!isfinite(height) || height<p.v[10] || height>p.v[11])return;
    max_atomic(hits+(moving[j]?p.h*p.w:0)+yy*p.w+xx,p.v[9]);
}
__global__ void support_overlay(const int8_t* raw,const uint8_t* allowed,const double* hits,double* out,SupportPolicy p) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,n=p.h*p.w;if(i>=n || !allowed[i] || raw[i]==100)return;
    if(p.v[9]-hits[i]<=p.v[13] || p.v[9]-hits[n+i]<=p.v[14]) {
        out[i]=100;out[n+i]=10;out[2*n+i]=fmax(hits[i],hits[n+i]);
    }
}
