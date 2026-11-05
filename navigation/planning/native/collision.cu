// Exact double-precision polygon/cell SAT, one block per sweep enclosure.
// No ROS, Python, PyTorch or third-party CUDA library dependency.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
void checked(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Context {
    signed char* grid=nullptr;double* polygons=nullptr;double* margins=nullptr;int* status=nullptr;
    const signed char* external_grid=nullptr;
    size_t cells=0,points=0,count=0;int height=0,width=0;
    ~Context(){cudaFree(grid);cudaFree(polygons);cudaFree(margins);cudaFree(status);}
};
template<class T> void reserve(T*& p,size_t size,size_t& capacity) {
    if(size<=capacity)return;cudaFree(p);p=nullptr;
    checked(cudaMalloc((void**)&p,size*sizeof(T)));capacity=size;
}
__global__ void certify(const signed char* grid,int height,int width,const double* polygons,const double* margins,
        int vertices,double ox,double oy,double resolution,int* output) {
    int k=blockIdx.x,tid=threadIdx.x;const double* p=polygons+k*vertices*2;
    __shared__ int lowx,lowy,highx,highy,status;
    __shared__ double ax[18],ay[18],minimum[18],maximum[18];
    if(tid==0) {
        double xmin=INFINITY,xmax=-INFINITY,ymin=INFINITY,ymax=-INFINITY;
        for(int v=0;v<vertices;++v) {
            xmin=fmin(xmin,p[2*v]);xmax=fmax(xmax,p[2*v]);ymin=fmin(ymin,p[2*v+1]);ymax=fmax(ymax,p[2*v+1]);
        }
        double m=margins[k];lowx=floor((xmin-m-ox)/resolution);highx=floor((xmax+m-ox)/resolution);
        lowy=floor((ymin-m-oy)/resolution);highy=floor((ymax+m-oy)/resolution);
        status=(lowx<0 || lowy<0 || highx>=width || highy>=height)?3:0;
        ax[0]=1;ay[0]=0;ax[1]=0;ay[1]=1;
        for(int v=0;v<vertices;++v) {
            int n=(v+1)%vertices;ax[v+2]=-(p[2*n+1]-p[2*v+1]);ay[v+2]=p[2*n]-p[2*v];
        }
        for(int a=0;a<vertices+2;++a) {
            double lo=INFINITY,hi=-INFINITY;
            for(int v=0;v<vertices;++v) {double z=p[2*v]*ax[a]+p[2*v+1]*ay[a];lo=fmin(lo,z);hi=fmax(hi,z);}
            minimum[a]=lo;maximum[a]=hi;
        }
    }
    __syncthreads();
    if(status==3) {if(tid==0)output[k]=3;return;}
    int nx=highx-lowx+1,n=(highy-lowy+1)*nx;
    for(int j=tid;j<n;j+=blockDim.x) {
        int x=lowx+j%nx,y=lowy+j/nx;int value=grid[y*width+x];if(value==0)continue;
        double cx=ox+(x+.5)*resolution,cy=oy+(y+.5)*resolution;bool overlap=true;
        for(int a=0;a<vertices+2;++a) {
            double r=(resolution/2+margins[k])*(fabs(ax[a])+fabs(ay[a]));
            double z=cx*ax[a]+cy*ay[a];
            if(z+r<minimum[a]-1e-10 || z-r>maximum[a]+1e-10) {overlap=false;break;}
        }
        if(overlap)atomicMax(&status,value==100?2:1);
    }
    __syncthreads();if(tid==0)output[k]=status;
}
}
extern "C" {
const char* sn_collision_error(){return error.c_str();}
void* sn_collision_create(){try{checked(cudaFree(nullptr));return new Context;}catch(const std::exception& e){error=e.what();return nullptr;}}
void sn_collision_destroy(void* p){delete static_cast<Context*>(p);}
int sn_collision_grid(void* raw,const signed char* input,int height,int width) {
    try {
        if(!raw || height<=0 || width<=0 || (size_t)height*width>4000000)throw std::runtime_error("Invalid collision grid");
        auto& c=*static_cast<Context*>(raw);size_t size=(size_t)height*width;reserve(c.grid,size,c.cells);
        checked(cudaMemcpy(c.grid,input,size,cudaMemcpyHostToDevice));c.external_grid=nullptr;c.height=height;c.width=width;return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
int sn_collision_device_grid(void* raw,const signed char* input,int height,int width) {
    if(!raw||!input||height<=0||width<=0||(size_t)height*width>4000000){error="Invalid resident collision grid";return -1;}
    auto& c=*static_cast<Context*>(raw);c.external_grid=input;c.height=height;c.width=width;return 0;
}
int sn_collision_batch_stream(void* raw,const double* polygons,const double* margins,int count,int vertices,
        double ox,double oy,double resolution,int* output,void* stream_ptr) {
    try {
        if(!raw || count<=0 || count>20000 || vertices<3 || vertices>16 || !std::isfinite(resolution) || resolution<=0)
            throw std::runtime_error("Invalid collision batch");
        auto& c=*static_cast<Context*>(raw);size_t points=(size_t)count*vertices*2;
        reserve(c.polygons,points,c.points);
        if((size_t)count>c.count) {
            cudaFree(c.margins);cudaFree(c.status);c.margins=nullptr;c.status=nullptr;
            checked(cudaMalloc((void**)&c.margins,count*sizeof(double)));checked(cudaMalloc((void**)&c.status,count*sizeof(int)));c.count=count;
        }
        auto stream=static_cast<cudaStream_t>(stream_ptr);
        checked(cudaMemcpyAsync(c.polygons,polygons,points*sizeof(double),cudaMemcpyHostToDevice,stream));
        checked(cudaMemcpyAsync(c.margins,margins,count*sizeof(double),cudaMemcpyHostToDevice,stream));
        certify<<<count,128,0,stream>>>(c.external_grid?c.external_grid:c.grid,c.height,c.width,c.polygons,c.margins,vertices,ox,oy,resolution,c.status);
        checked(cudaGetLastError());checked(cudaMemcpyAsync(output,c.status,count*sizeof(int),cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
int sn_collision_batch(void* raw,const double* polygons,const double* margins,int count,int vertices,
        double ox,double oy,double resolution,int* output) {
    return sn_collision_batch_stream(raw,polygons,margins,count,vertices,ox,oy,resolution,output,nullptr);
}
}
