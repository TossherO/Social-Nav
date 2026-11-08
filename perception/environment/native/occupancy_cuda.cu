// Rolling, world-aligned CUDA evidence with the same C ABI as the CPU backend.
// Ray traversal follows Amanatides-Woo and OctoMap's float direction, tie order,
// endpoint exclusion and discretization stop rule (OctoMap 1.9, BSD license).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <limits>
#include <cfloat>

namespace {
thread_local std::string error;
void check(cudaError_t result) { if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
struct Guard {
    int previous;
    explicit Guard(int device) { check(cudaGetDevice(&previous)); check(cudaSetDevice(device)); }
    ~Guard() { cudaSetDevice(previous); }
};
template<class T> struct Buffer {
    T* data=nullptr; size_t capacity=0;
    void reserve(size_t n) {
        if(n<=capacity) return;
        if(data) check(cudaFree(data)); data=nullptr; capacity=0;
        check(cudaMalloc((void**)&data,n*sizeof(T))); capacity=n;
    }
    ~Buffer() { if(data) cudaFree(data); }
};
struct Voxel { double stamp, hit_stamp,ordinary_stamp=-INFINITY; float odds; uint8_t valid, transient; };
struct NearCell {
    double classified, free_seen, hit_stamp, expires, hit_min, hit_max;
    float free_ground, free_coverage, ground;
    double ordinary_stamp=-INFINITY;
};
struct NearPolicy {
    int height,width; double x,y,resolution,stamp,minimum,maximum,coverage;
    double settings[10],pose[16],plane[3]; bool has_plane,history_suspended;
};
struct Grid {
    int3 lo,dim; double resolution; double bounds[6];
    __device__ int size() const { return dim.x*dim.y*dim.z; }
    __device__ int3 key(int linear) const {
        return make_int3(lo.x+linear%dim.x,lo.y+(linear/dim.x)%dim.y,lo.z+linear/(dim.x*dim.y));
    }
    __device__ bool contains(int3 k) const {
        return k.x>=lo.x && k.x<lo.x+dim.x && k.y>=lo.y && k.y<lo.y+dim.y && k.z>=lo.z && k.z<lo.z+dim.z;
    }
    __device__ int slot(int3 k) const {
        int x=(k.x%dim.x+dim.x)%dim.x, y=(k.y%dim.y+dim.y)%dim.y, z=(k.z%dim.z+dim.z)%dim.z;
        return (z*dim.y+y)*dim.x+x;
    }
    __device__ float3 center(int3 k) const {
        return make_float3(float((k.x+.5)*resolution),float((k.y+.5)*resolution),float((k.z+.5)*resolution));
    }
    __device__ bool inside(float3 p) const {
        return p.x>=bounds[0] && p.y>=bounds[1] && p.z>=bounds[2] && p.x<bounds[3] && p.y<bounds[4] && p.z<bounds[5];
    }
    __device__ int3 coordinate(float3 p) const {
        return make_int3(int(floor((1./resolution)*p.x)),int(floor((1./resolution)*p.y)),int(floor((1./resolution)*p.z)));
    }
};
struct Tree {
    int device,count=0; double resolution,obstacle_ttl=INFINITY; Grid grid{}; bool initialized=false,hit_invalidates_free=false;
    Buffer<Voxel> voxels; Buffer<unsigned> flags; Buffer<int> count_device;
    Buffer<float> points,origins,snapshot; Buffer<uint8_t> transient;
    Buffer<double> floor,endpoints,output; Buffer<uint8_t> support; Buffer<int> required; Buffer<unsigned> free;
    Buffer<NearCell> memory_a,memory_b; Buffer<unsigned> memory_flags;
    Buffer<double> view,ring,hit_bounds; Buffer<int> memory_counts; Buffer<float> distance_rows;
    Buffer<uint8_t> view_transient;Buffer<unsigned> person_columns;
    bool memory_initialized=false,history_suspended=false; int memory_height=0,memory_width=0;
    double memory_x=0,memory_y=0,memory_resolution=0,memory_stamp=0;
    Tree(double r,int d):device(d),resolution(r) { Guard g(d); count_device.reserve(1); }
};
__host__ __device__ bool equal(int3 a,int3 b) { return a.x==b.x && a.y==b.y && a.z==b.z; }
__device__ bool key_valid(int3 k) { return k.x>=-32768 && k.x<32768 && k.y>=-32768 && k.y<32768 && k.z>=-32768 && k.z<32768; }
__global__ void prepare(Voxel* voxels,unsigned* flags,Grid now,Grid previous,bool initialized,double stamp,double ttl,double transient_ttl,double obstacle_ttl) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=now.size()) return;
    int3 k=now.key(i); int slot=now.slot(k); auto& v=voxels[slot]; flags[slot]=0;
    if(!initialized || !previous.contains(k) || !now.inside(now.center(k)) ||
       (v.valid && (stamp-v.stamp>ttl || (v.odds>0 && stamp-v.hit_stamp>
            (v.transient?fmin(obstacle_ttl,transient_ttl):obstacle_ttl) && stamp-v.ordinary_stamp>obstacle_ttl)))) v=Voxel{};
}
__device__ void mark(unsigned* flags,Grid grid,int3 key,unsigned value) {
    if(grid.contains(key)) atomicOr(flags+grid.slot(key),value);
}
__global__ void trace(unsigned* flags,Grid grid,const float* xyz,int n,float3 origin,const float* origins,
                      int stride,double maxrange,const uint8_t* transient,bool omit) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float3 sensor=origins?make_float3(origins[3*i],origins[3*i+1],origins[3*i+2]):origin;
    float3 p=make_float3(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);
    float3 d=make_float3(p.x-sensor.x,p.y-sensor.y,p.z-sensor.z);
    double length=sqrt(double((d.x*d.x+d.y*d.y)+d.z*d.z));
    if(!isfinite(length) || length<.01) return;
    int3 endpoint=grid.coordinate(p);
    if(length<=maxrange && grid.inside(p) && key_valid(endpoint)) {
        bool moving=transient && transient[i];
        if(!moving || !omit) mark(flags,grid,endpoint,moving?2:4);
    }
    if(!stride || i%stride) return;
    if(length>maxrange) {
        float scale=float(maxrange/length);
        p=make_float3(sensor.x+d.x*scale,sensor.y+d.y*scale,sensor.z+d.z*scale);
    }
    int3 current=grid.coordinate(sensor); endpoint=grid.coordinate(p);
    if(!key_valid(current) || !key_valid(endpoint) || equal(current,endpoint)) return;
    if(grid.inside(grid.center(current))) mark(flags,grid,current,1);
    d=make_float3(p.x-sensor.x,p.y-sensor.y,p.z-sensor.z);
    float ray_length=float(sqrt(double((d.x*d.x+d.y*d.y)+d.z*d.z)));
    d=make_float3(d.x/ray_length,d.y/ray_length,d.z/ray_length);
    int step[3]; double limit[3],delta[3];
    float direction[3]={d.x,d.y,d.z},start[3]={sensor.x,sensor.y,sensor.z};
    int* key=&current.x;
    for(int axis=0;axis<3;++axis) {
        step[axis]=(direction[axis]>0)-(direction[axis]<0);
        if(step[axis]) {
            double border=(key[axis]+.5)*grid.resolution+float(step[axis]*grid.resolution*.5);
            limit[axis]=(border-start[axis])/direction[axis]; delta[axis]=grid.resolution/fabs(double(direction[axis]));
        } else limit[axis]=delta[axis]=DBL_MAX;
    }
    // Every step advances a key. The key-domain Manhattan distance bounds work.
    int budget=abs(current.x-endpoint.x)+abs(current.y-endpoint.y)+abs(current.z-endpoint.z)+3;
    for(int j=0;j<budget;++j) {
        int axis=limit[0]<limit[1]?(limit[0]<limit[2]?0:2):(limit[1]<limit[2]?1:2);
        key[axis]+=step[axis]; limit[axis]+=delta[axis];
        if(equal(current,endpoint) || fmin(fmin(limit[0],limit[1]),limit[2])>ray_length) break;
        if(grid.inside(grid.center(current))) mark(flags,grid,current,1);
    }
}
__global__ void apply(Voxel* voxels,const unsigned* flags,int size,double stamp,int* count,bool invalidate_free) {
    __shared__ int counts[256];
    int i=blockIdx.x*blockDim.x+threadIdx.x,valid=0;
    if(i<size) {
        auto& v=voxels[i]; unsigned f=flags[i];
        if(f) {
            bool hit=f&6;
            // OctoMap stores its sensor-model log odds and additions in float.
            float previous=(hit && invalidate_free)?fmaxf(0.f,v.odds):v.odds;
            float odds=previous+(hit?float(0.8472978603872034):float(-0.4054651081081643));
            v.odds=fmaxf(float(-2.000027830777221),fminf(float(3.5110306383048497),odds));
            v.stamp=stamp; v.valid=1; if(hit) { v.transient=!(f&4); v.hit_stamp=stamp;if(f&4)v.ordinary_stamp=stamp; }
        }
        valid=v.valid;
    }
    counts[threadIdx.x]=valid; __syncthreads();
    for(int span=128;span;span/=2) { if(threadIdx.x<span) counts[threadIdx.x]+=counts[threadIdx.x+span]; __syncthreads(); }
    if(!threadIdx.x) atomicAdd(count,counts[0]);
}
__device__ double max_atomic(double* ptr,double value) {
    auto p=(unsigned long long*)ptr; unsigned long long old=*p,assumed;
    do { assumed=old; if(__longlong_as_double(old)>=value) break;
        old=atomicCAS(p,assumed,__double_as_longlong(value)); } while(old!=assumed);
    return __longlong_as_double(old);
}
__device__ void min_atomic(double* ptr,double value) { // finite/infinite, never NaN
    auto p=(unsigned long long*)ptr; unsigned long long old=*p,assumed;
    do { assumed=old; if(__longlong_as_double(old)<=value) break;
        old=atomicCAS(p,assumed,__double_as_longlong(value)); } while(old!=assumed);
}
struct Projection { int height,width,bins; double x,y,resolution,minimum,maximum,occupied,free,coverage; bool float_height; };
__device__ int cell(double x,double y,Projection p) {
    int ix=int(floor((x-p.x)/p.resolution)),iy=int(floor((y-p.y)/p.resolution));
    return ix>=0 && ix<p.width && iy>=0 && iy<p.height?iy*p.width+ix:-1;
}
__global__ void clear_layers(double* output,int size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=size) return;
    output[i]=-1; output[size+i]=0; output[2*size+i]=-INFINITY;
    output[3*size+i]=0; output[4*size+i]=INFINITY; output[5*size+i]=-INFINITY;
}
__device__ void hit(double* output,int size,int i,double height) {
    max_atomic(output+i,100.); min_atomic(output+4*size+i,height); max_atomic(output+5*size+i,height);
}
__global__ void project_voxels(const Voxel* voxels,Grid grid,const double* floor,Projection p,
                               unsigned* free,double* output,double stamp) {
    int index=blockIdx.x*blockDim.x+threadIdx.x; if(index>=grid.size()) return;
    int3 k=grid.key(index); const auto& v=voxels[grid.slot(k)]; if(!v.valid) return;
    float3 xyz=grid.center(k); int i=cell(xyz.x,xyz.y,p),size=p.height*p.width; if(i<0) return;
    max_atomic(output+size+i,1.); max_atomic(output+2*size+i,stamp-double(float(stamp-v.stamp)));
    double h=p.float_height?double(float(xyz.z-float(floor[i]))):double(xyz.z)-floor[i];
    if(!isfinite(floor[i])) return;
    int first=ceil((floor[i]+p.minimum)/p.resolution-.5-1e-5);
    int last=::floor((floor[i]+p.maximum)/p.resolution-.5+1e-5);
    if(k.z<first || k.z>last) return;
    float probability=1.f-1.f/(1.f+expf(v.odds));
    if(probability>=float(p.occupied)) hit(output,size,i,h);
    if(probability<=float(p.free)) {
        int bin=k.z-first;
        atomicExch(free+i*p.bins+bin,1);
    }
}
__global__ void project_endpoints(const double* xyz,int n,const double* ground,Projection p,double* output) {
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=n) return;
    int i=cell(xyz[3*j],xyz[3*j+1],p); if(i<0) return;
    double h=xyz[3*j+2]-ground[i];
    if(isfinite(h) && h>=p.minimum && h<=p.maximum) hit(output,p.height*p.width,i,h);
}
__global__ void finish_layers(double* output,const unsigned* free,const uint8_t* support,const int* required,Projection p) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,size=p.height*p.width; if(i>=size) return;
    int count=0; for(int z=0;z<p.bins;++z) count+=free[i*p.bins+z];
    output[3*size+i]=float(fmin(1.,double(count)/required[i]));
    if(output[3*size+i]>=float(p.coverage) && support[i]==1 && output[i]!=100) output[i]=0;
}
__global__ void read_voxels(const Voxel* voxels,Grid grid,double stamp,float* out,int* count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=grid.size()) return;
    int3 k=grid.key(i); const auto& v=voxels[grid.slot(k)]; if(!v.valid) return;
    int j=atomicAdd(count,1); float3 p=grid.center(k);
    out[5*j]=p.x; out[5*j+1]=p.y; out[5*j+2]=p.z;
    out[5*j+3]=1.f-1.f/(1.f+expf(v.odds)); out[5*j+4]=stamp-v.stamp;
}
template<class T> void upload(Buffer<T>& target,const T* input,size_t count) {
    if(!count) return; target.reserve(count); check(cudaMemcpy(target.data,input,count*sizeof(T),cudaMemcpyHostToDevice));
}
int blocks(size_t count) { return int((count+255)/256); }
void project_resident(Tree& t,const double* floor,const uint8_t* support,const int32_t* required,
    int height,int width,const double* origin,const double* p,double stamp,const double* endpoints,int n) {
    int size=height*width,bins=std::ceil((p[2]-p[1])/p[0])+1;
    if(size<1 || size>360000 || bins<1 || bins>256) throw std::runtime_error("CUDA projection grid budget exceeded");
    Projection config{height,width,bins,origin[0],origin[1],p[0],p[1],p[2],p[3],p[4],p[5],p[6]!=0};
    upload(t.floor,floor,size);upload(t.support,support,size);upload(t.required,required,size);
    upload(t.endpoints,endpoints,size_t(n)*3);t.output.reserve(size_t(size)*6);t.free.reserve(size_t(size)*bins);
    check(cudaMemset(t.free.data,0,size_t(size)*bins*sizeof(unsigned)));
    clear_layers<<<blocks(size),256>>>(t.output.data,size);
    if(t.initialized) {
        int cells=t.grid.dim.x*t.grid.dim.y*t.grid.dim.z;
        project_voxels<<<blocks(cells),256>>>(t.voxels.data,t.grid,t.floor.data,config,t.free.data,t.output.data,stamp);
    }
    if(n) project_endpoints<<<blocks(n),256>>>(t.endpoints.data,n,t.floor.data,config,t.output.data);
    finish_layers<<<blocks(size),256>>>(t.output.data,t.free.data,t.support.data,t.required.data,config);
}
__global__ void person_voxels(const Voxel* voxels,Grid grid,const double* ground,Projection p,unsigned* tags,double stamp,double ordinary_ttl) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=grid.size())return;
    const auto& v=voxels[grid.slot(grid.key(j))];if(!v.valid)return;
    float probability=1.f-1.f/(1.f+expf(v.odds));if(probability<float(p.occupied))return;
    float3 xyz=grid.center(grid.key(j));int i=cell(xyz.x,xyz.y,p);if(i<0)return;
    double h=double(xyz.z)-ground[i];
    if(isfinite(h) && h>=p.minimum && h<=p.maximum)atomicOr(tags+i,v.transient && stamp-v.ordinary_stamp>ordinary_ttl?1u:2u);
}
__global__ void person_endpoints(const double* xyz,int n,const uint8_t* moving,const double* ground,Projection p,unsigned* tags) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    int i=cell(xyz[3*j],xyz[3*j+1],p);if(i<0)return;double h=xyz[3*j+2]-ground[i];
    if(isfinite(h) && h>=p.minimum && h<=p.maximum)atomicOr(tags+i,moving && moving[j]?1u:2u);
}
#include "local_view.cuh"
#include "support.cuh"
}
extern "C" {
int sn_occupancy_local_ground_conflicts() { return 1; }
int sn_navigation_support_with_ground_conflicts(const int8_t* raw,const double* seen,const double* prior,
    const double* previous,double* hits,double* out,uint8_t* allowed,const double* points,
    const uint8_t* moving,const uint8_t* conflicts,int count,const double* params,int h,int w,int ph,int pw,int dx,int dy,
    void* stream,int device) {
    try {
        Guard guard(device);SupportPolicy p{};std::copy(params,params+30,p.v);
        p.h=h;p.w=w;p.ph=ph;p.pw=pw;p.dx=dx;p.dy=dy;
        auto s=static_cast<cudaStream_t>(stream);
        support_cells<<<blocks(h*w),256,0,s>>>(raw,seen,prior,previous,hits,out,allowed,conflicts,p);
        if(count)support_hits<<<blocks(count),256,0,s>>>(points,moving,count,hits,p);
        support_overlay<<<blocks(h*w),256,0,s>>>(raw,allowed,hits,out,p);
        check(cudaGetLastError());return 0;
    } catch(const std::exception& e) { error=e.what();return -1; }
}
const char* sn_occupancy_error() { return error.c_str(); }
int sn_occupancy_obstacle_lifetime(void* ptr,double ttl) {
    if(!std::isfinite(ttl) || ttl<=0) { error="Invalid obstacle lifetime"; return -1; }
    static_cast<Tree*>(ptr)->obstacle_ttl=ttl; return 0;
}
int sn_occupancy_hit_invalidates_free(void* ptr,int enabled) {
    static_cast<Tree*>(ptr)->hit_invalidates_free=enabled!=0; return 0;
}
void* sn_occupancy_new_cuda(double resolution,int device) {
    try { return new Tree(resolution,device); } catch(const std::exception& e) { error=e.what(); return nullptr; }
}
void* sn_occupancy_new(double resolution) { return sn_occupancy_new_cuda(resolution,0); }
void sn_occupancy_free(void* ptr) {
    if(!ptr) return; auto* tree=static_cast<Tree*>(ptr); Guard guard(tree->device); delete tree;
}
int sn_occupancy_update_masked(void* ptr,const float* xyz,int n,const double* origin,const float* origins,
    const double* bounds,double stamp,double ttl,double maxrange,int stride,int maxvoxels,
    const uint8_t* transient,double transient_ttl,int omit) {
    try {
        auto& t=*static_cast<Tree*>(ptr); Guard guard(t.device); Grid next{}; next.resolution=t.resolution;
        int* lo=&next.lo.x; int* dim=&next.dim.x; size_t size=1;
        for(int a=0;a<3;++a) {
            lo[a]=std::floor(bounds[a]*(1./t.resolution));
            // Fixed span plus boundary padding prevents floating boundary noise
            // from changing dimensions and unnecessarily clearing rolling history.
            dim[a]=std::ceil((bounds[a+3]-bounds[a])/t.resolution-1.e-7)+2;
            if(dim[a]<1) throw std::runtime_error("Invalid CUDA voxel bounds"); size*=dim[a];
        }
        if(size>8000000) throw std::runtime_error("CUDA rolling storage budget exceeded (8 million cells)");
        std::copy(bounds,bounds+6,next.bounds);
        bool compatible=t.initialized && equal(t.grid.dim,next.dim);
        t.voxels.reserve(size); t.flags.reserve(size);
        prepare<<<blocks(size),256>>>(t.voxels.data,t.flags.data,next,t.grid,compatible,stamp,ttl,transient_ttl,t.obstacle_ttl);
        upload(t.points,xyz,size_t(n)*3); if(origins) upload(t.origins,origins,size_t(n)*3);
        if(transient) upload(t.transient,transient,size_t(n));
        if(n) trace<<<blocks(n),256>>>(t.flags.data,next,t.points.data,n,make_float3(origin[0],origin[1],origin[2]),
            origins?t.origins.data:nullptr,stride,maxrange,transient?t.transient.data:nullptr,omit);
        check(cudaMemset(t.count_device.data,0,sizeof(int)));
        apply<<<blocks(size),256>>>(t.voxels.data,t.flags.data,size,stamp,t.count_device.data,t.hit_invalidates_free);
        check(cudaGetLastError()); check(cudaMemcpy(&t.count,t.count_device.data,sizeof(int),cudaMemcpyDeviceToHost));
        t.grid=next; t.initialized=true;
        if(t.count>maxvoxels) { t.initialized=false; t.count=0; throw std::runtime_error("Occupancy voxel budget exceeded; map was reset to unknown"); }
        return t.count;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_update(void* p,const float* xyz,int n,const double* origin,const float* origins,
    const double* bounds,double stamp,double ttl,double range,int stride,int maxvoxels) {
    return sn_occupancy_update_masked(p,xyz,n,origin,origins,bounds,stamp,ttl,range,stride,maxvoxels,nullptr,ttl,0);
}
int sn_occupancy_size(void* ptr) { return static_cast<Tree*>(ptr)->count; }
int sn_occupancy_read(void* ptr,float* output,int capacity,double stamp) {
    try {
        auto& t=*static_cast<Tree*>(ptr); Guard guard(t.device);
        if(capacity<t.count) throw std::runtime_error("Voxel buffer is too small"); if(!t.count) return 0;
        t.snapshot.reserve(size_t(t.count)*5); check(cudaMemset(t.count_device.data,0,sizeof(int)));
        int size=t.grid.dim.x*t.grid.dim.y*t.grid.dim.z;
        read_voxels<<<blocks(size),256>>>(t.voxels.data,t.grid,stamp,t.snapshot.data,t.count_device.data);
        check(cudaGetLastError()); check(cudaMemcpy(output,t.snapshot.data,size_t(t.count)*5*sizeof(float),cudaMemcpyDeviceToHost));
        return t.count;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_project(void* ptr,const double* floor,const uint8_t* support,const int32_t* required,
    int height,int width,const double* origin,const double* p,double stamp,const double* endpoints,int n,double* out) {
    try {
        auto& t=*static_cast<Tree*>(ptr); Guard guard(t.device);
        project_resident(t,floor,support,required,height,width,origin,p,stamp,endpoints,n);
        check(cudaGetLastError()); check(cudaMemcpy(out,t.output.data,size_t(height)*width*6*sizeof(double),cudaMemcpyDeviceToHost));
        return t.count;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_project_view(void* ptr,const double* floor,const uint8_t* support,const int32_t* required,
    int height,int width,const double* origin,const double* p,double stamp,const double* endpoints,int n,
    const double* pose,const double* plane,const double* settings,const uint8_t* moving,double* out,int* counts,double* ring) {
    try {
        auto& t=*static_cast<Tree*>(ptr);Guard guard(t.device);
        project_resident(t,floor,support,required,height,width,origin,p,stamp,endpoints,n);
        local_view(t,height,width,origin,p,stamp,pose,plane,settings,moving,n,out,counts,ring);
        return t.count;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_project_people_view(void* ptr,const double* floor,const uint8_t* support,const int32_t* required,
    int height,int width,const double* origin,const double* p,double stamp,const double* endpoints,int n,
    const double* pose,const double* plane,const double* settings,const uint8_t* moving,double* out,int* counts,double* ring) {
    try {
        auto& t=*static_cast<Tree*>(ptr);Guard guard(t.device);
        project_resident(t,floor,support,required,height,width,origin,p,stamp,endpoints,n);
        local_view(t,height,width,origin,p,stamp,pose,plane,settings,moving,n,out,counts,ring,true);
        return t.count;
    } catch(const std::exception& e) { error=e.what();return -1; }
}
int sn_occupancy_view_from_layers(void* ptr,const double* raw,const double* floor,const uint8_t* support,
    int height,int width,const double* origin,const double* p,double stamp,const double* endpoints,int n,
    const double* pose,const double* plane,const double* settings,const uint8_t* moving,double* out,int* counts,double* ring) {
    try {
        auto& t=*static_cast<Tree*>(ptr);Guard guard(t.device);int size=height*width;
        if(size<1 || size>360000)throw std::runtime_error("CUDA local view grid budget exceeded");
        upload(t.output,raw,size_t(size)*6);upload(t.floor,floor,size);upload(t.support,support,size);
        upload(t.endpoints,endpoints,size_t(n)*3);
        local_view(t,height,width,origin,p,stamp,pose,plane,settings,moving,n,out,counts,ring);
        return 0;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_reset_view(void* ptr) { static_cast<Tree*>(ptr)->memory_initialized=false;return 0; }
int sn_occupancy_suspend_free_history(void* ptr,int suspended) {
    static_cast<Tree*>(ptr)->history_suspended=suspended!=0;return 0;
}
int sn_occupancy_project_device_view(void* ptr,const double* floor,const uint8_t* support,const int32_t* required,
    int height,int width,const double* origin,const double* p,double stamp,const double* endpoints,int n,
    const double* pose,const double* plane,const double* settings,const uint8_t* moving,double* device_out,int* counts,double* ring,
    int people) {
    try {
        auto& t=*static_cast<Tree*>(ptr);Guard guard(t.device);
        project_resident(t,floor,support,required,height,width,origin,p,stamp,endpoints,n);
        local_view(t,height,width,origin,p,stamp,pose,plane,settings,moving,n,device_out,counts,ring,people!=0,true);
        return t.count;
    } catch(const std::exception& e) { error=e.what();return -1; }
}
}
