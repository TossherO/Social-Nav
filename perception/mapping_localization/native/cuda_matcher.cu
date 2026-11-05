#include "cuda_matcher.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
using Key=unsigned long long;
constexpr Key empty=~Key(0);
constexpr int bias=1<<20;
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(std::string("FAST-LIO CUDA: ")+cudaGetErrorString(e)); }
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
template<class T> struct Buffer {
    T* data=nullptr;size_t capacity=0;
    ~Buffer(){cudaFree(data);}
    void reserve(size_t n,cudaStream_t stream,size_t keep=0) {
        if(n<=capacity)return;
        size_t cap=std::max(n,capacity+capacity/2+256);T* next;
        check(cudaMalloc(&next,cap*sizeof(T)));
        if(keep)check(cudaMemcpyAsync(next,data,keep*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        check(cudaStreamSynchronize(stream));check(cudaFree(data));data=next;capacity=cap;
    }
};
struct Entry {LioPoint p;uint32_t birth;};
struct Slot {Key key;int begin,end;};
__host__ __device__ Key key(int x,int y,int z) {
    if(x< -bias||x>=bias||y< -bias||y>=bias||z< -bias||z>=bias)return empty;
    return (Key(x+bias)<<42)|(Key(y+bias)<<21)|Key(z+bias);
}
__device__ size_t hash(Key k,size_t mask) {
    k^=k>>30;k*=0xbf58476d1ce4e5b9ULL;k^=k>>27;k*=0x94d049bb133111ebULL;k^=k>>31;return k&mask;
}
__device__ Slot* find(Slot* table,size_t mask,Key k) {
    if(k==empty)return nullptr;
    size_t h=hash(k,mask);
    while(table[h].key!=empty){if(table[h].key==k)return table+h;h=(h+1)&mask;}
    return nullptr;
}
__device__ Slot* claim(Slot* table,size_t mask,Key k) {
    size_t h=hash(k,mask);
    for(;;){Key old=atomicCAS(&table[h].key,empty,k);if(old==empty||old==k)return table+h;h=(h+1)&mask;}
}
__global__ void box_index(const LioBox* boxes,size_t n,Slot* table,size_t mask) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& b=boxes[i];auto* s=claim(table,mask,key(b.cell[0],b.cell[1],b.cell[2]));s->begin=int(i);
}
__device__ bool inside(const LioPoint& p,const LioBox& b) {
    return p.x>=b.low[0]&&p.x<b.high[0]&&p.y>=b.low[1]&&p.y<b.high[1]&&p.z>=b.low[2]&&p.z<b.high[2];
}
__global__ void erase_entries(Entry* points,size_t n,float leaf,Slot* table,size_t mask,
        const LioBox* small,const LioBox* large,size_t nl) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& e=points[i];int x=int(floorf(e.p.x/leaf)),y=int(floorf(e.p.y/leaf)),z=int(floorf(e.p.z/leaf));
    // Adjacent keys cover the reference tree's float-rounded voxel boundaries.
    for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz){
        auto* s=find(table,mask,key(x+dx,y+dy,z+dz));
        if(s&&small[s->begin].sequence>e.birth&&inside(e.p,small[s->begin])){e.birth=UINT32_MAX;return;}
    }
    for(size_t j=0;j<nl;++j)if(large[j].sequence>e.birth&&inside(e.p,large[j])){e.birth=UINT32_MAX;return;}
}
struct Removed {__host__ __device__ bool operator()(const Entry& e)const{return e.birth==UINT32_MAX;}};
__global__ void cell_keys(Entry* points,Key* keys,size_t n,int* invalid,float cell) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& e=points[i];e.birth=0;
    keys[i]=key(int(floorf(e.p.x/cell)),int(floorf(e.p.y/cell)),int(floorf(e.p.z/cell)));
    if(keys[i]==empty)atomicExch(invalid,1);
}
__global__ void map_index(const Key* keys,size_t n,Slot* table,size_t mask) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||i&&keys[i]==keys[i-1])return;
    size_t end=i+1;while(end<n&&keys[end]==keys[i])++end;
    auto* s=claim(table,mask,keys[i]);s->begin=int(i);s->end=int(end);
}
__device__ bool less(float a,float ax,float b,float bx) {return fabsf(a-b)<1e-10f?ax<bx:a<b;}
__device__ float distance(const LioPoint& a,const LioPoint& b) {
    float x=a.x-b.x,y=a.y-b.y,z=a.z-b.z;return x*x+y*y+z*z;
}
__device__ void fit(LioMatch& m,const LioPoint& world,const LioPoint& body,bool enabled,float tolerance,float cutoff,bool weighted) {
    if(!m.selected)return;
    m.selected=0;
    if(!enabled){m.flags|=2;return;}
    double a[3][4]={{0}},scale=0;
    for(int i=0;i<5;++i){double v[3]={m.neighbors[i].x,m.neighbors[i].y,m.neighbors[i].z};
        for(int j=0;j<3;++j){for(int k=0;k<3;++k)a[j][k]+=v[j]*v[k];a[j][3]-=v[j];}}
    for(int j=0;j<3;++j)scale=fmax(scale,fabs(a[j][j]));
    for(int j=0;j<3;++j){int pivot=j;for(int k=j+1;k<3;++k)if(fabs(a[k][j])>fabs(a[pivot][j]))pivot=k;
        if(fabs(a[pivot][j])<scale*1e-7){m.flags|=2;return;}
        for(int k=j;k<4;++k){double t=a[j][k];a[j][k]=a[pivot][k];a[pivot][k]=t;}
        double d=a[j][j];for(int k=j;k<4;++k)a[j][k]/=d;
        for(int i=0;i<3;++i)if(i!=j){double d=a[i][j];for(int k=j;k<4;++k)a[i][k]-=d*a[j][k];}}
    // Reference stores the QR solution and its normalization in float.
    float v[3]={float(a[0][3]),float(a[1][3]),float(a[2][3])};
    float length=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if(!isfinite(length)||length<1e-12f){m.flags|=2;return;}
    float n[4]={v[0]/length,v[1]/length,v[2]/length,1.f/length};
    for(int i=0;i<5;++i){const auto& p=m.neighbors[i];float d=fabsf(n[0]*p.x+n[1]*p.y+n[2]*p.z+n[3]);
        if(fabsf(d-tolerance)<1e-4f){m.flags|=2;return;}if(d>tolerance)return;}
    float pd=n[0]*world.x+n[1]*world.y+n[2]*world.z+n[3];
    double range=sqrt(sqrt(double(body.x)*body.x+double(body.y)*body.y+double(body.z)*body.z));
    if(weighted)range=sqrtf(sqrtf(body.x*body.x+body.y*body.y+body.z*body.z));
    float score=1.-.9*fabs(double(pd))/range;
    if(!isfinite(score)||fabsf(score-cutoff)<1e-5f){m.flags|=2;return;}
    if(score>cutoff){float weight=weighted?score:1.f;for(int j=0;j<3;++j)m.normal[j]=weight*n[j];m.normal[3]=weight*pd;m.selected=1;}
}
__global__ void query(const Entry* map,size_t map_count,Slot* table,size_t mask,const LioPoint* world,
        const LioPoint* body,LioMatch* cache,size_t n,bool search,int fit_mode,float limit,float cell) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& out=cache[i];out.flags=0;const auto p=world[i];
    if(search){
        if(map_count<5){out.flags=1;out.count=0;out.selected=0;return;}
        LioPoint near[6]{};float dist[6];for(int j=0;j<6;++j)dist[j]=limit+1e-5f;
        int count=0,cx=int(floorf(p.x/cell)),cy=int(floorf(p.y/cell)),cz=int(floorf(p.z/cell));
        for(int radius=0;radius<=int(ceilf(sqrtf(limit)/cell));++radius){
            for(int dx=-radius;dx<=radius;++dx)for(int dy=-radius;dy<=radius;++dy)for(int dz=-radius;dz<=radius;++dz){
                if(max(abs(dx),max(abs(dy),abs(dz)))!=radius)continue;
                double x=double(cx+dx)*cell,y=double(cy+dy)*cell,z=double(cz+dz)*cell;
                double ax=fmax(fmax(x-p.x,0.),p.x-(x+cell)),ay=fmax(fmax(y-p.y,0.),p.y-(y+cell)),az=fmax(fmax(z-p.z,0.),p.z-(z+cell));
                if(ax*ax+ay*ay+az*az>dist[5]+1e-5)continue;
                auto* s=find(table,mask,key(cx+dx,cy+dy,cz+dz));if(!s)continue;
                for(int j=s->begin;j<s->end;++j){auto q=map[j].p;float d=distance(p,q);if(d>limit+1e-5f)continue;
                    int pos=min(count,5);if(pos==5&&!less(d,q.x,dist[5],near[5].x))continue;
                    while(pos>0&&less(d,q.x,dist[pos-1],near[pos-1].x)){if(pos<6){dist[pos]=dist[pos-1];near[pos]=near[pos-1];}--pos;}
                    dist[pos]=d;near[pos]=q;count=min(count+1,6);}
            }
            double edge=fmin(fmin(p.x-double(cx-radius)*cell,double(cx+radius+1)*cell-p.x),fmin(fmin(p.y-double(cy-radius)*cell,double(cy+radius+1)*cell-p.y),fmin(p.z-double(cz-radius)*cell,double(cz+radius+1)*cell-p.z)));
            if(count>=5&&edge*edge>dist[4]+1e-5)break;
            if(edge*edge>limit+1e-5)break;
        }
        out.count=min(count,5);out.selected=count>=5&&dist[4]<=limit;
        for(int j=0;j<out.count;++j){out.neighbors[j]=near[j];out.distance[j]=dist[j];}
        // Sparse/out-of-radius queries still need the CPU's unbounded neighbors
        // for its original map insertion policy. Ties retain ikd-tree ordering.
        if(count<5||fabsf(dist[4]-limit)<1e-5f||count>5&&fabsf(dist[5]-dist[4])<1e-6f*fmaxf(1.f,dist[4]))out.flags|=1;
        for(int j=1;j<out.count;++j)if(fabsf(dist[j]-dist[j-1])<1e-7f)out.flags|=1;
    }
    if(!(out.flags&1))fit(out,p,fit_mode==2?p:body[i],fit_mode!=0,fit_mode==2?.2f:.1f,fit_mode==2?.1f:.9f,fit_mode==2);
}
__global__ void corrections(LioMatch* cache,const int* ids,const LioMatch* values,size_t n) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)cache[ids[i]]=values[i];
}
__global__ void selections(LioMatch* cache,const uint8_t* selected,size_t n) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)cache[i].selected=selected[i];
}
size_t slots(size_t n) {size_t s=256;while(s<n*2)s*=2;return s;}

__device__ void matvec(const double* a,const double* v,double* out) {
    for(int i=0;i<3;++i)out[i]=a[3*i]*v[0]+a[3*i+1]*v[1]+a[3*i+2]*v[2];
}
__device__ void compensate(LioTimedPoint& p,const LioImuPose& head,const LioImuPose& tail,const LioDeskewState& state) {
    double dt=double(p.offset_ms)/1000.-head.offset;
    const auto* w=tail.angular_velocity;double length=sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
    double exponential[9]={1,0,0,0,1,0,0,0,1};
    if(length>1e-7){double x=w[0]/length,y=w[1]/length,z=w[2]/length;
        double k[9]={0,-z,y,z,0,-x,-y,x,0},s=sin(length*dt),c=1-cos(length*dt);
        for(int r=0;r<3;++r)for(int col=0;col<3;++col){double square=0;
            for(int j=0;j<3;++j)square+=k[r*3+j]*k[j*3+col];exponential[r*3+col]+=(s*k[r*3+col]+c*square);}}
    double rotation[9];for(int r=0;r<3;++r)for(int col=0;col<3;++col)
        rotation[r*3+col]=head.rotation[r*3]*exponential[col]+head.rotation[r*3+1]*exponential[3+col]+head.rotation[r*3+2]*exponential[6+col];
    double body[3]={p.x,p.y,p.z},inertial[3],world[3],local[3],result[3];
    matvec(state.extrinsic,body,inertial);for(int j=0;j<3;++j)inertial[j]+=state.translation[j];
    matvec(rotation,inertial,world);
    for(int j=0;j<3;++j)world[j]+=head.position[j]+head.velocity[j]*dt+.5*tail.acceleration[j]*dt*dt-state.position[j];
    matvec(state.inverse_rotation,world,local);for(int j=0;j<3;++j)local[j]-=state.translation[j];
    matvec(state.inverse_extrinsic,local,result);p.x=result[0];p.y=result[1];p.z=result[2];
}
__global__ void deskew_points(LioTimedPoint* points,size_t n,const LioImuPose* poses,int np,LioDeskewState state) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||np<2)return;
    auto& p=points[i];double t=double(p.offset_ms)/1000.;
    // Entry zero is the scan anchor, not necessarily earlier than the first
    // carried IMU sample. Preserve the CPU's strict interval and first-point
    // traversal semantics rather than assuming the entire table is sorted.
    if(i==0){for(int h=np-2;h>=0;--h)if(t>poses[h].offset)compensate(p,poses[h],poses[h+1],state);return;}
    int lo=1,hi=np-1;
    while(lo<hi){int mid=(lo+hi)/2;if(poses[mid].offset<t)lo=mid+1;else hi=mid;}
    int idx=lo-1;if(idx==0&&t<=poses[0].offset)return;
    compensate(p,poses[idx],poses[idx+1],state);
}
__global__ void point_jacobians(const LioPoint* body,const LioMatch* matches,const LioPoint* normals,
        size_t n,const double* state,bool estimate,double* jacobian) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double* h=jacobian+i*13;for(int j=0;j<13;++j)h[j]=0;if(!matches[i].selected)return;
    auto b=body[i];LioPoint norm=normals?normals[i]:LioPoint{matches[i].normal[0],matches[i].normal[1],matches[i].normal[2],matches[i].normal[3]};
    double p[3]={b.x,b.y,b.z},normal[3]={norm.x,norm.y,norm.z},inertial[3],c[3];
    matvec(state,p,inertial);for(int j=0;j<3;++j)inertial[j]+=state[9+j];matvec(state+12,normal,c);
    h[0]=normal[0];h[1]=normal[1];h[2]=normal[2];
    h[3]=inertial[1]*c[2]-inertial[2]*c[1];h[4]=inertial[2]*c[0]-inertial[0]*c[2];h[5]=inertial[0]*c[1]-inertial[1]*c[0];
    if(estimate){double v[3];for(int j=0;j<3;++j)v[j]=state[j]*c[0]+state[3+j]*c[1]+state[6+j]*c[2];
        h[6]=p[1]*v[2]-p[2]*v[1];h[7]=p[2]*v[0]-p[0]*v[2];h[8]=p[0]*v[1]-p[1]*v[0];
        for(int j=0;j<3;++j)h[9+j]=c[j];}
    h[12]=-double(norm.intensity);
}
__global__ void normal_partials(const double* jacobian,size_t n,int chunks,double* partials) {
    int field=blockIdx.y,row=0,col=12;
    if(field<78){int index=field;while(index>=12-row){index-=12-row;++row;}col=row+index;}
    else row=field-78;
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;double v=i<n?jacobian[13*i+row]*jacobian[13*i+col]:0;
    __shared__ double values[128];values[threadIdx.x]=v;__syncthreads();
    for(int offset=64;offset;offset/=2){if(threadIdx.x<offset)values[threadIdx.x]+=values[threadIdx.x+offset];__syncthreads();}
    if(!threadIdx.x)partials[field*chunks+blockIdx.x]=values[0];
}
__global__ void finish_normals(const double* partials,int chunks,double* output) {
    int field=threadIdx.x;if(field>=90)return;double value=0;for(int j=0;j<chunks;++j)value+=partials[field*chunks+j];output[field]=value;
}
}

struct LioCuda::Impl {
    int device;float leaf,cell=1.f;bool plane_fit,surface_fit=false;cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
    Buffer<Entry> points;Buffer<Key> keys;Buffer<Slot> index,box_table;
    Buffer<LioBox> small,large;Buffer<LioPoint> body,world;
    Buffer<LioMatch> cache,fix_values;Buffer<int> fix_ids,invalid;
    Buffer<uint8_t> selected;
    Buffer<LioTimedPoint> timed;Buffer<LioImuPose> imu_poses;
    Buffer<LioPoint> normals;Buffer<double> linear_state,jacobian,partials,normal_output;
    std::vector<Entry> additions;std::map<Key,LioBox> replacements;std::vector<LioBox> erasures;
    size_t count=0,index_mask=0;uint32_t sequence=0;LioCudaStats stats;
    Impl(int d,float l,bool fit):device(d),leaf(l),plane_fit(fit) {
        check(cudaSetDevice(device));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaEventCreate(&begin));check(cudaEventCreate(&end));invalid.reserve(1,stream);
    }
    ~Impl(){cudaSetDevice(device);cudaStreamSynchronize(stream);cudaEventDestroy(begin);cudaEventDestroy(end);cudaStreamDestroy(stream);}
    template<class T>void upload(T* to,const T* from,size_t n) {
        if(n){check(cudaMemcpyAsync(to,from,n*sizeof(T),cudaMemcpyHostToDevice,stream));stats.h2d_bytes+=n*sizeof(T);}
    }
    void rebuild() {
        stats.points=count;
        if(!count)return;
        keys.reserve(count,stream);size_t cap=slots(count);index.reserve(cap,stream);index_mask=cap-1;
        check(cudaMemsetAsync(index.data,255,cap*sizeof(Slot),stream));check(cudaMemsetAsync(invalid.data,0,sizeof(int),stream));
        cell_keys<<<(count+255)/256,256,0,stream>>>(points.data,keys.data,count,invalid.data,cell);
        auto policy=thrust::cuda::par.on(stream);
        thrust::sort_by_key(policy,thrust::device_pointer_cast(keys.data),thrust::device_pointer_cast(keys.data+count),thrust::device_pointer_cast(points.data));
        map_index<<<(count+255)/256,256,0,stream>>>(keys.data,count,index.data,index_mask);
        int bad;check(cudaMemcpyAsync(&bad,invalid.data,sizeof(int),cudaMemcpyDeviceToHost,stream));stats.d2h_bytes+=sizeof(int);
        check(cudaStreamSynchronize(stream));if(bad)throw std::runtime_error("FAST-LIO CUDA map exceeds spatial index range");
        stats.points=count;
    }
};
LioCuda::LioCuda(int d,float l,bool fit):p_(new Impl(d,l,fit)){}
LioCuda::~LioCuda()=default;
void LioCuda::frame_begin(){check(cudaSetDevice(p_->device));p_->stats={};p_->stats.points=p_->count;}
void LioCuda::search_resolution(float meters){
    if(!std::isfinite(meters)||meters<.125f||meters>1.f||p_->count)throw std::invalid_argument("Set CUDA search resolution in [0.125,1] before loading the map");
    p_->cell=meters;
}
void LioCuda::surface_fitting(bool enabled){p_->surface_fit=enabled;}
void LioCuda::reset(const LioPoint* values,size_t n) {
    auto& p=*p_;auto start=Clock::now();p.additions.clear();p.replacements.clear();p.erasures.clear();p.sequence=0;
    std::vector<Entry> entries;entries.reserve(n);for(size_t i=0;i<n;++i)entries.push_back({values[i],0});
    p.points.reserve(n,p.stream);p.upload(p.points.data,entries.data(),n);p.count=n;p.rebuild();p.stats.map_ms+=ms(start);
}
void LioCuda::insert(const LioPoint& point,const LioBox* replaced) {
    auto& p=*p_;
    if(replaced){auto b=*replaced;b.sequence=++p.sequence;
        for(int j=0;j<3;++j)b.cell[j]=int(std::llround(double(b.low[j])/p.leaf));
        auto k=key(b.cell[0],b.cell[1],b.cell[2]);if(k==empty)throw std::runtime_error("FAST-LIO CUDA voxel index range exceeded");
        p.replacements[k]=b;}
    p.additions.push_back({point,++p.sequence});
}
void LioCuda::erase(const LioBox& value){auto b=value;b.sequence=++p_->sequence;p_->erasures.push_back(b);}
void LioCuda::sync_map() {
    auto& p=*p_;if(!p.sequence)return;auto start=Clock::now();
    p.points.reserve(p.count+p.additions.size(),p.stream,p.count);p.upload(p.points.data+p.count,p.additions.data(),p.additions.size());p.count+=p.additions.size();
    std::vector<LioBox> boxes;boxes.reserve(p.replacements.size());for(auto& item:p.replacements)boxes.push_back(item.second);
    p.small.reserve(boxes.size(),p.stream);p.large.reserve(p.erasures.size(),p.stream);
    p.upload(p.small.data,boxes.data(),boxes.size());p.upload(p.large.data,p.erasures.data(),p.erasures.size());
    size_t cap=slots(boxes.size());p.box_table.reserve(cap,p.stream);check(cudaMemsetAsync(p.box_table.data,255,cap*sizeof(Slot),p.stream));
    if(!boxes.empty())box_index<<<(boxes.size()+255)/256,256,0,p.stream>>>(p.small.data,boxes.size(),p.box_table.data,cap-1);
    if(p.count){erase_entries<<<(p.count+255)/256,256,0,p.stream>>>(p.points.data,p.count,p.leaf,p.box_table.data,cap-1,p.small.data,p.large.data,p.erasures.size());
        auto ptr=thrust::device_pointer_cast(p.points.data);p.count=thrust::remove_if(thrust::cuda::par.on(p.stream),ptr,ptr+p.count,Removed())-ptr;}
    p.additions.clear();p.replacements.clear();p.erasures.clear();p.sequence=0;p.rebuild();p.stats.points=p.count;p.stats.map_ms+=ms(start);
}
void LioCuda::begin_scan(const LioPoint* body,size_t n) {
    auto& p=*p_;p.body.reserve(n,p.stream);p.world.reserve(n,p.stream);p.cache.reserve(n,p.stream);
    p.upload(p.body.data,body,n);check(cudaMemsetAsync(p.cache.data,0,n*sizeof(LioMatch),p.stream));
}
void LioCuda::match(const LioPoint* world,size_t n,bool search,LioMatch* out) {
    auto& p=*p_;auto start=Clock::now();if(!n)return;
    p.upload(p.world.data,world,n);check(cudaEventRecord(p.begin,p.stream));
    query<<<(n+127)/128,128,0,p.stream>>>(p.points.data,p.count,p.index.data,p.index_mask,p.world.data,p.body.data,p.cache.data,n,search,p.plane_fit,5.f,p.cell);
    check(cudaEventRecord(p.end,p.stream));
    check(cudaMemcpyAsync(out,p.cache.data,n*sizeof(LioMatch),cudaMemcpyDeviceToHost,p.stream));p.stats.d2h_bytes+=n*sizeof(LioMatch);
    check(cudaStreamSynchronize(p.stream));float elapsed;check(cudaEventElapsedTime(&elapsed,p.begin,p.end));
    p.stats.query_device_ms+=elapsed;p.stats.match_ms+=ms(start);
}
void LioCuda::search(const LioPoint* world,size_t n,float limit,LioMatch* out) {
    auto& p=*p_;if(!n)return;
    if(!std::isfinite(limit)||limit<=0||limit>5)throw std::invalid_argument("CUDA search requires 0 < squared distance <= 5");
    check(cudaSetDevice(p.device));auto start=Clock::now();p.world.reserve(n,p.stream);p.cache.reserve(n,p.stream);
    p.upload(p.world.data,world,n);check(cudaEventRecord(p.begin,p.stream));
    query<<<(n+127)/128,128,0,p.stream>>>(p.points.data,p.count,p.index.data,p.index_mask,p.world.data,p.world.data,p.cache.data,n,true,p.surface_fit?2:0,limit,p.cell);
    check(cudaEventRecord(p.end,p.stream));
    check(cudaMemcpyAsync(out,p.cache.data,n*sizeof(LioMatch),cudaMemcpyDeviceToHost,p.stream));p.stats.d2h_bytes+=n*sizeof(LioMatch);
    check(cudaStreamSynchronize(p.stream));float elapsed;check(cudaEventElapsedTime(&elapsed,p.begin,p.end));
    p.stats.query_device_ms+=elapsed;p.stats.match_ms+=ms(start);
}
void LioCuda::correct(const int* ids,const LioMatch* values,size_t n) {
    auto& p=*p_;if(!n)return;p.fix_ids.reserve(n,p.stream);p.fix_values.reserve(n,p.stream);
    p.upload(p.fix_ids.data,ids,n);p.upload(p.fix_values.data,values,n);
    corrections<<<(n+255)/256,256,0,p.stream>>>(p.cache.data,p.fix_ids.data,p.fix_values.data,n);
    check(cudaStreamSynchronize(p.stream));
}
void LioCuda::select(const uint8_t* selected,size_t n) {
    auto& p=*p_;if(!n)return;p.selected.reserve(n,p.stream);p.upload(p.selected.data,selected,n);
    selections<<<(n+255)/256,256,0,p.stream>>>(p.cache.data,p.selected.data,n);
    check(cudaStreamSynchronize(p.stream));
}
void LioCuda::deskew(LioTimedPoint* points,size_t n,const LioImuPose* poses,size_t np,const LioDeskewState& state) {
    auto& p=*p_;if(!n||np<2)return;check(cudaSetDevice(p.device));
    p.timed.reserve(n,p.stream);p.imu_poses.reserve(np,p.stream);p.upload(p.timed.data,points,n);p.upload(p.imu_poses.data,poses,np);
    check(cudaEventRecord(p.begin,p.stream));deskew_points<<<(n+127)/128,128,0,p.stream>>>(p.timed.data,n,p.imu_poses.data,int(np),state);
    check(cudaEventRecord(p.end,p.stream));check(cudaMemcpyAsync(points,p.timed.data,n*sizeof(LioTimedPoint),cudaMemcpyDeviceToHost,p.stream));
    p.stats.d2h_bytes+=n*sizeof(LioTimedPoint);check(cudaStreamSynchronize(p.stream));float elapsed;check(cudaEventElapsedTime(&elapsed,p.begin,p.end));p.stats.deskew_device_ms+=elapsed;
}
void LioCuda::linearize(const LioPoint* normals,size_t n,const double* state,bool estimate,double* hth,double* htr) {
    auto& p=*p_;if(!n)throw std::invalid_argument("Cannot linearize an empty scan");
    check(cudaSetDevice(p.device));int chunks=int((n+127)/128);
    p.linear_state.reserve(21,p.stream);p.jacobian.reserve(n*13,p.stream);p.partials.reserve(chunks*90,p.stream);p.normal_output.reserve(90,p.stream);
    p.upload(p.linear_state.data,state,21);
    if(normals){p.normals.reserve(n,p.stream);p.upload(p.normals.data,normals,n);}
    check(cudaEventRecord(p.begin,p.stream));
    point_jacobians<<<chunks,128,0,p.stream>>>(p.body.data,p.cache.data,normals?p.normals.data:nullptr,n,p.linear_state.data,estimate,p.jacobian.data);
    normal_partials<<<dim3(chunks,90),128,0,p.stream>>>(p.jacobian.data,n,chunks,p.partials.data);
    finish_normals<<<1,128,0,p.stream>>>(p.partials.data,chunks,p.normal_output.data);
    check(cudaEventRecord(p.end,p.stream));double output[90];check(cudaMemcpyAsync(output,p.normal_output.data,sizeof(output),cudaMemcpyDeviceToHost,p.stream));p.stats.d2h_bytes+=sizeof(output);
    check(cudaStreamSynchronize(p.stream));float elapsed;check(cudaEventElapsedTime(&elapsed,p.begin,p.end));p.stats.linearize_device_ms+=elapsed;
    int field=0;for(int r=0;r<12;++r)for(int c=r;c<12;++c){hth[12*r+c]=hth[12*c+r]=output[field++];}
    for(int j=0;j<12;++j)htr[j]=output[78+j];
}
LioCudaStats LioCuda::stats()const{return p_->stats;}
