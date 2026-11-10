// A ROS-free C ABI keeps the geometry library usable from system and Conda Python.
// OctoMap source/licenses remain in the installed or private dependency prefix.
#include <octomap/OcTree.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
thread_local std::string error;
using Key = uint64_t;
Key pack(const octomap::OcTreeKey& k) { return (Key(k[0])<<32)|(Key(k[1])<<16)|Key(k[2]); }
octomap::OcTreeKey unpack(Key k) { return octomap::OcTreeKey((k>>32)&65535,(k>>16)&65535,k&65535); }
struct Voxel { float log_odds=0; double stamp=0, hit_stamp=0, ordinary_stamp=-INFINITY; bool transient=false; };
struct Votes {
    int hits=0, free=0;
    int64_t hit_bucket=-1, free_bucket=-1;
};
struct Tree {
    octomap::OcTree octree;
    // Use OctoMap's key/ray/sensor model with directly erasable bounded evidence.
    // This avoids old OctoMap's partial-node deletion path and stale evidence.
    std::unordered_map<Key,Voxel> seen;
    // Allocated only by offline static fusion; online CPU/CUDA semantics stay unchanged.
    std::unordered_map<Key,Votes> votes;
    double vote_interval=0, ray_end_margin=0, obstacle_ttl=INFINITY;
    bool hit_invalidates_free=false;
    bool static_retention=false, static_initialized=false;
    double static_first=0, static_last=0, static_box[6]{};
    std::vector<Key> outside_hits;
    explicit Tree(double resolution): octree(resolution) {}
};
bool inside(const octomap::point3d& p, const double* box) {
    return p.x()>=box[0] && p.y()>=box[1] && p.z()>=box[2] &&
           p.x()<box[3] && p.y()<box[4] && p.z()<box[5];
}
bool rayMayTouchBox(const octomap::point3d& a, const octomap::point3d& b, const double* box, double padding) {
    // Conservative slab intersection: voxel centers can be inside even when
    // the geometric ray is just outside. One full voxel of padding covers it.
    const double start[3]={a.x(),a.y(),a.z()}, end[3]={b.x(),b.y(),b.z()};
    double lo=0, hi=1;
    for(int axis=0;axis<3;++axis) {
        const double low=box[axis]-padding, high=box[axis+3]+padding, delta=end[axis]-start[axis];
        if(std::abs(delta)<1e-12) {
            if(start[axis]<low || start[axis]>high) return false;
        } else {
            double t0=(low-start[axis])/delta, t1=(high-start[axis])/delta;
            if(t0>t1) std::swap(t0,t1);
            lo=std::max(lo,t0); hi=std::min(hi,t1);
            if(lo>hi) return false;
        }
    }
    return true;
}
}
extern "C" {
const char* sn_occupancy_error() { return error.c_str(); }
void* sn_occupancy_new(double resolution) {
    try { return new Tree(resolution); } catch(const std::exception& e) { error=e.what(); return nullptr; }
}
void sn_occupancy_free(void* ptr) { delete static_cast<Tree*>(ptr); }
int sn_occupancy_obstacle_lifetime(void* ptr, double ttl) {
    if(!std::isfinite(ttl) || ttl<=0) { error="Invalid obstacle lifetime"; return -1; }
    static_cast<Tree*>(ptr)->obstacle_ttl=ttl; return 0;
}
int sn_occupancy_hit_invalidates_free(void* ptr, int enabled) {
    static_cast<Tree*>(ptr)->hit_invalidates_free=enabled!=0; return 0;
}
int sn_occupancy_enable_static_retention(void* ptr) {
    auto& t=*static_cast<Tree*>(ptr);
    if(!t.seen.empty()) { error="Static retention must be enabled before observations"; return -1; }
    t.static_retention=true; return 0;
}
int sn_occupancy_enable_static_review(void* ptr, double interval, double margin) {
    try {
        auto& t=*static_cast<Tree*>(ptr);
        if(!t.seen.empty() || !std::isfinite(interval) || interval<=0 || !std::isfinite(margin) || margin<0)
            throw std::runtime_error("Static review must be enabled before observations with valid interval/margin");
        t.vote_interval=interval; t.ray_end_margin=margin;
        return 0;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_static_votes(void* ptr, const float* xyz, int n, int32_t* output) {
    try {
        auto& t=*static_cast<Tree*>(ptr);
        if(t.vote_interval<=0) throw std::runtime_error("Static evidence collection was not enabled");
        for(int i=0;i<n;++i) {
            output[2*i]=output[2*i+1]=0;
            octomap::OcTreeKey key;
            if(!t.octree.coordToKeyChecked(octomap::point3d(xyz[3*i],xyz[3*i+1],xyz[3*i+2]),key)) continue;
            const auto found=t.votes.find(pack(key));
            if(found!=t.votes.end()) {
                output[2*i]=found->second.hits; output[2*i+1]=found->second.free;
            }
        }
        return n;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_update_masked(void* ptr, const float* xyz, int n, const double* origin, const float* ray_origins,
    const double* box, double stamp, double ttl, double maxrange, int ray_stride, int maxvoxels,
    const uint8_t* transient, double transient_ttl, int omit_transient) {
    try {
        auto& t=*static_cast<Tree*>(ptr);
        if(t.static_retention) {
            if(!t.static_initialized) {
                std::copy(box,box+6,t.static_box);
                t.static_first=t.static_last=stamp; t.static_initialized=true;
            }
            if(stamp<t.static_last || !std::equal(box,box+6,t.static_box) ||
                    stamp-t.static_first>ttl || stamp-t.static_first>t.obstacle_ttl ||
                    (transient && !omit_transient))
                throw std::runtime_error("Static retention requires fixed bounds and full-survey lifetimes");
            t.static_last=stamp;
            // A hit point inside the box can quantize to a voxel center outside
            // it. The ordinary path removes those exceptional keys next scan.
            for(const auto k:t.outside_hits) { t.seen.erase(k); t.votes.erase(k); }
            t.outside_hits.clear();
        } else for(auto it=t.seen.begin();it!=t.seen.end();) {
            const auto key=unpack(it->first);
            const auto& v=it->second;
            const double lifetime=v.transient?std::min(t.obstacle_ttl,transient_ttl):t.obstacle_ttl;
            if(stamp-v.stamp>ttl || (v.log_odds>0 && stamp-v.hit_stamp>lifetime && stamp-v.ordinary_stamp>t.obstacle_ttl) || !inside(t.octree.keyToCoord(key),box)) {
                if(t.vote_interval>0) t.votes.erase(it->first);
                it=t.seen.erase(it);
            } else ++it;
        }
        std::unordered_set<Key> hits, moving_hits, misses;
        std::unordered_set<Key> review_free, physical_hits;
        octomap::KeyRay ray;
        for(int i=0;i<n;++i) {
            octomap::point3d sensor=ray_origins ? octomap::point3d(ray_origins[3*i],ray_origins[3*i+1],ray_origins[3*i+2])
                                               : octomap::point3d(origin[0],origin[1],origin[2]);
            octomap::point3d p(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);
            const double length=(p-sensor).norm();
            if(!std::isfinite(length) || length<.01) continue;
            octomap::OcTreeKey key;
            if(length<=maxrange && inside(p,box) && t.octree.coordToKeyChecked(p,key)) {
                if(t.vote_interval>0) physical_hits.insert(pack(key));
                if(transient && transient[i]) {
                    if(!omit_transient) moving_hits.insert(pack(key));
                } else hits.insert(pack(key));
            }
            if(ray_stride<=0 || i%ray_stride) continue;
            if(length>maxrange) p=sensor+(p-sensor)*(maxrange/length);
            if(t.static_retention && !rayMayTouchBox(sensor,p,box,t.octree.getResolution()+1e-6)) continue;
            const auto direction=p-sensor;
            const double used_length=direction.norm();
            const double limit=(used_length-t.ray_end_margin)*used_length;
            if(t.octree.computeRayKeys(sensor,p,ray))
                for(const auto& k:ray) {
                    const auto center=t.octree.keyToCoord(k);
                    if(!inside(center,box)) continue;
                    misses.insert(pack(k));
                    // Only an actual visible ray prefix is negative evidence.
                    // Protect the return neighbourhood against surface/boundary noise.
                    if(t.vote_interval>0 && used_length>t.ray_end_margin &&
                            (center-sensor).dot(direction)<limit) review_free.insert(pack(k));
                }
        }
        auto update=[&](Key key,bool hit,bool moving=false) {
            auto& v=t.seen[key];
            // Opt-in local memory semantics: old negative evidence predates this
            // physical return. Subsequent misses must establish free space again.
            if(hit && t.hit_invalidates_free) v.log_odds=std::max(0.f,v.log_odds);
            v.log_odds=std::max(t.octree.getClampingThresMinLog(),
                std::min(t.octree.getClampingThresMaxLog(),
                    v.log_odds+(hit?t.octree.getProbHitLog():t.octree.getProbMissLog())));
            v.stamp=stamp;
            if(hit) { v.transient=moving; v.hit_stamp=stamp;if(!moving)v.ordinary_stamp=stamp; }
        };
        for(const auto k:misses) if(!hits.count(k) && !moving_hits.count(k)) {
            update(k,false);
        }
        for(const auto k:moving_hits) if(!hits.count(k)) update(k,true,true);
        for(const auto k:hits) update(k,true);
        if(t.static_retention) {
            for(const auto k:hits)
                if(!inside(t.octree.keyToCoord(unpack(k)),box)) t.outside_hits.push_back(k);
            for(const auto k:moving_hits)
                if(!inside(t.octree.keyToCoord(unpack(k)),box)) t.outside_hits.push_back(k);
        }
        if(t.vote_interval>0) {
            const int64_t bucket=std::floor(stamp/t.vote_interval);
            for(const auto k:review_free) if(!physical_hits.count(k)) {
                auto& v=t.votes[k];
                if(v.free_bucket!=bucket) { if(v.free<INT32_MAX) ++v.free; v.free_bucket=bucket; }
            }
            for(const auto k:hits) {
                auto& v=t.votes[k];
                if(v.hit_bucket!=bucket) { if(v.hits<INT32_MAX) ++v.hits; v.hit_bucket=bucket; }
            }
        }
        if(t.seen.size()>static_cast<size_t>(maxvoxels)) {
            t.seen.clear();
            t.votes.clear();
            t.outside_hits.clear();
            throw std::runtime_error("Occupancy voxel budget exceeded; map was reset to unknown");
        }
        return static_cast<int>(t.seen.size());
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
int sn_occupancy_update(void* ptr, const float* xyz, int n, const double* origin, const float* ray_origins,
    const double* box, double stamp, double ttl, double maxrange, int ray_stride, int maxvoxels) {
    return sn_occupancy_update_masked(ptr,xyz,n,origin,ray_origins,box,stamp,ttl,maxrange,ray_stride,maxvoxels,
                                      nullptr,ttl,0);
}
int sn_occupancy_size(void* ptr) { return static_cast<Tree*>(ptr)->seen.size(); }
int sn_occupancy_read(void* ptr, float* output, int capacity, double stamp) {
    try {
        auto& t=*static_cast<Tree*>(ptr); int i=0;
        if(capacity<static_cast<int>(t.seen.size())) throw std::runtime_error("Voxel buffer is too small");
        for(const auto& item:t.seen) {
            const auto key=unpack(item.first); const auto p=t.octree.keyToCoord(key);
            output[5*i]=p.x(); output[5*i+1]=p.y(); output[5*i+2]=p.z();
            output[5*i+3]=1.f-1.f/(1.f+std::exp(item.second.log_odds));
            output[5*i+4]=stamp-item.second.stamp; ++i;
        }
        return i;
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
// Exclusive person columns: ordinary occupied evidence always wins.
int sn_occupancy_person_columns(void* ptr,const double* floor,int height,int width,
    const double* origin,const double* p,double stamp,const double* endpoints,const uint8_t* moving,int n,uint8_t* out) {
    try {
        auto& t=*static_cast<Tree*>(ptr);std::vector<uint8_t> tags(height*width,0);
        auto mark=[&](double x,double y,double z,bool person) {
            int ix=std::floor((x-origin[0])/p[0]),iy=std::floor((y-origin[1])/p[0]);
            if(ix<0 || ix>=width || iy<0 || iy>=height)return;
            int i=iy*width+ix;double h=z-floor[i];
            if(std::isfinite(h) && h>=p[1] && h<=p[2])tags[i]|=person?1:2;
        };
        for(const auto& item:t.seen) {
            float probability=1.f-1.f/(1.f+std::exp(item.second.log_odds));
            if(probability<float(p[3]))continue;
            auto xyz=t.octree.keyToCoord(unpack(item.first));
            mark(xyz.x(),xyz.y(),xyz.z(),item.second.transient && stamp-item.second.ordinary_stamp>t.obstacle_ttl);
        }
        for(int j=0;j<n;++j)mark(endpoints[3*j],endpoints[3*j+1],endpoints[3*j+2],moving && moving[j]);
        for(int i=0;i<height*width;++i)out[i]=tags[i]==1;
        return 0;
    } catch(const std::exception& e) { error=e.what();return -1; }
}
// Project directly from resident evidence. Channel-major output:
// occupancy, observed, last_observed, coverage, obstacle_min_z, obstacle_max_z.
int sn_occupancy_project(void* ptr, const double* floor, const uint8_t* support,
    const int32_t* required, int height, int width, const double* origin,
    const double* p, double stamp, const double* endpoints, int n, double* out) {
    try {
        auto& t=*static_cast<Tree*>(ptr);
        const int size=height*width, bins=std::ceil((p[2]-p[1])/p[0])+1;
        std::vector<uint8_t> free(size*bins,0);
        for(int i=0;i<size;++i) {
            out[i]=-1; out[size+i]=0; out[2*size+i]=-INFINITY;
            out[3*size+i]=0; out[4*size+i]=INFINITY; out[5*size+i]=-INFINITY;
        }
        auto cell=[&](double x,double y) {
            const int ix=std::floor((x-origin[0])/p[0]), iy=std::floor((y-origin[1])/p[0]);
            return ix>=0 && ix<width && iy>=0 && iy<height ? iy*width+ix : -1;
        };
        auto hit=[&](int i,double h) {
            out[i]=100; out[4*size+i]=std::min(out[4*size+i],h); out[5*size+i]=std::max(out[5*size+i],h);
        };
        for(const auto& item:t.seen) {
            const auto xyz=t.octree.keyToCoord(unpack(item.first));
            const int i=cell(xyz.x(),xyz.y()); if(i<0) continue;
            out[size+i]=1;
            const float age=stamp-item.second.stamp;
            out[2*size+i]=std::max(out[2*size+i],stamp-double(age));
            const double h=p[6] ? double(float(xyz.z()-float(floor[i]))) : double(xyz.z())-floor[i];
            if(!std::isfinite(floor[i])) continue;
            const int first=std::ceil((floor[i]+p[1])/p[0]-.5-1e-5);
            const int last=std::floor((floor[i]+p[2])/p[0]-.5+1e-5);
            const int key=std::llround(double(xyz.z())/p[0]-.5);
            if(key<first || key>last) continue;
            const float probability=1.f-1.f/(1.f+std::exp(item.second.log_odds));
            if(probability>=float(p[3])) hit(i,h);
            if(probability<=float(p[4])) {
                free[i*bins+key-first]=1;
            }
        }
        for(int i=0;i<size;++i) {
            int count=0; for(int z=0;z<bins;++z) count+=free[i*bins+z];
            out[3*size+i]=float(std::min(1.,double(count)/required[i]));
            if(out[3*size+i]>=float(p[5]) && support[i] && out[i]!=100) out[i]=0;
        }
        for(int j=0;j<n;++j) {
            int i=cell(endpoints[3*j],endpoints[3*j+1]); if(i<0) continue;
            const double h=endpoints[3*j+2]-floor[i];
            if(std::isfinite(h) && h>=p[1] && h<=p[2]) hit(i,h);
        }
        return t.seen.size();
    } catch(const std::exception& e) { error=e.what(); return -1; }
}
}
