// A C ABI keeps the geometry library usable from system ROS and Conda Python.
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
struct Voxel { float log_odds=0; double stamp=0; };
struct Tree {
    octomap::OcTree octree;
    // Use OctoMap's key/ray/sensor model with directly erasable bounded evidence.
    // This avoids old OctoMap's partial-node deletion path and stale evidence.
    std::unordered_map<Key,Voxel> seen;
    explicit Tree(double resolution): octree(resolution) {}
};
bool inside(const octomap::point3d& p, const double* box) {
    return p.x()>=box[0] && p.y()>=box[1] && p.z()>=box[2] &&
           p.x()<box[3] && p.y()<box[4] && p.z()<box[5];
}
}
extern "C" {
const char* sn_occupancy_error() { return error.c_str(); }
void* sn_occupancy_new(double resolution) {
    try { return new Tree(resolution); } catch(const std::exception& e) { error=e.what(); return nullptr; }
}
void sn_occupancy_free(void* ptr) { delete static_cast<Tree*>(ptr); }
int sn_occupancy_update(void* ptr, const float* xyz, int n, const double* origin, const float* ray_origins,
    const double* box, double stamp, double ttl, double maxrange, int ray_stride, int maxvoxels) {
    try {
        auto& t=*static_cast<Tree*>(ptr);
        for(auto it=t.seen.begin();it!=t.seen.end();) {
            const auto key=unpack(it->first);
            if(stamp-it->second.stamp>ttl || !inside(t.octree.keyToCoord(key),box)) {
                it=t.seen.erase(it);
            } else ++it;
        }
        std::unordered_set<Key> hits, misses;
        octomap::KeyRay ray;
        for(int i=0;i<n;++i) {
            octomap::point3d sensor=ray_origins ? octomap::point3d(ray_origins[3*i],ray_origins[3*i+1],ray_origins[3*i+2])
                                               : octomap::point3d(origin[0],origin[1],origin[2]);
            octomap::point3d p(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);
            const double length=(p-sensor).norm();
            if(!std::isfinite(length) || length<.01) continue;
            octomap::OcTreeKey key;
            if(length<=maxrange && inside(p,box) && t.octree.coordToKeyChecked(p,key)) hits.insert(pack(key));
            if(ray_stride<=0 || i%ray_stride) continue;
            if(length>maxrange) p=sensor+(p-sensor)*(maxrange/length);
            if(t.octree.computeRayKeys(sensor,p,ray))
                for(const auto& k:ray) if(inside(t.octree.keyToCoord(k),box)) misses.insert(pack(k));
        }
        auto update=[&](Key key,bool hit) {
            auto& v=t.seen[key];
            v.log_odds=std::max(t.octree.getClampingThresMinLog(),
                std::min(t.octree.getClampingThresMaxLog(),
                    v.log_odds+(hit?t.octree.getProbHitLog():t.octree.getProbMissLog())));
            v.stamp=stamp;
        };
        for(const auto k:misses) if(!hits.count(k)) {
            update(k,false);
        }
        for(const auto k:hits) update(k,true);
        if(t.seen.size()>static_cast<size_t>(maxvoxels)) {
            t.seen.clear();
            throw std::runtime_error("Occupancy voxel budget exceeded; map was reset to unknown");
        }
        return static_cast<int>(t.seen.size());
    } catch(const std::exception& e) { error=e.what(); return -1; }
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
}
