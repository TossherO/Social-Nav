#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

// Plain host/device records. The grid is an exact search index, not a map
// downsampler: every active point of the CPU reference is retained.
struct LioPoint { float x,y,z,intensity; };
struct LioTimedPoint {float x,y,z,intensity,offset_ms;};
struct LioImuPose {double offset,rotation[9],velocity[3],position[3],acceleration[3],angular_velocity[3];};
struct LioDeskewState {double extrinsic[9],translation[3],inverse_rotation[9],position[3],inverse_extrinsic[9];};
struct LioBox { float low[3],high[3]; uint32_t sequence; int cell[3]; };
struct LioMatch {
    LioPoint neighbors[5];
    float distance[5],normal[4];
    int count,flags,selected;
};
struct LioCudaStats {
    size_t points=0,h2d_bytes=0,d2h_bytes=0;
    double map_ms=0,match_ms=0,query_device_ms=0;
    double deskew_device_ms=0,linearize_device_ms=0;
};

class LioCuda {
    struct Impl;
    std::unique_ptr<Impl> p_;
public:
    LioCuda(int device,float map_leaf,bool plane_fit=false);
    ~LioCuda();
    LioCuda(const LioCuda&)=delete;
    LioCuda& operator=(const LioCuda&)=delete;
    void frame_begin();
    void reset(const LioPoint*,size_t);
    void search_resolution(float meters);
    void surface_fitting(bool enabled);
    void insert(const LioPoint&,const LioBox* replaced);
    void erase(const LioBox&);
    void sync_map();
    void begin_scan(const LioPoint* body,size_t);
    void match(const LioPoint* world,size_t,bool search,LioMatch* output);
    // Immutable-map batch queries; used by ROLL with its own distance gate.
    void search(const LioPoint* world,size_t,float max_squared_distance,LioMatch* output);
    void correct(const int* indices,const LioMatch* matches,size_t);
    void select(const uint8_t* selected,size_t);
    void deskew(LioTimedPoint*,size_t,const LioImuPose*,size_t,const LioDeskewState&);
    // state: row-major offset rotation, offset translation, inverse world rotation.
    void linearize(const LioPoint* host_normals,size_t,const double* state,bool estimate_extrinsics,double* hth,double* htr);
    LioCudaStats stats() const;
};
