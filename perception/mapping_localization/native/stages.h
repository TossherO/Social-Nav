#pragma once
#include <stdint.h>

// Optional timing ABI; established pose/result records remain unchanged.
extern "C" {
struct SnLioStages {
    double imu_prepare_ms,imu_predict_ms,deskew_ms,downsample_ms,jacobian_ms,filter_solve_ms;
    double deskew_device_ms,linearize_device_ms;
    uint64_t deskew_h2d_bytes,deskew_d2h_bytes;
};
struct SnRollStages {
    double feature_ms,submap_ms,downsample_ms,matching_ms,graph_ms;
    double corner_ms,surface_ms,linear_solve_ms,map_upload_ms,query_device_ms;
    uint64_t h2d_bytes,d2h_bytes,fallbacks;
    int backend,iterations;
};
int sn_lio_stages(void*,SnLioStages*);
int sn_roll_stages(void*,SnRollStages*);
}
