#pragma once
#include <stddef.h>
#include <stdint.h>
#include "stages.h"

// POD-only ABI. All arrays belong to the caller; stamps are sensor seconds.
extern "C" {
struct SnLioResult {
    double stamp, world_from_lidar[16], velocity[3], gyro_bias[3], acc_bias[3];
    double residual, processing_ms, matching_ms, map_update_ms, initial_lidar_pose[16], reference_stamp;
    int valid, reason, effective_points, scan_points, deskewed_points;
};
// Separate optional extension: the established result ABI remains unchanged.
struct SnLioAcceleration {
    uint64_t map_points,reference_points,h2d_bytes,d2h_bytes,nn_fallbacks,plane_fallbacks;
    double map_sync_ms,match_wall_ms,query_device_ms;
    int backend; // 0=CPU, 1=CUDA
};
int sn_lio_acceleration(void* handle,SnLioAcceleration* result);
void* sn_lio_create(const char* yaml_path);
void sn_lio_destroy(void* handle);
const char* sn_lio_error();
int sn_lio_step(void* handle, const double* imu_t_acc_gyr, size_t imu_count,
    const float* xyzi_time_ms, size_t point_count, double scan_begin, double scan_end,
    SnLioResult* result);
int sn_lio_cloud(void* handle, float* xyzi, size_t capacity);
}
