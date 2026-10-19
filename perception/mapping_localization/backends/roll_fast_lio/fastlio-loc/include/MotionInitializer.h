#pragma once
#include "InertialAlignment.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>
#include <deque>

namespace social_nav_init {
class MotionInitializer {
public:
    using Cloud = pcl::PointCloud<pcl::PointXYZINormal>;
    struct Options {
        double min_duration=1.0, max_duration=6, acceleration_scale=1, voxel=.3;
        double max_residual=.06, max_acc_bias=1.5, max_icp_error=.04;
        double max_tilt_sensitivity=.01, stable_duration=.3;
        int min_scans=10;
    } options;
    Alignment result;
    M3 rotation_end=M3::Identity(), alignment=M3::Identity();
    V3 position_end=V3::Zero();
    std::string mode="collecting", reason="collecting";
    double reference_stamp=0, end_stamp=0, duration=0, icp_error=0;
    int scan_count=0, sample_count=0;
    AlignmentGate readiness;
    void setExtrinsic(const M3& r, const V3& t) { r_il_=r; t_il_=t; }
    bool process(const Cloud::Ptr& scan, double begin,
                 const std::deque<sensor_msgs::Imu::ConstPtr>& messages, Cloud& deskewed);
private:
    M3 r_il_=M3::Identity(); V3 t_il_=V3::Zero();
    Eigen::Matrix4d pose_=Eigen::Matrix4d::Identity();
    V3 velocity_=V3::Zero();
    Cloud::Ptr map_{new Cloud};
    std::vector<Pose> poses_;
    std::vector<Imu> imu_;
    void deskew(const Cloud& scan, double begin, double end, Cloud& output) const;
};
} // namespace social_nav_init
