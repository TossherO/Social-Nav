#include "MotionInitializer.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

namespace social_nav_init {
void MotionInitializer::deskew(const Cloud& scan, double begin, double end, Cloud& output) const {
    output=scan;
    std::vector<double> times{begin};
    std::vector<M3> rotations{M3::Identity()};
    // A cut bag can lack IMU for the first scan's beginning. Only that uncovered
    // prefix uses the first available angular rate; no acceleration is invented.
    V3 last=imu_.front().gyro;
    for (const auto& m:imu_) {
        if (m.t<=begin) { last=m.gyro; continue; }
        if (m.t>end) break;
        rotations.push_back(rotations.back()*rotation((last+m.gyro)*(.5*(m.t-times.back()))));
        times.push_back(m.t); last=m.gyro;
    }
    if (times.back()<end) { rotations.push_back(rotations.back()*rotation(last*(end-times.back()))); times.push_back(end); }
    const M3 end_inverse=rotations.back().transpose();
    const V3 velocity_body=pose_.block<3,3>(0,0).transpose()*velocity_;
    for (auto& p:output) {
        const double time=std::max(begin,std::min(end,begin+p.curvature*.001));
        auto b=std::lower_bound(times.begin(),times.end(),time);
        size_t k=std::min<size_t>(times.size()-1,b-times.begin());
        M3 r=rotations[k];
        if (k>0 && times[k]>times[k-1])
            r=Eigen::Quaterniond(rotations[k-1]).slerp((time-times[k-1])/(times[k]-times[k-1]),Eigen::Quaterniond(r)).toRotationMatrix();
        const V3 point(p.x,p.y,p.z);
        V3 corrected=r_il_.transpose()*(end_inverse*r*(r_il_*point+t_il_)-t_il_)+velocity_body*(time-end);
        p.x=corrected.x(); p.y=corrected.y(); p.z=corrected.z();
    }
}

bool MotionInitializer::process(const Cloud::Ptr& scan, double begin,
    const std::deque<sensor_msgs::Imu::ConstPtr>& messages, Cloud& deskewed) {
    if (messages.empty() || scan->empty()) return false;
    end_stamp=begin;
    for (const auto& p:*scan) end_stamp=std::max(end_stamp,begin+p.curvature*.001);
    for (const auto& m:messages) {
        double t=m->header.stamp.toSec();
        if (!imu_.empty() && t<=imu_.back().t) continue;
        const auto& a=m->linear_acceleration; const auto& g=m->angular_velocity;
        V3 acc(a.x,a.y,a.z), gyro(g.x,g.y,g.z);
        if (!acc.allFinite() || !gyro.allFinite()) { reason="nonfinite_imu"; return false; }
        imu_.push_back({t,acc*options.acceleration_scale,gyro});
    }
    if (imu_.size()<2) return false;
    auto oldest=std::lower_bound(imu_.begin(),imu_.end(),end_stamp-options.max_duration-.1,
        [](const Imu& m,double t) { return m.t<t; });
    if (oldest!=imu_.begin()) imu_.erase(imu_.begin(),oldest-1);
    deskew(*scan,begin,end_stamp,deskewed);
    Cloud::Ptr near(new Cloud), sampled(new Cloud);
    for (const auto& p:deskewed)
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
            p.x*p.x+p.y*p.y+p.z*p.z<900) near->push_back(p);
    pcl::VoxelGrid<Cloud::PointType> voxel;
    voxel.setLeafSize(options.voxel,options.voxel,options.voxel); voxel.setInputCloud(near); voxel.filter(*sampled);
    if (sampled->size()<100) { reason="insufficient_geometry"; readiness.reset(); return false; }
    if (poses_.empty()) {
        reference_stamp=end_stamp; *map_=*sampled;
    } else {
        const double dt=end_stamp-poses_.back().t;
        if (dt<=0 || dt>1) { reason="scan_gap"; readiness.reset(); return false; }
        Eigen::Matrix4d guess=pose_;
        guess.block<3,3>(0,0)=pose_.block<3,3>(0,0)*r_il_.transpose()*
            integrateGyro(imu_,poses_.back().t,end_stamp,V3::Zero())*r_il_;
        guess.block<3,1>(0,3)+=velocity_*dt;
        pcl::IterativeClosestPoint<Cloud::PointType,Cloud::PointType> icp;
        icp.setInputSource(sampled); icp.setInputTarget(map_);
        icp.setMaximumIterations(18); icp.setMaxCorrespondenceDistance(.8);
        icp.setTransformationEpsilon(1e-7); icp.setEuclideanFitnessEpsilon(1e-6);
        Cloud transformed; icp.align(transformed,guess.cast<float>());
        icp_error=icp.getFitnessScore(.5);
        Eigen::Matrix4d next=icp.getFinalTransformation().cast<double>();
        const V3 step=next.block<3,1>(0,3)-pose_.block<3,1>(0,3);
        if (!icp.hasConverged() || !next.allFinite() || icp_error>options.max_icp_error || step.norm()>5*dt) {
            reason="bootstrap_registration"; readiness.reset(); return false;
        }
        velocity_=step/dt; pose_=next;
        *map_+=transformed;
        Cloud::Ptr reduced(new Cloud); voxel.setInputCloud(map_); voxel.filter(*reduced); map_=reduced;
        if (map_->size()>30000) {
            Cloud::Ptr bounded(new Cloud); const size_t stride=(map_->size()+29999)/30000;
            for (size_t i=0;i<map_->size();i+=stride) bounded->push_back((*map_)[i]);
            map_=bounded;
        }
    }
    const M3 r=r_il_*pose_.block<3,3>(0,0)*r_il_.transpose();
    const V3 p=r_il_*pose_.block<3,1>(0,3)+t_il_-r*t_il_;
    poses_.push_back({end_stamp,r,p}); ++scan_count;
    while (poses_.size()>2 && end_stamp-poses_[1].t>options.max_duration) poses_.erase(poses_.begin());
    auto first=std::lower_bound(imu_.begin(),imu_.end(),poses_.front().t-.01,
        [](const Imu& m,double t) { return m.t<t; });
    if (first!=imu_.begin()) imu_.erase(imu_.begin(),first-1);
    duration=end_stamp-poses_.front().t; sample_count=imu_.size();
    if (duration<options.min_duration || int(poses_.size())<options.min_scans) return false;
    result=align(poses_,imu_,9.809,options.max_residual,options.max_acc_bias);
    mode="moving";
    V3 ma=V3::Zero(), mg=V3::Zero();
    for (const auto& m:imu_) { ma+=m.acc; mg+=m.gyro; }
    ma/=imu_.size(); mg/=imu_.size();
    double va=0,travel=0,angle=0;
    for (const auto& m:imu_) va+=(m.acc-ma).squaredNorm();
    // Raw 2kHz gyro noise can be large even while the platform is stationary.
    // Test motion using 50ms rate averages, together with actual LiDAR motion.
    std::vector<V3> rate_bins;
    V3 rate_sum=V3::Zero(); int rate_count=0; double bin_start=imu_.front().t;
    for (const auto& m:imu_) {
        if (m.t-bin_start>=.05 && rate_count) {
            rate_bins.push_back(rate_sum/rate_count);
            rate_sum.setZero(); rate_count=0; bin_start=m.t;
        }
        rate_sum+=m.gyro; ++rate_count;
    }
    double slow_variance=0;
    for (const auto& rate:rate_bins) slow_variance+=(rate-mg).squaredNorm();
    const double slow_rate_std=std::sqrt(slow_variance/std::max<size_t>(1,rate_bins.size()));
    for (const auto& q:poses_) {
        travel=std::max(travel,(q.p-poses_.front().p).norm());
        angle=std::max(angle,rotationVector(poses_.front().r.transpose()*q.r).norm());
    }
    if (travel<.05 && angle<.01 && std::sqrt(va/imu_.size())<.25 &&
        rate_bins.size()>=10 && slow_rate_std<.01 && mg.norm()<.08) {
        result.gravity=-poses_.front().r*ma.normalized()*9.809;
        result.velocity.setZero(); result.gyro_bias=mg;
        result.acc_bias=ma+poses_.front().r.transpose()*result.gravity;
        result.rms=travel; result.gyro_rms=slow_rate_std;
        result.valid=result.acc_bias.norm()<options.max_acc_bias;
        result.reason=result.valid ? "aligned" : "stationary_bias_budget";
        mode="stationary";
    }
    const bool ready=readiness.accept(result,end_stamp,mode=="stationary",
        options.max_tilt_sensitivity,options.stable_duration);
    reason=readiness.reason;
    if (!ready) return false;
    alignment=Eigen::Quaterniond::FromTwoVectors(-result.gravity,V3::UnitZ()).toRotationMatrix();
    rotation_end=alignment*r; position_end=alignment*p;
    result.velocity=alignment*result.velocity; result.gravity=alignment*result.gravity;
    reason="ready";
    return true;
}
} // namespace social_nav_init
