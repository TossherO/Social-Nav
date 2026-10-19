#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Short-window inertial alignment. All positions/rotations refer to the first
// IMU frame; acceleration is calibrated SI specific force, not gravity removed.
namespace social_nav_init {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
inline M3 rotation(const V3& v) {
    const double a = v.norm();
    return a < 1e-12 ? M3::Identity() : Eigen::AngleAxisd(a, v/a).toRotationMatrix();
}
inline V3 rotationVector(const M3& r) {
    Eigen::AngleAxisd a(r); return a.axis()*a.angle();
}
struct Imu { double t; V3 acc, gyro; };
struct Pose { double t; M3 r; V3 p; };
struct Alignment {
    V3 velocity = V3::Zero(), gravity = V3::Zero(), gyro_bias = V3::Zero(), acc_bias = V3::Zero();
    double rms = 0, gyro_rms = 0, gravity_raw_norm = 0;
    // Linearized tilt sensitivity to a 2cm (or larger fit-residual) position
    // perturbation. A conditioning indicator, not calibrated sensor covariance.
    double tilt_sensitivity = std::numeric_limits<double>::infinity();
    bool valid = false;
    std::string reason = "insufficient_window";
};
class AlignmentGate {
public:
    double stable_duration = 0;
    std::string reason = "alignment_stabilizing";
    bool accept(const Alignment& fit, double stamp, bool stationary,
                double maxTiltSensitivity, double requiredStableDuration) {
        if (!fit.valid || !std::isfinite(stamp) ||
            (!stationary && (!std::isfinite(fit.tilt_sensitivity) || fit.tilt_sensitivity>maxTiltSensitivity))) {
            anchored_=false; stable_duration=0;
            reason=fit.valid ? "alignment_sensitivity" : fit.reason;
            return false;
        }
        // Compare with the beginning of the stable interval, rather than only
        // adjacent fits: many tiny successive changes must not hide a drift.
        if (!anchored_ || stationary!=stationary_ || stamp<stamp_ ||
            rotationVector(Eigen::Quaterniond::FromTwoVectors(anchor_.gravity,fit.gravity).toRotationMatrix()).norm()>maxTiltSensitivity ||
            (anchor_.acc_bias-fit.acc_bias).norm()>.15 ||
            (anchor_.gyro_bias-fit.gyro_bias).norm()>.01) {
            anchor_=fit; stamp_=stamp; stationary_=stationary; anchored_=true;
        }
        stable_duration=stamp-stamp_;
        reason=stable_duration+1e-6>=requiredStableDuration ? "ready" : "alignment_stabilizing";
        return reason=="ready";
    }
    void reset() { anchored_=false; stable_duration=0; }
private:
    Alignment anchor_;
    double stamp_=0;
    bool anchored_=false, stationary_=false;
};
inline M3 orientationAt(const std::vector<Pose>& poses, double t) {
    auto b = std::lower_bound(poses.begin(), poses.end(), t,
        [](const Pose& p, double stamp) { return p.t < stamp; });
    if (b == poses.begin()) return b->r;
    if (b == poses.end()) return poses.back().r;
    const auto& a = *(b-1);
    return Eigen::Quaterniond(a.r).slerp((t-a.t)/(b->t-a.t), Eigen::Quaterniond(b->r)).toRotationMatrix();
}
inline M3 integrateGyro(const std::vector<Imu>& imu, double begin, double end, const V3& bias) {
    M3 r = M3::Identity();
    for (size_t k = 1; k < imu.size(); ++k) {
        double a = std::max(begin, imu[k-1].t), b = std::min(end, imu[k].t);
        if (b > a) r = r * rotation(((imu[k-1].gyro+imu[k].gyro)*.5-bias)*(b-a));
    }
    return r;
}
inline Alignment align(const std::vector<Pose>& poses, const std::vector<Imu>& imu,
                       double gravityNorm, double maxResidual, double maxBias) {
    Alignment out;
    if (poses.size() < 8 || imu.size() < 10 || poses.back().t-poses.front().t < .5 ||
        imu.front().t > poses.front().t+.01 || imu.back().t < poses.back().t-.01) return out;
    for (size_t i=1; i<poses.size(); ++i)
        if (poses[i].t <= poses[i-1].t || !poses[i].r.allFinite() || !poses[i].p.allFinite()) return out;
    for (size_t i=1; i<imu.size(); ++i)
        if (imu[i].t <= imu[i-1].t || !imu[i].acc.allFinite() || !imu[i].gyro.allFinite()) return out;

    // Rotation constraints estimate gyro bias independently of acceleration.
    for (int iteration=0; iteration<3; ++iteration) {
        M3 h = M3::Identity()*1e-5; V3 rhs = V3::Zero();
        double sum = 0;
        for (size_t j=1; j<poses.size(); ++j) {
            const auto& a=poses[j-1]; const auto& b=poses[j];
            M3 measured = a.r.transpose()*b.r;
            V3 residual=rotationVector(integrateGyro(imu,a.t,b.t,out.gyro_bias).transpose()*measured);
            M3 jac;
            for (int k=0;k<3;++k) {
                V3 bias=out.gyro_bias; bias[k]+=1e-5;
                jac.col(k)=(rotationVector(integrateGyro(imu,a.t,b.t,bias).transpose()*measured)-residual)/1e-5;
            }
            h+=jac.transpose()*jac; rhs-=jac.transpose()*residual; sum+=residual.squaredNorm();
        }
        out.gyro_bias+=h.ldlt().solve(rhs);
        out.gyro_rms=std::sqrt(sum/(poses.size()-1));
    }
    V3 axis=V3::Zero();
    for (const auto& m:imu) axis+=m.acc;
    if (axis.norm()<1) return out;
    axis.normalize(); // Only this one accelerometer-bias component is estimated.
    std::vector<V3> alpha(poses.size(),V3::Zero()), beta(alpha), velocityIntegral(alpha), biasIntegral(alpha);
    V3 ap=V3::Zero(), av=V3::Zero(), bp=V3::Zero(), bv=V3::Zero();
    size_t sample=1;
    double time=poses.front().t;
    for (size_t j=1;j<poses.size();++j) {
        while (time < poses[j].t-1e-9) {
            while (sample<imu.size() && imu[sample].t<=time+1e-9) ++sample;
            // The synchronized last IMU may precede scan end by one sample.
            // Hold it over that bounded (<10ms) tail, matching coverage checks.
            const size_t k=std::min(sample,imu.size()-1);
            const double end=sample>=imu.size() ? poses[j].t : std::min(poses[j].t,imu[k].t), dt=end-time;
            const auto& a=imu[k-1]; const auto& b=imu[k];
            const double f=std::max(0.0,std::min(1.0,((time+end)*.5-a.t)/(b.t-a.t)));
            const M3 r=orientationAt(poses,(time+end)*.5);
            const V3 acc=r*(a.acc+(b.acc-a.acc)*f), direction=r*axis;
            ap+=av*dt+acc*(.5*dt*dt); av+=acc*dt;
            bp+=bv*dt+direction*(.5*dt*dt); bv+=direction*dt;
            time=end;
        }
        alpha[j]=ap; beta[j]=bp; velocityIntegral[j]=av; biasIntegral[j]=bv;
    }
    const int n=3*(poses.size()-1);
    Eigen::MatrixXd a(n,6); Eigen::VectorXd rhs(n);
    for (size_t j=1;j<poses.size();++j) {
        const double dt=poses[j].t-poses.front().t; const int row=3*(j-1);
        a.block<3,3>(row,0)=M3::Identity()*dt;
        a.block<3,3>(row,3)=M3::Identity()*(.5*dt*dt);
        rhs.segment<3>(row)=poses[j].p-poses.front().p-alpha[j];
    }
    Eigen::VectorXd x=a.colPivHouseholderQr().solve(rhs);
    V3 gravity=x.tail<3>(), v0=x.head<3>();
    out.gravity_raw_norm=gravity.norm();
    if (gravity.norm()<7 || gravity.norm()>12) { out.reason="gravity_inconsistent"; return out; }
    gravity=gravity.normalized()*gravityNorm;
    double bias=0;
    // Fix gravity magnitude and limit accelerometer bias to its axial component.
    // The other two bias components cannot be separated from tilt in a short,
    // mostly planar trajectory and remain zero with uncertainty in the EKF.
    for (int iteration=0;iteration<6;++iteration) {
        const V3 u=gravity.normalized(), b1=u.unitOrthogonal(), b2=u.cross(b1);
        Eigen::Matrix<double,3,2> tangent; tangent<<b1,b2;
        for (size_t j=1;j<poses.size();++j) {
            const double dt=poses[j].t-poses.front().t; const int row=3*(j-1);
            a.block<3,3>(row,0)=M3::Identity()*dt;
            a.block<3,2>(row,3)=tangent*(.5*dt*dt); a.block<3,1>(row,5)=-beta[j];
            rhs.segment<3>(row)=poses[j].p-poses.front().p-(v0*dt+gravity*(.5*dt*dt)+alpha[j]-beta[j]*bias);
        }
        const Eigen::VectorXd step=a.colPivHouseholderQr().solve(rhs);
        if (!step.allFinite()) return out;
        v0+=step.head<3>(); gravity=(gravity+tangent*step.segment<2>(3)).normalized()*gravityNorm;
        bias+=step[5];
    }
    double error=0;
    for (size_t j=1;j<poses.size();++j) {
        const double dt=poses[j].t-poses.front().t;
        error+=(poses[j].p-poses.front().p-v0*dt-gravity*(.5*dt*dt)-alpha[j]+beta[j]*bias).squaredNorm();
    }
    out.gravity=gravity; out.acc_bias=axis*bias;
    out.velocity=v0+gravity*(poses.back().t-poses.front().t)+velocityIntegral.back()-biasIntegral.back()*bias;
    out.rms=std::sqrt(error/(poses.size()-1));
    const auto svd=a.jacobiSvd(Eigen::ComputeThinV);
    if (svd.singularValues().minCoeff()<1e-8) { out.reason="alignment_unobservable"; return out; }
    const Eigen::Matrix<double,6,6> sensitivity=svd.matrixV()*
        svd.singularValues().array().square().inverse().matrix().asDiagonal()*svd.matrixV().transpose();
    const Eigen::Matrix2d tilt=sensitivity.block<2,2>(3,3);
    out.tilt_sensitivity=std::max(.02,out.rms/std::sqrt(3.0))*
        std::sqrt(std::max(0.0,tilt.selfadjointView<Eigen::Lower>().eigenvalues().maxCoeff()))/gravityNorm;
    out.valid=out.velocity.allFinite() && out.velocity.norm()<5 && out.gyro_bias.norm()<.15 &&
        out.gyro_rms<.02 && std::abs(bias)<=maxBias && out.rms<=maxResidual;
    out.reason=out.valid ? "aligned" : "alignment_quality";
    return out;
}
} // namespace social_nav_init
