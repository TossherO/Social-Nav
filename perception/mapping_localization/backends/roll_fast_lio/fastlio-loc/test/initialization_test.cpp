#include <gtest/gtest.h>
#include "InertialAlignment.h"
using namespace social_nav_init;

struct Trajectory {
    std::vector<Pose> poses;
    std::vector<Imu> imu;
    V3 gravity=V3(.6,-.4,-9.7).normalized()*9.809;
    V3 v0=V3(.8,.2,-.1), acceleration=V3::Zero(), gyro_bias=V3(.01,-.005,.003);
    double yaw_rate=0, axial_bias=0;
    void make() {
        poses.clear(); imu.clear();
        for (int k=0;k<=400;++k) {
            double t=k*.005; M3 r=rotation(V3(0,0,yaw_rate*t));
            V3 a=r.transpose()*(acceleration-gravity)+V3(0,0,axial_bias);
            imu.push_back({t,a,V3(0,0,yaw_rate)+gyro_bias});
            if (k%20==0) poses.push_back({t,r,v0*t+acceleration*(.5*t*t)});
        }
    }
};
TEST(InertialAlignment, RecoversNonzeroVelocityAndTiltDuringConstantMotion) {
    Trajectory t; t.make(); auto result=align(t.poses,t.imu,9.809,.06,1.5);
    ASSERT_TRUE(result.valid) << result.reason;
    EXPECT_LT((result.velocity-t.v0).norm(),.005);
    EXPECT_LT((result.gravity-t.gravity).norm(),.005);
    EXPECT_LT((result.gyro_bias-t.gyro_bias).norm(),.0001);
}
TEST(InertialAlignment, HandlesAccelerationAndTurningWithoutTreatingSpecificForceAsGravity) {
    Trajectory t; t.acceleration=V3(.25,-.1,.05); t.yaw_rate=.2; t.make();
    auto result=align(t.poses,t.imu,9.809,.06,1.5);
    ASSERT_TRUE(result.valid) << result.reason;
    EXPECT_LT((result.velocity-(t.v0+t.acceleration*2)).norm(),.01);
    EXPECT_LT((result.gravity-t.gravity).norm(),.01);
    EXPECT_LT((result.gyro_bias-t.gyro_bias).norm(),.0001);
}
TEST(InertialAlignment, SeparatesAxialBiasFromFixedGravityMagnitude) {
    Trajectory t; t.gravity=V3(0,0,-9.809); t.v0.setZero(); t.axial_bias=-.6; t.make();
    auto result=align(t.poses,t.imu,9.809,.06,1.5);
    ASSERT_TRUE(result.valid) << result.reason;
    EXPECT_LT(result.velocity.norm(),.005);
    EXPECT_NEAR(result.acc_bias.z(),-.6,.005);
    EXPECT_NEAR(result.gravity.norm(),9.809,1e-8);
}
TEST(InertialAlignment, RejectsShortWindowAndMissingImuCoverage) {
    Trajectory t; t.make(); auto poses=t.poses; poses.resize(3);
    EXPECT_FALSE(align(poses,t.imu,9.809,.06,1.5).valid);
    t.imu.erase(t.imu.begin(),t.imu.begin()+50);
    EXPECT_FALSE(align(t.poses,t.imu,9.809,.06,1.5).valid);
}
TEST(InertialAlignment, RejectsNonMonotonicAndNonfiniteInput) {
    Trajectory t; t.make(); t.imu[3].t=t.imu[2].t;
    EXPECT_FALSE(align(t.poses,t.imu,9.809,.06,1.5).valid);
    t.make(); t.poses[3].p.x()=std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(align(t.poses,t.imu,9.809,.06,1.5).valid);
}
TEST(InertialAlignment, RejectsBiasBeyondCalibrationBudget) {
    Trajectory t; t.gravity=V3(0,0,-9.809); t.axial_bias=1.0; t.make();
    EXPECT_FALSE(align(t.poses,t.imu,9.809,.06,.3).valid);
}
TEST(InertialAlignment, AcceptsOneSampleTailButRejectsStaleImu) {
    Trajectory t; t.make(); t.imu.pop_back();
    EXPECT_TRUE(align(t.poses,t.imu,9.809,.06,1.5).valid);
    for (int i=0;i<5;++i) t.imu.pop_back();
    EXPECT_FALSE(align(t.poses,t.imu,9.809,.06,1.5).valid);
}
TEST(InertialAlignment, ShortWindowNeedsMoreGeometryEvenWithZeroResidual) {
    Trajectory t; t.make(); auto short_poses=t.poses; short_poses.resize(11);
    const auto short_fit=align(short_poses,t.imu,9.809,.06,1.5);
    const auto long_fit=align(t.poses,t.imu,9.809,.06,1.5);
    ASSERT_TRUE(short_fit.valid); ASSERT_TRUE(long_fit.valid);
    EXPECT_LT(short_fit.rms,1e-6);
    EXPECT_GT(short_fit.tilt_sensitivity,.01);
    EXPECT_LT(long_fit.tilt_sensitivity,.01);
    EXPECT_GT(short_fit.tilt_sensitivity,3*long_fit.tilt_sensitivity);
    AlignmentGate gate;
    EXPECT_FALSE(gate.accept(short_fit,1,false,.01,.3));
    EXPECT_EQ(gate.reason,"alignment_sensitivity");
    EXPECT_FALSE(gate.accept(long_fit,2,false,.01,.3));
    EXPECT_FALSE(gate.accept(long_fit,2.1,false,.01,.3));
    EXPECT_TRUE(gate.accept(long_fit,2.3,false,.01,.3));
}
TEST(InertialAlignment, StableIntervalCannotChainAdjacentDriftingFits) {
    Trajectory t; t.make(); auto fit=align(t.poses,t.imu,9.809,.06,1.5);
    ASSERT_TRUE(fit.valid); const V3 gravity=fit.gravity;
    AlignmentGate gate;
    for (int k=0;k<10;++k) {
        fit.gravity=rotation(V3(.004*k,0,0))*gravity;
        EXPECT_FALSE(gate.accept(fit,2+k*.1,false,.01,.3));
    }
    EXPECT_LT(gate.stable_duration,.3);
    EXPECT_TRUE(gate.accept(fit,3.2,false,.01,.3));
    fit.valid=false; fit.reason="alignment_quality";
    EXPECT_FALSE(gate.accept(fit,3.3,false,.01,.3));
    fit.valid=true;
    EXPECT_FALSE(gate.accept(fit,3.4,false,.01,.3));
}
TEST(InertialAlignment, StationaryStillNeedsConfirmationAndModeChangeRestartsIt) {
    Trajectory t; t.make(); auto fit=align(t.poses,t.imu,9.809,.06,1.5);
    ASSERT_TRUE(fit.valid); fit.tilt_sensitivity=std::numeric_limits<double>::infinity();
    AlignmentGate gate;
    EXPECT_FALSE(gate.accept(fit,1,true,.01,.3));
    EXPECT_TRUE(gate.accept(fit,1.3,true,.01,.3));
    fit.tilt_sensitivity=.005;
    EXPECT_FALSE(gate.accept(fit,1.4,false,.01,.3));
    EXPECT_FALSE(gate.accept(fit,1.2,false,.01,.3));
    EXPECT_TRUE(gate.accept(fit,1.5,false,.01,.3));
}
int main(int argc,char** argv) {
    ::testing::InitGoogleTest(&argc,argv); return RUN_ALL_TESTS();
}
