#pragma once
#include <cstdint>
#include <memory>

#ifdef SOCIAL_NAV_NATIVE
namespace social_nav_native {
struct Stamp {
    double seconds=0;
    double toSec() const { return seconds; }
};
struct Vector3 { double x=0,y=0,z=0; };
struct ImuSample {
    using Ptr=std::shared_ptr<ImuSample>;
    using ConstPtr=std::shared_ptr<const ImuSample>;
    struct { Stamp stamp; } header;
    Vector3 linear_acceleration,angular_velocity;
};
struct Pose6D {
    double offset_time=0,acc[3]{},gyr[3]{},vel[3]{},pos[3]{},rot[9]{};
};
}
using SocialNavImu=social_nav_native::ImuSample;
using SocialNavImuConstPtr=SocialNavImu::ConstPtr;
#else
#include <sensor_msgs/Imu.h>
using SocialNavImu=sensor_msgs::Imu;
using SocialNavImuConstPtr=sensor_msgs::ImuConstPtr;
#endif
