#!/usr/bin/env bash
set -eo pipefail
SOCIAL_NAV_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-all}"
JOBS="${BUILD_JOBS:-2}"
case "$PROFILE" in
  interfaces) PACKAGES='social_nav_localization;social_nav_people';;
  mapping) PACKAGES='social_nav_localization;social_nav_people;fast_lio;roll';;
  environment) PACKAGES='social_nav_localization;social_nav_people;social_nav_environment;fast_lio;roll';;
  sensors) PACKAGES='wheelchair_sensors;rslidar_sdk;realsense2_camera;realsense2_description';;
  visualization) PACKAGES='social_nav_rviz';;
  all) PACKAGES='';;
  *) echo 'Usage: build_ros1.sh interfaces|mapping|environment|sensors|visualization|all'; exit 2;;
esac
# An explicit system toolchain also avoids accidentally linking Conda libstdc++/Python.
unset PYTHONPATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc CXX=/usr/bin/g++
/usr/bin/python3 "$SOCIAL_NAV_PROJECT/perception/scripts/prepare_ros1_dependencies.py" --profile "$PROFILE"
source "$SOCIAL_NAV_PROJECT/perception/scripts/ros1_env.sh"
# Keep the ROS underlay available even after an interrupted CMake regeneration
# has left the generated workspace setup with an incomplete prefix list.
source /opt/ros/noetic/setup.bash --extend
BUILD_ROOT="$SOCIAL_NAV_ROS_OUTPUT/build"
if [[ "$PROFILE" == mapping || "$PROFILE" == all ]]; then
  cmake -S "$SOCIAL_NAV_ROS_OUTPUT/sources/gtsam-4.0.2" -B "$BUILD_ROOT/gtsam" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SOCIAL_NAV_ROS_OUTPUT/dependencies/gtsam" \
    -DBoost_NO_BOOST_CMAKE=ON -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_WRAP=OFF \
    -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_WITH_TBB=OFF
  cmake --build "$BUILD_ROOT/gtsam" --target install -j"$JOBS"
fi
if [[ "$PROFILE" == sensors || "$PROFILE" == all ]]; then
  cmake -S "$SOCIAL_NAV_ROS_OUTPUT/sources/librealsense" -B "$BUILD_ROOT/realsense" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SOCIAL_NAV_ROS_OUTPUT/dependencies/realsense" \
    -DFORCE_RSUSB_BACKEND=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_GLSL_EXTENSIONS=OFF -DBUILD_TOOLS=ON -DBUILD_PYTHON_BINDINGS=OFF \
    -DIMPORT_DEPTH_CAM_FW=OFF -DCHECK_FOR_UPDATES=OFF -DBUILD_WITH_CUDA=OFF
  cmake --build "$BUILD_ROOT/realsense" --target install -j"$JOBS"
fi
cd "$SOCIAL_NAV_ROS_OUTPUT/ros1_ws"
catkin_make --force-cmake -j"$JOBS" -l"$JOBS" -DCMAKE_BUILD_TYPE=Release \
  -DCATKIN_WHITELIST_PACKAGES="$PACKAGES" -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
  -DGTSAM_DIR="$SOCIAL_NAV_ROS_OUTPUT/dependencies/gtsam/lib/cmake/GTSAM" \
  -DCeres_DIR="$SOCIAL_NAV_ROS_OUTPUT/dependencies/roll/usr/lib/cmake/Ceres" \
  -DGLOG_INCLUDE_DIR="$SOCIAL_NAV_ROS_OUTPUT/dependencies/roll/usr/include" \
  -DGLOG_LIBRARY="$SOCIAL_NAV_ROS_OUTPUT/dependencies/roll/usr/lib/x86_64-linux-gnu/libglog.so" \
  -Drealsense2_DIR="$SOCIAL_NAV_ROS_OUTPUT/dependencies/realsense/lib/cmake/realsense2" \
  -Doctomap_DIR="$SOCIAL_NAV_ROS_OUTPUT/dependencies/octomap/usr/lib/x86_64-linux-gnu/octomap" \
  -DDISABLE_PCAP_PARSE=OFF
