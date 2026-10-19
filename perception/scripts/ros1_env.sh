#!/usr/bin/env bash
# Source in the terminal which will run ROS; never changes shell startup files.
SOCIAL_NAV_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep command arguments such as --help out of catkin's setup utility.
social_nav_source_setup() {
  local SOCIAL_NAV_SETUP_FILE="$1"
  shift
  source "$SOCIAL_NAV_SETUP_FILE"
}
social_nav_source_setup /opt/ros/noetic/setup.bash
SOCIAL_NAV_ROS_OUTPUT="$SOCIAL_NAV_PROJECT/outputs/perception"
export CMAKE_PREFIX_PATH="$SOCIAL_NAV_ROS_OUTPUT/dependencies/gtsam:$SOCIAL_NAV_ROS_OUTPUT/dependencies/roll/usr:$SOCIAL_NAV_ROS_OUTPUT/dependencies/realsense:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$SOCIAL_NAV_ROS_OUTPUT/dependencies/gtsam/lib:$SOCIAL_NAV_ROS_OUTPUT/dependencies/roll/usr/lib/x86_64-linux-gnu:$SOCIAL_NAV_ROS_OUTPUT/dependencies/realsense/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$SOCIAL_NAV_ROS_OUTPUT/ros1_ws/devel/setup.bash" ]]; then
  social_nav_source_setup "$SOCIAL_NAV_ROS_OUTPUT/ros1_ws/devel/setup.bash"
fi
unset -f social_nav_source_setup
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOCIAL_NAV_PROJECT:${PYTHONPATH:-}"
