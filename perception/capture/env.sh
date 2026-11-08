#!/usr/bin/env bash
# Machine-local environment. Source only from this directory's launchers.
CAPTURE_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CAPTURE_DRIVER_ENV="${SOCIAL_NAV_SENSOR_ENV:-/home/tl/hys/sensor_migration/scripts/env.sh}"
if [[ ! -f "$CAPTURE_DRIVER_ENV" ]]; then
  echo "未找到本机传感器环境：$CAPTURE_DRIVER_ENV" >&2
  return 1
fi
unset PYTHONHOME
# catkin setup utilities must not receive launcher arguments such as --help.
capture_load_driver_env() {
  source "$CAPTURE_DRIVER_ENV"
}
capture_load_driver_env || return
unset -f capture_load_driver_env
export ROS_PACKAGE_PATH="$CAPTURE_PROJECT/perception/sensors/ros:${ROS_PACKAGE_PATH:-}"
export PYTHONPATH="$CAPTURE_PROJECT:${PYTHONPATH:-}"
export ROS_HOME="$CAPTURE_PROJECT/outputs/local_capture/ros_home"
export ROS_LOG_DIR="$CAPTURE_PROJECT/outputs/local_capture/ros_logs"
export PYTHONNOUSERSITE=1
cd "$CAPTURE_PROJECT" || return
