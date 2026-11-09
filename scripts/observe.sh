#!/usr/bin/env bash
# This machine-local wrapper selects live shadow observation only.
set -eo pipefail
SOCIAL_NAV_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOCIAL_NAV_LOCAL_ROOT"
SOCIAL_NAV_DRY_RUN=0
if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && "$1" == --help ]]; then
    printf '%s\n' '用法：bash scripts/observe.sh [--dry-run]' \
      '先在另一终端运行 bash perception/capture/start_sensors.sh。' \
      '本入口打开RViz并进行在线观察与规划，不接入底盘。' \
      '用Ground Pose Estimate选择当前雷达位置与朝向，再选择目标；q回车退出。'
    exit 0
  elif [[ $# -eq 1 && "$1" == --dry-run ]]; then
    SOCIAL_NAV_DRY_RUN=1
  else
    printf '%s\n' '只支持 --help 或 --dry-run；实机执行不通过此观察入口开启。' >&2
    exit 2
  fi
fi
source outputs/runtime/environment/activate.sh
source perception/scripts/ros1_env.sh
source navigation/commissioning/performance.env
SOCIAL_NAV_SESSION_DIR="$SOCIAL_NAV_LOCAL_ROOT/outputs/navigation/live_observation/$(date +%Y%m%d_%H%M%S_%N)"
SOCIAL_NAV_OBSERVE_COMMAND=(taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" /home/tl/miniconda3/envs/social-nav-runtime/bin/python -B -m runtime.interactive \
  --live --gpu "${GPU_ID:-0}" --config "$SOCIAL_NAV_LOCAL_ROOT/navigation/commissioning/interactive_observation.yaml" \
  --motion-profile trial \
  --environment-update-hz "$SOCIAL_NAV_COMMISSIONING_ENVIRONMENT_HZ" --environment-window-m "$SOCIAL_NAV_COMMISSIONING_WINDOW_M" \
  --vehicle-profile "$SOCIAL_NAV_LOCAL_ROOT/configs/wheelchair_vehicle.yaml" \
  --output "$SOCIAL_NAV_SESSION_DIR")
if [[ "$SOCIAL_NAV_DRY_RUN" == 1 ]]; then
  printf '%q ' "${SOCIAL_NAV_OBSERVE_COMMAND[@]}"
  printf '\n'
  exit 0
fi
# Actual online observation must use wall time, not a leftover bag clock.
/home/tl/miniconda3/envs/social-nav-runtime/bin/python -B - <<'PY'
import rosgraph
master=rosgraph.Master('/social_nav_live_observation_start')
try:
    master.getPid()
except Exception as error:
    raise SystemExit('未连接ROS master，请先运行 bash perception/capture/start_sensors.sh。') from error
try:
    simulated=master.getParam('/use_sim_time')
except rosgraph.MasterError:
    simulated=False
if simulated:
    raise SystemExit('当前master启用了/use_sim_time，请使用真实传感器会话后再启动在线观察。')
PY
printf '%s\n' "本次观察记录：$SOCIAL_NAV_SESSION_DIR" \
  '请在RViz选择轮椅当前雷达位置与朝向；观察关联范围见 navigation/commissioning/coverage_observation_extended.yaml。' \
  '本入口仅记录规划建议；终端输入 q 并回车退出。'
exec "${SOCIAL_NAV_OBSERVE_COMMAND[@]}"
