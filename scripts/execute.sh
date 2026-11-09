#!/usr/bin/env bash
# Prepared execution entry. Profile checks are not physical verification.
set -euo pipefail
SOCIAL_NAV_EXECUTION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOCIAL_NAV_EXECUTION_ROOT"
SOCIAL_NAV_EXECUTION_MODE="${1:---dry-run}"
if [[ $# -gt 1 ]]; then exit 2; fi
case "$SOCIAL_NAV_EXECUTION_MODE" in
  --dry-run|--check|--execute) ;;
  --help)
    printf '%s\n' '用法：bash scripts/execute.sh [--dry-run|--check|--execute]' \
      '默认只打印真实导航命令；--check只读检查候选车辆档案；--execute也必须先通过档案检查。' \
      '档案检查只验证字段和数值一致性，不验证实际制动或失联停车。执行还需现场准备、当前地图初始化和单独人工使能。'
    exit 0;;
  *) exit 2;;
esac
source navigation/commissioning/performance.env
SOCIAL_NAV_EXECUTION_COMMAND=(taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" /home/tl/miniconda3/envs/social-nav-runtime/bin/python -B -m runtime.interactive \
  --live --gpu "${GPU_ID:-0}" --config navigation/commissioning/interactive.yaml \
  --motion-profile trial \
  --environment-update-hz "$SOCIAL_NAV_COMMISSIONING_ENVIRONMENT_HZ" --environment-window-m "$SOCIAL_NAV_COMMISSIONING_WINDOW_M" \
  --vehicle-profile configs/wheelchair_vehicle.yaml \
  --execute --chassis-port /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 \
  --output "outputs/navigation/live_execution/$(date +%Y%m%d_%H%M%S_%N)")
if [[ "$SOCIAL_NAV_EXECUTION_MODE" == --dry-run ]]; then
  printf '%q ' "${SOCIAL_NAV_EXECUTION_COMMAND[@]}"; printf '\n'; exit 0
fi
source outputs/runtime/environment/activate.sh
python -B - <<'PY'
from navigation.control.vehicle import load_vehicle_profile
from navigation.planning.motion import load_motion_config
try:
    load_vehicle_profile('configs/wheelchair_vehicle.yaml',
        load_motion_config(profile='trial'),require_execution=True)
except ValueError as error:
    raise SystemExit('执行准备未通过：'+str(error)+'。未启动ROS、GPU计算或底盘串口。')
print('Vehicle profile passed; physical preparation and separate operator arming still required.')
PY
if [[ "$SOCIAL_NAV_EXECUTION_MODE" == --check ]]; then exit 0; fi
source perception/scripts/ros1_env.sh
exec "${SOCIAL_NAV_EXECUTION_COMMAND[@]}"
