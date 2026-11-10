#!/usr/bin/env bash
# 通用交互入口，以及本车观察、执行和部署检查的预设。
set -eo pipefail
case "${1:-}" in
  observe|execute|preflight) ;;
  *)
    if [[ "${1:-}" == --help ]]; then
      printf '%s\n' '本车预设：observe [--dry-run]、execute [--dry-run|--check|--execute]、preflight [--help]。'
    fi
    source "$(dirname -- "${BASH_SOURCE[0]}")/../perception/scripts/ros1_env.sh"
    exec "${SOCIAL_NAV_PYTHON:-python}" -B -m runtime.interactive "$@"
    ;;
esac
SOCIAL_NAV_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOCIAL_NAV_LOCAL_ROOT"
SOCIAL_NAV_LAUNCH_KIND="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
SOCIAL_NAV_AGENT_OBSERVATIONS=0
SOCIAL_NAV_MODE_ARG=''
for SOCIAL_NAV_ARG in "$@"; do
  if [[ "$SOCIAL_NAV_ARG" == --agent-observations ]]; then
    SOCIAL_NAV_AGENT_OBSERVATIONS=1
  elif [[ -z "$SOCIAL_NAV_MODE_ARG" ]]; then
    SOCIAL_NAV_MODE_ARG="$SOCIAL_NAV_ARG"
  else
    echo '参数过多，请使用 --help 查看用法。' >&2; exit 2
  fi
done
case "$SOCIAL_NAV_LAUNCH_KIND" in
  observe)
    SOCIAL_NAV_LAUNCH_MODE="${SOCIAL_NAV_MODE_ARG:---start}"
    case "$SOCIAL_NAV_LAUNCH_MODE" in
      --help)
        printf '%s\n' '用法：bash scripts/interactive.sh observe [--dry-run] [--agent-observations]' \
          '先运行传感器入口。本入口打开 RViz 进行在线观察与规划，不连接底盘。'
        exit 0;;
      --start|--dry-run) ;;
      *) echo '观察入口只支持 --help 或 --dry-run。' >&2; exit 2;;
    esac
    SOCIAL_NAV_LAUNCH_CONFIG=navigation/commissioning/interactive_observation.yaml
    SOCIAL_NAV_LAUNCH_OUTPUT=live_observation
    ;;
  execute)
    SOCIAL_NAV_LAUNCH_MODE="${SOCIAL_NAV_MODE_ARG:---dry-run}"
    case "$SOCIAL_NAV_LAUNCH_MODE" in
      --help)
        printf '%s\n' '用法：bash scripts/interactive.sh execute [--dry-run|--check|--execute] [--agent-observations]' \
          '默认只打印命令；--check 只检查车辆档案；--execute 启动实机会话，仍需单独人工使能。' \
          '档案检查只验证字段和数值一致性，不验证实际制动或失联停车。'
        exit 0;;
      --dry-run|--check|--execute) ;;
      *) echo '执行入口只支持 --help、--dry-run、--check 或 --execute。' >&2; exit 2;;
    esac
    SOCIAL_NAV_LAUNCH_CONFIG=navigation/commissioning/interactive.yaml
    SOCIAL_NAV_LAUNCH_OUTPUT=live_execution
    ;;
  preflight)
    if [[ $# -eq 1 && "$1" == --help ]]; then
      printf '%s\n' '用法：bash scripts/interactive.sh preflight' '只读检查地图、ROS、RViz、原生库与 GPU，不连接底盘。'
      exit 0
    fi
    if [[ $# -ne 0 ]]; then echo '此检查入口不接受其他参数。' >&2; exit 2; fi
    SOCIAL_NAV_LAUNCH_MODE=--check
    ;;
  *) echo '请通过观察、执行或部署检查入口启动。' >&2; exit 2;;
esac

# Keep the wheelchair's existing local activation when present. An explicitly
# selected Python works on other machines without creating an artificial file.
SOCIAL_NAV_REQUESTED_PYTHON="${SOCIAL_NAV_PYTHON:-}"
if [[ "$SOCIAL_NAV_LAUNCH_MODE" != --dry-run && -f outputs/runtime/environment/activate.sh ]]; then
  source outputs/runtime/environment/activate.sh
fi
if [[ -n "$SOCIAL_NAV_REQUESTED_PYTHON" ]]; then
  SOCIAL_NAV_LAUNCH_PYTHON="$SOCIAL_NAV_REQUESTED_PYTHON"
elif [[ -n "${SOCIAL_NAV_PYTHON:-}" ]]; then
  SOCIAL_NAV_LAUNCH_PYTHON="$SOCIAL_NAV_PYTHON"
elif [[ -x /home/tl/miniconda3/envs/social-nav-runtime/bin/python ]]; then
  SOCIAL_NAV_LAUNCH_PYTHON=/home/tl/miniconda3/envs/social-nav-runtime/bin/python
else
  SOCIAL_NAV_LAUNCH_PYTHON=python
fi
if [[ "$SOCIAL_NAV_LAUNCH_MODE" != --dry-run && ! -f outputs/runtime/environment/activate.sh ]]; then
  echo '未找到本机 activate.sh，使用当前环境和所选 Python。' >&2
fi

if [[ "$SOCIAL_NAV_LAUNCH_KIND" == preflight ]]; then
  source perception/scripts/ros1_env.sh
  exec "$SOCIAL_NAV_LAUNCH_PYTHON" -B -m runtime.preflight \
    --config navigation/commissioning/interactive.yaml --motion-profile trial --gpu "${GPU_ID:-0}"
fi

source navigation/commissioning/performance.env
SOCIAL_NAV_SESSION_DIR="outputs/navigation/$SOCIAL_NAV_LAUNCH_OUTPUT/$(date +%Y%m%d_%H%M%S_%N)"
SOCIAL_NAV_LAUNCH_COMMAND=(taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" "$SOCIAL_NAV_LAUNCH_PYTHON" -B -m runtime.interactive
  --live --gpu "${GPU_ID:-0}" --config "$SOCIAL_NAV_LAUNCH_CONFIG" --motion-profile trial
  --environment-update-hz "$SOCIAL_NAV_COMMISSIONING_ENVIRONMENT_HZ" --environment-window-m "$SOCIAL_NAV_COMMISSIONING_WINDOW_M"
  --vehicle-profile configs/wheelchair_vehicle.yaml)
if [[ "$SOCIAL_NAV_AGENT_OBSERVATIONS" == 1 ]]; then
  SOCIAL_NAV_LAUNCH_COMMAND+=(--agent-observations)
fi
if [[ "$SOCIAL_NAV_LAUNCH_KIND" == execute ]]; then
  SOCIAL_NAV_LAUNCH_COMMAND+=(--execute --chassis-port /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0)
fi
SOCIAL_NAV_LAUNCH_COMMAND+=(--output "$SOCIAL_NAV_SESSION_DIR")
if [[ "$SOCIAL_NAV_LAUNCH_MODE" == --dry-run ]]; then
  printf '%q ' "${SOCIAL_NAV_LAUNCH_COMMAND[@]}"; printf '\n'; exit 0
fi

if [[ "$SOCIAL_NAV_LAUNCH_KIND" == execute ]]; then
  "$SOCIAL_NAV_LAUNCH_PYTHON" -B - <<'PY'
from navigation.control.vehicle import load_vehicle_profile
from navigation.planning.motion import load_motion_config
try:
    load_vehicle_profile('configs/wheelchair_vehicle.yaml',
        load_motion_config(profile='trial'), require_execution=True)
except ValueError as error:
    raise SystemExit('执行准备未通过：'+str(error)+'。未启动 ROS、GPU 计算或底盘串口。')
print('车辆档案检查通过；实际执行仍需现场准备与单独人工使能。')
PY
  if [[ "$SOCIAL_NAV_LAUNCH_MODE" == --check ]]; then exit 0; fi
fi

source perception/scripts/ros1_env.sh
# Both live presets require a real-time ROS master; a stale bag clock must not
# be interpreted as a live sensor clock.
"$SOCIAL_NAV_LAUNCH_PYTHON" -B - <<'PY'
import rosgraph
master=rosgraph.Master('/social_nav_live_start')
try:
    master.getPid()
except Exception as error:
    raise SystemExit('未连接 ROS master，请先运行 bash perception/capture/start_sensors.sh。') from error
try:
    simulated=master.getParam('/use_sim_time')
except rosgraph.MasterError:
    simulated=False
if simulated:
    raise SystemExit('当前 master 启用了 /use_sim_time，请使用真实传感器会话后再启动。')
PY
printf '%s\n' "本次记录：$SOCIAL_NAV_SESSION_DIR" \
  '请在 RViz 选择轮椅当前雷达位置与朝向，然后选择目标。'
if [[ "$SOCIAL_NAV_LAUNCH_KIND" == observe ]]; then
  printf '%s\n' '本入口仅记录规划建议，不连接底盘；终端输入 q 并回车退出。'
else
  printf '%s\n' '执行会话需要单独人工使能；终端输入 q 并回车退出。'
fi
exec "${SOCIAL_NAV_LAUNCH_COMMAND[@]}"
