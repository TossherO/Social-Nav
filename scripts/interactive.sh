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
SOCIAL_NAV_AGENT_FOLLOW_CANDIDATES=0
SOCIAL_NAV_T2_BACKEND=''
SOCIAL_NAV_T2_IMAGE_FORMAT=jpeg
SOCIAL_NAV_T2_JPEG_QUALITY=80
SOCIAL_NAV_T2_IMAGE_OPTION_SET=0
SOCIAL_NAV_T2_RAW_RGB=0
SOCIAL_NAV_AGENT_SEATING=0
SOCIAL_NAV_AGENT_EXECUTION=0
SOCIAL_NAV_TRACE_TIMING=0
SOCIAL_NAV_AGENT_FLOOR=''
SOCIAL_NAV_INITIAL_Z=''
SOCIAL_NAV_INITIAL_Z_SET=0
SOCIAL_NAV_MODE_ARG=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-observations) SOCIAL_NAV_AGENT_OBSERVATIONS=1 ;;
    --agent-follow-candidates) SOCIAL_NAV_AGENT_FOLLOW_CANDIDATES=1 ;;
    --t2-raw-rgb) SOCIAL_NAV_T2_RAW_RGB=1 ;;
    --t2-backend)
      if [[ $# -lt 2 || -n "$SOCIAL_NAV_T2_BACKEND" || ( "$2" != dspark && "$2" != hosted ) ]]; then echo '--t2-backend 仅接受 dspark 或 hosted。' >&2; exit 2; fi
      SOCIAL_NAV_T2_BACKEND="$2"; shift ;;
    --t2-image-format)
      if [[ $# -lt 2 || ( "$2" != jpeg && "$2" != png ) ]]; then echo '--t2-image-format 仅接受 jpeg 或 png。' >&2; exit 2; fi
      SOCIAL_NAV_T2_IMAGE_FORMAT="$2"; SOCIAL_NAV_T2_IMAGE_OPTION_SET=1; shift ;;
    --t2-jpeg-quality)
      if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]?$ ]] || ((10#$2 > 95)); then echo '--t2-jpeg-quality 仅接受 1 至 95 的整数。' >&2; exit 2; fi
      SOCIAL_NAV_T2_JPEG_QUALITY="$2"; SOCIAL_NAV_T2_IMAGE_OPTION_SET=1; shift ;;
    --agent-seating) SOCIAL_NAV_AGENT_SEATING=1 ;;
    --agent-execution) SOCIAL_NAV_AGENT_EXECUTION=1 ;;
    --trace-timing) SOCIAL_NAV_TRACE_TIMING=1 ;;
    --floor)
      if [[ $# -lt 2 || -n "$SOCIAL_NAV_AGENT_FLOOR" || ( "$2" != floor1 && "$2" != floor2 ) ]]; then echo '--floor 仅接受 floor1 或 floor2，且只能指定一次。' >&2; exit 2; fi
      SOCIAL_NAV_AGENT_FLOOR="$2"; shift ;;
    --initial-z)
      if [[ $# -lt 2 || "$SOCIAL_NAV_INITIAL_Z_SET" == 1 || -z "$2" ]]; then echo '--initial-z 需要一个高度值且只能指定一次。' >&2; exit 2; fi
      SOCIAL_NAV_INITIAL_Z="$2"; SOCIAL_NAV_INITIAL_Z_SET=1; shift ;;
    --initial-z=*)
      if [[ "$SOCIAL_NAV_INITIAL_Z_SET" == 1 || -z "${1#--initial-z=}" ]]; then echo '--initial-z 需要一个高度值且只能指定一次。' >&2; exit 2; fi
      SOCIAL_NAV_INITIAL_Z="${1#--initial-z=}"; SOCIAL_NAV_INITIAL_Z_SET=1 ;;
    *)
      if [[ -z "$SOCIAL_NAV_MODE_ARG" ]]; then SOCIAL_NAV_MODE_ARG="$1"
      else echo '参数过多，请使用 --help 查看用法。' >&2; exit 2; fi ;;
  esac
  shift
done
case "$SOCIAL_NAV_LAUNCH_KIND" in
  observe)
    SOCIAL_NAV_LAUNCH_MODE="${SOCIAL_NAV_MODE_ARG:---start}"
    case "$SOCIAL_NAV_LAUNCH_MODE" in
      --help)
        printf '%s\n' '用法：bash scripts/interactive.sh observe [--dry-run] [--agent-observations] [--floor floor1|floor2] [--agent-follow-candidates] [--t2-raw-rgb] [--t2-backend dspark|hosted] [--t2-image-format jpeg|png] [--t2-jpeg-quality 1..95] [--agent-seating] [--initial-z 米]' \
          '先运行传感器入口。本入口打开 RViz 进行在线观察与规划，不连接底盘。'
        exit 0;;
      --start|--dry-run) ;;
      *) echo '观察入口只支持 --help 或 --dry-run。' >&2; exit 2;;
    esac
    SOCIAL_NAV_LAUNCH_CONFIG=navigation/commissioning/interactive_observation.yaml
    if [[ "$SOCIAL_NAV_AGENT_OBSERVATIONS" == 1 ]]; then
      SOCIAL_NAV_LAUNCH_CONFIG=navigation/commissioning/interactive_agent.yaml
      if [[ "$SOCIAL_NAV_AGENT_FLOOR" == floor1 ]]; then SOCIAL_NAV_LAUNCH_CONFIG=navigation/commissioning/interactive_agent_floor1.yaml; fi
    elif [[ -n "$SOCIAL_NAV_AGENT_FLOOR" ]]; then
      echo '--floor 需要 --agent-observations。' >&2; exit 2
    fi
    SOCIAL_NAV_LAUNCH_OUTPUT=live_observation
    ;;
  execute)
    if [[ -n "$SOCIAL_NAV_AGENT_FLOOR" ]]; then echo '--floor 仅用于人工驾驶观察入口。' >&2; exit 2; fi
    SOCIAL_NAV_LAUNCH_MODE="${SOCIAL_NAV_MODE_ARG:---dry-run}"
    case "$SOCIAL_NAV_LAUNCH_MODE" in
      --help)
        printf '%s\n' '用法：bash scripts/interactive.sh execute [--dry-run|--check|--execute] [--agent-observations] [--agent-follow-candidates] [--t2-backend dspark|hosted] [--t2-image-format jpeg|png] [--t2-jpeg-quality 1..95] [--agent-seating] [--agent-execution] [--initial-z 米]' \
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
    if [[ "$SOCIAL_NAV_MODE_ARG" == --help && -z "$SOCIAL_NAV_AGENT_FLOOR" && -z "$SOCIAL_NAV_T2_BACKEND" && "$SOCIAL_NAV_T2_IMAGE_OPTION_SET" == 0 && "$SOCIAL_NAV_T2_RAW_RGB" == 0 && "$SOCIAL_NAV_INITIAL_Z_SET" == 0 && "$SOCIAL_NAV_AGENT_OBSERVATIONS" == 0 && "$SOCIAL_NAV_AGENT_FOLLOW_CANDIDATES" == 0 && "$SOCIAL_NAV_AGENT_SEATING" == 0 && "$SOCIAL_NAV_AGENT_EXECUTION" == 0 && "$SOCIAL_NAV_TRACE_TIMING" == 0 ]]; then
      printf '%s\n' '用法：bash scripts/interactive.sh preflight' '只读检查地图、ROS、RViz、原生库与 GPU，不连接底盘。'
      exit 0
    fi
    if [[ -n "$SOCIAL_NAV_MODE_ARG" || -n "$SOCIAL_NAV_AGENT_FLOOR" || -n "$SOCIAL_NAV_T2_BACKEND" || "$SOCIAL_NAV_T2_IMAGE_OPTION_SET" == 1 || "$SOCIAL_NAV_T2_RAW_RGB" == 1 || "$SOCIAL_NAV_INITIAL_Z_SET" == 1 || "$SOCIAL_NAV_AGENT_OBSERVATIONS" == 1 || "$SOCIAL_NAV_AGENT_FOLLOW_CANDIDATES" == 1 || "$SOCIAL_NAV_AGENT_SEATING" == 1 || "$SOCIAL_NAV_AGENT_EXECUTION" == 1 || "$SOCIAL_NAV_TRACE_TIMING" == 1 ]]; then echo '此检查入口不接受其他参数。' >&2; exit 2; fi
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
    --config navigation/commissioning/interactive.yaml --gpu "${GPU_ID:-0}"
fi

source navigation/commissioning/performance.env
SOCIAL_NAV_SESSION_DIR="outputs/navigation/$SOCIAL_NAV_LAUNCH_OUTPUT/$(date +%Y%m%d_%H%M%S_%N)"
SOCIAL_NAV_LAUNCH_COMMAND=(taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" "$SOCIAL_NAV_LAUNCH_PYTHON" -B -m runtime.interactive
  --live --gpu "${GPU_ID:-0}" --config "$SOCIAL_NAV_LAUNCH_CONFIG"
  --environment-update-hz "$SOCIAL_NAV_COMMISSIONING_ENVIRONMENT_HZ" --environment-window-m "$SOCIAL_NAV_COMMISSIONING_WINDOW_M"
  --roll-gravity-deg "$SOCIAL_NAV_COMMISSIONING_ROLL_GRAVITY_DEG"
  --vehicle-profile configs/wheelchair_vehicle.yaml)
if [[ "$SOCIAL_NAV_AGENT_OBSERVATIONS" == 1 ]]; then
  SOCIAL_NAV_LAUNCH_COMMAND+=(--agent-observations)
fi
if [[ "$SOCIAL_NAV_AGENT_FOLLOW_CANDIDATES" == 1 ]]; then
  if [[ "$SOCIAL_NAV_AGENT_OBSERVATIONS" != 1 ]]; then
    echo '--agent-follow-candidates 要求同时启用 --agent-observations。' >&2; exit 2
  fi
  SOCIAL_NAV_LAUNCH_COMMAND+=(--agent-follow-candidates)
  if [[ "$SOCIAL_NAV_T2_RAW_RGB" == 1 ]]; then
    if [[ "$SOCIAL_NAV_LAUNCH_KIND" != observe ]]; then echo '--t2-raw-rgb 仅支持非执行观察。' >&2; exit 2; fi
    SOCIAL_NAV_LAUNCH_COMMAND+=(--t2-raw-rgb)
  fi
  if [[ "$SOCIAL_NAV_T2_IMAGE_FORMAT" == jpeg ]]; then
    SOCIAL_NAV_LAUNCH_COMMAND+=(--t2-jpeg-quality "$SOCIAL_NAV_T2_JPEG_QUALITY")
  fi
elif [[ "$SOCIAL_NAV_T2_IMAGE_OPTION_SET" == 1 || "$SOCIAL_NAV_T2_RAW_RGB" == 1 ]]; then
  echo 'T2 图像选项要求启用 --agent-follow-candidates。' >&2; exit 2
fi
if [[ -n "$SOCIAL_NAV_T2_BACKEND" ]]; then
  if [[ "$SOCIAL_NAV_AGENT_FOLLOW_CANDIDATES" != 1 ]]; then echo '--t2-backend 要求启用 --agent-follow-candidates。' >&2; exit 2; fi
  SOCIAL_NAV_LAUNCH_COMMAND+=(--t2-backend "$SOCIAL_NAV_T2_BACKEND")
fi
if [[ "$SOCIAL_NAV_AGENT_SEATING" == 1 ]]; then
  if [[ "$SOCIAL_NAV_AGENT_OBSERVATIONS" != 1 ]]; then
    echo '--agent-seating 要求同时启用 --agent-observations。' >&2; exit 2
  fi
  SOCIAL_NAV_LAUNCH_COMMAND+=(--agent-seating)
fi
if [[ "$SOCIAL_NAV_AGENT_EXECUTION" == 1 ]]; then
  if [[ "$SOCIAL_NAV_LAUNCH_KIND" != execute || "$SOCIAL_NAV_AGENT_OBSERVATIONS" != 1 ]]; then
    echo '--agent-execution 要求执行入口并同时启用 --agent-observations。' >&2; exit 2
  fi
  SOCIAL_NAV_LAUNCH_COMMAND+=(--agent-execution)
fi
if [[ "$SOCIAL_NAV_TRACE_TIMING" == 1 ]]; then
  SOCIAL_NAV_LAUNCH_COMMAND+=(--trace-timing)
fi
if [[ -n "$SOCIAL_NAV_INITIAL_Z" ]]; then
  SOCIAL_NAV_LAUNCH_COMMAND+=(--initial-z "$SOCIAL_NAV_INITIAL_Z")
fi
if [[ "$SOCIAL_NAV_LAUNCH_KIND" == execute ]]; then
  SOCIAL_NAV_LAUNCH_COMMAND+=(--execute --chassis-port /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0)
fi
SOCIAL_NAV_LAUNCH_COMMAND+=(--output "$SOCIAL_NAV_SESSION_DIR")
if [[ "$SOCIAL_NAV_LAUNCH_MODE" == --dry-run ]]; then
  printf '%q ' "${SOCIAL_NAV_LAUNCH_COMMAND[@]}"; printf '\n'; exit 0
fi

if [[ "$SOCIAL_NAV_LAUNCH_KIND" == execute ]]; then
  "$SOCIAL_NAV_LAUNCH_PYTHON" -B - "$SOCIAL_NAV_LAUNCH_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
from runtime.interactive import motion_selection
from navigation.control.vehicle import load_vehicle_profile
from navigation.planning.motion import load_motion_config
try:
    path, profile = motion_selection(yaml.safe_load(Path(sys.argv[1]).read_text()))
    load_vehicle_profile('configs/wheelchair_vehicle.yaml',
        load_motion_config(path, profile=profile), require_execution=True)
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
