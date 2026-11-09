#!/usr/bin/env bash
# Read-only resource inspection; no ROS master, sensor or chassis is started.
set -eo pipefail
SOCIAL_NAV_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SOCIAL_NAV_LOCAL_ROOT"
if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && "$1" == --help ]]; then
    printf '%s\n' '用法：bash navigation/commissioning/check.sh' '检查本次地图、ROS、RViz、原生库与GPU，不连接底盘。'
    exit 0
  fi
  printf '%s\n' '此本机检查入口不接受其他参数。' >&2
  exit 2
fi
source outputs/runtime/environment/activate.sh
source perception/scripts/ros1_env.sh
exec /home/tl/miniconda3/envs/social-nav-runtime/bin/python -B -m runtime.preflight \
  --config navigation/commissioning/interactive.yaml \
  --motion-profile trial --gpu "${GPU_ID:-0}"
