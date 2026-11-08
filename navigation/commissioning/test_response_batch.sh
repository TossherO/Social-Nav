#!/usr/bin/env bash
# Fixed six-stage response check; no hardware access without --execute.
set -euo pipefail
SOCIAL_NAV_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SOCIAL_NAV_LOCAL_ROOT"
SOCIAL_NAV_RESPONSE_EXECUTE=0
if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && "$1" == --execute ]]; then
    SOCIAL_NAV_RESPONSE_EXECUTE=1
  elif [[ $# -eq 1 && "$1" == --dry-run ]]; then
    :
  elif [[ $# -eq 1 && "$1" == --help ]]; then
    printf '%s\n' '用法：bash navigation/commissioning/test_response_batch.sh [--dry-run|--execute]' \
      '固定六段：前进0.01、0.03、0.06 m/s，倒车0.05 m/s，左右原地转向各0.10 rad/s。每段发令2秒，段间自动归零并检查静止。' \
      '实际执行须轮椅无人乘坐、手柄回中，前后及转动范围空旷，且现场看护可直接操作物理急停。' \
      '默认只打印计划。任一安全异常终止整个批次，不自动重试。'
    exit 0
  else
    printf '%s\n' '只支持 --dry-run、--execute 或 --help，不接受自定义速度或动作。' >&2
    exit 2
  fi
fi
SOCIAL_NAV_RESPONSE_COMMAND=(/usr/bin/python3 -B -m navigation.control.response_probe)
if [[ "$SOCIAL_NAV_RESPONSE_EXECUTE" == 1 ]]; then
  SOCIAL_NAV_RESPONSE_OUTPUT="$SOCIAL_NAV_LOCAL_ROOT/outputs/navigation/ground_probe/response_$(date +%Y%m%d_%H%M%S_%N)"
  SOCIAL_NAV_RESPONSE_COMMAND+=(--execute \
    --port /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 \
    --output "$SOCIAL_NAV_RESPONSE_OUTPUT")
  printf '%s\n' "本次批次记录：$SOCIAL_NAV_RESPONSE_OUTPUT" '即将执行六段固定低速动作；请全程保持急停看护，异常时立即停车。'
fi
exec "${SOCIAL_NAV_RESPONSE_COMMAND[@]}"
