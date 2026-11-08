#!/usr/bin/env bash
# One fixed supervised firmware silence probe; default prints the plan only.
set -euo pipefail
SOCIAL_NAV_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SOCIAL_NAV_LOCAL_ROOT"
SOCIAL_NAV_DO_SILENCE_TEST=0
if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && "$1" == --execute ]]; then
    SOCIAL_NAV_DO_SILENCE_TEST=1
  elif [[ $# -eq 1 && "$1" == --dry-run ]]; then
    :
  elif [[ $# -eq 1 && "$1" == --help ]]; then
    printf '%s\n' '用法：bash navigation/commissioning/test_silence.sh [--dry-run|--execute]' \
      '默认只打印计划。--execute执行一轮最高0.10 m/s、发令1.5秒、最长断发0.8秒的检查。' \
      '执行时必须停稳、无人乘坐、手柄回中，处于平整空旷场地，且看护可直接操作物理急停。' \
      '不要拔USB、关闭程序或操作手柄；断发和恢复零速由程序处理，异常时立即按急停。'
    exit 0
  else
    printf '%s\n' '只支持 --dry-run、--execute 或 --help，不接受其他参数。' >&2
    exit 2
  fi
fi
SOCIAL_NAV_SILENCE_COMMAND=(/usr/bin/python3 -B -m navigation.control.ground_probe \
  --speed 0.10 --pulse-s 1.5 --silence-s 0.8)
if [[ "$SOCIAL_NAV_DO_SILENCE_TEST" == 1 ]]; then
  SOCIAL_NAV_SILENCE_OUTPUT="$SOCIAL_NAV_LOCAL_ROOT/outputs/navigation/ground_probe/silence_$(date +%Y%m%d_%H%M%S_%N)"
  SOCIAL_NAV_SILENCE_COMMAND+=(--execute --motion \
    --port /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 \
    --output "$SOCIAL_NAV_SILENCE_OUTPUT")
  printf '%s\n' "本次检查记录：$SOCIAL_NAV_SILENCE_OUTPUT" \
    '即将检查静止与手柄中立，随后短时直行、断发观察并恢复零速。请保持物理急停看护。'
fi
exec "${SOCIAL_NAV_SILENCE_COMMAND[@]}"
