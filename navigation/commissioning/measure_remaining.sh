#!/usr/bin/env bash
set -eo pipefail
SOCIAL_NAV_MEASURE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SOCIAL_NAV_MEASURE_ROOT"
source perception/capture/env.sh
source outputs/runtime/environment/mapping.sh
exec /usr/bin/python3 -B navigation/commissioning/measure_remaining.py "$@"
