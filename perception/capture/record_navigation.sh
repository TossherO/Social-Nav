#!/usr/bin/env bash
set -eo pipefail
CAPTURE_TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CAPTURE_TOOLS/env.sh"
if [[ -n "${SOCIAL_NAV_RECORD_ROS_MASTER_URI:-}" ]]; then
  export ROS_MASTER_URI="$SOCIAL_NAV_RECORD_ROS_MASTER_URI"
fi
exec /usr/bin/python3 -B "$CAPTURE_TOOLS/capture.py" record navigation "$@"
