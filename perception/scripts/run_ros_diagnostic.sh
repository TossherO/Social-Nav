#!/usr/bin/env bash
# Explicit ROS orchestration for research/diagnostics and legacy preparation.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ros1_env.sh"
exec /usr/bin/python3 -B -m perception.mapping_localization "$@"
