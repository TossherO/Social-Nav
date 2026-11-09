#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../perception/scripts/ros1_env.sh"
exec "${SOCIAL_NAV_PYTHON:-python}" -B -m runtime "$@"
