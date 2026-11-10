#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${SOCIAL_NAV_PYTHON:-python3}" -B -m perception.visualization.map_editor "$@"
