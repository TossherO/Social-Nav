#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ros1_env.sh"
ANNOTATION_HTTP_PYTHON="${SOCIAL_NAV_ANNOTATION_SERVICE_PYTHON:-python}"
if ! command -v "$ANNOTATION_HTTP_PYTHON" >/dev/null 2>&1; then
    echo 'Activate the project Conda environment or set SOCIAL_NAV_ANNOTATION_SERVICE_PYTHON; see docs/系统运行手册.md.' >&2
    exit 1
fi
exec "$ANNOTATION_HTTP_PYTHON" -B -m perception.detection_tracking.annotation serve "$@"
