#!/usr/bin/env bash
set -eo pipefail
CAPTURE_TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CAPTURE_TOOLS/env.sh"
exec /usr/bin/python3 -B "$CAPTURE_TOOLS/capture.py" sensors "$@"
