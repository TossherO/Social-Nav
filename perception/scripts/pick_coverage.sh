#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec /usr/bin/python3 -B -m perception.mapping_localization.pick_coverage "$@"
