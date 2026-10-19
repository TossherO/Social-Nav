#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ros1_env.sh"
exec /usr/bin/python3 -m perception.mapping_localization.pick_start "$@"
