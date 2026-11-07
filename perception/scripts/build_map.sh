#!/usr/bin/env bash
# Offline FAST-LIO + ROLL mapping, without detector/navigation processes.
set -eo pipefail
MAP_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$MAP_PROJECT/perception/scripts/ros1_env.sh"
MAP_LOCAL_ENV="${SOCIAL_NAV_MAPPING_ENV:-$MAP_PROJECT/outputs/runtime/environment/mapping.sh}"
if [[ -f "$MAP_LOCAL_ENV" ]]; then
  map_load_local_environment() { source "$MAP_LOCAL_ENV"; }
  map_load_local_environment
  unset -f map_load_local_environment
elif [[ -n "${SOCIAL_NAV_MAPPING_ENV:-}" ]]; then
  echo "Mapping environment file not found: $MAP_LOCAL_ENV" >&2
  exit 2
fi
export PYTHONPATH="$MAP_PROJECT:${PYTHONPATH:-}"
exec "${SOCIAL_NAV_MAPPING_PYTHON:-/usr/bin/python3}" -B -m perception.mapping_localization.build_map "$@"
