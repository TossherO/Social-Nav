#!/usr/bin/env bash
# Select a bag start, then build a reviewable fixed-map update candidate.
set -eo pipefail
UPDATE_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
UPDATE_ENV="${SOCIAL_NAV_RUNTIME_ENV:-$UPDATE_PROJECT/outputs/runtime/environment/activate.sh}"
if [[ -f "$UPDATE_ENV" ]]; then
  update_load_environment() { source "$UPDATE_ENV"; }
  update_load_environment
  unset -f update_load_environment
elif [[ -n "${SOCIAL_NAV_RUNTIME_ENV:-}" ]]; then
  echo "Runtime environment file not found: $UPDATE_ENV" >&2
  exit 2
fi
export PYTHONPATH="$UPDATE_PROJECT:${PYTHONPATH:-}"
exec "${SOCIAL_NAV_PYTHON:-python}" -B -m perception.mapping_localization.update_map_workflow "$@"
