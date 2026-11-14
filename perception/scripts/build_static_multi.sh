#!/usr/bin/env bash
# Select each survey's initial pose, then register and fuse all recorded scans.
set -eo pipefail
STATIC_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STATIC_ENV="${SOCIAL_NAV_RUNTIME_ENV:-$STATIC_PROJECT/outputs/runtime/environment/activate.sh}"
if [[ -f "$STATIC_ENV" ]]; then
  static_load_environment() { source "$STATIC_ENV"; }
  static_load_environment
  unset -f static_load_environment
elif [[ -n "${SOCIAL_NAV_RUNTIME_ENV:-}" ]]; then
  echo "Runtime environment file not found: $STATIC_ENV" >&2
  exit 2
fi
export PYTHONPATH="$STATIC_PROJECT:${PYTHONPATH:-}"
exec "${SOCIAL_NAV_PYTHON:-python}" -B -m perception.environment.multi_workflow "$@"
