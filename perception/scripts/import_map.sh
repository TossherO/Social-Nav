#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ros1_env.sh"
if [[ $# != 2 ]]; then echo 'Usage: import_map.sh SOURCE_MAP NEW_PROJECT_OUTPUT_MAP'; exit 2; fi
mapfile -t MAP_PATHS < <(/usr/bin/python3 - "$SOCIAL_NAV_PROJECT" "$1" "$2" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()
source, output = map(path, sys.argv[2:])
if (root / 'outputs').resolve() not in output.parents or output.exists():
    raise SystemExit('Map output must be a new directory beneath project outputs/')
if not (source / 'keyframes/poses.txt').is_file():
    raise SystemExit('Source map has no keyframe poses')
output.parent.mkdir(parents=True, exist_ok=True)
print(source)
print(output)
PY
)
if [[ ${#MAP_PATHS[@]} != 2 ]]; then exit 2; fi
exec "$SOCIAL_NAV_ROS_OUTPUT/ros1_ws/devel/lib/roll/roll_import_map" "${MAP_PATHS[@]}"
