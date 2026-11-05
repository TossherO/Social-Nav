"""Read-only inspection of the project's dataset symlink entry points.

Never creates/replaces links, loads pickle, imports CUDA, or starts training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("coda_raw", "wheelchair_raw",
         "coda", "coda_downsample", "wheelchair", "wheelchair_weak", "wheelchair_annotations")
DETECTOR_DATA = "perception/detection_tracking/training/simpledet3d/data"


def inspect_links(root=ROOT, names=None):
    names = tuple(dict.fromkeys(NAMES if names is None else names))
    if not names or any(name not in NAMES for name in names):
        raise ValueError('Select at least one registered dataset name')
    root = Path(root).absolute()
    entry = root / "data/datasets"
    rows = []
    for name in names:
        path = entry / name
        try:
            target, available = str(path.resolve()), path.is_dir()
        except (OSError, RuntimeError):
            target, available = None, False
        rows.append({"name": name, "path": str(path), "is_symlink": path.is_symlink(),
                     "target": target, "directory_available": available})
    detector = root / DETECTOR_DATA
    try:
        bound = detector.is_symlink() and detector.is_dir() and detector.resolve() == entry.resolve()
    except (OSError, RuntimeError):
        bound = False
    return {"status": "ready" if bound and all(r["is_symlink"] and r["directory_available"] for r in rows)
            else "missing_or_invalid_links", "datasets": rows, "detector_data_alias_valid": bound,
            "checked_datasets": list(names),
            "unselected_datasets": [name for name in NAMES if name not in names],
            "dataset_contents_modified": False, "pickle_loaded": False, "training_executed": False,
            "training_readiness_verified": False, "network_requests": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=NAMES,
                        help='Check only these aliases; default: all registered datasets')
    args = parser.parse_args(argv)
    report = inspect_links(names=args.datasets)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
