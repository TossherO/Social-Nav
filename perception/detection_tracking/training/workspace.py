"""Read-only source audit and print-only detector dry-run command.

Never imports SimpleDet3d, Torch or CUDA; never loads checkpoints/pickle files,
creates resource links, compiles extensions or starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

TRAINING_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = TRAINING_ROOT / "simpledet3d"
MANIFEST_PATH = TRAINING_ROOT / "source_snapshot.json"


def inspect_snapshot(training_root=TRAINING_ROOT):
    training_root = Path(training_root).resolve()
    copied = training_root / "simpledet3d"
    manifest = json.loads((training_root / "source_snapshot.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Unsupported detector source snapshot")
    missing, changed = [], []
    for name, record in manifest["files"].items():
        path = (copied / name).resolve()
        if not path.is_relative_to(copied.resolve()):
            raise ValueError("Snapshot file escapes the copied source directory")
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            changed.append(name)
    return {"component": "perception.detection_tracking.training", "source_root": str(copied),
            "snapshot_matches": not missing and not changed,
            "copied_files": len(manifest["files"]), "missing": missing, "changed": changed,
            "data_binding_present": (copied / "data").is_dir(),
            "training_readiness_verified": False, "online_inference_checked": False,
            "training_executed": False, "cuda_environment_checked": False,
            "notice": "Source integrity only. Bind resources and prepare the detector's own environment separately."}


def dry_run_command(checkpoint, run_tag="social_nav_detection", conda_env=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_tag):
        raise ValueError("Invalid run tag")
    if conda_env is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", conda_env):
        raise ValueError("Use an explicit valid existing conda environment name")
    command = ["env", "DRY_RUN=1", f"CODA_CKPT={Path(checkpoint).expanduser().resolve()}", f"RUN_TAG={run_tag}"]
    if conda_env:
        command += ["conda", "run", "--no-capture-output", "-n", conda_env]
    command += ["bash", "train_wheelchair_3stage.sh"]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true", help="Hash copied source only; no environment/GPU checks")
    actions.add_argument("--print-command", action="store_true", help="Print manual three-stage DRY_RUN=1 command only")
    parser.add_argument("--checkpoint", type=Path, help="Verified CODA initialization checkpoint, not loaded")
    parser.add_argument("--run-tag", default="social_nav_detection")
    parser.add_argument("--conda-env", help="Optional explicit detector environment; never creates/modifies it")
    args = parser.parse_args(argv)
    if args.check:
        report = inspect_snapshot()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["snapshot_matches"] else 1
    if args.checkpoint is None:
        parser.error("--print-command requires --checkpoint; no checkpoint is selected automatically")
    command = dry_run_command(args.checkpoint, args.run_tag, args.conda_env)
    print("# Print only; source/data/CUDA/weights are not checked and no training is started.")
    print("cd " + shlex.quote(str(SOURCE_ROOT)))
    print(shlex.join(command))
    return 0
