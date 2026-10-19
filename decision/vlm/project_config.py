"""Public client defaults and repository paths; no environment changes."""
from __future__ import annotations

import json
from pathlib import Path

VLM_ROOT = Path(__file__).resolve().parent
ROOT = VLM_ROOT.parents[1]
CONFIG_DIR = VLM_ROOT / "configs"
CONFIG_PATH = CONFIG_DIR / "project.json"


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sections = {"dataset_roots", "model", "deployment", "inference", "benchmark", "development"}
    if not isinstance(config, dict) or any(not isinstance(config.get(name), dict) for name in sections):
        raise ValueError("Expected the complete project configuration")
    return config


def project_path(value):
    """Resolve a configured relative path against the repo, preserving aliases."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def dataset_path(name):
    return project_path(load_config()["dataset_roots"][name])
