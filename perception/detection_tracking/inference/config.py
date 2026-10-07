"""Public policy plus one explicitly named, ignored machine resource file."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / 'perception/detection_tracking/configs/inference.json'
LOCAL = ROOT / 'configs/detection_tracking.local.json'


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f'Unknown inference configuration key: {key}')
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path=None, local_path=None):
    config = json.loads(DEFAULT.read_text(encoding='utf-8'))
    if path is not None and Path(path).resolve() != DEFAULT.resolve():
        config = merge(config, json.loads(Path(path).read_text(encoding='utf-8')))
    local = LOCAL if local_path is None else Path(local_path)
    if local.is_file():
        config = merge(config, json.loads(local.read_text(encoding='utf-8')))
    return config
