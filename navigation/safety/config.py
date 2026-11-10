"""Shared finite-horizon geometry settings; watchdogs are configured separately."""
from pathlib import Path
import yaml


def load_safety_config():
    return yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
