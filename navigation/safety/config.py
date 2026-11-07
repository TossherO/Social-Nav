"""Shared stopping-model settings; process watchdogs are configured separately."""
import math
from pathlib import Path
import yaml


def load_stopping_config(motion=None):
    settings=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    override=(motion or {}).get('stopping_model')
    if override is not None:
        if not isinstance(override,dict) or set(override)-{'reaction_s','actuator_delay_s'}:
            raise ValueError('Stopping model may override only reaction_s and actuator_delay_s')
        for key,value in override.items():
            if type(value) not in (int,float) or not math.isfinite(value) or value<settings[key]:
                raise ValueError('Stopping budget must be finite and at least the default: '+key)
            settings[key]=float(value)
    return settings
