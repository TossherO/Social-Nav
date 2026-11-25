"""Validate runtime transport and display configuration."""
import math
from pathlib import Path
import yaml
from interfaces.shared import validate_pool_capacity
from perception.mapping_localization.configuration import merge_config

ROOT=Path(__file__).resolve().parents[1]


def runtime_settings(path):
    """Partial machine overrides inherit current transport/display defaults."""
    defaults=yaml.safe_load((ROOT/'configs/runtime.yaml').read_text())
    settings=merge_config(defaults,yaml.safe_load(Path(path).read_text()))
    profile=settings.get('observation_vehicle_profile')
    if profile is not None and (not isinstance(profile,str) or not profile.strip()):
        raise ValueError('observation_vehicle_profile must be a vehicle profile path or null')
    for pool in ('scan', 'safety'):
        validate_pool_capacity(settings[f'shared_{pool}_slots'], settings[f'shared_{pool}_slot_bytes'])
    for key in ('control_period_s', 'scene_export_hz', 'sensor_input_timeout_s', 'startup_timeout_s', 'finalization_timeout_s'):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite runtime setting required: ' + key)
    if not .01 <= settings['control_period_s'] <= .2:
        raise ValueError('Control period must stay in [0.01,0.2] seconds')
    display=settings['display'];route=settings['route_replay']
    for key in ('scan_hz','people_hz','map_hz','rgb_hz','navigation_hz'):
        value=display[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=60:
            raise ValueError('Display frequency must be in [0,60]: '+key)
    if type(display['max_scan_points']) is not int or not 1<=display['max_scan_points']<=200000:
        raise ValueError('Invalid display point budget')
    if type(display['path_z_m']) not in (int,float) or not math.isfinite(display['path_z_m']):
        raise ValueError('Display drawing height must be finite')
    clearance=display['vehicle_clearance_m']
    if type(clearance) not in (int,float) or not math.isfinite(clearance) or clearance<0:
        raise ValueError('Display vehicle ground clearance must be nonnegative and finite')
    if display['height_mode'] not in ('ground','plane'):raise ValueError('Invalid display height mode')
    radius=display['ground_plane_radius_m']
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<0:
        raise ValueError('Display floor-fit radius must be nonnegative and finite')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in route.values()):
        raise ValueError('Route replay bounds must be positive and finite')
    return settings
