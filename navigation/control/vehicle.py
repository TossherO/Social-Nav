"""Explicit measured vehicle profile; historical geometry is never executable."""
import hashlib
import math
from pathlib import Path
import yaml
from navigation.planning.footprint import VehicleGeometry


def load_vehicle_profile(path, motion, *, require_execution=False):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    profile = yaml.safe_load(data)
    if not isinstance(profile, dict) or profile.get('format') != 'wheelchair_vehicle_profile':
        raise ValueError('Expected wheelchair_vehicle_profile')
    verification = profile.get('verification', {})
    if not profile.get('vehicle_id') or not verification.get('source'):
        raise ValueError('Vehicle identity and measurement source are required')
    for name in ('geometry_confirmed', 'motion_confirmed', 'protocol_confirmed'):
        if verification.get(name) is not True:
            raise ValueError('Vehicle profile is not confirmed: '+name)
    source = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                  vehicle_id=profile['vehicle_id'], verification=verification)
    geometry = VehicleGeometry(**profile['geometry'], source=source, assumed=False)
    limits = profile['measured_limits']
    names = ('max_forward_mps', 'max_reverse_mps', 'max_angular_rps',
             'max_acceleration_mps2', 'max_angular_acceleration_rps2',
             'minimum_braking_mps2', 'minimum_angular_braking_rps2',
             'maximum_actuator_delay_s', 'firmware_watchdog_s')
    for name in names:
        value = limits.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive measured limit required: '+name)
    for name in names[:5]:
        if motion[name] > limits[name]:
            raise ValueError('Selected motion exceeds measured limit: '+name)
    for name, measured in (('max_acceleration_mps2', 'minimum_braking_mps2'),
                           ('max_angular_acceleration_rps2', 'minimum_angular_braking_rps2')):
        if motion[name]*motion['brake_deceleration_scale'] > limits[measured]:
            raise ValueError('Stopping model exceeds guaranteed braking: '+measured)
    safety = yaml.safe_load(Path(__file__).parents[1].joinpath('safety/config.yaml').read_text())
    if limits['maximum_actuator_delay_s'] > safety['actuator_delay_s']:
        raise ValueError('Measured actuator delay exceeds stopping model')
    if limits['firmware_watchdog_s'] > safety['reaction_s']:
        raise ValueError('Firmware watchdog exceeds stopping reaction budget')
    if require_execution and profile.get('protocol') != 'af01_12_bf01_32_payload_sum_mode1_3':
        raise ValueError('Confirm the exact tested serial protocol before execution')
    selected = dict(motion, parameters_assumed=False, vehicle_source=source)
    return geometry, selected, profile
