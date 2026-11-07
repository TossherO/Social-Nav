"""Explicit measured vehicle profile; historical geometry is never executable."""
import hashlib
import math
from pathlib import Path
import yaml
from navigation.planning.footprint import VehicleGeometry
from navigation.safety.config import load_stopping_config


def controller_settings(motion):
    """Explicit wire fields are distinct from host velocity command caps."""
    declared=motion.get('controller')
    if declared is None:
        return dict(velocity_limit_field=math.ceil(max(motion['max_forward_mps'],motion['max_reverse_mps'])*1000),
                    feedback_max_linear_mps=max(motion['max_forward_mps'],motion['max_reverse_mps']),
                    feedback_max_angular_rps=motion['max_angular_rps'])
    names={'velocity_limit_field','feedback_max_linear_mps','feedback_max_angular_rps'}
    if not isinstance(declared,dict) or set(declared)!=names:
        raise ValueError('Controller needs explicit wire-limit and feedback-limit fields')
    limit=declared['velocity_limit_field']
    if type(limit) is not int or not 1<=limit<=32767:
        raise ValueError('Invalid controller velocity_limit_field')
    for key,minimum in (('feedback_max_linear_mps',max(motion['max_forward_mps'],motion['max_reverse_mps'])),
                        ('feedback_max_angular_rps',motion['max_angular_rps'])):
        value=declared[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<minimum:
            raise ValueError('Invalid controller feedback bound: '+key)
    return dict(declared)


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
    selected=dict(motion,parameters_assumed=False,vehicle_source=source)
    for key in ('stopping_model','controller'):
        if profile.get(key) is not None:selected[key]=profile[key]
    safety = load_stopping_config(selected)
    if limits['maximum_actuator_delay_s'] > safety['actuator_delay_s']:
        raise ValueError('Measured actuator delay exceeds stopping model')
    if limits['firmware_watchdog_s'] > safety['reaction_s']:
        raise ValueError('Firmware watchdog exceeds stopping reaction budget')
    if require_execution and profile.get('protocol') != 'af01_12_bf01_32_payload_sum_mode1_3':
        raise ValueError('Confirm the exact tested serial protocol before execution')
    if require_execution and profile.get('controller') is None:
        raise ValueError('Execution requires explicit tested controller wire and feedback fields')
    controller_settings(selected)
    return geometry, selected, profile
