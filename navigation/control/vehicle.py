"""Explicit measured vehicle profile; historical geometry is never executable."""
import hashlib
import math
from pathlib import Path
import yaml
from navigation.planning.footprint import VehicleGeometry


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
             'max_acceleration_mps2', 'max_angular_acceleration_rps2')
    for name in names:
        value = limits.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive measured limit required: '+name)
    for name in names:
        if motion[name] > limits[name]:
            raise ValueError('Selected motion exceeds measured limit: '+name)
    selected=dict(motion,parameters_assumed=False,vehicle_source=source)
    minimum=profile.get('minimum_command_speeds')
    if minimum is not None:
        names={'forward_mps','reverse_mps','angular_rps'}
        if not isinstance(minimum,dict) or set(minimum)!=names:
            raise ValueError('Minimum command speeds require forward_mps, reverse_mps and angular_rps')
        for name,limit in (('forward_mps',motion['max_forward_mps']),('reverse_mps',motion['max_reverse_mps']),('angular_rps',motion['max_angular_rps'])):
            value=minimum[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=limit:
                raise ValueError('Invalid minimum command speed: '+name)
        selected['minimum_command_speeds']=dict(minimum)
    if profile.get('controller') is not None:selected['controller']=profile['controller']
    if require_execution and profile.get('protocol') != 'af01_12_bf01_32_payload_sum_mode1_3':
        raise ValueError('Confirm the exact tested serial protocol before execution')
    if require_execution and profile.get('controller') is None:
        raise ValueError('Execution requires explicit tested controller wire and feedback fields')
    if require_execution:
        # Existing deployed profiles used a misleading planar name for this
        # same longitudinal-only check. Do not silently accept contradictions.
        bound=profile.get('execution_max_longitudinal_speed_mps')
        legacy=profile.get('execution_max_planar_speed_mps')
        if bound is None:bound=legacy
        elif legacy is not None and legacy!=bound:
            raise ValueError('Conflicting execution speed bounds')
        if type(bound) not in (int,float) or not math.isfinite(bound) or bound < max(limits['max_forward_mps'], limits['max_reverse_mps']):
            raise ValueError('Invalid execution longitudinal-speed bound')
        selected['execution_max_longitudinal_speed_mps']=bound
        selected['execution_max_angular_rps'] = limits['max_angular_rps']
    controller_settings(selected)
    return geometry, selected, profile
