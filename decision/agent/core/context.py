"""Model-facing projections; original observations and bindings stay unchanged."""
import copy
import json
from .store import encode


def robot_context(navigation):
    geometry = getattr(navigation,'geometry',None)
    motion = getattr(navigation,'motion',None)
    body = dict(status='unavailable')
    if geometry is not None:
        low,high = geometry.footprint.min(axis=0),geometry.footprint.max(axis=0)
        body = dict(status='assumed' if geometry.assumed else 'configured',
                    use='simulation_geometry' if navigation.scope=='simulation' else 'observer_geometry_reference',
                    footprint_xy_m=geometry.footprint.tolist(),length_m=float(high[0]-low[0]),width_m=float(high[1]-low[1]),
                    base_link_x_bounds_m=[float(low[0]),float(high[0])],base_link_y_bounds_m=[float(low[1]),float(high[1])],
                    source_status=geometry.source.get('status') if isinstance(geometry.source,dict) else None)
    keys = ('allow_reverse','allow_in_place_rotation','min_turning_radius_m','max_forward_mps','max_reverse_mps',
            'max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2')
    limits = dict(status='adapter_configured' if motion is not None else 'unavailable',
                  source='active_adapter_motion' if motion is not None else 'not_published_by_this_adapter',
                  parameters_assumed=None if motion is None else motion.get('parameters_assumed'),
                  **{key:None if motion is None else motion.get(key) for key in keys})
    return dict(type='intelligent_wheelchair',scope=navigation.scope,pose_reference='base_link',kinematics_interface='differential',
                velocity_fields=['forward_mps','yaw_rate_rad_s'],lateral_command_supported=False,
                body=body,motion=limits,physical_platform_performance_verified=False)


def same_fixed_annotations(annotations, reference):
    def fixed(value):
        return dict(points=[p for p in value.get('points',[]) if p['place_id']!='mission_start'],
                    regions=value.get('regions',[]),landmarks=value.get('landmarks',[]))
    return fixed(annotations)==fixed(reference.record['annotations'])


def view_context(record, reference):
    """An explicit projection, not a new record claiming the original hash."""
    value = copy.deepcopy(record)
    if (reference is None or record.get('frame_id')!=reference.record['frame_id'] or
            record.get('global_binding')!=reference.record['global_binding'] or
            not same_fixed_annotations(record.get('annotations',{}),reference)):
        return value
    annotations = value.pop('annotations')
    value['annotation_reference'] = dict(reference_map_id=reference.record['reference_map_id'],
        task_points=[p for p in annotations.get('points',[]) if p['place_id']=='mission_start'])
    value['record_representation'] = 'projection_of_original_view_id'
    return value


def project_dialogue(messages, reference):
    material = copy.deepcopy(messages)
    if reference is None:
        return material
    reference.verify()
    for message in material:
        if not isinstance(message.get('content'),str):
            continue
        try:
            value = json.loads(message['content'])
        except ValueError:
            continue
        if not isinstance(value,dict):
            continue
        if value.get('source')=='initial_map_catalog' and isinstance(value.get('catalog'),dict):
            catalog = value['catalog']
            if (catalog.get('frame_id')==reference.record['frame_id'] and
                    same_fixed_annotations(catalog.get('annotations',{}),reference)):
                annotations = catalog.pop('annotations')
                catalog['annotation_reference'] = dict(reference_map_id=reference.record['reference_map_id'],
                    task_points=[p for p in annotations.get('points',[]) if p['place_id']=='mission_start'])
                message['content'] = encode(value)
        elif message.get('role')=='tool' and isinstance(value.get('view'),dict):
            value['view'] = view_context(value['view'],reference)
            message['content'] = encode(value)
    return material


def visual_inputs(state, observation, spatial_view, reference, keyframes, status):
    feedback = state['navigation']['feedback']
    now = feedback['stamp_s']
    def current(record, identity, stamp):
        if record['clock_domain']!=feedback['clock_domain'] or (stamp is not None and stamp>now):
            raise ValueError('Current image source is future or belongs to another clock')
        return dict(available=True,**{identity:record[identity]},source_stamp_s=stamp,
                    age_s=now-stamp if stamp is not None else None)
    rgb = dict(available=False,reason=status.get('rgb','not_provided'))
    if observation is not None:
        rgb = current(observation.record,'observation_id',observation.record['stamp_s'])
        rgb['rgb_stamp_s'] = observation.record['rgb']['stamp_s']
    maps = dict(available=False,reason=status.get('maps','not_provided'))
    if spatial_view is not None:
        maps = current(spatial_view.record,'view_id',spatial_view.record['source_stamp_s'])
        maps['local_source'] = spatial_view.record['local_source']
    return dict(rgb=rgb,maps=maps,reference_map_available=reference is not None,
                historical_keyframe_ids=list(keyframes),historical_images_are_current=False)
