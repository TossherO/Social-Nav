"""Read-only scene and map query helpers; no goal execution or robot control."""
from perception.environment.context import json_safe
from perception.environment.layers import load_config


def query_scene(bundle, request, *, settings=None):
    """Query one unified scene; scene identity also binds RGB, people and optional static geometry."""
    from perception.environment.context import query_scene as query_facts
    bundle.verify()
    result = query_facts(bundle,request,settings or load_config()['consumer'])
    return dict(contract='read_only_scene_facts',data=result,navigation_reachability_verified=False)


def query_environment(package, request, *, settings=None):
    settings = settings or load_config()['consumer']
    allowed = {'operation','snapshot_id','frame','bounds','layers','points'}
    if not isinstance(request,dict) or set(request)-allowed:
        raise ValueError('Unsupported environment tool arguments')
    snapshot = package.get_snapshot(request['snapshot_id'])
    operation = request.get('operation')
    if operation == 'get_snapshot':
        data = snapshot.metadata
    elif operation == 'query_geometry':
        if len(request['points']) > 256:
            raise ValueError('Too many geometry query points')
        data = package.query_geometry(request['points'],frame=request['frame'])
    elif operation == 'crop_map':
        data = package.crop_map(request['bounds'],frame=request['frame'],layers=request.get('layers',['occupancy']))
        if any(array.size > settings['tool_crop_cells'] for array in data['layers'].values()):
            raise ValueError('Tool crop exceeds its response budget')
    else:
        raise ValueError('Only explicit read-only environment queries are supported')
    return json_safe(dict(entity_type='map',map_type='static_environment_map',
        snapshot_id=snapshot.metadata['snapshot_id'],frame_id=snapshot.metadata['frame_id'],
        time_scope=package.manifest['time_scope'],source=package.manifest['source'],data=data,
        contract='read_only_environment_facts',navigation_reachability_verified=False))
