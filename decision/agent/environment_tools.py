"""Read-only environment tools; no agent loop, goal execution or robot control."""
from perception.environment.context import json_safe
from perception.environment.layers import load_config


def query_scene(bundle, request, *, settings=None):
    """Query one unified scene; scene identity also binds RGB, people and memory."""
    from perception.environment.context import query_scene as query_facts
    bundle.verify()
    result = query_facts(bundle,request,settings or load_config()['consumer'])
    return dict(contract='read_only_scene_facts',data=result,navigation_reachability_verified=False)


def query_environment(package, request, *, settings=None):
    settings = settings or load_config()['consumer']
    allowed = {'operation','snapshot_id','frame','bounds','layers','points','as_of',
               'name','place','position','radius','heading','limit'}
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
    elif operation == 'query_places':
        data = package.query_places(name=request.get('name'),bounds=request.get('bounds'),
                                    as_of=request['as_of'],frame=request['frame'])
    elif operation == 'query_observations':
        limit = request.get('limit',settings['history_limit'])
        if type(limit) is not int or not 1 <= limit <= settings['history_limit']:
            raise ValueError('Tool history limit exceeds its budget')
        data = package.query_observations(as_of=request['as_of'],frame=request['frame'],
            place=request.get('place'),position=request.get('position'),radius=request.get('radius'),
            heading=request.get('heading'),limit=limit)
    else:
        raise ValueError('Only explicit read-only environment queries are supported')
    return json_safe(dict(snapshot_id=snapshot.metadata['snapshot_id'],frame_id=snapshot.metadata['frame_id'],
        time_scope=package.manifest['time_scope'],source=package.manifest['source'],data=data,
        contract='read_only_environment_facts',navigation_reachability_verified=False))
