"""Bounded reuse of old geometry as a reference or initialization, never feedback."""
import numpy as np
from perception.environment.context import fingerprint
from .motion import identity


def local_goal(request,scene,geometry,asset=None,config=None):
    """Goal in the continuous planar local frame, using the current map anchor."""
    goal=np.asarray(request['goal'],float)
    if goal.shape not in ((2,),(3,)) or not np.isfinite(goal).all():raise ValueError('invalid_goal')
    r=scene.record
    if request['mode']=='local':
        if request['frame_id']!=r['frame_id']:raise ValueError('goal_frame_mismatch')
        return goal.copy()
    from .inputs import association_status,load_config
    from .planner import se2,transform_poses
    if request['mode']!='mapped':raise ValueError('unsupported_live_goal_mode')
    reason=association_status(scene,asset,config if config is not None else load_config())
    if reason:raise ValueError(reason)
    if request['frame_id']!=asset.snapshot.metadata['frame_id'] or request.get('map_snapshot_id')!=asset.manifest['snapshot_id']:
        raise ValueError('goal_map_identity_mismatch')
    world=np.asarray(r['localization']['world_from_lidar'])
    pose=geometry.pose(world);mapped=geometry.pose(np.asarray(r['prior']['map_from_local'])@world)
    out=transform_poses(np.array([[*goal[:2],goal[2] if len(goal)==3 else 0.]]),se2(pose)@np.linalg.inv(se2(mapped)))[0]
    return out[:len(goal)]


def goal_binding(request):
    value={k:request.get(k) for k in ('task_id','goal_revision','goal','mode','frame_id',
        'map_snapshot_id','target_reference','motion_intent','algorithm')}
    value['goal_behavior']=request.get('goal_behavior','pass_through')
    value['completion_policy']=request.get('completion_policy','goal')
    return fingerprint(value)


def context_binding(request,scene,asset):
    r=scene.record;a=r['prior']
    lineage=None if request['mode']=='local' else {k:a.get(k) for k in (
        'valid','map_id','asset_fingerprint','map_frame','association_session',
        'association_generation','map_revision','coverage_fingerprint')}
    static=None if asset is None else {k:asset.manifest[k] for k in ('snapshot_id','layers_sha256')}
    return fingerprint([goal_binding(request),identity(scene),r['clock_domain'],static,lineage])


def content_valid(result):
    return result.get('result_binding')==fingerprint({k:v for k,v in result.items() if k!='result_binding'})


def reference_envelope(result,request,scene,asset):
    if not content_valid(result):raise ValueError('reference_result_binding_mismatch')
    out=dict(result=result,context_binding=context_binding(request,scene,asset),
        as_of=scene.record['as_of'],clock_domain=scene.record['clock_domain'])
    out['binding']=fingerprint(out)
    return out


def check_reference(envelope,request,scene,asset,max_age,*,require_path=True):
    if envelope.get('binding')!=fingerprint({k:v for k,v in envelope.items() if k!='binding'}):
        raise ValueError('reference_envelope_binding_mismatch')
    result=envelope['result'];source=result['input_binding'];r=scene.record
    if not content_valid(result) or envelope['context_binding']!=context_binding(request,scene,asset):
        raise ValueError('reference_context_mismatch')
    if goal_binding(result['request'])!=goal_binding(request) or result.get('frame_id')!=request['frame_id']:
        raise ValueError('reference_goal_mismatch')
    if (envelope['clock_domain']!=r['clock_domain'] or source['clock_domain']!=r['clock_domain'] or
            source['as_of']!=envelope['as_of'] or not 0<=r['as_of']-envelope['as_of']<=max_age):
        raise ValueError('reference_future_or_stale')
    if require_path and (result['status']!='path_found' or len(result.get('path',[]))<2):
        raise ValueError('reference_has_no_path')
