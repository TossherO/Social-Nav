"""Small, observation-bound search poses after losing a follow target."""
from functools import lru_cache
import math
import numpy as np

from navigation.planning.footprint import swept_check
from perception.environment.context import fingerprint


def _se2(pose):
    c,s=math.cos(pose[2]),math.sin(pose[2])
    return np.array([[c,-s,pose[0]],[s,c,pose[1]],[0.,0.,1.]])


def _inputs(inputs, *, now, clock_domain):
    stamp=inputs.get('stamp_s');pose=np.asarray(inputs.get('pose'),dtype=float)
    if (inputs.get('clock_domain')!=clock_domain or type(stamp) not in (int,float) or
            not 0<=now-stamp<=.5 or pose.shape!=(3,) or not np.isfinite(pose).all()):
        raise ValueError('search_current_pose_or_observation_unavailable')
    geometry=inputs.get('geometry');global_grid=inputs.get('global_grid');scene=inputs.get('scene')
    if geometry is None or global_grid is None:raise ValueError('search_geometry_or_map_unavailable')
    if scene is not None:
        scene.verify();local=scene.geometry
        if (scene.record['as_of']!=stamp or scene.record['clock_domain']!=clock_domain or
                scene.record['geometry']['status']!='valid' or scene.record['localization']['status']!='valid'):
            raise ValueError('search_scene_source_mismatch')
        local_pose=inputs.get('local_pose')
        if local.metadata['frame_id']==global_grid.metadata['frame_id']:transform=np.eye(3)
        elif local_pose is not None:transform=_se2(pose)@np.linalg.inv(_se2(local_pose))
        else:raise ValueError('search_local_transform_unavailable')
    else:
        local=inputs.get('local_grid');matrix=np.asarray(inputs.get('local_grid_map_from_local'),dtype=float)
        if local is None or matrix.shape!=(4,4) or not np.isfinite(matrix).all():
            raise ValueError('search_local_grid_unavailable')
        local_pose=np.asarray(inputs.get('local_pose'),dtype=float)
        if local_pose.shape!=(3,) or not np.isfinite(local_pose).all():
            raise ValueError('search_local_transform_unavailable')
        transform=_se2(pose)@np.linalg.inv(_se2(local_pose))
    return pose,geometry,global_grid,local,transform


def check_pose(inputs, target, *, now, clock_domain, anchor, margin=.05, radius_m=1.5):
    """Check the short VLM search pose against current and mapped geometry."""
    pose,geometry,grid,local,transform=_inputs(inputs,now=now,clock_domain=clock_domain)
    target=np.asarray(target,dtype=float);anchor=np.asarray(anchor,dtype=float)
    if (target.shape!=(3,) or anchor.shape!=(2,) or not np.isfinite(target).all() or
            not np.isfinite(anchor).all() or np.linalg.norm(target[:2]-anchor)>radius_m+1e-9 or
            np.linalg.norm(pose[:2]-anchor)>radius_m+1e-9 or np.linalg.norm(target[:2]-pose[:2])>.8+1e-9):
        raise ValueError('search_pose_outside_budget')
    delta=target[:2]-pose[:2]
    distance=np.linalg.norm(delta)
    if distance>1e-8 and delta@np.array([math.cos(pose[2]),math.sin(pose[2])])<=0:
        raise ValueError('search_approach_not_ahead')
    inverse=np.linalg.inv(transform);yaw=math.atan2(transform[1,0],transform[0,0])
    def local_pose(p):
        xy=inverse@np.array([p[0],p[1],1.])
        return [float(xy[0]),float(xy[1]),float(p[2]-yaw)]
    mapped=np.stack([pose,target]);local_path=np.array([local_pose(pose),local_pose(target)])
    static=swept_check(grid,geometry,mapped,margin,512,prefix=False)
    observed=swept_check(local,geometry,local_path,margin,512,prefix=False)
    return dict(valid=static['status']=='free' and observed['status']=='free',
                static_status=static['status'],local_status=observed['status'],
                source_stamp_s=inputs['stamp_s'],clock_domain=clock_domain,
                scope='current_swept_geometry_not_dynamic_or_path_proof')


def current_search_pose(choice, pose):
    """Refresh a selected search action from the current vehicle pose."""
    target=np.asarray(choice.get('pose'),dtype=float)
    pose=np.asarray(pose,dtype=float)
    if target.shape!=(3,) or pose.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(pose).all():
        raise ValueError('search_candidate_pose_invalid')
    if choice['kind']=='rotate':
        return [*pose[:2].tolist(),float(target[2])]
    if choice['kind']!='approach':
        raise ValueError('search_candidate_kind_invalid')
    front=np.array([math.cos(pose[2]),math.sin(pose[2])])
    if (target[:2]-pose[:2])@front<=0:
        raise ValueError('search_approach_already_passed')
    destination=np.asarray(choice.get('last_observed_pose',target),dtype=float)
    if destination.shape!=(3,) or not np.isfinite(destination).all():
        raise ValueError('search_last_observation_pose_invalid')
    delta=destination[:2]-pose[:2]
    distance=float(np.linalg.norm(delta))
    if distance<.2 or delta@front<=0:
        raise ValueError('search_destination_not_ahead')
    endpoint=pose[:2]+delta*(min(distance,.8)/distance)
    yaw=destination[2] if distance<=.8 else math.atan2(delta[1],delta[0])
    return [*endpoint.tolist(),float(yaw)]


def _last_point(last, *, now, clock_domain, local_identity, transform):
    if (isinstance(last,dict) and last.get('binding')==fingerprint({k:v for k,v in last.items() if k!='binding'}) and
            last.get('clock_domain')==clock_domain and last.get('local_identity')==local_identity and
            type(last.get('stamp_s')) in (int,float) and 0<=now-last['stamp_s']<=45.):
        xy=np.asarray(last.get('position_xy_m'),dtype=float)
        if xy.shape==(2,) and np.isfinite(xy).all():
            return (transform@np.r_[xy,1.])[:2]
    return None


@lru_cache(maxsize=1)
def _arrival_tolerances():
    from navigation.control.tracking import load_config
    tracking = load_config()
    return tracking['goal_position_tolerance_m'], tracking['goal_yaw_tolerance_rad']


def check_last_position(scene, asset, geometry, target, planning, *, now, clock_domain):
    """Check current occupancy and the requested endpoint before mapped planning."""
    from navigation.planning.continuity import context_binding
    from navigation.planning.footprint import footprint_status, wrap
    from navigation.planning.inputs import association_status, scene_status
    from navigation.planning.planner import goal_cell_issue
    from perception.environment.navigation import navigation_geometry

    scene.verify()
    reason = scene_status(scene, now, clock_domain, planning) or association_status(scene, asset, planning)
    if reason:
        raise ValueError(reason)
    point = np.asarray(target, dtype=float)
    if point.shape not in ((2,), (3,)) or not np.isfinite(point).all():
        raise ValueError('follow_last_position_invalid')
    record = scene.record
    if not 0 <= now-record['prior']['measurement_stamp_s'] <= planning['max_association_age_s']:
        raise ValueError('association_future_or_stale')
    local_from_lidar = np.asarray(record['localization']['world_from_lidar'])
    local_pose = geometry.pose(local_from_lidar)
    map_pose = geometry.pose(np.asarray(record['prior']['map_from_local']) @ local_from_lidar)
    xy_tolerance, yaw_tolerance = _arrival_tolerances()
    reached_xy = np.linalg.norm(point[:2]-map_pose[:2]) <= xy_tolerance
    aligned = len(point) == 2 or abs(wrap(point[2]-map_pose[2])) <= yaw_tolerance
    if reached_xy and aligned:
        raise ValueError('follow_last_position_reached')
    margin = planning['safety_margin_m']
    reason = goal_cell_issue(asset.snapshot, geometry, point, margin)
    if reason:
        raise ValueError(reason)
    if len(point) == 3 and footprint_status(asset.snapshot, geometry, point, margin) != 'free':
        raise ValueError('follow_last_position_map_footprint_not_free')
    local = navigation_geometry(scene, now=now, asset=asset, check_asset=True)
    if footprint_status(local, geometry, local_pose, margin) != 'free':
        raise ValueError('follow_last_position_start_not_free')
    local_from_map = _se2(local_pose) @ np.linalg.inv(_se2(map_pose))
    local_target = (local_from_map @ np.r_[point[:2], 1.])[:2].tolist()
    reason = goal_cell_issue(local, geometry, local_target, margin)
    if len(point) == 3:
        yaw = point[2]+local_pose[2]-map_pose[2]
        if footprint_status(local, geometry, [*local_target, yaw], margin) == 'occupied':
            raise ValueError('follow_last_position_currently_occupied')
    if reason in ('goal_occupied', 'goal_footprint_occupied'):
        raise ValueError('follow_last_position_currently_occupied')
    return dict(valid=True, source_stamp_s=record['as_of'], clock_domain=clock_domain,
                map_context=context_binding(dict(mode='mapped'), scene, asset),
                scope='mapped_goal_and_current_footprint')


def last_position_candidate(scene, asset, geometry, last, planning, continuation=None, loss=None):
    record = scene.record
    stamp = record['as_of']
    clock = record['clock_domain']
    identity = [record[k] for k in ('session_id', 'generation', 'frame_id')]
    local_from_lidar = np.asarray(record['localization']['world_from_lidar'])
    map_pose = geometry.pose(np.asarray(record['prior']['map_from_local']) @ local_from_lidar)
    local_pose = geometry.pose(local_from_lidar)
    transform = _se2(map_pose) @ np.linalg.inv(_se2(local_pose))
    point = _last_point(last, now=stamp, clock_domain=clock, local_identity=identity, transform=transform)
    if point is None or continuation is not None and stamp >= continuation['expires_s']:
        return None
    target = point.tolist() if continuation is None else continuation['target']
    departure_yaw = None if loss is None else loss.get('departure_yaw_rad')
    if continuation is None and departure_yaw is not None:
        yaw = departure_yaw + map_pose[2] - local_pose[2]
        target.append(math.atan2(math.sin(yaw), math.cos(yaw)))
    check = check_last_position(scene, asset, geometry, target, planning, now=stamp, clock_domain=clock)
    result = dict(id=1, kind='approach', pose=target, check=check, last_evidence_binding=last['binding'])
    if continuation is not None:
        result['continuation_id'] = continuation['intent_id']
    return result


def candidates(inputs, last, *, now, clock_domain, local_identity, anchor, margin=.05):
    pose,_,_,_,transform=_inputs(inputs,now=now,clock_domain=clock_domain)
    anchor=np.asarray(anchor,dtype=float);headings=[]
    point=_last_point(last,now=now,clock_domain=clock_domain,local_identity=local_identity,transform=transform)
    if point is not None:
        delta=point-pose[:2]
        if np.linalg.norm(delta)>.05:headings.append(math.atan2(delta[1],delta[0]))
    results=[dict(id=0,kind='wait',pose=None)]
    headings.extend([pose[2]+math.pi/4,pose[2]-math.pi/4])
    proposed=[]
    for heading in headings:
        heading=math.atan2(math.sin(heading),math.cos(heading))
        if abs(math.atan2(math.sin(heading-pose[2]),math.cos(heading-pose[2])))<.1:continue
        if any(abs(math.atan2(math.sin(heading-p[2]),math.cos(heading-p[2])))<.1 for p in proposed):continue
        proposed.append([float(pose[0]),float(pose[1]),heading])
    destination=None
    if point is not None:
        delta=point-pose[:2];distance=float(np.linalg.norm(delta));step=min(.8,distance)
        direction=np.asarray(last.get('velocity_xy_mps'),dtype=float)
        if direction.shape==(2,) and np.isfinite(direction).all() and np.linalg.norm(direction)>=.15:
            mapped_direction=transform[:2,:2]@direction
            yaw=math.atan2(mapped_direction[1],mapped_direction[0])
        else:
            yaw=math.atan2(delta[1],delta[0]) if distance>.05 else float(pose[2])
        destination=[float(point[0]),float(point[1]),yaw]
        if step>=.2:
            end=pose[:2]+delta/distance*step
            proposed.append([float(end[0]),float(end[1]),yaw if distance<=.8 else math.atan2(delta[1],delta[0])])
    for candidate in proposed:
        try:check=check_pose(inputs,candidate,now=now,clock_domain=clock_domain,anchor=anchor,margin=margin)
        except ValueError:continue
        if check['valid']:
            kind='rotate' if np.linalg.norm(np.asarray(candidate[:2])-pose[:2])<1e-8 else 'approach'
            item=dict(id=len(results),kind=kind,pose=candidate,check=check)
            if kind=='approach' and destination is not None:item['last_observed_pose']=destination
            results.append(item)
    return results


def follow_candidates(scene, asset, geometry, last=None, *, anchor=None, margin=.05,
                      continuation=None, local_reid=False, planning=None, loss=None):
    """Build search choices or check the existing fixed target against this scene."""
    from perception.environment.context import fingerprint
    from navigation.planning.inputs import association_status, load_config
    scene.verify();record=scene.record;stamp=record['as_of']
    identity=[record[k] for k in ('session_id','generation','frame_id')]
    results=[dict(id=0,kind='wait',pose=None)]
    planning = load_config() if planning is None else planning
    if local_reid:
        if asset is not None and not association_status(scene, asset, planning):
            try:
                candidate = last_position_candidate(scene, asset, geometry, last, planning, continuation, loss)
            except (ValueError, KeyError, TypeError):
                candidate = None
            if candidate is not None:
                results.append(candidate)
    elif asset is not None and not association_status(scene,asset,planning):
        matrix=np.asarray(record['localization'].get('world_from_lidar'),dtype=float)
        transform=np.asarray(record['prior'].get('map_from_local'),dtype=float)
        if matrix.shape==(4,4) and transform.shape==(4,4) and np.isfinite(matrix).all() and np.isfinite(transform).all():
            local_pose=geometry.pose(matrix);map_pose=geometry.pose(transform@matrix)
            inputs=dict(stamp_s=stamp,clock_domain=record['clock_domain'],scene=scene,geometry=geometry,
                        global_grid=asset.snapshot,local_pose=local_pose,pose=map_pose)
            search_anchor=map_pose[:2] if anchor is None else anchor
            try:
                results=candidates(inputs,last,now=stamp,clock_domain=record['clock_domain'],
                    local_identity=identity,anchor=search_anchor,margin=margin)
            except (ValueError,KeyError,TypeError):
                pass
    value=dict(clock_domain=record['clock_domain'],source_stamp_s=stamp,scene_binding=record['input_fingerprint'],
               candidates=results)
    value['binding']=fingerprint(value)
    return value


def bounds_status(request, scene, asset, planning, poses, *, margin,geometry):
    """Check the actual proposed motion remains inside the mapped search area."""
    from navigation.planning.inputs import association_status
    from perception.detection_tracking.inference.geometry import validate_transform
    bounds=request.get('search_bounds')
    if bounds is None:return None
    if (not isinstance(bounds,dict) or set(bounds)!={'anchor_xy_m','radius_m','map_snapshot_id','loss_binding'} or
            request.get('mode')!='mapped' or asset is None or bounds.get('map_snapshot_id')!=request.get('map_snapshot_id') or
            bounds.get('map_snapshot_id')!=asset.manifest['snapshot_id'] or not isinstance(bounds.get('loss_binding'),str) or
            not bounds['loss_binding'] or type(bounds.get('radius_m')) not in (int,float) or
            not 0<bounds['radius_m']<=1.5):
        return 'follow_search_bounds_invalid'
    anchor=np.asarray(bounds['anchor_xy_m'],dtype=float);poses=np.asarray(poses,dtype=float)
    if (anchor.shape!=(2,) or not np.isfinite(anchor).all() or poses.ndim!=2 or poses.shape[1]!=3 or
            not np.isfinite(poses).all() or not np.isfinite(margin) or margin<0):
        return 'follow_search_bounds_invalid'
    if association_status(scene,asset,planning):return 'follow_search_map_association_invalid'
    transform=validate_transform(scene.record['prior']['map_from_local'])
    local_from_lidar=validate_transform(scene.record['localization']['world_from_lidar'])
    local_pose=geometry.pose(local_from_lidar);map_pose=geometry.pose(transform@local_from_lidar)
    planar=_se2(map_pose)@np.linalg.inv(_se2(local_pose))
    mapped=poses[:,:2]@planar[:2,:2].T+planar[:2,2]
    if np.any(np.linalg.norm(mapped-anchor,axis=1)+margin>bounds['radius_m']):
        return 'follow_search_radius_exceeded'
    return None
