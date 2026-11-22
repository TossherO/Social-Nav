"""Common waypoint checks and measured progress; never dispatches motion."""
import copy
import math
import numpy as np


def route_replacement_evidence(stage):
    """Return the withdrawn route's confirmed obstruction, with its original observation time."""
    selected=stage.get('selected') or {}
    blockage=stage.get('recovery_blockage') or {}
    if not selected.get('route') or not selected.get('handle') or blockage.get('handle')!=selected['handle']:
        return None
    if (blockage.get('kind')=='route_blocked' or
            blockage.get('kind')=='destination_occupied' and blockage.get('current') is True):
        return blockage
    return None


def record_route_visit(route, visits, handle, state, index, tolerance):
    """Record one ordered waypoint from a fresh, measured pose."""
    points=route['waypoints']
    if index in route.get('checkpoint_waypoint_indices',()):
        tolerance=min(tolerance,.2)
    pose=state.get('origin_pose')
    stamp=state.get('origin_stamp_s')
    now=state.get('stamp_s')
    if (index != len(visits) or not 0<=index<len(points) or state.get('input_stale') or
            state.get('localization_status','valid')!='valid' or
            type(stamp) not in (int,float) or type(now) not in (int,float) or
            not math.isfinite(stamp) or not math.isfinite(now) or not 0<=now-stamp<=.5 or
            visits and stamp<=visits[-1]['stamp_s'] or
            np.shape(pose)!=(3,) or not np.isfinite(pose).all() or
            np.linalg.norm(np.asarray(pose[:2])-points[index])>tolerance):
        return False
    visits.append(dict(index=index,waypoint=copy.deepcopy(points[index]),pose=copy.deepcopy(pose),
        stamp_s=stamp,frame_id=state.get('origin_frame_id'),clock_domain=state.get('clock_domain'),
        local_identity=[state.get(k) for k in ('session_id','generation','frame_id')],
        map_binding=copy.deepcopy(state.get('map_binding')),handle=copy.deepcopy(handle),
        source=state.get('source')))
    return True


def validate_route_progress(route, proof):
    """Verify every saved waypoint measurement belongs to this completed route."""
    if not isinstance(route,dict) or not isinstance(route.get('waypoints'),list):
        return dict(status='unproven',reason='route_definition_missing')
    points=route['waypoints'];visits=proof.get('route_visits')
    if not isinstance(visits,list) or len(visits)!=len(points):
        return dict(status='unproven',reason='ordered_waypoint_evidence_incomplete')
    context=proof.get('context') or {};last=None
    end_stamp=proof.get('stamp_s')
    if (not points or not isinstance(proof.get('handle'),dict) or not proof.get('frame_id') or
            not proof.get('source') or not context.get('clock_domain') or
            not isinstance(context.get('local_identity'),list) or
            type(end_stamp) not in (int,float) or not math.isfinite(end_stamp)):
        return dict(status='unproven',reason='route_evidence_context_invalid')
    for index,(point,visit) in enumerate(zip(points,visits)):
        if not isinstance(visit,dict):
            return dict(status='unproven',reason='waypoint_evidence_invalid',index=index)
        stamp=visit.get('stamp_s');pose=visit.get('pose')
        tolerance=.2 if index in (0,len(points)-1) or index in route.get('checkpoint_waypoint_indices',()) else .5
        if (visit.get('index')!=index or visit.get('handle')!=proof.get('handle') or
                visit.get('waypoint')!=point or visit.get('frame_id')!=proof.get('frame_id') or
                visit.get('source')!=proof.get('source') or
                visit.get('clock_domain')!=context.get('clock_domain') or
                visit.get('local_identity')!=context.get('local_identity') or
                visit.get('map_binding')!=context.get('map_binding') or
                type(stamp) not in (int,float) or not math.isfinite(stamp) or
                last is not None and stamp<=last or stamp>end_stamp or
                np.shape(pose)!=(3,) or not np.isfinite(pose).all() or
                np.linalg.norm(np.asarray(pose[:2])-point)>tolerance):
            return dict(status='unproven',reason='waypoint_evidence_invalid',index=index)
        last=stamp
    return dict(status='met',waypoint_count=len(visits),visits=copy.deepcopy(visits),
                source='ordered_measured_waypoint_visits_not_semantic_completion')


def route_attempt_evidence(selected, state, trajectory, blockage):
    """Preserve measurements from a retired call under its original handle."""
    handle=selected['handle'];route=selected['route']
    start=selected.get('route_dispatched_stamp_s')
    identity=[state.get(key) for key in ('session_id','generation','frame_id')]
    visits=[]
    observed_visits=(state.get('route_visits') if state.get('handle')==handle else None)
    if blockage.get('handle')==handle:
        for index,visit in enumerate(observed_visits or blockage.get('route_visits') or []):
            stamp=visit.get('stamp_s')
            pose=visit.get('pose')
            tolerance=.2 if index in (0,len(route['waypoints'])-1) or index in route.get('checkpoint_waypoint_indices',()) else .5
            if (index>=len(route['waypoints']) or visit.get('index')!=index or
                    visit.get('handle')!=handle or visit.get('waypoint')!=route['waypoints'][index] or
                    visit.get('frame_id')!=selected['frame_id'] or
                    visit.get('source')!=state.get('source') or visit.get('clock_domain')!=state.get('clock_domain') or
                    visit.get('local_identity')!=identity or visit.get('map_binding')!=state.get('map_binding') or
                    np.shape(pose)!=(3,) or not np.isfinite(pose).all() or
                    math.dist(pose[:2],route['waypoints'][index])>tolerance or
                    type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp>state['stamp_s'] or
                    start is not None and stamp<start or visits and stamp<=visits[-1]['stamp_s']):
                break
            visits.append(copy.deepcopy(visit))
    measured=[]
    if start is not None:
        for segment in trajectory:
            if segment['local_identity']!=identity or segment['map_binding']!=state.get('map_binding'):
                continue
            samples=[sample for sample in segment['samples'] if start<sample['stamp_s']<=state['stamp_s'] and
                     sample.get('source')==state.get('source')]
            if samples:
                measured.append(dict(segment,samples=samples))
    return dict(handle=copy.deepcopy(handle),route=copy.deepcopy(route),reason=blockage['reason'],
                failed_index=blockage.get('route_index'),blocked_position=copy.deepcopy(blockage.get('blocked_position')),
                source_stamp_s=blockage.get('source_stamp_s'),visits=visits,measured_trajectory=measured)


def repeated_route_segment(points, origin, attempts, tolerance):
    """Reject a previously blocked plan or a direct return to the latest failed target."""
    for attempt in attempts:
        old=attempt['route']['waypoints']
        if len(points)==len(old) and all(math.dist(point,prior)<=tolerance for point,prior in zip(points,old)):
            return True
    latest=attempts[-1]
    old=latest['route']['waypoints']
    index=latest['failed_index']
    if type(index) is not int or not 0<=index<len(old):
        return False
    remaining=old[index:]
    if len(points)==len(remaining) and all(math.dist(point,prior)<=tolerance for point,prior in zip(points,remaining)):
        return True
    target=next((point for point in points if math.dist(point,origin)>tolerance),points[-1])
    return math.dist(target,old[index])<=tolerance


def visited_checkpoints(attempts, checkpoints, checkpoint_met, *, context, handle, current_visits=()):
    """Count ordered checkpoints from retired calls and, if supplied, the current call."""
    batches=[]
    for attempt in attempts:
        old=attempt['handle']
        if handle is None or (current_visits and old==handle):
            continue
        if all(old.get(k)==handle.get(k) for k in ('task_id','plan_revision','goal_id','task_goal_binding','catalog_fingerprint')):
            batches.append((attempt['visits'],len(checkpoints)-1))
    batches.append((current_visits,len(checkpoints)))
    index=0
    previous_stamp=None
    for visits,limit in batches:
        for visit in visits:
            if any(visit.get(k)!=context[k] for k in ('clock_domain','local_identity','map_binding')):
                continue
            stamp=visit['stamp_s']
            if previous_stamp is not None and stamp<=previous_stamp:
                continue
            previous_stamp=stamp
            if index<limit and checkpoint_met(checkpoints[index],visit['pose'][:2]):
                index+=1
    return index


def reference_crosses_blockage(path, position, tolerance):
    """Test the planner's actual base-reference polyline against a known position."""
    path=np.asarray(path,dtype=float);point=np.asarray(position,dtype=float)
    if (path.ndim!=2 or path.shape[0]<2 or path.shape[1]<2 or not np.isfinite(path).all() or
            point.shape!=(2,) or not np.isfinite(point).all()):
        return None
    xy=path[:,:2]
    offsets=point-xy[:-1]
    segments=np.diff(xy,axis=0)
    lengths=(segments*segments).sum(axis=1)
    projections=np.divide((offsets*segments).sum(axis=1),lengths,out=np.zeros_like(lengths),where=lengths>0)
    nearest=xy[:-1]+np.clip(projections,0.,1.)[:,None]*segments
    return bool(np.any(np.linalg.norm(nearest-point,axis=1)<=tolerance))


def waypoint_occupancy_diagnostics(inputs, points, margin):
    """Check waypoint cells and yaw-independent necessary footprint, not route segments."""
    from ..tools.spatial import sample,se2
    from navigation.planning.footprint import goal_obstruction
    grid=inputs['global_grid'];points=np.asarray(points,dtype=float)
    values,valid=sample(grid,points);geometry=inputs.get('geometry')
    local=transform=None
    scene=inputs.get('scene')
    if scene is not None and scene.record['geometry']['status']=='valid':
        local=scene.geometry
        if local.metadata['frame_id']==grid.metadata['frame_id']:
            transform=np.eye(3)
        elif inputs.get('pose') is not None and inputs.get('local_pose') is not None:
            transform=se2(inputs['pose'])@np.linalg.inv(se2(inputs['local_pose']))
    elif inputs.get('local_grid') is not None:
        matrix=np.asarray(inputs.get('local_grid_map_from_local'),dtype=float)
        if matrix.shape==(4,4) and np.isfinite(matrix).all():
            local=inputs['local_grid'];transform=matrix[np.ix_([0,1,3],[0,1,3])]
    local_points=local_values=local_valid=None
    if local is not None and transform is not None:
        try:
            inverse=np.linalg.inv(transform)
            local_points=points@inverse[:2,:2].T+inverse[:2,2]
            local_values,local_valid=sample(local,local_points)
        except np.linalg.LinAlgError:
            local=local_points=None
    names={0:'free',100:'occupied',-1:'unknown'}
    rows=[]
    for i,point in enumerate(points):
        static=names.get(int(values[i]),'unknown') if valid[i] else 'outside'
        static_body=goal_obstruction(grid,geometry,point,margin) if geometry is not None else None
        current='unavailable'
        current_body=None
        if local_points is not None:
            current=names.get(int(local_values[i]),'unknown') if local_valid[i] else 'not_covered'
            if local_valid[i] and geometry is not None:
                current_body=goal_obstruction(local,geometry,local_points[i],margin)
        local_clear=(current=='free' and geometry is not None and current_body is None)
        if static=='outside' or static_body=='outside' and not local_clear:reason='waypoint_outside_map'
        elif current=='occupied' or current_body=='occupied':reason='waypoint_currently_occupied'
        elif (static=='occupied' or static_body=='occupied') and not local_clear:reason='waypoint_statically_occupied'
        elif (static=='unknown' or static_body=='unknown') and not local_clear:reason='waypoint_unknown'
        else:reason=None
        rows.append(dict(index=i,position=point.tolist(),static_cell=static,local_cell=current,
                         static_necessary_footprint=static_body or 'not_proven_blocked',
                         local_necessary_footprint=('unavailable' if geometry is None or current in ('unavailable','not_covered')
                                                    else current_body or 'not_proven_blocked'),
                         prior_override_by_current=bool(local_clear and (static=='occupied' or static_body=='occupied')),
                         reason=reason))
    return dict(waypoints=rows,scope='waypoint_only_not_full_footprint_or_planned_path',
                full_footprint_checked=False,route_checked=False)
