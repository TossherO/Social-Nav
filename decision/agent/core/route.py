"""Common waypoint checks and measured progress; never dispatches motion."""
import copy
import math
import numpy as np


def record_route_visit(route, visits, handle, state, index, tolerance):
    """Record one ordered waypoint from a fresh, measured pose."""
    points=route['waypoints']
    pose=state.get('origin_pose')
    stamp=state.get('origin_stamp_s',state.get('stamp_s'))
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
        stamp=visit.get('stamp_s');pose=visit.get('pose');tolerance=.2 if index in (0,len(points)-1) else .5
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
