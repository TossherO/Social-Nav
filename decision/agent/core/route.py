"""Task-level path constraints and evidence; never dispatches motion."""
import math
import numpy as np
from .tasks import centroid,inside


def _cross(a,b,c):
    return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))


def _touches(a,b,c,d):
    eps=1e-9
    def on(p,q,r):
        return abs(_cross(p,q,r))<=eps and np.dot(r-p,r-q)<=eps
    x,y,z,w=_cross(a,b,c),_cross(a,b,d),_cross(c,d,a),_cross(c,d,b)
    return ((x*y < -eps and z*w < -eps) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b))


def path_outside_region(path, polygon):
    points=np.asarray(path,dtype=float)
    vertices=np.asarray(polygon,dtype=float)
    if (points.ndim!=2 or points.shape[1]<2 or len(points)<1 or not np.isfinite(points).all() or
            vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<3 or not np.isfinite(vertices).all()):
        return False
    points=points[:,:2]
    if any(inside(p,vertices) for p in points):
        return False
    edges=list(zip(vertices,np.roll(vertices,-1,axis=0)))
    return not any(_touches(a,b,c,d) for a,b in zip(points,points[1:]) for c,d in edges)


def circuit_plan_diagnostics(points,polygon,direction,position_tolerance):
    points=np.asarray(points,dtype=float)
    vertices=np.asarray(polygon,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all() or
            vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<3 or not np.isfinite(vertices).all() or
            direction not in ('clockwise','counterclockwise') or not math.isfinite(position_tolerance) or position_tolerance<=0):
        raise ValueError('Invalid circuit plan geometry or direction')
    closed=np.linalg.norm(points[-1]-points[0])<1e-6
    anchors=points[:-1] if closed else points
    if len(anchors)<3 or len({tuple(p) for p in anchors})<3:
        raise ValueError('Circuit requires at least three distinct waypoints')
    ring=np.vstack((anchors,anchors[0]))
    inside_indices=[i for i,p in enumerate(anchors) if inside(p,vertices)]
    center=centroid(polygon)
    valid_center=inside(center,polygon)
    winding=reverse=geometric_winding=None
    actual_direction='undetermined'
    if valid_center:
        angles=np.arctan2(ring[:,1]-center[1],ring[:,0]-center[0])
        change=np.arctan2(np.sin(np.diff(angles)),np.cos(np.diff(angles)))
        sign=-1 if direction=='clockwise' else 1
        geometric_winding=float(change.sum())
        winding=sign*geometric_winding
        reverse=float(np.maximum(-sign*change,0).sum())
        if geometric_winding>=2*math.pi-.2:actual_direction='counterclockwise'
        elif geometric_winding<=-2*math.pi+.2:actual_direction='clockwise'
    issues=[]
    if inside_indices:issues.append('waypoint_inside_region')
    if not valid_center:issues.append('region_centroid_outside')
    if valid_center and not 2*math.pi-.2<=winding<=2*math.pi+.2:
        issues.append('direction_opposite_to_requested' if actual_direction not in ('undetermined',direction)
                      else 'route_does_not_make_requested_circuit')
    if reverse is not None and reverse>math.pi/2:issues.append('excessive_reverse_winding')
    if inside_indices:reason='circuit_waypoint_inside_region'
    elif not valid_center:reason='region_centroid_outside_region'
    elif not 2*math.pi-.2 <= winding <= 2*math.pi+.2:reason='proposed_route_does_not_make_requested_circuit'
    elif reverse>math.pi/2:reason='proposed_route_excessive_reverse_winding'
    else:reason=None
    return dict(reason=reason,inside_waypoint_indices=inside_indices,
                waypoint_index_reference='zero_based',closure_added=not closed,
                requested_direction=direction,actual_direction=actual_direction,geometric_winding_rad=geometric_winding,
                signed_winding_rad=winding,reverse_winding_rad=reverse,issues=issues,
                scope='waypoint_order_only_not_planned_path_or_execution_check')


def circuit_plan_status(points,polygon,direction,position_tolerance):
    return circuit_plan_diagnostics(points,polygon,direction,position_tolerance)['reason']


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


def static_route_line_diagnostics(grid,points):
    """Describe the submitted straight segments; never certify a planned path."""
    from ..tools.spatial import sample
    points=np.asarray(points,dtype=float)
    spacing=min(.05,float(grid.metadata['resolution'])/2)
    if not math.isfinite(spacing) or spacing<=0:raise ValueError('Invalid static grid resolution')
    segments=[]
    for index,(a,b) in enumerate(zip(points,points[1:])):
        length=float(np.linalg.norm(b-a))
        if not math.isfinite(length) or np.abs(np.r_[a,b]).max()>1e8:
            segments.append(dict(segment=index,status='outside_sample_range'))
            continue
        count=min(4097,max(2,math.ceil(length/spacing)+1))
        xy=a[None,:]+np.linspace(0.,1.,count)[:,None]*(b-a)[None,:]
        values,valid=sample(grid,xy)
        segments.append(dict(segment=index,occupied_samples=int(((values==100)&valid).sum()),
            unknown_samples=int(((values==-1)&valid).sum()),outside_samples=int((~valid).sum()),
            sample_step_m=length/(count-1)))
    return dict(segments=segments,scope='static_centerline_only_not_vehicle_or_actual_planned_path')


def sequence_evidence(segments, checkpoints, checkpoint_met, start_stamp, end_stamp):
    if (not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start_stamp,end_stamp)) or
            end_stamp<start_stamp):
        return dict(status='unproven',reason='invalid_sequence_evidence_context')
    groups=[[s for s in segment['samples'] if start_stamp-1e-8<=s['stamp_s']<=end_stamp+1e-8] for segment in segments]
    groups=[group for group in groups if group]
    if len(groups)!=1 or len(groups[0])<2 or groups[0][0]['stamp_s']-start_stamp>.3 or end_stamp-groups[0][-1]['stamp_s']>.3:
        return dict(status='unproven',reason='continuous_measured_path_unavailable')
    samples=groups[0]
    stamps=np.asarray([s['stamp_s'] for s in samples],dtype=float)
    poses=np.asarray([s['pose'] for s in samples],dtype=float)
    if (poses.shape!=(len(samples),3) or not np.isfinite(poses).all() or not np.isfinite(stamps).all() or
            (np.diff(stamps)<=0).any() or (np.diff(stamps)>3.).any()):
        return dict(status='unproven',reason='invalid_or_discontinuous_measured_path')
    index=0;matched=[]
    for stamp,pose in zip(stamps,poses):
        if index<len(checkpoints) and checkpoint_met(checkpoints[index],pose[:2]):
            matched.append(dict(checkpoint=checkpoints[index],stamp_s=float(stamp)))
            index+=1
    return dict(status='met' if index==len(checkpoints) else 'not_met',matched=matched,
                source='measured_localization_not_planned_route')


def circuit_evidence(segments, polygon, direction, start_stamp, end_stamp, position_tolerance):
    """Check one continuous measured trace, not planned vertices or ACKs."""
    if (direction not in ('clockwise','counterclockwise') or
            not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start_stamp,end_stamp,position_tolerance)) or
            end_stamp<start_stamp or position_tolerance<=0):
        return dict(status='unproven',reason='invalid_circuit_evidence_context')
    matching=[]
    for segment in segments:
        samples=[s for s in segment['samples'] if start_stamp-1e-8<=s['stamp_s']<=end_stamp+1e-8]
        if samples:matching.append(samples)
    if len(matching)!=1 or len(matching[0])<3:
        return dict(status='unproven',reason='continuous_measured_path_unavailable')
    samples=matching[0]
    if samples[0]['stamp_s']-start_stamp>.3 or end_stamp-samples[-1]['stamp_s']>.3:
        return dict(status='unproven',reason='measured_path_does_not_cover_circuit')
    stamps=np.asarray([s['stamp_s'] for s in samples],dtype=float)
    poses=np.asarray([s['pose'] for s in samples],dtype=float)
    if (poses.shape!=(len(samples),3) or not np.isfinite(poses).all() or not np.isfinite(stamps).all() or
            (np.diff(stamps)<=0).any() or (np.diff(stamps)>3.).any()):
        return dict(status='unproven',reason='invalid_or_discontinuous_measured_path')
    if not path_outside_region(poses,polygon):
        return dict(status='not_met',reason='measured_path_intersects_region')
    center=centroid(polygon)
    if not inside(center,polygon):
        return dict(status='unproven',reason='region_centroid_outside_region')
    delta=poses[:,:2]-center
    angles=np.arctan2(delta[:,1],delta[:,0])
    changes=np.arctan2(np.sin(np.diff(angles)),np.cos(np.diff(angles)))
    if (np.abs(changes)>math.pi/2).any():
        return dict(status='unproven',reason='angular_sampling_gap')
    sign=-1 if direction=='clockwise' else 1
    winding=float(sign*changes.sum())
    reverse=float(np.maximum(-sign*changes,0).sum())
    closed=bool(np.linalg.norm(poses[-1,:2]-poses[0,:2])<=2*position_tolerance)
    met=closed and 2*math.pi-.2<=winding<=2*math.pi+.2 and reverse<=math.pi/2
    return dict(status='met' if met else 'not_met',winding_rad=winding,reverse_rad=reverse,
                returned_near_start=closed,sample_count=len(samples),source='measured_localization_not_planned_route')
