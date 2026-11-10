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


def circuit_plan_status(points,polygon,direction,position_tolerance):
    points=np.asarray(points,dtype=float)
    if not path_outside_region(points,polygon):return 'proposed_route_intersects_task_region'
    if np.linalg.norm(points[-1]-points[0])>2*position_tolerance:return 'circuit_route_must_return_to_its_start'
    center=centroid(polygon)
    if not inside(center,polygon):return 'region_centroid_outside_region'
    angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
    change=np.arctan2(np.sin(np.diff(angles)),np.cos(np.diff(angles)))
    sign=-1 if direction=='clockwise' else 1
    if not 2*math.pi-.2 <= sign*float(change.sum()) <= 2*math.pi+.2:
        return 'proposed_route_does_not_make_requested_circuit'
    return None


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
