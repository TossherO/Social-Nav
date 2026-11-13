"""Small, observation-bound search poses after losing a follow target."""
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
    pose,geometry,grid,local,transform=_inputs(inputs,now=now,clock_domain=clock_domain)
    target=np.asarray(target,dtype=float);anchor=np.asarray(anchor,dtype=float)
    if (target.shape!=(3,) or anchor.shape!=(2,) or not np.isfinite(target).all() or
            not np.isfinite(anchor).all() or np.linalg.norm(target[:2]-anchor)>radius_m+1e-9 or
            np.linalg.norm(pose[:2]-anchor)>radius_m+1e-9 or np.linalg.norm(target[:2]-pose[:2])>.8+1e-9):
        raise ValueError('search_pose_outside_budget')
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


def candidates(inputs, last, *, now, clock_domain, local_identity, anchor, margin=.05):
    pose,_,_,_,transform=_inputs(inputs,now=now,clock_domain=clock_domain)
    anchor=np.asarray(anchor,dtype=float);headings=[];point=None
    if (isinstance(last,dict) and last.get('binding')==fingerprint({k:v for k,v in last.items() if k!='binding'}) and
            last.get('clock_domain')==clock_domain and last.get('local_identity')==local_identity and
            type(last.get('stamp_s')) in (int,float) and 0<=now-last['stamp_s']<=45.):
        xy=np.asarray(last.get('position_xy_m'),dtype=float)
        if xy.shape==(2,) and np.isfinite(xy).all():
            point=(transform@np.r_[xy,1.])[:2]
            if np.linalg.norm(point-pose[:2])>.05:headings.append(math.atan2(point[1]-pose[1],point[0]-pose[0]))
    headings.extend([pose[2]+math.pi/4,pose[2]-math.pi/4])
    proposed=[]
    for heading in headings:
        heading=math.atan2(math.sin(heading),math.cos(heading))
        if abs(math.atan2(math.sin(heading-pose[2]),math.cos(heading-pose[2])))<.1:continue
        if any(abs(math.atan2(math.sin(heading-p[2]),math.cos(heading-p[2])))<.1 for p in proposed):continue
        proposed.append([float(pose[0]),float(pose[1]),heading])
    if point is not None:
        delta=point-pose[:2];distance=float(np.linalg.norm(delta));step=min(.8,max(0.,distance-1.5))
        if step>=.2:
            end=pose[:2]+delta/distance*step
            proposed.append([float(end[0]),float(end[1]),math.atan2(delta[1],delta[0])])
    results=[dict(id=0,kind='wait',pose=None)]
    for candidate in proposed:
        try:check=check_pose(inputs,candidate,now=now,clock_domain=clock_domain,anchor=anchor,margin=margin)
        except ValueError:continue
        if check['valid']:
            results.append(dict(id=len(results),kind='rotate' if np.linalg.norm(np.asarray(candidate[:2])-pose[:2])<1e-8 else 'approach',
                                pose=candidate,check=check))
    return results


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
