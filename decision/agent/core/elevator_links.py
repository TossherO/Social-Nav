"""Find candidate elevator annotations without assuming a physical choice."""
import copy
import math

import numpy as np


def candidate_links(links, from_catalog, to_catalog, pose, *, max_distance_m, ambiguity_m):
    """Return plausible corresponding doors, not a vehicle pose or confirmation.

    The caller must validate the pose's map identity, age and origin. A nearby
    labelled door is merely a clue and does not prove which elevator was used.
    """
    source=from_catalog.value['floor_id'];target=to_catalog.value['floor_id']
    if (source==target or not isinstance(links,list) or not links or len(pose)!=3 or
            not all(math.isfinite(float(v)) for v in pose) or
            not math.isfinite(max_distance_m) or max_distance_m<=0 or
            not math.isfinite(ambiguity_m) or ambiguity_m<0):
        raise ValueError('Invalid cross-floor candidate context')
    result=[];seen=set()
    for link in links:
        if source not in link or target not in link:
            raise ValueError('Missing floor in elevator correspondence')
        pair=(link[source],link[target])
        if pair in seen:raise ValueError('Duplicate elevator correspondence')
        seen.add(pair)
        a=from_catalog.places.get(pair[0]);b=to_catalog.places.get(pair[1])
        if a is None or b is None or a.get('pose') is None or b.get('pose') is None:
            raise ValueError('Corresponding elevator annotation is unavailable')
        distance=math.hypot(float(pose[0])-float(a['pose'][0]),float(pose[1])-float(a['pose'][1]))
        if distance<=max_distance_m:
            result.append(dict(source_id=pair[0],destination_id=pair[1],distance_m=distance,
                               destination_reference_xy=copy.deepcopy(b['pose'][:2])))
    result.sort(key=lambda item:item['distance_m'])
    plausible=[item for item in result if item['distance_m']<=result[0]['distance_m']+ambiguity_m] if result else []
    return dict(status='unique_candidate' if len(plausible)==1 else 'ambiguous' if plausible else 'unavailable',
                candidates=plausible,verified_elevator=False,verified_pose=False)


def fit_floor_transform(links, from_catalog, to_catalog):
    """Fit an SE(2) seed from explicitly corresponding annotated doors.

    A low residual is a property of the annotations, not evidence that the
    wheelchair has reached the destination floor or that localization works.
    """
    source=from_catalog.value['floor_id'];target=to_catalog.value['floor_id']
    if source==target or not isinstance(links,list) or len(links)<2:
        raise ValueError('At least two cross-floor correspondences are required')
    source_points=[];target_points=[];ids=[];seen_source=set();seen_target=set()
    for link in links:
        a,b=link.get(source),link.get(target)
        if not isinstance(a,str) or not isinstance(b,str) or a in seen_source or b in seen_target:
            raise ValueError('Corresponding elevator identities must be unique')
        seen_source.add(a);seen_target.add(b)
        first=from_catalog.places.get(a);second=to_catalog.places.get(b)
        if first is None or second is None or first.get('pose') is None or second.get('pose') is None:
            raise ValueError('Corresponding elevator annotation is unavailable')
        source_points.append(first['pose'][:2]);target_points.append(second['pose'][:2]);ids.append([a,b])
    x,y=np.asarray(source_points,dtype=float),np.asarray(target_points,dtype=float)
    if x.shape!=(len(links),2) or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Corresponding elevator positions must be finite XY')
    xc,yc=x.mean(axis=0),y.mean(axis=0)
    if np.linalg.norm(x-xc)<1e-6 or np.linalg.norm(y-yc)<1e-6:
        raise ValueError('Elevator correspondences have no spatial extent')
    u,_,vt=np.linalg.svd((x-xc).T@(y-yc))
    rotation=vt.T@np.diag([1.,np.linalg.det(vt.T@u.T)])@u.T
    translation=yc-rotation@xc
    residual=np.linalg.norm((x@rotation.T+translation)-y,axis=1)
    transform=np.eye(3);transform[:2,:2]=rotation;transform[:2,2]=translation
    return dict(from_floor=source,to_floor=target,from_catalog=from_catalog.fingerprint,
                to_catalog=to_catalog.fingerprint,from_map_id=from_catalog.value.get('localization_map_id'),
                to_map_id=to_catalog.value.get('localization_map_id'),correspondences=ids,matrix=transform.tolist(),
                residual_m=residual.tolist(),max_residual_m=float(residual.max()),
                source='user_confirmed_elevator_annotations',verified_localization=False)


def transform_seed_pose(transform, pose):
    """Apply the annotation fit to a source-frame base pose as an initial guess."""
    matrix=np.asarray(transform['matrix'],dtype=float);p=np.asarray(pose,dtype=float)
    if (matrix.shape!=(3,3) or p.shape!=(3,) or not np.isfinite(matrix).all() or not np.isfinite(p).all() or
            not np.allclose(matrix[2],[0.,0.,1.]) or
            not np.allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-6) or
            not np.isclose(np.linalg.det(matrix[:2,:2]),1.,atol=1e-6)):
        raise ValueError('Invalid floor transform or measured base pose')
    xy=matrix[:2,:2]@p[:2]+matrix[:2,2]
    angle=p[2]+math.atan2(matrix[1,0],matrix[0,0])
    return [float(xy[0]),float(xy[1]),float(math.atan2(math.sin(angle),math.cos(angle)))]


def capture_source_pose(state, catalog):
    """Freeze the last valid source-floor base pose before entering the elevator."""
    binding=state.get('map_binding') or {};stamp=state.get('origin_stamp_s');now=state.get('stamp_s')
    pose=state.get('origin_pose')
    if (state.get('input_stale') or state.get('localization_status')!='valid' or
            binding.get('valid') is not True or binding.get('map_id')!=catalog.value.get('localization_map_id') or
            binding.get('map_frame')!=catalog.value.get('frame_id') or state.get('origin_frame_id')!=binding['map_frame'] or
            type(stamp) not in (int,float) or type(now) not in (int,float) or
            not math.isfinite(stamp) or not math.isfinite(now) or not 0<=now-stamp<=.5 or
            not isinstance(pose,(list,tuple)) or len(pose)!=3 or not all(math.isfinite(float(x)) for x in pose)):
        raise ValueError('A current source-floor mapped base pose is required')
    return dict(pose=list(pose),stamp_s=stamp,clock_domain=state.get('clock_domain'),
                local_identity=[state.get(k) for k in ('session_id','generation','frame_id')],
                map_binding=copy.deepcopy(binding),floor_id=catalog.value['floor_id'],
                source='source_floor_localization_before_elevator')


def destination_lidar_seed(transform, source, base_from_lidar):
    """Transform the saved base pose into a provisional destination LiDAR XY/yaw."""
    if source.get('floor_id')!=transform.get('from_floor'):
        raise ValueError('Saved pose and floor transform do not match')
    extrinsic=np.asarray(base_from_lidar,dtype=float)
    if (extrinsic.shape!=(4,4) or not np.isfinite(extrinsic).all() or
            not np.allclose(extrinsic[3],[0.,0.,0.,1.]) or
            not np.allclose(extrinsic[:3,:3].T@extrinsic[:3,:3],np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(extrinsic[:3,:3]),1.,atol=1e-6)):
        raise ValueError('Invalid LiDAR-to-base extrinsic')
    base=transform_seed_pose(transform,source['pose']);c,s=math.cos(base[2]),math.sin(base[2])
    xy=np.asarray(base[:2])+np.array([[c,-s],[s,c]])@extrinsic[:2,3]
    yaw=base[2]+math.atan2(extrinsic[1,0],extrinsic[0,0])
    return dict(floor_id=transform['to_floor'],lidar_xy_yaw=[float(xy[0]),float(xy[1]),
        float(math.atan2(math.sin(yaw),math.cos(yaw)))],source_pose=copy.deepcopy(source),
        floor_transform=copy.deepcopy(transform),verified_localization=False,
        limitation='before_elevator_pose_is_only_an_initial_guess')
