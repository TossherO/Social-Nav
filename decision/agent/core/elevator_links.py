"""Convert a measured destination-floor base pose proposal to a provisional LiDAR seed."""
import math
import copy

import numpy as np


def standard_destination_seed(transition, standard, scan):
    """Bind a user-selected, map-specific LiDAR seed to a fresh post-ride scan."""
    pose=standard.get('pose_rpy_xyz')
    if (transition.status!='awaiting_destination_localization' or
            standard.get('map_id')!=transition.destination_map_id or
            standard.get('frame_id')!='map' or standard.get('reference')!='rslidar' or
            not isinstance(pose,list) or len(pose)!=6 or
            not all(type(v) in (int,float) and math.isfinite(v) for v in pose) or
            scan.get('source')!='local_lidar_returns' or scan.get('clock_domain')!='ros_time' or
            not isinstance(scan.get('frame_id'),str) or scan['frame_id'].lstrip('/')!=standard['reference'] or
            type(scan.get('stamp_s')) not in (int,float) or
            not math.isfinite(scan['stamp_s']) or scan['stamp_s']<=transition.last_observation_stamp_s):
        raise ValueError('Standard start requires a matching map and a fresh post-ride scan')
    return dict(floor_id=transition.to_floor,scan_stamp_s=scan['stamp_s'],task_id=transition.task_id,
                goal_id=transition.goal_id,exit_stamp_s=transition.last_observation_stamp_s,
                scan_frame=scan['frame_id'],source='standard_destination_start',
                destination=dict(from_floor=transition.from_floor,to_floor=transition.to_floor,
                                 to_map_id=transition.destination_map_id),
                lidar_xy_yaw=[pose[3],pose[4],pose[2]],lidar_pose_rpy_xyz=copy.deepcopy(pose),
                verified_localization=False)


def destination_seed(transition, base_pose, scan, base_from_lidar):
    if (transition.status!='awaiting_destination_localization' or
            scan.get('source')!='local_lidar_returns' or scan.get('clock_domain')!='ros_time' or
            not scan.get('frame_id') or not math.isfinite(scan['stamp_s']) or
            scan['stamp_s']<=transition.last_observation_stamp_s or
            np.shape(base_pose)!=(3,) or not np.isfinite(base_pose).all()):
        raise ValueError('Initial pose must use a scan measured after exit')
    return dict(floor_id=transition.to_floor,base_pose=list(base_pose),scan_stamp_s=scan['stamp_s'],
                task_id=transition.task_id,goal_id=transition.goal_id,exit_stamp_s=transition.last_observation_stamp_s,
                scan_frame=scan['frame_id'],source='agent_destination_scan_and_map',
                destination=dict(from_floor=transition.from_floor,to_floor=transition.to_floor,
                                 to_map_id=transition.destination_map_id),
                lidar_xy_yaw=lidar_xy_yaw(base_pose,base_from_lidar),verified_localization=False)


def lidar_xy_yaw(base, base_from_lidar):
    extrinsic=np.asarray(base_from_lidar,dtype=float)
    if (extrinsic.shape!=(4,4) or not np.isfinite(extrinsic).all() or
            not np.allclose(extrinsic[3],[0.,0.,0.,1.]) or
            not np.allclose(extrinsic[:3,:3].T@extrinsic[:3,:3],np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(extrinsic[:3,:3]),1.,atol=1e-6)):
        raise ValueError('Invalid LiDAR-to-base extrinsic')
    c,s=math.cos(base[2]),math.sin(base[2])
    xy=np.asarray(base[:2])+np.array([[c,-s],[s,c]])@extrinsic[:2,3]
    yaw=base[2]+math.atan2(extrinsic[1,0],extrinsic[0,0])
    return [float(xy[0]),float(xy[1]),float(math.atan2(math.sin(yaw),math.cos(yaw)))]
