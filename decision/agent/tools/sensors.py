"""Configured sensor geometry and causal pinhole field of view for the Agent."""
import math
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]


def sensor_context(geometry, camera_info=None, *, rgb_shape=None, rgb_frame=None, rgb_stamp=None):
    result = dict(status='unavailable', reason='vehicle_geometry_unavailable', reference='base_link')
    if geometry is None or geometry.assumed:
        return result
    vehicle = yaml.safe_load((ROOT/'configs/wheelchair_vehicle.yaml').read_text())
    if not isinstance(geometry.source,dict) or geometry.source.get('vehicle_id') != vehicle.get('vehicle_id'):
        return dict(status='unavailable',reason='sensor_calibration_vehicle_identity_mismatch',reference='base_link')
    base_lidar = np.asarray(geometry.base_from_lidar, dtype=float)
    if base_lidar.shape != (4, 4) or not np.isfinite(base_lidar).all():
        return result
    camera = yaml.safe_load((ROOT/'perception/sensors/calibration/front_camera_lidar.yaml').read_text())
    imu = yaml.safe_load((ROOT/'perception/sensors/calibration/lidar_imu.yaml').read_text())
    lidar_camera = np.asarray(camera['camera_optical_to_lidar']['matrix'], dtype=float)
    imu_lidar = np.eye(4)
    imu_lidar[:3,:3] = np.asarray(imu['mapping']['extrinsic_R'],dtype=float).reshape(3,3)
    imu_lidar[:3,3] = imu['mapping']['extrinsic_T']
    base_camera = base_lidar@lidar_camera
    base_imu = base_lidar@np.linalg.inv(imu_lidar)
    result = dict(status='configured', reference='base_link', axes='x_forward_y_left_z_up',
        source_status='configured_transforms_not_independent_mount_validation',
        lidar=dict(position_m=base_lidar[:3,3].tolist(),source='vehicle_profile.base_from_lidar'),
        imu=dict(position_m=base_imu[:3,3].tolist(),source='lidar_imu.yaml',
                 calibration_status=imu['calibration']['status']),
        camera=dict(position_m=base_camera[:3,3].tolist(),optical_axis=base_camera[:3,2].tolist(),
                    source='front_camera_lidar.yaml_and_vehicle_profile',calibration_status=camera['status'],
                    view=dict(status='unavailable',reason='matching_camera_info_required')))
    c = result['camera']
    if camera_info is None or rgb_shape is None or rgb_stamp is None:
        return result
    try:
        width, height = int(camera_info['width']), int(camera_info['height'])
        k = np.asarray(camera_info['K'], dtype=float).reshape(3,3)
        stamp = camera_info['stamp_ns']*1e-9
        if (list(rgb_shape[:2]) != [height,width] or camera_info['frame_id'] != rgb_frame or
                rgb_frame != camera['front_camera']['optical_frame'] or
                not math.isfinite(stamp) or stamp > rgb_stamp or width <= 0 or height <= 0 or
                not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0):
            return result
        # Pinhole rays through the horizontal image edges. Raw lens distortion
        # is not inverted: this is a visual approximation, not visibility proof.
        rays = []
        for u in (0., float(width-1)):
            ray = base_camera[:3,:3]@np.array([(u-k[0,2])/k[0,0], 0., 1.])
            rays.append(math.atan2(float(ray[1]),float(ray[0])))
        c['view'] = dict(status='approximate_pinhole',edge_yaw_rad=rays,
            camera_info_stamp_s=stamp,rgb_stamp_s=rgb_stamp,image_size=[width,height],
            frame_id=rgb_frame,distortion_model=camera_info.get('distortion_model'),
            interpretation='horizontal_projection_not_occlusion_or_visibility_proof')
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return result
