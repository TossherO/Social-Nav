"""Offline final-pose fusion of recorded, moving-origin deskewed scans."""
import hashlib

import numpy as np

from .layers import GridSnapshot, project_evidence
from .memory import encoded
from .occupancy import OccupancyTree
from .scans import read_scans


def fuse_scans(candidate, path, timeline, config, fingerprint):
    meta = candidate.metadata
    origin = np.asarray(meta['origin_xy'])
    shape = np.asarray(meta['shape'])
    floor, support = candidate.layers['ground_z'], candidate.layers['ground_support']
    sensors = np.asarray([row['position'] for row in timeline.rows if row['health'] == 0])
    finite_floor = floor[np.isfinite(floor)]
    low_z = float(finite_floor.min()-.5) if len(finite_floor) else float(sensors[:, 2].min()-3)
    high_z = float(finite_floor.max()+config['max_height_m']+.5) if len(finite_floor) else float(sensors[:, 2].max()+3)
    bounds = [*origin, low_z, *(origin+shape[::-1]*config['resolution']), high_z]
    settings = dict(config, voxel_ttl_s=max(config['voxel_ttl_s'], float(timeline.times[-1]-timeline.times[0]+1)))
    settings['max_voxels'] = config['static']['max_voxels']
    count, unbound, rays, last_stamp = 0, 0, 0, None
    tree = OccupancyTree(config['resolution'])
    try:
        for record, data in read_scans(path):
            stamp = record['stamp_ns']/1e9
            pose = timeline.at(stamp, exact_ns=record['stamp_ns'])
            if pose is None:
                unbound += 1
                continue
            if ((pose['session_id'], pose['generation'], pose['local_frame']) !=
                    (record['session_id'], record['generation'], record['frame_id']) or
                    not np.allclose(pose['local_from_lidar'], data['local_from_lidar'], atol=1e-8)):
                raise ValueError('Recorded scan and localization source do not agree')
            local, final = np.asarray(pose['local_from_lidar']), np.asarray(pose['world_from_lidar'])
            correction = final @ np.linalg.inv(local)
            points = data['points'] @ final[:3, :3].T+final[:3, 3]
            origins = data.get('origins')
            if origins is not None:
                origins = origins @ correction[:3, :3].T+correction[:3, 3]
            model = record['ray_origin_model']
            if model == 'pose_interpolated_origins' and origins is None:
                raise ValueError('Moving-origin evidence was lost from the scan journal')
            use_rays = bool(record['rays_used'])
            if use_rays and model not in ('pose_interpolated_origins', 'bounded_scan_end_approximation'):
                raise ValueError('Unsupported scan ray origin model')
            tree.update(points, final[:3, 3], bounds, stamp, settings, rays=use_rays,
                        ray_origins=origins, read=False)
            count += 1; rays += int(use_rays); last_stamp = stamp
        if not count:
            raise ValueError('No scan could be bound to a final-map pose')
        voxels = tree.snapshot(last_stamp)
    finally:
        tree.close()
    layers = project_evidence(voxels, floor, support, origin, config, last_stamp, candidates=candidate.layers)
    layers['last_observed'][~np.isfinite(layers['last_observed'])] = np.nan
    layers['endpoint_count'] = candidate.layers['endpoint_count']
    signature = hashlib.sha256(encoded([meta['snapshot_id'], fingerprint]).encode()).hexdigest()[:20]
    metadata = dict(meta, snapshot_id=meta['map_id']+':scan_fusion:'+signature, stamp_s=last_stamp,
                    geometry_status='STATIC_SCAN_OBSERVED' if rays else 'STATIC_SCAN_HITS_ONLY',
                    source_evidence='deskewed_scans_and_final_map_endpoints', rays_used=bool(rays),
                    free_space_supported=bool(rays), per_cell_observation_time_available=True,
                    voxel_count=len(voxels), evidence_scans=count, unbound_scans=unbound, ray_scans=rays,
                    final_endpoint_candidates_retained=True, dynamic_residual_cleanup_verified=False)
    for array in layers.values():
        array.flags.writeable = False
    return GridSnapshot(metadata, layers)
