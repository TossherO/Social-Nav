"""Ordered raw datasets/manifests and fingerprint-bound detection replay; no pickle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import validate_boxes, validate_transform
from .pipeline import Frame


def read_xyzi(path):
    path = Path(path)
    if path.stat().st_size % 16:
        raise ValueError(f'Point file is not float32 XYZI: {path}')
    points = np.fromfile(path, dtype='<f4').reshape(-1, 4)
    if not len(points):
        raise ValueError(f'Empty point cloud: {path}')
    return points


def read_pose(path):
    values = Path(path).read_text(encoding='utf-8').split()
    if len(values) != 8 or not values[0].startswith('pose'):
        raise ValueError(f'Invalid Wheelchair pose: {path}')
    position = np.asarray(values[1:4], dtype=float)
    quaternion = np.asarray(values[4:8], dtype=float)  # qx qy qz qw
    if not np.all(np.isfinite(quaternion)) or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
        raise ValueError('Invalid pose quaternion')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return validate_transform(matrix)


class WheelchairSource:
    def __init__(self, root, scene, nominal_rate_hz=10, timestamps=None, start=0, limit=None):
        root = Path(root).resolve()
        self.scene = str(scene)
        self.path = (root/scene).resolve()
        if not self.path.is_relative_to(root) or not (self.path/'points').is_dir():
            raise ValueError('Choose a scene below the Wheelchair dataset root with points/')
        if not np.isfinite(nominal_rate_hz) or nominal_rate_hz <= 0:
            raise ValueError('nominal_rate_hz must be positive')
        if start < 0 or (limit is not None and limit < 1):
            raise ValueError('Invalid start/limit')
        files = sorted((self.path/'points').glob('*.bin'), key=lambda p: int(p.stem))
        self.files = files[start:None if limit is None else start+limit]
        if not self.files:
            raise ValueError('No input frames in the requested slice')
        self.rate = nominal_rate_hz
        self.timestamps = None
        if timestamps is not None:
            self.timestamps = json.loads(Path(timestamps).read_text(encoding='utf-8'))
        self.metadata = {'dataset': 'wheelchair', 'scene': self.scene,
                         'timestamp_source': 'explicit_manifest' if timestamps else 'nominal_frame_index',
                         'nominal_rate_hz': None if timestamps else nominal_rate_hz,
                         'pose_semantics': 'dataset poses are lidar-to-world; no raw odometry fallback'}

    def __iter__(self):
        for path in self.files:
            index = int(path.stem)
            stamp = float(self.timestamps[str(index)]) if self.timestamps is not None else index/self.rate
            pose_path = self.path/'poses'/f'{index:04d}.txt'
            pose, health = None, 'invalid'
            if pose_path.is_file():
                try:
                    pose, health = read_pose(pose_path), 'valid'
                except (ValueError, OSError):
                    health = 'invalid'
            rgb = self.path/'imgs'/f'{index:04d}.png'
            yield Frame(f'{self.scene}/{path.stem}', self.scene, stamp, read_xyzi(path),
                        timestamp_source=self.metadata['timestamp_source'], world_from_lidar=pose,
                        pose_source='dataset_pose' if pose is not None else 'missing', pose_health=health,
                        rgb_asset_uri=str(rgb) if rgb.is_file() else None,
                        rgb_stamp_s=stamp if rgb.is_file() else None)


class ManifestSource:
    """Portable JSONL: explicit time/frames, relative asset paths, optional rigid pose."""
    def __init__(self, path, start=0, limit=None):
        self.path = Path(path).resolve()
        rows = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.rows = rows[start:None if limit is None else start+limit]
        if not self.rows:
            raise ValueError('Empty frame manifest')
        self.metadata = {'dataset': 'explicit_manifest', 'path': str(self.path)}

    def __iter__(self):
        for row in self.rows:
            pose = row.get('world_from_lidar')
            if pose is not None:
                pose = validate_transform(pose)
            rgb = row.get('rgb_asset_uri')
            yield Frame(row['observation_id'], row['sequence_id'], float(row['stamp_s']),
                        read_xyzi(self.path.parent/row['pointcloud']),
                        sensor_frame=row['sensor_frame'], clock_domain=row.get('clock_domain', 'dataset'),
                        timestamp_source=row['timestamp_source'], world_from_lidar=pose,
                        world_frame=row['world_frame'], pose_source=row.get('pose_source', 'missing'),
                        pose_health=row.get('pose_health', 'invalid'),
                        pose_stamp_s=row.get('pose_stamp_s'),
                        rgb_asset_uri=str(self.path.parent/rgb) if rgb else None,
                        rgb_stamp_s=row.get('rgb_stamp_s'))


class CachedDetector:
    def __init__(self, path):
        self.rows = {}
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['sequence_id'], row['observation_id'])
            if key in self.rows or row.get('health') == 'invalid':
                raise ValueError('Duplicate/invalid cached inference observation')
            validate_boxes(row['detections'])
            self.rows[key] = row
        self.metadata = {'backend': 'fingerprint_bound_detection_replay', 'source': str(Path(path).resolve())}

    def predict(self, frame):
        row = self.rows[(frame.sequence_id, frame.observation_id)]
        if (row['stamp_s'] != frame.stamp_s or row['detection_frame_id'] != frame.sensor_frame
                or row['input_fingerprint'] != frame.input_fingerprint()):
            raise ValueError('Cached detections do not match this source point cloud/time/frame')
        return validate_boxes(row['detections'])
