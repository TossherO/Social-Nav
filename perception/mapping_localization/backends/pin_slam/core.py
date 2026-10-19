"""Project lifecycle around the fixed official PIN mapping/registration/PGO core."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from .map_io import POINT_FIELDS, load_map, portable_config, save_map

# Upstream uses top-level imports. This module belongs to a dedicated process.
VENDOR = Path(__file__).parent / 'upstream'
if 'utils' in sys.modules and not str(getattr(sys.modules['utils'], '__file__', '')).startswith(str(VENDOR)):
    raise RuntimeError('PIN core must run in its own process: conflicting upstream utils module')
sys.path.insert(0, str(VENDOR))
from utils.config import Config
from utils.tools import deskewing, freeze_decoders, seed_anything, transform_torch, voxel_down_sample_torch
from utils.tracker import Tracker
from utils.mapper import Mapper
from utils.pgo import PoseGraphManager
from utils.loop_detector import NeuralPointMapContextManager, detect_local_loop
from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from .map_refinement import refine_global_features


def pose_matrix(pose):
    value = np.asarray(pose, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError('Expected finite roll pitch yaw x y z, radians and metres')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', value[:3]).as_matrix()
    matrix[:3, 3] = value[3:]
    return matrix


def scaled_motion(delta, ratio):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(delta[:3, :3]).as_rotvec() * ratio).as_matrix()
    result[:3, 3] = delta[:3, 3] * ratio
    return result


@dataclass
class Result:
    stamp_s: float
    valid: bool
    status: str
    reason: str
    local_pose: np.ndarray | None
    map_pose: np.ndarray | None
    map_revision: int


class PinSlam:
    # Sampling and compute budgets can change without changing the trained
    # field. Decoder, feature and query geometry remain bound to the map.
    LOCALIZATION_RUNTIME_KEYS = ('device', 'vox_down_m', 'source_vox_down_m', 'infer_bs', 'reg_iter_n',
                                 'motion_prediction_max_gap_s')

    def __init__(self, config_path, mode, initial_pose, map_path=None):
        if mode not in ('mapping', 'localization'):
            raise ValueError('Unknown PIN mode')
        cfg = Config()
        cfg.motion_prediction_max_gap_s = .5
        cfg.final_map_refine_iters = 5000
        cfg.final_map_refine_bs = 2048
        cfg.final_map_refine_offset_m = .08
        values = yaml.safe_load(Path(config_path).read_text())
        self._configure(cfg, values)
        payload = None
        if mode == 'localization':
            payload, self.map_manifest = load_map(map_path)
            self._configure(cfg, payload['config'])
            self._configure(cfg, {name: values[name] for name in self.LOCALIZATION_RUNTIME_KEYS if name in values})
            cfg.pgo_on = False
            cfg.decoder_freezed = True
        cfg.setup_dtype()
        cfg.run_with_ros, cfg.use_dataloader, cfg.track_on = True, False, True
        cfg.pc_path = cfg.pose_path = cfg.calib_path = ''
        cfg.wandb_vis_on = cfg.o3d_vis_on = False
        if cfg.color_on or cfg.semantic_on or cfg.ba_freq_frame or cfg.local_map_context or cfg.loop_with_feature:
            raise ValueError('Adapter supports geometry-only maps, scan-context loops and fixed per-frame poses')
        positive = ('voxel_size_m', 'vox_down_m', 'source_vox_down_m', 'local_map_radius',
                    'buffer_size', 'pool_capacity', 'bs', 'iters', 'init_iter_ratio', 'reg_iter_n')
        positive += ('motion_prediction_max_gap_s',)
        if any(not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0 for name in positive):
            raise ValueError('PIN resolutions, capacities and iteration budgets must be positive')
        if (type(cfg.final_map_refine_iters) is not int or cfg.final_map_refine_iters < 0 or
                type(cfg.final_map_refine_bs) is not int or cfg.final_map_refine_bs <= 0 or
                not 0 < cfg.final_map_refine_offset_m <= cfg.surface_sample_range_m):
            raise ValueError('Invalid final map fitting budget or surface neighbourhood')
        if not cfg.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('PIN requires an explicitly available CUDA device')
        seed_anything(cfg.seed)
        self.config, self.mode = cfg, mode
        self.session_id, self.generation = uuid.uuid4().hex, 0
        self.map_revision, self.frames, self.valid_frames, self.lost_frames = 0, 0, 0, 0
        self.initial_pose = list(initial_pose)
        self.map_pose = pose_matrix(initial_pose)
        self.local_pose = self.map_pose.copy()
        self.delta, self.delta_dt = np.eye(4), None
        self.last_valid_stamp, self.last_input_stamp = None, None
        self.clock_fault = False
        self.neural = NeuralPoints(cfg)
        self.decoder = Decoder(cfg, cfg.geo_mlp_hidden_dim, cfg.geo_mlp_level, 1)
        self.decoders = {'sdf': self.decoder, 'semantic': None, 'color': None}
        self.dataset = SLAMDataset(cfg)
        self.dataset.cur_pose_ref = self.map_pose.copy()
        self.dataset.last_pose_ref = self.map_pose.copy()
        self.dataset.cur_pose_torch = self._tensor(self.map_pose, torch.float64)
        if payload is not None:
            for name in POINT_FIELDS:
                setattr(self.neural, name, payload[name].to(cfg.device))
            for name in ('cur_ts', 'max_ts', 'reboot_ts'):
                setattr(self.neural, name, payload[name])
            # query_feature must use the same point-local coordinate frames as
            # the map that was optimized and saved, including PGO rotations.
            self.neural.after_pgo = payload['after_pgo']
            self.neural.travel_dist = payload['travel_dist'].to(cfg.device)
            self.neural.position_encoder_geo.load_state_dict(payload['encoder_geo'])
            self.neural.position_encoder_color.load_state_dict(payload['encoder_color'])
            self.decoder.load_state_dict(payload['decoder'])
            self.neural.temporal_local_map_on = False
            self.neural.recreate_hash(self._tensor(self.map_pose[:3, 3]), None, True, False)
            freeze_decoders(self.decoders, cfg)
            self.neural.requires_grad_(False)
        self.tracker = Tracker(cfg, self.neural, self.decoders)
        self.tracker.reg_local_map = mode == 'mapping'
        # No mapper, pool, optimizer or pose graph exists in fixed-map mode.
        self.mapper = Mapper(cfg, self.dataset, self.neural, self.decoders) if mode == 'mapping' else None
        self.pgm = PoseGraphManager(cfg) if cfg.pgo_on else None
        self.context = NeuralPointMapContextManager(cfg) if cfg.pgo_on else None
        self.loop_failures = 0
        self.last_registration = {}
        self.final_map_refinement = None
        if self.pgm is not None:
            self.pgm.add_pose_prior(0, self.map_pose, fixed=True)

    @staticmethod
    def _configure(cfg, values):
        if not isinstance(values, dict):
            raise ValueError('PIN config must be a mapping')
        for name, value in values.items():
            if not hasattr(cfg, name):
                raise ValueError('Unknown PIN config key: ' + name)
            setattr(cfg, name, value)

    def _tensor(self, value, dtype=None):
        return torch.as_tensor(value, device=self.config.device, dtype=dtype or self.config.dtype)

    def invalid(self, stamp, reason):
        self.lost_frames += 1
        self.dataset.lose_track = True
        # Never expose an old pose as a current measurement.
        return Result(stamp, False, 'LOST', reason, None, None, self.map_revision)

    def process(self, scan):
        cfg, ds, i = self.config, self.dataset, self.frames
        if self.clock_fault or self.last_input_stamp is not None and scan.stamp_s <= self.last_input_stamp:
            self.clock_fault = True
            return self.invalid(scan.stamp_s, 'NONMONOTONIC_SCAN_TIME_RESTART_REQUIRED')
        self.last_input_stamp = scan.stamp_s
        if i >= len(ds.travel_dist):
            raise RuntimeError('PIN pose history capacity reached; save and restart explicitly')
        points = self._tensor(scan.points[:, :3])
        times = self._tensor(scan.times) if scan.times is not None else None
        if cfg.deskew and times is None:
            return self.invalid(scan.stamp_s, 'MISSING_POINT_TIMES')
        ranges = points.norm(dim=1)
        keep = (ranges >= cfg.min_range) & (ranges <= cfg.max_range) & (points[:, 2] >= cfg.min_z) & (points[:, 2] <= cfg.max_z)
        points = points[keep]
        times = times[keep] if times is not None else None
        if len(points) < 30:
            return self.invalid(scan.stamp_s, 'TOO_FEW_POINTS_AFTER_CROP')
        indices = voxel_down_sample_torch(points, cfg.vox_down_m)
        points, times = points[indices], times[indices] if times is not None else None
        source_indices = voxel_down_sample_torch(points, cfg.source_vox_down_m)
        source = points[source_indices].clone()
        if len(source) < 30:
            return self.invalid(scan.stamp_s, 'TOO_FEW_REGISTRATION_POINTS')
        source_times = times[source_indices] if times is not None else None
        ds.processed_frame = i
        dt = scan.stamp_s - self.last_valid_stamp if self.last_valid_stamp is not None else None
        guess = self.map_pose.copy()
        # A rejected measurement does not erase the last observed velocity.
        # Keep it only as a short-lived initial guess, never as a valid output.
        if dt is not None and self.delta_dt is not None and dt <= cfg.motion_prediction_max_gap_s:
            guess = guess @ scaled_motion(self.delta, dt / self.delta_dt)
            if cfg.deskew:
                source = deskewing(source, source_times, self._tensor(scaled_motion(self.delta, scan.duration_s / self.delta_dt)))
        first_map = self.mode == 'mapping' and i == 0
        valid = True
        if first_map:
            pose = self.map_pose.copy()
        else:
            pose_t, _, _, valid = self.tracker.tracking(source, self._tensor(guess, torch.float64))
            self.last_registration = dict(getattr(self.tracker, 'last_diagnostics', {}))
            pose = pose_t.detach().cpu().numpy() if pose_t is not None else guess
            valid = bool(valid) and np.isfinite(pose).all()
            if valid and dt is not None:
                valid = np.linalg.norm((np.linalg.inv(self.map_pose) @ pose)[:3, 3]) <= cfg.surface_sample_range_m * 20
                if not valid:
                    self.last_registration.setdefault('reasons', []).append('EXCESSIVE_POSE_STEP')
        # Save a history row even on failure, but never train on the failed scan.
        if valid:
            delta = np.linalg.inv(self.map_pose) @ pose
            if self.last_valid_stamp is None:
                self.local_pose = pose.copy()
            else:
                self.local_pose = self.local_pose @ delta
                self.delta, self.delta_dt = delta, dt
            self.map_pose = pose.copy()
            self.last_valid_stamp = scan.stamp_s
            ds.lose_track = False
            ds.last_odom_tran = delta
            ds.stop_count = ds.stop_count + 1 if np.linalg.norm(delta - np.eye(4)) < 1e-3 else 0
            ds.stop_status = ds.stop_count > cfg.stop_frame_thre
            if cfg.deskew and dt is not None:
                points = deskewing(points, times, self._tensor(scaled_motion(delta, scan.duration_s / dt)))
        else:
            ds.lose_track = True
        ds.cur_pose_ref = ds.last_pose_ref = self.map_pose.copy()
        ds.cur_pose_torch = self._tensor(self.map_pose, torch.float64)
        ds.cur_point_cloud_torch, ds.cur_source_points = points, source
        ds.odom_poses[i] = self.local_pose
        ds.travel_dist[i] = (ds.travel_dist[i-1] if i else 0) + (np.linalg.norm(ds.last_odom_tran[:3, 3]) if valid and i else 0)
        if cfg.pgo_on:
            ds.pgo_poses[i] = self.map_pose
        if self.mode == 'mapping':
            self.neural.travel_dist = self._tensor(ds.travel_dist[:i+1])
            if cfg.pgo_on:
                self._loop(i, valid)
            if valid and (first_map or not ds.stop_status):
                self.mapper.process_frame(points, None, ds.cur_pose_torch, i, cfg.dynamic_filter_on and not first_map)
                if i >= cfg.freeze_after_frame and not cfg.decoder_freezed:
                    freeze_decoders(self.decoders, cfg)
                    cfg.decoder_freezed = True
                self.mapper.mapping(cfg.iters * cfg.init_iter_ratio if first_map else cfg.iters)
            elif self.neural.count():
                self.neural.reset_local_map(ds.cur_pose_torch[:3, 3], None, i)
        self.frames += 1
        if first_map:
            return Result(scan.stamp_s, False, 'INITIALIZING', 'FIRST_SCAN_BUILDS_MAP_NO_REGISTRATION', None, None, self.map_revision)
        if not valid:
            detail = '+'.join(self.last_registration.get('reasons', []))
            return self.invalid(scan.stamp_s, 'PIN_REGISTRATION_REJECTED' + (':'+detail if detail else ''))
        self.valid_frames += 1
        return Result(scan.stamp_s, True, 'TRACKING', '', self.local_pose.copy(), self.map_pose.copy(), self.map_revision)

    def _loop(self, i, valid):
        """Official scan-context candidate, implicit registration and GTSAM correction."""
        cfg, ds, pgm, context = self.config, self.dataset, self.pgm, self.context
        poses, travel = ds.pgo_poses[:i+1], ds.travel_dist[:i+1]
        context.add_node(i, ds.cur_point_cloud_torch, valid_flag=valid)
        pgm.add_frame_node(i, poses[i])
        pgm.init_poses = poses
        if i:
            pgm.add_odometry_factor(i, i-1, np.linalg.inv(poses[i-1]) @ poses[i])
            pgm.estimate_drift(travel, i, correct_ratio=0.01)
        if not valid or ds.stop_status or i-pgm.last_loop_idx <= cfg.pgo_freq:
            return
        candidates = (travel[-1]-travel) > cfg.min_loop_travel_dist_ratio * cfg.local_map_radius
        if not candidates.any():
            return
        loop_id, _, transform = detect_local_loop(poses, candidates, pgm.drift_radius, i, self.loop_failures,
                                                  cfg.local_loop_dist_thre, cfg.local_loop_dist_thre*3, cfg.silence)
        if loop_id is None and cfg.global_loop_on:
            loop_id, _, transform, _ = context.detect_global_loop(poses, pgm.drift_radius*cfg.loop_dist_drift_ratio_thre,
                                                                 candidates, self.neural)
        if loop_id is None or not context.valid_flags[loop_id]:
            return
        if cfg.loop_z_check_on and abs(transform[2, 3]) > cfg.voxel_size_m*4:
            return
        initial = self._tensor(poses[loop_id] @ transform, torch.float64)
        # Flush local optimized features before swapping to a history neighbourhood.
        self.neural.assign_local_to_global()
        self.neural.recreate_hash(initial[:3, 3], None, True, True, loop_id)
        refined, _, _, accepted = self.tracker.tracking(ds.cur_source_points, initial, loop_reg=True)
        if accepted:
            transform = np.linalg.inv(poses[loop_id]) @ refined.detach().cpu().numpy()
            accepted = pgm.add_loop_factor(i, loop_id, transform)
        if accepted:
            pgm.optimize_pose_graph()
            differences = self._tensor(pgm.get_pose_diff())
            self.neural.adjust_map(differences)
            self.map_pose = pgm.cur_pose.copy()
            ds.update_poses_after_pgo(pgm.pgo_poses)
            ds.cur_pose_torch = self._tensor(self.map_pose, torch.float64)
            self.mapper.transform_data_pool(differences)
            pgm.last_loop_idx = i
            pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
            self.map_revision += 1
            self.loop_failures = 0
        else:
            self.loop_failures += 1
        # Keep global points; do not compact/prune away alternative history features.
        self.neural.recreate_hash(ds.cur_pose_torch[:3, 3], None, True, True, i)

    def save(self, path, source):
        if self.mode != 'mapping':
            raise RuntimeError('Fixed-map localization cannot save or update its input map')
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError('Map destination already exists')
        if self.final_map_refinement is None:
            self.final_map_refinement = refine_global_features(self.neural, self.decoder, self.config)
        return save_map(path, self.neural, self.decoder, self.config,
                        dict(input=source, initial_pose=self.initial_pose, frames=self.frames,
                             registered_frames=self.valid_frames, lost_frames=self.lost_frames,
                             map_revision=self.map_revision, session_id=self.session_id,
                             final_map_refinement=self.final_map_refinement))

    def parameters(self):
        return portable_config(self.config)
