"""Lazy SimpleDet3d adapter with causal, pose-compensated scan history."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import json
from pathlib import Path
import sys
import sysconfig
from collections import deque

import numpy as np

from .config import ROOT, project_path
from .geometry import validate_boxes, validate_transform

SOURCE = ROOT / 'perception/detection_tracking/training/simpledet3d'
EXTENSIONS = (
    'ops.iou3d_nms.iou3d_nms_cuda',
    'ops.roiaware_pool3d.roiaware_pool3d_cuda',
    'ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda',
    'ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_single_frame(points, intensity_scale=255.0, zero_intensity=False):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('Detector requires raw [x,y,z,intensity] points of shape (N,4)')
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError('Invalid intensity normalization')
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError('No finite input points; not a valid empty-scene observation')
    # Exact ego-square removal used by CodaDataset.get_lidar_with_sweeps.
    points = points[~((np.abs(points[:, 0]) < 1) & (np.abs(points[:, 1]) < 1))]
    points[:, 3] = 0 if zero_intensity else points[:, 3]/intensity_scale
    return np.c_[points, np.zeros(len(points), dtype=np.float32)].astype(np.float32)


class CausalSweeps:
    """Current scan plus nearest past scans from one continuous pose lifecycle."""
    def __init__(self, max_sweeps, max_time_lag_s=.35, max_pose_age_s=.05):
        if type(max_sweeps) is not int or max_sweeps not in (1, 2, 3, 4):
            raise ValueError('MAX_SWEEPS must be an integer from 1 to 4')
        if not np.isfinite([max_time_lag_s, max_pose_age_s]).all() or min(max_time_lag_s, max_pose_age_s) <= 0:
            raise ValueError('History time limits must be positive and finite')
        self.max_sweeps, self.max_time_lag_s = max_sweeps, float(max_time_lag_s)
        self.max_pose_age_s = float(max_pose_age_s)
        self.history = deque(maxlen=max_sweeps - 1)
        self.reset()

    def reset(self):
        self.history.clear()
        self.identity = self.last_stamp = None

    def prepare(self, frame, intensity_scale=255., zero_intensity=False):
        points = prepare_single_frame(frame.points, intensity_scale, zero_intensity)
        if not np.isfinite(frame.stamp_s):
            raise ValueError('Nonfinite scan time')
        identity = (frame.sequence_id, frame.sensor_frame, frame.clock_domain, frame.world_frame,
                    frame.pose_source, tuple(frame.pose_identity or ()))
        if identity != self.identity:
            self.reset()
        if self.last_stamp is not None and frame.stamp_s <= self.last_stamp:
            raise ValueError('History requires strictly increasing scan times')
        self.identity, self.last_stamp = identity, frame.stamp_s
        context = dict(max_sweeps=self.max_sweeps, max_time_lag_s=self.max_time_lag_s,
                       max_pose_age_s=self.max_pose_age_s, used_sweeps=1,
                       input_fingerprint=frame.input_fingerprint(), history=[],
                       current_pose=None, pose_identity=list(frame.pose_identity or ()),
                       status='single_frame' if self.max_sweeps == 1 else 'unavailable')
        if self.max_sweeps == 1:
            return points, context
        valid = (frame.world_from_lidar is not None and frame.pose_health == 'valid'
                 and frame.pose_source in ('dataset_pose', 'localization', 'stationary_assumption'))
        if frame.pose_stamp_s is not None:
            age = frame.stamp_s - frame.pose_stamp_s
            if not np.isfinite(age) or age < 0:
                raise ValueError('History pose is nonfinite or from the future')
            valid = valid and age <= self.max_pose_age_s + 1e-7
        if not valid:
            self.history.clear()
            context['reason'] = 'reliable_current_pose_unavailable'
            return points, context
        transform = validate_transform(frame.world_from_lidar)
        context['current_pose'] = transform.tolist()
        selected = [past for past in self.history
                    if 0 < frame.stamp_s - past['stamp_s'] <= self.max_time_lag_s]
        merged = [points]
        for past in selected:
            relative = np.linalg.solve(transform, past['pose']).astype(np.float32)
            previous = past['points'].copy()
            previous[:, :3] = previous[:, :3] @ relative[:3, :3].T + relative[:3, 3]
            previous[:, 4] = frame.stamp_s - past['stamp_s']
            merged.append(previous)
            context['history'].append(dict(observation_id=past['observation_id'], stamp_s=past['stamp_s'],
                input_fingerprint=past['fingerprint'], current_from_history=relative.tolist()))
        context['used_sweeps'] += len(selected)
        context['status'] = 'ready' if len(selected) == self.max_sweeps - 1 else 'partial'
        self.history.append(dict(points=points, pose=transform.copy(), stamp_s=frame.stamp_s,
                                 observation_id=frame.observation_id, fingerprint=context['input_fingerprint']))
        return np.concatenate(merged), context


def _load_extensions(extension_root):
    records = []
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    for name in EXTENSIONS:
        qualified = 'simple_det3d.'+name
        relative = Path('simple_det3d') / (name.replace('.', '/')+suffix)
        local = SOURCE / relative
        path = local if local.is_file() else (project_path(extension_root) / relative if extension_root else local)
        if not path.is_file():
            raise RuntimeError(f'Missing compiled extension {qualified}; configure compatible binaries or build separately')
        # Import ONLY an explicitly selected binary, never the external Python source tree.
        importlib.import_module(qualified.rsplit('.', 1)[0])
        if qualified in sys.modules:
            if Path(sys.modules[qualified].__file__).resolve() != path.resolve():
                raise RuntimeError(f'Conflicting extension already imported: {qualified}')
        else:
            spec = importlib.util.spec_from_file_location(qualified, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[qualified] = module
        records.append({'module': qualified, 'path': str(path.resolve()), 'sha256': file_sha256(path)})
    return records


class SimpleDet3dDetector:
    def __init__(self, config, *, metrics=None):
        self.metrics=metrics
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('Detector needs CUDA; check sandbox/device access before changing the environment')
        if not config.get('checkpoint'):
            raise ValueError('Select a trusted checkpoint explicitly or in detection_tracking.local.json')
        checkpoint = project_path(config['checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        sys.path.insert(0, str(SOURCE))
        package = importlib.import_module('simple_det3d')
        if Path(package.__file__).resolve().parent != (SOURCE/'simple_det3d').resolve():
            raise RuntimeError('An unrelated simple_det3d package is already loaded')
        extension_records = _load_extensions(config.get('extension_root'))
        from simple_det3d.utils.config import load_config
        from simple_det3d.datasets.coda_dataset import CodaDataset
        from simple_det3d.models import build_model, load_data_to_gpu
        from simple_det3d.utils.precision_utils import autocast_context, validate_precision_support

        cfg = load_config(project_path(config['config']))
        if cfg.CLASS_NAMES != ['Pedestrian']:
            raise ValueError('This deployment adapter requires Pedestrian')
        if cfg.DATA_CONFIG.get('SWEEP_SELECT_MODE', 'near') != 'near':
            raise ValueError('Causal inference requires nearest past sweeps')
        self.sweeps = CausalSweeps(cfg.DATA_CONFIG.MAX_SWEEPS,
            cfg.DATA_CONFIG.get('MAX_SWEEP_TIME_LAG', .35), cfg.DATA_CONFIG.get('HISTORY_MAX_POSE_AGE_S', .05))
        self.last_input_context = None
        if list(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list) != ['x','y','z','intensity','timestamp']:
            raise ValueError('Unexpected checkpoint point-feature convention')
        cfg.DATA_CONFIG.INFO_PATH = {'train': [], 'test': []}
        logger = logging.getLogger('social_nav.detector')
        self.dataset = CodaDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
                                   root_path=SOURCE/'data/wheelchair_weak', logger=logger)
        self.model = build_model(cfg.MODEL, len(cfg.CLASS_NAMES), self.dataset)
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(loaded, dict) or not isinstance(loaded.get('model_state'), dict):
            raise ValueError('Expected a trusted checkpoint with model_state')
        if 'max_sweeps' in loaded and loaded['max_sweeps'] != self.sweeps.max_sweeps:
            raise ValueError('Checkpoint temporal input does not match MAX_SWEEPS')
        state, matched = self.model._load_state_dict(loaded['model_state'], strict=False)
        missing = sorted(set(state)-set(matched))
        if missing:
            raise RuntimeError(f'Incomplete deployment weights ({len(matched)}/{len(state)}); missing: {missing[:8]}')
        self.precision = config['precision']
        validate_precision_support(self.precision)
        self.model.cuda().eval()
        self.load_data_to_gpu, self.autocast = load_data_to_gpu, autocast_context
        self.cfg = cfg
        self.metadata = {
            'backend': 'simpledet3d', 'checkpoint_sha256': file_sha256(checkpoint),
            'config_sha256': file_sha256(project_path(config['config'])),
            'resolved_config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            'intensity_normalization': float(cfg.DATA_CONFIG.get('INTENSITY_NORMALIZATION', 1)),
            'matched_parameters': len(matched), 'max_sweeps': self.sweeps.max_sweeps, 'precision': self.precision,
            'max_sweep_time_lag_s': self.sweeps.max_time_lag_s,
            'history_max_pose_age_s': self.sweeps.max_pose_age_s,
            'history_compensation': 'continuous_local_relative_pose',
            'candidate_score_threshold': float(cfg.MODEL.POST_PROCESSING.SCORE_THRESH),
            'point_cloud_range': list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            'extensions': extension_records,
        }

    def reset_history(self):
        self.sweeps.reset()
        self.last_input_context = None

    def prepare_frame(self, frame):
        cfg = self.cfg.DATA_CONFIG
        points, self.last_input_context = self.sweeps.prepare(frame,
            float(cfg.get('INTENSITY_NORMALIZATION', 1)), bool(cfg.get('ZERO_INTENSITY', False)))
        return points

    def predict_device(self, frame):
        """Keep detections on the selected device; host export is an explicit boundary."""
        points = self.prepare_frame(frame)
        data = self.dataset.prepare_data({'points': points, 'frame_id': frame.observation_id})
        if not len(data['points']):
            return self.torch.empty((0,8),device=next(self.model.parameters()).device)
        batch = self.dataset.collate_batch([data])
        if self.metrics is not None:
            excluded={'frame_id','metadata','calib','image_paths','ori_shape','img_process_infos','images'}
            self.metrics.transfer('h2d',sum(v.size*4 for k,v in batch.items() if isinstance(v,np.ndarray) and k not in excluded),'detector_batch')
        self.load_data_to_gpu(batch)
        # Match the existing extension-compatible test.py behavior, not inference_mode.
        with self.torch.no_grad(), self.autocast(self.precision):
            predictions, _ = self.model(batch)
        prediction = predictions[0]
        if not self.torch.all(prediction['pred_labels'] == 1):
            raise RuntimeError('Unexpected detector class ID')
        boxes = prediction['pred_boxes'].detach().float()
        scores = prediction['pred_scores'].detach().float()
        return self.torch.cat((boxes[:,:7],scores[:,None]),dim=1)

    def predict(self, frame):
        result=self.predict_device(frame).cpu().numpy()
        if self.metrics is not None:self.metrics.transfer('d2h',result.nbytes+1,'detection_boxes_and_class_guard')
        return validate_boxes(result)
