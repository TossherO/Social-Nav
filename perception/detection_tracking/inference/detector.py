"""Lazy, single-frame SimpleDet3d adapter. No infos/GT/pickle dataset loading."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import json
from pathlib import Path
import sys
import sysconfig

import numpy as np

from .config import ROOT, project_path
from .geometry import validate_boxes

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
    points = points[np.all(np.isfinite(points), axis=1)].copy()
    if not len(points):
        raise ValueError('No finite input points; not a valid empty-scene observation')
    # Exact ego-square removal used by CodaDataset.get_lidar_with_sweeps.
    points = points[~((np.abs(points[:, 0]) < 1) & (np.abs(points[:, 1]) < 1))]
    points[:, 3] = 0 if zero_intensity else points[:, 3]/intensity_scale
    return np.c_[points, np.zeros(len(points), dtype=np.float32)].astype(np.float32)


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
    def __init__(self, config):
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
        if cfg.CLASS_NAMES != ['Pedestrian'] or cfg.DATA_CONFIG.MAX_SWEEPS != 1:
            raise ValueError('This deployment adapter requires Pedestrian and MAX_SWEEPS=1')
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
            'matched_parameters': len(matched), 'max_sweeps': 1, 'precision': self.precision,
            'candidate_score_threshold': float(cfg.MODEL.POST_PROCESSING.SCORE_THRESH),
            'point_cloud_range': list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            'extensions': extension_records,
        }

    def predict(self, frame):
        cfg = self.cfg.DATA_CONFIG
        points = prepare_single_frame(frame.points, float(cfg.get('INTENSITY_NORMALIZATION', 1)),
                                      bool(cfg.get('ZERO_INTENSITY', False)))
        data = self.dataset.prepare_data({'points': points, 'frame_id': frame.observation_id})
        if not len(data['points']):
            return np.empty((0, 8))
        batch = self.dataset.collate_batch([data])
        self.load_data_to_gpu(batch)
        # Match the existing extension-compatible test.py behavior, not inference_mode.
        with self.torch.no_grad(), self.autocast(self.precision):
            predictions, _ = self.model(batch)
        prediction = predictions[0]
        if not self.torch.all(prediction['pred_labels'] == 1):
            raise RuntimeError('Unexpected detector class ID')
        boxes = prediction['pred_boxes'].detach().float().cpu().numpy()
        scores = prediction['pred_scores'].detach().float().cpu().numpy()
        return validate_boxes(np.c_[boxes[:, :7], scores])
