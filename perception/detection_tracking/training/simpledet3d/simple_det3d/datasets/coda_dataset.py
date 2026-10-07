"""CODa / Wheelchair dataset class for simple_det3d.

Supports both CODa and Wheelchair datasets differentiated by dataset_cfg,
including conservative center-only/partial-box supervision masks.
"""

import copy
import pickle
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from ..utils import common_utils, box_utils
from ..ops.iou3d_nms import iou3d_nms_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from .data_augmentor import DataAugmentor
from .data_processor import DataProcessor, PointFeatureEncoder


def _dataset_path(value):
    """Relative dataset configuration is rooted in this detector project."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


class CodaDataset(torch_data.Dataset):
    """Single dataset class for CODa and Wheelchair datasets.

    Loads point clouds from .bin files with optional multi-sweep support,
    applies data augmentation, point feature encoding, and voxelization.
    """

    def __init__(self, dataset_cfg, class_names, training=True,
                 root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.root_path = _dataset_path(root_path if root_path is not None else self.dataset_cfg.DATA_PATH)
        raw_data_path = self.dataset_cfg.get('RAW_DATA_PATH', None)
        self.raw_data_path = _dataset_path(raw_data_path).resolve() if raw_data_path else None
        if self.raw_data_path is not None and not self.raw_data_path.exists():
            self.logger.warning('Raw point-cloud root not found: %s', self.raw_data_path)
        self.infos = []

        # Point feature encoder
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        # Data augmentor (only in training)
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR,
            self.class_names, logger=self.logger
        ) if self.training else None

        # Data processor (voxelization etc.)
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size

        # Load data
        self.include_coda_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    def include_coda_data(self, mode):
        """Load pkl info files listed in dataset_cfg.INFO_PATH[mode]."""
        self.logger.info('Loading dataset info files for mode: %s', mode)
        coda_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                self.logger.warning('Info file not found: %s', info_path)
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                coda_infos.extend(infos)

        self.infos.extend(coda_infos)
        self.logger.info('Total samples loaded: %d', len(coda_infos))

    def balanced_infos_resampling(self, infos):
        """Class-balanced re-sampling (see https://arxiv.org/abs/1908.09492)."""
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum(len(v) for _, v in cls_infos.items())
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []
        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(cls_infos.values(), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()

        self.logger.info('Total samples after balanced resampling: %d', len(sampled_infos))
        return sampled_infos

    # ------------------------------------------------------------------
    #  Point cloud loading
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def _resolve_lidar_path(self, stored_path):
        """Resolve a point-cloud path stored in an info pkl.

        New info files store paths relative to ``RAW_DATA_PATH``.  Older info
        files may contain an absolute path from another machine; when that path
        no longer exists, retain the CODa point-cloud suffix (for example
        ``3d_comp/os1/0/file.bin``) and relocate it below ``RAW_DATA_PATH``.
        """
        lidar_path = Path(stored_path).expanduser()
        if not lidar_path.is_absolute():
            base_path = self.raw_data_path if self.raw_data_path is not None else self.root_path
            wheelchair_categories = {
                'indoor1', 'indoor2', 'outdoor1', 'outdoor2',
                'people', 'old1', 'old2',
            }
            if (
                len(lidar_path.parts) >= 2
                and lidar_path.parts[0] == 'data'
                and lidar_path.parts[1] in wheelchair_categories
            ):
                lidar_path = Path(*lidar_path.parts[1:])
            return base_path / lidar_path

        if self.raw_data_path is None:
            return lidar_path

        # Wheelchair weak-info files use paths relative to the raw root, while
        # the upstream pseudo-label file prefixes them with ``data/``.  Accept
        # both representations and relocate absolute paths from old machines.
        wheelchair_categories = {
            'indoor1', 'indoor2', 'outdoor1', 'outdoor2',
            'people', 'old1', 'old2',
        }
        for idx, part in enumerate(lidar_path.parts):
            if part in wheelchair_categories:
                return self.raw_data_path.joinpath(*lidar_path.parts[idx:])

        for point_cloud_dir in ('3d_comp', '3d_comp_downsample'):
            if point_cloud_dir in lidar_path.parts:
                suffix_start = lidar_path.parts.index(point_cloud_dir)
                return self.raw_data_path.joinpath(*lidar_path.parts[suffix_start:])

        if lidar_path.exists():
            return lidar_path

        # Last-resort compatibility for an old root that has the same basename
        # as the configured raw-data root (normally ``CODa_full``).
        matching_indices = [
            idx for idx, part in enumerate(lidar_path.parts)
            if part == self.raw_data_path.name
        ]
        if matching_indices:
            suffix_start = matching_indices[-1] + 1
            return self.raw_data_path.joinpath(*lidar_path.parts[suffix_start:])

        return lidar_path

    def get_sweep(self, sweep_info):
        """Load and transform a single sweep."""
        lidar_path = self._resolve_lidar_path(sweep_info['lidar_path'])
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])
        points_sweep = points_sweep[np.all(np.isfinite(points_sweep), axis=1)]
        points_sweep = self._remove_ego_points(points_sweep).T
        transform_matrix = np.asarray(sweep_info['transform_matrix'], dtype=np.float32)
        num_points = points_sweep.shape[1]
        points_sweep[:3, :] = transform_matrix.dot(
            np.vstack((points_sweep[:3, :], np.ones(num_points, dtype=np.float32)))
        )[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def _select_sweeps(self, info, max_sweeps):
        """Select valid history sweeps for one sample.

        ``MAX_SWEEPS`` counts the current frame, so at most
        ``max_sweeps - 1`` history frames are returned.  Validation is also
        performed here to protect training from stale or malformed entries in
        pkl files generated by older versions of the preprocessing script.
        """
        num_sweeps = max(0, int(max_sweeps) - 1)
        if num_sweeps == 0:
            return []

        max_time_lag = self.dataset_cfg.get('MAX_SWEEP_TIME_LAG', None)
        max_time_lag = float(max_time_lag) if max_time_lag is not None else np.inf
        valid_sweeps = []

        for sweep in info.get('sweeps', []):
            try:
                time_lag = float(sweep['time_lag'])
            except (KeyError, TypeError, ValueError):
                continue

            if not np.isfinite(time_lag) or time_lag <= 0 or time_lag > max_time_lag:
                continue

            transform_matrix = sweep.get('transform_matrix', None)
            if transform_matrix is None:
                continue
            transform_matrix = np.asarray(transform_matrix)
            if transform_matrix.shape != (4, 4) or not np.all(np.isfinite(transform_matrix)):
                continue

            valid_sweeps.append(sweep)

        select_mode = self.dataset_cfg.get('SWEEP_SELECT_MODE', 'near')
        if select_mode == 'near':
            selected = sorted(valid_sweeps, key=lambda x: float(x['time_lag']))[:num_sweeps]
        elif select_mode == 'random':
            num_selected = min(num_sweeps, len(valid_sweeps))
            if num_selected == 0:
                return []
            indices = np.random.choice(len(valid_sweeps), num_selected, replace=False)
            selected = [valid_sweeps[k] for k in indices]
        else:
            raise ValueError(
                f'Unsupported SWEEP_SELECT_MODE={select_mode!r}; expected "near" or "random"'
            )

        # Preserve chronological concatenation order (oldest to newest).  Point
        # order is not a temporal signal, but deterministic ordering simplifies
        # diagnostics and makes preprocessing reproducible when shuffle is off.
        return sorted(selected, key=lambda x: float(x['time_lag']), reverse=True)

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        """Load main lidar and merge with sweep frames."""
        info = self.infos[index]
        lidar_path = self._resolve_lidar_path(info['lidar_path'])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])
        points = points[np.all(np.isfinite(points), axis=1)]
        points = self._remove_ego_points(points)

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for sweep in self._select_sweeps(info, max_sweeps):
            points_sweep, times_sweep = self.get_sweep(sweep)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
        points = np.concatenate((points, times), axis=1)

        intensity_scale = self.dataset_cfg.get('INTENSITY_NORMALIZATION', None)
        if intensity_scale is not None:
            intensity_scale = float(intensity_scale)
            if not np.isfinite(intensity_scale) or intensity_scale <= 0:
                raise ValueError('INTENSITY_NORMALIZATION must be a positive finite value')
            points[:, 3] /= intensity_scale

        if self.dataset_cfg.get('ZERO_INTENSITY', False):
            points[:, 3] = 0
        return points

    # ------------------------------------------------------------------
    #  Core dataset methods
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(
            index, max_sweeps=self.dataset_cfg.get('MAX_SWEEPS', 1)
        )

        input_dict = {
            'points': points,
            # Basenames repeat in every Wheelchair scene.  Prefer the unique
            # converter token to avoid evaluation-map collisions.
            'frame_id': info.get('token', Path(info['lidar_path']).stem),
        }

        if 'gt_boxes' in info:
            num_gt = len(info['gt_boxes'])
            is_target_domain = bool(info.get('is_target_domain', False))
            center_label_is_exhaustive = bool(info.get(
                'center_label_is_exhaustive',
                info.get('label_is_exhaustive', True),
            ))
            weak_cfg = self.dataset_cfg.get('WEAK_SUPERVISION', {})
            background_weight = 0.0
            if center_label_is_exhaustive:
                background_weight = float(
                    weak_cfg.get('EXHAUSTIVE_BACKGROUND_WEIGHT', 1.0)
                    if is_target_domain else 1.0
                )
            roi_center_cfg = weak_cfg.get('ROI_CENTER_SUPERVISION', {})
            box_reg_mask = np.asarray(
                info.get('gt_box_reg_mask', np.ones((num_gt, 8))),
                dtype=np.float32,
            ).copy()
            if (
                is_target_domain
                and bool(weak_cfg.get('FORCE_CENTER_ONLY_REGRESSION', False))
                and box_reg_mask.shape[1] > 2
            ):
                box_reg_mask[:, 2:] = 0.0
            input_dict.update({
                'gt_names': info['gt_names'],
                'gt_boxes': np.asarray(info['gt_boxes'], dtype=np.float32),
                'gt_cls_weights': np.asarray(
                    info.get('gt_cls_weights', np.ones(num_gt)), dtype=np.float32
                ),
                'gt_reg_weights': np.asarray(
                    info.get('gt_reg_weights', np.ones(num_gt)), dtype=np.float32
                ),
                'gt_box_reg_mask': box_reg_mask,
                'gt_supervision_types': np.asarray(
                    info.get('gt_supervision_types', np.full(num_gt, 3)), dtype=np.int32
                ),
                'label_is_exhaustive': np.asarray(
                    center_label_is_exhaustive, dtype=np.float32
                ),
                'center_label_is_exhaustive': np.asarray(
                    center_label_is_exhaustive, dtype=np.float32
                ),
                'is_target_domain': np.asarray(
                    is_target_domain, dtype=np.float32
                ),
                'heatmap_background_weight': np.asarray(
                    background_weight, dtype=np.float32
                ),
                'roi_supervision_enabled': np.asarray(
                    bool(info.get('roi_supervision_enabled', True)), dtype=np.float32
                ),
                'roi_center_supervision_enabled': np.asarray(
                    is_target_domain
                    and center_label_is_exhaustive
                    and bool(roi_center_cfg.get('ENABLED', False)),
                    dtype=np.float32,
                ),
            })

        if 'lidar2global' in info:
            input_dict['lidar2global'] = info['lidar2global']

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def set_lidar_aug_matrix(self, data_dict):
        """Record the cumulative lidar augment matrix for each sample."""
        lidar_aug_matrix = np.eye(4)
        if data_dict.get('flip_x', False):
            lidar_aug_matrix[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3, :3]
        if data_dict.get('flip_y', False):
            lidar_aug_matrix[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3, :3]
        if 'noise_rot' in data_dict:
            noise_rot = data_dict['noise_rot']
            lidar_aug_matrix[:3, :3] = common_utils.angle2matrix(torch.tensor(noise_rot)) @ lidar_aug_matrix[:3, :3]
        if 'noise_scale' in data_dict:
            lidar_aug_matrix[:3, :3] *= data_dict['noise_scale']
        if 'noise_translate' in data_dict:
            lidar_aug_matrix[:3, 3:4] = data_dict['noise_translate'].T
        data_dict['lidar_aug_matrix'] = lidar_aug_matrix
        return data_dict

    def prepare_data(self, data_dict):
        """Full data preparation: augment, filter, encode, voxelize.

        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N,) string

        Returns:
            data_dict with voxels, voxel_coords, etc.
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_
            )
            data_dict = self.data_augmentor.forward(
                data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
            )

        data_dict = self.set_lidar_aug_matrix(data_dict)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            for key in (
                'gt_cls_weights', 'gt_reg_weights', 'gt_box_reg_mask',
                'gt_supervision_types',
            ):
                if key in data_dict:
                    data_dict[key] = data_dict[key][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32
            )
            gt_boxes = np.concatenate(
                (data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1
            )
            data_dict['gt_boxes'] = gt_boxes

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        allow_empty_gt = bool(self.dataset_cfg.get('ALLOW_EMPTY_GT', False))
        label_is_exhaustive = bool(data_dict.get('label_is_exhaustive', True))
        if self.training and len(data_dict['gt_boxes']) == 0 and not (
            allow_empty_gt or not label_is_exhaustive
        ):
            new_index = np.random.randint(len(self))
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        return data_dict

    # ------------------------------------------------------------------
    #  Batch collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """Collate a list of data dicts into a batched dict."""
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val = [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'gt_box_reg_mask']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_cls_weights', 'gt_reg_weights', 'gt_supervision_types']:
                    max_gt = max([len(x) for x in val])
                    batch_values = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_values[k, :len(val[k])] = val[k]
                    ret[key] = batch_values
                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, val[0].shape[0], max_gt), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except Exception:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret

    # ------------------------------------------------------------------
    #  Evaluation (AP3D and APBEV)
    # ------------------------------------------------------------------

    def evaluation(self, det_annos, class_names, **kwargs):
        """Self-contained AP calculation for 3D and BEV.

        Args:
            det_annos: list of dicts with keys 'name', 'boxes_lidar', 'score', 'frame_id'
            class_names: list of class name strings

        Returns:
            result_str: formatted string with per-class and mean AP
            result_dict: dict with AP3D and APBEV per-class values
        """
        weak_cfg = self.dataset_cfg.get('WEAK_SUPERVISION', {})
        if weak_cfg.get('CENTER_EVAL', False):
            return self._evaluation_center_only(det_annos, weak_cfg)

        if len(self.infos) == 0 or 'gt_boxes' not in self.infos[0]:
            return 'No ground-truth boxes for evaluation', {}

        iou_thresholds = {
            'Car': 0.7,
            'Pedestrian': 0.5,
            'Cyclist': 0.5
        }

        # Build per-frame ground truth lookup
        # Only keep GT boxes whose center falls within point_cloud_range,
        # consistent with training-time filtering (USE_CENTER_TO_FILTER=True).
        gt_by_frame = {}
        for info in self.infos:
            frame_id = info.get('token', Path(info['lidar_path']).stem)
            gt_names = np.array(info['gt_names'])
            gt_boxes = np.array(info['gt_boxes'])
            mask = np.array([n in class_names for n in gt_names])
            if len(gt_boxes) > 0 and mask.any():
                range_mask = box_utils.mask_boxes_outside_range_numpy(
                    gt_boxes[mask], self.point_cloud_range,
                    use_center_to_filter=True
                )
                mask[mask] = range_mask
            gt_by_frame[frame_id] = {
                'names': gt_names[mask],
                'boxes': gt_boxes[mask]
            }

        def compute_iou_mats(boxes_pred, boxes_gt):
            if boxes_pred.shape[0] == 0 or boxes_gt.shape[0] == 0:
                empty = np.zeros((boxes_pred.shape[0], boxes_gt.shape[0]), dtype=np.float32)
                return empty, empty

            bev_iou = iou3d_nms_utils.boxes_bev_iou_cpu(boxes_pred, boxes_gt)

            area_pred = (boxes_pred[:, 3] * boxes_pred[:, 4])[:, None]
            area_gt = (boxes_gt[:, 3] * boxes_gt[:, 4])[None, :]
            overlap_area = bev_iou * (area_pred + area_gt) / np.maximum(1.0 + bev_iou, 1e-7)

            zmax_pred = (boxes_pred[:, 2] + boxes_pred[:, 5] * 0.5)[:, None]
            zmin_pred = (boxes_pred[:, 2] - boxes_pred[:, 5] * 0.5)[:, None]
            zmax_gt = (boxes_gt[:, 2] + boxes_gt[:, 5] * 0.5)[None, :]
            zmin_gt = (boxes_gt[:, 2] - boxes_gt[:, 5] * 0.5)[None, :]
            overlap_h = np.maximum(0.0, np.minimum(zmax_pred, zmax_gt) - np.maximum(zmin_pred, zmin_gt))

            overlap_vol = overlap_area * overlap_h
            vol_pred = (boxes_pred[:, 3] * boxes_pred[:, 4] * boxes_pred[:, 5])[:, None]
            vol_gt = (boxes_gt[:, 3] * boxes_gt[:, 4] * boxes_gt[:, 5])[None, :]
            iou_3d = overlap_vol / np.maximum(vol_pred + vol_gt - overlap_vol, 1e-7)
            return bev_iou, iou_3d

        def compute_ap(tp, fp, num_gt):
            if num_gt == 0:
                return 0.0
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / float(num_gt)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)

            recall_levels = np.linspace(0.0, 1.0, 41)
            precisions = []
            for rl in recall_levels:
                mask = recall >= rl
                precisions.append(np.max(precision[mask]) if mask.any() else 0.0)
            return float(np.mean(precisions))

        results = {'AP3D': {}, 'APBEV': {}}

        for cls_name in class_names:
            cls_thresh = iou_thresholds.get(cls_name, 0.5)
            preds_cls = []
            for det in det_annos:
                names = np.array(det['name'])
                boxes = np.array(det['boxes_lidar'])
                scores = np.array(det['score'])
                mask = names == cls_name
                if not np.any(mask):
                    continue
                for box, score in zip(boxes[mask], scores[mask]):
                    preds_cls.append((det['frame_id'], float(score), box))

            preds_cls.sort(key=lambda x: x[1], reverse=True)
            gt_counts = 0
            gt_used = {}
            for frame_id, gt_data in gt_by_frame.items():
                cls_mask = gt_data['names'] == cls_name
                gt_boxes_cls = gt_data['boxes'][cls_mask]
                gt_counts += len(gt_boxes_cls)
                gt_used[frame_id] = np.zeros(len(gt_boxes_cls), dtype=bool)

            tp_bev, fp_bev, tp_3d, fp_3d = [], [], [], []
            for frame_id, score, pred_box in preds_cls:
                gt_data = gt_by_frame.get(frame_id, {'names': np.array([]), 'boxes': np.zeros((0, 7))})
                cls_mask = gt_data['names'] == cls_name
                gt_boxes_cls = gt_data['boxes'][cls_mask]
                used_flags = gt_used.get(frame_id)
                if gt_boxes_cls.shape[0] == 0:
                    fp_bev.append(1.0)
                    fp_3d.append(1.0)
                    tp_bev.append(0.0)
                    tp_3d.append(0.0)
                    continue
                bev_iou_mat, iou3d_mat = compute_iou_mats(pred_box[None, :], gt_boxes_cls)
                bev_best_idx = int(np.argmax(bev_iou_mat[0]))
                bev_best = bev_iou_mat[0, bev_best_idx]
                iou3d_best = iou3d_mat[0, bev_best_idx]
                matched = used_flags is not None and used_flags[bev_best_idx]
                if bev_best >= cls_thresh and not matched:
                    tp_bev.append(1.0)
                    fp_bev.append(0.0)
                else:
                    tp_bev.append(0.0)
                    fp_bev.append(1.0)
                if iou3d_best >= cls_thresh and not matched:
                    tp_3d.append(1.0)
                    fp_3d.append(0.0)
                    if used_flags is not None:
                        used_flags[bev_best_idx] = True
                else:
                    tp_3d.append(0.0)
                    fp_3d.append(1.0)

            tp_bev = np.array(tp_bev)
            fp_bev = np.array(fp_bev)
            tp_3d = np.array(tp_3d)
            fp_3d = np.array(fp_3d)
            results['APBEV'][cls_name] = compute_ap(tp_bev, fp_bev, gt_counts)
            results['AP3D'][cls_name] = compute_ap(tp_3d, fp_3d, gt_counts)

        ap_result_lines = []
        for key in ['AP3D', 'APBEV']:
            ap_result_lines.append(f"{key}:")
            ap_vals = []
            for cls_name in class_names:
                ap_val = results[key][cls_name] * 100
                ap_vals.append(ap_val)
                ap_result_lines.append(f"  {cls_name}: {ap_val:.2f}")
            ap_result_lines.append(f"  mAP: {np.mean(ap_vals):.2f}")

        return '\n'.join(ap_result_lines), results

    @staticmethod
    def _greedy_center_matches(pred_xy, gt_xy, threshold):
        """Return one-to-one greedy center matches within ``threshold``."""

        pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
        gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
        if len(pred_xy) == 0 or len(gt_xy) == 0:
            return []
        distances = np.linalg.norm(
            pred_xy[:, None, :] - gt_xy[None, :, :], axis=-1
        )
        matches = []
        while distances.size:
            pred_idx, gt_idx = np.unravel_index(
                int(np.argmin(distances)), distances.shape
            )
            distance = float(distances[pred_idx, gt_idx])
            if distance > threshold:
                break
            matches.append((distance, pred_idx, gt_idx))
            distances[pred_idx, :] = np.inf
            distances[:, gt_idx] = np.inf
        return matches

    def _evaluation_center_only(self, det_annos, weak_cfg):
        """Evaluate center labels, including precision on completed frames.

        Frames explicitly marked ``center_label_is_exhaustive`` use the union
        of manual and original pseudo-label targets stored in ``gt_boxes``.
        Other frames retain the old recall-only manual-center evaluation.
        """

        thresholds = [float(x) for x in weak_cfg.get(
            'CENTER_HIT_THRESHOLDS', [0.5, 1.0]
        )]
        score_thresh = float(weak_cfg.get('CENTER_SCORE_THRESH', 0.1))
        gt_by_frame = {}
        exhaustive_frames = set()
        total_manual_centers = 0
        outside_range_centers = 0
        for info in self.infos:
            is_exhaustive = bool(info.get('center_label_is_exhaustive', False))
            if is_exhaustive:
                boxes = np.asarray(
                    info.get('gt_boxes', np.empty((0, 7))), dtype=np.float32
                )
                centers = boxes[:, :2] if len(boxes) else np.empty((0, 2), dtype=np.float32)
            else:
                centers = np.asarray(
                    info.get('weak_centers', []), dtype=np.float32
                ).reshape(-1, 2)
                if len(centers) == 0:
                    continue
            total_manual_centers += len(centers)
            finite = np.all(np.isfinite(centers), axis=1)
            in_range = (
                finite
                & (centers[:, 0] >= self.point_cloud_range[0])
                & (centers[:, 0] <= self.point_cloud_range[3])
                & (centers[:, 1] >= self.point_cloud_range[1])
                & (centers[:, 1] <= self.point_cloud_range[4])
            )
            outside_range_centers += int((~in_range).sum())
            frame_id = str(info.get('token', Path(info['lidar_path']).stem))
            if is_exhaustive:
                exhaustive_frames.add(frame_id)
                gt_by_frame[frame_id] = centers[in_range]
            elif in_range.any():
                gt_by_frame[frame_id] = centers[in_range]

        hits = {threshold: 0 for threshold in thresholds}
        exhaustive_hits = {threshold: 0 for threshold in thresholds}
        exhaustive_predictions = {threshold: 0 for threshold in thresholds}
        exhaustive_gt = sum(
            len(gt_by_frame.get(frame_id, [])) for frame_id in exhaustive_frames
        )
        total_gt = sum(len(centers) for centers in gt_by_frame.values())
        total_predictions = 0
        matched_distances = []

        for anno in det_annos:
            frame_id = str(anno['frame_id'])
            gt_xy = gt_by_frame.get(frame_id)
            if gt_xy is None:
                continue
            names = np.asarray(anno.get('name', []), dtype=str)
            scores = np.asarray(anno.get('score', []), dtype=np.float32)
            boxes = np.asarray(anno.get('boxes_lidar', []), dtype=np.float32)
            if boxes.size == 0:
                boxes = np.empty((0, 7), dtype=np.float32)
            elif boxes.ndim != 2 or boxes.shape[1] < 2:
                raise ValueError('boxes_lidar must have shape (N, >=2)')
            pred_mask = (names == 'Pedestrian') & (scores >= score_thresh)
            pred_xy = boxes[pred_mask, :2]
            total_predictions += len(pred_xy)
            for threshold in thresholds:
                matches = self._greedy_center_matches(pred_xy, gt_xy, threshold)
                hits[threshold] += len(matches)
                if frame_id in exhaustive_frames:
                    exhaustive_hits[threshold] += len(matches)
                    exhaustive_predictions[threshold] += len(pred_xy)
                if threshold == max(thresholds):
                    matched_distances.extend(match[0] for match in matches)

        result_dict = {
            f'CenterHit@{threshold:.2f}m': hits[threshold] / max(total_gt, 1)
            for threshold in thresholds
        }
        result_dict.update({
            'CenterMeanError': float(np.mean(matched_distances))
            if matched_distances else float('nan'),
            'CenterMedianError': float(np.median(matched_distances))
            if matched_distances else float('nan'),
            'ManualCenterCount': int(total_gt),
            'ManualCenterOutsideRange': int(outside_range_centers),
            'ManualCenterRangeCoverage': total_gt / max(total_manual_centers, 1),
            'LabeledCenterCount': int(total_gt),
            'LabeledCenterOutsideRange': int(outside_range_centers),
            'LabeledCenterRangeCoverage': total_gt / max(total_manual_centers, 1),
            'PredictionCountOnLabeledFrames': int(total_predictions),
        })
        if exhaustive_frames:
            for threshold in thresholds:
                tp = exhaustive_hits[threshold]
                num_pred = exhaustive_predictions[threshold]
                precision = tp / max(num_pred, 1)
                recall = tp / max(exhaustive_gt, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                result_dict.update({
                    f'CenterPrecision@{threshold:.2f}m': precision,
                    f'CenterRecall@{threshold:.2f}m': recall,
                    f'CenterF1@{threshold:.2f}m': f1,
                    f'CenterFalsePositiveCount@{threshold:.2f}m': int(num_pred - tp),
                })
            primary_threshold = thresholds[0]
            primary_fp = result_dict[
                f'CenterFalsePositiveCount@{primary_threshold:.2f}m'
            ]
            result_dict.update({
                'ExhaustiveCenterCount': int(exhaustive_gt),
                'ExhaustiveFrameCount': int(len(exhaustive_frames)),
                'CenterFalsePositivesPerFrame': (
                    primary_fp / max(len(exhaustive_frames), 1)
                ),
            })
            lines = ['Wheelchair exhaustive-center evaluation (3D AP unavailable):']
        else:
            lines = ['Weak Wheelchair center evaluation (precision/AP unavailable):']
        for threshold in thresholds:
            key = f'CenterHit@{threshold:.2f}m'
            lines.append(f'  {key}: {result_dict[key]:.4f}')
            precision_key = f'CenterPrecision@{threshold:.2f}m'
            if precision_key in result_dict:
                lines.append(
                    f'  Center P/R/F1@{threshold:.2f}m: '
                    f'{result_dict[precision_key]:.4f} / '
                    f'{result_dict[f"CenterRecall@{threshold:.2f}m"]:.4f} / '
                    f'{result_dict[f"CenterF1@{threshold:.2f}m"]:.4f}'
                )
        lines.extend([
            '  Center mean/median error: '
            f'{result_dict["CenterMeanError"]:.4f} / '
            f'{result_dict["CenterMedianError"]:.4f} m',
            f'  Labeled centers: {total_gt}',
            f'  Labeled centers outside model range: {outside_range_centers}',
            f'  Predictions on labeled frames: {total_predictions}',
        ])
        if exhaustive_frames:
            lines.extend([
                f'  Exhaustive center frames/targets: '
                f'{len(exhaustive_frames)} / {exhaustive_gt}',
                f'  False positives/frame @{thresholds[0]:.2f}m: '
                f'{result_dict["CenterFalsePositivesPerFrame"]:.4f}',
            ])
        return '\n'.join(lines), result_dict

    # ------------------------------------------------------------------
    #  Prediction dict generation
    # ------------------------------------------------------------------

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """Convert model output to standard annotation format.

        Args:
            batch_dict: contains 'frame_id'
            pred_dicts: list of dicts with 'pred_boxes', 'pred_scores', 'pred_labels'
            class_names: list of class name strings
            output_path: optional path for saving

        Returns:
            annos: list of annotation dicts
        """

        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]),
                'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            # Keep serialized predictions and NumPy evaluation stable across
            # fp32/fp16/bf16 inference modes.
            pred_scores = box_dict['pred_scores'].detach().float().cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].detach().float().cpu().numpy()
            pred_labels = box_dict['pred_labels'].detach().cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    # ------------------------------------------------------------------
    #  Ground-truth database creation (for GT sampling)
    # ------------------------------------------------------------------

    def create_groundtruth_database(self, info_path=None, used_classes=None,
                                    split='train', max_sweeps=None, save_name=None):
        """Create a ground-truth database of per-object point clouds.

        This is called once offline to build the database used by
        DataBaseSampler during training for GT-sampling augmentation.
        """
        import torch as _torch

        infos = self.infos if info_path is None else None
        if info_path is not None:
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

        used_classes = self.class_names if used_classes is None else used_classes
        max_sweeps = self.dataset_cfg.get('MAX_SWEEPS', 1) if max_sweeps is None else max_sweeps
        database_dir_name = 'gt_database_%s' % save_name if save_name is not None else (
            'gt_database' if split == 'train' else ('gt_database_%s' % split)
        )
        db_info_file_name = (
            '%s_dbinfos_%s.pkl' % (save_name, split) if save_name is not None
            else ('coda_dbinfos_%s.pkl' % split)
        )
        database_save_path = Path(self.root_path) / database_dir_name
        db_info_save_path = Path(self.root_path) / db_info_file_name

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        original_infos = self.infos
        self.infos = infos

        try:
            for k in range(len(infos)):
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
                info = infos[k]
                sample_idx = info['token']
                points = self.get_lidar_with_sweeps(k, max_sweeps=max_sweeps)

                if 'annos' in info:
                    annos = info['annos']
                    names = annos['name']
                    gt_boxes = annos.get('gt_boxes', annos.get('gt_boxes_lidar'))
                else:
                    names = info['gt_names']
                    gt_boxes = info['gt_boxes']

                gt_boxes = np.asarray(gt_boxes)
                if gt_boxes.shape[1] > 7:
                    gt_boxes = gt_boxes[:, :7]

                num_obj = gt_boxes.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    _torch.from_numpy(points[:, 0:3]), _torch.from_numpy(gt_boxes)
                ).numpy()

                for i in range(num_obj):
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]

                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'wb') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))
                        db_info = {
                            'name': names[i],
                            'path': db_path,
                            'image_idx': sample_idx,
                            'gt_idx': i,
                            'box3d_lidar': gt_boxes[i],
                            'num_points_in_gt': gt_points.shape[0]
                        }
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
        finally:
            self.infos = original_infos

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
