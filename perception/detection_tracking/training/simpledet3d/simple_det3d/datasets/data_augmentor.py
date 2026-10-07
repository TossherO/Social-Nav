"""Data augmentor with GT sampling, flip, rotation, scaling, and translation."""

from functools import partial
import copy
import pickle

import numpy as np
import torch

from ..utils import common_utils
from ..utils import box_utils
from ..ops.iou3d_nms import iou3d_nms_utils


GT_ALIGNED_KEYS = (
    'gt_cls_weights',
    'gt_reg_weights',
    'gt_box_reg_mask',
    'gt_supervision_types',
)


def filter_gt_aligned_fields(data_dict, mask):
    """Apply a GT selection mask to boxes, names, and weak-label metadata."""

    data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
    data_dict['gt_names'] = data_dict['gt_names'][mask]
    for key in GT_ALIGNED_KEYS:
        if key in data_dict:
            data_dict[key] = data_dict[key][mask]
    return data_dict


# ---------------------------------------------------------------------------
#  Helper functions (from augmentor_utils.py)
# ---------------------------------------------------------------------------

def random_flip_along_x(gt_boxes, points, return_flip=False, enable=None):
    """Flip along the x-axis (i.e., negate y coordinates)."""
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, return_flip=False, enable=None):
    """Flip along the y-axis (i.e., negate x coordinates)."""
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, return_rot=False, noise_rotation=None):
    """Apply a random global rotation around the z-axis."""
    if noise_rotation is None:
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(
        points[np.newaxis, :, :], np.array([noise_rotation])
    )[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(
        gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation])
    )[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]
    if return_rot:
        return gt_boxes, points, noise_rotation
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, return_scale=False):
    """Apply a random global scaling."""
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:] *= noise_scale
    if return_scale:
        return gt_boxes, points, noise_scale
    return gt_boxes, points


def global_scaling_with_roi_boxes(gt_boxes, roi_boxes, points, scale_range, return_scale=False):
    """Apply a random global scaling that also transforms roi_boxes."""
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, roi_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    roi_boxes[:, :, [0, 1, 2, 3, 4, 5, 7, 8]] *= noise_scale
    if return_scale:
        return gt_boxes, roi_boxes, points, noise_scale
    return gt_boxes, roi_boxes, points


# ---------------------------------------------------------------------------
#  DataBaseSampler
# ---------------------------------------------------------------------------

class DataBaseSampler(object):
    """Sample objects from a ground-truth database for copy-paste augmentation."""

    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            if not db_info_path.exists():
                if sampler_cfg.get('BACKUP_DB_INFO') is not None:
                    assert len(sampler_cfg.DB_INFO_PATH) == 1
                    sampler_cfg.DB_INFO_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_INFO_PATH']
                    sampler_cfg.DB_DATA_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_DATA_PATH']
                    db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH[0]
                    sampler_cfg.NUM_POINT_FEATURES = sampler_cfg.BACKUP_DB_INFO['NUM_POINT_FEATURES']
                else:
                    if self.logger is not None:
                        self.logger.warning(
                            f'GT database not found at {db_info_path} and no BACKUP_DB_INFO configured. '
                            f'gt_sampling will be disabled. Run tools/generate_gt_database.py first.'
                        )
                    continue

            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' %
                                 (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos
        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        sample_num = int(sample_group['sample_num'])
        pointer = sample_group['pointer']
        indices = sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer:pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        aligned_values = {
            key: data_dict[key][gt_boxes_mask]
            for key in GT_ALIGNED_KEYS if key in data_dict
        }
        points = data_dict['points']

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES]
            )
            if obj_points.shape[0] != info['num_points_in_gt']:
                obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(
                    -1, self.sampler_cfg.NUM_POINT_FEATURES
                )

            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                time_mask = np.logical_and(
                    obj_points[:, -1] < max_time + 1e-6,
                    obj_points[:, -1] > min_time - 1e-6
                )
                obj_points = obj_points[time_mask]
            elif obj_points.shape[-1] != points.shape[-1]:
                time_mask = np.logical_and(
                    obj_points[:, -1] < 1e-6, obj_points[:, -1] > -1e-6
                )
                obj_points = obj_points[time_mask]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        num_sampled = len(sampled_gt_boxes)
        if 'gt_cls_weights' in aligned_values:
            data_dict['gt_cls_weights'] = np.concatenate([
                aligned_values['gt_cls_weights'], np.ones(num_sampled, dtype=np.float32)
            ])
        if 'gt_reg_weights' in aligned_values:
            data_dict['gt_reg_weights'] = np.concatenate([
                aligned_values['gt_reg_weights'], np.ones(num_sampled, dtype=np.float32)
            ])
        if 'gt_box_reg_mask' in aligned_values:
            code_dim = aligned_values['gt_box_reg_mask'].shape[-1]
            data_dict['gt_box_reg_mask'] = np.concatenate([
                aligned_values['gt_box_reg_mask'],
                np.ones((num_sampled, code_dim), dtype=np.float32),
            ], axis=0)
        if 'gt_supervision_types' in aligned_values:
            data_dict['gt_supervision_types'] = np.concatenate([
                aligned_values['gt_supervision_types'],
                np.full(num_sampled, 3, dtype=np.int32),
            ])
        return data_dict

    def __call__(self, data_dict):
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                if len(sampled_dict) == 0:
                    continue

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                valid_mask = valid_mask.nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate(
                    (existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0
                )
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        if len(total_valid_sampled_dict) > 0:
            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict
            )
        else:
            data_dict = filter_gt_aligned_fields(
                data_dict, data_dict['gt_boxes_mask']
            )

        data_dict.pop('gt_boxes_mask')
        return data_dict


# ---------------------------------------------------------------------------
#  DataAugmentor
# ---------------------------------------------------------------------------

class DataAugmentor(object):
    """Apply data augmentation transforms to point clouds and boxes."""

    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def gt_sampling(self, config=None):
        db_sampler = DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            flip_func = globals()['random_flip_along_%s' % cur_axis]
            gt_boxes, points, enable = flip_func(gt_boxes, points, return_flip=True)
            data_dict['flip_%s' % cur_axis] = enable
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois, dim = data_dict['roi_boxes'].shape
                flip_func2 = globals()['random_flip_along_%s' % cur_axis]
                roi_boxes, _, _ = flip_func2(
                    data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]),
                    return_flip=True, enable=enable)
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois, dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois, dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = global_rotation(
                data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]),
                rot_range=rot_range, return_rot=True, noise_rotation=noise_rot
            )
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois, dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        if 'roi_boxes' in data_dict.keys():
            gt_boxes, roi_boxes, points, noise_scale = global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'],
                config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = global_scaling(
                data_dict['gt_boxes'], data_dict['points'],
                config['WORLD_SCALE_RANGE'], return_scale=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ], dtype=np.float32).T

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        points[:, :3] += noise_translate
        gt_boxes[:, :3] += noise_translate

        if 'roi_boxes' in data_dict.keys():
            data_dict['roi_boxes'][:, :, :3] += noise_translate

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_translate'] = noise_translate
        return data_dict

    def forward(self, data_dict):
        """Run all augmentation steps in order."""
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict = filter_gt_aligned_fields(data_dict, gt_boxes_mask)
            data_dict.pop('gt_boxes_mask')
        return data_dict
