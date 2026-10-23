"""NumPy-only adapter from portable bag center annotations to detector targets."""
import json
from pathlib import Path

import numpy as np


def _rigid(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-4)):
        raise ValueError('运动补偿变换不是有限刚体4x4矩阵')
    return matrix


def _cloud_pose(cloud, max_age_s):
    pose = cloud.get('pose')
    if not isinstance(pose, dict) or type(pose.get('valid')) is not bool:
        raise ValueError('点云缺少显式位姿有效性')
    if not pose['valid']:
        if pose.get('local_from_lidar') is not None or not pose.get('reason'):
            raise ValueError('无效位姿不能携带替代变换，须记录原因')
        return None
    age = (cloud['stamp_ns'] - pose['stamp_ns']) / 1e9
    if (pose.get('source_id') != cloud['source_id'] or pose.get('query_stamp_ns') != cloud['stamp_ns'] or
            pose.get('child_frame_id') != cloud['sensor_frame'] or
            not pose.get('frame_id') or not pose.get('session_id') or
            type(pose.get('generation')) is not int or pose['generation'] < 0 or
            not 0 <= age <= max_age_s or not np.isclose(age, pose['age_s'], atol=1e-9)):
        raise ValueError('点云位姿的来源、时刻、坐标或生命周期绑定无效')
    return _rigid(pose['local_from_lidar'])


def _motion_sweeps(sample, cloud, pointclouds, max_age_s):
    current = _cloud_pose(cloud, max_age_s)
    sweeps = []
    for history in sample['history']:
        past = pointclouds[history['pointcloud_id']]
        lag = (cloud['stamp_ns'] - past['stamp_ns']) / 1e9
        if (past['source_id'] != cloud['source_id'] or past['sensor_frame'] != cloud['sensor_frame'] or
                lag <= 0 or not np.isclose(lag, history['time_lag_s'], atol=1e-9)):
            raise ValueError('历史点云跨子包、坐标不一致或包含非因果时间')
        previous = _cloud_pose(past, max_age_s)
        same_identity = all(cloud['pose'].get(key) == past['pose'].get(key)
                            for key in ('source_id', 'frame_id', 'child_frame_id', 'session_id', 'generation'))
        expected_valid = current is not None and previous is not None and same_identity
        if 'current_from_history' not in history:
            raise ValueError('历史点云缺少运动补偿变换记录')
        matrix = history['current_from_history']
        if not expected_valid:
            if matrix is not None or not history.get('pose_reason'):
                raise ValueError('历史缺少有效同生命周期位姿，不得代入变换')
            continue
        if matrix is None or history.get('pose_reason') is not None:
            raise ValueError('有效历史位姿缺少对应变换')
        matrix = _rigid(matrix)
        expected = np.linalg.solve(current, previous)
        if not np.allclose(matrix, expected, atol=1e-5, rtol=1e-6):
            raise ValueError('历史到当前的变换与绑定的局部位姿不一致')
        sweeps.append(dict(lidar_path=past['path'], time_lag=lag, transform_matrix=matrix))
    return current, sweeps


def load_rosbag_centers(path, split, *, max_sweeps=1, point_cloud_range=None,
                       group_assignments=None, complete_only=False):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('format') != 'rosbag_people_centers' or document.get('unverified_policy') != 'ignore':
        raise ValueError('不是支持的rosbag人工中心数据集')
    if type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError('MAX_SWEEPS须为正整数')
    history_config = document.get('history', {})
    prepared = (history_config.get('transforms_available') is True and
                history_config.get('compensation') == 'continuous_local_relative_pose_on_load')
    if max_sweeps > 1 and not prepared:
        raise ValueError('导出历史点云未提供运动补偿变换；须先关联可靠定位结果')
    if history_config.get('motion_compensated'):
        raise ValueError('此适配器读取原始点云并在加载时补偿，不接受已变换点云')
    max_pose_age_s = float(history_config.get('max_pose_age_s', .02))
    if not np.isfinite(max_pose_age_s) or max_pose_age_s <= 0:
        raise ValueError('位姿配对时间窗无效')
    bounds = document['point_cloud_range']
    if point_cloud_range is not None and not np.allclose(bounds, point_cloud_range):
        raise ValueError('训练范围与人工复核范围不一致')
    sources = {s['id']: dict(s) for s in document['sources']}
    # Training configuration can assign an as-yet unsplit capture without
    # rewriting the frozen export or the human annotation database.
    for source in sources.values():
        assignment = (group_assignments or {}).get(source['root_group'])
        if assignment is None:
            continue
        for key in ('split', 'location_group'):
            value = assignment[key]
            if source[key] not in ('', 'unassigned', value):
                raise ValueError('不能覆盖已有来源划分或地点组：' + source['id'])
            source[key] = value
    pointclouds = {p['id']: p for p in document['pointclouds']}
    used_groups = {}
    for ident in {s['source_id'] for s in document['samples']}:
        source = sources[ident]
        if source['split'] not in ('train', 'val', 'holdout') or not source['root_group'] or not source['location_group']:
            raise ValueError('训练前须明确采集组、地点组和划分；未划分数据集仍可用于存储与查看')
        for kind in ('root_group', 'location_group'):
            group = kind, source[kind]
            if group in used_groups and used_groups[group] != source['split']:
                raise ValueError('同一采集／重复地点不能跨训练验证划分：' + source[kind])
            used_groups[group] = source['split']
    infos = []
    for sample in document['samples']:
        if sources[sample['source_id']]['split'] != split:
            continue
        if complete_only and sample['status'] != 'complete':
            continue
        annotation = sample['annotation']
        exhaustive = annotation['center_label_is_exhaustive']
        if (sample['status'] not in ('draft', 'complete') or
                exhaustive and (sample['status'] != 'complete' or annotation['ignored_candidate_count'])):
            raise ValueError('未核验候选或待标帧不能提供完整背景监督')
        centers = np.asarray([c['xy'] for c in annotation['centers']], dtype=np.float32).reshape(-1, 2)
        if (not np.isfinite(centers).all() or
                np.any(centers < np.asarray(bounds[:2])) or np.any(centers > np.asarray(bounds[3:5]))):
            raise ValueError('标注中心非有限或超出人工复核范围')
        count = len(centers)
        boxes = np.zeros((count, 7), dtype=np.float32)
        boxes[:, :2] = centers
        boxes[:, 3:6] = [.6, .6, 1.7]  # Internal proposal placeholders, never exported 3D truth.
        cloud = pointclouds[sample['pointcloud_id']]
        if cloud['source_id'] != sample['source_id']:
            raise ValueError('当前点云来源与标注不一致')
        current, sweeps = _motion_sweeps(sample, cloud, pointclouds, max_pose_age_s) if prepared else (None, [])
        infos.append(dict(token=sample['id'], scene_id=sample['source_id'], frame_index=sample['ordinal'],
                          root_group=sources[sample['source_id']]['root_group'],
                          location_group=sources[sample['source_id']]['location_group'],
                          bag_name=sources[sample['source_id']]['bag_name'],
                          lidar_path=cloud['path'], sweeps=sweeps, raw_history=sample['history'],
                          has_pose=current is not None, local_from_lidar=current,
                          motion_compensation_prepared=prepared, motion=sample.get('motion'),
                          pose_source=sources[sample['source_id']].get('pose_source'),
                          image_path=sample['image']['path'] if sample['image'] else None,
                          annotation_status=sample['status'], label_is_exhaustive=bool(exhaustive),
                          center_label_is_exhaustive=bool(exhaustive), is_target_domain=True,
                          roi_supervision_enabled=False, gt_boxes=boxes,
                          gt_names=np.full(count, 'Pedestrian', dtype='<U10'),
                          gt_cls_weights=np.ones(count, dtype=np.float32),
                          gt_reg_weights=np.ones(count, dtype=np.float32),
                          gt_box_reg_mask=np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (count, 1)),
                          gt_supervision_types=np.ones(count, dtype=np.int32), weak_centers=centers))
    if not infos:
        raise ValueError('数据集中没有所需划分的样本：' + str(split))
    return infos
