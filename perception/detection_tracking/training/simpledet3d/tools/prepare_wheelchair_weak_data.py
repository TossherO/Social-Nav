#!/usr/bin/env python3
"""Convert Wheelchair pseudo/point annotations into weak-training info files.

The converter never treats a rejected pseudo box as background.  Geometry that
fails the conservative box filter is downgraded to center-only supervision when
its center remains finite; otherwise it is ignored.

This first implementation intentionally does not promote boxes with detector
Teacher predictions.  It produces a safe offline bootstrap dataset that can be
refreshed by the EMA/Teacher stage later.
"""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simple_det3d.datasets.weak_label_utils import (  # noqa: E402
    PedestrianBoxFilterConfig,
    SUPERVISION_BOX,
    SUPERVISION_CENTER,
    SUPERVISION_IGNORE,
    classify_pseudo_box,
    temporal_support_counts,
)
from tools.generate_wheelchair_info import parse_pose_file  # noqa: E402


MANUAL_PATTERN = re.compile(r'^annotations_(indoor[12])_(.+)\.pkl$')
DEFAULT_CATEGORIES = ('indoor1', 'indoor2', 'outdoor1', 'outdoor2', 'people')
DEFAULT_POINT_CLOUD_RANGE = (-4.8, -9.6, -1.5, 14.4, 9.6, 4.5)
DEFAULT_PEDESTRIAN_SIZE = (0.8, 0.6, 1.73)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare conservatively filtered Wheelchair weak labels'
    )
    parser.add_argument(
        '--data-root', default=str(PROJECT_ROOT / 'data/wheelchair_raw'),
        help='Wheelchair raw-data root containing indoor1/indoor2/...',
    )
    parser.add_argument(
        '--annotation-root', default=str(PROJECT_ROOT / 'data/wheelchair_annotations'),
        help='Directory containing the scene pkl and annotations_*.pkl files',
    )
    parser.add_argument(
        '--pseudo-info', default='wheelchair_infos_train_full_by_scene.pkl',
        help='Pseudo-label pkl, absolute or relative to --annotation-root',
    )
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'output/data_preparation/wheelchair_weak'))
    parser.add_argument(
        '--categories', nargs='+', default=list(DEFAULT_CATEGORIES),
        help='Raw-data category directories to include',
    )
    parser.add_argument(
        '--scene', action='append', default=None,
        help=(
            'Restrict conversion to an exact scene name or category/scene; '
            'repeatable. Omit to process every scene in the selected categories.'
        ),
    )
    parser.add_argument('--weak-val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--frame-rate', type=float, default=10.0)
    parser.add_argument(
        '--sweep-frame-len', type=int, default=3,
        help='Number of history frames; 3 produces a four-frame input',
    )
    parser.add_argument('--max-sweep-time-lag', type=float, default=0.35)
    parser.add_argument(
        '--point-cloud-range', type=float, nargs=6,
        default=list(DEFAULT_POINT_CLOUD_RANGE),
    )
    parser.add_argument('--pseudo-center-weight', type=float, default=0.2)
    parser.add_argument('--pseudo-box-weight', type=float, default=0.3)
    parser.add_argument('--manual-center-weight', type=float, default=1.0)
    parser.add_argument(
        '--exhaustive-pseudo-cls-min', type=float, default=0.5,
        help=(
            'Minimum classification weight for original pseudo labels on '
            'manually completed frames; regression weights are unchanged'
        ),
    )
    parser.add_argument('--min-horizontal-size', type=float, default=0.15)
    parser.add_argument('--max-horizontal-size', type=float, default=1.50)
    parser.add_argument('--min-height', type=float, default=0.60)
    parser.add_argument('--max-height', type=float, default=2.40)
    parser.add_argument('--near-distance', type=float, default=8.0)
    parser.add_argument('--far-distance', type=float, default=15.0)
    parser.add_argument('--near-min-points', type=int, default=5)
    parser.add_argument('--mid-min-points', type=int, default=3)
    parser.add_argument('--far-min-points', type=int, default=1)
    parser.add_argument('--min-bidirectional-box-score', type=float, default=0.60)
    parser.add_argument('--temporal-base-distance', type=float, default=0.25)
    parser.add_argument('--max-pedestrian-speed', type=float, default=4.0)
    parser.add_argument(
        '--static-no-pose-categories', nargs='+', default=['indoor1', 'people'],
        help='Categories known to have a stationary platform when pose is absent',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Scene-level preprocessing threads (point files are I/O-bound)',
    )
    parser.add_argument(
        '--max-frames-per-scene', type=int, default=None,
        help='Diagnostic limit; omit for complete conversion',
    )
    return parser.parse_args()


def _resolve_input_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path


def load_pseudo_scenes(pseudo_path):
    with Path(pseudo_path).open('rb') as f:
        scene_entries = pickle.load(f)

    pseudo_by_frame = {}
    frame_indices_by_scene = {}
    for scene_entry in scene_entries:
        category = str(scene_entry['scene_dir'])
        scene = str(scene_entry['scene'])
        scene_key = (category, scene)
        if scene_key in frame_indices_by_scene:
            raise ValueError(f'Duplicate scene in pseudo info: {category}/{scene}')
        frame_indices = []
        for frame in scene_entry.get('frames', []):
            frame_idx = int(frame['frame_index'])
            frame_key = (category, scene, frame_idx)
            if frame_key in pseudo_by_frame:
                raise ValueError(
                    f'Duplicate frame in pseudo info: {category}/{scene}/{frame_idx:04d}'
                )
            pseudo_by_frame[frame_key] = frame
            frame_indices.append(frame_idx)
        frame_indices_by_scene[scene_key] = tuple(frame_indices)
    return pseudo_by_frame, frame_indices_by_scene


def load_manual_centers(
    annotation_root, frame_indices_by_scene, return_frame_keys=False,
):
    """Load manual centers using annotation keys as source-frame list positions."""
    manual_by_frame = defaultdict(list)
    manual_scenes = set()
    manual_frame_keys = set()
    for path in sorted(Path(annotation_root).glob('annotations_*.pkl')):
        match = MANUAL_PATTERN.match(path.name)
        if match is None:
            continue
        category, scene = match.groups()
        scene_key = (category, scene)
        if scene_key not in frame_indices_by_scene:
            raise ValueError(
                f'Manual annotation scene is absent from pseudo info: '
                f'{category}/{scene} ({path})'
            )
        frame_indices = frame_indices_by_scene[scene_key]
        manual_scenes.add(scene_key)
        with path.open('rb') as f:
            annotations = pickle.load(f)
        for raw_frame_key, objects in annotations.items():
            try:
                frame_position = int(raw_frame_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Invalid manual frame-list index {raw_frame_key!r} in {path}'
                ) from exc
            if not 0 <= frame_position < len(frame_indices):
                raise ValueError(
                    f'Manual frame-list index {frame_position} is outside [0, '
                    f'{len(frame_indices)}) for {category}/{scene} ({path})'
                )
            frame_idx = frame_indices[frame_position]
            frame_key = (category, scene, frame_idx)
            manual_frame_keys.add(frame_key)
            for obj in objects:
                if len(obj) < 4 or str(obj[3]) != 'Pedestrian':
                    continue
                xy = np.asarray(obj[:2], dtype=np.float32)
                if np.all(np.isfinite(xy)):
                    manual_by_frame[frame_key].append(xy)

    manual_by_frame = {
        key: np.asarray(value, dtype=np.float32).reshape(-1, 2)
        for key, value in manual_by_frame.items()
    }
    if return_frame_keys:
        return manual_by_frame, manual_scenes, manual_frame_keys
    return manual_by_frame, manual_scenes


def split_manual_scenes(manual_scenes, val_ratio, seed):
    if not 0 <= val_ratio < 1:
        raise ValueError('--weak-val-ratio must be in [0, 1)')
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for category, scene in sorted(manual_scenes):
        grouped[category].append(scene)

    val_scenes = set()
    if val_ratio == 0:
        return val_scenes
    for category, scenes in grouped.items():
        scenes = list(scenes)
        rng.shuffle(scenes)
        num_val = max(1, int(round(len(scenes) * val_ratio)))
        num_val = min(num_val, max(0, len(scenes) - 1))
        val_scenes.update((category, scene) for scene in scenes[:num_val])
    return val_scenes


def load_scene_poses(scene_dir, frame_indices, static_without_pose):
    poses_dir = scene_dir / 'poses'
    poses = {}
    pose_found = {}
    pose_usable = {}
    for frame_idx in frame_indices:
        pose = None
        if poses_dir.exists():
            pose = parse_pose_file(poses_dir / f'{frame_idx:04d}.txt')
        valid = pose is not None
        if pose is None:
            pose = np.eye(4, dtype=np.float64)
        poses[frame_idx] = np.asarray(pose, dtype=np.float32)
        pose_found[frame_idx] = valid
        pose_usable[frame_idx] = valid or static_without_pose
    return poses, pose_found, pose_usable


def build_sweeps(
    frame_idx,
    frame_indices,
    relative_paths,
    poses,
    pose_valid,
    sweep_frame_len,
    frame_rate,
    max_time_lag,
):
    if sweep_frame_len <= 0:
        return []
    frame_position = bisect.bisect_left(frame_indices, frame_idx)
    histories = frame_indices[max(0, frame_position - sweep_frame_len):frame_position]
    sweeps = []
    for history_idx in histories:
        frame_gap = frame_idx - history_idx
        time_lag = frame_gap / frame_rate
        if frame_gap <= 0 or time_lag > max_time_lag:
            continue
        if not (pose_valid[frame_idx] and pose_valid[history_idx]):
            continue
        transform = np.linalg.inv(poses[frame_idx].astype(np.float64)) @ poses[history_idx]
        if not np.all(np.isfinite(transform)):
            continue
        sweeps.append({
            'frame_id': history_idx,
            'lidar_path': relative_paths[history_idx],
            'time_lag': float(time_lag),
            'transform_matrix': transform.astype(np.float32),
        })
    return sweeps


def _pseudo_arrays(frame):
    if frame is None:
        return {
            'name': np.empty(0, dtype=str),
            'center_3d': np.empty((0, 3), dtype=np.float32),
            'bbox_3d': np.empty((0, 7), dtype=np.float32),
            'score': np.empty(0, dtype=np.float32),
        }
    annos = frame.get('pseudo_annos', {})
    names = np.asarray(annos.get('name', []), dtype=str)
    count = len(names)
    centers = np.asarray(annos.get('center_3d', np.empty((count, 3))), dtype=np.float32).reshape(count, 3)
    boxes = np.asarray(annos.get('bbox_3d', np.empty((count, 7))), dtype=np.float32).reshape(count, 7)
    scores = np.asarray(annos.get('score', np.zeros(count)), dtype=np.float32).reshape(count)
    return {'name': names, 'center_3d': centers, 'bbox_3d': boxes, 'score': scores}


def compute_scene_temporal_support(
    category, scene, frame_indices, poses, pose_usable,
    pseudo_by_frame, cfg, frame_rate,
):
    frame_centers = {}
    frame_poses = {}
    pedestrian_original_indices = {}
    for frame_idx in frame_indices:
        arrays = _pseudo_arrays(pseudo_by_frame.get((category, scene, frame_idx)))
        keep = np.flatnonzero(arrays['name'] == 'Pedestrian')
        pedestrian_original_indices[frame_idx] = keep
        frame_centers[frame_idx] = arrays['center_3d'][keep] if pose_usable[frame_idx] \
            else np.empty((0, 3), dtype=np.float32)
        frame_poses[frame_idx] = poses[frame_idx]

    compact_support = temporal_support_counts(
        frame_centers, frame_poses, frame_rate, cfg, max_frame_gap=2
    )
    support = {}
    for frame_idx, original_indices in pedestrian_original_indices.items():
        for compact_idx, original_idx in enumerate(original_indices):
            support[(frame_idx, int(original_idx))] = compact_support.get((frame_idx, compact_idx), 0)
    return support


def _placeholder_box(center, default_size):
    box = np.zeros(7, dtype=np.float32)
    box[:3] = np.asarray(center, dtype=np.float32)[:3]
    box[3:6] = np.asarray(default_size, dtype=np.float32)
    return box


def make_weak_targets(
    pseudo_frame,
    manual_centers,
    temporal_support,
    points,
    cfg,
    point_cloud_range,
    pseudo_center_weight,
    pseudo_box_weight,
    manual_center_weight,
    label_is_exhaustive=False,
    exhaustive_pseudo_cls_min=0.5,
):
    arrays = _pseudo_arrays(pseudo_frame)
    manual_centers = np.asarray(manual_centers, dtype=np.float32).reshape(-1, 2)
    targets = []
    frame_stats = Counter()

    for obj_idx, name in enumerate(arrays['name']):
        if name != 'Pedestrian':
            frame_stats['non_pedestrian_ignored'] += 1
            continue

        raw_box = arrays['bbox_3d'][obj_idx]
        center = arrays['center_3d'][obj_idx]
        if not np.all(np.isfinite(center)):
            frame_stats['invalid_center'] += 1
            continue

        has_positive_size = np.all(np.isfinite(raw_box)) and np.all(raw_box[3:6] > 0)
        if has_positive_size:
            supervision, diagnostics = classify_pseudo_box(
                raw_box,
                points,
                temporal_support.get(obj_idx, 0),
                cfg,
                point_cloud_range,
                pseudo_score=float(arrays['score'][obj_idx]),
            )
        else:
            supervision = SUPERVISION_CENTER
            diagnostics = {
                'reason': 'point_only',
                'point_count': 0,
            }

        if supervision == SUPERVISION_IGNORE:
            frame_stats['ignored'] += 1
            frame_stats[f'filter_reason_{diagnostics["reason"]}'] += 1
            continue

        if supervision == SUPERVISION_BOX:
            box = raw_box.astype(np.float32)
            # Offline pseudo boxes do not have sufficient yaw validation yet.
            code_mask = np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32)
            cls_weight = pseudo_box_weight
            reg_weight = pseudo_box_weight
            frame_stats['box'] += 1
        else:
            box = _placeholder_box(center, DEFAULT_PEDESTRIAN_SIZE)
            code_mask = np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            cls_weight = pseudo_center_weight
            reg_weight = pseudo_center_weight
            frame_stats['center'] += 1
        if label_is_exhaustive:
            pseudo_score = float(arrays['score'][obj_idx])
            if not np.isfinite(pseudo_score):
                pseudo_score = exhaustive_pseudo_cls_min
            cls_weight = max(
                float(cls_weight),
                float(np.clip(pseudo_score, exhaustive_pseudo_cls_min, 1.0)),
            )
        frame_stats[f'filter_reason_{diagnostics["reason"]}'] += 1

        targets.append({
            'box': box,
            'cls_weight': float(cls_weight),
            'reg_weight': float(reg_weight),
            'code_mask': code_mask,
            'supervision': int(supervision),
            'score': float(arrays['score'][obj_idx]),
            'reason': diagnostics['reason'],
            'origin': 'pseudo',
        })

    for manual_xy in manual_centers:
        center = np.asarray([manual_xy[0], manual_xy[1], 0.0], dtype=np.float32)
        targets.append({
            'box': _placeholder_box(center, DEFAULT_PEDESTRIAN_SIZE),
            'cls_weight': float(manual_center_weight),
            'reg_weight': float(manual_center_weight),
            'code_mask': np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            'supervision': SUPERVISION_CENTER,
            'score': 1.0,
            'reason': 'manual_center',
            'origin': 'manual',
        })
        frame_stats['manual_added'] += 1

    # The two label sources are complementary by annotation protocol: a target
    # with an existing pseudo 3D box is not clicked again.  Deduplicate only
    # within the pseudo-label source; cross-source distance matching can merge
    # two different pedestrians in crowded scenes.
    targets.sort(
        key=lambda item: (item['cls_weight'], item['supervision'], item['score']),
        reverse=True,
    )
    deduplicated = []
    for target in targets:
        is_duplicate_pseudo = target['origin'] == 'pseudo' and any(
            kept['origin'] == 'pseudo'
            and np.linalg.norm(target['box'][:2] - kept['box'][:2]) < 0.20
            for kept in deduplicated
        )
        if is_duplicate_pseudo:
            frame_stats['duplicate_removed'] += 1
            continue
        deduplicated.append(target)

    if deduplicated:
        boxes = np.stack([item['box'] for item in deduplicated]).astype(np.float32)
        cls_weights = np.asarray([item['cls_weight'] for item in deduplicated], dtype=np.float32)
        reg_weights = np.asarray([item['reg_weight'] for item in deduplicated], dtype=np.float32)
        code_masks = np.stack([item['code_mask'] for item in deduplicated]).astype(np.float32)
        supervision_types = np.asarray([item['supervision'] for item in deduplicated], dtype=np.int32)
    else:
        boxes = np.empty((0, 7), dtype=np.float32)
        cls_weights = np.empty(0, dtype=np.float32)
        reg_weights = np.empty(0, dtype=np.float32)
        code_masks = np.empty((0, 8), dtype=np.float32)
        supervision_types = np.empty(0, dtype=np.int32)

    return {
        'gt_boxes': boxes,
        'gt_names': np.full(len(boxes), 'Pedestrian', dtype='<U10'),
        'gt_cls_weights': cls_weights,
        'gt_reg_weights': reg_weights,
        'gt_box_reg_mask': code_masks,
        'gt_supervision_types': supervision_types,
        'frame_stats': frame_stats,
    }


def process_scene(
    category,
    scene_dir,
    data_root,
    pseudo_by_frame,
    manual_by_frame,
    manual_frame_keys,
    val_scenes,
    args,
    filter_cfg,
):
    points_dir = scene_dir / 'points'
    point_files = sorted(points_dir.glob('*.bin')) if points_dir.exists() else []
    if args.max_frames_per_scene is not None:
        point_files = point_files[:args.max_frames_per_scene]
    frame_indices = [int(path.stem) for path in point_files]
    if not frame_indices:
        return [], Counter()

    scene = scene_dir.name
    static_without_pose = category in set(args.static_no_pose_categories)
    poses, pose_found, pose_usable = load_scene_poses(
        scene_dir, frame_indices, static_without_pose
    )
    relative_paths = {
        frame_idx: str(point_path.relative_to(data_root))
        for frame_idx, point_path in zip(frame_indices, point_files)
    }
    temporal = compute_scene_temporal_support(
        category, scene, frame_indices, poses, pose_usable,
        pseudo_by_frame, filter_cfg, args.frame_rate,
    )

    infos = []
    scene_stats = Counter()
    is_val_scene = (category, scene) in val_scenes
    for frame_idx, point_path in zip(frame_indices, point_files):
        if point_path.stat().st_size % (4 * np.dtype(np.float32).itemsize) != 0:
            scene_stats['malformed_point_file'] += 1
            continue
        key = (category, scene, frame_idx)
        label_is_exhaustive = key in manual_frame_keys
        manual_centers = manual_by_frame.get(key, np.empty((0, 2), dtype=np.float32))
        pseudo_frame = pseudo_by_frame.get(key)
        pseudo_arrays = _pseudo_arrays(pseudo_frame)
        pedestrian_mask = pseudo_arrays['name'] == 'Pedestrian'
        candidate_boxes = pseudo_arrays['bbox_3d'][pedestrian_mask]
        needs_point_filter = bool(
            len(candidate_boxes)
            and np.any(
                np.all(np.isfinite(candidate_boxes), axis=1)
                & np.all(candidate_boxes[:, 3:6] > 0, axis=1)
            )
        )
        if needs_point_filter:
            points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)
            points = points[np.all(np.isfinite(points), axis=1)]
            ego_mask = ~(
                (np.abs(points[:, 0]) < 1.0)
                & (np.abs(points[:, 1]) < 1.0)
            )
            points = points[ego_mask]
            scene_stats['point_files_read_for_box_filter'] += 1
        else:
            points = np.empty((0, 4), dtype=np.float32)
        frame_temporal = {
            obj_idx: temporal.get((frame_idx, obj_idx), 0)
            for obj_idx in range(len(pseudo_arrays['name']))
        }
        weak = make_weak_targets(
            pseudo_frame,
            manual_centers,
            frame_temporal,
            points,
            filter_cfg,
            args.point_cloud_range,
            args.pseudo_center_weight,
            args.pseudo_box_weight,
            args.manual_center_weight,
            label_is_exhaustive=label_is_exhaustive,
            exhaustive_pseudo_cls_min=args.exhaustive_pseudo_cls_min,
        )
        scene_stats.update(weak.pop('frame_stats'))

        info = {
            'token': f'{category}/{scene}/{frame_idx:04d}',
            'scene_id': f'{category}/{scene}',
            'frame_index': frame_idx,
            'lidar_path': relative_paths[frame_idx],
            'sweeps': build_sweeps(
                frame_idx,
                frame_indices,
                relative_paths,
                poses,
                pose_usable,
                args.sweep_frame_len,
                args.frame_rate,
                args.max_sweep_time_lag,
            ),
            'lidar2global': poses[frame_idx],
            'has_pose': bool(pose_found[frame_idx]),
            'domain': 'wheelchair_indoor' if category.startswith('indoor') else 'wheelchair_other',
            # Manual BEV clicks complement, rather than duplicate, the original
            # pseudo labels.  Their union is center/class exhaustive on frames
            # explicitly present in an annotations_*.pkl file.  This says
            # nothing about z/size/yaw reliability.
            'label_is_exhaustive': bool(label_is_exhaustive),
            'center_label_is_exhaustive': bool(label_is_exhaustive),
            'is_target_domain': True,
            'roi_supervision_enabled': False,
            'weak_centers': manual_centers.astype(np.float32),
            **weak,
        }

        # Validation keeps every explicitly reviewed frame, including a
        # possible future empty-background annotation.
        if not is_val_scene or label_is_exhaustive:
            infos.append(info)
        scene_stats['frames'] += 1
        scene_stats['exhaustive_frames'] += int(label_is_exhaustive)
        scene_stats['manual_centers'] += len(manual_centers)
        scene_stats['pseudo_frames'] += int(pseudo_frame is not None)
    return infos, scene_stats


def save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def summarize_infos(infos):
    supervision = [
        np.asarray(info['gt_supervision_types'], dtype=np.int32)
        for info in infos if len(info['gt_supervision_types'])
    ]
    supervision = np.concatenate(supervision) if supervision else np.empty(0, dtype=np.int32)
    return {
        'frames': len(infos),
        'exhaustive_frames': int(sum(
            bool(info.get('center_label_is_exhaustive', False)) for info in infos
        )),
        'empty_frames': int(sum(len(info['gt_boxes']) == 0 for info in infos)),
        'targets': int(sum(len(info['gt_boxes']) for info in infos)),
        'box_targets': int((supervision == SUPERVISION_BOX).sum()),
        'center_targets': int((supervision == SUPERVISION_CENTER).sum()),
        'manual_centers': int(sum(len(info['weak_centers']) for info in infos)),
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    pseudo_path = _resolve_input_path(args.pseudo_info, annotation_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Wheelchair data root not found: {data_root}')
    if not pseudo_path.exists():
        raise FileNotFoundError(f'Pseudo-label pkl not found: {pseudo_path}')
    if args.frame_rate <= 0 or args.max_sweep_time_lag <= 0:
        raise ValueError('frame rate and max sweep lag must be positive')

    pseudo_by_frame, frame_indices_by_scene = load_pseudo_scenes(pseudo_path)
    manual_by_frame, manual_scenes, manual_frame_keys = load_manual_centers(
        annotation_root, frame_indices_by_scene, return_frame_keys=True,
    )
    val_scenes = split_manual_scenes(manual_scenes, args.weak_val_ratio, args.seed)
    filter_cfg = PedestrianBoxFilterConfig(
        min_horizontal_size=args.min_horizontal_size,
        max_horizontal_size=args.max_horizontal_size,
        min_height=args.min_height,
        max_height=args.max_height,
        near_distance=args.near_distance,
        far_distance=args.far_distance,
        near_min_points=args.near_min_points,
        mid_min_points=args.mid_min_points,
        far_min_points=args.far_min_points,
        min_bidirectional_box_score=args.min_bidirectional_box_score,
        temporal_base_distance=args.temporal_base_distance,
        max_pedestrian_speed=args.max_pedestrian_speed,
    )
    weights = (
        args.pseudo_center_weight,
        args.pseudo_box_weight,
        args.manual_center_weight,
        args.exhaustive_pseudo_cls_min,
    )
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError('All weak-label weights must be in [0, 1]')
    if not (
        0 < filter_cfg.min_horizontal_size <= filter_cfg.max_horizontal_size
        and 0 < filter_cfg.min_height <= filter_cfg.max_height
        and 0 < filter_cfg.near_distance <= filter_cfg.far_distance
    ):
        raise ValueError('Invalid weak pseudo-box filter ranges')

    train_all = []
    train_indoor = []
    train_other = []
    train_indoor_exhaustive = []
    train_indoor_non_exhaustive = []
    weak_val = []
    totals = Counter()
    scene_summaries = {}

    scene_counter = 0
    requested_scenes = set(args.scene or [])
    scene_jobs = []
    for category in args.categories:
        category_dir = data_root / category
        if not category_dir.exists():
            print(f'WARNING: category not found, skipping: {category_dir}')
            continue
        for scene_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            if requested_scenes and not (
                scene_dir.name in requested_scenes
                or f'{category}/{scene_dir.name}' in requested_scenes
            ):
                continue
            scene_jobs.append((category, scene_dir))
    total_scene_count = len(scene_jobs)
    if requested_scenes and not scene_jobs:
        raise ValueError(
            f'No scenes matched --scene values: {sorted(requested_scenes)}'
        )

    def run_scene(job):
        category, scene_dir = job
        return process_scene(
            category,
            scene_dir,
            data_root,
            pseudo_by_frame,
            manual_by_frame,
            manual_frame_keys,
            val_scenes,
            args,
            filter_cfg,
        )

    if args.workers <= 0:
        raise ValueError('--workers must be positive')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        scene_results = executor.map(run_scene, scene_jobs)
        for (category, scene_dir), (infos, stats) in zip(scene_jobs, scene_results):
            scene_counter += 1
            scene_key = (category, scene_dir.name)
            scene_summaries[f'{category}/{scene_dir.name}'] = dict(stats)
            totals.update(stats)
            if scene_key in val_scenes:
                weak_val.extend(infos)
            else:
                train_all.extend(infos)
                if category.startswith('indoor'):
                    train_indoor.extend(infos)
                    train_indoor_exhaustive.extend(
                        info for info in infos
                        if info.get('center_label_is_exhaustive', False)
                    )
                    train_indoor_non_exhaustive.extend(
                        info for info in infos
                        if not info.get('center_label_is_exhaustive', False)
                    )
                else:
                    train_other.extend(infos)
            print(
                f'[{scene_counter}/{total_scene_count}] '
                f'{category}/{scene_dir.name}: frames={stats.get("frames", 0)}, '
                f'box={stats.get("box", 0)}, center={stats.get("center", 0)}',
                flush=True,
            )

    summary = {
        'data_root': str(data_root),
        'pseudo_info': str(pseudo_path),
        'manual_frame_mapping': 'scene_frames_list_index',
        'filter_config': asdict(filter_cfg),
        'num_pseudo_frames_loaded': len(pseudo_by_frame),
        'num_manual_scenes_loaded': len(manual_scenes),
        'num_manual_frames_loaded': len(manual_by_frame),
        'num_manual_annotation_frame_keys': len(manual_frame_keys),
        'num_manual_centers_loaded': int(sum(len(x) for x in manual_by_frame.values())),
        'num_manual_centers_not_mapped': int(
            sum(len(x) for x in manual_by_frame.values())
            - totals.get('manual_centers', 0)
        ) if args.max_frames_per_scene is None else None,
        'diagnostic_max_frames_per_scene': args.max_frames_per_scene,
        'requested_scenes': sorted(requested_scenes),
        'val_scenes': [f'{category}/{scene}' for category, scene in sorted(val_scenes)],
        'num_train_all': len(train_all),
        'num_train_indoor': len(train_indoor),
        'num_train_indoor_exhaustive': len(train_indoor_exhaustive),
        'num_train_indoor_non_exhaustive': len(train_indoor_non_exhaustive),
        'num_train_other': len(train_other),
        'num_weak_val': len(weak_val),
        'output_summaries': {
            'train_all': summarize_infos(train_all),
            'train_indoor': summarize_infos(train_indoor),
            'train_indoor_exhaustive': summarize_infos(train_indoor_exhaustive),
            'train_indoor_non_exhaustive': summarize_infos(train_indoor_non_exhaustive),
            'train_other': summarize_infos(train_other),
            'weak_val': summarize_infos(weak_val),
        },
        'totals': dict(totals),
        'scenes': scene_summaries,
    }

    print(json.dumps({key: value for key, value in summary.items() if key != 'scenes'}, indent=2))
    if args.dry_run:
        print('Dry run: no files written.')
        return

    outputs = {
        'wheelchair_weak_infos_train_all.pkl': train_all,
        'wheelchair_weak_infos_train_indoor.pkl': train_indoor,
        'wheelchair_weak_infos_train_indoor_exhaustive.pkl': train_indoor_exhaustive,
        'wheelchair_weak_infos_train_indoor_non_exhaustive.pkl': train_indoor_non_exhaustive,
        'wheelchair_weak_infos_train_other.pkl': train_other,
        'wheelchair_weak_infos_val_indoor.pkl': weak_val,
        'wheelchair_weak_infos_val_indoor_exhaustive.pkl': weak_val,
    }
    for filename, infos in outputs.items():
        path = output_dir / filename
        save_pickle(path, infos)
        print(f'Saved {len(infos):,} infos to {path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'wheelchair_weak_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary to {output_dir / "wheelchair_weak_summary.json"}')


if __name__ == '__main__':
    main()
