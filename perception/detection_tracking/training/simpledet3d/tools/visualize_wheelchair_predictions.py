#!/usr/bin/env python3
"""Run Wheelchair inference and render auditable BEV prediction overlays.

The Wheelchair labels are incomplete.  This tool therefore reports manual
center coverage and predictions requiring review, never precision or false
positive counts.  It saves individual PNG files, CSV/JSON summaries, and the
raw prediction pickle for human inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simple_det3d.datasets.weak_label_utils import (  # noqa: E402
    PedestrianBoxFilterConfig,
    SUPERVISION_BOX,
    SUPERVISION_CENTER,
    count_points_in_oriented_box,
    required_point_count,
)


DEFAULT_CFG = 'configs/voxelrcnn_wheelchair_weak_stage3.yaml'
DEFAULT_CKPT = (
    'output/voxelrcnn_wheelchair_weak_stage3/'
    'wheelchair_exhaustive_center_10_12_5/'
    'ckpt/best_model.pth'
)
SELECTION_MODES = (
    'missed', 'matched', 'crowded', 'low_support',
    'random', 'uniform', 'all',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a trained detector and visualize Wheelchair predictions in BEV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg_file', default=DEFAULT_CFG)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument(
        '--info_path', nargs='+', default=None,
        help='Override DATA_CONFIG.INFO_PATH.test; paths are relative to root_path',
    )
    parser.add_argument('--root_path', default=None, help='Override pkl root directory')
    parser.add_argument('--raw_data_path', default=None, help='Override raw point-cloud root')
    parser.add_argument(
        '--max_sweeps', type=int, default=None,
        help='Override DATA_CONFIG.MAX_SWEEPS; 0/1 means current frame only',
    )
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument(
        '--selection', choices=SELECTION_MODES, default='missed',
        help='How to choose frames after inference',
    )
    parser.add_argument('--num_samples', type=int, default=40,
                        help='Number of PNG files; 0 means all eligible frames')
    parser.add_argument(
        '--max_inference_samples', type=int, default=0,
        help='Uniformly limit inference candidates before ranking; 0 means no limit',
    )
    parser.add_argument(
        '--scene', action='append', default=None,
        help='Restrict to an exact scene_id such as indoor1/sequence2_001; repeatable',
    )
    parser.add_argument(
        '--frame_id', action='append', default=None,
        help='Restrict to an exact token such as indoor1/sequence2_001/0001; repeatable',
    )
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='Display/matching threshold; model inference is unchanged')
    parser.add_argument(
        '--final_nms_thresh', type=float, default=None,
        help='Override MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH for an inference ablation',
    )
    parser.add_argument('--match_radius', type=float, default=0.5,
                        help='Manual-center matching radius in metres')
    parser.add_argument('--min_box_points', type=int, default=3,
                        help='Flag a prediction with fewer current-frame points')
    parser.add_argument(
        '--filter_by_point_support', action='store_true',
        help='Reject predictions that fail the distance-adaptive current-frame point rule',
    )
    parser.add_argument('--point_support_near_distance', type=float, default=8.0)
    parser.add_argument('--point_support_far_distance', type=float, default=15.0)
    parser.add_argument('--point_support_near_min_points', type=int, default=5)
    parser.add_argument('--point_support_mid_min_points', type=int, default=3)
    parser.add_argument('--point_support_far_min_points', type=int, default=1)
    parser.add_argument('--show_history', action='store_true',
                        help='Also draw history sweep points in light blue')
    parser.add_argument('--max_points', type=int, default=60000,
                        help='Maximum plotted points per image')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--seed', type=int, default=666)
    return parser.parse_args()


def oriented_bev_corners(box):
    """Return the closed, rotated BEV polygon for a 7D LiDAR box."""

    box = np.asarray(box, dtype=np.float32)
    if box.shape[0] < 7:
        raise ValueError('box must contain at least seven values')
    local = np.asarray([
        [box[3] / 2, box[4] / 2],
        [box[3] / 2, -box[4] / 2],
        [-box[3] / 2, -box[4] / 2],
        [-box[3] / 2, box[4] / 2],
        [box[3] / 2, box[4] / 2],
    ], dtype=np.float32)
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return local @ rotation.T + box[None, :2]


def greedy_center_matches(pred_xy, manual_xy, radius):
    """One-to-one greedy matches as ``(distance, pred_idx, manual_idx)``."""

    pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
    manual_xy = np.asarray(manual_xy, dtype=np.float32).reshape(-1, 2)
    if len(pred_xy) == 0 or len(manual_xy) == 0:
        return []
    distances = np.linalg.norm(
        pred_xy[:, None, :] - manual_xy[None, :, :], axis=-1,
    )
    matches = []
    while distances.size:
        pred_idx, manual_idx = np.unravel_index(
            int(np.argmin(distances)), distances.shape,
        )
        distance = float(distances[pred_idx, manual_idx])
        if distance > radius:
            break
        matches.append((distance, int(pred_idx), int(manual_idx)))
        distances[pred_idx, :] = np.inf
        distances[:, manual_idx] = np.inf
    return matches


def uniform_subset(items, count):
    """Select ordered, approximately uniform elements without duplicates."""

    items = list(items)
    if count <= 0 or count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[index] for index in indices]


def _finite_boxes_and_scores(anno, score_thresh):
    names = np.asarray(anno.get('name', []), dtype=str)
    boxes = np.asarray(anno.get('boxes_lidar', []), dtype=np.float32)
    scores = np.asarray(anno.get('score', []), dtype=np.float32)
    if boxes.size == 0:
        boxes = np.empty((0, 7), dtype=np.float32)
    valid = (
        (names == 'Pedestrian')
        & np.isfinite(scores)
        & (scores >= score_thresh)
    )
    if len(boxes):
        valid &= np.all(np.isfinite(boxes[:, :7]), axis=1)
        valid &= np.all(boxes[:, 3:6] > 0, axis=1)
    return boxes[valid, :7], scores[valid]


def _manual_centers_in_range(info, point_cloud_range):
    centers = np.asarray(info.get('weak_centers', []), dtype=np.float32).reshape(-1, 2)
    if len(centers) == 0:
        return centers
    pc_range = np.asarray(point_cloud_range, dtype=np.float32)
    keep = (
        np.all(np.isfinite(centers), axis=1)
        & (centers[:, 0] >= pc_range[0])
        & (centers[:, 0] <= pc_range[3])
        & (centers[:, 1] >= pc_range[1])
        & (centers[:, 1] <= pc_range[4])
    )
    return centers[keep]


def frame_audit_stats(index, info, anno, point_cloud_range,
                      score_thresh, match_radius):
    """Compute label-aware frame statistics without claiming precision."""

    boxes, scores = _finite_boxes_and_scores(anno, score_thresh)
    manual_all = np.asarray(
        info.get('weak_centers', []), dtype=np.float32,
    ).reshape(-1, 2)
    manual = _manual_centers_in_range(info, point_cloud_range)
    matches = greedy_center_matches(boxes[:, :2], manual, match_radius)
    matched_manual = {match[2] for match in matches}

    if len(manual) and len(boxes):
        nearest = np.linalg.norm(
            manual[:, None, :] - boxes[None, :, :2], axis=-1,
        ).min(axis=1)
        mean_nearest = float(nearest.mean())
        max_nearest = float(nearest.max())
    elif len(manual):
        mean_nearest = math.inf
        max_nearest = math.inf
    else:
        mean_nearest = math.nan
        max_nearest = math.nan

    supervision = np.asarray(
        info.get('gt_supervision_types', []), dtype=np.int32,
    )
    match_distances = [match[0] for match in matches]
    return {
        'dataset_index': int(index),
        'frame_id': str(info.get('token', anno.get('frame_id', index))),
        'scene_id': str(info.get('scene_id', '')),
        'frame_index': int(info.get('frame_index', -1)),
        'raw_predictions': int(len(anno.get('score', []))),
        'shown_predictions': int(len(boxes)),
        'unmatched_predictions_for_review': int(len(boxes) - len(matches)),
        'manual_centers_total': int(len(manual_all)),
        'manual_centers_in_range': int(len(manual)),
        'manual_matched': int(len(matched_manual)),
        'manual_missed': int(len(manual) - len(matched_manual)),
        'center_hit': float(len(matched_manual) / len(manual)) if len(manual) else math.nan,
        'mean_match_distance': float(np.mean(match_distances)) if match_distances else math.nan,
        'mean_nearest_manual_distance': mean_nearest,
        'max_nearest_manual_distance': max_nearest,
        'mean_shown_score': float(scores.mean()) if len(scores) else math.nan,
        'weak_box_targets': int(np.sum(supervision == SUPERVISION_BOX)),
        'weak_center_targets': int(np.sum(supervision == SUPERVISION_CENTER)),
        'low_support_predictions': None,
    }


def select_audit_frames(stats, mode, count, seed=666):
    """Rank/select per-frame audit dictionaries."""

    records = list(stats)
    if mode == 'all':
        selected = records
    elif mode == 'missed':
        selected = [record for record in records if record['manual_missed'] > 0]
        selected.sort(key=lambda record: (
            record['manual_missed'],
            record['max_nearest_manual_distance'],
            -record['shown_predictions'],
        ), reverse=True)
    elif mode == 'matched':
        selected = [record for record in records if record['manual_matched'] > 0]
        selected.sort(key=lambda record: (
            record['manual_matched'],
            -record['mean_match_distance'],
        ), reverse=True)
    elif mode == 'crowded':
        selected = sorted(
            records, key=lambda record: record['shown_predictions'], reverse=True,
        )
    elif mode == 'low_support':
        selected = sorted(
            records,
            key=lambda record: record['low_support_predictions'] or 0,
            reverse=True,
        )
    elif mode == 'random':
        generator = np.random.default_rng(seed)
        selected = records.copy()
        generator.shuffle(selected)
    elif mode == 'uniform':
        selected = uniform_subset(records, count)
        return selected
    else:
        raise ValueError(f'Unknown selection mode: {mode}')

    if count > 0:
        selected = selected[:count]
    return selected


def _weak_overlays(info):
    boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
    if boxes.size == 0:
        boxes = np.empty((0, 7), dtype=np.float32)
    supervision = np.asarray(
        info.get('gt_supervision_types', []), dtype=np.int32,
    )
    manual = np.asarray(info.get('weak_centers', []), dtype=np.float32).reshape(-1, 2)
    weak_boxes = boxes[supervision == SUPERVISION_BOX, :7]
    center_targets = boxes[supervision == SUPERVISION_CENTER, :2]
    if len(center_targets) and len(manual):
        exact_manual = np.any(
            np.linalg.norm(
                center_targets[:, None, :] - manual[None, :, :], axis=-1,
            ) < 1e-4,
            axis=1,
        )
        pseudo_centers = center_targets[~exact_manual]
    else:
        pseudo_centers = center_targets
    return weak_boxes, pseudo_centers


def _prediction_point_support(current_points, boxes):
    return [
        count_points_in_oriented_box(current_points[:, :3], box)
        for box in boxes
    ]


def filter_prediction_anno_by_point_support(current_points, anno, filter_cfg):
    """Filter one prediction annotation with current-frame LiDAR support.

    The returned annotation retains the regular prediction fields and adds the
    measured/required point counts for auditing.  Filtering is deliberately
    independent of the score display threshold so the saved prediction pickle
    represents the actual post-processing experiment.
    """

    boxes = np.asarray(anno.get('boxes_lidar', []), dtype=np.float32)
    num_predictions = len(boxes)
    if num_predictions == 0:
        filtered = dict(anno)
        filtered['point_support'] = np.empty((0,), dtype=np.int32)
        filtered['required_point_support'] = np.empty((0,), dtype=np.int32)
        return filtered, {
            'input_predictions': 0,
            'rejected_predictions': 0,
        }

    supports = np.asarray(
        _prediction_point_support(current_points, boxes), dtype=np.int32,
    )
    distances = np.linalg.norm(boxes[:, :2], axis=1)
    required = np.asarray([
        required_point_count(float(distance), filter_cfg)
        for distance in distances
    ], dtype=np.int32)
    keep = supports >= required

    filtered = {}
    for key, value in anno.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == num_predictions:
            filtered[key] = value[keep]
        else:
            filtered[key] = value
    filtered['point_support'] = supports[keep]
    filtered['required_point_support'] = required[keep]
    return filtered, {
        'input_predictions': int(num_predictions),
        'rejected_predictions': int((~keep).sum()),
    }


def _apply_point_support_filter(dataset, candidate_indices, annos_by_index, filter_cfg):
    filtered_by_index = {}
    audit_by_index = {}
    for index in candidate_indices:
        current_points = dataset.get_lidar_with_sweeps(index, max_sweeps=1)
        filtered, audit = filter_prediction_anno_by_point_support(
            current_points, annos_by_index[index], filter_cfg,
        )
        filtered_by_index[index] = filtered
        audit_by_index[index] = audit
    return filtered_by_index, audit_by_index


def _plot_box(ax, box, color, linestyle='-', linewidth=1.4, alpha=1.0):
    corners = oriented_bev_corners(box)
    ax.plot(
        corners[:, 0], corners[:, 1], color=color,
        linestyle=linestyle, linewidth=linewidth, alpha=alpha,
    )
    direction = np.asarray([np.cos(box[6]), np.sin(box[6])]) * box[3] / 2
    ax.plot(
        [box[0], box[0] + direction[0]],
        [box[1], box[1] + direction[1]],
        color=color, linewidth=linewidth, alpha=alpha,
    )


def _subsample_points(points, maximum):
    if maximum <= 0 or len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _plot_point_subsets(current_points, history_points, maximum):
    """Limit total plotted points while giving current-frame points priority."""

    if maximum <= 0:
        return current_points, history_points
    current_plot = _subsample_points(current_points, maximum)
    history_budget = maximum - len(current_plot)
    if history_budget <= 0:
        return current_plot, history_points[:0]
    return current_plot, _subsample_points(history_points, history_budget)


def render_bev(dataset, record, anno, args, output_path):
    """Render aligned full-range label and prediction BEV panels."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    index = record['dataset_index']
    info = dataset.infos[index]
    current_points = dataset.get_lidar_with_sweeps(index, max_sweeps=1)
    if args.show_history:
        all_points = dataset.get_lidar_with_sweeps(
            index, max_sweeps=dataset.dataset_cfg.get('MAX_SWEEPS', 1),
        )
        history_points = all_points[np.abs(all_points[:, 4]) > 1e-6]
    else:
        history_points = np.empty((0, current_points.shape[1]), dtype=np.float32)

    pc_range = np.asarray(dataset.point_cloud_range, dtype=np.float32)
    current_keep = (
        (current_points[:, 0] >= pc_range[0])
        & (current_points[:, 0] <= pc_range[3])
        & (current_points[:, 1] >= pc_range[1])
        & (current_points[:, 1] <= pc_range[4])
    )
    current_points = current_points[current_keep]
    if len(history_points):
        history_keep = (
            (history_points[:, 0] >= pc_range[0])
            & (history_points[:, 0] <= pc_range[3])
            & (history_points[:, 1] >= pc_range[1])
            & (history_points[:, 1] <= pc_range[4])
        )
        history_points = history_points[history_keep]

    boxes, scores = _finite_boxes_and_scores(anno, args.score_thresh)
    supports = _prediction_point_support(current_points, boxes)
    record['low_support_predictions'] = int(
        sum(support < args.min_box_points for support in supports)
    )

    manual = _manual_centers_in_range(info, pc_range)
    matches = greedy_center_matches(boxes[:, :2], manual, args.match_radius)
    weak_boxes, pseudo_centers = _weak_overlays(info)

    current_plot, history_plot = _plot_point_subsets(
        current_points, history_points, args.max_points,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8), dpi=args.dpi,
        sharex=True, sharey=True, layout='constrained',
    )
    label_ax, prediction_ax = axes
    for ax in axes:
        if len(history_plot):
            ax.scatter(
                history_plot[:, 0], history_plot[:, 1], s=0.18,
                c='#8ecae6', alpha=0.28, rasterized=True, zorder=1,
            )
        if len(current_plot):
            ax.scatter(
                current_plot[:, 0], current_plot[:, 1], s=0.25,
                c='0.28', alpha=0.62, rasterized=True, zorder=2,
            )
        ax.scatter(0, 0, marker='^', s=45, c='black', zorder=9)
        ax.axhline(0, color='0.75', linewidth=0.5)
        ax.axvline(0, color='0.75', linewidth=0.5)
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_aspect('equal')
        ax.set_xlabel('LiDAR x / forward (m)')
        ax.set_ylabel('LiDAR y / left (m)')
        ax.grid(alpha=0.15)

    for weak_box in weak_boxes:
        _plot_box(label_ax, weak_box, '#f28e2b', linestyle='--', linewidth=1.4)
    if len(pseudo_centers):
        label_ax.scatter(
            pseudo_centers[:, 0], pseudo_centers[:, 1], marker='+',
            s=65, linewidths=1.6, c='#f28e2b', zorder=6,
        )
    if len(manual):
        label_ax.scatter(
            manual[:, 0], manual[:, 1], marker='x',
            s=90, linewidths=2.2, c='#d62728', zorder=7,
        )

    for box, score, support in zip(boxes, scores, supports):
        color = plt.cm.viridis(float(np.clip(
            (score - args.score_thresh) / max(1e-6, 1 - args.score_thresh), 0, 1,
        )))
        linestyle = ':' if support < args.min_box_points else '-'
        _plot_box(
            prediction_ax, box, color, linestyle=linestyle,
            linewidth=1.2 + 1.2 * float(score),
        )
        prediction_ax.text(
            box[0], box[1], f'{score:.2f}|p{support}', color=color,
            fontsize=6.5, zorder=7, clip_on=True,
            bbox={
                'facecolor': 'white', 'alpha': 0.62,
                'edgecolor': 'none', 'pad': 0.4,
            },
        )

    label_ax.set_title(
        f'Available labels / weak ground truth (incomplete)\n'
        f'manual centers={len(manual)}, weak boxes={len(weak_boxes)}, '
        f'other weak centers={len(pseudo_centers)}'
    )
    prediction_ax.set_title(
        f'Model predictions (score >= {args.score_thresh:g})\n'
        f'boxes={len(boxes)}, low-point boxes={record["low_support_predictions"]}'
    )

    point_legend = [
        Line2D([], [], marker='^', linestyle='None', color='black', label='LiDAR origin'),
        Line2D([], [], marker='.', linestyle='None', color='0.28',
               label='current-frame points'),
    ]
    if args.show_history:
        point_legend.append(
            Line2D([], [], marker='.', linestyle='None', color='#8ecae6',
                   label='history sweep points')
        )
    label_legend = point_legend + [
        Line2D([], [], marker='x', linestyle='None', color='#d62728',
               markersize=8, label='manual BEV center'),
        Line2D([], [], color='#f28e2b', linestyle='--', label='filtered weak 3D box'),
        Line2D([], [], marker='+', linestyle='None', color='#f28e2b',
               markersize=8, label='non-manual weak center'),
    ]
    prediction_legend = point_legend + [
        Line2D([], [], color='#31688e', label='prediction: score|current points'),
        Line2D([], [], color='#31688e', linestyle=':', label='prediction with low point support'),
    ]
    label_ax.legend(handles=label_legend, loc='upper right', fontsize=7)
    prediction_ax.legend(handles=prediction_legend, loc='upper right', fontsize=7)

    warning = 'Labels are incomplete: visual differences are not automatically detection errors.'
    fig.suptitle(
        f"{record['frame_id']}  |  manual-center hit@{args.match_radius:g}m="
        f"{len(matches)}/{len(manual)}\n"
        f'{warning}',
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _compute_low_support_for_all(dataset, stats, annos_by_index,
                                 score_thresh, min_box_points):
    for record in stats:
        index = record['dataset_index']
        boxes, _ = _finite_boxes_and_scores(annos_by_index[index], score_thresh)
        points = dataset.get_lidar_with_sweeps(index, max_sweeps=1)
        supports = _prediction_point_support(points, boxes)
        record['low_support_predictions'] = int(
            sum(support < min_box_points for support in supports)
        )


def _csv_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return value


def _write_csv(path, records):
    records = list(records)
    if not records:
        path.write_text('', encoding='utf-8')
        return
    # Rendering adds ``image_name`` only to selected records.  Build the union
    # of keys so all_frames.csv remains valid even when its first row was not
    # selected for rendering.
    fieldnames = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(value) for key, value in record.items()})


def _safe_image_name(frame_id):
    return frame_id.replace('/', '__').replace('\\', '__') + '.png'


def _resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_args(args):
    if not 0 <= args.score_thresh <= 1:
        raise ValueError('--score_thresh must be in [0, 1]')
    if args.final_nms_thresh is not None and not 0 <= args.final_nms_thresh <= 1:
        raise ValueError('--final_nms_thresh must be in [0, 1]')
    if not math.isfinite(args.match_radius) or args.match_radius <= 0:
        raise ValueError('--match_radius must be positive')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive')
    if args.workers < 0:
        raise ValueError('--workers must be non-negative')
    if getattr(args, 'max_sweeps', None) is not None and args.max_sweeps < 0:
        raise ValueError('--max_sweeps must be non-negative')
    if args.num_samples < 0 or args.max_inference_samples < 0:
        raise ValueError('sample limits must be non-negative')
    if args.min_box_points < 0 or args.max_points < 0:
        raise ValueError('point limits must be non-negative')
    if args.point_support_near_distance <= 0:
        raise ValueError('--point_support_near_distance must be positive')
    if args.point_support_far_distance <= args.point_support_near_distance:
        raise ValueError('--point_support_far_distance must exceed near distance')
    if min(
        args.point_support_near_min_points,
        args.point_support_mid_min_points,
        args.point_support_far_min_points,
    ) < 0:
        raise ValueError('point-support minimum counts must be non-negative')
    if args.dpi <= 0:
        raise ValueError('--dpi must be positive')


def main():
    args = parse_args()
    _validate_args(args)

    import torch
    import tqdm
    from torch.utils.data import DataLoader, Subset

    from simple_det3d.datasets.coda_dataset import CodaDataset
    from simple_det3d.models import build_model, load_data_to_gpu
    from simple_det3d.utils.common_utils import create_logger
    from simple_det3d.utils.config import load_config
    from simple_det3d.utils.precision_utils import (
        autocast_context,
        resolve_precision,
        validate_precision_support,
    )

    cfg_path = _resolve_project_path(args.cfg_file)
    checkpoint_path = _resolve_project_path(args.ckpt)
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = load_config(cfg_path)
    cfg.TAG = cfg_path.stem
    cfg.LOCAL_RANK = 0
    cfg.ROOT_DIR = PROJECT_ROOT
    if args.root_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.root_path
    if args.raw_data_path is not None:
        cfg.DATA_CONFIG.RAW_DATA_PATH = args.raw_data_path
    if args.info_path is not None:
        cfg.DATA_CONFIG.INFO_PATH.test = args.info_path
    if args.max_sweeps is not None:
        cfg.DATA_CONFIG.MAX_SWEEPS = max(1, args.max_sweeps)
    if args.final_nms_thresh is not None:
        cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = args.final_nms_thresh
    args.precision = resolve_precision(
        cli_precision=args.precision,
        config_precision=cfg.get('PRECISION', 'fp32'),
    )
    validate_precision_support(args.precision, device=0)

    if args.output_dir:
        output_dir = _resolve_project_path(args.output_dir)
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        output_dir = (
            PROJECT_ROOT / 'output' / 'wheelchair_prediction_vis'
            / f'{cfg.TAG}_{checkpoint_path.stem}_{timestamp}'
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir / 'visualize.log')

    root_path = Path(cfg.DATA_CONFIG.DATA_PATH).expanduser()
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    dataset = CodaDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=root_path,
        logger=logger,
    )
    if not dataset.infos:
        raise RuntimeError('No dataset infos were loaded')

    requested_scenes = set(args.scene or [])
    requested_frames = set(args.frame_id or [])
    candidate_indices = []
    for index, info in enumerate(dataset.infos):
        token = str(info.get('token', index))
        scene_id = str(info.get('scene_id', ''))
        if requested_scenes and scene_id not in requested_scenes:
            continue
        if requested_frames and token not in requested_frames:
            continue
        if args.selection in ('missed', 'matched') and not len(info.get('weak_centers', [])):
            continue
        candidate_indices.append(index)
    if not candidate_indices:
        raise RuntimeError('No frames match the requested scene/frame filters')
    if args.max_inference_samples > 0:
        candidate_indices = uniform_subset(
            candidate_indices, args.max_inference_samples,
        )

    logger.info('Config: %s', cfg_path)
    logger.info('Checkpoint: %s', checkpoint_path)
    logger.info('Dataset frames: %d; inference candidates: %d',
                len(dataset), len(candidate_indices))
    logger.info('Selection=%s score_thresh=%.3f match_radius=%.3f',
                args.selection, args.score_thresh, args.match_radius)
    logger.info('Input sweeps (including current frame): %d',
                int(cfg.DATA_CONFIG.MAX_SWEEPS))
    logger.info(
        'Final NMS threshold: %.3f; point-support filter: %s',
        float(cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH),
        args.filter_by_point_support,
    )

    subset = Subset(dataset, candidate_indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=dataset.collate_batch,
        drop_last=False,
    )

    model = build_model(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(str(checkpoint_path), logger=logger)
    model.cuda().eval()

    annos_by_index = {}
    all_annos = []
    cursor = 0
    inference_start = time.time()
    progress = tqdm.tqdm(dataloader, desc='infer', dynamic_ncols=True)
    with torch.no_grad():
        for batch_dict in progress:
            batch_count = len(batch_dict['frame_id'])
            batch_indices = candidate_indices[cursor:cursor + batch_count]
            cursor += batch_count
            load_data_to_gpu(batch_dict)
            with autocast_context(args.precision):
                pred_dicts, _ = model(batch_dict)
            batch_annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, cfg.CLASS_NAMES,
            )
            for index, anno in zip(batch_indices, batch_annos):
                expected_token = str(dataset.infos[index].get('token', index))
                if str(anno['frame_id']) != expected_token:
                    raise RuntimeError(
                        f'Prediction/index mismatch: {anno["frame_id"]} != {expected_token}'
                    )
                annos_by_index[index] = anno
                all_annos.append(anno)
    inference_seconds = time.time() - inference_start

    raw_annos_by_index = annos_by_index
    raw_all_annos = all_annos
    point_filter_audit = {}
    if args.filter_by_point_support:
        logger.info('Applying distance-adaptive current-frame point-support filter...')
        point_filter_cfg = PedestrianBoxFilterConfig(
            near_distance=args.point_support_near_distance,
            far_distance=args.point_support_far_distance,
            near_min_points=args.point_support_near_min_points,
            mid_min_points=args.point_support_mid_min_points,
            far_min_points=args.point_support_far_min_points,
        )
        annos_by_index, point_filter_audit = _apply_point_support_filter(
            dataset, candidate_indices, raw_annos_by_index, point_filter_cfg,
        )
        all_annos = [annos_by_index[index] for index in candidate_indices]

    stats = [
        frame_audit_stats(
            index, dataset.infos[index], annos_by_index[index],
            dataset.point_cloud_range, args.score_thresh, args.match_radius,
        )
        for index in candidate_indices
    ]
    for record in stats:
        index = record['dataset_index']
        record['model_predictions_before_point_filter'] = int(
            len(raw_annos_by_index[index].get('score', []))
        )
        record['point_support_rejected'] = int(
            point_filter_audit.get(index, {}).get('rejected_predictions', 0)
        )
    if args.selection == 'low_support':
        logger.info('Computing current-frame point support for all candidates...')
        _compute_low_support_for_all(
            dataset, stats, annos_by_index,
            args.score_thresh, args.min_box_points,
        )

    selected = select_audit_frames(
        stats, args.selection, args.num_samples, seed=args.seed,
    )
    if not selected:
        logger.warning(
            'No frames qualify for selection=%r; summary files will still be written',
            args.selection,
        )

    for record in tqdm.tqdm(selected, desc='render', dynamic_ncols=True):
        image_name = _safe_image_name(record['frame_id'])
        record['image_name'] = image_name
        render_bev(
            dataset, record, annos_by_index[record['dataset_index']], args,
            image_dir / image_name,
        )

    total_manual = sum(record['manual_centers_in_range'] for record in stats)
    total_matched = sum(record['manual_matched'] for record in stats)
    total_predictions = sum(record['shown_predictions'] for record in stats)
    summary = {
        'config': str(cfg_path),
        'checkpoint': str(checkpoint_path),
        'precision': args.precision,
        'selection': args.selection,
        'score_thresh': args.score_thresh,
        'final_nms_thresh': float(cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH),
        'point_support_filter_enabled': args.filter_by_point_support,
        'point_support_rejected': int(sum(
            record['point_support_rejected'] for record in stats
        )),
        'match_radius': args.match_radius,
        'inference_frames': len(stats),
        'rendered_frames': len(selected),
        'inference_seconds': inference_seconds,
        'manual_centers_in_range': total_manual,
        'manual_centers_matched': total_matched,
        'center_hit': total_matched / total_manual if total_manual else None,
        'predictions_for_review': total_predictions,
        'warning': (
            'Wheelchair labels are incomplete; unmatched predictions are not '
            'automatically false positives and precision/AP are not reported.'
        ),
        'arguments': {
            key: value for key, value in vars(args).items()
            if key not in {'cfg_file', 'ckpt'}
        },
    }
    with (output_dir / 'predictions.pkl').open('wb') as stream:
        pickle.dump(all_annos, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if args.filter_by_point_support:
        with (output_dir / 'raw_predictions.pkl').open('wb') as stream:
            pickle.dump(raw_all_annos, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    _write_csv(output_dir / 'all_frames.csv', stats)
    _write_csv(output_dir / 'selected_frames.csv', selected)

    logger.info('Inference finished in %.2f s (%.4f s/frame)',
                inference_seconds, inference_seconds / len(stats))
    logger.info('Manual center hit: %d/%d = %s',
                total_matched, total_manual,
                f'{total_matched / total_manual:.4f}' if total_manual else 'N/A')
    logger.info('Rendered %d frames to %s', len(selected), output_dir)
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
