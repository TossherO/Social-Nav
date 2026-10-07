#!/usr/bin/env python3
"""Render aligned Wheelchair GT, one-frame, and four-frame BEV panels.

The script consumes prediction pickle files produced by
``visualize_wheelchair_predictions.py``.  Reusing the saved full-validation
predictions guarantees that comparison rendering does not accidentally change
the input sweep count, score threshold, NMS, or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.visualize_wheelchair_predictions import (  # noqa: E402
    _finite_boxes_and_scores,
    _manual_centers_in_range,
    _plot_box,
    _plot_point_subsets,
    _prediction_point_support,
    _weak_overlays,
    frame_audit_stats,
    greedy_center_matches,
    uniform_subset,
)


DEFAULT_CFG = 'configs/voxelrcnn_wheelchair_weak_stage3.yaml'
DEFAULT_SINGLE_PREDICTIONS = (
    'output/wheelchair_prediction_vis/'
    'exhaustive_center_1frame_10_10_5_missed20/predictions.pkl'
)
DEFAULT_FOUR_PREDICTIONS = (
    'output/wheelchair_prediction_vis/'
    'exhaustive_center_10_12_5_missed20/predictions.pkl'
)
DEFAULT_OUTPUT_DIR = (
    'output/wheelchair_prediction_vis/'
    'comparison_1frame_vs_4frame_balanced20'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render aligned GT / one-frame / four-frame Wheelchair BEV comparisons',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg_file', default=DEFAULT_CFG)
    parser.add_argument('--single_predictions', default=DEFAULT_SINGLE_PREDICTIONS)
    parser.add_argument('--four_predictions', default=DEFAULT_FOUR_PREDICTIONS)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--root_path', default=None)
    parser.add_argument('--raw_data_path', default=None)
    parser.add_argument('--info_path', nargs='+', default=None)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument(
        '--frame_id', action='append', default=None,
        help='Render an exact frame token; repeatable and overrides automatic selection',
    )
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--match_radius', type=float, default=0.5)
    parser.add_argument('--min_box_points', type=int, default=3)
    parser.add_argument('--min_frame_gap', type=int, default=10)
    parser.add_argument('--max_points', type=int, default=60000)
    parser.add_argument('--dpi', type=int, default=150)
    return parser.parse_args()


def _resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_args(args):
    if args.num_samples <= 0:
        raise ValueError('--num_samples must be positive')
    if not 0 <= args.score_thresh <= 1:
        raise ValueError('--score_thresh must be in [0, 1]')
    if args.match_radius <= 0:
        raise ValueError('--match_radius must be positive')
    if args.min_box_points < 0 or args.min_frame_gap < 0 or args.max_points < 0:
        raise ValueError('point counts and frame gap must be non-negative')
    if args.dpi <= 0:
        raise ValueError('--dpi must be positive')


def load_predictions(path):
    with path.open('rb') as stream:
        annos = pickle.load(stream)
    if not isinstance(annos, list):
        raise TypeError(f'Prediction file must contain a list: {path}')

    by_frame = {}
    for anno in annos:
        frame_id = str(anno.get('frame_id', ''))
        if not frame_id:
            raise ValueError(f'Prediction without frame_id in {path}')
        if frame_id in by_frame:
            raise ValueError(f'Duplicate frame_id {frame_id!r} in {path}')
        by_frame[frame_id] = anno
    return by_frame


def build_comparison_records(dataset, single_by_frame, four_by_frame,
                             score_thresh, match_radius):
    records = []
    for index, info in enumerate(dataset.infos):
        frame_id = str(info.get('token', index))
        if frame_id not in single_by_frame or frame_id not in four_by_frame:
            continue
        single_stats = frame_audit_stats(
            index, info, single_by_frame[frame_id], dataset.point_cloud_range,
            score_thresh, match_radius,
        )
        four_stats = frame_audit_stats(
            index, info, four_by_frame[frame_id], dataset.point_cloud_range,
            score_thresh, match_radius,
        )
        if single_stats['manual_centers_in_range'] <= 0:
            continue
        records.append({
            'dataset_index': index,
            'frame_id': frame_id,
            'scene_id': str(info.get('scene_id', '')),
            'frame_index': int(info.get('frame_index', -1)),
            'manual_centers': single_stats['manual_centers_in_range'],
            'single_hits': single_stats['manual_matched'],
            'four_hits': four_stats['manual_matched'],
            'single_misses': single_stats['manual_missed'],
            'four_misses': four_stats['manual_missed'],
            'single_boxes': single_stats['shown_predictions'],
            'four_boxes': four_stats['shown_predictions'],
            'single_unmatched': single_stats['unmatched_predictions_for_review'],
            'four_unmatched': four_stats['unmatched_predictions_for_review'],
            'hit_delta': single_stats['manual_matched'] - four_stats['manual_matched'],
            'unmatched_delta': (
                four_stats['unmatched_predictions_for_review']
                - single_stats['unmatched_predictions_for_review']
            ),
        })
    return records


def _take_diverse(candidates, count, used, min_frame_gap):
    selected = []
    for enforce_gap in (True, False):
        for record in candidates:
            if len(selected) >= count:
                break
            if record['frame_id'] in used:
                continue
            if enforce_gap and any(
                previous['scene_id'] == record['scene_id']
                and abs(previous['frame_index'] - record['frame_index']) < min_frame_gap
                for previous in selected
            ):
                continue
            selected.append(record)
            used.add(record['frame_id'])
        if len(selected) >= count:
            break
    return selected


def select_balanced_records(records, count, min_frame_gap=10):
    """Select wins for each model, cleaner ties, and temporal coverage."""

    if count <= 0 or count >= len(records):
        return [dict(record, category='all') for record in records]

    quota_base, remainder = divmod(count, 4)
    quotas = [quota_base + (index < remainder) for index in range(4)]
    groups = [
        (
            'single_better',
            sorted(
                (record for record in records if record['hit_delta'] > 0),
                key=lambda record: (
                    -record['hit_delta'], -record['unmatched_delta'],
                    -record['manual_centers'], record['dataset_index'],
                ),
            ),
        ),
        (
            'four_better',
            sorted(
                (record for record in records if record['hit_delta'] < 0),
                key=lambda record: (
                    record['hit_delta'], record['unmatched_delta'],
                    -record['manual_centers'], record['dataset_index'],
                ),
            ),
        ),
        (
            'single_cleaner_tie',
            sorted(
                (
                    record for record in records
                    if record['hit_delta'] == 0 and record['unmatched_delta'] > 0
                ),
                key=lambda record: (
                    -record['unmatched_delta'], -record['manual_centers'],
                    record['dataset_index'],
                ),
            ),
        ),
        (
            'uniform',
            uniform_subset(sorted(records, key=lambda record: record['dataset_index']), count),
        ),
    ]

    selected = []
    used = set()
    for (category, candidates), quota in zip(groups, quotas):
        chosen = _take_diverse(candidates, quota, used, min_frame_gap)
        selected.extend(dict(record, category=category) for record in chosen)

    if len(selected) < count:
        fallback = sorted(
            records,
            key=lambda record: (
                -abs(record['hit_delta']), -abs(record['unmatched_delta']),
                -record['manual_centers'], record['dataset_index'],
            ),
        )
        chosen = _take_diverse(fallback, count - len(selected), used, min_frame_gap)
        selected.extend(dict(record, category='balanced_fallback') for record in chosen)
    return selected[:count]


def _plot_common_background(ax, points, point_cloud_range):
    if len(points):
        ax.scatter(
            points[:, 0], points[:, 1], s=0.25, c='0.28', alpha=0.62,
            rasterized=True, zorder=2,
        )
    ax.scatter(0, 0, marker='^', s=45, c='black', zorder=9)
    ax.axhline(0, color='0.75', linewidth=0.5)
    ax.axvline(0, color='0.75', linewidth=0.5)
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('LiDAR x / forward (m)')
    ax.set_ylabel('LiDAR y / left (m)')
    ax.grid(alpha=0.15)


def _plot_predictions(ax, current_points, anno, score_thresh,
                      min_box_points, color):
    boxes, scores = _finite_boxes_and_scores(anno, score_thresh)
    supports = _prediction_point_support(current_points, boxes)
    for box, score, support in zip(boxes, scores, supports):
        linestyle = ':' if support < min_box_points else '-'
        _plot_box(
            ax, box, color, linestyle=linestyle,
            linewidth=1.2 + 1.2 * float(score), alpha=0.85,
        )
        ax.text(
            box[0], box[1], f'{score:.2f}|p{support}', color=color,
            fontsize=6.2, zorder=7, clip_on=True,
            bbox={
                'facecolor': 'white', 'alpha': 0.62,
                'edgecolor': 'none', 'pad': 0.4,
            },
        )
    return boxes, scores, supports


def render_comparison(dataset, record, single_anno, four_anno, args, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    info = dataset.infos[record['dataset_index']]
    current_points = dataset.get_lidar_with_sweeps(
        record['dataset_index'], max_sweeps=1,
    )
    point_cloud_range = np.asarray(dataset.point_cloud_range, dtype=np.float32)
    keep = (
        (current_points[:, 0] >= point_cloud_range[0])
        & (current_points[:, 0] <= point_cloud_range[3])
        & (current_points[:, 1] >= point_cloud_range[1])
        & (current_points[:, 1] <= point_cloud_range[4])
    )
    current_points = current_points[keep]
    current_plot, _ = _plot_point_subsets(
        current_points, current_points[:0], args.max_points,
    )

    manual = _manual_centers_in_range(info, point_cloud_range)
    weak_boxes, pseudo_centers = _weak_overlays(info)

    fig, axes = plt.subplots(
        1, 3, figsize=(26, 8), dpi=args.dpi,
        sharex=True, sharey=True, layout='constrained',
    )
    label_ax, single_ax, four_ax = axes
    for ax in axes:
        _plot_common_background(ax, current_plot, point_cloud_range)

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

    single_boxes, _, single_supports = _plot_predictions(
        single_ax, current_points, single_anno, args.score_thresh,
        args.min_box_points, '#0072b2',
    )
    four_boxes, _, four_supports = _plot_predictions(
        four_ax, current_points, four_anno, args.score_thresh,
        args.min_box_points, '#cc79a7',
    )
    single_hits = len(greedy_center_matches(
        single_boxes[:, :2], manual, args.match_radius,
    ))
    four_hits = len(greedy_center_matches(
        four_boxes[:, :2], manual, args.match_radius,
    ))

    label_ax.set_title(
        'Available labels / weak ground truth\n'
        f'manual centers={len(manual)}, weak boxes={len(weak_boxes)}, '
        f'other weak centers={len(pseudo_centers)}'
    )
    single_ax.set_title(
        '1-frame model predictions\n'
        f'boxes={len(single_boxes)}, manual-center hit={single_hits}/{len(manual)}, '
        f'low-point={sum(value < args.min_box_points for value in single_supports)}'
    )
    four_ax.set_title(
        '4-frame model predictions\n'
        f'boxes={len(four_boxes)}, manual-center hit={four_hits}/{len(manual)}, '
        f'low-point={sum(value < args.min_box_points for value in four_supports)}'
    )

    point_handles = [
        Line2D([], [], marker='^', linestyle='None', color='black', label='LiDAR origin'),
        Line2D([], [], marker='.', linestyle='None', color='0.28', label='current-frame points'),
    ]
    label_ax.legend(handles=point_handles + [
        Line2D([], [], marker='x', linestyle='None', color='#d62728', label='manual BEV center'),
        Line2D([], [], color='#f28e2b', linestyle='--', label='filtered weak 3D box'),
        Line2D([], [], marker='+', linestyle='None', color='#f28e2b', label='other weak center'),
    ], loc='upper right', fontsize=7)
    single_ax.legend(handles=point_handles + [
        Line2D([], [], color='#0072b2', label='1-frame prediction: score|points'),
        Line2D([], [], color='#0072b2', linestyle=':', label='low point support'),
    ], loc='upper right', fontsize=7)
    four_ax.legend(handles=point_handles + [
        Line2D([], [], color='#cc79a7', label='4-frame prediction: score|points'),
        Line2D([], [], color='#cc79a7', linestyle=':', label='low point support'),
    ], loc='upper right', fontsize=7)

    fig.suptitle(
        f"{record['frame_id']} | category={record['category']} | "
        f'1-frame hit={single_hits}/{len(manual)} | 4-frame hit={four_hits}/{len(manual)}\n'
        'All panels show current-frame points; models were inferred with 1 and 4 sweeps respectively.',
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _safe_image_name(index, record):
    token = record['frame_id'].replace('/', '__').replace('\\', '__')
    return f"{index:02d}_{record['category']}__{token}.png"


def main():
    args = parse_args()
    _validate_args(args)

    from simple_det3d.datasets.coda_dataset import CodaDataset
    from simple_det3d.utils.common_utils import create_logger
    from simple_det3d.utils.config import load_config

    cfg_path = _resolve_project_path(args.cfg_file)
    single_path = _resolve_project_path(args.single_predictions)
    four_path = _resolve_project_path(args.four_predictions)
    output_dir = _resolve_project_path(args.output_dir)
    for path in (cfg_path, single_path, four_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir / 'compare.log')

    cfg = load_config(cfg_path)
    if args.root_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.root_path
    if args.raw_data_path is not None:
        cfg.DATA_CONFIG.RAW_DATA_PATH = args.raw_data_path
    if args.info_path is not None:
        cfg.DATA_CONFIG.INFO_PATH.test = args.info_path
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

    single_by_frame = load_predictions(single_path)
    four_by_frame = load_predictions(four_path)
    records = build_comparison_records(
        dataset, single_by_frame, four_by_frame,
        args.score_thresh, args.match_radius,
    )
    if args.frame_id:
        by_frame = {record['frame_id']: record for record in records}
        missing = [frame_id for frame_id in args.frame_id if frame_id not in by_frame]
        if missing:
            raise ValueError(f'Frames unavailable or without manual centers: {missing}')
        selected = [dict(by_frame[frame_id], category='requested') for frame_id in args.frame_id]
    else:
        selected = select_balanced_records(
            records, args.num_samples, args.min_frame_gap,
        )

    for image_index, record in enumerate(selected, start=1):
        image_name = _safe_image_name(image_index, record)
        record['image_name'] = image_name
        render_comparison(
            dataset, record,
            single_by_frame[record['frame_id']],
            four_by_frame[record['frame_id']],
            args, image_dir / image_name,
        )

    _write_csv(output_dir / 'selected_frames.csv', selected)
    summary = {
        'config': str(cfg_path),
        'single_predictions': str(single_path),
        'four_predictions': str(four_path),
        'score_thresh': args.score_thresh,
        'match_radius': args.match_radius,
        'available_aligned_frames_with_manual_centers': len(records),
        'rendered_frames': len(selected),
        'category_counts': {
            category: sum(record['category'] == category for record in selected)
            for category in sorted({record['category'] for record in selected})
        },
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    logger.info('Rendered %d aligned comparisons to %s', len(selected), output_dir)
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
