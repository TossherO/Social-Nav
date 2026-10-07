#!/usr/bin/env python3
"""Render Wheelchair labels in overview, local BEV, and pseudo-box side view.

Manual annotation keys are resolved through the corresponding scene's ordered
``frames`` list in the source pseudo-info file.  The side view audits the raw
pseudo-box convention ``[x, y, z, dx, dy, dz, yaw]`` with center-based ``z``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simple_det3d.datasets.weak_label_utils import (  # noqa: E402
    PedestrianBoxFilterConfig,
    validate_box_geometry,
)
from tools.prepare_wheelchair_weak_data import (  # noqa: E402
    DEFAULT_POINT_CLOUD_RANGE,
    _pseudo_arrays,
    load_manual_centers,
    load_pseudo_scenes,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Wheelchair weak labels in BEV')
    parser.add_argument('--category', required=True, choices=[
        'indoor1', 'indoor2', 'outdoor1', 'outdoor2', 'people',
    ])
    parser.add_argument('--scene', required=True)
    parser.add_argument('--frame', required=True, type=int, help='Raw point-cloud frame index')
    parser.add_argument(
        '--data-root', default=str(PROJECT_ROOT / 'data/wheelchair_raw')
    )
    parser.add_argument(
        '--annotation-root', default=str(PROJECT_ROOT / 'data/wheelchair_annotations')
    )
    parser.add_argument(
        '--pseudo-info', default='wheelchair_infos_train_full_by_scene.pkl'
    )
    parser.add_argument('--point-cloud-range', nargs=6, type=float,
                        default=list(DEFAULT_POINT_CLOUD_RANGE))
    parser.add_argument('--max-points', type=int, default=80000)
    parser.add_argument(
        '--zoom-radius', type=float, default=1.5,
        help='Half-width of the local BEV panel around the first manual center',
    )
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def oriented_bev_corners(box):
    box = np.asarray(box, dtype=np.float32)
    local = np.asarray([
        [box[3] / 2, box[4] / 2],
        [box[3] / 2, -box[4] / 2],
        [-box[3] / 2, -box[4] / 2],
        [-box[3] / 2, box[4] / 2],
        [box[3] / 2, box[4] / 2],
    ])
    cos_yaw, sin_yaw = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return local @ rotation.T + box[None, :2]


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    pseudo_path = Path(args.pseudo_info).expanduser()
    if not pseudo_path.is_absolute():
        pseudo_path = annotation_root / pseudo_path

    point_path = (
        data_root / args.category / args.scene / 'points' / f'{args.frame:04d}.bin'
    )
    if not point_path.exists():
        raise FileNotFoundError(point_path)
    raw = np.fromfile(point_path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f'Malformed point cloud: {point_path}')
    points = raw.reshape(-1, 4)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) > args.max_points:
        indices = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points = points[indices]

    pseudo_by_frame, frame_indices_by_scene = load_pseudo_scenes(pseudo_path)
    manual_by_frame, _ = load_manual_centers(
        annotation_root, frame_indices_by_scene
    )
    key = (args.category, args.scene, args.frame)
    pseudo = _pseudo_arrays(pseudo_by_frame.get(key))
    manual = manual_by_frame.get(key, np.empty((0, 2), dtype=np.float32))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 3, figsize=(20, 7), dpi=150, layout='constrained'
    )
    overview_ax, local_ax, side_ax = axes
    for ax in (overview_ax, local_ax):
        ax.scatter(
            points[:, 0], points[:, 1], s=0.12, c=points[:, 2], cmap='gray',
            vmin=-1.5, vmax=2.5, rasterized=True,
        )

    pedestrian_count = 0
    valid_box_count = 0
    valid_boxes = []
    filter_cfg = PedestrianBoxFilterConfig()
    for name, center, box, score in zip(
        pseudo['name'], pseudo['center_3d'], pseudo['bbox_3d'], pseudo['score']
    ):
        if name != 'Pedestrian':
            continue
        pedestrian_count += 1
        for ax in (overview_ax, local_ax):
            ax.scatter(center[0], center[1], marker='+', s=80, linewidths=1.5,
                       c='darkorange', zorder=4)
        valid, _ = validate_box_geometry(box, filter_cfg, args.point_cloud_range)
        if valid:
            valid_box_count += 1
            valid_boxes.append((np.asarray(box), float(score)))
            corners = oriented_bev_corners(box)
            for ax in (overview_ax, local_ax):
                ax.plot(corners[:, 0], corners[:, 1], color='limegreen', linewidth=1.3)
                ax.text(box[0], box[1], f'{float(score):.2f}', fontsize=6,
                        color='limegreen')

    if len(manual):
        for ax in (overview_ax, local_ax):
            ax.scatter(manual[:, 0], manual[:, 1], marker='x', s=90,
                       linewidths=2.0, c='red', label='manual BEV center', zorder=5)
    valid_boxes.sort(key=lambda item: item[1], reverse=True)
    overview_ax.scatter([], [], marker='+', s=80, c='darkorange', label='pseudo center')
    overview_ax.plot([], [], color='limegreen', label='geometry-valid raw pseudo box')

    pc_range = np.asarray(args.point_cloud_range)
    overview_ax.set_xlim(pc_range[0], pc_range[3])
    overview_ax.set_ylim(pc_range[1], pc_range[4])
    overview_ax.set_title(
        f'Overview: {args.category}/{args.scene}/{args.frame:04d}\n'
        f'mapping=source frames[index], manual={len(manual)}, '
        f'pseudo={pedestrian_count}, valid boxes={valid_box_count}'
    )
    overview_ax.legend(loc='upper right')

    pseudo_centers = pseudo['center_3d'][pseudo['name'] == 'Pedestrian']
    if len(manual):
        focus_xy = manual[0]
        focus_name = 'first manual center'
    elif len(valid_boxes):
        focus_xy = valid_boxes[0][0][:2]
        focus_name = 'first valid pseudo box'
    elif len(pseudo_centers):
        focus_xy = pseudo_centers[0, :2]
        focus_name = 'first pseudo center'
    else:
        focus_xy = np.zeros(2)
        focus_name = 'origin'
    local_ax.set_xlim(focus_xy[0] - args.zoom_radius, focus_xy[0] + args.zoom_radius)
    local_ax.set_ylim(focus_xy[1] - args.zoom_radius, focus_xy[1] + args.zoom_radius)
    local_ax.set_title(f'Local BEV around {focus_name}')

    side_summary = 'no geometry-valid pseudo box'
    if valid_boxes:
        box, score = valid_boxes[0]
        relative = points[:, :3] - box[None, :3]
        cos_yaw, sin_yaw = np.cos(box[6]), np.sin(box[6])
        local_x = relative[:, 0] * cos_yaw + relative[:, 1] * sin_yaw
        local_y = -relative[:, 0] * sin_yaw + relative[:, 1] * cos_yaw
        side_keep = (
            (np.abs(local_x) <= max(1.0, box[3]))
            & (np.abs(local_y) <= max(0.75, box[4] * 0.75))
        )
        side_ax.scatter(
            local_x[side_keep], points[side_keep, 2], s=2.0,
            c='0.25', rasterized=True,
        )
        rectangle_x = np.asarray([
            -box[3] / 2, box[3] / 2, box[3] / 2,
            -box[3] / 2, -box[3] / 2,
        ])
        rectangle_z = np.asarray([
            box[2] - box[5] / 2, box[2] - box[5] / 2,
            box[2] + box[5] / 2, box[2] + box[5] / 2,
            box[2] - box[5] / 2,
        ])
        side_ax.plot(rectangle_x, rectangle_z, color='limegreen', linewidth=2)
        bev_keep = (
            (np.abs(local_x) <= box[3] / 2)
            & (np.abs(local_y) <= box[4] / 2)
        )
        z_relative = relative[bev_keep, 2]
        center_count = int(np.sum(np.abs(z_relative) <= box[5] / 2))
        bottom_count = int(np.sum((z_relative >= 0) & (z_relative <= box[5])))
        side_summary = (
            f'score={score:.2f}, dims={box[3]:.2f}/{box[4]:.2f}/{box[5]:.2f} m\n'
            f'points in center-z box={center_count}; bottom-z box={bottom_count}'
        )
        side_ax.set_xlim(-max(1.0, box[3]), max(1.0, box[3]))
        side_ax.set_title('First valid box: local-x / z\n' + side_summary)
    else:
        side_ax.text(0.5, 0.5, side_summary, ha='center', va='center',
                     transform=side_ax.transAxes)
        side_ax.set_title('Pseudo-box side view')

    for ax in (overview_ax, local_ax):
        ax.set_aspect('equal')
        ax.set_xlabel('LiDAR x (m)')
        ax.set_ylabel('LiDAR y (m)')
    side_ax.set_xlabel('box-local x (m)')
    side_ax.set_ylabel('LiDAR z (m)')
    for ax in axes:
        ax.grid(alpha=0.2)

    output = Path(args.output).expanduser() if args.output else (
        PROJECT_ROOT / 'output' / 'wheelchair_label_audit'
        / f'{args.category}_{args.scene}_{args.frame:04d}_source_index.png'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f'Saved {output}')
    print(f'Focus: {focus_name} at ({focus_xy[0]:.3f}, {focus_xy[1]:.3f})')
    print(f'Side-view audit: {side_summary}')


if __name__ == '__main__':
    main()
