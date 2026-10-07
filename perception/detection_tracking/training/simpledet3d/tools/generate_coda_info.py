"""
Generate pkl info files from CODa dataset for SimpleDet3D.

CODa dataset structure:
    CODa_full/
        3d_comp/os1/{seq_id}/3d_comp_os1_{seq_id}_{frame_id}.bin   # point cloud (N,4) [x,y,z,intensity]
        3d_bbox/os1/{seq_id}/3d_bbox_os1_{seq_id}_{frame_id}.json   # 3D bbox annotations
        calibrations/{seq_id}/calib_os1_to_base.yaml                 # LiDAR → base extrinsic
        poses/dense/{seq_id}.txt                                     # per line: timestamp x y z qw qx qy qz
        timestamps/{seq_id}.txt                                      # per line: unix timestamp
        metadata/{seq_id}.json                                       # scene meta + train/val/test split

Usage:
    python tools/generate_coda_info.py
    python tools/generate_coda_info.py \
        --data_root data/coda_raw \
        --output_dir output/data_preparation/coda_new
    python tools/generate_coda_info.py --pc_dir 3d_comp_downsample/os1  # use downsampled point clouds
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pyquaternion import Quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------------------
# Class mapping: CODa raw class names → SimpleDet3D target classes
# ---------------------------------------------------------------------------
CODA_CLASS_MAPPING = {
    'Pedestrian': 'Pedestrian',
    'Car': 'Car',
    'Pickup Truck': 'Car',
    'Service Vehicle': 'Car',
    'Delivery Truck': 'Car',
    'Utility Vehicle': 'Car',
    'Bus': 'Car',
    'Bike': 'Cyclist',
    'Motorcycle': 'Cyclist',
    'Scooter': 'Cyclist',
    'Segway': 'Cyclist',
    'Skateboard': 'Cyclist',
}

TARGET_CLASSES = ['Car', 'Pedestrian', 'Cyclist']


# ---------------------------------------------------------------------------
#  Skip-tracking helper
# ---------------------------------------------------------------------------

class SkipStats:
    """Track skipped frames by reason, keeping a few example tokens for each."""

    def __init__(self):
        self._counters = defaultdict(int)
        self._examples = defaultdict(list)  # reason → [token, ...]
        self._max_examples = 5

    def add(self, reason, token=None):
        self._counters[reason] += 1
        if token is not None and len(self._examples[reason]) < self._max_examples:
            self._examples[reason].append(token)

    def summary(self, prefix="  "):
        if not self._counters:
            return f"{prefix}(no frames skipped)"
        lines = []
        for reason in sorted(self._counters.keys()):
            cnt = self._counters[reason]
            tokens = self._examples[reason]
            extra = f'  (e.g. {", ".join(tokens)})' if tokens else ''
            lines.append(f"{prefix}{reason}: {cnt}{extra}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate CODa pkl info files')
    parser.add_argument('--data_root', type=str,
                        default=str(PROJECT_ROOT / 'data/coda_raw'),
                        help='Path to CODa dataset root')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / 'output/data_preparation/coda'),
                        help='Output directory for pkl files (relative to project root)')
    parser.add_argument('--sweep_frame_len', type=int, default=3,
                        help='Number of previous frames to include as sweeps')
    parser.add_argument('--max_sweep_time_lag', type=float, default=0.35,
                        help='Maximum history time lag in seconds (default: 0.35)')
    parser.add_argument('--pc_dir', type=str, default='3d_comp/os1',
                        help='Point cloud subdirectory under data_root '
                             '(e.g. 3d_comp/os1 or 3d_comp_downsample/os1)')
    parser.add_argument('--filter_class', type=str, default=None,
                        choices=TARGET_CLASSES,
                        help='Only keep frames that contain at least one box of '
                             'the specified class (e.g. Pedestrian). '
                             'Other frames remain available as unlabeled sweeps.')
    parser.add_argument('--seq_ids', type=int, nargs='+', default=None,
                        help='Sequence IDs to process (default: all 0-21)')
    return parser.parse_args()


def load_pose(pose_path):
    """Load pose file: each line = timestamp x y z qw qx qy qz → Nx8 array."""
    data = np.loadtxt(pose_path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data  # columns: [ts, x, y, z, qw, qx, qy, qz]


def load_calibration(calib_path):
    """Load calib_os1_to_base.yaml → (4, 4) extrinsic matrix."""
    with open(calib_path) as f:
        cfg = yaml.safe_load(f)
    data = np.array(cfg['extrinsic_matrix']['data'], dtype=np.float64)
    return data.reshape(cfg['extrinsic_matrix']['rows'], cfg['extrinsic_matrix']['cols'])


def pose_to_matrix(ts, x, y, z, qw, qx, qy, qz):
    """Convert pose line to (4, 4) base_link→global transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(w=qw, x=qx, y=qy, z=qz).rotation_matrix
    T[:3, 3] = [x, y, z]
    return T


def parse_annotation(json_path, class_mapping=None, skip_stats=None):
    """
    Parse a CODa 3d_bbox JSON file.

    Returns:
        gt_boxes:  (N, 7) float32  [x, y, z, dx, dy, dz, heading]
        gt_names:  (N,)   str
        valid:     bool — False when the frame should be skipped entirely.
    """
    if class_mapping is None:
        class_mapping = CODA_CLASS_MAPPING

    token = os.path.splitext(os.path.basename(json_path))[0]

    if not os.path.exists(json_path):
        if skip_stats is not None:
            skip_stats.add('annotation_file_missing', token)
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str), True

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        if skip_stats is not None:
            skip_stats.add('annotation_parse_error', token)
        print(f"  WARNING: Failed to parse {json_path}: {e}")
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str), True

    gt_boxes = []
    gt_names = []

    for obj in data.get('3dbbox', []):
        raw_cls = obj['classId']
        mapped_cls = class_mapping.get(raw_cls)
        if mapped_cls is None:
            continue  # skip non-target classes (Pole, Tree, ...)

        # CODa annotation: cX, cY, cZ (center), l, w, h (dims), r, p, y (RPY)
        cX = float(obj['cX'])
        cY = float(obj['cY'])
        cZ = float(obj['cZ'])
        l = float(obj['l'])
        w = float(obj['w'])
        h = float(obj['h'])
        yaw = float(obj['y'])

        # SimpleDet3D format: [x, y, z, dx, dy, dz, heading]
        gt_boxes.append([cX, cY, cZ, l, w, h, yaw])
        gt_names.append(mapped_cls)

    if gt_boxes:
        boxes = np.array(gt_boxes, dtype=np.float32)
        names = np.array(gt_names, dtype=str)

        # Validate boxes: no NaN / Inf, positive dimensions
        if not np.all(np.isfinite(boxes)):
            if skip_stats is not None:
                skip_stats.add('corrupt_boxes_nan_inf', token)
            print(f"  WARNING: {json_path}: NaN/Inf in boxes, dropping those boxes")
            valid_mask = np.all(np.isfinite(boxes), axis=1)
            boxes = boxes[valid_mask]
            names = names[valid_mask]
            if len(boxes) == 0:
                return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str), True

        # Check for non-positive dimensions
        dims = boxes[:, 3:6]
        if np.any(dims <= 0):
            bad = np.where(~np.all(dims > 0, axis=1))[0]
            if skip_stats is not None:
                skip_stats.add('non_positive_dimensions', token)
            print(f"  WARNING: {json_path}: {len(bad)} box(es) have non-positive "
                  f"dimensions at indices {bad.tolist()}, dropping them")
            boxes = boxes[np.all(dims > 0, axis=1)]
            names = names[np.all(dims > 0, axis=1)]
            if len(boxes) == 0:
                return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str), True

        return boxes, names, True
    else:
        # JSON exists but contains 0 target-class objects
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str), True


def validate_bin_file(bin_path):
    """Check that a .bin file exists, is readable, and has the expected shape.

    Returns:
        valid:  bool — True if the file passes all checks.
    """
    if not bin_path.exists():
        return False

    try:
        points = np.fromfile(str(bin_path), dtype=np.float32, count=-1)
    except (IOError, ValueError):
        return False

    if len(points) == 0:
        return False
    if len(points) % 4 != 0:
        return False  # expected [x, y, z, intensity]

    # Quick sanity: no all-NaN blocks
    if not np.all(np.isfinite(points)):
        return False

    return True


def index_lidar_frames(comp_seq_dir):
    """Index structurally valid LiDAR files by physical frame ID.

    Historical sweeps do not need annotations, so the index is built from the
    complete sensor stream instead of one metadata split.  A cheap file-size
    check avoids reading every raw frame during info generation; target frames
    are still fully validated by :func:`validate_bin_file`.
    """
    lidar_paths = {}
    for bin_path in comp_seq_dir.glob('*.bin'):
        try:
            frame_id = int(bin_path.stem.rsplit('_', 1)[-1])
            size_bytes = bin_path.stat().st_size
        except (OSError, ValueError):
            continue
        if size_bytes > 0 and size_bytes % (4 * np.dtype(np.float32).itemsize) == 0:
            lidar_paths[frame_id] = bin_path
    return lidar_paths


def build_sweep_infos(frame_id, current_lidar2global, current_timestamp,
                      lidar_paths, poses, lidar2base, data_root,
                      sweep_frame_len, max_sweep_time_lag):
    """Build sweeps from consecutive physical frames before ``frame_id``."""
    if sweep_frame_len <= 0:
        return []

    current_from_global = np.linalg.inv(current_lidar2global)
    sweeps_info = []
    first_frame_id = max(0, frame_id - sweep_frame_len)

    for sweep_frame_id in range(first_frame_id, frame_id):
        lidar_path = lidar_paths.get(sweep_frame_id)
        if lidar_path is None or sweep_frame_id >= len(poses):
            continue

        pose_line = poses[sweep_frame_id]
        if not np.all(np.isfinite(pose_line)):
            continue

        time_lag = float(current_timestamp - pose_line[0])
        if (not np.isfinite(time_lag) or time_lag <= 0
                or time_lag > max_sweep_time_lag):
            continue

        sweep_base2global = pose_to_matrix(*pose_line)
        sweep_lidar2global = sweep_base2global @ lidar2base
        transform_matrix = current_from_global @ sweep_lidar2global
        if not np.all(np.isfinite(transform_matrix)):
            continue

        sweeps_info.append({
            'frame_id': sweep_frame_id,
            'lidar_path': lidar_path.relative_to(data_root).as_posix(),
            'time_lag': time_lag,
            'transform_matrix': transform_matrix.astype(np.float32),
        })

    return sweeps_info


# ---------------------------------------------------------------------------
#  Main generation
# ---------------------------------------------------------------------------

def generate_coda_infos(data_root, output_dir, sweep_frame_len=3, seq_ids=None,
                        pc_dir='3d_comp/os1', filter_class=None,
                        max_sweep_time_lag=0.35):
    """
    Generate pkl info files for CODa dataset.

    For each sequence:
      1. Load metadata → get train/val/test frame lists
      2. Load calibration: lidar → base
      3. Load timestamps + poses (indexed by frame_id)
      4. For each frame in each split:
         - parse annotation → gt_boxes, gt_names
         - compute lidar2global = base2global @ lidar2base
         - accumulate sweeps from previous frames in the same sequence

    Invalid frames (missing or corrupt point clouds, corrupt annotations,
    empty GT boxes) are skipped with a warning.  A summary of all skipped
    frames is printed at the end.
    """
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sweep_frame_len < 0:
        raise ValueError('sweep_frame_len must be non-negative')
    if max_sweep_time_lag <= 0:
        raise ValueError('max_sweep_time_lag must be positive')

    if seq_ids is None:
        comp_dir = data_root / pc_dir
        seq_ids = sorted([int(d.name) for d in comp_dir.iterdir() if d.is_dir()])

    print(f"Processing sequences: {seq_ids}")

    infos = {'train': [], 'val': [], 'test': []}
    global_skip = SkipStats()

    for seq_id in seq_ids:
        print(f"\n{'='*60}")
        print(f"Processing sequence {seq_id} ...")

        seq_skip = SkipStats()

        # --- Paths ---
        comp_seq_dir = data_root / pc_dir / str(seq_id)
        annot_seq_dir = data_root / '3d_bbox' / 'os1' / str(seq_id)
        calib_path = data_root / 'calibrations' / str(seq_id) / 'calib_os1_to_base.yaml'
        pose_path = data_root / 'poses' / 'dense' / f'{seq_id}.txt'
        meta_path = data_root / 'metadata' / f'{seq_id}.json'

        # --- Load calibration ---
        if not calib_path.exists():
            print(f"  SKIP: calibration not found at {calib_path}")
            continue
        T_lidar2base = load_calibration(calib_path)

        # --- Load poses ---
        if not pose_path.exists():
            print(f"  SKIP: poses not found at {pose_path}")
            continue
        poses = load_pose(pose_path)

        # Validate poses
        n_bad_poses = np.sum(~np.isfinite(poses))
        if n_bad_poses > 0:
            print(f"  WARNING: {n_bad_poses} NaN/Inf pose values; affected frames "
                  f"will not be used as targets or sweeps")

        lidar_paths = index_lidar_frames(comp_seq_dir)
        if not lidar_paths:
            print(f"  SKIP: no valid point clouds found at {comp_seq_dir}")
            continue

        # --- Load metadata (split info) ---
        if not meta_path.exists():
            print(f"  SKIP: metadata not found at {meta_path}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        obj_tracking = meta.get('ObjectTracking', {})

        # Parse split lists
        split_frames = {}
        for split_name in ['training', 'validation', 'testing']:
            paths = obj_tracking.get(split_name, [])
            frame_ids = []
            for p in paths:
                fname = os.path.basename(p)
                stem = os.path.splitext(fname)[0]
                frame_id = int(stem.split('_')[-1])
                frame_ids.append(frame_id)
            split_frames[split_name] = sorted(frame_ids)

        print(f"  training: {len(split_frames['training'])} frames  "
              f"validation: {len(split_frames['validation'])} frames  "
              f"testing: {len(split_frames['testing'])} frames")
        print(f"  poses loaded: {len(poses)}  calibration: loaded")

        # --- Process each split ---
        split_map = {'training': 'train', 'validation': 'val', 'testing': 'test'}

        for raw_split, out_split in split_map.items():
            frame_ids = split_frames[raw_split]
            if not frame_ids:
                continue

            for frame_id in frame_ids:
                token = f'{seq_id}_{frame_id}'

                # --- frame_id bounds ---
                if frame_id >= len(poses):
                    seq_skip.add('frame_id_exceeds_poses', token)
                    continue

                # --- Validate point cloud ---
                bin_path = lidar_paths.get(frame_id)
                if bin_path is None:
                    bin_name = f'3d_comp_os1_{seq_id}_{frame_id}.bin'
                    bin_path = comp_seq_dir / bin_name
                if not validate_bin_file(bin_path):
                    reason = ('bin_missing' if not bin_path.exists()
                              else 'bin_corrupt_or_empty')
                    seq_skip.add(reason, token)
                    continue  # do NOT add to sweep_buffer either
                lidar_path = bin_path.relative_to(data_root).as_posix()

                # --- Annotation ---
                annot_name = f'3d_bbox_os1_{seq_id}_{frame_id}.json'
                annot_path = annot_seq_dir / annot_name
                gt_boxes, gt_names, _ = parse_annotation(str(annot_path),
                                                          skip_stats=seq_skip)

                # Skip frames with no GT boxes (waste of training time)
                if len(gt_boxes) == 0:
                    seq_skip.add('empty_gt_boxes', token)
                    continue

                # Optional class filter: only keep frames containing a specific class
                if filter_class is not None:
                    if filter_class not in gt_names:
                        seq_skip.add(f'filter_no_{filter_class}', token)
                        continue
                    # Frame-level filter passed — also remove boxes of other classes
                    mask = gt_names == filter_class
                    gt_boxes = gt_boxes[mask]
                    gt_names = gt_names[mask]

                # --- Pose ---
                pose_line = poses[frame_id]
                if not np.all(np.isfinite(pose_line)):
                    seq_skip.add('invalid_pose', token)
                    continue
                ts_val = pose_line[0]
                T_base2global = pose_to_matrix(*pose_line)

                # --- lidar2global ---
                T_lidar2global = T_base2global @ T_lidar2base
                T_lidar2global = T_lidar2global.astype(np.float32)

                # --- Sweeps ---
                sweeps_info = build_sweep_infos(
                    frame_id=frame_id,
                    current_lidar2global=T_lidar2global,
                    current_timestamp=ts_val,
                    lidar_paths=lidar_paths,
                    poses=poses,
                    lidar2base=T_lidar2base,
                    data_root=data_root,
                    sweep_frame_len=sweep_frame_len,
                    max_sweep_time_lag=max_sweep_time_lag,
                )

                # --- Frame info ---
                info = {
                    'token': token,
                    'lidar_path': lidar_path,
                    'gt_boxes': gt_boxes,
                    'gt_names': gt_names,
                    'sweeps': sweeps_info,
                    'lidar2global': T_lidar2global,
                }
                infos[out_split].append(info)

        # Per-sequence skip summary
        print(f"\n  Sequence {seq_id} skip summary:")
        print(seq_skip.summary())

        # Merge into global stats
        for reason, cnt in seq_skip._counters.items():
            global_skip._counters[reason] += cnt
        for reason, tokens in seq_skip._examples.items():
            existing = global_skip._examples[reason]
            for t in tokens:
                if t not in existing:
                    existing.append(t)
            global_skip._examples[reason] = existing[:global_skip._max_examples]

        print(f"  Sequence {seq_id} done: "
              f"train={len([x for x in infos['train'] if x['token'].split('_', 1)[0] == str(seq_id)])}, "
              f"val={len([x for x in infos['val'] if x['token'].split('_', 1)[0] == str(seq_id)])}, "
              f"test={len([x for x in infos['test'] if x['token'].split('_', 1)[0] == str(seq_id)])}")

    # ------------------------------------------------------------------
    #  Final counts
    # ------------------------------------------------------------------
    total_skipped = sum(global_skip._counters.values())
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total valid frames: {sum(len(v) for v in infos.values())}")
    print(f"Total skipped:       {total_skipped}")
    print()
    print("Skip reason breakdown:")
    print(global_skip.summary(prefix="  "))

    # ------------------------------------------------------------------
    #  Save pkl files
    # ------------------------------------------------------------------
    for split_name in ['train', 'val', 'test']:
        if not infos[split_name]:
            print(f"\nNo frames in {split_name} split, skipping.")
            continue
        save_path = output_dir / f'coda_infos_{split_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(infos[split_name], f)
        print(f"\nSaved {len(infos[split_name])} frames to {save_path}")

        class_counts = defaultdict(int)
        empty_count = 0
        for info in infos[split_name]:
            if len(info['gt_names']) == 0:
                empty_count += 1
            for name in info['gt_names']:
                class_counts[name] += 1
        print(f"  Empty frames: {empty_count}")
        print(f"  Class distribution:")
        for cls in sorted(class_counts.keys()):
            print(f"    {cls}: {class_counts[cls]}")

    return infos


def main():
    args = parse_args()
    generate_coda_infos(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sweep_frame_len=args.sweep_frame_len,
        max_sweep_time_lag=args.max_sweep_time_lag,
        seq_ids=args.seq_ids,
        pc_dir=args.pc_dir,
        filter_class=args.filter_class,
    )


if __name__ == '__main__':
    main()
