"""
Generate pkl info files from Wheelchair dataset for SimpleDet3D.

Wheelchair 数据集原始没有标注，本脚本生成仅含点云路径、位姿和 sweeps 的 pkl，
gt_boxes 和 gt_names 为空，供推理使用。标注需另行补充。

Wheelchair dataset structure:
    data_root/
        indoor1/, indoor2/, outdoor1/, outdoor2/, people/, old1/, old2/
            <scene>/
                points/     # {frame_id:04d}.bin  (N,4) [x,y,z,intensity]
                imgs/       # {frame_id:04d}.png  (未使用)
                poses/      # {frame_id:04d}.txt  (可选) 格式: pose{id}: x y z qx qy qz qw
            calib.txt        # 相机标定 (未使用，poses 直接为 LiDAR 位姿)

Usage:
    python tools/generate_wheelchair_info.py
    python tools/generate_wheelchair_info.py --data_root data/wheelchair_raw --output_dir output/data_preparation/wheelchair_new
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Wheelchair pkl info files')
    parser.add_argument('--data_root', type=str,
                        default=str(PROJECT_ROOT / 'data/wheelchair_raw'),
                        help='Path to Wheelchair dataset root')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / 'output/data_preparation/wheelchair'),
                        help='Output directory for pkl files')
    parser.add_argument('--sweep_frame_len', type=int, default=3,
                        help='Number of previous frames to include as sweeps')
    parser.add_argument('--max_sweep_time_lag', type=float, default=0.35,
                        help='Maximum history time lag in seconds (default: 0.35)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of scenes to hold out for validation (per category)')
    return parser.parse_args()


CATEGORY_DIRS = [
    'indoor1', 'indoor2', 'outdoor1', 'outdoor2', 'people',
    # old1/old2 包含额外数据，根据需要可取消注释
    # 'old1', 'old2',
]

# The platform is stationary in these categories when pose files are absent,
# so identity transforms are valid for temporal point-cloud fusion.
STATIC_WITHOUT_POSE_CATEGORIES = {'indoor1', 'people'}


def parse_pose_file(pose_path):
    """
    解析单个 pose 文件，返回 (4,4) lidar2global 变换矩阵。

    Pose 格式: pose{frame_id}: x y z qx qy qz qw
    若无文件或格式错误，返回 None。
    """
    try:
        with open(pose_path) as f:
            line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        if len(parts) < 8:
            return None
        # parts[0] = "poseXXXX:", parts[1:8] = x y z qx qy qz qw
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
    except (FileNotFoundError, ValueError, IndexError):
        return None

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(w=qw, x=qx, y=qy, z=qz).rotation_matrix
    T[:3, 3] = [x, y, z]
    return T


def process_category(category_dir, category_name, output_dir, sweep_frame_len,
                     max_sweep_time_lag, val_scenes):
    """
    处理一个类别目录（如 indoor1），返回 train/val 帧信息列表。

    Args:
        category_dir: Path, 类别目录
        category_name: str
        output_dir: Path (unused, just for output)
        sweep_frame_len: int
        max_sweep_time_lag: float
        val_scenes: set of scene names to put in val split

    Returns:
        train_infos, val_infos: lists of info dicts
    """
    train_infos = []
    val_infos = []
    category_dir = Path(category_dir).resolve()
    data_root = category_dir.parent

    scene_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])
    print(f"  {category_name}: {len(scene_dirs)} scenes")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        points_dir = scene_dir / 'points'
        poses_dir = scene_dir / 'poses'

        if not points_dir.exists():
            print(f"    WARNING: no points/ in {scene_name}, skipping")
            continue

        # 获取所有点云文件
        bin_files = sorted(points_dir.glob('*.bin'))
        if not bin_files:
            print(f"    WARNING: no .bin files in {scene_name}/points/, skipping")
            continue

        # 确定属于 train 还是 val
        is_val = scene_name in val_scenes
        target_list = val_infos if is_val else train_infos

        # 提取 frame indices
        frame_indices = sorted([int(f.stem) for f in bin_files])
        split_label = 'val' if is_val else 'train'
        print(f"    {scene_name}: {len(frame_indices)} frames → {split_label}")

        # Sweep buffer: 前一帧的 (lidar_path, lidar2global)
        sweep_buffer = []

        for frame_idx in frame_indices:
            bin_path = points_dir / f'{frame_idx:04d}.bin'
            if not bin_path.exists():
                # 回退: 尝试直接用文件名
                bin_candidates = list(points_dir.glob(f'{frame_idx:04d}*'))
                if bin_candidates:
                    bin_path = bin_candidates[0]
                else:
                    continue

            # --- lidar2global ---
            lidar2global = np.eye(4, dtype=np.float64)
            pose_valid = False
            if poses_dir.exists():
                pose_path = poses_dir / f'{frame_idx:04d}.txt'
                T = parse_pose_file(pose_path)
                if T is not None:
                    lidar2global = T
                    pose_valid = True

            lidar2global = lidar2global.astype(np.float32)
            pose_usable = pose_valid or category_name in STATIC_WITHOUT_POSE_CATEGORIES

            # --- Sweeps ---
            # Remove stale frames before they can be used at a discontinuity.
            sweep_buffer = [
                sweep for sweep in sweep_buffer
                if 0 < frame_idx - sweep['frame_index'] <= sweep_frame_len
            ]
            sweeps_info = []
            candidate_sweeps = sweep_buffer if pose_usable else []
            for sweep in candidate_sweeps:
                if not sweep['pose_usable']:
                    continue
                time_lag = (frame_idx - sweep['frame_index']) * 0.1  # 假设 10Hz
                if time_lag <= 0 or time_lag > max_sweep_time_lag:
                    continue
                # 变换: sweep lidar → current lidar
                T_rel = np.linalg.inv(lidar2global) @ sweep['lidar2global']
                if not np.all(np.isfinite(T_rel)):
                    continue
                sweeps_info.append({
                    'frame_id': sweep['frame_index'],
                    'lidar_path': sweep['lidar_path'],
                    'time_lag': time_lag,
                    'transform_matrix': T_rel.astype(np.float32),
                })

            # 只保留最近的 sweep_frame_len 个
            sweeps_info = (
                sweeps_info[-sweep_frame_len:] if sweep_frame_len > 0 else []
            )

            # --- Token ---
            token = f'{category_name}_{scene_name}_{frame_idx:04d}'
            lidar_path = bin_path.relative_to(data_root).as_posix()

            info = {
                'token': token,
                'lidar_path': lidar_path,
                'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                'gt_names': np.array([], dtype=str),
                'sweeps': sweeps_info,
                'lidar2global': lidar2global,
                'has_pose': pose_valid,
            }
            target_list.append(info)

            # 将当前帧加入 sweep buffer
            sweep_buffer.append({
                'frame_index': frame_idx,
                'lidar_path': lidar_path,
                'lidar2global': lidar2global,
                'pose_valid': pose_valid,
                'pose_usable': pose_usable,
            })

    return train_infos, val_infos


def generate_wheelchair_infos(data_root, output_dir, sweep_frame_len=3,
                              max_sweep_time_lag=0.35, val_split=0.1):
    """
    生成 Wheelchair 数据集 pkl 信息文件。
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sweep_frame_len < 0:
        raise ValueError('sweep_frame_len must be non-negative')
    if max_sweep_time_lag <= 0:
        raise ValueError('max_sweep_time_lag must be positive')

    all_train = []
    all_val = []

    # 分场景: 按每个类别取最后一小部分场景作为验证集
    for cat_name in CATEGORY_DIRS:
        cat_dir = data_root / cat_name
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} not found, skipping")
            continue

        scenes = sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])
        if not scenes:
            continue

        n_val = max(1, int(len(scenes) * val_split))
        val_scenes = set(scenes[-n_val:])

        print(f"\nProcessing {cat_name}/ ({len(scenes)} scenes, {n_val} val)")
        train_infos, val_infos = process_category(
            cat_dir, cat_name, output_dir, sweep_frame_len,
            max_sweep_time_lag, val_scenes
        )
        all_train.extend(train_infos)
        all_val.extend(val_infos)
        print(f"  → train: {len(train_infos)} frames, val: {len(val_infos)} frames")

    # --- 统计 ---
    total_train_frames = len(all_train)
    total_val_frames = len(all_val)
    total_frames = total_train_frames + total_val_frames

    train_with_pose = sum(d['has_pose'] for d in all_train)
    val_with_pose = sum(d['has_pose'] for d in all_val)

    print(f"\n{'='*60}")
    print(f"Total: {total_frames} frames ({total_train_frames} train / {total_val_frames} val)")
    print(f"With valid pose: {train_with_pose} train / {val_with_pose} val")
    print(f"Without pose: {total_train_frames - train_with_pose} train / {total_val_frames - val_with_pose} val")

    # --- 保存 ---
    for split_name, infos in [('train', all_train), ('val', all_val)]:
        if not infos:
            print(f"\nNo frames in {split_name}, skipping.")
            continue
        save_path = output_dir / f'wheelchair_infos_{split_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(infos, f)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"\nSaved {len(infos)} frames to {save_path} ({size_mb:.1f} MB)")

    return all_train, all_val


def main():
    args = parse_args()
    generate_wheelchair_infos(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sweep_frame_len=args.sweep_frame_len,
        max_sweep_time_lag=args.max_sweep_time_lag,
        val_split=args.val_split,
    )


if __name__ == '__main__':
    main()
