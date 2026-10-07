"""
Generate ground-truth database for GT-sampling data augmentation.

Usage:
    python tools/generate_gt_database.py --cfg_file configs/voxelrcnn_coda.yaml --split train
    python tools/generate_gt_database.py --cfg_file configs/centerpoint_coda.yaml --split train --save_name coda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_det3d.utils.config import load_config
from simple_det3d.utils.common_utils import create_logger
from simple_det3d.datasets.coda_dataset import CodaDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate GT database')
    parser.add_argument('--cfg_file', type=str, required=True, help='Config file')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Override info/GT-database root path')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Override raw point-cloud dataset root path')
    parser.add_argument('--info_path', type=str, default=None, help='Existing info pkl file')
    parser.add_argument('--split', type=str, default='train', help='Data split')
    parser.add_argument('--save_name', type=str, default=None, help='Custom database folder name')
    parser.add_argument('--used_classes', type=str, nargs='*', default=None, help='Filter by classes')
    parser.add_argument('--max_sweeps', type=int, default=None, help='Override max sweeps')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.cfg_file)
    cfg.ROOT_DIR = Path(__file__).resolve().parent.parent
    if args.raw_data_path is not None:
        cfg.DATA_CONFIG.RAW_DATA_PATH = args.raw_data_path

    root_path = Path(args.root_path) if args.root_path else Path(cfg.DATA_CONFIG.DATA_PATH)
    if not root_path.is_absolute():
        root_path = cfg.ROOT_DIR / root_path

    logger = create_logger()
    logger.info(f"Generating GT database for {args.cfg_file}")

    dataset = CodaDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=root_path,
        training=True,  # Use train mode to access gt_boxes
        logger=logger,
    )

    dataset.create_groundtruth_database(
        info_path=args.info_path,
        used_classes=args.used_classes,
        split=args.split,
        max_sweeps=args.max_sweeps,
        save_name=args.save_name,
    )

    logger.info("Done!")


if __name__ == '__main__':
    main()
