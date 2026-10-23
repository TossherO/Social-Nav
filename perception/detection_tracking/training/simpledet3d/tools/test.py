"""
Simplified inference/test script for 3D object detection.
Usage:
    python tools/test.py --cfg_file configs/centerpoint_coda.yaml --ckpt output/xxx/ckpt/checkpoint_epoch_20.pth
    python tools/test.py --cfg_file configs/centerpoint_coda.yaml --ckpt output/xxx/ckpt/checkpoint_epoch_20.pth --infer_time
    python tools/test.py --cfg_file configs/centerpoint_coda.yaml --ckpt output/xxx/ckpt/checkpoint_epoch_20.pth --save_vis
"""
import argparse
import datetime
import glob
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_det3d.utils.config import load_config, log_config_to_file
from simple_det3d.utils.common_utils import create_logger, AverageMeter
from simple_det3d.datasets.coda_dataset import CodaDataset
from simple_det3d.datasets.task_scene_dataset import build_task_scene_dataset
from simple_det3d.models import build_model, load_data_to_gpu
from simple_det3d.utils.precision_utils import (
    autocast_context,
    resolve_precision,
    validate_precision_support,
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Test 3D object detector')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--extra_tag', type=str, default='default', help='Extra tag')
    parser.add_argument('--infer_time', action='store_true', default=False, help='Measure inference latency')
    parser.add_argument('--save_vis', action='store_true', default=False, help='Save BEV visualizations')
    parser.add_argument('--vis_output_dir', type=str, default=None, help='Visualization output directory')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default=None,
                        help='Runtime precision (default: cfg.PRECISION or fp32)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Alias for --precision fp16')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Override info/GT-database root path')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Override raw point-cloud dataset root path')
    parser.add_argument('--max_sweeps', type=int, default=None, help='Match the checkpoint temporal input; default from config')

    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.LOCAL_RANK = 0
    cfg.ROOT_DIR = Path(__file__).resolve().parent.parent
    if cfg.get('TASK_SCENE', {}).get('ENABLED') and (args.root_path is not None or args.raw_data_path is not None):
        parser.error('Task-scene data roots must be set in each TASK_SCENE dataset CONFIG')

    if args.root_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.root_path
    if args.raw_data_path is not None:
        cfg.DATA_CONFIG.RAW_DATA_PATH = args.raw_data_path
    if args.max_sweeps is not None:
        if args.max_sweeps < 0:
            parser.error('--max_sweeps must be non-negative')
        cfg.DATA_CONFIG.MAX_SWEEPS = max(1, args.max_sweeps)

    args.precision = resolve_precision(
        cli_precision=args.precision,
        config_precision=cfg.get('PRECISION', 'fp32'),
        use_amp=args.use_amp,
    )

    return args, cfg


def eval_one_epoch(model, dataloader, cfg, args, logger, result_dir=None):
    """Run evaluation on the entire dataset."""
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)

    metric = {'gt_num': 0}
    center_only = cfg.DATA_CONFIG.get('WEAK_SUPERVISION', {}).get('CENTER_EVAL', False) if hasattr(cfg, 'DATA_CONFIG') else False
    recall_thresholds = [] if center_only else cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
    for cur_thresh in recall_thresholds:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    class_names = dataloader.dataset.class_names
    det_annos = []

    logger.info('*************** EVALUATION *****************')
    model.eval()

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    # Keep no_grad instead of inference_mode: the bundled PointNet2 CUDA
    # extensions do not fully support inference tensors in BF16 autocast.
    with torch.no_grad():
        for i, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)

            with autocast_context(args.precision):
                pred_dicts, ret_dict = model(batch_dict)

            disp_dict = {}

            # Update recall stats
            for cur_thresh in recall_thresholds:
                metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
                metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
            metric['gt_num'] += ret_dict.get('gt', 0)

            if recall_thresholds:
                min_thresh = recall_thresholds[0]
                disp_dict['recall_%s' % str(min_thresh)] = \
                    '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)],
                                       metric['recall_rcnn_%s' % str(min_thresh)],
                                       metric['gt_num'])

            annos = dataloader.dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
            )
            det_annos += annos
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    progress_bar.close()

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished (sec_per_example: %.4f second).' % sec_per_example)

    # Compute recall
    gt_num_cnt = metric['gt_num']
    ret_dict = {}
    for cur_thresh in recall_thresholds:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # Save results
    if result_dir is not None:
        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)

    # Run evaluation
    result_str, result_dict = dataloader.dataset.evaluation(det_annos, class_names)
    logger.info(result_str)
    ret_dict.update(result_dict)

    # BEV visualization
    # ``eval_one_epoch`` is shared by the standalone test entry point and
    # train.py's final/mid-training evaluation.  Training arguments do not
    # define the test-only visualization switch, so default it to disabled.
    if getattr(args, 'save_vis', False) and result_dir is not None:
        save_bev_visualization(dataloader.dataset, det_annos, args, result_dir)

    logger.info('****************Evaluation done.*****************')
    return ret_dict


def save_bev_visualization(dataset, det_annos, args, result_dir):
    """Save BEV visualization images."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    vis_dir = result_dir / 'bev_vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    class_colors = {
        'Car': '#FF0000',
        'Pedestrian': '#00FF00',
        'Cyclist': '#0000FF',
    }

    # Get point cloud range
    pc_range = dataset.point_cloud_range

    for i, det in enumerate(det_annos[:50]):  # Limit to first 50 frames
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Draw point cloud range
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f"BEV Detection - {det['frame_id']}")
        ax.grid(True, alpha=0.3)

        # Draw predicted boxes
        names = det['name']
        boxes = det['boxes_lidar']
        scores = det['score']

        for name, box, score in zip(names, boxes, scores):
            x, y, z, dx, dy, dz, heading = box[:7]
            color = class_colors.get(name, '#FFFFFF')

            # Draw box center
            ax.plot(x, y, 'o', color=color, markersize=3)

            # Draw box outline (simplified as rectangle for BEV)
            corners_x = [x - dx / 2, x + dx / 2, x + dx / 2, x - dx / 2, x - dx / 2]
            corners_y = [y - dy / 2, y - dy / 2, y + dy / 2, y + dy / 2, y - dy / 2]
            ax.plot(corners_x, corners_y, '-', color=color, linewidth=0.5, alpha=0.7)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=n) for n, c in class_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(vis_dir / f"{det['frame_id']}_bev.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {min(50, len(det_annos))} BEV visualizations to {vis_dir}")


def measure_inference_time(model, dataloader, cfg, args, logger):
    """Measure inference time."""
    logger.info('*************** MEASURING INFERENCE TIME *****************')
    model.eval()

    # Warmup
    logger.info('Warmup...')
    for i, batch_dict in enumerate(dataloader):
        if i >= 10:
            break
        load_data_to_gpu(batch_dict)
        with torch.no_grad(), autocast_context(args.precision):
            model(batch_dict)

    # Measure
    logger.info('Measuring...')
    infer_time_meter = AverageMeter()
    torch.cuda.synchronize()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad(), autocast_context(args.precision):
            model(batch_dict)

        torch.cuda.synchronize()
        t_end = time.time()

        infer_time_meter.update((t_end - t_start) * 1000)  # ms

        if i >= 100:
            break

    logger.info(f'Average inference time: {infer_time_meter.avg:.2f} ms')
    logger.info(f'FPS: {1000.0 / infer_time_meter.avg:.2f}')
    return infer_time_meter.avg


def eval_checkpoints(model, test_loader, args, eval_output_dir, logger, ckpt_dir, cfg):
    """Evaluate all checkpoints in ckpt_dir."""
    ckpt_list = glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    if not ckpt_list:
        logger.info('No checkpoints found for evaluation')
        return
    logger.info(f'Found {len(ckpt_list)} checkpoints to evaluate')


def main():
    args, cfg = parse_args()
    validate_precision_support(args.precision, device=0)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    # Setup output
    output_dir = Path('./output') / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval' / 'eval_single'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = create_logger(log_file)

    logger.info('**********************Start logging**********************')
    logger.info('CUDA_VISIBLE_DEVICES=%s' % os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL'))

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # Build dataset
    root_path = Path(cfg.DATA_CONFIG.DATA_PATH)
    if not root_path.is_absolute():
        root_path = cfg.ROOT_DIR / root_path

    test_set = build_task_scene_dataset(cfg, training=False, logger=logger) if cfg.get('TASK_SCENE', {}).get('ENABLED') else CodaDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=root_path,
        logger=logger,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        shuffle=False,
        collate_fn=test_set.collate_batch,
        drop_last=False,
    )

    # Build model and load checkpoint
    model = build_model(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger)
    model.cuda()

    logger.info(f'Loaded checkpoint from {args.ckpt}')

    if args.infer_time:
        measure_inference_time(model, test_loader, cfg, args, logger)
    else:
        eval_one_epoch(model, test_loader, cfg, args, logger, result_dir=eval_output_dir)

    logger.info('**********************Done**********************')


if __name__ == '__main__':
    main()
