"""
Simplified training script for 3D object detection.

Usage (single GPU):
    python tools/train.py --cfg_file configs/centerpoint_coda.yaml

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=2 tools/train.py --cfg_file configs/voxelrcnn_coda.yaml
    torchrun --nproc_per_node=2 tools/train.py --cfg_file configs/voxelrcnn_coda.yaml \
        --batch_size 8 --eval_interval 5
"""
import argparse
import copy
import datetime
import glob
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_det3d.utils.config import load_config, log_config_to_file
from simple_det3d.utils.common_utils import create_logger, set_random_seed, get_dist_info
from simple_det3d.datasets.coda_dataset import CodaDataset
from simple_det3d.datasets.mixed_domain_dataset import (
    MixedDomainDataset,
    MultiDomainDataset,
)
from simple_det3d.models import build_model
from simple_det3d.train.optimization import build_optimizer, build_scheduler
from simple_det3d.train.ema_utils import (
    create_ema_teacher,
    freeze_modules_by_name,
    keep_frozen_modules_eval,
    load_ema_teacher_from_checkpoint,
)
from simple_det3d.train.train_utils import (
    load_best_metric_state,
    model_fn_decorator,
    train_model,
)
from simple_det3d.utils.precision_utils import (
    create_grad_scaler,
    resolve_precision,
    validate_precision_support,
)

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D object detector')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--extra_tag', type=str, default='default', help='Extra tag for output directory')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Pretrained model weights')
    parser.add_argument('--fix_random_seed', action='store_true', default=False)
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='Max number of saved checkpoints')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='Log every N iterations')
    parser.add_argument(
        '--log_gpu_memory', action='store_true',
        help='Log per-rank point/voxel counts and CUDA memory at log intervals',
    )
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='Save latest model every N seconds')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default=None,
                        help='Runtime precision (default: cfg.PRECISION or fp32)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Deprecated alias for --precision fp16')
    parser.add_argument('--eval_interval', type=int, default=0,
                        help='Evaluate on val set every N epochs (0 = disable)')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Override info/GT-database root path')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Override raw point-cloud dataset root path')
    parser.add_argument(
        '--max_sweeps', type=int, default=None,
        help='Override MAX_SWEEPS for target and source datasets; 0/1 means current frame only',
    )
    parser.add_argument('--local_rank', type=int, default=None,
                        help='Local rank for distributed training (set by torchrun)')

    args = parser.parse_args()

    # Resolve local_rank: torchrun sets LOCAL_RANK env var; --local_rank overrides
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Load config
    cfg = load_config(args.cfg_file)

    if args.root_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.root_path
    if args.raw_data_path is not None:
        cfg.DATA_CONFIG.RAW_DATA_PATH = args.raw_data_path
    if args.max_sweeps is not None:
        if args.max_sweeps < 0:
            parser.error('--max_sweeps must be non-negative')
        cfg.DATA_CONFIG.MAX_SWEEPS = args.max_sweeps
        if 'SOURCE_DATA_CONFIG' in cfg:
            cfg.SOURCE_DATA_CONFIG.MAX_SWEEPS = args.max_sweeps

    args.precision = resolve_precision(
        cli_precision=args.precision,
        config_precision=cfg.get('PRECISION', 'fp32'),
        use_amp=args.use_amp,
    )

    cfg.TAG = Path(args.cfg_file).stem
    cfg.LOCAL_RANK = args.local_rank
    cfg.ROOT_DIR = Path(__file__).resolve().parent.parent

    return args, cfg


def build_dataloader(dataset_cfg, class_names, batch_size, training, workers=4,
                     root_path=None, logger=None, seed=None, sampler=None,
                     dataset=None):
    """Build dataset and dataloader.

    Args:
        sampler:  optional ``DistributedSampler``.  When provided, ``shuffle``
                  and the sampler are mutually exclusive in DataLoader.
        dataset:  pre-built ``CodaDataset`` (if provided, skips creation).
    """
    if dataset is None:
        dataset = CodaDataset(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        persistent_workers=(workers > 0),
        shuffle=(training and sampler is None),
        sampler=sampler,
        collate_fn=dataset.collate_batch,
        drop_last=False,
    )

    return dataset, dataloader


def main():
    args, cfg = parse_args()

    # ------------------------------------------------------------------
    #  Distributed training initialisation
    # ------------------------------------------------------------------
    # Bind the local CUDA device before NCCL initialization.  Otherwise every
    # torchrun child first creates a context on the default logical GPU 0.
    torch.cuda.set_device(args.local_rank)
    if args.local_rank > 0 or 'RANK' in os.environ:
        dist.init_process_group(
            backend='nccl', device_id=torch.device('cuda', args.local_rank),
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    validate_precision_support(args.precision, device=args.local_rank)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    if args.epochs is None:
        args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS

    if args.fix_random_seed:
        set_random_seed(666)

    # ------------------------------------------------------------------
    #  Output directory & logging (rank 0 only)
    # ------------------------------------------------------------------
    output_dir = Path('./output') / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = create_logger(log_file if rank == 0 else None, rank=rank)

    if rank == 0:
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
        logger.info('World size: %d  Rank: %d  local_rank: %d' % (world_size, rank, args.local_rank))

        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if rank == 0 else None

    # ------------------------------------------------------------------
    #  Dataset & DataLoader
    # ------------------------------------------------------------------
    root_path = Path(cfg.DATA_CONFIG.DATA_PATH)
    if not root_path.is_absolute():
        root_path = cfg.ROOT_DIR / root_path

    logger.info("----------- Create dataloader & network & optimizer -----------")

    # Training data: optionally mix a fully supervised source dataset with the
    # weakly supervised target dataset.  Both datasets must share geometry and
    # point-feature definitions.
    target_train_set = CodaDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,
        root_path=root_path,
        logger=logger,
    )
    mixed_cfg = cfg.get('MIXED_DOMAIN', {})
    if mixed_cfg.get('ENABLED', False):
        if cfg.get('SOURCE_DATA_CONFIG') is None:
            raise ValueError('MIXED_DOMAIN.ENABLED requires SOURCE_DATA_CONFIG')
        source_root_path = Path(cfg.SOURCE_DATA_CONFIG.DATA_PATH)
        if not source_root_path.is_absolute():
            source_root_path = cfg.ROOT_DIR / source_root_path
        source_train_set = CodaDataset(
            dataset_cfg=cfg.SOURCE_DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=True,
            root_path=source_root_path,
            logger=logger,
        )
        secondary_cfg = mixed_cfg.get('SECONDARY_TARGET', {})
        if secondary_cfg.get('ENABLED', False):
            secondary_dataset_cfg = copy.deepcopy(cfg.DATA_CONFIG)
            secondary_dataset_cfg.INFO_PATH.train = list(secondary_cfg.INFO_PATH)
            secondary_target_set = CodaDataset(
                dataset_cfg=secondary_dataset_cfg,
                class_names=cfg.CLASS_NAMES,
                training=True,
                root_path=root_path,
                logger=logger,
            )
            source_ratio = float(mixed_cfg.get('SOURCE_RATIO', 0.25))
            primary_ratio = float(mixed_cfg.get('TARGET_PRIMARY_RATIO', 0.5))
            secondary_ratio = float(secondary_cfg.get('RATIO', 0.25))
            train_set = MultiDomainDataset(
                datasets=[source_train_set, target_train_set, secondary_target_set],
                ratios=[source_ratio, primary_ratio, secondary_ratio],
                names=['source', 'target_exhaustive', 'target_non_exhaustive'],
                epoch_size=mixed_cfg.get('EPOCH_SIZE', None),
                reference_dataset_index=1,
            )
            logger.info(
                'Three-way training: source=%d exhaustive=%d non_exhaustive=%d '
                'epoch_size=%d ratios=%.3f/%.3f/%.3f',
                len(source_train_set), len(target_train_set),
                len(secondary_target_set), len(train_set), source_ratio,
                primary_ratio, secondary_ratio,
            )
        else:
            train_set = MixedDomainDataset(
                source_dataset=source_train_set,
                target_dataset=target_train_set,
                source_ratio=float(mixed_cfg.get('SOURCE_RATIO', 0.5)),
                epoch_size=mixed_cfg.get('EPOCH_SIZE', None),
            )
            logger.info(
                'Mixed-domain training: source=%d target=%d epoch_size=%d source_ratio=%.3f',
                len(source_train_set), len(target_train_set), len(train_set),
                train_set.source_ratio,
            )
    else:
        train_set = target_train_set

    train_sampler = None
    if dist.is_initialized():
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True,
        )

    train_set, train_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        training=True,
        workers=args.workers,
        root_path=root_path,
        logger=logger,
        seed=666 if args.fix_random_seed else None,
        sampler=train_sampler,
        dataset=train_set,  # reuse the pre-built dataset
    )

    # ------------------------------------------------------------------
    #  Model & Optimizer
    # ------------------------------------------------------------------
    model = build_model(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda()

    freeze_cfg = cfg.get('MODEL_FREEZE', {})
    frozen_names = list(freeze_cfg.get('MODULE_NAMES', []))
    freeze_train_mode_names = list(freeze_cfg.get('TRAIN_MODE_MODULE_NAMES', []))
    freeze_stats = freeze_modules_by_name(
        model, frozen_names, train_mode_module_names=freeze_train_mode_names,
    )
    if rank == 0 and frozen_names:
        logger.info(
            'Frozen model modules: %s (%d parameters)',
            ', '.join(frozen_names), freeze_stats['parameters'],
        )

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    scaler = create_grad_scaler(args.precision)

    # Wrap model with DDP
    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
        )
    checkpoint_model = model.module if isinstance(
        model, nn.parallel.DistributedDataParallel
    ) else model

    if rank == 0:
        logger.info('Precision mode: %s (GradScaler: %s)',
                    args.precision, 'enabled' if scaler.is_enabled() else 'disabled')

    # ------------------------------------------------------------------
    #  Checkpoint loading
    # ------------------------------------------------------------------
    start_epoch = it = 0
    last_epoch = -1
    resume_checkpoint_path = None
    if args.pretrained_model is not None:
        checkpoint_model.load_params_from_file(
            filename=args.pretrained_model, to_cpu=True, logger=logger,
        )

    if args.ckpt is not None:
        it, start_epoch = checkpoint_model.load_params_with_optimizer(
            args.ckpt, to_cpu=True, optimizer=optimizer, scaler=scaler, logger=logger
        )
        resume_checkpoint_path = args.ckpt
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = checkpoint_model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=True, optimizer=optimizer,
                        scaler=scaler, logger=logger,
                    )
                    resume_checkpoint_path = ckpt_list[-1]
                    last_epoch = start_epoch + 1
                    break
                except Exception:
                    ckpt_list = ckpt_list[:-1]

    weak_cfg = cfg.DATA_CONFIG.get('WEAK_SUPERVISION', {})
    best_metric_name = weak_cfg.get('BEST_METRIC', None)
    best_metric_state = {
        'best_metric': float('-inf'),
        'best_epoch': None,
    }
    if rank == 0 and resume_checkpoint_path is not None:
        best_metric_state = load_best_metric_state(
            [resume_checkpoint_path, ckpt_dir / 'best_model.pth'],
            expected_metric_name=best_metric_name,
            logger=logger,
        )

    ema_cfg = cfg.get('EMA_TEACHER', {})
    ema_teacher = None
    if ema_cfg.get('ENABLED', False):
        ema_teacher = create_ema_teacher(checkpoint_model).cuda()
        if resume_checkpoint_path is not None:
            load_ema_teacher_from_checkpoint(
                ema_teacher, resume_checkpoint_path,
                logger=logger if rank == 0 else None,
            )
        if rank == 0:
            logger.info(
                'EMA Teacher enabled: decay=%.6f score_thresh=%.3f '
                'allow_unmatched=%s',
                float(ema_cfg.get('DECAY', 0.999)),
                float(ema_cfg.get('SCORE_THRESH', 0.7)),
                bool(ema_cfg.get('ALLOW_UNMATCHED', False)),
            )

    model.train()
    keep_frozen_modules_eval(checkpoint_model)

    if rank == 0:
        param_count = sum([m.numel() for m in model.parameters()])
        logger.info(f'Model {cfg.MODEL.NAME} created, param count: {param_count}')

    # ------------------------------------------------------------------
    #  LR Scheduler
    # ------------------------------------------------------------------
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # ------------------------------------------------------------------
    #  Validation dataloader (rank 0 only for mid-training evaluation)
    # ------------------------------------------------------------------
    test_loader = None
    if args.eval_interval > 0 and rank == 0:
        _, test_loader = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            training=False,
            workers=args.workers,
            root_path=root_path,
            logger=logger,
        )
        logger.info('Mid-training evaluation enabled: every %d epoch(s)' % args.eval_interval)

    # ------------------------------------------------------------------
    #  Start training
    # ------------------------------------------------------------------
    logger.info('**********************Start training %s(%s)**********************'
                % (cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=rank,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        scaler=scaler,
        precision=args.precision,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        log_gpu_memory=args.log_gpu_memory,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        eval_interval=args.eval_interval,
        test_loader=test_loader,
        cfg=cfg,
        ema_teacher=ema_teacher,
        ema_cfg=ema_cfg,
        point_cloud_range=train_set.point_cloud_range,
        initial_best_metric=best_metric_state['best_metric'],
        initial_best_epoch=best_metric_state['best_epoch'],
        best_metric_name=best_metric_name,
    )

    # ------------------------------------------------------------------
    #  Final evaluation (rank 0 only)
    # ------------------------------------------------------------------
    if rank == 0:
        logger.info('**********************Start final evaluation %s(%s)**********************'
                    % (cfg.TAG, args.extra_tag))

        if test_loader is None:
            _, test_loader = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                training=False,
                workers=args.workers,
                root_path=root_path,
                logger=logger,
            )

        # Try best_model first, fall back to current model state
        best_path = ckpt_dir / 'best_model.pth'
        eval_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        if best_path.exists():
            eval_model.load_params_from_file(filename=str(best_path), logger=logger)
            logger.info('Loaded best_model.pth for final evaluation')
        elif ema_teacher is not None and bool(ema_cfg.get('EVAL_WITH_TEACHER', True)):
            eval_model = ema_teacher
            logger.info('No best_model.pth found, evaluating current EMA Teacher')
        else:
            logger.info('No best_model.pth found, evaluating current model state')

        from test import eval_one_epoch
        eval_output_dir = output_dir / 'eval' / 'final'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_one_epoch(eval_model, test_loader, cfg, args, logger, result_dir=eval_output_dir)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logger.info('**********************Done**********************')


if __name__ == '__main__':
    main()
