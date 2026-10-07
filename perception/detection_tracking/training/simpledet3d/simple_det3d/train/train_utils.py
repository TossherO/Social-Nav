import os
import glob
import time
import numpy as np
import torch
import tqdm
from collections import namedtuple
from torch.nn.utils import clip_grad_norm_

from simple_det3d.utils.precision_utils import (
    autocast_context,
    create_grad_scaler,
    normalize_precision,
)
from simple_det3d.train.ema_utils import (
    inject_teacher_targets,
    keep_frozen_modules_eval,
    update_ema_teacher,
)


def eval_epoch(model, test_loader, cfg, logger, cur_epoch, tb_log=None,
               accumulated_iter=0, precision='fp32'):
    """
    Run evaluation on the validation set during training.

    This is a lightweight evaluation: runs inference, computes metrics (AP3D,
    APBEV), and logs results to logger and TensorBoard.  No result files or
    visualizations are saved.

    Args:
        model:        model in eval mode (restored to train mode before return).
        test_loader:  DataLoader for the validation set.
        cfg:          EasyDict config.
        logger:       logging.Logger.
        cur_epoch:    current epoch (int), for display.
        tb_log:       TensorBoard SummaryWriter or None.
        accumulated_iter: global iteration, for TensorBoard x-axis alignment.

    Returns:
        eval_result:  flat dict of metric_name → float value, e.g.
                      {'Car_AP3D': 0.85, 'Pedestrian_AP3D': 0.62, ...}
    """
    from simple_det3d.models import load_data_to_gpu

    class_names = test_loader.dataset.class_names
    logger.info(f'{"="*60}')
    logger.info(f'Evaluation at epoch {cur_epoch}')
    logger.info(f'{"="*60}')

    det_annos = []
    was_training = model.training
    model.eval()

    eval_pbar = tqdm.tqdm(total=len(test_loader), leave=False,
                          desc=f'eval_epoch_{cur_epoch}', dynamic_ncols=True)
    # The bundled PointNet2 CUDA extensions are compatible with no_grad, but
    # do not fully support PyTorch inference tensors in every precision mode.
    with torch.no_grad():
        for batch_dict in test_loader:
            load_data_to_gpu(batch_dict)
            with autocast_context(precision):
                pred_dicts, _ = model(batch_dict)
            annos = test_loader.dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
            )
            det_annos += annos
            eval_pbar.update()
    eval_pbar.close()

    model.train(was_training)
    keep_frozen_modules_eval(model)

    result_str, result_dict = test_loader.dataset.evaluation(det_annos, class_names)
    logger.info(result_str)

    # Log to TensorBoard (flatten nested dict)
    if tb_log is not None:
        for key, val in result_dict.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    tb_log.add_scalar(f'eval/{key}_{sub_key}', sub_val, accumulated_iter)
            else:
                tb_log.add_scalar(f'eval/{key}', val, accumulated_iter)

    # Build a clean flat dict
    eval_result = {}
    for key, val in result_dict.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                eval_result[f'{key}_{sub_key}'] = sub_val
        else:
            eval_result[key] = val

    logger.info(f'Evaluation at epoch {cur_epoch} finished.')
    return eval_result


def _log_distributed_cuda_memory(batch, rank, logger):
    """Collect lightweight workload/memory diagnostics from every DDP rank."""
    if not torch.cuda.is_available():
        return

    points = batch.get('points')
    num_points = points.shape[0] if torch.is_tensor(points) else 0
    values = torch.tensor(
        [
            num_points,
            batch.get('vfe_num_voxels_before_limit', 0),
            batch.get('vfe_num_voxels', 0),
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
        ],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )

    if torch.distributed.is_initialized():
        gathered = [torch.empty_like(values) for _ in range(
            torch.distributed.get_world_size()
        )]
        torch.distributed.all_gather(gathered, values)
    else:
        gathered = [values]

    if rank == 0 and logger is not None:
        stats = [item.cpu().tolist() for item in gathered]
        gib = 1024 ** 3
        logger.info(
            'GPU workload by rank [points, voxels before->after]: %s',
            '  '.join(
                'r%d=%d,%d->%d' % (
                    idx, int(item[0]), int(item[1]), int(item[2])
                )
                for idx, item in enumerate(stats)
            ),
        )
        logger.info(
            'GPU memory GiB by rank [allocated/reserved, peak allocated/reserved]: %s',
            '  '.join(
                'r%d=%.1f/%.1f,%.1f/%.1f' % (
                    idx,
                    item[3] / gib,
                    item[4] / gib,
                    item[5] / gib,
                    item[6] / gib,
                )
                for idx, item in enumerate(stats)
            ),
        )

    torch.cuda.reset_peak_memory_stats()


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter,
                    optim_cfg, rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None,
                    leave_pbar=False, logger=None, logger_iter_interval=50, cur_epoch=None,
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300,
                    scaler=None, precision='fp32', ema_teacher=None, ema_cfg=None,
                    point_cloud_range=None, log_gpu_memory=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    precision = normalize_precision(precision)
    if scaler is None:
        scaler = create_grad_scaler(precision)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = AverageMeter()
        batch_time = AverageMeter()
        forward_time = AverageMeter()
        losses_m = AverageMeter()

    end = time.time()
    for cur_it in range(0, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        student_model = model.module if hasattr(model, 'module') else model
        keep_frozen_modules_eval(student_model)
        optimizer.zero_grad()

        teacher_stats = None
        if ema_teacher is not None:
            from simple_det3d.models import load_data_to_gpu

            load_data_to_gpu(batch)
            teacher_batch = dict(batch)
            ema_teacher.eval()
            with torch.no_grad(), autocast_context(precision):
                teacher_predictions, _ = ema_teacher(teacher_batch)
            teacher_stats = inject_teacher_targets(
                batch,
                teacher_predictions,
                ema_cfg,
                point_cloud_range=point_cloud_range,
            )
            # The model forward path stores the complete sparse feature pyramid
            # in teacher_batch.  It is no longer needed once pseudo targets are
            # injected and otherwise overlaps with Student forward/backward.
            if hasattr(ema_teacher, 'dense_head'):
                ema_teacher.dense_head.forward_ret_dict.clear()
            if (
                hasattr(ema_teacher, 'roi_head')
                and ema_teacher.roi_head is not None
            ):
                ema_teacher.roi_head.forward_ret_dict = None
            del teacher_predictions, teacher_batch

        with autocast_context(precision):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if ema_teacher is not None:
            update_ema_teacher(
                ema_teacher,
                student_model,
                decay=float(ema_cfg.get('DECAY', 0.999)),
            )
            if teacher_stats is not None:
                tb_dict.update(teacher_stats)

        accumulated_iter += 1

        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        should_log = (
            accumulated_iter % logger_iter_interval == 0
            or cur_it + 1 == total_it_each_epoch
        )

        if rank == 0:
            batch_size = batch.get('batch_size', None)
            data_time.update(cur_data_time)
            forward_time.update(cur_forward_time)
            batch_time.update(cur_batch_time)
            losses_m.update(loss.item(), batch_size)

            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr,
                'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            if should_log:
                logger.info(
                    'Train: {:>4d}/{} ({:>3.0f}%)  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.2f}({data_time.avg:.2f})  '
                    'Forward: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                    'Batch: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                        cur_epoch + 1, total_epochs, 100. * cur_it / total_it_each_epoch,
                        loss=losses_m, lr=cur_lr,
                        data_time=data_time, forward_time=forward_time, batch_time=batch_time
                    )
                )

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                if scaler.is_enabled():
                    tb_log.add_scalar('meta_data/grad_scale', scaler.get_scale(), accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

        if log_gpu_memory and should_log:
            _log_distributed_cuda_memory(batch, rank=rank, logger=logger)

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False,
                scaler=None, precision='fp32',
                logger=None, logger_iter_interval=None, ckpt_save_time_interval=None,
                eval_interval=0, test_loader=None, cfg=None, ema_teacher=None,
                ema_cfg=None, point_cloud_range=None,
                initial_best_metric=-np.inf, initial_best_epoch=None,
                best_metric_name=None, log_gpu_memory=False):
    """
    Train the model across multiple epochs.

    Args:
        eval_interval:  Evaluate on ``test_loader`` every N epochs.
                        0 or None disables mid-training evaluation.
        test_loader:    DataLoader for the validation set (required if
                        eval_interval > 0).
        cfg:            EasyDict config (required if eval_interval > 0).

    When ``eval_interval > 0``, after every N epochs the model is temporarily
    switched to eval mode, evaluated on the entire validation set, and the
    results are logged to both the text logger and TensorBoard (under the
    ``eval/`` prefix).  The best model (according to the mean of all returned
    metrics) is saved as ``best_model.pth`` in the checkpoint directory.
    """
    precision = normalize_precision(precision)
    if scaler is None:
        scaler = create_grad_scaler(precision)

    accumulated_iter = start_iter
    best_metric = float(initial_best_metric)
    best_epoch = initial_best_epoch
    if best_metric_name is None and cfg is not None:
        weak_cfg = cfg.DATA_CONFIG.get('WEAK_SUPERVISION', {})
        best_metric_name = weak_cfg.get('BEST_METRIC', None)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                scaler=scaler, precision=precision,
                ema_teacher=ema_teacher, ema_cfg=ema_cfg,
                point_cloud_range=point_cloud_range,
                log_gpu_memory=log_gpu_memory,
            )

            trained_epoch = cur_epoch + 1
            # --- Evaluation during training ---
            should_evaluate = (
                eval_interval and eval_interval > 0
                and cfg is not None
                and trained_epoch % eval_interval == 0
            )
            if should_evaluate and rank == 0:
                if test_loader is None:
                    raise ValueError('Rank 0 requires test_loader for evaluation')
                eval_model = (
                    ema_teacher
                    if ema_teacher is not None
                    and bool(ema_cfg.get('EVAL_WITH_TEACHER', True))
                    else (model.module if hasattr(model, 'module') else model)
                )
                eval_result = eval_epoch(
                    eval_model, test_loader, cfg, logger,
                    cur_epoch=trained_epoch,
                    tb_log=tb_log,
                    accumulated_iter=accumulated_iter,
                    precision=precision,
                )

                # Compute aggregate metric for best-model selection
                if eval_result:
                    if best_metric_name is not None:
                        if best_metric_name not in eval_result:
                            raise KeyError(
                                f'Configured BEST_METRIC={best_metric_name!r} '
                                f'not found in evaluation results'
                            )
                        current_metric = float(eval_result[best_metric_name])
                    else:
                        finite_metrics = [
                            float(value) for value in eval_result.values()
                            if np.isscalar(value) and np.isfinite(value)
                        ]
                        current_metric = np.mean(finite_metrics) if finite_metrics else -np.inf
                    logger.info(
                        'Eval epoch %d  selection_metric%s=%.4f  best=%.4f'
                        % (
                            trained_epoch,
                            f'[{best_metric_name}]' if best_metric_name else '',
                            current_metric,
                            best_metric,
                        )
                    )
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_epoch = trained_epoch
                        best_path = ckpt_save_dir / 'best_model'
                        save_checkpoint(
                            checkpoint_state(
                                eval_model, None, trained_epoch, accumulated_iter,
                                scaler=None, precision=precision,
                                ema_teacher=ema_teacher,
                                best_metric=best_metric,
                                best_epoch=best_epoch,
                                best_metric_name=best_metric_name,
                            ),
                            filename=best_path,
                        )
                        logger.info(
                            'New best model saved (selection_metric=%.4f, epoch=%d)'
                            % (best_metric, trained_epoch)
                        )

            # Save after evaluation so the epoch checkpoint contains the
            # selection state produced by this epoch.  Non-zero ranks wait at
            # the barrier below while rank 0 evaluates and writes checkpoint.
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(
                        model, optimizer, trained_epoch, accumulated_iter,
                        scaler=scaler, precision=precision,
                        ema_teacher=ema_teacher,
                        best_metric=best_metric,
                        best_epoch=best_epoch,
                        best_metric_name=best_metric_name,
                    ),
                    filename=ckpt_name,
                )

            # Rank-0-only evaluation must finish before another DDP rank
            # starts the next epoch.  ``eval_model`` above is deliberately
            # unwrapped, so its forward pass performs no DDP broadcasts.
            if should_evaluate and torch.distributed.is_initialized():
                torch.distributed.barrier()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None,
                     scaler=None, precision='fp32', ema_teacher=None,
                     best_metric=None, best_epoch=None,
                     best_metric_name=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model_state_to_cpu(model.state_dict())
    else:
        model_state = None

    scaler_state = scaler.state_dict() if scaler is not None else None
    state = {
        'epoch': epoch,
        'it': it,
        'model_state': model_state,
        'optimizer_state': optim_state,
        'scaler_state': scaler_state,
        'precision': normalize_precision(precision),
        'best_metric': float(best_metric) if best_metric is not None else None,
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_metric_name': best_metric_name,
    }
    if ema_teacher is not None:
        state['ema_model_state'] = model_state_to_cpu(ema_teacher.state_dict())
    return state


def load_best_metric_state(checkpoint_paths, expected_metric_name=None, logger=None):
    """Recover the best finite selection metric from compatible checkpoints."""

    best_state = {
        'best_metric': -np.inf,
        'best_epoch': None,
        'best_metric_name': expected_metric_name,
        'source': None,
    }
    seen_paths = set()
    existing_paths = 0
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path is None:
            continue
        checkpoint_path = os.fspath(checkpoint_path)
        if checkpoint_path in seen_paths or not os.path.isfile(checkpoint_path):
            continue
        seen_paths.add(checkpoint_path)
        existing_paths += 1
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    'Unable to read best-metric metadata from %s: %s',
                    checkpoint_path, exc,
                )
            continue

        metric = checkpoint.get('best_metric')
        if metric is None:
            continue
        try:
            metric = float(metric)
        except (TypeError, ValueError):
            if logger is not None:
                logger.warning(
                    'Ignoring invalid best_metric=%r in %s', metric, checkpoint_path,
                )
            continue
        if not np.isfinite(metric):
            continue

        stored_metric_name = checkpoint.get('best_metric_name')
        if stored_metric_name != expected_metric_name:
            if logger is not None:
                logger.warning(
                    'Ignoring best metric from %s because metric name %r does '
                    'not match current configuration %r',
                    checkpoint_path, stored_metric_name, expected_metric_name,
                )
            continue

        epoch = checkpoint.get('best_epoch')
        try:
            epoch = int(epoch) if epoch is not None else None
        except (TypeError, ValueError):
            epoch = None
        current_epoch = best_state['best_epoch']
        is_better = metric > best_state['best_metric']
        is_later_tie = (
            metric == best_state['best_metric']
            and epoch is not None
            and (current_epoch is None or epoch > current_epoch)
        )
        if is_better or is_later_tie:
            best_state.update({
                'best_metric': metric,
                'best_epoch': epoch,
                'source': checkpoint_path,
            })

    if best_state['source'] is not None and logger is not None:
        logger.info(
            'Restored best selection metric%s=%.4f from epoch %s (%s)',
            f'[{expected_metric_name}]' if expected_metric_name else '',
            best_state['best_metric'],
            best_state['best_epoch'] if best_state['best_epoch'] is not None else 'unknown',
            best_state['source'],
        )
    elif existing_paths and logger is not None:
        logger.info(
            'Resume checkpoints contain no compatible best-metric metadata; '
            'best-model tracking starts fresh (legacy checkpoint compatible).'
        )
    return best_state


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        from simple_det3d.models import load_data_to_gpu

        # Inputs remain FP32; autocast selects lower precision per operation.
        # Coordinates and other discrete tensors retain their integer dtype.
        load_data_to_gpu(batch_dict)

        ret_dict, tb_dict, disp_dict = model(batch_dict)
        loss = ret_dict['loss'].mean()
        model.module.update_global_step() if hasattr(model, 'module') else model.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
