"""EMA Teacher helpers for conservative weak-label refinement."""

from __future__ import annotations

import copy
from pathlib import Path

import torch


def create_ema_teacher(student_model):
    """Create a frozen eval-mode copy of an initialized student model."""

    teacher = copy.deepcopy(student_model)
    teacher.eval()
    # Teacher runs in eval mode but must use the same dynamic-voxel cap as the
    # training Student so both paths see the same deterministic voxel subset.
    if hasattr(teacher, 'vfe') and hasattr(
        teacher.vfe, 'max_num_voxels_mode'
    ):
        teacher.vfe.max_num_voxels_mode = 'train'
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, decay=0.999):
    """Update Teacher parameters and buffers from the DDP-unwrapped student."""

    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError('EMA decay must be in [0, 1)')

    teacher_params = dict(teacher.named_parameters())
    for name, student_param in student.named_parameters():
        teacher_params[name].mul_(decay).add_(student_param.detach(), alpha=1.0 - decay)

    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        if name not in teacher_buffers:
            continue
        teacher_buffer = teacher_buffers[name]
        if teacher_buffer.is_floating_point():
            teacher_buffer.mul_(decay).add_(student_buffer.detach(), alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer.detach())
    teacher.eval()


def load_ema_teacher_from_checkpoint(teacher, checkpoint_path, logger=None):
    """Restore an EMA state when resuming a stage-2/3 checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    ema_state = checkpoint.get('ema_model_state')
    if ema_state is None:
        if logger is not None:
            logger.info(
                'Checkpoint %s has no EMA state; initializing Teacher from student',
                checkpoint_path,
            )
        return False

    if hasattr(teacher, '_load_state_dict'):
        teacher._load_state_dict(ema_state, strict=True)
    else:
        teacher.load_state_dict(ema_state, strict=True)
    teacher.eval()
    if logger is not None:
        logger.info('Restored EMA Teacher from %s', checkpoint_path)
    return True


def freeze_modules_by_name(model, module_names, train_mode_module_names=()):
    """Freeze exact submodules and configure their runtime training mode.

    Some container modules (notably ``VoxelRCNNHead``) branch on
    ``self.training`` to choose target assignment versus inference.  Such a
    root must remain in training mode even while all of its parameterized
    children (BatchNorm/Dropout included) remain in eval mode.
    """

    requested = tuple(str(name) for name in module_names)
    if not requested:
        model._frozen_module_names = ()
        model._frozen_train_mode_names = ()
        return {'modules': 0, 'parameters': 0}

    named_modules = dict(model.named_modules())
    missing = [name for name in requested if name not in named_modules]
    if missing:
        raise KeyError(f'Unknown module names requested for freezing: {missing}')
    train_mode_names = tuple(str(name) for name in train_mode_module_names)
    invalid_train_mode = [name for name in train_mode_names if name not in requested]
    if invalid_train_mode:
        raise ValueError(
            'train_mode_module_names must also appear in module_names: '
            f'{invalid_train_mode}'
        )

    frozen_parameter_ids = set()
    for name in requested:
        module = named_modules[name]
        module.requires_grad_(False)
        frozen_parameter_ids.update(id(parameter) for parameter in module.parameters())

    model._frozen_module_names = requested
    model._frozen_train_mode_names = train_mode_names
    keep_frozen_modules_eval(model)
    return {
        'modules': len(requested),
        'parameters': sum(
            parameter.numel()
            for parameter in model.parameters()
            if id(parameter) in frozen_parameter_ids
        ),
    }


def keep_frozen_modules_eval(model):
    """Re-apply eval mode after a global ``model.train()`` call."""

    names = getattr(model, '_frozen_module_names', ())
    if not names:
        return
    named_modules = dict(model.named_modules())
    train_mode_names = set(getattr(model, '_frozen_train_mode_names', ()))
    for name in names:
        module = named_modules[name]
        if name in train_mode_names:
            if model.training:
                module.training = True
                for child in module.children():
                    child.eval()
            else:
                module.eval()
        else:
            module.eval()


def _required_points(distance, cfg):
    if distance <= float(cfg.get('NEAR_DISTANCE', 8.0)):
        return int(cfg.get('NEAR_MIN_POINTS', 5))
    if distance <= float(cfg.get('FAR_DISTANCE', 15.0)):
        return int(cfg.get('MID_MIN_POINTS', 3))
    return int(cfg.get('FAR_MIN_POINTS', 1))


def _box_geometry_mask(boxes, point_cloud_range, cfg):
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    finite = torch.isfinite(boxes[:, :7]).all(dim=1)
    horizontal_min = float(cfg.get('MIN_HORIZONTAL_SIZE', 0.15))
    horizontal_max = float(cfg.get('MAX_HORIZONTAL_SIZE', 1.5))
    min_height = float(cfg.get('MIN_HEIGHT', 0.6))
    max_height = float(cfg.get('MAX_HEIGHT', 2.4))
    size_valid = (
        (boxes[:, 3] >= horizontal_min)
        & (boxes[:, 3] <= horizontal_max)
        & (boxes[:, 4] >= horizontal_min)
        & (boxes[:, 4] <= horizontal_max)
        & (boxes[:, 5] >= min_height)
        & (boxes[:, 5] <= max_height)
    )
    pc_range = boxes.new_tensor(point_cloud_range)
    range_valid = (
        (boxes[:, :3] >= pc_range[:3]).all(dim=1)
        & (boxes[:, :3] <= pc_range[3:]).all(dim=1)
    )
    return finite & size_valid & range_valid


def _count_points_in_box(points_xyz, box):
    if len(points_xyz) == 0:
        return 0
    rel = points_xyz - box[None, :3]
    cos_yaw = torch.cos(box[6])
    sin_yaw = torch.sin(box[6])
    local_x = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
    local_y = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw
    inside = (
        (local_x.abs() <= box[3] * 0.5)
        & (local_y.abs() <= box[4] * 0.5)
        & (rel[:, 2].abs() <= box[5] * 0.5)
    )
    return int(inside.sum().item())


@torch.no_grad()
def inject_teacher_targets(batch_dict, pred_dicts, cfg, point_cloud_range):
    """Upgrade target-domain center labels with conservative Teacher boxes.

    Only target-domain samples are changed.  Target-domain identity is kept
    separate from center-label exhaustiveness: manually completed Wheelchair
    frames may safely supervise background while still receiving conservative
    Teacher geometry upgrades.  By default unmatched Teacher boxes are not
    added, preventing unconstrained self-confirmation.
    """

    required_keys = (
        'gt_boxes', 'gt_cls_weights', 'gt_reg_weights',
        'gt_box_reg_mask', 'gt_supervision_types', 'label_is_exhaustive',
    )
    missing = [key for key in required_keys if key not in batch_dict]
    if missing:
        raise KeyError(f'EMA target injection requires batch keys: {missing}')

    gt_boxes = batch_dict['gt_boxes']
    cls_weights = batch_dict['gt_cls_weights']
    reg_weights = batch_dict['gt_reg_weights']
    code_masks = batch_dict['gt_box_reg_mask']
    supervision_types = batch_dict['gt_supervision_types']
    batch_size = int(batch_dict['batch_size'])
    if len(pred_dicts) != batch_size:
        raise ValueError('Teacher prediction count does not match batch size')

    match_distance = float(cfg.get('MATCH_DISTANCE', 0.5))
    score_threshold = float(cfg.get('SCORE_THRESH', 0.7))
    unmatched_threshold = float(cfg.get('UNMATCHED_SCORE_THRESH', 0.9))
    allow_unmatched = bool(cfg.get('ALLOW_UNMATCHED', False))
    teacher_cls_weight = float(cfg.get('CLS_WEIGHT', 0.5))
    teacher_box_weight = float(cfg.get('BOX_WEIGHT', 0.3))

    point_tensor = batch_dict.get('points')
    per_sample = []
    stats = {'teacher_candidates': 0, 'teacher_upgraded': 0, 'teacher_added': 0}
    for batch_idx in range(batch_size):
        valid_gt = gt_boxes[batch_idx, :, -1] > 0
        cur_boxes = gt_boxes[batch_idx, valid_gt].clone()
        cur_cls_weights = cls_weights[batch_idx, valid_gt].clone()
        cur_reg_weights = reg_weights[batch_idx, valid_gt].clone()
        cur_code_masks = code_masks[batch_idx, valid_gt].clone()
        cur_supervision = supervision_types[batch_idx, valid_gt].clone()

        domain_flags = batch_dict.get('is_target_domain')
        if domain_flags is None:
            # Backward compatibility for checkpoints/tests produced before
            # domain identity was made explicit.
            is_target = batch_dict['label_is_exhaustive'][batch_idx].item() < 0.5
        else:
            is_target = domain_flags[batch_idx].item() > 0.5
        if not is_target:
            per_sample.append((
                cur_boxes, cur_cls_weights, cur_reg_weights,
                cur_code_masks, cur_supervision,
            ))
            continue

        pred = pred_dicts[batch_idx]
        pred_boxes = pred['pred_boxes'][:, :7].detach()
        pred_scores = pred['pred_scores'].detach()
        pred_labels = pred['pred_labels'].detach()
        candidate_mask = (
            (pred_labels == 1)
            & (pred_scores >= score_threshold)
            & _box_geometry_mask(pred_boxes, point_cloud_range, cfg)
        )
        candidate_indices = candidate_mask.nonzero(as_tuple=False).view(-1)
        if len(candidate_indices):
            order = torch.argsort(pred_scores[candidate_indices], descending=True)
            candidate_indices = candidate_indices[order]
        stats['teacher_candidates'] += int(len(candidate_indices))

        if point_tensor is None:
            sample_points = pred_boxes.new_empty((0, 3))
        else:
            sample_mask = point_tensor[:, 0].long() == batch_idx
            # Count current-frame returns only.  History sweeps can otherwise
            # inflate support for a ghosted or stale prediction.
            if point_tensor.shape[1] >= 6:
                sample_mask &= point_tensor[:, -1].abs() < 1e-6
            sample_points = point_tensor[sample_mask, 1:4]

        for pred_idx in candidate_indices.tolist():
            teacher_box = pred_boxes[pred_idx]
            score = float(pred_scores[pred_idx].item())
            distance = float(torch.linalg.norm(teacher_box[:2]).item())
            point_count = _count_points_in_box(sample_points, teacher_box)
            if point_count < _required_points(distance, cfg):
                continue

            nearest_idx = None
            nearest_distance = float('inf')
            if len(cur_boxes):
                distances = torch.linalg.norm(cur_boxes[:, :2] - teacher_box[None, :2], dim=1)
                nearest_distance_tensor, nearest_idx_tensor = torch.min(distances, dim=0)
                nearest_distance = float(nearest_distance_tensor.item())
                nearest_idx = int(nearest_idx_tensor.item())

            if nearest_idx is not None and nearest_distance <= match_distance:
                if int(cur_supervision[nearest_idx].item()) == 1:
                    # Preserve the human/image center, and borrow only geometry
                    # from the Teacher.  Fractional code-mask values reduce the
                    # uncertain z/size gradients independently of xy.
                    cur_boxes[nearest_idx, 2:7] = teacher_box[2:7]
                    center_weight = torch.clamp(
                        cur_reg_weights[nearest_idx], min=0.0, max=1.0
                    )
                    cur_code_masks[nearest_idx, :2] = center_weight
                    cur_code_masks[nearest_idx, 2:6] = teacher_box_weight
                    cur_code_masks[nearest_idx, 6:8] = 0.0
                    cur_reg_weights[nearest_idx] = 1.0
                    cur_cls_weights[nearest_idx] = torch.maximum(
                        cur_cls_weights[nearest_idx],
                        cur_cls_weights.new_tensor(score * teacher_cls_weight),
                    )
                    cur_supervision[nearest_idx] = 2
                    stats['teacher_upgraded'] += 1
                continue

            if not allow_unmatched or score < unmatched_threshold:
                continue
            new_box = torch.cat([teacher_box, teacher_box.new_ones(1)])
            cur_boxes = torch.cat([cur_boxes, new_box[None, :]], dim=0)
            cur_cls_weights = torch.cat([
                cur_cls_weights,
                cur_cls_weights.new_tensor([score * teacher_cls_weight]),
            ])
            cur_reg_weights = torch.cat([
                cur_reg_weights, cur_reg_weights.new_ones(1),
            ])
            new_code_mask = cur_code_masks.new_tensor([
                1.0, 1.0,
                teacher_box_weight, teacher_box_weight,
                teacher_box_weight, teacher_box_weight,
                0.0, 0.0,
            ])
            cur_code_masks = torch.cat([cur_code_masks, new_code_mask[None, :]], dim=0)
            cur_supervision = torch.cat([
                cur_supervision, cur_supervision.new_tensor([2]),
            ])
            stats['teacher_added'] += 1

        per_sample.append((
            cur_boxes, cur_cls_weights, cur_reg_weights,
            cur_code_masks, cur_supervision,
        ))

    max_gt = max(1, max(len(sample[0]) for sample in per_sample))
    batch_dict['gt_boxes'] = gt_boxes.new_zeros((batch_size, max_gt, gt_boxes.shape[-1]))
    batch_dict['gt_cls_weights'] = cls_weights.new_zeros((batch_size, max_gt))
    batch_dict['gt_reg_weights'] = reg_weights.new_zeros((batch_size, max_gt))
    batch_dict['gt_box_reg_mask'] = code_masks.new_zeros(
        (batch_size, max_gt, code_masks.shape[-1])
    )
    batch_dict['gt_supervision_types'] = supervision_types.new_zeros((batch_size, max_gt))
    for batch_idx, sample in enumerate(per_sample):
        num_gt = len(sample[0])
        batch_dict['gt_boxes'][batch_idx, :num_gt] = sample[0]
        batch_dict['gt_cls_weights'][batch_idx, :num_gt] = sample[1]
        batch_dict['gt_reg_weights'][batch_idx, :num_gt] = sample[2]
        batch_dict['gt_box_reg_mask'][batch_idx, :num_gt] = sample[3]
        batch_dict['gt_supervision_types'][batch_idx, :num_gt] = sample[4]
    return stats
