from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch
from easydict import EasyDict

from simple_det3d.datasets.coda_dataset import CodaDataset
from simple_det3d.datasets.mixed_domain_dataset import MultiDomainDataset
from simple_det3d.datasets.weak_label_utils import (
    PedestrianBoxFilterConfig,
    SUPERVISION_BOX,
    SUPERVISION_CENTER,
    classify_pseudo_box,
    count_points_in_oriented_box,
    temporal_support_counts,
    validate_box_geometry,
)
from simple_det3d.models.center_head import CenterHead
from simple_det3d.models.voxel_rcnn_head import ProposalTargetLayer
from simple_det3d.train.ema_utils import (
    freeze_modules_by_name,
    inject_teacher_targets,
    keep_frozen_modules_eval,
    update_ema_teacher,
)
from simple_det3d.utils.config import load_config
from simple_det3d.utils.loss_utils import _reg_loss
from simple_det3d.utils.loss_utils import neg_loss_cornernet
from tools.prepare_wheelchair_weak_data import (
    load_manual_centers,
    load_pseudo_scenes,
    make_weak_targets,
)


def test_manual_annotation_key_is_scene_frame_list_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pseudo_path = root / 'scenes.pkl'
        scene_entries = [{
            'scene_dir': 'indoor2',
            'scene': 'sequence_sparse',
            'frames': [
                {'frame_index': 45},
                {'frame_index': 117},
                {'frame_index': 550},
            ],
        }]
        with pseudo_path.open('wb') as f:
            pickle.dump(scene_entries, f)
        with (root / 'annotations_indoor2_sequence_sparse.pkl').open('wb') as f:
            pickle.dump({
                0: [[1.0, 2.0, 0.5, 'Pedestrian']],
                2: [[3.0, 4.0, 0.5, 'Pedestrian']],
            }, f)

        pseudo_by_frame, frame_indices_by_scene = load_pseudo_scenes(pseudo_path)
        manual_by_frame, manual_scenes = load_manual_centers(
            root, frame_indices_by_scene
        )

        assert ('indoor2', 'sequence_sparse', 45) in pseudo_by_frame
        assert manual_scenes == {('indoor2', 'sequence_sparse')}
        assert np.allclose(
            manual_by_frame[('indoor2', 'sequence_sparse', 45)], [[1.0, 2.0]]
        )
        assert np.allclose(
            manual_by_frame[('indoor2', 'sequence_sparse', 550)], [[3.0, 4.0]]
        )
        assert ('indoor2', 'sequence_sparse', 1) not in manual_by_frame

        _, _, manual_frame_keys = load_manual_centers(
            root, frame_indices_by_scene, return_frame_keys=True
        )
        assert manual_frame_keys == {
            ('indoor2', 'sequence_sparse', 45),
            ('indoor2', 'sequence_sparse', 550),
        }


def test_manual_annotation_frame_list_index_must_be_in_range():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        with (root / 'annotations_indoor1_sequence_short.pkl').open('wb') as f:
            pickle.dump({1: [[1.0, 2.0, 0.5, 'Pedestrian']]}, f)

        with np.testing.assert_raises_regex(ValueError, 'outside'):
            load_manual_centers(
                root, {('indoor1', 'sequence_short'): (73,)}
            )


def test_box_geometry_and_oriented_point_count():
    cfg = PedestrianBoxFilterConfig()
    box = np.asarray([2.0, 0.0, 1.0, 0.8, 0.6, 1.8, np.pi / 2])
    valid, reason = validate_box_geometry(box, cfg, [-5, -5, -2, 10, 5, 4])
    assert valid and reason == 'ok'

    points = np.asarray([
        [2.0, 0.0, 1.0, 1.0],
        [2.0, 0.3, 1.0, 1.0],
        [2.0, -0.3, 1.0, 1.0],
        [2.2, 0.0, 1.0, 1.0],
        [4.0, 0.0, 1.0, 1.0],
    ])
    assert count_points_in_oriented_box(points, box) == 4


def test_invalid_box_is_downgraded_to_center():
    cfg = PedestrianBoxFilterConfig()
    invalid_box = np.asarray([2.0, 0.0, 1.0, 0.0, 0.6, 1.8, 0.0])
    supervision, diagnostics = classify_pseudo_box(
        invalid_box,
        np.empty((0, 4)),
        temporal_support=2,
        cfg=cfg,
    )
    assert supervision == SUPERVISION_CENTER
    assert diagnostics['reason'] == 'non_positive_size'


def test_supported_box_passes_conservative_filter():
    cfg = PedestrianBoxFilterConfig(near_min_points=3)
    box = np.asarray([2.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0])
    points = np.asarray([
        [2.0, 0.0, 1.0, 1.0],
        [2.1, 0.0, 1.0, 1.0],
        [1.9, 0.0, 1.0, 1.0],
    ])
    supervision, diagnostics = classify_pseudo_box(
        box,
        points,
        temporal_support=2,
        cfg=cfg,
        pseudo_score=0.8,
    )
    assert supervision == SUPERVISION_BOX
    assert diagnostics['reason'] == 'box_supported'


def test_bidirectional_box_needs_minimum_image_score():
    cfg = PedestrianBoxFilterConfig(near_min_points=1, min_bidirectional_box_score=0.6)
    box = np.asarray([2.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0])
    supervision, diagnostics = classify_pseudo_box(
        box,
        np.asarray([[2.0, 0.0, 1.0, 1.0]]),
        temporal_support=2,
        cfg=cfg,
        pseudo_score=0.59,
    )
    assert supervision == SUPERVISION_CENTER
    assert diagnostics['reason'] == 'low_box_score'


def test_manual_center_and_nearby_pseudo_box_remain_complementary_targets():
    pseudo_frame = {'pseudo_annos': {
        'name': np.asarray(['Pedestrian']),
        'center_3d': np.asarray([[2.0, 0.0, 1.0]], dtype=np.float32),
        'bbox_3d': np.asarray([[2.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0]], dtype=np.float32),
        'score': np.asarray([0.8], dtype=np.float32),
    }}
    result = make_weak_targets(
        pseudo_frame=pseudo_frame,
        # Deliberately closer than the pseudo-only dedup threshold.  Annotation
        # protocol says this is a different target, not duplicate evidence.
        manual_centers=np.asarray([[2.1, 0.0]], dtype=np.float32),
        temporal_support={0: 2},
        points=np.asarray([[2.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        cfg=PedestrianBoxFilterConfig(near_min_points=1),
        point_cloud_range=[-5, -5, -2, 10, 5, 4],
        pseudo_center_weight=0.2,
        pseudo_box_weight=0.3,
        manual_center_weight=1.0,
    )
    assert result['gt_supervision_types'].tolist() == [SUPERVISION_CENTER, SUPERVISION_BOX]
    assert np.allclose(result['gt_boxes'][0, :2], [2.1, 0.0])
    assert np.allclose(result['gt_boxes'][1, :2], [2.0, 0.0])
    assert result['gt_box_reg_mask'][0].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert result['gt_box_reg_mask'][1].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    assert np.allclose(result['gt_cls_weights'], [1.0, 0.3])
    assert result['frame_stats']['manual_added'] == 1
    assert 'manual_matched' not in result['frame_stats']


def test_exhaustive_frame_raises_only_pseudo_classification_confidence():
    pseudo_frame = {'pseudo_annos': {
        'name': np.asarray(['Pedestrian']),
        'center_3d': np.asarray([[2.0, 0.0, 1.0]], dtype=np.float32),
        'bbox_3d': np.asarray([[2.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0]], dtype=np.float32),
        'score': np.asarray([0.7], dtype=np.float32),
    }}
    result = make_weak_targets(
        pseudo_frame=pseudo_frame,
        manual_centers=np.empty((0, 2), dtype=np.float32),
        temporal_support={0: 2},
        points=np.asarray([[2.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        cfg=PedestrianBoxFilterConfig(near_min_points=1),
        point_cloud_range=[-5, -5, -2, 10, 5, 4],
        pseudo_center_weight=0.2,
        pseudo_box_weight=0.3,
        manual_center_weight=1.0,
        label_is_exhaustive=True,
        exhaustive_pseudo_cls_min=0.5,
    )
    assert np.allclose(result['gt_cls_weights'], [0.7])
    assert np.allclose(result['gt_reg_weights'], [0.3])
    assert result['gt_box_reg_mask'][0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]


def test_temporal_support_uses_both_sides():
    cfg = PedestrianBoxFilterConfig(max_pedestrian_speed=1.0)
    centers = {
        1: np.asarray([[1.0, 0.0, 0.0]]),
        2: np.asarray([[1.05, 0.0, 0.0]]),
        3: np.asarray([[1.10, 0.0, 0.0]]),
    }
    poses = {idx: np.eye(4) for idx in centers}
    support = temporal_support_counts(centers, poses, 10.0, cfg, max_frame_gap=1)
    assert support[(2, 0)] == 2
    assert support[(1, 0)] == 1


def test_regression_loss_respects_code_mask_and_weight():
    prediction = torch.zeros((1, 1, 2), dtype=torch.float32)
    target = torch.ones((1, 1, 2), dtype=torch.float32)
    object_mask = torch.ones((1, 1), dtype=torch.float32)
    code_mask = torch.tensor([[[1.0, 0.0]]])
    weights = torch.tensor([[0.25]])
    loss = _reg_loss(
        prediction, target, object_mask, code_mask=code_mask, weights=weights
    )
    assert torch.allclose(loss, torch.tensor([0.25, 0.0]))

    fractional_code_mask = torch.tensor([[[0.3, 0.0]]])
    fractional_loss = _reg_loss(
        prediction, target, object_mask, code_mask=fractional_code_mask
    )
    assert torch.allclose(fractional_loss, torch.tensor([0.3, 0.0]))


def test_focal_mask_confidence_is_not_cancelled_by_normalization():
    prediction = torch.tensor([[[[0.8]]]], dtype=torch.float32)
    target = torch.ones_like(prediction)
    full_loss = neg_loss_cornernet(
        prediction, target, mask=torch.ones((1, 1, 1))
    )
    weak_loss = neg_loss_cornernet(
        prediction, target, mask=torch.full((1, 1, 1), 0.2)
    )
    assert torch.allclose(weak_loss, full_loss * 0.2)


def test_collate_pads_weak_supervision_fields():
    empty = {
        'gt_boxes': np.empty((0, 8), dtype=np.float32),
        'gt_box_reg_mask': np.empty((0, 8), dtype=np.float32),
        'gt_cls_weights': np.empty(0, dtype=np.float32),
        'gt_reg_weights': np.empty(0, dtype=np.float32),
        'gt_supervision_types': np.empty(0, dtype=np.int32),
    }
    populated = {
        'gt_boxes': np.ones((1, 8), dtype=np.float32),
        'gt_box_reg_mask': np.ones((1, 8), dtype=np.float32),
        'gt_cls_weights': np.asarray([0.5], dtype=np.float32),
        'gt_reg_weights': np.asarray([0.25], dtype=np.float32),
        'gt_supervision_types': np.asarray([2], dtype=np.int32),
    }
    batch = CodaDataset.collate_batch([empty, populated])
    assert batch['gt_boxes'].shape == (2, 1, 8)
    assert batch['gt_box_reg_mask'].shape == (2, 1, 8)
    assert batch['gt_cls_weights'].tolist() == [[0.0], [0.5]]
    assert batch['gt_reg_weights'].tolist() == [[0.0], [0.25]]


def test_recursive_config_inheritance_for_all_three_stages():
    project_root = Path(__file__).resolve().parents[1]
    stage3 = load_config(project_root / 'configs/voxelrcnn_wheelchair_weak_stage3.yaml')
    assert stage3.MODEL.NAME == 'VoxelRCNN'
    assert stage3.CLASS_NAMES == ['Pedestrian']
    assert stage3.MIXED_DOMAIN.ENABLED is False
    assert stage3.DATA_CONFIG.POINT_CLOUD_RANGE == [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5]


def test_center_head_builds_unknown_mask_and_partial_regression_target():
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.point_cloud_range = np.asarray([-4.8, -9.6, -1.5, 14.4, 9.6, 4.5])
    head.voxel_size = np.asarray([0.05, 0.05, 0.2])
    gt_boxes = torch.tensor([[2.0, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0, 1.0]])
    outputs = head.assign_target_of_single_head(
        num_classes=1,
        gt_boxes=gt_boxes,
        feature_map_size=np.asarray([48, 48]),
        feature_map_stride=8,
        num_max_objs=4,
        gt_cls_weights=torch.tensor([0.2]),
        gt_reg_weights=torch.tensor([0.3]),
        gt_box_reg_mask=torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]]),
        label_is_exhaustive=False,
    )
    heatmap, _, _, object_mask, heatmap_mask, reg_weights, reg_code_mask, _ = outputs
    assert heatmap.max().item() == 1.0
    assert 0 < heatmap_mask.max().item() <= 0.2 + 1e-6
    assert heatmap_mask.min().item() == 0.0
    assert object_mask.tolist() == [1, 0, 0, 0]
    assert torch.allclose(reg_weights, torch.tensor([0.3, 0.0, 0.0, 0.0]))
    assert reg_code_mask[0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_center_head_separates_exhaustive_background_and_positive_weights():
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.point_cloud_range = np.asarray([-4.8, -9.6, -1.5, 14.4, 9.6, 4.5])
    head.voxel_size = np.asarray([0.05, 0.05, 0.2])
    gt_boxes = torch.tensor([[2.0, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0, 1.0]])
    outputs = head.assign_target_of_single_head(
        num_classes=1,
        gt_boxes=gt_boxes,
        feature_map_size=np.asarray([48, 48]),
        feature_map_stride=8,
        num_max_objs=4,
        gt_cls_weights=torch.tensor([0.7]),
        label_is_exhaustive=True,
        heatmap_background_weight=0.25,
    )
    heatmap_mask = outputs[4]
    assert torch.isclose(heatmap_mask.min(), torch.tensor(0.25))
    assert torch.isclose(heatmap_mask.max(), torch.tensor(0.7))


def test_ema_teacher_upgrades_matching_center_but_not_roi_state():
    batch = {
        'batch_size': 1,
        'label_is_exhaustive': torch.tensor([0.0]),
        'gt_boxes': torch.tensor([[[2.0, 0.0, 0.0, 0.8, 0.6, 1.73, 0.0, 1.0]]]),
        'gt_cls_weights': torch.tensor([[1.0]]),
        'gt_reg_weights': torch.tensor([[1.0]]),
        'gt_box_reg_mask': torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        'gt_supervision_types': torch.tensor([[1.0]]),
        'points': torch.tensor([
            [0.0, 2.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 2.1, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.9, 0.0, 1.0, 1.0, 0.0],
            [0.0, 2.0, 0.1, 1.0, 1.0, 0.0],
            [0.0, 2.0, -0.1, 1.0, 1.0, 0.0],
        ]),
    }
    predictions = [{
        'pred_boxes': torch.tensor([[2.1, 0.0, 1.0, 0.8, 0.6, 1.8, 0.2]]),
        'pred_scores': torch.tensor([0.8]),
        'pred_labels': torch.tensor([1]),
    }]
    stats = inject_teacher_targets(
        batch,
        predictions,
        {'SCORE_THRESH': 0.7, 'MATCH_DISTANCE': 0.5, 'BOX_WEIGHT': 0.3},
        [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5],
    )
    assert stats['teacher_upgraded'] == 1
    assert torch.allclose(batch['gt_boxes'][0, 0, :2], torch.tensor([2.0, 0.0]))
    assert torch.allclose(batch['gt_boxes'][0, 0, 2:6], predictions[0]['pred_boxes'][0, 2:6])
    assert torch.allclose(
        batch['gt_box_reg_mask'][0, 0],
        torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.0, 0.0]),
    )


def test_ema_target_identity_is_independent_of_exhaustive_labels():
    batch = {
        'batch_size': 1,
        'label_is_exhaustive': torch.tensor([1.0]),
        'is_target_domain': torch.tensor([1.0]),
        'gt_boxes': torch.tensor([[[2.0, 0.0, 0.0, 0.8, 0.6, 1.73, 0.0, 1.0]]]),
        'gt_cls_weights': torch.tensor([[1.0]]),
        'gt_reg_weights': torch.tensor([[1.0]]),
        'gt_box_reg_mask': torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        'gt_supervision_types': torch.tensor([[1.0]]),
        'points': torch.tensor([
            [0.0, 2.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 2.1, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.9, 0.0, 1.0, 1.0, 0.0],
            [0.0, 2.0, 0.1, 1.0, 1.0, 0.0],
            [0.0, 2.0, -0.1, 1.0, 1.0, 0.0],
        ]),
    }
    predictions = [{
        'pred_boxes': torch.tensor([[2.1, 0.0, 1.0, 0.8, 0.6, 1.8, 0.2]]),
        'pred_scores': torch.tensor([0.8]),
        'pred_labels': torch.tensor([1]),
    }]
    stats = inject_teacher_targets(
        batch, predictions,
        {'SCORE_THRESH': 0.7, 'MATCH_DISTANCE': 0.5, 'BOX_WEIGHT': 0.3},
        [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5],
    )
    assert stats['teacher_upgraded'] == 1


def test_ema_parameter_update():
    student = torch.nn.Linear(2, 1, bias=False)
    teacher = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.fill_(0.0)
    update_ema_teacher(teacher, student, decay=0.75)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))


def test_ema_point_filter_ignores_history_sweep_points():
    batch = {
        'batch_size': 1,
        'label_is_exhaustive': torch.tensor([0.0]),
        'gt_boxes': torch.tensor([[[2.0, 0.0, 0.0, 0.8, 0.6, 1.73, 0.0, 1.0]]]),
        'gt_cls_weights': torch.tensor([[0.2]]),
        'gt_reg_weights': torch.tensor([[0.2]]),
        'gt_box_reg_mask': torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        'gt_supervision_types': torch.tensor([[1.0]]),
        # [batch, x, y, z, intensity, timestamp]: only one current point.
        'points': torch.tensor([
            [0.0, 2.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 2.1, 0.0, 1.0, 1.0, 0.1],
            [0.0, 1.9, 0.0, 1.0, 1.0, 0.1],
            [0.0, 2.0, 0.1, 1.0, 1.0, 0.2],
            [0.0, 2.0, -0.1, 1.0, 1.0, 0.3],
        ]),
    }
    predictions = [{
        'pred_boxes': torch.tensor([[2.1, 0.0, 1.0, 0.8, 0.6, 1.8, 0.2]]),
        'pred_scores': torch.tensor([0.8]),
        'pred_labels': torch.tensor([1]),
    }]
    stats = inject_teacher_targets(
        batch, predictions, {'SCORE_THRESH': 0.7},
        [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5],
    )
    assert stats['teacher_upgraded'] == 0
    assert batch['gt_supervision_types'][0, 0].item() == SUPERVISION_CENTER


def test_frozen_modules_remain_in_eval_mode():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1)),
    )
    stats = freeze_modules_by_name(model, ['1'])
    assert stats['modules'] == 1
    assert stats['parameters'] == sum(p.numel() for p in model[1].parameters())
    assert not any(p.requires_grad for p in model[1].parameters())

    model.train()
    assert model[1].training
    keep_frozen_modules_eval(model)
    assert not model[1].training
    assert not model[1][0].training

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Dropout()),
    )
    freeze_modules_by_name(model, ['1'], train_mode_module_names=['1'])
    model.train()
    keep_frozen_modules_eval(model)
    assert model[1].training
    assert not model[1][0].training
    assert not model[1][1].training
    model.eval()
    keep_frozen_modules_eval(model)
    assert not model[1].training


def test_center_only_evaluation_reports_recall_not_ap():
    dataset = CodaDataset.__new__(CodaDataset)
    dataset.dataset_cfg = {
        'WEAK_SUPERVISION': {
            'CENTER_EVAL': True,
            'CENTER_HIT_THRESHOLDS': [0.5, 1.0],
            'CENTER_SCORE_THRESH': 0.1,
        }
    }
    dataset.point_cloud_range = np.asarray([-4.8, -9.6, -1.5, 14.4, 9.6, 4.5])
    dataset.infos = [{
        'token': 'indoor1/scene/0001',
        'lidar_path': 'indoor1/scene/points/0001.bin',
        'weak_centers': np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
    }]
    result_text, metrics = dataset.evaluation([{
        'frame_id': 'indoor1/scene/0001',
        'name': np.asarray(['Pedestrian']),
        'score': np.asarray([0.9]),
        'boxes_lidar': np.asarray([[2.2, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0]]),
    }], ['Pedestrian'])
    assert 'precision/AP unavailable' in result_text
    assert metrics['CenterHit@0.50m'] == 0.5
    assert metrics['CenterHit@1.00m'] == 0.5
    assert 'AP3D' not in metrics


def test_exhaustive_center_evaluation_reports_precision_recall_and_f1():
    dataset = CodaDataset.__new__(CodaDataset)
    dataset.dataset_cfg = {
        'WEAK_SUPERVISION': {
            'CENTER_EVAL': True,
            'CENTER_HIT_THRESHOLDS': [0.5],
            'CENTER_SCORE_THRESH': 0.3,
        }
    }
    dataset.point_cloud_range = np.asarray([-4.8, -9.6, -1.5, 14.4, 9.6, 4.5])
    dataset.infos = [{
        'token': 'indoor1/scene/0001',
        'lidar_path': 'indoor1/scene/points/0001.bin',
        'center_label_is_exhaustive': True,
        'weak_centers': np.asarray([[2.0, 0.0]], dtype=np.float32),
        'gt_boxes': np.asarray([
            [2.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0],
            [4.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0],
        ], dtype=np.float32),
    }]
    text, metrics = dataset.evaluation([{
        'frame_id': 'indoor1/scene/0001',
        'name': np.asarray(['Pedestrian', 'Pedestrian']),
        'score': np.asarray([0.9, 0.8]),
        'boxes_lidar': np.asarray([
            [2.1, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0],
            [8.0, 0.0, 1.0, 0.8, 0.6, 1.8, 0.0],
        ]),
    }], ['Pedestrian'])
    assert 'exhaustive-center' in text
    assert metrics['CenterPrecision@0.50m'] == 0.5
    assert metrics['CenterRecall@0.50m'] == 0.5
    assert metrics['CenterF1@0.50m'] == 0.5
    assert metrics['CenterFalsePositivesPerFrame'] == 1.0


def test_roi_center_labels_are_one_to_one_and_keep_ambiguity_ignored():
    layer = ProposalTargetLayer({
        'ROI_PER_IMAGE': 4,
        'FG_RATIO': 0.5,
        'CENTER_POS_DISTANCE': 0.5,
        'CENTER_NEG_DISTANCE': 1.0,
    })
    distances = torch.tensor([0.1, 0.2, 0.7, 1.2, 0.1])
    assignments = torch.tensor([0, 0, 0, 0, 1])
    labels = layer.get_center_cls_labels(distances, assignments, num_gt=2)
    assert labels.tolist() == [1.0, -1.0, -1.0, 0.0, 1.0]


def test_roi_center_supervision_trains_classification_but_not_regression():
    layer = ProposalTargetLayer(EasyDict({
        'ROI_PER_IMAGE': 4,
        'FG_RATIO': 0.5,
        'CENTER_POS_DISTANCE': 0.5,
        'CENTER_NEG_DISTANCE': 1.0,
        'CLS_SCORE_TYPE': 'roi_iou',
        'CLS_BG_THRESH': 0.25,
        'CLS_FG_THRESH': 0.75,
        'REG_FG_THRESH': 0.55,
        'SAMPLE_ROI_BY_EACH_CLASS': False,
    }))
    batch = {
        'batch_size': 1,
        'rois': torch.tensor([[[
            2.1, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0
        ], [
            4.1, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0
        ], [
            8.0, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0
        ], [
            9.0, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0
        ]]]),
        'roi_scores': torch.ones((1, 4)),
        'roi_labels': torch.ones((1, 4), dtype=torch.long),
        'gt_boxes': torch.tensor([[[
            2.0, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0, 1.0
        ], [
            4.0, 0.0, 1.0, 0.8, 0.6, 1.7, 0.0, 1.0
        ]]]),
        'roi_supervision_enabled': torch.tensor([0.0]),
        'roi_center_supervision_enabled': torch.tensor([1.0]),
    }
    targets = layer(batch)
    assert (targets['rcnn_cls_labels'] == 1).sum().item() == 2
    assert (targets['rcnn_cls_labels'] == 0).sum().item() == 2
    assert targets['reg_valid_mask'].sum().item() == 0


def test_multi_domain_dataset_routes_exact_bucket_ratios():
    class DummyDataset:
        class_names = ['Pedestrian']
        point_cloud_range = np.asarray([-1, -1, -1, 1, 1, 1])
        voxel_size = np.asarray([0.1, 0.1, 0.1])
        point_feature_encoder = type('Encoder', (), {'num_point_features': 5})()
        data_processor = object()
        grid_size = np.asarray([20, 20, 20])
        dataset_cfg = {}

        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def __getitem__(self, _index):
            return self.value

    mixed = MultiDomainDataset(
        [DummyDataset('s'), DummyDataset('e'), DummyDataset('n')],
        [0.25, 0.50, 0.25],
        epoch_size=10000,
        reference_dataset_index=1,
    )
    counts = {value: 0 for value in 'sen'}
    for index in range(10000):
        counts[mixed[index]] += 1
    assert counts == {'s': 2500, 'e': 5000, 'n': 2500}


class WeakSupervisionTest(unittest.TestCase):
    def test_manual_annotation_key_is_scene_frame_list_index(self):
        test_manual_annotation_key_is_scene_frame_list_index()

    def test_manual_annotation_frame_list_index_must_be_in_range(self):
        test_manual_annotation_frame_list_index_must_be_in_range()

    def test_box_geometry_and_oriented_point_count(self):
        test_box_geometry_and_oriented_point_count()

    def test_invalid_box_is_downgraded_to_center(self):
        test_invalid_box_is_downgraded_to_center()

    def test_supported_box_passes_conservative_filter(self):
        test_supported_box_passes_conservative_filter()

    def test_bidirectional_box_needs_minimum_image_score(self):
        test_bidirectional_box_needs_minimum_image_score()

    def test_manual_center_and_nearby_pseudo_box_remain_complementary_targets(self):
        test_manual_center_and_nearby_pseudo_box_remain_complementary_targets()

    def test_exhaustive_frame_raises_only_pseudo_classification_confidence(self):
        test_exhaustive_frame_raises_only_pseudo_classification_confidence()

    def test_temporal_support_uses_both_sides(self):
        test_temporal_support_uses_both_sides()

    def test_regression_loss_respects_code_mask_and_weight(self):
        test_regression_loss_respects_code_mask_and_weight()

    def test_focal_mask_confidence_is_not_cancelled_by_normalization(self):
        test_focal_mask_confidence_is_not_cancelled_by_normalization()

    def test_collate_pads_weak_supervision_fields(self):
        test_collate_pads_weak_supervision_fields()

    def test_recursive_config_inheritance_for_all_three_stages(self):
        test_recursive_config_inheritance_for_all_three_stages()

    def test_center_head_builds_unknown_mask_and_partial_regression_target(self):
        test_center_head_builds_unknown_mask_and_partial_regression_target()

    def test_center_head_separates_exhaustive_background_and_positive_weights(self):
        test_center_head_separates_exhaustive_background_and_positive_weights()

    def test_ema_teacher_upgrades_matching_center_but_not_roi_state(self):
        test_ema_teacher_upgrades_matching_center_but_not_roi_state()

    def test_ema_target_identity_is_independent_of_exhaustive_labels(self):
        test_ema_target_identity_is_independent_of_exhaustive_labels()

    def test_ema_parameter_update(self):
        test_ema_parameter_update()

    def test_ema_point_filter_ignores_history_sweep_points(self):
        test_ema_point_filter_ignores_history_sweep_points()

    def test_frozen_modules_remain_in_eval_mode(self):
        test_frozen_modules_remain_in_eval_mode()

    def test_center_only_evaluation_reports_recall_not_ap(self):
        test_center_only_evaluation_reports_recall_not_ap()

    def test_exhaustive_center_evaluation_reports_precision_recall_and_f1(self):
        test_exhaustive_center_evaluation_reports_precision_recall_and_f1()

    def test_roi_center_labels_are_one_to_one_and_keep_ambiguity_ignored(self):
        test_roi_center_labels_are_one_to_one_and_keep_ambiguity_ignored()

    def test_roi_center_supervision_trains_classification_but_not_regression(self):
        test_roi_center_supervision_trains_classification_but_not_regression()

    def test_multi_domain_dataset_routes_exact_bucket_ratios(self):
        test_multi_domain_dataset_routes_exact_bucket_ratios()


if __name__ == '__main__':
    unittest.main()
