"""Prepare task inputs, inspect sampling, and compare two detector checkpoints."""
import argparse
import json
import logging
import math
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_det3d.datasets.task_scene_dataset import (
    TaskSceneDataset, build_task_scene_dataset, check_train_validation,
    data_snapshot, fingerprint, make_task_sampler,
)
from simple_det3d.utils.config import load_config


def prepare(args):
    if args.out.exists():
        raise FileExistsError('Input config already exists; use a new file')
    base = Path(__file__).resolve().parents[1] / 'configs/voxelrcnn_ros_centers_finetune.yaml'
    specs, bindings = [], []
    for index, filename in enumerate(args.annotations):
        filename = filename.expanduser().resolve()
        if any(item['path'] == str(filename) for item in bindings):
            raise ValueError('Do not list the same annotation export twice')
        document = json.loads(filename.read_text(encoding='utf-8'))
        if document.get('format') != 'rosbag_people_centers':
            raise ValueError('Expected a portable rosbag annotation export')
        groups = {s['root_group'] for s in document['sources']}
        if not all(groups):
            raise ValueError('Each source needs its original capture group')
        if any(s['split'] not in ('unassigned', 'train') or
               s['location_group'] not in ('', args.location_group) for s in document['sources']):
            raise ValueError('Prepare cannot reassign an existing validation split or location group')
        assignments = {g: dict(split='train', location_group=args.location_group) for g in groups}
        specs.append(dict(NAME='task' if index == 0 else 'task_' + str(index + 1), WEIGHT=1.,
                          CONFIG=dict(DATA_PATH=str(filename.parent), RAW_DATA_PATH=str(filename.parent),
                                      INFO_PATH=dict(train=[filename.name], test=[filename.name]),
                                      ROS_GROUP_ASSIGNMENTS=assignments, ROS_COMPLETE_ONLY=args.complete_only)))
        bindings.append(fingerprint(filename))
    config = dict(_BASE_CONFIG_=str(base), TASK_SCENE=dict(
        TRAIN_DATASETS=specs + [dict(NAME='legacy', WEIGHT=.2, CONFIG=dict(
            DATA_PATH='data/wheelchair_weak', RAW_DATA_PATH='data/wheelchair_raw',
            INFO_PATH=dict(train=['wheelchair_weak_infos_train_indoor_exhaustive.pkl'])))],
        INPUT_BINDINGS=bindings))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(str(args.out.resolve()))


def inspect_data(args, cfg, logger):
    train = build_task_scene_dataset(cfg, training=True, logger=logger)
    validation = build_task_scene_dataset(cfg, training=False, logger=logger)
    check_train_validation(train, validation)
    sampler = make_task_sampler(train, cfg, world_size=args.world_size)
    snapshot = data_snapshot(train, validation, sampler, cfg)
    result = dict(training=[{k: v for k, v in s.items() if k != 'data_config'} for s in train.summary],
                  validation=[{k: v for k, v in s.items() if k != 'data_config'} for s in validation.summary],
                  validation_role=snapshot['validation_role'], sampling=snapshot['sampling'])
    if args.load_samples:
        selected = []
        start = 0
        for dataset in train.datasets:
            for predicate in (lambda i: bool(len(i['gt_boxes'])),
                              lambda i: not len(i['gt_boxes']) and i.get('center_label_is_exhaustive'),
                              lambda i: bool(len(i['gt_boxes'])) and not i.get('center_label_is_exhaustive')):
                match = next((n for n, info in enumerate(dataset.infos) if predicate(info)), None)
                if match is not None and start + match not in selected:
                    selected.append(start + match)
            start += len(dataset)
        samples = [train[i] for i in selected]
        batch = train.collate_batch(samples)
        if not np.isfinite(batch['points']).all() or not np.isfinite(batch['gt_boxes']).all():
            raise ValueError('Non-finite data in representative mixed batch')
        result['mixed_batch'] = dict(frames=batch['frame_id'].tolist(), points=len(batch['points']),
                                     gt_counts=[len(s['gt_boxes']) for s in samples],
                                     background_weights=batch['heatmap_background_weight'].tolist(),
                                     roi_center_enabled=batch['roi_center_supervision_enabled'].tolist(),
                                     regression_masks=[s['gt_box_reg_mask'].tolist() for s in samples])
    print(json.dumps(result, ensure_ascii=False, indent=2))


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def metric_comparison(baseline, candidate, *, role, max_recall_drop=None, limited=False,
                      complete_human_labels=True):
    keys = ['CenterPrecision@0.50m', 'CenterRecall@0.50m', 'CenterF1@0.50m',
            'CenterFalsePositivesPerFrame', 'CenterMissedCount@0.50m',
            'EmptyFalsePositivesPerFrame', 'EmptyFalsePositiveFrameRate']
    keys += [key for key in baseline if key.startswith('DistanceWeighted') or (
        key.startswith('Distance/') and
        (key.endswith('CenterRecall@0.50m') or key.endswith('CenterFalsePositivesPerFrame')))]
    delta = {key: candidate[key] - baseline[key] for key in keys if key in baseline and key in candidate}
    eligible = (role == 'task_development' and complete_human_labels and not limited and max_recall_drop is not None and
                all(key in delta and np.isfinite(delta[key])
                    for key in ('CenterRecall@0.50m', 'CenterFalsePositivesPerFrame')))
    return dict(delta=delta, acceptance_assessed=eligible,
                meets_requested_tradeoff=(delta['CenterRecall@0.50m'] >= -max_recall_drop and
                                          delta['CenterFalsePositivesPerFrame'] < 0) if eligible else None)


def compare(args, cfg, logger):
    import torch
    from torch.utils.data import DataLoader
    from simple_det3d.models import build_model
    from simple_det3d.train.train_utils import eval_epoch
    from simple_det3d.utils.common_utils import load_checkpoint
    from simple_det3d.utils.precision_utils import validate_precision_support

    if args.out.exists():
        raise FileExistsError('Comparison output already exists')
    validate_precision_support('fp32', device=0)
    dataset = build_task_scene_dataset(cfg, training=False, logger=logger,
                                       diagnostic_training=args.diagnostic_training, names=args.datasets)
    if not args.diagnostic_training:
        train = build_task_scene_dataset(cfg, training=True, logger=logger)
        check_train_validation(train, dataset)
        del train
    if args.limit_per_dataset:
        for component in dataset.datasets:
            count = min(args.limit_per_dataset, len(component))
            component.infos = [component.infos[i] for i in np.linspace(0, len(component)-1, count).astype(int)]
        dataset = TaskSceneDataset(dataset.datasets, dataset.names)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, collate_fn=dataset.collate_batch, pin_memory=True)
    metrics = {}
    for name, checkpoint_path in (('baseline', args.baseline), ('candidate', args.candidate)):
        model = build_model(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
        checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
        model._load_state_dict(checkpoint['model_state'], strict=True)
        del checkpoint
        model.cuda()
        metrics[name] = eval_epoch(model, loader, cfg, logger, cur_epoch=name, precision='fp32')
        del model
        torch.cuda.empty_cache()
    role = 'training_resubstitution' if args.diagnostic_training else cfg.TASK_SCENE.get('VALIDATION_ROLE', 'development')
    result = dict(role=role, limit_per_dataset=args.limit_per_dataset, samples=len(dataset),
                  inputs=dataset.input_files, datasets=dataset.summary,
                  checkpoints=dict(baseline=fingerprint(args.baseline), candidate=fingerprint(args.candidate)),
                  post_processing=dict(score=cfg.MODEL.POST_PROCESSING.SCORE_THRESH,
                                       nms=cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH,
                                       distance_evaluation=dict(cfg.DATA_CONFIG.WEAK_SUPERVISION.get('DISTANCE_EVAL', {}))),
                  metrics=metrics, comparison=metric_comparison(
                      metrics['baseline'], metrics['candidate'], role=role,
                      max_recall_drop=args.max_recall_drop, limited=bool(args.limit_per_dataset),
                      complete_human_labels=all(i.get('annotation_status') == 'complete' for i in dataset.infos)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(finite_json(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(finite_json(dict(output=str(args.out), comparison=result['comparison'])), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Explicitly assign exported capture groups to training, leaving labels unchanged')
    prep.add_argument('--annotations', type=Path, action='append', required=True)
    prep.add_argument('--location-group', required=True, help='Conservative route/site grouping for this training batch')
    prep.add_argument('--complete-only', action='store_true')
    prep.add_argument('--out', type=Path, required=True, help='New YAML input configuration')
    inspect = commands.add_parser('inspect', help='CPU-only data/sampling summary, optionally read a representative batch')
    inspect.add_argument('--cfg_file', required=True)
    inspect.add_argument('--world-size', type=int, default=2)
    inspect.add_argument('--load-samples', action='store_true')
    inspect.add_argument('--max-sweeps', type=int)
    evaluation = commands.add_parser('compare', help='Compare two checkpoints on the same unweighted evaluation frames')
    evaluation.add_argument('--cfg_file', required=True)
    evaluation.add_argument('--baseline', type=Path, required=True)
    evaluation.add_argument('--candidate', type=Path, required=True)
    evaluation.add_argument('--out', type=Path, required=True, help='New comparison JSON')
    evaluation.add_argument('--batch-size', type=int, default=8)
    evaluation.add_argument('--workers', type=int, default=2)
    evaluation.add_argument('--diagnostic-training', action='store_true', help='Explicit training-set diagnosis, never independent validation')
    evaluation.add_argument('--datasets', nargs='+', help='Select configured evaluation dataset names')
    evaluation.add_argument('--limit-per-dataset', type=int, help='Diagnostic limit; disables acceptance assessment')
    evaluation.add_argument('--max-recall-drop', type=float, help='Absolute permitted recall drop, chosen before development evaluation')
    evaluation.add_argument('--max-sweeps', type=int, help='Explicit temporal input for both models; default from config')
    evaluation.add_argument('--score-threshold', type=float,
                            help='Override both final detection and center-evaluation score thresholds')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.location_group.strip():
            parser.error('A nonempty location group is required')
        return prepare(args)
    if args.command == 'compare':
        if args.batch_size < 1 or args.workers < 0 or (args.limit_per_dataset is not None and args.limit_per_dataset < 1):
            parser.error('Invalid batch, worker count or sample budget')
        if args.max_recall_drop is not None and (not math.isfinite(args.max_recall_drop) or not 0 <= args.max_recall_drop <= 1):
            parser.error('Recall tolerance must be finite and in [0, 1]')
        if args.score_threshold is not None and (not math.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1):
            parser.error('Score threshold must be finite and in [0, 1]')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('task_scene')
    cfg = load_config(args.cfg_file)
    if not cfg.get('TASK_SCENE', {}).get('ENABLED'):
        parser.error('This tool requires a TASK_SCENE configuration')
    if args.command == 'compare' and args.score_threshold is not None:
        cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_threshold
        cfg.DATA_CONFIG.WEAK_SUPERVISION.CENTER_SCORE_THRESH = args.score_threshold
        for spec in list(cfg.TASK_SCENE.TRAIN_DATASETS) + list(cfg.TASK_SCENE.VALIDATION_DATASETS):
            if 'WEAK_SUPERVISION' in spec.get('CONFIG', {}):
                spec.CONFIG.WEAK_SUPERVISION.CENTER_SCORE_THRESH = args.score_threshold
    if args.max_sweeps is not None:
        if args.max_sweeps < 1:
            parser.error('--max-sweeps must be positive')
        cfg.DATA_CONFIG.MAX_SWEEPS = args.max_sweeps
    return inspect_data(args, cfg, logger) if args.command == 'inspect' else compare(args, cfg, logger)


if __name__ == '__main__':
    main()
