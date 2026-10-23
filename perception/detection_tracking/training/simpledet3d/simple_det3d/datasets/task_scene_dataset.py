"""Task annotation/replay composition; sampling never changes label weights."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

from .coda_dataset import CodaDataset
from .mixed_domain_dataset import MultiDomainDataset
from ..train.sampling import CyclicCoverageSampler, WeightedDistributedSampler


def merge_config(base, patch):
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path.resolve()), size=path.stat().st_size, sha256=digest.hexdigest())


class TaskSceneDataset(ConcatDataset):
    def __init__(self, datasets, names, weights=None):
        if not datasets or any(not len(d) for d in datasets):
            raise ValueError('Each configured task dataset must have usable samples')
        if len(names) != len(datasets) or len(set(names)) != len(names):
            raise ValueError('Task dataset names must be unique')
        if any(not name or '/' in name or '::' in name for name in names):
            raise ValueError('Task dataset names must be nonempty path-free names')
        super().__init__(datasets)
        self.names = list(names)
        reference = datasets[0]
        MultiDomainDataset._validate_compatibility(self, reference)
        for key in ('class_names', 'point_cloud_range', 'point_feature_encoder', 'data_processor',
                    'grid_size', 'voxel_size', 'dataset_cfg', 'training'):
            setattr(self, key, getattr(reference, key))
        self.infos = []
        self.sample_weights = []
        self.summary = []
        self.input_files = []
        seen_frames = set()
        for index, (name, dataset) in enumerate(zip(names, datasets)):
            if any(not info.get('is_target_domain') for info in dataset.infos):
                raise ValueError('Task-scene adaptation accepts target-domain data only; CODa is excluded')
            weight = float(weights[index]) if weights is not None else 1.
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError('Each per-frame sample weight must be positive and finite')
            self.sample_weights.extend([weight] * len(dataset))
            for info in dataset.infos:
                identity = (('ros', info['scene_id'], info['frame_index']) if 'annotation_status' in info else
                            ('asset', str(dataset._resolve_lidar_path(info['lidar_path']).resolve())))
                if identity in seen_frames:
                    raise ValueError('Duplicate frame across task inputs; use one consistent annotation snapshot')
                seen_frames.add(identity)
                original = info.get('token', Path(info['lidar_path']).stem)
                self.infos.append(dict(info, token=name + '::' + original, dataset_name=name,
                                       original_token=original))
            self.summary.append(dict(name=name, samples=len(dataset), per_frame_weight=weight,
                                     empty_complete=sum(not len(i['gt_boxes']) and i.get('center_label_is_exhaustive', False)
                                                        for i in dataset.infos),
                                     positive_only=sum(not i.get('center_label_is_exhaustive', False) for i in dataset.infos),
                                     data_config=dict(dataset.dataset_cfg)))
            for item in dataset.dataset_cfg.INFO_PATH[dataset.mode]:
                self.input_files.append(fingerprint(dataset.root_path / item))
        mass = sum(self.sample_weights)
        for entry in self.summary:
            entry['expected_fraction'] = entry['samples'] * entry['per_frame_weight'] / mass

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # Distinct exports may reuse bag/frame names; keep evaluation identities unique.
        return dict(data, frame_id=self.infos[index]['token'])

    collate_batch = staticmethod(CodaDataset.collate_batch)
    generate_prediction_dicts = CodaDataset.generate_prediction_dicts
    _greedy_center_matches = staticmethod(CodaDataset._greedy_center_matches)
    _evaluation_center_only = CodaDataset._evaluation_center_only

    def evaluation(self, det_annos, class_names, **kwargs):
        tokens = [str(a['frame_id']) for a in det_annos]
        if len(tokens) != len(set(tokens)) or set(tokens) != {i['token'] for i in self.infos}:
            raise ValueError('Evaluation must provide exactly one prediction record per selected frame')
        text, metrics = self._evaluation_center_only(det_annos, self.dataset_cfg.WEAK_SUPERVISION)
        # Only explicitly known groups are used; absent location metadata is not invented.
        for kind in ('dataset_name', 'root_group', 'location_group', 'scene_id'):
            groups = sorted({str(i[kind]) for i in self.infos if i.get(kind)})
            if len(groups) < 2:
                continue
            for group in groups:
                probe = copy.copy(self)
                probe.infos = [i for i in self.infos if str(i.get(kind)) == group]
                tokens = {i['token'] for i in probe.infos}
                _, values = probe._evaluation_center_only([a for a in det_annos if a['frame_id'] in tokens],
                                                           self.dataset_cfg.WEAK_SUPERVISION)
                metrics.update({kind + '/' + group + '/' + k: v for k, v in values.items()})
        return text, metrics


def build_task_scene_dataset(cfg, *, training, logger=None, diagnostic_training=False, names=None):
    task = cfg.TASK_SCENE
    if cfg.get('MIXED_DOMAIN', {}).get('ENABLED'):
        raise ValueError('TASK_SCENE and fixed-ratio MIXED_DOMAIN cannot both be enabled')
    specs = task.TRAIN_DATASETS if training or diagnostic_training else task.VALIDATION_DATASETS
    datasets, selected_names, weights = [], [], []
    for spec in specs:
        if names and spec.NAME not in names:
            continue
        data_cfg = merge_config(cfg.DATA_CONFIG, spec.get('CONFIG', {}))
        # Lists replace, rather than inherit, the old stage's data selection.
        if diagnostic_training:
            data_cfg.INFO_PATH.test = list(data_cfg.INFO_PATH.train)
            data_cfg.DATA_SPLIT.test = data_cfg.DATA_SPLIT.train
        dataset = CodaDataset(data_cfg, cfg.CLASS_NAMES, training=training, logger=logger)
        if training:
            before = len(dataset.infos)
            dataset.infos = [i for i in dataset.infos if len(i['gt_boxes']) or i.get('center_label_is_exhaustive', False)]
            if logger is not None and before != len(dataset.infos):
                logger.info('%s: excluded %d zero-center, non-exhaustive frames', spec.NAME, before-len(dataset.infos))
        if spec.get('DIAGNOSTIC_LIMIT') is not None:
            limit = spec.DIAGNOSTIC_LIMIT
            if training or task.get('VALIDATION_ROLE') != 'smoke_diagnostic' or type(limit) is not int or limit < 1:
                raise ValueError('DIAGNOSTIC_LIMIT is only allowed for smoke_diagnostic validation')
            indices = np.linspace(0, len(dataset)-1, min(limit, len(dataset))).astype(int)
            dataset.infos = [dataset.infos[i] for i in indices]
        datasets.append(dataset)
        selected_names.append(spec.NAME)
        weights.append(spec.get('WEIGHT', 1.))
    if names and set(names) != set(selected_names):
        raise ValueError('Unknown requested task dataset')
    result = TaskSceneDataset(datasets, selected_names, weights if training else None)
    frozen = {i['path']: i for i in task.get('INPUT_BINDINGS', [])}
    for item in result.input_files:
        if item['path'] in frozen and item != frozen[item['path']]:
            raise ValueError('Bound annotation export changed; prepare a new input config and experiment')
    return result


def check_train_validation(train, validation):
    train_assets = {(str(d._resolve_lidar_path(i['lidar_path']).resolve()))
                    for d in train.datasets for i in d.infos}
    val_assets = {(str(d._resolve_lidar_path(i['lidar_path']).resolve()))
                  for d in validation.datasets for i in d.infos}
    if train_assets & val_assets:
        raise ValueError('Training and validation share current point-cloud frames')
    for key in ('root_group', 'location_group'):
        first = {i[key] for i in train.infos if i.get(key)}
        second = {i[key] for i in validation.infos if i.get(key)}
        if first & second:
            raise ValueError('Training/validation share ' + key + ': ' + str(sorted(first & second)))


def make_task_sampler(dataset, cfg, world_size=1, rank=0):
    name = cfg.TASK_SCENE.get('SAMPLER', 'weighted_replacement')
    classes = {'cyclic_coverage': CyclicCoverageSampler, 'weighted_replacement': WeightedDistributedSampler}
    if name not in classes:
        raise ValueError('Unknown task sampler: ' + str(name))
    sampler = classes[name](dataset.sample_weights, cfg.TASK_SCENE.get('EPOCH_SAMPLES'),
                            num_replicas=world_size, rank=rank, seed=cfg.TASK_SCENE.get('SEED', 666))
    if isinstance(sampler, CyclicCoverageSampler):
        exposures = {g['weight']: g['exposure_per_frame'] for g in sampler.coverage}
        for entry in dataset.summary:
            entry['exposure_per_frame'] = exposures[entry['per_frame_weight']]
            entry['expected_epoch_samples'] = entry['samples'] * entry['exposure_per_frame']
            entry['expected_fraction'] = entry['expected_epoch_samples'] / sampler.total_size
    return sampler


def data_snapshot(train, validation, sampler, cfg):
    sampling = dict(requested_epoch_samples=sampler.requested_size,
                    actual_epoch_samples=sampler.total_size, seed=sampler.seed,
                    replacement=sampler.replacement, world_size=sampler.num_replicas)
    if isinstance(sampler, CyclicCoverageSampler):
        sampling.update(strategy='cyclic_coverage', pools=sampler.coverage,
                        resume_position='reconstructed_from_seed_and_epoch')
    return dict(training=train.summary, validation=validation.summary,
                validation_role=cfg.TASK_SCENE.get('VALIDATION_ROLE', 'development'),
                inputs=train.input_files + validation.input_files,
                sampling=sampling)


def save_data_snapshot(path, snapshot):
    path = Path(path)
    text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(text):
            raise ValueError('Training data/configuration changed; use a new experiment directory')
    else:
        path.write_text(text, encoding='utf-8')


def check_resume_inputs(checkpoint_path, snapshot):
    path = Path(checkpoint_path).resolve().parent.parent / 'training_data.json'
    if not path.is_file() or json.loads(path.read_text()) != json.loads(json.dumps(snapshot)):
        raise ValueError('Full resume requires the same recorded task inputs; use --pretrained_model for new data')
