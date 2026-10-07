"""Deterministic source/target dataset mixing for domain adaptation."""

from __future__ import annotations

import math

import numpy as np
import torch.utils.data as torch_data

from .coda_dataset import CodaDataset


class MixedDomainDataset(torch_data.Dataset):
    """Mix two compatible datasets with a fixed source-sample ratio.

    The dataset index determines the domain, which keeps the aggregate ratio
    deterministic under ``DistributedSampler``.  The dataloader still shuffles
    indices each epoch, so source and target samples are interleaved.
    """

    _RATIO_BUCKETS = 1000
    _INDEX_MULTIPLIER = 104729

    def __init__(self, source_dataset, target_dataset, source_ratio=0.5, epoch_size=None):
        if not 0.0 <= source_ratio <= 1.0:
            raise ValueError('source_ratio must be in [0, 1]')
        if len(source_dataset) == 0 or len(target_dataset) == 0:
            raise ValueError('source and target datasets must both be non-empty')

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_ratio = float(source_ratio)
        self.source_buckets = int(round(self.source_ratio * self._RATIO_BUCKETS))

        if epoch_size is None:
            source_need = len(source_dataset) / max(self.source_ratio, 1e-12) \
                if self.source_ratio > 0 else 0
            target_need = len(target_dataset) / max(1.0 - self.source_ratio, 1e-12) \
                if self.source_ratio < 1 else 0
            epoch_size = int(math.ceil(max(source_need, target_need)))
        if int(epoch_size) <= 0:
            raise ValueError('epoch_size must be positive')
        self.epoch_size = int(epoch_size)

        self._validate_compatibility()

        # Model construction reads these attributes from its dataset.
        self.class_names = target_dataset.class_names
        self.point_cloud_range = target_dataset.point_cloud_range
        self.point_feature_encoder = target_dataset.point_feature_encoder
        self.data_processor = target_dataset.data_processor
        self.grid_size = target_dataset.grid_size
        self.voxel_size = target_dataset.voxel_size
        self.dataset_cfg = target_dataset.dataset_cfg
        self.training = True

    def _validate_compatibility(self):
        if list(self.source_dataset.class_names) != list(self.target_dataset.class_names):
            raise ValueError('source and target class_names must match')
        if not np.allclose(
            self.source_dataset.point_cloud_range,
            self.target_dataset.point_cloud_range,
        ):
            raise ValueError('source and target POINT_CLOUD_RANGE must match')
        if not np.allclose(self.source_dataset.voxel_size, self.target_dataset.voxel_size):
            raise ValueError('source and target VOXEL_SIZE must match')
        source_features = self.source_dataset.point_feature_encoder.num_point_features
        target_features = self.target_dataset.point_feature_encoder.num_point_features
        if source_features != target_features:
            raise ValueError('source and target point feature dimensions must match')

    def __len__(self):
        return self.epoch_size

    def is_source_index(self, index):
        return int(index) % self._RATIO_BUCKETS < self.source_buckets

    def __getitem__(self, index):
        index = int(index)
        mapped_index = (index * self._INDEX_MULTIPLIER + index // self._RATIO_BUCKETS)
        if self.is_source_index(index):
            return self.source_dataset[mapped_index % len(self.source_dataset)]
        return self.target_dataset[mapped_index % len(self.target_dataset)]

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        return CodaDataset.collate_batch(batch_list, _unused=_unused)


class MultiDomainDataset(torch_data.Dataset):
    """Mix two or more compatible datasets with deterministic ratios.

    This is used by Wheelchair stage 2 for source / exhaustive target /
    non-exhaustive target sampling.  Index-based routing remains deterministic
    under ``DistributedSampler`` and avoids rank-dependent domain ratios.
    """

    _RATIO_BUCKETS = 10000
    _INDEX_MULTIPLIER = 104729

    def __init__(self, datasets, ratios, names=None, epoch_size=None,
                 reference_dataset_index=0):
        if len(datasets) < 2 or len(datasets) != len(ratios):
            raise ValueError('datasets and ratios must have the same length >= 2')
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError('all mixed datasets must be non-empty')
        ratios = np.asarray(ratios, dtype=np.float64)
        if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-6):
            raise ValueError('multi-domain ratios must be non-negative and sum to 1')
        if not 0 <= int(reference_dataset_index) < len(datasets):
            raise ValueError('reference_dataset_index is out of range')

        self.datasets = list(datasets)
        self.ratios = ratios.tolist()
        self.names = list(names) if names is not None else [
            f'dataset_{idx}' for idx in range(len(datasets))
        ]
        if len(self.names) != len(self.datasets):
            raise ValueError('names must match datasets')

        bucket_counts = np.floor(ratios * self._RATIO_BUCKETS).astype(np.int64)
        remainder = self._RATIO_BUCKETS - int(bucket_counts.sum())
        fractional = ratios * self._RATIO_BUCKETS - bucket_counts
        for idx in np.argsort(-fractional)[:remainder]:
            bucket_counts[idx] += 1
        self.bucket_boundaries = np.cumsum(bucket_counts)

        if epoch_size is None:
            requirements = [
                len(dataset) / ratio
                for dataset, ratio in zip(self.datasets, ratios)
                if ratio > 0
            ]
            epoch_size = int(math.ceil(max(requirements)))
        if int(epoch_size) <= 0:
            raise ValueError('epoch_size must be positive')
        self.epoch_size = int(epoch_size)

        reference = self.datasets[int(reference_dataset_index)]
        self._validate_compatibility(reference)
        self.class_names = reference.class_names
        self.point_cloud_range = reference.point_cloud_range
        self.point_feature_encoder = reference.point_feature_encoder
        self.data_processor = reference.data_processor
        self.grid_size = reference.grid_size
        self.voxel_size = reference.voxel_size
        self.dataset_cfg = reference.dataset_cfg
        self.training = True

    def _validate_compatibility(self, reference):
        for dataset in self.datasets:
            if list(dataset.class_names) != list(reference.class_names):
                raise ValueError('mixed class_names must match')
            if not np.allclose(dataset.point_cloud_range, reference.point_cloud_range):
                raise ValueError('mixed POINT_CLOUD_RANGE values must match')
            if not np.allclose(dataset.voxel_size, reference.voxel_size):
                raise ValueError('mixed VOXEL_SIZE values must match')
            if (
                dataset.point_feature_encoder.num_point_features
                != reference.point_feature_encoder.num_point_features
            ):
                raise ValueError('mixed point feature dimensions must match')

    def __len__(self):
        return self.epoch_size

    def dataset_index(self, index):
        bucket = int(index) % self._RATIO_BUCKETS
        return int(np.searchsorted(self.bucket_boundaries, bucket, side='right'))

    def __getitem__(self, index):
        index = int(index)
        dataset_idx = self.dataset_index(index)
        mapped_index = (
            index * self._INDEX_MULTIPLIER
            + index // self._RATIO_BUCKETS
            + dataset_idx * 1009
        )
        dataset = self.datasets[dataset_idx]
        return dataset[mapped_index % len(dataset)]

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        return CodaDataset.collate_batch(batch_list, _unused=_unused)
