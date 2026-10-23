"""Deterministic distributed sampling for task adaptation and historical replay."""
import math

import torch
from torch.utils.data import Sampler


class WeightedDistributedSampler(Sampler):
    replacement = True

    def __init__(self, weights, epoch_size=None, *, num_replicas=1, rank=0, seed=666):
        self.weights = torch.as_tensor(weights, dtype=torch.double, device='cpu')
        if (self.weights.ndim != 1 or not len(self.weights) or
                not torch.isfinite(self.weights).all() or (self.weights <= 0).any()):
            raise ValueError('Sample weights must be finite and positive')
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError('Invalid sampler rank/world size')
        epoch_size = math.ceil(float(self.weights.sum())) if epoch_size is None else epoch_size
        if type(epoch_size) is not int or epoch_size < 1:
            raise ValueError('EPOCH_SAMPLES must be a positive integer or null')
        self.requested_size = epoch_size
        self.num_replicas, self.rank, self.seed = num_replicas, rank, int(seed)
        self.num_samples = math.ceil(epoch_size / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(draws[self.rank::self.num_replicas].tolist())


class CyclicCoverageSampler(WeightedDistributedSampler):
    """Full-weight samples every epoch; fractional weights get cyclic replay.

    Per-weight quotas are fixed across epochs, allowing a fixed LR schedule.
    Pool cursors are reconstructed from seed and epoch, so resume needs no
    extra state. A cycle boundary never repeats a sample within one epoch.
    """

    replacement = False

    def __init__(self, weights, epoch_size=None, *, num_replicas=1, rank=0, seed=666):
        super().__init__(weights, epoch_size, num_replicas=num_replicas, rank=rank, seed=seed)
        if (self.weights > 1).any():
            raise ValueError('Cyclic coverage requires per-frame weights in (0, 1]')
        full_count = int((self.weights == 1).sum())
        if self.requested_size < full_count:
            raise ValueError('EPOCH_SAMPLES cannot be smaller than the full-weight sample count')
        if self.total_size > len(self.weights):
            raise ValueError('DDP-aligned EPOCH_SAMPLES exceeds the unique sample count')
        self.groups = []
        for weight in self.weights.unique(sorted=True).tolist():
            indices = (self.weights == weight).nonzero().flatten()
            self.groups.append(dict(weight=weight, indices=indices, quota=len(indices) if weight == 1 else 0))
        remaining = self.total_size - full_count
        active = [i for i,g in enumerate(self.groups) if g['weight'] < 1]
        # Capped proportional allocation; integer remainders use a stable order.
        targets = {}
        while active:
            mass = sum(self.groups[i]['weight'] * len(self.groups[i]['indices']) for i in active)
            shares = {i: remaining * self.groups[i]['weight'] * len(self.groups[i]['indices']) / mass
                      for i in active}
            saturated = [i for i in active if shares[i] >= len(self.groups[i]['indices'])]
            if not saturated:
                targets.update(shares)
                break
            for i in saturated:
                count = len(self.groups[i]['indices'])
                targets[i] = float(count)
                remaining -= count
                active.remove(i)
        for i,target in targets.items():
            self.groups[i]['quota'] = math.floor(target)
        remainder = self.total_size - sum(g['quota'] for g in self.groups)
        order = sorted(targets, key=lambda i: (-(targets[i] - math.floor(targets[i])), i))
        for i in order:
            if remainder and self.groups[i]['quota'] < len(self.groups[i]['indices']):
                self.groups[i]['quota'] += 1
                remainder -= 1
        if remainder:
            raise ValueError('Unable to allocate a unique, full-coverage epoch')

    @property
    def coverage(self):
        return [dict(weight=g['weight'], samples=len(g['indices']), per_epoch=g['quota'],
                     exposure_per_frame=g['quota'] / len(g['indices'])) for g in self.groups]

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if self.epoch < 0:
            raise ValueError('Epoch must be nonnegative')

    def _pool_slice(self, group, group_id):
        indices, quota = group['indices'], group['quota']
        size = len(indices)
        if quota == size:
            return indices
        if not quota:
            return indices[:0]
        generator = torch.Generator().manual_seed(self.seed + group_id + 1)
        cursor = size
        # Replay only index selection, not data loading or model computation.
        for _ in range(self.epoch + 1):
            selected, needed = [], quota
            while needed:
                if cursor == size:
                    pool = torch.randperm(size, generator=generator)
                    if selected:
                        used = torch.zeros(size, dtype=torch.bool)
                        used[torch.cat(selected)] = True
                        blocked = used[pool]
                        pool = torch.cat((pool[~blocked], pool[blocked]))
                    cursor = 0
                take = min(needed, size - cursor)
                selected.append(pool[cursor:cursor + take])
                cursor += take
                needed -= take
        return indices[torch.cat(selected)]

    def __iter__(self):
        draws = torch.cat([self._pool_slice(group, index) for index,group in enumerate(self.groups)])
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = draws[torch.randperm(self.total_size, generator=generator)]
        return iter(draws[self.rank::self.num_replicas].tolist())
