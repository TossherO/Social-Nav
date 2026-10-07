import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    pass

from ..utils.spconv_utils import replace_feature


def limit_voxels_per_sample(unq_coords, unq_inv, scale_xyz, max_voxels):
    """Apply a deterministic per-sample cap to dynamic non-empty voxels.

    ``unq_coords`` contains the merged ``batch * scale_xyz + xyz`` key and
    ``unq_inv`` maps every point to one entry in ``unq_coords``.  Returning a
    point mask as ``None`` keeps the uncapped fast path allocation-free.
    """
    if max_voxels is None or int(max_voxels) <= 0 or unq_coords.numel() == 0:
        return unq_coords, unq_inv, None

    max_voxels = int(max_voxels)
    voxel_batch_indices = torch.div(
        unq_coords, int(scale_xyz), rounding_mode='floor'
    )
    _, batch_counts = torch.unique_consecutive(
        voxel_batch_indices, return_counts=True
    )
    batch_counts_cpu = batch_counts.tolist()
    if max(batch_counts_cpu) <= max_voxels:
        return unq_coords, unq_inv, None

    keep_voxels = torch.zeros(
        unq_coords.numel(), dtype=torch.bool, device=unq_coords.device
    )
    # Multiplication modulo this prime gives a deterministic pseudo-random
    # ordering of local voxel keys.  It avoids retaining only one spatial
    # prefix while keeping Teacher and Student selections identical.
    hash_modulus = 2147483647
    hash_multiplier = 1103515245
    sample_start = 0
    for batch_count in batch_counts_cpu:
        sample_indices = torch.arange(
            sample_start, sample_start + batch_count,
            device=unq_coords.device,
        )
        sample_start += batch_count
        if batch_count <= max_voxels:
            keep_voxels[sample_indices] = True
            continue

        local_coords = torch.remainder(
            unq_coords[sample_indices].long(), int(scale_xyz)
        )
        hashes = torch.remainder(
            local_coords * hash_multiplier + 12345, hash_modulus
        )
        selected_local = torch.topk(
            hashes, k=max_voxels, largest=False, sorted=False
        ).indices
        keep_voxels[sample_indices[selected_local]] = True

    selected_voxel_indices = torch.nonzero(
        keep_voxels, as_tuple=False
    ).flatten()
    old_to_new = torch.full(
        (unq_coords.numel(),), -1, dtype=unq_inv.dtype,
        device=unq_inv.device,
    )
    old_to_new[selected_voxel_indices] = torch.arange(
        selected_voxel_indices.numel(), dtype=unq_inv.dtype,
        device=unq_inv.device,
    )
    remapped_inv = old_to_new[unq_inv]
    point_keep = remapped_inv >= 0
    return (
        unq_coords[selected_voxel_indices],
        remapped_inv[point_keep],
        point_keep,
    )


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class MeanVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, point_cloud_range, voxel_size, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict


class DynamicVoxelVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size,
                 point_cloud_range, max_num_voxels=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.max_points_per_voxel = self.model_cfg.get('MAX_POINTS_PER_VOXEL', -1)
        self.max_num_voxels = max_num_voxels
        self.max_num_voxels_mode = None

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        # Runtime geometry follows model.to(device) but stays out of
        # checkpoints, preserving compatibility with existing weights.
        self.register_buffer(
            'grid_size', torch.as_tensor(grid_size), persistent=False
        )
        self.register_buffer(
            'voxel_size', torch.as_tensor(voxel_size), persistent=False
        )
        self.register_buffer(
            'point_cloud_range', torch.as_tensor(point_cloud_range),
            persistent=False,
        )

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def _get_max_num_voxels(self):
        if self.max_num_voxels is None:
            return None
        if hasattr(self.max_num_voxels, 'get'):
            mode = self.max_num_voxels_mode
            if mode is None:
                mode = 'train' if self.training else 'test'
            value = self.max_num_voxels.get(mode, None)
        else:
            value = self.max_num_voxels
        if value is None or int(value) <= 0:
            return None
        return int(value)

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv = torch.unique(
            merge_coords, return_inverse=True, dim=0
        )
        num_voxels_before_limit = unq_coords.numel()
        unq_coords, unq_inv, point_keep = limit_voxels_per_sample(
            unq_coords,
            unq_inv,
            scale_xyz=self.scale_xyz,
            max_voxels=self._get_max_num_voxels(),
        )
        if point_keep is not None:
            points = points[point_keep]
            points_coords = points_coords[point_keep]
            points_xyz = points_xyz[point_keep]

        batch_dict['vfe_num_voxels_before_limit'] = num_voxels_before_limit
        batch_dict['vfe_num_voxels'] = unq_coords.numel()

        # 限制每个体素内的最大点数（参考 transform_points_to_voxels 的 MAX_POINTS_PER_VOXEL）
        # 由于 shuffle_points 已随机打乱点顺序，直接取每体素前 K 个点即可实现随机采样
        if self.max_points_per_voxel > 0:
            num_points = points.shape[0]
            # 按体素 ID 分组排序，同一体素的点聚集到连续位置
            sorted_idx = torch.argsort(unq_inv)
            sorted_inv = unq_inv[sorted_idx]
            # 标记每个体素组的起始位置
            boundary = torch.cat([
                torch.tensor([True], device=points.device),
                sorted_inv[1:] != sorted_inv[:-1]
            ])
            # 计算每个点在排序后序列中的组内位置
            group_ids_sorted = torch.cumsum(boundary, dim=0) - 1
            all_pos = torch.arange(num_points, device=points.device)
            group_starts = all_pos[boundary]
            expanded_starts = group_starts[group_ids_sorted]
            pos_in_group = all_pos - expanded_starts
            # 每个体素保留前 max_points_per_voxel 个点
            keep_sorted = pos_in_group < self.max_points_per_voxel
            keep = torch.zeros(num_points, dtype=torch.bool, device=points.device)
            keep[sorted_idx] = keep_sorted
            # 应用过滤
            points = points[keep]
            points_coords = points_coords[keep]
            points_xyz = points_xyz[keep]
            unq_inv = unq_inv[keep]

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict
