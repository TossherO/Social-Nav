"""Bounded final fitting of the global field on observed mapping surfaces."""
import time

import numpy as np
import open3d as o3d
import torch

from utils.loss import sdf_bce_loss
from utils.tracker import Tracker


def refine_global_features(neural, decoder, config):
    iterations = config.final_map_refine_iters
    if iterations == 0:
        return {'iterations': 0, 'scope': 'disabled'}
    started = time.monotonic()
    neural.assign_local_to_global()
    neural.temporal_local_map_on = False
    # This is the same representative selection and query used after loading.
    neural.recreate_hash(None, None, True, False)
    indices = torch.unique(neural.buffer_pt_index[neural.buffer_pt_index >= 0])
    indices = indices[neural.point_certainties[indices] >= 5.]
    if len(indices) < 30:
        raise ValueError('Too few observed surface points for final map fitting')
    centers = neural.neural_points[indices].detach().clone()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers.cpu().numpy()))
    radius = config.voxel_size_m * 2.5
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = torch.as_tensor(np.asarray(cloud.normals).copy(), device=config.device, dtype=config.dtype)
    decoder.requires_grad_(False)
    config.decoder_freezed = True
    tracker = Tracker(config, neural, {'sdf': decoder, 'semantic': None, 'color': None})
    _, gradient, *_ = tracker.query_source_points(centers, config.infer_bs, query_locally=False)
    # Geometric normals are unoriented; retain the learned field's sign.
    normals[(normals * gradient).sum(dim=1) < 0] *= -1
    centers = centers.detach()
    neural.geo_features = torch.nn.Parameter(neural.geo_features.detach().clone())
    optimizer = torch.optim.Adam([neural.geo_features], lr=config.lr)
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    offsets = torch.tensor([0., config.final_map_refine_offset_m, -config.final_map_refine_offset_m],
                           device=config.device, dtype=config.dtype).repeat_interleave(config.final_map_refine_bs)
    first_loss = None
    for step in range(iterations):
        selected = torch.randint(len(centers), (config.final_map_refine_bs,),
                                 device=config.device, generator=generator)
        coordinates = centers[selected].repeat(3, 1) + normals[selected].repeat(3, 1) * offsets[:, None]
        features, _, weights, _, _ = neural.query_feature(coordinates, training_mode=False,
                                                         query_locally=False, query_color_feature=False)
        prediction = decoder.sdf(features)
        if not config.weighted_first:
            prediction = (prediction * weights).sum(dim=1).squeeze(1)
        loss = sdf_bce_loss(prediction, offsets, tracker.sdf_scale, torch.ones_like(offsets))
        if first_loss is None:
            first_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not torch.isfinite(neural.geo_features).all():
        raise ValueError('Nonfinite features after final map fitting')
    neural.geo_features.requires_grad_(False)
    # Saving flushes local features: refresh their copies after global fitting
    # so an old local cache cannot overwrite the newly optimized field.
    neural.reset_local_map(torch.zeros(3, device=config.device, dtype=config.dtype), None, neural.max_ts)
    neural.requires_grad_(False)
    return dict(scope='observed_mapping_surfaces_only', iterations=iterations,
                surface_points=len(centers), batch_size=config.final_map_refine_bs,
                normal_offset_m=config.final_map_refine_offset_m, normal_radius_m=radius,
                loss='sdf_bce', initial_loss=first_loss, final_loss=float(loss.detach()),
                frozen_decoder=True, frozen_point_geometry=True,
                elapsed_s=time.monotonic()-started)
