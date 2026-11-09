"""CUDA local views through the existing C ABI, also usable by system ROS Python."""

import ctypes as C

import numpy as np

from .near_field import HISTORY_DTYPES


# Native channel order. Wire dtypes remain independent of double device scratch.
VIEW_DTYPES = dict(
    occupancy=np.int8, observed=bool, last_observed=np.float64, coverage=np.float32,
    obstacle_min_z=np.float32, obstacle_max_z=np.float32, ground_z=np.float32,
    ground_support=bool, clearance=np.float32, unknown_distance=np.float32,
    frontier=bool, **HISTORY_DTYPES,
)
_DOUBLE = C.POINTER(C.c_double)
_BYTE = C.POINTER(C.c_uint8)
_INT = C.POINTER(C.c_int32)


def _pointer(array, dtype):
    return None if array is None else array.ctypes.data_as(dtype)


def _policy_inputs(tree, policy, metadata, config, stamp):
    """Reset incompatible history, then prepare the native near-field policy."""
    key = tuple(metadata[k] for k in (
        'session_id', 'generation', 'frame_id', 'resolution'
    )) + tuple(metadata['shape'])
    if key != policy.key or (policy.stamp is not None and stamp <= policy.stamp):
        policy.reset()
        reset = tree.lib.sn_occupancy_reset_view
        reset.argtypes = [C.c_void_p]
        reset(tree.ptr)
    policy.key = key
    policy.stamp = stamp

    pose = np.ascontiguousarray(metadata['world_from_lidar'], np.float64)
    value = policy.prepare_plane(metadata.get('ground_tracking', {}), pose)
    suspend = tree.lib.sn_occupancy_suspend_free_history
    suspend.argtypes = [C.c_void_p, C.c_int]
    suspend(tree.ptr, int(policy.history_suspended))
    plane = None if value is None else np.ascontiguousarray(value, np.float64)
    settings = policy.settings
    native = np.array([
        1., settings['lowest_beam_deg'], settings['max_radius_m'],
        settings['boundary_margin_m'], settings['max_sensor_tilt_rad'],
        settings['ground_change_m'], settings['remember_outside_ring'],
        settings['temporary_obstacle_overlay'], config['obstacle_ttl_s'],
        config.get('dynamic', {}).get('transient_ttl_s', config['obstacle_ttl_s']),
    ], np.float64)
    return pose, plane, native


def _decode_layers(out, *, policy, people, device):
    """Convert native channels to their public dtypes without changing storage side."""
    if device is not None:
        import torch

    layers = {}
    for index, (key, dtype) in enumerate(VIEW_DTYPES.items()):
        if index >= 11 and policy is None:
            continue
        if device is None:
            value = out[index].astype(dtype)
            if key in ('obstacle_min_z', 'obstacle_max_z'):
                value[~np.isfinite(value)] = np.nan
        else:
            value = out[index].to(dtype=torch.from_numpy(np.empty(0, dtype)).dtype)
            if key in ('obstacle_min_z', 'obstacle_max_z'):
                value = value.masked_fill(~torch.isfinite(value), float('nan'))
        layers[key] = value
    if people:
        layers['person_background_free'] = (
            out[16].astype(bool) if device is None else out[16].bool()
        )
    return layers


def native_view(tree, floor, support, origin, config, stamp, endpoints, *, required=None,
                source_layers=None, metadata=None, policy=None, transient_mask=None,
                device=None, metrics=None,ground_conflict=None):
    """Project evidence, decode layers and return the near-field description."""
    people = source_layers is None and policy is not None and transient_mask is not None
    if device is not None:
        if source_layers is not None:
            raise ValueError('Device projection requires resident evidence')
        name = 'sn_occupancy_project_device_view'
    elif source_layers is not None:
        name = 'sn_occupancy_view_from_layers'
    elif people:
        name = 'sn_occupancy_project_people_view'
    else:
        name = 'sn_occupancy_project_view'
    if tree.backend != 'cuda' or not hasattr(tree.lib, name):
        raise RuntimeError('Build the CUDA environment library for local history and distance fields')

    original = np.asarray(floor)
    floor = np.ascontiguousarray(original, np.float64)
    support = np.ascontiguousarray(support, np.uint8)
    if ground_conflict is None and source_layers is not None:
        ground_conflict=source_layers.get('ground_conflict')
    if ground_conflict is not None:
        ground_conflict=np.asarray(ground_conflict)
        if ground_conflict.dtype!=bool or ground_conflict.shape!=floor.shape:
            raise ValueError('Ground conflicts must be a boolean projection grid')
        # Internal C ABI states: 0 missing, 1 supported, 2 conflicting. Public
        # ground_support remains boolean; no extra native input upload is needed.
        support=support.copy();support[ground_conflict]=2
    origin = np.ascontiguousarray(origin, np.float64)
    points = np.ascontiguousarray(endpoints, np.float64)
    if (floor.ndim != 2 or support.shape != floor.shape or origin.shape != (2,) or
            points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
        raise ValueError('Invalid CUDA local view inputs')
    params = np.array([config[k] for k in (
        'resolution', 'min_obstacle_height_m', 'max_height_m',
        'occupied_probability', 'free_probability', 'free_vertical_coverage',
    )] + [original.dtype == np.float32], np.float64)
    counts = np.zeros(6, np.int32)
    ring = np.zeros(4, np.float64)
    pose = plane = settings = mask = None
    if policy is not None:
        pose, plane, settings = _policy_inputs(tree, policy, metadata, config, stamp)
    if transient_mask is not None:
        mask = np.ascontiguousarray(transient_mask, np.uint8)
        if mask.shape != (len(points),) or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid transient view mask')

    shape = (17 if people else 16, *floor.shape)
    if device is None:
        out = np.empty(shape, np.float64)
    else:
        import torch

        out = torch.empty(shape, dtype=torch.float64, device=device)
    method = getattr(tree.lib, name)
    if source_layers is None:
        required = np.ascontiguousarray(required, np.int32)
        prefix_types = [C.c_void_p, _DOUBLE, _BYTE, _INT]
        prefix = [tree.ptr, floor.ctypes.data_as(_DOUBLE),
                  support.ctypes.data_as(_BYTE), required.ctypes.data_as(_INT)]
    else:
        raw = np.ascontiguousarray(
            [source_layers[k] for k in list(VIEW_DTYPES)[:6]], np.float64
        )
        prefix_types = [C.c_void_p, _DOUBLE, _DOUBLE, _BYTE]
        prefix = [tree.ptr, raw.ctypes.data_as(_DOUBLE),
                  floor.ctypes.data_as(_DOUBLE), support.ctypes.data_as(_BYTE)]
    method.argtypes = prefix_types + [
        C.c_int, C.c_int, _DOUBLE, _DOUBLE, C.c_double, _DOUBLE, C.c_int,
        _DOUBLE, _DOUBLE, _DOUBLE, _BYTE, _DOUBLE, _INT, _DOUBLE,
    ]
    if device is not None:
        method.argtypes.append(C.c_int)
    args = [
        *prefix, *floor.shape, origin.ctypes.data_as(_DOUBLE), params.ctypes.data_as(_DOUBLE),
        stamp, points.ctypes.data_as(_DOUBLE), len(points), _pointer(pose, _DOUBLE),
        _pointer(plane, _DOUBLE), _pointer(settings, _DOUBLE), _pointer(mask, _BYTE),
        out.ctypes.data_as(_DOUBLE) if device is None else C.cast(out.data_ptr(), _DOUBLE),
        counts.ctypes.data_as(_INT), ring.ctypes.data_as(_DOUBLE),
    ]
    if device is not None:
        args.append(int(people))
        # The historical C ABI uses the CUDA default stream. Preserve ordering
        # with the current runtime's asynchronous map stream and tensor allocator.
        current=torch.cuda.current_stream(out.device)
        default=torch.cuda.default_stream(out.device)
        default.wait_stream(current)
        out.record_stream(default)
    result = method(*args)
    if result < 0:
        raise RuntimeError(tree.lib.sn_occupancy_error().decode())
    if device is not None:current.wait_stream(default)

    if metrics is not None:
        metrics.transfer(
            'h2d', sum(v.nbytes for v in (floor, support, required, points) if v is not None),
            'environment_projection_inputs',
        )
        metrics.transfer('d2h', counts.nbytes + ring.nbytes, 'environment_policy_summary')
        if device is None:
            metrics.transfer('d2h', out.nbytes, 'environment_host_layers')
        else:
            metrics.transfer('d2d', out.numel() * out.element_size(), 'environment_owned_device_layers')

    layers = _decode_layers(out, policy=policy, people=people, device=device)
    if ground_conflict is not None:
        value=ground_conflict.copy()
        if device is not None:
            # A blocking copy here would wait for all preceding map-stream
            # work. Private pinned storage stays leased by the CUDA copy.
            value=torch.from_numpy(value).pin_memory().to(device,non_blocking=True)
            if metrics is not None:metrics.transfer('h2d',ground_conflict.nbytes,'ground_conflict_consumer_layer')
        else:value.flags.writeable=False
        layers['ground_conflict']=value
    description = None
    if policy is not None:
        state = {
            0: 'ground_ring_unavailable',
            1: 'available',
            2: 'sensor_tilt_outside_ring_model',
            3: 'invalid_ring_clearance',
        }[int(ring[0])]
        info = dict(status=state)
        if state == 'available':
            info.update(
                clearance_m=float(ring[1]), radius_min_m=float(ring[2]),
                radius_max_m=float(ring[3]),
                method='lowest_beam_intersection_with_tracked_plane_whole_cell_inside',
            )
        description = policy.describe(metadata['snapshot_id'], counts, info, 'cuda')
    return layers, description
