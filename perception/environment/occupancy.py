"""Small checked OctoMap C ABI; no Python/CUDA extension ABI dependency."""
import ctypes as C
from pathlib import Path

import numpy as np


class OccupancyTree:
    def __init__(self, resolution, library=None):
        root = Path(__file__).resolve().parents[2]
        path = Path(library) if library else root/'outputs/perception/ros1_ws/devel/lib/libsocial_nav_occupancy.so'
        if not path.is_file():
            raise RuntimeError('Build the environment profile first: bash perception/scripts/build_ros1.sh environment')
        self.lib = lib = C.CDLL(str(path))
        fp, dp = C.POINTER(C.c_float), C.POINTER(C.c_double)
        lib.sn_occupancy_new.argtypes, lib.sn_occupancy_new.restype = [C.c_double], C.c_void_p
        lib.sn_occupancy_free.argtypes = [C.c_void_p]
        lib.sn_occupancy_error.restype = C.c_char_p
        lib.sn_occupancy_update.argtypes = [C.c_void_p, fp, C.c_int, dp, fp, dp, C.c_double,
                                           C.c_double, C.c_double, C.c_int, C.c_int]
        lib.sn_occupancy_size.argtypes = [C.c_void_p]
        lib.sn_occupancy_read.argtypes = [C.c_void_p, fp, C.c_int, C.c_double]
        self.ptr = lib.sn_occupancy_new(resolution)
        if not self.ptr:
            raise RuntimeError(lib.sn_occupancy_error().decode())

    def close(self):
        if self.ptr:
            self.lib.sn_occupancy_free(self.ptr)
            self.ptr = None

    def update(self, points, origin, bounds, stamp, config, rays=True, ray_origins=None, *, read=True):
        points = np.ascontiguousarray(points, dtype=np.float32)
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        box = np.ascontiguousarray(bounds, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or origin.shape != (3,) or box.shape != (6,)
                or not np.isfinite(points).all() or not np.isfinite(origin).all()
                or not np.isfinite(box).all() or not np.isfinite(stamp)):
            raise ValueError('Invalid occupancy observation')
        stride = max(1, int(np.ceil(len(points)/config['max_rays']))) if rays else 0
        origins = None
        if ray_origins is not None:
            origins = np.ascontiguousarray(ray_origins,dtype=np.float32)
            if origins.shape != points.shape or not np.isfinite(origins).all():
                raise ValueError('Ray origins must match the point observations')
        n = self.lib.sn_occupancy_update(self.ptr, points.ctypes.data_as(C.POINTER(C.c_float)), len(points),
            origin.ctypes.data_as(C.POINTER(C.c_double)),
            origins.ctypes.data_as(C.POINTER(C.c_float)) if origins is not None else None,
            box.ctypes.data_as(C.POINTER(C.c_double)), stamp,
            config['voxel_ttl_s'], config['max_range_m'], stride, config['max_voxels'])
        if n < 0:
            raise RuntimeError(self.lib.sn_occupancy_error().decode())
        return self.snapshot(stamp) if read else n

    def snapshot(self, stamp):
        n = self.lib.sn_occupancy_size(self.ptr)
        if n < 0 or not np.isfinite(stamp):
            raise ValueError('Invalid occupancy snapshot time/size')
        values = np.empty((n,5), dtype=np.float32)
        count = self.lib.sn_occupancy_read(self.ptr, values.ctypes.data_as(C.POINTER(C.c_float)), n, stamp)
        if count < 0:
            raise RuntimeError(self.lib.sn_occupancy_error().decode())
        return values[:count]
