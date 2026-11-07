"""CPU/CUDA evidence through C ABI, independent of Python extension ABI."""
import ctypes as C
from pathlib import Path

import numpy as np


class OccupancyTree:
    def __init__(self, resolution, library=None, *, backend='cpu', device=0):
        if backend not in ('cpu','cuda'):
            raise ValueError('Environment compute backend must be cpu or cuda')
        self.backend = backend
        root = Path(__file__).resolve().parents[2]
        name = 'libsocial_nav_occupancy_cuda.so' if backend == 'cuda' else 'libsocial_nav_occupancy.so'
        path = Path(library) if library else root/'outputs/perception/ros1_ws/devel/lib'/name
        if not path.is_file():
            raise RuntimeError('Build the environment library first: '+str(path))
        self.lib = lib = C.CDLL(str(path))
        if backend=='cuda' and not hasattr(lib,'sn_occupancy_suspend_free_history'):
            raise RuntimeError('Rebuild the CUDA environment library for ground-history suspension')
        fp, dp = C.POINTER(C.c_float), C.POINTER(C.c_double)
        lib.sn_occupancy_new.argtypes, lib.sn_occupancy_new.restype = [C.c_double], C.c_void_p
        lib.sn_occupancy_free.argtypes = [C.c_void_p]
        lib.sn_occupancy_error.restype = C.c_char_p
        lib.sn_occupancy_update.argtypes = [C.c_void_p, fp, C.c_int, dp, fp, dp, C.c_double,
                                           C.c_double, C.c_double, C.c_int, C.c_int]
        if hasattr(lib, 'sn_occupancy_update_masked'):
            lib.sn_occupancy_update_masked.argtypes = lib.sn_occupancy_update.argtypes + [C.POINTER(C.c_uint8), C.c_double, C.c_int]
        lib.sn_occupancy_size.argtypes = [C.c_void_p]
        lib.sn_occupancy_read.argtypes = [C.c_void_p, fp, C.c_int, C.c_double]
        if backend == 'cuda':
            lib.sn_occupancy_new_cuda.argtypes, lib.sn_occupancy_new_cuda.restype = [C.c_double,C.c_int],C.c_void_p
            self.ptr = lib.sn_occupancy_new_cuda(resolution,device)
        else:
            self.ptr = lib.sn_occupancy_new(resolution)
        if not self.ptr:
            raise RuntimeError(lib.sn_occupancy_error().decode())

    def close(self):
        if self.ptr:
            self.lib.sn_occupancy_free(self.ptr)
            self.ptr = None

    def update(self, points, origin, bounds, stamp, config, rays=True, ray_origins=None, *, read=True,
               transient_mask=None, transient_ttl=None, omit_transient=False):
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
        arguments = [self.ptr, points.ctypes.data_as(C.POINTER(C.c_float)), len(points),
            origin.ctypes.data_as(C.POINTER(C.c_double)),
            origins.ctypes.data_as(C.POINTER(C.c_float)) if origins is not None else None,
            box.ctypes.data_as(C.POINTER(C.c_double)), stamp,
            config['voxel_ttl_s'], config['max_range_m'], stride, config['max_voxels']]
        lifetime = config.get('obstacle_ttl_s', config['voxel_ttl_s'])
        invalidate = config.get('hit_invalidates_free',False)
        if type(invalidate) is not bool:
            raise ValueError('hit_invalidates_free must be boolean')
        if hasattr(self.lib,'sn_occupancy_hit_invalidates_free'):
            method=self.lib.sn_occupancy_hit_invalidates_free
            method.argtypes=[C.c_void_p,C.c_int]
            method(self.ptr,int(invalidate))
        elif invalidate:
            raise RuntimeError('Rebuild the environment library for hit invalidation')
        if hasattr(self.lib, 'sn_occupancy_obstacle_lifetime'):
            method = self.lib.sn_occupancy_obstacle_lifetime
            method.argtypes = [C.c_void_p, C.c_double]
            if method(self.ptr, lifetime) < 0:
                raise RuntimeError(self.lib.sn_occupancy_error().decode())
        elif lifetime != config['voxel_ttl_s']:
            raise RuntimeError('Rebuild the environment library for separate obstacle expiration')
        if transient_mask is None:
            n = self.lib.sn_occupancy_update(*arguments)
        else:
            mask = np.ascontiguousarray(transient_mask, dtype=np.uint8)
            if (mask.shape != (len(points),) or not np.isin(mask, [0, 1]).all() or
                    transient_ttl is None or not np.isfinite(transient_ttl) or transient_ttl < 0):
                raise ValueError('Invalid transient point mask/lifetime')
            if not hasattr(self.lib, 'sn_occupancy_update_masked'):
                raise RuntimeError('Rebuild the environment occupancy library for transient point masks')
            n = self.lib.sn_occupancy_update_masked(*arguments, mask.ctypes.data_as(C.POINTER(C.c_uint8)),
                                                   transient_ttl, int(omit_transient))
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

    def enable_static_review(self, interval, margin):
        """Opt-in CPU evidence counts for offline export, never online expiration."""
        if self.backend != 'cpu' or not hasattr(self.lib, 'sn_occupancy_enable_static_review'):
            raise RuntimeError('Rebuild the CPU environment library for offline static review')
        method = self.lib.sn_occupancy_enable_static_review
        method.argtypes = [C.c_void_p, C.c_double, C.c_double]
        if method(self.ptr, interval, margin) < 0:
            raise RuntimeError(self.lib.sn_occupancy_error().decode())

    def static_votes(self, points):
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Static review requires finite Nx3 points')
        if self.backend != 'cpu' or not hasattr(self.lib, 'sn_occupancy_static_votes'):
            raise RuntimeError('Rebuild the CPU environment library for offline static review')
        votes = np.empty((len(points), 2), np.int32)
        method = self.lib.sn_occupancy_static_votes
        method.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.POINTER(C.c_int32)]
        n = method(self.ptr, points.ctypes.data_as(C.POINTER(C.c_float)), len(points),
                   votes.ctypes.data_as(C.POINTER(C.c_int32)))
        if n < 0:
            raise RuntimeError(self.lib.sn_occupancy_error().decode())
        return votes

    def project(self, floor, support, origin, config, stamp, *, endpoints, view_metadata=None,
                near_field=None, transient_mask=None, device=None, metrics=None):
        """Project resident evidence without exporting the entire 3D voxel map."""
        from .layers import finish_evidence
        if not hasattr(self.lib,'sn_occupancy_project'):
            raise RuntimeError('Rebuild the environment library for resident evidence projection')
        fp64, u8, i32 = C.POINTER(C.c_double),C.POINTER(C.c_uint8),C.POINTER(C.c_int32)
        method = self.lib.sn_occupancy_project
        method.argtypes = [C.c_void_p,fp64,u8,i32,C.c_int,C.c_int,fp64,fp64,C.c_double,fp64,C.c_int,fp64]
        original = np.asarray(floor)
        from .geometry import voxel_height_band
        _, required = voxel_height_band(original, config['resolution'], config['min_obstacle_height_m'], config['max_height_m'])
        required = np.maximum(1, required)
        if self.backend=='cuda':
            from .cuda_view import native_view
            layers,self.view_metadata=native_view(self,original,support,origin,config,stamp,endpoints,
                required=required,metadata=view_metadata,policy=near_field,transient_mask=transient_mask,
                device=device,metrics=metrics)
            return layers
        if device is not None:raise ValueError('Device layers require CUDA evidence')
        floor = np.ascontiguousarray(original,np.float64)
        support = np.ascontiguousarray(support,np.uint8)
        origin = np.ascontiguousarray(origin,np.float64)
        endpoints = np.ascontiguousarray(endpoints,np.float64)
        params = np.array([config[k] for k in ('resolution','min_obstacle_height_m','max_height_m',
             'occupied_probability','free_probability','free_vertical_coverage')]+[original.dtype==np.float32],np.float64)
        out = np.empty((6,*floor.shape),np.float64)
        n = method(self.ptr,floor.ctypes.data_as(fp64),support.ctypes.data_as(u8),required.ctypes.data_as(i32),
            *floor.shape,origin.ctypes.data_as(fp64),params.ctypes.data_as(fp64),stamp,
            endpoints.ctypes.data_as(fp64),len(endpoints),out.ctypes.data_as(fp64))
        if n<0:
            raise RuntimeError(self.lib.sn_occupancy_error().decode())
        bottom,top = out[4],out[5]
        bottom[~np.isfinite(bottom)],top[~np.isfinite(top)] = np.nan,np.nan
        self.person_only=None
        if transient_mask is not None:
            if not hasattr(self.lib,'sn_occupancy_person_columns'):
                raise RuntimeError('Rebuild the environment library for exclusive person evidence')
            method=self.lib.sn_occupancy_person_columns
            method.argtypes=[C.c_void_p,fp64,C.c_int,C.c_int,fp64,fp64,C.c_double,fp64,u8,C.c_int,u8]
            mask=np.ascontiguousarray(transient_mask,np.uint8);columns=np.zeros(floor.shape,np.uint8)
            if method(self.ptr,floor.ctypes.data_as(fp64),*floor.shape,origin.ctypes.data_as(fp64),params.ctypes.data_as(fp64),stamp,
                    endpoints.ctypes.data_as(fp64),mask.ctypes.data_as(u8),len(endpoints),columns.ctypes.data_as(u8))<0:
                raise RuntimeError(self.lib.sn_occupancy_error().decode())
            self.person_only=columns.astype(bool)
        return finish_evidence(out[0].astype(np.int8),out[1].astype(bool),out[2],out[3].astype(np.float32),
                               bottom,top,floor,support.astype(bool),config['resolution'])
