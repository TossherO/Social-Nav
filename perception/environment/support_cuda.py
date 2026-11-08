"""Stream-ordered 2D support kernels using cached, immutable static layers."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def library(path):
    lib=C.CDLL(str(path))
    capability=getattr(lib,'sn_occupancy_local_ground_conflicts',None)
    if capability is None or capability()!=1:
        raise RuntimeError('Rebuild environment CUDA library: local ground conflicts are required')
    lib.sn_navigation_support_with_ground_conflicts.argtypes=[C.c_void_p]*10+[C.c_int,C.c_void_p]+[C.c_int]*6+[C.c_void_p,C.c_int]
    lib.sn_occupancy_error.restype=C.c_char_p
    return lib


def project(snapshot,layers,asset,prior,plane,proof,observation,previous,origin,metrics,*,library_path=None):
    import torch
    m=snapshot.metadata;pm=asset.snapshot.metadata;device=prior.device;shape=tuple(m['shape'])
    transform=np.asarray(m['association']['map_from_local'])
    affine=transform[:2,:2]
    extent=np.abs(affine*m['resolution']/2).sum(axis=1)+np.abs(transform[:2,2])*np.diff(proof['height_band'])[0]/2
    if np.prod(np.ceil(2*extent/pm['resolution']).astype(int)+1)>81:
        raise ValueError('Navigation prior projection exceeds overlap budget')
    delta=np.zeros(2) if previous is None else (origin-np.asarray(m['origin_xy']))/m['resolution']
    if not np.allclose(delta,np.rint(delta),rtol=0,atol=1e-6):raise ValueError('Support grid lattice changed')
    params=np.r_[m['resolution'],pm['resolution'],m['origin_xy'],pm['origin_xy'],
        np.asarray(m['world_from_lidar'])[:2,3],proof['radius_m'],m['stamp_s'],proof['height_band'],
        proof['ground_tolerance_m'],proof['obstacle_retention_s'],proof['person_retention_s'],plane,transform[:3].ravel()].astype(np.float64)
    points=torch.from_numpy(observation['endpoints']).to(device)
    moving=torch.from_numpy(observation['transient_mask']).to(device)
    hits=torch.empty((2,*shape),dtype=torch.float64,device=device)
    output=torch.empty((3,*shape),dtype=torch.float64,device=device)
    allowed=torch.empty(shape,dtype=torch.uint8,device=device)
    buffers=[layers['occupancy'],layers['last_observed'],prior,previous,hits,output,allowed,points,moving,layers.get('ground_conflict')]
    stream=torch.cuda.current_stream(device)
    native=library(library_path or Path(__file__).resolve().parents[2]/'outputs/perception/ros1_ws/devel/lib/libsocial_nav_occupancy_cuda.so')
    result=native.sn_navigation_support_with_ground_conflicts(*[None if t is None else t.data_ptr() for t in buffers],len(points),params.ctypes.data,
        *shape,*pm['shape'],*np.rint(delta).astype(int),stream.cuda_stream,device.index)
    if result<0:raise RuntimeError(native.sn_occupancy_error().decode())
    for tensor in buffers:
        if tensor is not None:tensor.record_stream(stream)
    if metrics is not None:
        metrics.transfer('h2d',observation['endpoints'].nbytes+observation['transient_mask'].nbytes,'navigation_current_endpoints')
    return output[0].to(torch.int8),output[1].to(torch.uint8),output[2],hits
