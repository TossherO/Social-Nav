"""Batched free-sweep certificates, with the existing exact CPU fallback."""
import ctypes as C
from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
from .footprint import swept_check,wrap

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY=ROOT/'outputs/navigation/build/libsocial_nav_collision_cuda.so'


def build_cuda(arch):
    if not str(arch).isdigit() or not 50<=int(arch)<=130:raise ValueError('Specify the target CUDA architecture explicitly')
    compiler=os.environ.get('CUDACXX') or shutil.which('nvcc')
    if not compiler:raise RuntimeError('Set CUDACXX to the target CUDA nvcc')
    DEFAULT_LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    host=['-ccbin',os.environ['CUDAHOSTCXX']] if os.environ.get('CUDAHOSTCXX') else []
    subprocess.run([compiler,*host,'-O3','--shared','-Xcompiler','-fPIC','-std=c++14','-arch=sm_'+str(arch),
        str(Path(__file__).with_name('native')/'collision.cu'),'-o',str(DEFAULT_LIBRARY)],check=True)
    return str(DEFAULT_LIBRARY)


class SweepChecker:
    """Own one validated snapshot cache per planner/safety process.

    A complete batch of free polygon enclosures certifies the sweep. Any
    inconclusive enclosure uses the unchanged adaptive CPU checker, so an
    obstacle near an enclosure does not itself become a new rejection.
    """
    def __init__(self,backend='cpu',library=None):
        if backend not in ('cpu','cuda'):raise ValueError('Unknown collision backend')
        self.backend=backend;self.library=library or str(DEFAULT_LIBRARY)
        self.snapshot=None;self.binding=None;self.ptr=None;self.lib=None;self.integral=None;self.device_grid=None;self.metrics=None

    def bind_device(self,snapshot,tensor,*,binding):
        import torch
        if self.backend!='cuda' or not tensor.is_cuda or tensor.dtype!=torch.int8 or not tensor.is_contiguous() or tuple(tensor.shape)!=tuple(snapshot.metadata['shape']):
            raise ValueError('Invalid resident collision map')
        self.warmup()
        if not hasattr(self.lib,'sn_collision_device_grid'):raise RuntimeError('Build resident collision C ABI')
        method=self.lib.sn_collision_device_grid;method.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int]
        if method(self.ptr,C.c_void_p(tensor.data_ptr()),*tensor.shape)<0:raise RuntimeError(self.lib.sn_collision_error().decode())
        self.snapshot,self.binding,self.device_grid=snapshot,binding,tensor
        self.device_stream=torch.cuda.current_stream(tensor.device).cuda_stream

    def warmup(self):
        if self.backend!='cuda' or self.ptr is not None:return
        lib=C.CDLL(self.library);self.lib=lib
        lib.sn_collision_create.restype=C.c_void_p
        lib.sn_collision_error.restype=C.c_char_p
        lib.sn_collision_destroy.argtypes=[C.c_void_p]
        lib.sn_collision_grid.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int]
        lib.sn_collision_batch.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double,C.c_void_p]
        if hasattr(lib,'sn_collision_batch_stream'):lib.sn_collision_batch_stream.argtypes=lib.sn_collision_batch.argtypes+[C.c_void_p]
        self.ptr=lib.sn_collision_create()
        if not self.ptr:raise RuntimeError(lib.sn_collision_error().decode())

    def close(self):
        if getattr(self,'ptr',None) is not None:self.lib.sn_collision_destroy(self.ptr);self.ptr=None

    def __del__(self):
        self.close()

    def _grid(self,snapshot,binding):
        if self.snapshot is snapshot and self.binding==binding:return
        self.device_grid=None
        cells=np.ascontiguousarray(snapshot.layers['occupancy'],np.int8)
        if self.backend=='cuda':
            self.warmup()
            if self.lib.sn_collision_grid(self.ptr,C.c_void_p(cells.ctypes.data),*cells.shape)<0:
                raise RuntimeError(self.lib.sn_collision_error().decode())
        else:
            self.integral=np.pad((cells!=0).astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
        self.snapshot=snapshot;self.binding=binding

    def check(self,snapshot,geometry,poses,margin,max_samples,*,binding):
        poses=np.asarray(poses,float)
        if len(poses)<2 or len(poses)>max_samples:
            return swept_check(snapshot,geometry,poses,margin,max_samples,prefix=False)
        self._grid(snapshot,binding)
        turn=wrap(np.diff(poses[:,2]));delta=np.diff(poses[:,:2],axis=0)
        mids=np.c_[(poses[:-1,:2]+poses[1:,:2])/2,poses[:-1,2]+turn/2]
        centres=np.vstack([poses[0],mids]);c,s=np.cos(centres[:,2]),np.sin(centres[:,2])
        fp=geometry.footprint
        polygons=np.empty((len(centres),len(fp),2))
        polygons[:,:,0]=centres[:,0,None]+c[:,None]*fp[:,0]-s[:,None]*fp[:,1]
        polygons[:,:,1]=centres[:,1,None]+s[:,None]*fp[:,0]+c[:,None]*fp[:,1]
        margins=margin+np.r_[0.,(np.linalg.norm(delta,axis=1)+geometry.radius*abs(turn))/2]
        if self.backend=='cuda':
            status=np.empty(len(polygons),np.int32);polygons=np.ascontiguousarray(polygons)
            origin=snapshot.metadata['origin_xy'];r=float(snapshot.metadata['resolution'])
            arguments=[self.ptr,C.c_void_p(polygons.ctypes.data),C.c_void_p(margins.ctypes.data),len(polygons),len(fp),*map(float,origin),r,C.c_void_p(status.ctypes.data)]
            if self.device_grid is not None:
                result=self.lib.sn_collision_batch_stream(*arguments,C.c_void_p(self.device_stream))
            else:result=self.lib.sn_collision_batch(*arguments)
            if result<0:raise RuntimeError(self.lib.sn_collision_error().decode())
            if self.metrics is not None:
                self.metrics.transfer('h2d',polygons.nbytes+margins.nbytes,'selected_sweep_enclosures')
                self.metrics.transfer('d2h',status.nbytes,'selected_sweep_certificates')
            free=not status.any()
        else:
            r=snapshot.metadata['resolution'];origin=np.asarray(snapshot.metadata['origin_xy'])
            low=np.floor((polygons.min(1)-margins[:,None]-origin)/r).astype(int)
            high=np.floor((polygons.max(1)+margins[:,None]-origin)/r).astype(int)
            shape=np.array(snapshot.layers['occupancy'].shape)[::-1]
            free=bool((low>=0).all() and (high<shape).all())
            if free:
                x,y=low.T;xx,yy=(high+1).T;v=self.integral
                free=not np.any(v[yy,xx]-v[y,xx]-v[yy,x]+v[y,x])
        if free:
            return dict(status='free',checked_samples=len(polygons),valid_distance_m=float(np.linalg.norm(delta,axis=1).sum()),
                last_valid_pose=poses[-1].tolist(),batch_certificate=self.backend)
        if self.metrics is not None:self.metrics.event('exact_cpu_adaptive_sweep_fallback')
        result=swept_check(snapshot,geometry,poses,margin,max_samples,prefix=False)
        return dict(result,batch_certificate=self.backend,adaptive_fallback=True)
