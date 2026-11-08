"""Batched free-sweep certificates, with the existing exact CPU fallback."""
import ctypes as C
from pathlib import Path
import os
import shutil
import subprocess
import time
import numpy as np
from .footprint import swept_check,wrap

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY=ROOT/'outputs/navigation/build/libsocial_nav_collision_cuda.so'


class CertificationBudget:
    """One call shares a monotonic deadline and exact geometry-query allowance."""
    def __init__(self, deadline, max_checks):
        self.deadline=deadline;self.max_checks=max_checks;self.checked=0;self.reason=None

    def check(self, count=0):
        if self.reason is None:
            if time.perf_counter()>=self.deadline:self.reason='certification_time_budget_exceeded'
            elif self.checked+count>self.max_checks:self.reason='certification_work_budget_exceeded'
        if self.reason:return False
        self.checked+=count
        return True

    def summary(self):
        return dict(exact_checks=self.checked,max_exact_checks=self.max_checks,reason=self.reason)


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
    Inconclusive intervals use the adaptive CPU checker. Already certified
    intervals are reused within this call only, with identical map and margins.
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

    @staticmethod
    def _polygons(geometry,poses):
        c,s=np.cos(poses[:,2]),np.sin(poses[:,2]);fp=geometry.footprint
        polygons=np.empty((len(poses),len(fp),2))
        polygons[:,:,0]=poses[:,0,None]+c[:,None]*fp[:,0]-s[:,None]*fp[:,1]
        polygons[:,:,1]=poses[:,1,None]+s[:,None]*fp[:,0]+c[:,None]*fp[:,1]
        return polygons

    def _cuda_batch(self,snapshot,polygons,margins):
        polygons=np.ascontiguousarray(polygons);margins=np.ascontiguousarray(margins)
        status=np.empty(len(polygons),np.int32)
        arguments=[self.ptr,C.c_void_p(polygons.ctypes.data),C.c_void_p(margins.ctypes.data),
            len(polygons),polygons.shape[1],*map(float,snapshot.metadata['origin_xy']),
            float(snapshot.metadata['resolution']),C.c_void_p(status.ctypes.data)]
        if self.device_grid is not None:
            result=self.lib.sn_collision_batch_stream(*arguments,C.c_void_p(self.device_stream))
        else:result=self.lib.sn_collision_batch(*arguments)
        if result<0:raise RuntimeError(self.lib.sn_collision_error().decode())
        if self.metrics is not None:
            self.metrics.transfer('h2d',polygons.nbytes+margins.nbytes,'selected_sweep_enclosures')
            self.metrics.transfer('d2h',status.nbytes,'selected_sweep_certificates')
        return status

    def _refine(self,snapshot,geometry,poses,margin,certified,max_samples,budget):
        """Prove uncertain intervals in small GPU batches; never infer collision.

        Every root becomes free only after all its covering leaves are free.
        Unresolved roots retain the exact CPU fallback and original margins.
        """
        if not certified[0]:return 0,0
        roots=np.flatnonzero(~certified[1:]);a,b=poses[roots],poses[roots+1]
        batches=enclosures=0
        for _ in range(4):
            if not len(roots) or len(roots)*2>min(max_samples,20000):break
            if budget is not None and not budget.check():break
            turn=wrap(b[:,2]-a[:,2]);middle=np.c_[(a[:,:2]+b[:,:2])/2,a[:,2]+turn/2]
            left=np.stack((a,middle),axis=1).reshape(-1,3)
            right=np.stack((middle,b),axis=1).reshape(-1,3)
            ids=np.repeat(roots,2);turn=wrap(right[:,2]-left[:,2])
            centres=np.c_[(left[:,:2]+right[:,:2])/2,left[:,2]+turn/2]
            padding=margin+(np.linalg.norm(right[:,:2]-left[:,:2],axis=1)+geometry.radius*abs(turn))/2
            pending=self._cuda_batch(snapshot,self._polygons(geometry,centres),padding)!=0
            batches+=1;enclosures+=len(ids)
            unresolved=np.zeros(len(poses)-1,bool);unresolved[ids[pending]]=True
            completed=np.unique(roots);certified[completed[~unresolved[completed]]+1]=True
            roots,a,b=ids[pending],left[pending],right[pending]
        return batches,enclosures

    def check(self,snapshot,geometry,poses,margin,max_samples,*,binding,budget=None):
        began=time.perf_counter()
        if budget is not None and not budget.check():
            return dict(status='budget_exceeded',checked_samples=0,timing_ms=dict(enclosures=0.,cpu_fallback=0.))
        poses=np.asarray(poses,float)
        if len(poses)<2 or len(poses)>max_samples:
            return swept_check(snapshot,geometry,poses,margin,max_samples,prefix=False,budget=budget)
        self._grid(snapshot,binding)
        turn=wrap(np.diff(poses[:,2]));delta=np.diff(poses[:,:2],axis=0)
        mids=np.c_[(poses[:-1,:2]+poses[1:,:2])/2,poses[:-1,2]+turn/2]
        centres=np.vstack([poses[0],mids]);polygons=self._polygons(geometry,centres)
        margins=margin+np.r_[0.,(np.linalg.norm(delta,axis=1)+geometry.radius*abs(turn))/2]
        refinement_batches=refinement_enclosures=0
        if self.backend=='cuda':
            certified=self._cuda_batch(snapshot,polygons,margins)==0
            refinement_batches,refinement_enclosures=self._refine(snapshot,geometry,poses,margin,certified,max_samples,budget)
        else:
            r=snapshot.metadata['resolution'];origin=np.asarray(snapshot.metadata['origin_xy'])
            low=np.floor((polygons.min(1)-margins[:,None]-origin)/r).astype(int)
            high=np.floor((polygons.max(1)+margins[:,None]-origin)/r).astype(int)
            shape=np.array(snapshot.layers['occupancy'].shape)[::-1]
            certified=((low>=0)&(high<shape)).all(axis=1)
            valid=np.flatnonzero(certified)
            if len(valid):
                x,y=low[valid].T;xx,yy=(high[valid]+1).T;v=self.integral
                certified[valid]=(v[yy,xx]-v[y,xx]-v[yy,x]+v[y,x])==0
        enclosure_ms=(time.perf_counter()-began)*1000
        refinement=dict(gpu_refinement_batches=refinement_batches,gpu_refinement_enclosures=refinement_enclosures)
        if budget is not None and not budget.check():
            return dict(status='budget_exceeded',checked_samples=0,timing_ms=dict(enclosures=enclosure_ms,cpu_fallback=0.),**refinement)
        if certified.all():
            return dict(status='free',checked_samples=len(polygons),valid_distance_m=float(np.linalg.norm(delta,axis=1).sum()),
                last_valid_pose=poses[-1].tolist(),batch_certificate=self.backend,
                timing_ms=dict(enclosures=enclosure_ms,cpu_fallback=0.),reused_segments=int(certified[1:].sum()),**refinement)
        if self.metrics is not None:self.metrics.event('exact_cpu_adaptive_sweep_fallback')
        began=time.perf_counter()
        result=swept_check(snapshot,geometry,poses,margin,max_samples,prefix=False,certified=certified,budget=budget)
        return dict(result,batch_certificate=self.backend,adaptive_fallback=True,
            timing_ms=dict(enclosures=enclosure_ms,cpu_fallback=(time.perf_counter()-began)*1000),
            reused_segments=int(certified[1:].sum()),**refinement)
