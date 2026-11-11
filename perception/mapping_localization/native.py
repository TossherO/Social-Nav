"""ROS-free ctypes boundary for the owned FAST-LIO numerical core."""
import ctypes as C
from pathlib import Path
import numpy as np


class Result(C.Structure):
    _fields_=[('stamp',C.c_double),('world_from_lidar',C.c_double*16),('velocity',C.c_double*3),
        ('gyro_bias',C.c_double*3),('acc_bias',C.c_double*3),('residual',C.c_double),
        ('processing_ms',C.c_double),('matching_ms',C.c_double),('map_update_ms',C.c_double),
        ('initial_lidar_pose',C.c_double*16),('reference_stamp',C.c_double),
        ('valid',C.c_int),('reason',C.c_int),('effective_points',C.c_int),('scan_points',C.c_int),('deskewed_points',C.c_int)]


class Acceleration(C.Structure):
    _fields_=[(k,C.c_uint64) for k in ('map_points','reference_points','h2d_bytes','d2h_bytes','nn_fallbacks','plane_fallbacks')]+[
        (k,C.c_double) for k in ('map_sync_ms','match_wall_ms','query_device_ms')]+[('backend',C.c_int)]


class LioStages(C.Structure):
    _fields_=[(k,C.c_double) for k in ('imu_prepare_ms','imu_predict_ms','deskew_ms','downsample_ms','jacobian_ms','filter_solve_ms',
        'deskew_device_ms','linearize_device_ms')]+[(k,C.c_uint64) for k in ('deskew_h2d_bytes','deskew_d2h_bytes')]


class RollStages(C.Structure):
    _fields_=[(k,C.c_double) for k in ('feature_ms','submap_ms','downsample_ms','matching_ms','graph_ms','corner_ms','surface_ms',
        'linear_solve_ms','map_upload_ms','query_device_ms')]+[(k,C.c_uint64) for k in ('h2d_bytes','d2h_bytes','fallbacks')]+[
        (k,C.c_int) for k in ('backend','iterations')]


class NativeFrontend:
    def __init__(self,config,library=None):
        root=Path(__file__).resolve().parents[2]
        path=Path(library or root/'outputs/runtime/build/localization/libsocial_nav_lio.so')
        self.lib=C.CDLL(str(path));lib=self.lib
        lib.sn_lio_error.restype=C.c_char_p
        lib.sn_lio_create.argtypes=[C.c_char_p];lib.sn_lio_create.restype=C.c_void_p
        lib.sn_lio_destroy.argtypes=[C.c_void_p]
        lib.sn_lio_step.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_double,C.c_double,C.POINTER(Result)]
        lib.sn_lio_cloud.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t]
        if hasattr(lib,'sn_lio_acceleration'):
            lib.sn_lio_acceleration.argtypes=[C.c_void_p,C.POINTER(Acceleration)]
        if hasattr(lib,'sn_lio_stages'):lib.sn_lio_stages.argtypes=[C.c_void_p,C.POINTER(LioStages)]
        self.ptr=lib.sn_lio_create(str(config).encode())
        if not self.ptr:raise RuntimeError(lib.sn_lio_error().decode())

    def step(self,imu,points,begin,end):
        if not self.ptr:raise RuntimeError('Native frontend is closed')
        imu=np.ascontiguousarray(imu,np.float64);points=np.ascontiguousarray(points,np.float32)
        if imu.ndim!=2 or imu.shape[1]!=7 or points.ndim!=2 or points.shape[1]!=5 or not np.isfinite(imu).all() or not np.isfinite(points).all():
            raise ValueError('Native LIO expects finite IMU [t,ax,ay,az,gx,gy,gz] and cloud [x,y,z,i,offset_ms]')
        out=Result()
        if self.lib.sn_lio_step(self.ptr,imu.ctypes.data,len(imu),points.ctypes.data,len(points),begin,end,C.byref(out))<0:
            raise RuntimeError(self.lib.sn_lio_error().decode())
        cloud=np.empty((out.deskewed_points,5),np.float32)
        if len(cloud) and self.lib.sn_lio_cloud(self.ptr,cloud.ctypes.data,len(cloud))!=len(cloud):
            raise RuntimeError('Native deskewed scan changed before consumption')
        pose=np.array(out.world_from_lidar).reshape(4,4)
        acceleration=None
        if hasattr(self.lib,'sn_lio_acceleration'):
            a=Acceleration();self.lib.sn_lio_acceleration(self.ptr,C.byref(a))
            acceleration={k:getattr(a,k) for k,_ in a._fields_};acceleration['backend']='cuda' if a.backend else 'cpu'
        stages=None
        if hasattr(self.lib,'sn_lio_stages'):
            s=LioStages();self.lib.sn_lio_stages(self.ptr,C.byref(s));stages={k:getattr(s,k) for k,_ in s._fields_}
        return dict(stamp_s=out.stamp,world_from_lidar=pose,valid=bool(out.valid),reason=out.reason,
            velocity=np.array(out.velocity),gyro_bias=np.array(out.gyro_bias),acc_bias=np.array(out.acc_bias),
            residual=out.residual,effective_points=out.effective_points,scan_points=out.scan_points,
            initial_lidar_pose=np.array(out.initial_lidar_pose).reshape(4,4),reference_stamp=out.reference_stamp,
            processing_ms=out.processing_ms,matching_ms=out.matching_ms,map_update_ms=out.map_update_ms,acceleration=acceleration,stages=stages),cloud

    def close(self):
        if self.ptr:self.lib.sn_lio_destroy(self.ptr);self.ptr=None


def rs16_preprocess(fields,config,stamp):
    """The deployed type-4, feature-disabled RS16 timestamped scan convention.

    Keep original point indices when applying point_filter_num, as the ROS core
    does. Unsupported feature/timestamp modes fail explicitly.
    """
    pre=config['preprocess']
    if pre['lidar_type']!=4 or pre['scan_line']!=16 or config['feature_extract_enable']:
        raise ValueError('Native RS16 adapter requires deployed type 4 / 16 beams / no feature extraction')
    xyz=np.column_stack([fields[k] for k in ('x','y','z')]).astype(np.float32)
    times=np.asarray(fields['timestamp'],np.float64).copy()
    if len(times)<2:raise ValueError('Empty RS16 scan')
    if times[0]>1000:times-=times[0]
    if not times[-1]>1e-7:raise ValueError('Native RS16 adapter requires measured point timestamps')
    xy=np.sqrt(xyz[:,0]*xyz[:,0]+xyz[:,1]*xyz[:,1])
    with np.errstate(invalid='ignore',divide='ignore'):
        angle=(np.arctan(xyz[:,2]/xy)*180/np.pi).astype(np.float32)
        ring=np.trunc((angle+15)/2+.5)
    ring[~np.isfinite(ring)]=-1
    valid=np.isfinite(xyz).all(axis=1)&np.isfinite(times)&(xy>=1e-6)&(ring>=0)&(ring<16)
    valid&=(np.arange(len(xyz))%config['point_filter_num']==0)&((xyz*xyz).sum(axis=1)>pre['blind']**2)
    points=np.column_stack([xyz,fields['intensity'],times*1000]).astype(np.float32)[valid]
    if len(points)<2:raise ValueError('No usable RS16 points')
    duration=float(points[-1,4])/1000
    end=float(stamp) if pre.get('stamp_at_scan_end',True) else float(stamp)+duration
    return points,end-duration,end


class NativeRoll:
    def __init__(self,config,library=None):
        root=Path(__file__).resolve().parents[2]
        self.lib=C.CDLL(str(library or root/'outputs/runtime/build/localization/libsocial_nav_roll.so'))
        lib=self.lib;lib.sn_roll_error.restype=C.c_char_p
        lib.sn_roll_create.argtypes=[C.c_char_p];lib.sn_roll_create.restype=C.c_void_p
        lib.sn_roll_destroy.argtypes=[C.c_void_p]
        lib.sn_roll_step.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,C.c_double,C.c_void_p]
        lib.sn_roll_save.argtypes=[C.c_void_p,C.c_char_p]
        if hasattr(lib,'sn_roll_stages'):lib.sn_roll_stages.argtypes=[C.c_void_p,C.POINTER(RollStages)]
        if hasattr(lib,'sn_roll_gate_diagnostics'):lib.sn_roll_gate_diagnostics.argtypes=[C.c_void_p,C.c_void_p]
        self.ptr=lib.sn_roll_create(str(config).encode())
        if not self.ptr:raise RuntimeError(lib.sn_roll_error().decode())

    def step(self,cloud,pose,reference,stamp):
        if not self.ptr:raise RuntimeError('Native ROLL is closed')
        cloud=np.ascontiguousarray(cloud,np.float32);pose=np.ascontiguousarray(pose,np.float64)
        reference=np.ascontiguousarray(reference,np.float64)
        if cloud.ndim!=2 or cloud.shape[1]!=4 or pose.shape!=(4,4) or reference.shape!=(4,4):
            raise ValueError('Invalid native ROLL scan/pose/reference')
        out=np.empty(19,np.float64)
        valid=self.lib.sn_roll_step(self.ptr,cloud.ctypes.data,len(cloud),pose.ctypes.data,reference.ctypes.data,float(stamp),out.ctypes.data)
        if valid<0:raise RuntimeError(self.lib.sn_roll_error().decode())
        stages=None
        if hasattr(self.lib,'sn_roll_stages'):
            s=RollStages();self.lib.sn_roll_stages(self.ptr,C.byref(s));stages={k:getattr(s,k) for k,_ in s._fields_}
        diagnostics=None
        if valid!=2 and hasattr(self.lib,'sn_roll_gate_diagnostics'):
            d=np.empty(5,np.float64);self.lib.sn_roll_gate_diagnostics(self.ptr,d.ctypes.data)
            diagnostics=dict(rejection_mask=int(d[0]),**{key:(float(value) if np.isfinite(value) else None)
                for key,value in zip(('candidate_clearance_m','correction_step_m','correction_angle_rad','gravity_angle_rad'),d[1:])})
        return dict(map_from_lidar=out[:16].reshape(4,4),available=valid!=2,valid=valid==1,inlier_ratio=float(out[16]),registration_error=float(out[17]),map_revision=int(out[18]),stages=stages,gate_diagnostics=diagnostics)

    def save(self,directory):
        # The shared legacy writer expects a newly created empty destination.
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
        if self.lib.sn_roll_save(self.ptr,str(directory).encode())<0:raise RuntimeError(self.lib.sn_roll_error().decode())

    def close(self):
        if self.ptr:self.lib.sn_roll_destroy(self.ptr);self.ptr=None
