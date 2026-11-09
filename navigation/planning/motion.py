"""Explicit offline motion assumptions and causal velocity from algorithm poses."""
from pathlib import Path
import json
import numpy as np
import yaml
from perception.environment.context import fingerprint
from perception.environment.storage import content_hash
from perception.mapping_localization.context import pose_matrix
from .backend import ROOT
from .footprint import wrap


def load_motion_config(path=None, *, profile=None):
    default=Path(__file__).with_name('configs')/'motion.yaml'
    source=Path(path or default)
    cfg=yaml.safe_load(source.read_text())
    if not isinstance(cfg,dict):raise ValueError('Motion configuration must be a mapping')
    if 'profiles' in cfg or 'common' in cfg or 'default_profile' in cfg:
        if set(cfg)!={'default_profile','common','profiles'} or not isinstance(cfg['common'],dict) or not isinstance(cfg['profiles'],dict):
            raise ValueError('Motion profiles require default_profile, common and profiles')
        name=cfg['default_profile'] if profile is None else profile
        if not isinstance(name,str) or name not in cfg['profiles'] or not isinstance(cfg['profiles'][name],dict):
            raise ValueError('Unknown motion profile: '+str(name))
        selected=cfg['profiles'][name]
        required={'max_forward_mps','max_reverse_mps','max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2'}
        if not required<=selected.keys() or set(selected)&set(cfg['common']):
            raise ValueError('Motion profile must declare all limits without overriding common settings')
        cfg=dict(cfg['common'],**selected)
    elif profile is not None:
        raise ValueError('A flat motion configuration does not define profiles')
    else:
        defaults=yaml.safe_load(default.read_text())['common']
        for key in ('stationary_min_window_s','stationary_position_span_m','stationary_yaw_span_rad',
                    'stationary_max_linear_mps','stationary_max_angular_rps'):
            cfg.setdefault(key,defaults[key])
    if cfg.get('performance_reference'):
        source=Path(cfg['performance_reference']);source=source if source.is_absolute() else ROOT/source
        reference=json.loads(source.read_text());profile=reference['profiles'][cfg['performance_profile']]['fields']
        names=dict(max_forward_mps='max_vel_x',max_reverse_mps='max_vel_x_backwards',max_angular_rps='max_vel_theta',
            max_acceleration_mps2='acc_lim_x',max_angular_acceleration_rps2='acc_lim_theta')
        for target,key in names.items():cfg.setdefault(target,profile[key])
        cfg['parameter_provenance']=dict(path=str(source),sha256=content_hash(source),source_image=reference['source_image'],
            status=reference['status'],profile=cfg['performance_profile'])
    cfg.setdefault('allow_in_place_rotation',False)
    if cfg['model']!='differential' or any(type(cfg[k]) is not bool for k in ('parameters_assumed','allow_reverse','allow_in_place_rotation')):
        raise ValueError('Explicit differential model and assumption/reverse flags are required')
    if cfg.get('collision_backend','cpu') not in ('cpu','cuda'):raise ValueError('Unknown collision backend')
    for k,v in cfg.items():
        if k not in ('model','parameters_assumed','allow_reverse','allow_in_place_rotation','performance_reference','performance_profile','parameter_provenance','collision_backend','collision_library'):
            if not np.isfinite(v) or v<=0:raise ValueError('Invalid motion setting: '+k)
    if cfg['state_window_s']<cfg['state_max_gap_s']:raise ValueError('Invalid state window')
    if cfg['stationary_min_window_s']>cfg['state_window_s']:raise ValueError('Stationary evidence exceeds pose window')
    if cfg['brake_deceleration_scale']>1:raise ValueError('Braking assumption cannot exceed nominal acceleration')
    if 'collision_library' in cfg:
        value=Path(cfg['collision_library']);cfg['collision_library']=str(value if value.is_absolute() else ROOT/value)
    return cfg


def identity(scene):
    r=scene.record
    return [r['session_id'],r['generation'],r['frame_id']]


def estimate_motion(rows,scene,geometry,config):
    """Past-only median interval twists; an invalid boundary ends the usable history."""
    cutoff=scene.record['as_of'];key=identity(scene);selected=[];reason='insufficient_pose_history'
    for row in reversed(rows):
        if row['stamp_s']>cutoff+1e-7:continue
        if cutoff-row['stamp_s']>config['state_window_s']:break
        if row['health']!=0 or [row['session_id'],row['generation'],row['frame_id']]!=key:break
        selected.append(row)
    selected.reverse()
    if len(selected)>=2:
        times=np.array([r['stamp_s'] for r in selected]);dt=np.diff(times)
        if (dt<=0).any() or (dt>config['state_max_gap_s']).any():reason='pose_history_gap'
        elif cutoff-times[-1]>config['state_max_age_s']:reason='motion_state_stale'
        else:
            poses=np.array([geometry.pose(pose_matrix(r['position'],r['quaternion_xyzw'])) for r in selected])
            rates=twists(poses,dt)
            velocity=np.median(rates,axis=0)
            result=dict(status='valid',source='causal_algorithm_pose_intervals',stamp_s=float(times[-1]),
                local_identity=key,clock_domain=scene.record['clock_domain'],velocity=velocity.tolist(),
                window_s=[float(times[0]),float(times[-1])],samples=len(selected),
                input_fingerprint=fingerprint(selected),assumed=False)
            span=float(times[-1]-times[0])
            position_span=float(np.linalg.norm(poses[:,:2]-poses[0,:2],axis=1).max())
            angles=np.unwrap(poses[:,2]);yaw_span=float(np.ptp(angles))
            linear=float(np.linalg.norm(rates[:,:2],axis=1).max());angular=float(abs(rates[:,2]).max())
            if (len(selected)>=3 and span>=config['stationary_min_window_s']-1e-7 and
                    position_span<=config['stationary_position_span_m'] and yaw_span<=config['stationary_yaw_span_rad'] and
                    linear<=config['stationary_max_linear_mps'] and angular<=config['stationary_max_angular_rps']):
                result.update(velocity_raw=velocity.tolist(),velocity=[0.,0.,0.],
                    stationary_evidence=dict(model='bounded_past_pose_noise',window_s=span,position_span_m=position_span,
                        yaw_span_rad=yaw_span,linear_bound_mps=max(.01,linear),angular_bound_rps=max(.01,angular)))
            return result
    return dict(status='unavailable',reason=reason,local_identity=key,clock_domain=scene.record['clock_domain'])


def motion_uncertainty_margin(state,geometry,horizon_s):
    """Conservative extra envelope for a supported zero nominal estimate only."""
    evidence=state.get('stationary_evidence')
    if evidence is None:return 0.
    values=np.asarray([evidence[k] for k in ('position_span_m','yaw_span_rad','linear_bound_mps','angular_bound_rps')],float)
    if not np.isfinite(values).all() or (values<0).any() or not np.isfinite(horizon_s) or horizon_s<0:
        raise ValueError('Invalid stationary uncertainty envelope')
    return float(evidence['position_span_m']+geometry.radius*evidence['yaw_span_rad']+
        horizon_s*(evidence['linear_bound_mps']+geometry.radius*evidence['angular_bound_rps']))


def twists(poses,dt):
    """SE(2) logarithm: body-frame vx, vy, omega on each constant-twist interval."""
    a=np.asarray(poses)[:-1];d=np.diff(np.asarray(poses)[:,:2],axis=0);angle=wrap(np.diff(np.asarray(poses)[:,2]))
    c,s=np.cos(a[:,2]),np.sin(a[:,2]);x=c*d[:,0]+s*d[:,1];y=-s*d[:,0]+c*d[:,1]
    A=np.sinc(angle/np.pi);B=angle/2*np.sinc(angle/(2*np.pi))**2
    denom=A*A+B*B
    return np.c_[(A*x+B*y)/denom/dt,(-B*x+A*y)/denom/dt,angle/dt]


def interpolate_se2(poses,resolution,limit):
    """Fine exponential interpolation and a bound on each chord's translation error."""
    poses=np.asarray(poses)
    if len(poses)==1:return poses.copy(),0.
    rates=twists(poses,np.ones(len(poses)-1));turn=wrap(np.diff(poses[:,2]));length=np.linalg.norm(rates[:,:2],axis=1)
    counts=np.maximum(1,np.ceil(np.maximum(length/(resolution/2),abs(turn)/.05)).astype(int))
    total=int(counts.sum())
    if total+1>limit:raise ValueError('trajectory_sampling_budget_exceeded')
    index=np.repeat(np.arange(len(counts)),counts)
    u=(np.arange(total)-np.repeat(np.cumsum(counts)-counts,counts)+1)/counts[index]
    theta=turn[index]*u;rate=rates[index];a=poses[index]
    A=u*np.sinc(theta/np.pi);B=u*theta/2*np.sinc(theta/(2*np.pi))**2
    x=A*rate[:,0]-B*rate[:,1];y=B*rate[:,0]+A*rate[:,1]
    c,s=np.cos(a[:,2]),np.sin(a[:,2]);out=np.empty((total+1,3));out[0]=poses[0]
    out[1:,0]=a[:,0]+c*x-s*y;out[1:,1]=a[:,1]+s*x+c*y;out[1:,2]=a[:,2]+theta
    return out,float(np.max(length/counts*abs(turn/counts)/8,initial=0.))
