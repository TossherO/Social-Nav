"""Planar differential-drive model quantities, never measured chassis dynamics."""
import numpy as np


def turning_residual(velocity,motion):
    """Curvature restriction for arc-only vehicles; differential spin has none.

    Speed, angular speed and acceleration limits are checked separately. The
    independent v/w domain also admits continuous bounded transitions to spin.
    """
    value=np.asarray(velocity,float)
    if motion.get('allow_in_place_rotation',False):return np.zeros(value.shape[:-1])
    v=abs(value[...,0]) if motion['allow_reverse'] else value[...,0]
    return v/motion['min_turning_radius_m']-abs(value[...,1])


def bound_target(velocity,motion):
    """Reduce a requested turn without inventing forward motion or reversing."""
    value=np.asarray(velocity,float);v,w=value[...,0],value[...,1]
    v=np.clip(v,-motion['max_reverse_mps'] if motion['allow_reverse'] else 0.,motion['max_forward_mps'])
    limit=motion['max_angular_rps'] if motion.get('allow_in_place_rotation',False) else np.minimum(motion['max_angular_rps'],abs(v)/motion['min_turning_radius_m'])
    return np.stack([v,np.clip(w,-limit,limit)],axis=-1)


def velocity_step(velocity,target,motion,dt):
    """Common interpolation preserves the cone and both acceleration limits.

    In particular braking scales v and omega together, rather than allowing a
    residual spin when the linear component reaches zero first.
    """
    v=np.asarray(velocity,float);target=np.broadcast_to(np.asarray(target,float),v.shape)
    # The two signed velocity cones are not jointly convex. Brake in the
    # current cone before accelerating in the other, including before zero.
    target=np.where((v[...,0]*target[...,0]<0)[...,None],0.,target)
    limits=np.array([motion['max_acceleration_mps2'],motion['max_angular_acceleration_rps2']])
    delta=target-v
    fraction=np.minimum(1.,np.min(limits*dt/np.maximum(abs(delta),1e-15),axis=-1))
    return v+fraction[...,None]*delta


def advance(pose,velocity,dt):
    pose=np.asarray(pose);value=np.asarray(velocity,float);v,w=value[...,0],value[...,1]
    angle=w*dt;heading=pose[...,2]+angle/2
    length=v*dt*np.sinc(angle/(2*np.pi))
    return pose+np.stack([length*np.cos(heading),length*np.sin(heading),angle],axis=-1)


def sample_trajectory(points,times):
    """Vectorized piecewise-SE(2) samples and body rates, without extrapolation."""
    from navigation.planning.motion import twists
    points=np.asarray(points,float);times=np.atleast_1d(times).astype(float)
    durations=np.diff(points[:,0]);rates=twists(points[:,1:],durations)
    clipped=np.clip(times,points[0,0],points[-1,0])
    index=np.clip(np.searchsorted(points[:,0],clipped,side='right')-1,0,len(points)-2)
    elapsed=clipped-points[index,0];rate=rates[index];angle=rate[:,2]*elapsed
    A=elapsed*np.sinc(angle/np.pi);B=elapsed*angle/2*np.sinc(angle/(2*np.pi))**2
    local=np.c_[A*rate[:,0]-B*rate[:,1],B*rate[:,0]+A*rate[:,1]]
    pose=points[index,1:].copy();c,s=np.cos(pose[:,2]),np.sin(pose[:,2])
    pose[:,:2]+=np.c_[c*local[:,0]-s*local[:,1],s*local[:,0]+c*local[:,1]];pose[:,2]+=angle
    return pose,rate


def ramp_path(pose,velocity,target,motion,duration,dt=.1,*,timed=False):
    """Piecewise midpoint integration of bounded acceleration to a velocity target."""
    poses=[np.asarray(pose,float)];velocity=np.asarray(velocity,float)
    n=max(1,int(np.ceil(duration/dt)))
    for step in np.full(n,duration/n):
        after=velocity_step(velocity,target,motion,step)
        poses.append(advance(poses[-1],(velocity+after)/2,step));velocity=after
    return (np.c_[np.linspace(0,duration,n+1),poses] if timed else np.asarray(poses)),velocity
