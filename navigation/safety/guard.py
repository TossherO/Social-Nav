"""Independent simulation/shadow gate with current geometry and stopping sweep."""
from pathlib import Path
import numpy as np
import yaml
from perception.environment.context import fingerprint
from navigation.kinematics import ramp_path,braking_path,turning_residual,advance,velocity_step
from navigation.planning.motion import interpolate_se2,motion_uncertainty_margin
from navigation.planning.people import PeopleModel
from navigation.planning.inputs import request_status
from navigation.planning.collision import SweepChecker
from navigation.planning.footprint import footprint_status
from perception.environment.navigation import navigation_geometry


def stopping_trajectory(feedback,desired,motion,settings,dt,*,pending=None,braking_motion=None,observation_age_s=0.):
    """Shared hypothetical actuator delay, response and braking; not execution."""
    velocity=np.asarray(feedback['velocity'])
    if pending is None:pending=velocity
    if braking_motion is None:
        braking_motion=dict(motion)
        for key in ('max_acceleration_mps2','max_angular_acceleration_rps2'):
            braking_motion[key]*=motion['brake_deceleration_scale']
    delay,v=ramp_path(feedback['pose'],velocity,pending,motion,settings['actuator_delay_s']+observation_age_s,dt,timed=True)
    reaction,v=ramp_path(delay[-1,1:],v,desired,motion,settings['reaction_s']+dt,dt,timed=True)
    braking=braking_path(reaction[-1,1:],v,braking_motion,0.,timed=True)
    reaction[:,0]+=delay[-1,0];braking[:,0]+=reaction[-1,0]
    points=np.vstack([delay,reaction[1:],braking[1:]])
    return points[np.r_[True,np.diff(points[:,0])>1e-10]]


def stopping_trajectories(feedback,desired,motion,settings,dt,*,pending=None,observation_age_s=0.):
    """Same scalar stopping model for a batch, including unequal stop times.

    Braking scales both velocity components together. Its exact integrated arc
    is v0*T*(u-u*u/2), eliminating a Python loop for each candidate.
    """
    desired=np.asarray(desired,float);velocity=np.asarray(feedback['velocity'],float)
    pending=velocity if pending is None else pending
    delay,v=ramp_path(feedback['pose'],velocity,pending,motion,settings['actuator_delay_s']+observation_age_s,dt,timed=True)
    duration=settings['reaction_s']+dt;n=max(1,int(np.ceil(duration/dt)));step=duration/n
    poses=[np.broadcast_to(delay[-1,1:],(len(desired),3)).copy()]
    velocities=np.broadcast_to(v,desired.shape).copy()
    for _ in range(n):
        after=velocity_step(velocities,desired,motion,step)
        poses.append(advance(poses[-1],(velocities+after)/2,step));velocities=after
    poses=np.asarray(poses);times=np.arange(n+1)*step+delay[-1,0]
    limits=np.array([motion['max_acceleration_mps2'],motion['max_angular_acceleration_rps2']])*motion['brake_deceleration_scale']
    durations=np.max(abs(velocities)/limits,axis=1);counts=np.maximum(1,np.ceil(durations/dt).astype(int))
    indices=np.arange(counts.max()+1)[:,None];fraction=np.minimum(indices/counts[None],1.)
    braking=advance(poses[-1][None],velocities[None],durations[None]*(fraction-fraction**2/2))
    results=[]
    for j,count in enumerate(counts):
        response=np.c_[times,poses[:,j]]
        stop=np.c_[np.arange(count+1)*durations[j]/count+times[-1],braking[:count+1,j]]
        points=np.vstack([delay,response[1:],stop[1:]])
        results.append(points[np.r_[True,np.diff(points[:,0])>1e-10]])
    return results


class LocalSafety:
    def __init__(self,geometry,motion,margin,*,settings=None,people_settings=None,simulation=False,shadow=False,asset=None):
        if bool(simulation)==bool(shadow):raise ValueError('Choose explicit simulation or non-executing shadow mode')
        self.shadow=shadow
        self.asset=asset
        self.geometry=geometry;self.motion=motion;self.margin=margin
        self.settings=settings or yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        if any(not np.isfinite(v) or v<=0 for v in self.settings.values()):raise ValueError('Invalid safety setting')
        if not 0<motion['brake_deceleration_scale']<=1:raise ValueError('Invalid braking assumption')
        self.braking_motion=dict(motion)
        for key in ('max_acceleration_mps2','max_angular_acceleration_rps2'):
            self.braking_motion[key]*=motion['brake_deceleration_scale']
        self.last_target=np.zeros(2)
        if people_settings is None:
            from navigation.planning.inputs import load_config
            people_settings=load_config()['people']
        self.people_settings=people_settings
        self._people_cache=None
        self.sweep=SweepChecker(motion.get('collision_backend','cpu'),motion.get('collision_library'))
        if shadow:self.sweep.warmup()

    def _people(self,scene):
        # Called only after scene.verify(), including on cache hits. An ID alone
        # cannot certify unchanged source bytes or a changed prediction policy.
        binding=(scene.record['input_fingerprint'],fingerprint(self.people_settings))
        cached=self._people_cache
        if cached is None or cached[0] is not scene or cached[1]!=binding:
            cached=(scene,binding,PeopleModel(scene,self.people_settings))
            self._people_cache=cached
        return cached[2]

    def gate(self,proposal,request,scene,feedback,*,now,clock_domain,dt,emergency_stop=False,manual_override=False):
        result=dict(velocity=[0.,0.],state='braking',reason=None,executable=False,scope='shadow_safety' if self.shadow else 'simulation_safety',
            request_binding=fingerprint(request),scene_binding=scene.record['input_fingerprint'])
        # Deliberately independent of the controller's validity function.
        r=scene.record
        try:
            invalid=(feedback is None or feedback.get('status')!='valid' or not feedback.get('source') or
                feedback.get('local_identity')!=[r['session_id'],r['generation'],r['frame_id']] or
                feedback.get('clock_domain')!=clock_domain or np.shape(feedback.get('velocity'))!=(2,) or
                np.shape(feedback.get('pose'))!=(3,) or not np.isfinite(feedback['pose']).all() or
                not np.isfinite(feedback['velocity']).all() or not 0<dt<=.2 or
                not 0<=now-feedback.get('stamp_s',-np.inf)<=self.settings['feedback_max_age_s'])
        except (ValueError,TypeError):invalid=True
        if invalid:
            self.last_target=np.zeros(2)
            return dict(result,reason='feedback_invalid',state='stop_request')
        velocity=np.asarray(feedback['velocity'])
        ratio=self.motion.get('max_initial_profile_ratio',1.)
        reverse=self.motion['max_reverse_mps']*ratio if self.motion['allow_reverse'] else 0.
        if (not -reverse<=velocity[0]<=self.motion['max_forward_mps']*ratio or
                abs(velocity[1])>self.motion['max_angular_rps']*ratio or
                turning_residual(velocity,self.motion)<-1e-7):
            self.last_target=np.zeros(2)
            return dict(result,reason='feedback_outside_motion_model',state='stop_request')
        reason=None;desired=np.zeros(2)
        if emergency_stop:reason='emergency_stop'
        elif manual_override:reason='manual_override'
        elif request_status(request,scene,now,clock_domain):reason=request_status(request,scene,now,clock_domain)
        elif (clock_domain!=r['clock_domain'] or r['localization']['status']!='valid' or r['geometry']['status']!='valid' or
                not 0<=now-r['as_of']<=self.settings['max_input_age_s']):reason='observation_invalid_or_stale'
        elif proposal is None or proposal.get('status')!='candidate':reason='planner_unavailable'
        elif proposal.get('control_mode')!='dwa':reason='unsupported_control_mode'
        elif (proposal.get('request_binding')!=fingerprint(request) or proposal.get('scene_binding')!=r['input_fingerprint'] or
                not 0<=now-proposal.get('trajectory_as_of',-np.inf)<=self.settings['max_input_age_s']):reason='proposal_binding_or_age_invalid'
        else:
            desired=np.asarray(proposal.get('velocity'),float)
            if desired.shape!=(2,) or not np.isfinite(desired).all():desired=np.zeros(2);reason='invalid_velocity'
        if reason:desired=np.zeros(2)
        reverse=self.motion['max_reverse_mps'] if self.motion['allow_reverse'] else 0.
        desired=np.clip(desired,[-reverse,-self.motion['max_angular_rps']],
            [self.motion['max_forward_mps'],self.motion['max_angular_rps']])
        if not reason and turning_residual(desired,self.motion)<-1e-7:
            reason='turning_radius_violation'
        if reason:
            self.last_target=np.zeros(2)
            return dict(result,reason=reason,state='stop_request')
        try:
            from navigation.planning.inputs import validate_grid
            scene.verify();validate_grid(scene.geometry)
            contact_grid=navigation_geometry(scene,now=now,asset=self.asset,check_asset=True)
        except (ValueError,KeyError,TypeError):
            self.last_target=np.zeros(2)
            return dict(result,reason='scene_binding_invalid',state='stop_request')
        # A target is not an instantaneous velocity. Model the outstanding target,
        # bounded actuator response, watchdog reaction interval, and actual braking.
        # A shadow recommendation was never sent to an actuator. Hold the
        # observed velocity through delay, never invent an outstanding command.
        pending=velocity if self.shadow else self.last_target
        people=self._people(scene);requested=desired.copy();attempts=[]
        grid,grid_binding,dynamic=people.navigation_grid(scene)
        result['dynamic_geometry']=dict(dynamic)
        # Never erase current contact by moving a person into a predicted future.
        result['navigation_geometry']=scene.geometry.metadata.get('navigation_fusion')
        current=footprint_status(contact_grid,self.geometry,feedback['pose'],self.margin)
        if current!='free':
            self.last_target=np.zeros(2)
            return dict(result,state='stop_request',reason='stopping_sweep_'+current)
        points=stopping_trajectory(feedback,desired,self.motion,self.settings,dt,pending=pending,braking_motion=self.braking_motion,
            observation_age_s=now-feedback['stamp_s'])
        prediction_start=feedback['stamp_s']
        result['stopping_sweep']=dict(points=points.tolist(),frame_id=r['frame_id'],prediction_start_s=float(prediction_start),
            feedback_stamp_s=feedback['stamp_s'],scope='hypothetical_not_executed')
        dense,error=interpolate_se2(points[:,1:],scene.geometry.metadata['resolution'],self.settings['max_sweep_samples'])
        uncertainty=motion_uncertainty_margin(feedback,self.geometry,float(points[-1,0]))
        result['motion_uncertainty_margin_m']=uncertainty
        check=self.sweep.check(grid,self.geometry,dense,self.margin+error+uncertainty,self.settings['max_sweep_samples'],binding=grid_binding)
        reason='stopping_sweep_'+check['status'] if check['status']!='free' else None
        prediction=None if reason else people.check(points,self.geometry,self.margin+uncertainty+dynamic.get('projection_margin_m',0.),start_time=prediction_start)
        if prediction is not None and not prediction['valid']:reason=prediction['reason']
        attempts.append(dict(scale=1.,reason=reason,minimum_clearance_m=None if prediction is None else prediction.get('minimum_clearance_m')))
        result.update(check=check,people_check=prediction,velocity_filter=dict(requested_velocity=requested.tolist(),
            selected_scale=1. if reason is None else None,attempts=attempts))
        if reason is None:
            self.last_target=desired.copy()
            return dict(result,state='allowed',reason=None,velocity=desired.tolist(),trajectory_binding=proposal['trajectory_binding'])
        self.last_target=np.zeros(2)
        return dict(result,reason=reason,state='stop_request')
