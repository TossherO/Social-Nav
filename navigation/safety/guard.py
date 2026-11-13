"""Independent simulation/shadow gate with finite-horizon motion checks."""
import time
import numpy as np
from perception.environment.context import fingerprint
from navigation.kinematics import ramp_path,turning_residual
from navigation.planning.motion import interpolate_se2,motion_uncertainty_margin
from navigation.planning.people import PeopleModel
from navigation.planning.inputs import request_status
from navigation.planning.collision import SweepChecker
from navigation.planning.footprint import footprint_status
from perception.environment.navigation import navigation_geometry


def predicted_trajectory(feedback, desired, motion, settings, *, pending=None, observation_age_s=0.):
    """Recompute a finite-horizon motion from the original feedback timestamp."""
    velocity=np.asarray(feedback['velocity'],float)
    pending=velocity if pending is None else pending
    step=settings['sample_s']
    aged,initial=ramp_path(feedback['pose'],velocity,pending,motion,observation_age_s,step,timed=True)
    future,_=ramp_path(aged[-1,1:],initial,desired,motion,settings['prediction_s'],
                       min(step,settings['prediction_s']/2),timed=True)
    future[:,0]+=observation_age_s
    points=np.vstack([aged,future[1:]])
    return points[np.r_[True,np.diff(points[:,0])>1e-10]]


class LocalSafety:
    def __init__(self,geometry,motion,margin,*,settings=None,people_settings=None,simulation=False,shadow=False,execution=False,asset=None):
        if sum(map(bool,(simulation,shadow,execution)))!=1:raise ValueError('Choose one explicit safety mode')
        if execution and (geometry.assumed or motion.get('parameters_assumed',True)):
            raise ValueError('Execution requires measured vehicle geometry and motion')
        if execution:
            for key,minimum in (('execution_max_longitudinal_speed_mps',max(motion['max_forward_mps'],motion['max_reverse_mps'])),
                                ('execution_max_angular_rps',motion['max_angular_rps'])):
                value=motion.get(key)
                if type(value) not in (int,float) or not np.isfinite(value) or value<minimum:
                    raise ValueError('Execution requires explicit feedback speed bounds covering commands')
        self.execution=execution
        self.shadow=shadow
        self.asset=asset
        self.geometry=geometry;self.motion=motion;self.margin=margin
        from .config import load_safety_config
        self.settings=settings if settings is not None else load_safety_config()
        if any(not np.isfinite(v) or v<=0 for v in self.settings.values()):raise ValueError('Invalid safety setting')
        self.last_target=np.zeros(2)
        if people_settings is None:
            from navigation.planning.inputs import load_config
            people_settings=load_config()['people']
        self.people_settings=people_settings
        self._people_cache=None
        self._search_planning=None
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
        began=time.perf_counter()
        result=dict(velocity=[0.,0.],state='braking',reason=None,executable=False,scope='execution_check' if self.execution else 'shadow_safety' if self.shadow else 'simulation_safety',
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
        if self.execution and (abs(velocity[0])>self.motion['execution_max_longitudinal_speed_mps'] or
                abs(velocity[1])>self.motion['execution_max_angular_rps']):
            self.last_target=np.zeros(2)
            return dict(result,reason='feedback_exceeds_execution_speed_bound',state='stop_request')
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
        # Account for feedback age before predicting the selected target.
        # A shadow recommendation was never sent to an actuator; propagate
        # feedback age using observed velocity instead of an imagined command.
        pending=velocity if self.shadow else self.last_target
        people=self._people(scene);requested=desired.copy();attempts=[]
        grid,grid_binding,dynamic=people.navigation_grid(scene)
        result['dynamic_geometry']=dict(dynamic)
        # Never erase current contact by moving a person into a predicted future.
        result['navigation_geometry']=scene.geometry.metadata.get('navigation_fusion')
        current=footprint_status(contact_grid,self.geometry,feedback['pose'],self.margin)
        if current!='free':
            self.last_target=np.zeros(2)
            return dict(result,state='stop_request',reason='motion_sweep_'+current)
        age=now-feedback['stamp_s']
        points=predicted_trajectory(feedback,desired,self.motion,self.settings,pending=pending,
                                   observation_age_s=age)
        prediction_start=feedback['stamp_s']
        result['motion_sweep']=dict(points=points.tolist(),frame_id=r['frame_id'],prediction_start_s=float(prediction_start),
            feedback_stamp_s=feedback['stamp_s'],scope='hypothetical_not_executed')
        dense,error=interpolate_se2(points[:,1:],scene.geometry.metadata['resolution'],self.settings['max_sweep_samples'])
        uncertainty=motion_uncertainty_margin(feedback,self.geometry,float(points[-1,0]))
        result['motion_uncertainty_margin_m']=uncertainty
        if request.get('search_bounds') is not None:
            from navigation.planning.inputs import load_config
            from navigation.search import bounds_status
            try:
                if self._search_planning is None:self._search_planning=load_config()
                reason=bounds_status(request,scene,self.asset,self._search_planning,dense,
                                     margin=error+uncertainty,geometry=self.geometry)
            except (ValueError,KeyError,TypeError):reason='follow_search_bounds_invalid'
            if reason:
                self.last_target=np.zeros(2)
                return dict(result,reason=reason,state='stop_request')
        from navigation.planning.collision import CertificationBudget
        budget=CertificationBudget(began+dt,self.settings['max_sweep_samples'])
        check=self.sweep.check(grid,self.geometry,dense,self.margin+error+uncertainty,self.settings['max_sweep_samples'],binding=grid_binding,budget=budget)
        reason='motion_sweep_'+check['status'] if check['status']!='free' else None
        # PeopleModel measures cutoff from the feedback timestamp. Include the
        # feedback age so the configured horizon starts at this check's now.
        people_horizon=age+people.config.get('hard_horizon_s',people.config['horizon_s'])
        prediction=None if reason else people.check(points,self.geometry,self.margin+uncertainty+dynamic.get('projection_margin_m',0.),
            start_time=prediction_start,hard_until=people_horizon)
        if prediction is not None and not prediction['valid']:reason=prediction['reason']
        if not budget.check():reason='motion_'+budget.reason
        result['certification']=budget.summary()
        attempts.append(dict(scale=1.,reason=reason,minimum_clearance_m=None if prediction is None else prediction.get('minimum_clearance_m')))
        result.update(check=check,people_check=prediction,velocity_filter=dict(requested_velocity=requested.tolist(),
            selected_scale=1. if reason is None else None,attempts=attempts))
        if reason is None:
            self.last_target=desired.copy()
            return dict(result,state='allowed',reason=None,velocity=desired.tolist(),trajectory_binding=proposal['trajectory_binding'])
        self.last_target=np.zeros(2)
        return dict(result,reason=reason,state='stop_request')
