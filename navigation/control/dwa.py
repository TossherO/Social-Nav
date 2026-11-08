"""Bounded velocity sampling with causal inputs and complete braking certificates.

Uses the dynamic-window formulation of Fox/Burgard/Thrun (1997) and ROS Noetic
sampling/early rejection semantics; no ROS plugin, optimizer or second tracker.
"""
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
import yaml
from perception.environment.context import fingerprint
from perception.environment.navigation import navigation_geometry
from navigation.kinematics import advance,velocity_step,turning_residual,ramp_path
from navigation.planning.continuity import check_reference,local_goal,context_binding
from navigation.planning.collision import SweepChecker,CertificationBudget
from navigation.planning.footprint import footprint_status,wrap
from navigation.planning.inputs import scene_status,request_status,validate_grid
from navigation.planning.motion import interpolate_se2,motion_uncertainty_margin
from navigation.planning.people import PeopleModel,social_groups
from navigation.planning.rollout import segments,project_progress
from navigation.planning.planner import se2,transform_poses
from navigation.safety.guard import stopping_trajectories
from .tracking import feedback_valid,load_config as tracking_config


def load_config(path=None):
    settings=yaml.safe_load(Path(path or Path(__file__).with_name('dwa.yaml')).read_text())
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in settings.values()):
        raise ValueError('Invalid DWA configuration')
    if any(type(settings[k]) is not int for k in ('linear_samples','angular_samples','max_candidates','certification_max_checks')):
        raise ValueError('DWA sample counts must be integers')
    if settings['linear_samples']*settings['angular_samples']+3>settings['max_candidates'] or settings['max_candidates']>64:
        raise ValueError('DWA candidate budget exceeded')
    if settings['certification_budget_s']>settings['max_compute_s']:
        raise ValueError('Certification budget exceeds the whole control computation budget')
    return settings


class DwaController:
    def __init__(self,planning,motion,geometry,*,asset=None,settings=None,device_batch=None):
        self.device_batch=device_batch
        self.diagnostics=False
        self.planning=copy.deepcopy(planning);self.motion=copy.deepcopy(motion);self.geometry=geometry;self.asset=asset
        self.settings=copy.deepcopy(settings) if settings is not None else load_config();self.goal_settings=tracking_config()
        from navigation.safety.config import load_stopping_config
        self.safety_settings=load_stopping_config(motion)
        self.sweep=SweepChecker(motion.get('collision_backend','cpu'),motion.get('collision_library'))
        self.sweep.metrics=None if device_batch is None else device_batch.metrics
        sources=[Path(__file__),Path(__file__).parents[1]/'kinematics.py',Path(__file__).parents[1]/'safety/guard.py',
            Path(__file__).parents[1]/'planning/collision.py',Path(__file__).parents[1]/'planning/footprint.py']
        if self.sweep.backend=='cuda':sources.append(Path(self.sweep.library))
        if device_batch is not None:sources.extend([Path(__file__).with_name('device.py'),Path(device_batch.library)])
        self.backend_binding={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        self.configuration_binding=fingerprint([self.planning,self.motion,self.settings,self.goal_settings,self.safety_settings,
            geometry.footprint.tolist(),geometry.base_from_lidar.tolist(),self.backend_binding])
        self.sweep.warmup();self.clear()

    def clear(self):
        self.previous=None;self.result=None;self._scene=None;self._scene_binding=None;self._people=None;self._path_key=None;self._progress=0.;self._section=0

    def _model(self,scene):
        binding=scene.record['input_fingerprint']
        if self._scene is not scene or self._scene_binding!=binding:
            self._scene,self._scene_binding,self._people=scene,binding,PeopleModel(scene,self.planning['people'])
        return self._people

    def _coarse_clear(self,points,grid):
        """Definite collision rejection only; success still needs a full sweep."""
        poses=points[:,1:];c,s=np.cos(poses[:,2]),np.sin(poses[:,2]);fp=self.geometry.footprint
        xy=np.empty((len(poses),len(fp),2))
        xy[:,:,0]=poses[:,0,None]+c[:,None]*fp[:,0]-s[:,None]*fp[:,1]
        xy[:,:,1]=poses[:,1,None]+s[:,None]*fp[:,0]+c[:,None]*fp[:,1]
        ij=np.floor((xy-grid.metadata['origin_xy'])/grid.metadata['resolution']).astype(int)
        shape=np.array(grid.layers['occupancy'].shape)[::-1]
        return bool((ij>=0).all() and (ij<shape).all() and not np.any(grid.layers['occupancy'][ij[:,:,1],ij[:,:,0]]!=0))

    def command(self,request,scene,feedback,reference,*,now,clock_domain,dt=.1,pending=None):
        began=time.perf_counter();self.result=None;timing={};budget=None;certification_start=None
        certification=dict(prediction_sweep_ms=0.,braking_sweep_ms=0.,people_ms=0.,enclosures_ms=0.,cpu_fallback_ms=0.,
            cpu_fallbacks=0,reused_segments=0,gpu_refinement_batches=0,gpu_refinement_enclosures=0)
        out=dict(status='unavailable',reason=None,velocity=[0.,0.],control_mode='dwa',executable=False,
            request_binding=fingerprint(request),scene_binding=scene.record['input_fingerprint'])
        try:
            if not 0<dt<=.2:raise ValueError('invalid_control_period')
            reason=scene_status(scene,now,clock_domain,self.planning) or request_status(request,scene,now,clock_domain)
            if reason:raise ValueError(reason)
            if request.get('goal_behavior','pass_through') not in ('stop','pass_through'):raise ValueError('invalid_goal_behavior')
            if request.get('motion_intent','forward_preferred')!='forward_preferred':raise ValueError('reverse_or_unknown_intent_not_supported')
            if not feedback_valid(feedback,scene,now,clock_domain,self.goal_settings['feedback_max_age_s']):
                raise ValueError('feedback_invalid')
            scene.verify();validate_grid(scene.geometry)
            if reference is None:raise ValueError('waiting_for_background_reference')
            check_reference(reference,request,scene,self.asset,self.settings['reference_max_age_s'])
            model=self._model(scene)
            if model.status=='invalid' or model.status=='missing' and model.config['required']:raise ValueError(model.reason)
            velocity=np.asarray(feedback['velocity'],float);age=now-feedback['stamp_s']
            reverse=self.motion['max_reverse_mps']*self.motion['max_initial_profile_ratio'] if self.motion['allow_reverse'] else self.motion['reverse_tolerance_mps']
            if (velocity[0]<-reverse or
                    velocity[0]>self.motion['max_forward_mps']*self.motion['max_initial_profile_ratio'] or
                    abs(velocity[1])>self.motion['max_angular_rps']*self.motion['max_initial_profile_ratio'] or
                    turning_residual(velocity,self.motion)<-1e-7):raise ValueError('feedback_outside_motion_model')
            contact_grid=navigation_geometry(scene,now=now,asset=self.asset,check_asset=True)
            out['navigation_geometry']=scene.geometry.metadata.get('navigation_fusion')
            if footprint_status(contact_grid,self.geometry,feedback['pose'],self.planning['safety_margin_m'])!='free':
                raise ValueError('current_footprint_not_free')
            grid,binding,dynamic=model.navigation_grid(scene);groups,_=social_groups(request,scene,model)
            timing['inputs_reference']=(time.perf_counter()-began)*1000;stage=time.perf_counter()
            actual_pending=velocity if pending is None else np.asarray(pending,float)
            aged,initial=ramp_path(feedback['pose'],velocity,actual_pending,self.motion,age,self.settings['sample_s'],timed=True)
            start=aged[-1,1:];path=np.asarray(reference['result']['path'],float)
            if request['mode']=='mapped':
                world=np.asarray(scene.record['localization']['world_from_lidar'])
                path=transform_poses(path,se2(self.geometry.pose(world))@np.linalg.inv(se2(self.geometry.pose(np.asarray(scene.record['prior']['map_from_local'])@world))))
            path=path[np.r_[True,(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)>1e-7)|
                (abs(wrap(np.diff(path[:,2])))>1e-7)]]
            if len(path)<2:raise ValueError('empty_reference')
            yaw=path[:-1,2]+wrap(np.diff(path[:,2]))/2
            direction=np.where(np.sum(np.diff(path[:,:2],axis=0)*np.c_[np.cos(yaw),np.sin(yaw)],axis=1)<0,-1,1)
            rotating=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)<=1e-7
            if rotating.any() and not self.motion.get('allow_in_place_rotation',False):raise ValueError('reference_requires_rotation')
            direction=np.where(rotating,np.where(wrap(np.diff(path[:,2]))>0,2,-2),direction)
            if not self.motion['allow_reverse'] and (direction==-1).any():raise ValueError('reference_requires_reverse')
            cuts=np.r_[0,1+np.flatnonzero(np.diff(direction)),len(direction)]
            # Keep long rotations in their certified direction rather than
            # replacing a 270-degree route by an obstructed -90-degree turn.
            extra=[]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                if abs(direction[lo])!=2:continue
                accumulated=0.
                for j in range(lo,hi):
                    angle=abs(wrap(path[j+1,2]-path[j,2]))
                    if accumulated+angle>np.pi/2 and j>lo:extra.append(j);accumulated=0.
                    accumulated+=angle
            cuts=np.unique(np.r_[cuts,extra]).astype(int)
            key=reference['binding']
            if key!=self._path_key:self._progress=0.;self._section=0;self._path_key=key
            # A cusp is a stop-and-change-direction point, not a lookahead
            # shortcut onto an overlapping later section of the reference.
            while self._section<len(cuts)-2:
                cusp=path[cuts[self._section+1]]
                if (np.linalg.norm(start[:2]-cusp[:2])>self.goal_settings['goal_position_tolerance_m'] or
                        abs(wrap(start[2]-cusp[2]))>self.goal_settings['goal_yaw_tolerance_rad'] or
                        abs(velocity[0])>self.goal_settings['stopped_linear_mps'] or
                        abs(velocity[1])>self.goal_settings['stopped_angular_rps']):break
                self._section+=1;self._progress=0.
            lo,hi=cuts[self._section:self._section+2];rotation_mode=abs(direction[lo])==2
            gear=0 if rotation_mode else int(direction[lo]);cusp_stop=hi<len(direction)
            path=path[lo:hi+1]
            out['reference_motion']=dict(direction=gear,section=self._section,stop_at_cusp=bool(cusp_stop))
            if rotation_mode:out['reference_motion']['rotation']=True
            prepared=segments(path)
            if not rotation_mode and not len(prepared[2]):raise ValueError('empty_reference')
            goal=local_goal(request,scene,self.geometry,self.asset,self.planning)
            stop=request.get('goal_behavior','pass_through')=='stop'
            if rotation_mode:
                if np.linalg.norm(start[:2]-path[0,:2])>self.goal_settings['goal_position_tolerance_m']:
                    raise ValueError('rotation_anchor_not_reached')
                progress=0.;arc=np.zeros(len(path));speed=0.
                error=wrap(path[-1,2]-start[2])
                brake=self.motion['max_angular_acceleration_rps2']*self.motion['brake_deceleration_scale']
                delay=self.safety_settings['reaction_s']+self.safety_settings['actuator_delay_s']+age
                remaining=max(0.,abs(error)-self.goal_settings['goal_yaw_tolerance_rad']/2)
                limit=min(self.motion['max_angular_rps'],np.sqrt((brake*delay)**2+2*brake*remaining)-brake*delay)
                target_w=np.sign(error)*limit
                dw=self.motion['max_angular_acceleration_rps2']*dt
                low=np.array([0.,max(-limit,initial[1]-dw)])
                high=np.array([0.,min(limit,initial[1]+dw)])
                choices=[np.zeros(2)];speed_limits=dict(angular_stopping=float(limit))
                # Arrive at the pivot and brake translation before rotating.
                if abs(initial[0])<=self.goal_settings['stopped_linear_mps']:
                    if high[1]>=low[1]:
                        choices.extend(np.array([0.,w]) for w in np.linspace(low[1],high[1],self.settings['angular_samples']))
                        choices.append(np.array([0.,np.clip(target_w,low[1],high[1])]))
            else:
                a,delta,length,arc=prepared
                fraction=np.clip(np.sum((start[:2]-a)*delta,axis=1)/length**2,0,1)
                distance=np.linalg.norm(a+fraction[:,None]*delta-start[:2],axis=1)
                candidates=(arc[:-1]<=self._progress+self.settings['path_window_m'])&(arc[1:]>=self._progress-.2)
                reference_yaw=path[:-1,2]+fraction*wrap(np.diff(path[:,2]))
                match=int(np.argmin(np.where(candidates,distance+.02*abs(wrap(reference_yaw-start[2])),np.inf)))
                progress=float(arc[match]+fraction[match]*length[match]);self._progress=max(self._progress,progress)
                target_s=min(arc[-1],progress+self.settings['lookahead_m'])
                target=np.array([np.interp(target_s,arc,path[:,k]) for k in range(2)])
                if len(goal)==3 and not cusp_stop:
                    target+=gear*max(0.,progress+self.settings['lookahead_m']-arc[-1])*np.array([np.cos(goal[2]),np.sin(goal[2])])
                error=target-start[:2];bearing=wrap(np.arctan2(error[1],error[0])-start[2])
                curvature=np.clip(2*np.sin(bearing)/max(.2,np.linalg.norm(error)),-1/self.motion['min_turning_radius_m'],1/self.motion['min_turning_radius_m'])
                speed_limit=self.motion['max_forward_mps'] if gear>0 else self.motion['max_reverse_mps']
                speed=min(speed_limit,self.motion['max_angular_rps']/max(abs(curvature),1e-9))
                speed_limits=dict(pursuit=float(speed))
                remaining=max(0.,arc[-1]-progress-self.goal_settings['goal_position_tolerance_m']/2)
                if stop or cusp_stop:
                    brake=self.motion['max_acceleration_mps2']*self.motion['brake_deceleration_scale']
                    delay=self.safety_settings['reaction_s']+self.safety_settings['actuator_delay_s']+age
                    speed=min(speed,np.sqrt((brake*delay)**2+2*brake*remaining)-brake*delay)
                    speed_limits['stopping']=float(speed)
                curve=abs(wrap(np.diff(path[:,2])))/np.maximum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1),1e-9)
                if len(curve)==len(length):
                    curve_speed=np.minimum(speed_limit,self.motion['max_angular_rps']*.8/np.maximum(curve,1e-9))
                    distance=np.maximum(0.,arc[:-1]-progress-abs(initial[0])*.4)
                    cap=np.sqrt(curve_speed**2+2*self.motion['max_acceleration_mps2']*self.motion['brake_deceleration_scale']*distance)
                    speed=min(speed,float(np.min(np.where(arc[1:]>=progress,cap,np.inf))))
                    speed_limits['path_curvature']=float(speed)
                acceleration=self.motion['max_acceleration_mps2'];deceleration=acceleration*self.motion['brake_deceleration_scale']
                low=np.maximum([0. if gear>0 else -speed,-self.motion['max_angular_rps']],initial-np.array([
                    acceleration if initial[0]<0 else deceleration,self.motion['max_angular_acceleration_rps2']])*dt)
                high=np.minimum([speed if gear>0 else 0.,self.motion['max_angular_rps']],initial+np.array([
                    deceleration if initial[0]<0 else acceleration,self.motion['max_angular_acceleration_rps2']])*dt)
                # A zero target is a braking action, not an instantaneous zero state.
                choices=[np.zeros(2)]
                if initial[0]*gear>=0 and np.all(high>=low):
                    choices.extend([np.array([v,w]) for v in np.linspace(low[0],high[0],self.settings['linear_samples'])
                        for w in np.linspace(low[1],high[1],self.settings['angular_samples'])])
                    choices.extend([np.clip([gear*speed,gear*speed*curvature],low,high),np.clip(initial if self.previous is None else self.previous,low,high)])
                elif initial[0]*gear>speed:
                    # A bounded actuator target for speed recovery. The measured
                    # overspeed is never clipped or replaced by this target state.
                    choices.append(np.array([gear*speed,np.clip(initial[1],-min(self.motion['max_angular_rps'],speed/self.motion['min_turning_radius_m']),
                        min(self.motion['max_angular_rps'],speed/self.motion['min_turning_radius_m']))]))
            choices=np.unique(np.round(choices,12),axis=0)
            choices=choices[np.asarray(turning_residual(choices,self.motion))>=-1e-9]
            if self.diagnostics:
                out['sampling']=dict(initial=initial.tolist(),observation_velocity=velocity.tolist(),dt=dt,
                    speed_limits=speed_limits,speed=float(speed),low=low.tolist(),high=high.tolist(),
                    empty_window=bool(np.any(high<low)),choices=choices.tolist())
            if (stop and np.linalg.norm(start[:2]-goal[:2])<=self.goal_settings['goal_position_tolerance_m'] and
                    (len(goal)==2 or abs(wrap(start[2]-goal[2]))<=self.goal_settings['goal_yaw_tolerance_rad'])):
                # Hold braking while GoalMonitor checks measured dwell. A noisy
                # reference refresh must not restart an already settled approach.
                choices=np.zeros((1,2))
            if len(choices)>self.settings['max_candidates']:raise ValueError('candidate_budget_exceeded')
            if self.device_batch is None:
                # Shared current-state propagation; candidate arrays remain bounded.
                n=max(2,int(np.ceil(self.settings['prediction_s']/self.settings['sample_s'])))
                step=self.settings['prediction_s']/n;poses=np.empty((n+1,len(choices),3));poses[0]=start
                current=np.broadcast_to(initial,(len(choices),2)).copy()
                for i in range(n):
                    targets=np.broadcast_to(actual_pending,current.shape) if i*step<self.safety_settings['actuator_delay_s'] else choices
                    after=velocity_step(current,targets,self.motion,step)
                    poses[i+1]=advance(poses[i],(current+after)/2,step);current=after
                brakes=stopping_trajectories(feedback,choices,self.motion,self.safety_settings,dt,pending=actual_pending,observation_age_s=age)
                timing['candidate_generation']=(time.perf_counter()-stage)*1000;stage=time.perf_counter()
                results=[];rejects={}
                for j,desired in enumerate(choices):
                    points=np.c_[np.arange(n+1)*step,poses[:,j]]
                    reason=None if self._coarse_clear(points,grid) and self._coarse_clear(brakes[j],grid) else 'coarse_obstacle_or_unknown'
                    if reason:
                        rejects[reason]=rejects.get(reason,0)+1;continue
                    if rotation_mode:
                        projected=np.zeros(len(points));offset=np.linalg.norm(points[:,1:3]-path[0,:2],axis=1)
                    else:projected,offset,total=project_progress(points[:,1:3],path,prepared)
                    end=points[-1,1:]
                    ref_yaw=path[-1,2] if rotation_mode else np.interp(projected[-1],arc,np.unwrap(path[:,2]))
                    heading=wrap(ref_yaw-end[2])
                    score=(self.settings['path_weight']*float(np.mean(offset)+offset[-1])-
                        self.settings['progress_weight']*float(projected[-1]-progress)+self.settings['heading_weight']*abs(heading)+
                        self.settings['change_weight']*float(np.linalg.norm(desired-(initial if self.previous is None else self.previous)))+
                        model.social_cost(points,groups))
                    # Use supplied clearance only as a soft preference, never a certificate.
                    if 'clearance' in grid.layers:
                        ij=np.floor((end[:2]-grid.metadata['origin_xy'])/grid.metadata['resolution']).astype(int)
                        if 0<=ij[1]<grid.layers['clearance'].shape[0] and 0<=ij[0]<grid.layers['clearance'].shape[1]:
                            c=grid.layers['clearance'][ij[1],ij[0]]
                            if np.isfinite(c):score+=self.settings['clearance_weight']/max(.1,float(c))
                    results.append((score,j,points))
                timing['ranking']=(time.perf_counter()-stage)*1000;stage=time.perf_counter()
            else:
                results,rejected=self.device_batch.evaluate(self,scene,model,groups,grid,dynamic,choices,start,initial,
                    actual_pending,feedback,age,dt,path,prepared,progress)
                rejects={'coarse_obstacle_or_unknown':rejected} if rejected else {}
                brakes=None
                timing['candidate_generation_and_ranking']=(time.perf_counter()-stage)*1000
                stage=time.perf_counter()
                if self.sweep.backend=='cuda':self.sweep.bind_device(grid,self.device_batch.navigation_occupancy,binding=binding)
            chosen=None;certified=0
            certification_start=time.perf_counter()
            budget=CertificationBudget(min(began+self.settings['max_compute_s'],
                certification_start+self.settings['certification_budget_s']),self.settings['certification_max_checks'])

            def sweep(poses,margin,phase):
                started=time.perf_counter()
                check=self.sweep.check(grid,self.geometry,poses,margin,self.safety_settings['max_sweep_samples'],binding=binding,budget=budget)
                certification[phase+'_sweep_ms']+=(time.perf_counter()-started)*1000
                for key,value in check.get('timing_ms',{}).items():certification[key+'_ms']+=value
                certification['cpu_fallbacks']+=int(check.get('adaptive_fallback',False))
                certification['reused_segments']+=check.get('reused_segments',0)
                for key in ('gpu_refinement_batches','gpu_refinement_enclosures'):certification[key]+=check.get(key,0)
                if not budget.check():raise ValueError(budget.reason)
                return check

            def people_check(points,margin,start_time,**options):
                if not budget.check():raise ValueError(budget.reason)
                started=time.perf_counter()
                try:result=model.check(points,self.geometry,margin,start_time=start_time,**options)
                finally:certification['people_ms']+=(time.perf_counter()-started)*1000
                if not budget.check():raise ValueError(budget.reason)
                return result

            out.update(candidate_count=len(choices),rejections=rejects,valid_candidates=0,certification_attempts=0)
            if not budget.check():raise ValueError(budget.reason)
            for item in sorted(results,key=lambda x:x[0]):
                if not budget.check():raise ValueError(budget.reason)
                _,j,points=item;certified+=1
                out['certification_attempts']=certified
                if points is None:points=self.device_batch.trajectory(j)
                dense,err=interpolate_se2(points[:,1:],grid.metadata['resolution'],self.safety_settings['max_sweep_samples'])
                margin=self.planning['safety_margin_m']+err+motion_uncertainty_margin(feedback,self.geometry,self.settings['prediction_s'])
                check=sweep(dense,margin,'prediction')
                reason=None if check['status']=='free' else 'sweep_'+check['status']
                prediction=None
                if reason is None:
                    prediction=people_check(points,margin+dynamic.get('projection_margin_m',0.),start_time=now,
                        hard_until=model.config.get('hard_horizon_s'))
                    if not prediction['valid']:reason=prediction['reason']
                if reason is None:
                    braking=brakes[j] if brakes is not None else self.device_batch.trajectory(j,brake=True)
                    dense,err=interpolate_se2(braking[:,1:],grid.metadata['resolution'],self.safety_settings['max_sweep_samples'])
                    stop_margin=self.planning['safety_margin_m']+err+motion_uncertainty_margin(feedback,self.geometry,float(braking[-1,0]))
                    certificate=sweep(dense,stop_margin,'braking')
                    if certificate['status']!='free':reason='braking_'+certificate['status']
                    elif not people_check(braking,stop_margin+dynamic.get('projection_margin_m',0.),start_time=feedback['stamp_s'])['valid']:
                        reason='braking_people_conflict'
                if not budget.check():raise ValueError(budget.reason)
                if reason:rejects[reason]=rejects.get(reason,0)+1
                else:chosen=(item[0],j,points,prediction);break
            timing['certification']=(time.perf_counter()-stage)*1000
            out.update(candidate_count=len(choices),rejections=rejects,valid_candidates=int(chosen is not None),certification_attempts=certified)
            if chosen is None:raise ValueError('no_admissible_velocity')
            _,j,points,prediction=chosen;desired=choices[j];self.previous=desired.copy()
            # Prediction starts now; input_binding separately retains both source times.
            points[:,0]+=age
            result=dict(status='control_ready',reason=None,control_mode='dwa',request=copy.deepcopy(request),request_binding=fingerprint(request),
                configuration_binding=self.configuration_binding,context_binding=context_binding(request,scene,self.asset),
                input_binding=dict(scene=scene.record['input_fingerprint'],as_of=feedback['stamp_s'],geometry_as_of=scene.record['as_of'],
                    frame_id=scene.record['frame_id'],clock_domain=clock_domain,feedback_source_binding=feedback.get('source_binding',fingerprint(feedback))),
                local_goal=dict(pose=goal.tolist(),actual_goal=True,goal_behavior=request.get('goal_behavior','pass_through')),
                trajectory=dict(points=points.tolist(),frame_id=scene.record['frame_id'],initialization='dwa',people_check=prediction,
                    candidate_velocity=dict(linear_x_mps=float(desired[0]),angular_z_rps=float(desired[1]),executable=False)),
                reference_binding=reference['binding'],candidate_count=len(choices),valid_candidates=1,certification_attempts=certified,rejections=rejects,
                timing_ms=dict(timing),executable=False,task_completed=False)
            result['certification']=dict(certification,**budget.summary())
            result['result_binding']=fingerprint(result)
            if not budget.check():raise ValueError(budget.reason)
            self.result=result
            out.update(status='candidate',reason=None,velocity=desired.tolist(),trajectory_as_of=feedback['stamp_s'],
                trajectory_binding=result['result_binding'],source_scene_binding=scene.record['input_fingerprint'],
                reference_binding=reference['binding'],reference_as_of=reference['as_of'],continuation=False)
        except (ValueError,KeyError,TypeError,RuntimeError) as error:
            out.update(reason=str(error),velocity=[0.,0.],valid_candidates=0);self.previous=None
        if budget is not None:
            timing['certification']=(time.perf_counter()-certification_start)*1000
            out['certification']=dict(certification,**budget.summary())
        out['compute_ms']=(time.perf_counter()-began)*1000
        out['timing_ms']=dict(timing)
        return out
