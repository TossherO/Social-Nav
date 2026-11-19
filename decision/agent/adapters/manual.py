"""Operator-driven travel with read-only runtime observations."""
import copy
import math
import time
from .shadow import RuntimeShadowNavigation, apply_runtime_mapped_pose
from ..core.route import record_route_visit,validate_route_progress


class ManualNavigation(RuntimeShadowNavigation):
    scope = 'manual'

    def __init__(self,catalog,client,*,automatic_arrival=False):
        client.require_observation_only()
        from navigation.control.tracking import load_config
        self.settings=load_config()
        # Manual confirmation is a task-level observation, not an automatic
        # control tolerance. Keep the original heading and stopped checks.
        self.settings['goal_position_tolerance_m']=.20
        self.status='idle'
        self.confirmation=None
        self.automatic_arrival=automatic_arrival
        self.automatic_evidence=None
        self.arrival_since=None
        self.arrival_last_stamp=None
        self.route=None
        self.route_index=None
        self.route_start_stamp_s=None
        self.route_visits=[]
        self.released_handle=None
        self.arrival_invalidation=None
        self.arrival_events=[]
        self.route_progress_tolerance_m=.5
        super().__init__(catalog,client)
        if self.clock_domain != 'runtime_live':
            raise ValueError('Manual navigation requires live observations')

    def start(self,place,handle):
        if (self.handle is not None and self.status not in ('arrived','stopped') and
                self.released_handle!=self.handle):
            raise ValueError('An operator target is already active')
        self.follow_active=False
        self.handle=copy.deepcopy(handle)
        self.released_handle=None
        self.arrival_invalidation=None
        self.route=self.route_index=self.route_start_stamp_s=None
        self.route_visits=[]
        requirement=place.get('requirement') or {}
        require_yaw=(requirement.get('stage_type')=='seating' or 'required_yaw' in requirement or any(requirement.get(key) for key in
                     ('facing','semantic_conditions','visual_preferences')))
        self.request=dict(place_id=place['place_id'],pose=copy.deepcopy(place['pose']),require_yaw=require_yaw)
        self.confirmation=None
        self.automatic_evidence=None;self.arrival_since=None;self.arrival_last_stamp=None
        self.cancel_requested=False
        self.status='running'
        return dict(accepted=True,handle=copy.deepcopy(handle),execution_feedback=False)

    def start_route(self,place,handle):
        ack=self.start(place,handle)
        self.route=copy.deepcopy(place['route'])
        self.route_index=0
        return ack

    def start_follow(self,intent,handle):
        ack=super().start_follow(intent,handle)
        self.status='running'
        self.route=self.route_index=self.route_start_stamp_s=None
        self.confirmation=self.automatic_evidence=self.released_handle=None
        self.route_visits=[]
        return ack

    def release_target(self,handle):
        """Permit a later stage to replace a task already reviewed as complete."""
        if handle!=self.handle:
            raise ValueError('Manual target identity changed before stage completion')
        self.released_handle=copy.deepcopy(handle)

    def drain_arrival_events(self):
        events=self.arrival_events
        self.arrival_events=[]
        return events

    def _ensure_final_visit(self,state):
        if self.route is None:return True
        if self.route_index!=len(self.route['waypoints'])-1:return False
        if len(self.route_visits)==self.route_index:
            return record_route_visit(self.route,self.route_visits,self.handle,state,self.route_index,
                                      self.settings['goal_position_tolerance_m'])
        if len(self.route_visits)!=len(self.route['waypoints']):return False
        context=dict(clock_domain=state.get('clock_domain'),
            local_identity=[state.get(k) for k in ('session_id','generation','frame_id')],
            map_binding=copy.deepcopy(state.get('map_binding')))
        proof=dict(route_visits=self.route_visits,handle=self.handle,frame_id=state.get('origin_frame_id'),
            stamp_s=state.get('stamp_s'),source=state.get('source'),context=context)
        return validate_route_progress(self.route,proof)['status']=='met'

    def tick(self):
        state=super().tick()
        if self.follow_active:return state
        if (self.route is not None and self.status=='running' and
                self.route_index < len(self.route['waypoints'])-1 and
                not state.get('input_stale') and state.get('localization_status')=='valid' and
                state.get('origin_frame_id')==self.catalog.value['frame_id'] and
                state.get('origin_pose') is not None):
            point=self.route['waypoints'][self.route_index]
            pose=state['origin_pose']
            tolerance=(self.settings['goal_position_tolerance_m'] if self.route_index==0
                       else self.route_progress_tolerance_m)
            if record_route_visit(self.route,self.route_visits,self.handle,state,self.route_index,tolerance):
                if self.route_index==0:
                    self.route_start_stamp_s=state['origin_stamp_s']
                self.route_index+=1
                state=self.snapshot()
        if self.automatic_arrival and self.status=='running' and self.handle is not None:
            pose=state.get('origin_pose');stamp=state.get('origin_stamp_s');now=state.get('stamp_s')
            final=self.route is None or (self.route_index==len(self.route['waypoints'])-1 and
                                         self.route_start_stamp_s is not None)
            current=(final and not state.get('input_stale') and state.get('localization_status')=='valid' and
                     state.get('origin_frame_id')==self.catalog.value['frame_id'] and
                     isinstance(stamp,(int,float)) and isinstance(now,(int,float)) and
                     0<=now-stamp<=.5 and pose is not None and state.get('stopped'))
            if current:
                target=self.request['pose']
                distance=math.hypot(pose[0]-target[0],pose[1]-target[1])
                angle=abs(math.atan2(math.sin(pose[2]-target[2]),math.cos(pose[2]-target[2])))
                current=(distance<=self.settings['goal_position_tolerance_m'] and
                         (not self.request['require_yaw'] or angle<=self.settings['goal_yaw_tolerance_rad']))
            if not current:
                self.arrival_since=self.arrival_last_stamp=None
            elif (self.arrival_since is None or self.arrival_last_stamp is None or stamp<self.arrival_last_stamp or
                  stamp-self.arrival_last_stamp>.5):
                self.arrival_since=self.arrival_last_stamp=stamp
            elif stamp-self.arrival_since>=self.settings['goal_dwell_s']:
                if not self._ensure_final_visit(state):
                    return state
                self.automatic_evidence=dict(handle=copy.deepcopy(self.handle),stamp_s=stamp,
                    pose=copy.deepcopy(pose),local_identity=[state.get(k) for k in ('session_id','generation','frame_id')],
                    map_binding=copy.deepcopy(state.get('map_binding')),source='automatic_localization_and_stop_estimate')
                if self.arrival_invalidation is not None:
                    self.arrival_events.append(dict(event='manual_arrival_recovered',stamp_s=stamp,
                        handle=copy.deepcopy(self.handle),source='automatic_localization_and_stop_estimate'))
                    self.arrival_invalidation=None
                self.status='arrived';state=self.snapshot()
            else:self.arrival_last_stamp=stamp
        return state

    def cancel(self,handle):
        if handle != self.handle:
            raise ValueError('Wrong manual navigation target')
        if self.follow_active:
            return super().cancel(handle)
        self.confirmation=None
        self.automatic_evidence=None;self.arrival_since=None;self.arrival_last_stamp=None
        self.status='stopped'
        self.cancel_requested=True

    def snapshot(self):
        state=apply_runtime_mapped_pose(super().snapshot(),self.value['control'],self.catalog)
        control=self.value['control']
        feedback=control.get('idle_feedback') or {}
        velocity=feedback.get('velocity')
        stamp=feedback.get('stamp_s')
        feedback_current=(feedback.get('status')=='valid' and
            feedback.get('source')=='causal_algorithm_poses_not_chassis_feedback' and
            feedback.get('local_identity')==[state.get(k) for k in ('session_id','generation','frame_id')] and
            type(stamp) in (int,float) and math.isfinite(stamp) and
            0 <= state['stamp_s']-stamp <= self.settings['feedback_max_age_s'] and
            isinstance(velocity,list) and len(velocity)==2 and
            all(type(v) in (int,float) and math.isfinite(v) for v in velocity))
        stopped=(feedback_current and
            abs(velocity[0])<=self.settings['stopped_linear_mps'] and
            abs(velocity[1])<=self.settings['stopped_angular_rps'])
        # Operator driving uses the idle estimator even with an Agent display target.
        state['pose'] = copy.deepcopy(feedback.get('pose')) if feedback_current else None
        if self.follow_active:
            follow=state.get('follow_status') or {}
            acknowledged=(follow.get('handle')==self.handle and follow.get('status') in ('failed','stopped'))
            confirmed=bool(acknowledged and stopped and not state['input_stale'] and
                           state['localization_status']=='valid' and state.get('origin_pose') is not None)
            if follow.get('status') == 'preempted':
                self.status = 'preempted'
            else:
                self.status = 'stopped' if confirmed else 'running'
            state.update(source='runtime_manual_observer',handle=copy.deepcopy(self.handle),stopped=bool(stopped),
                         status=('failed' if confirmed and follow.get('status')=='failed' else self.status),
                         stop_observation=confirmed,stop_measurement_stamp_s=stamp,proposal_cancelled=confirmed,
                         manual_arrival_observation=False,velocity=copy.deepcopy(velocity))
            return state
        pose=state.get('origin_pose')
        confirmation=self.confirmation or self.automatic_evidence
        current=(not state['input_stale'] and state['localization_status']=='valid' and pose is not None and
                 feedback.get('status')=='valid')
        confirmed=(confirmation is not None and self.handle==confirmation['handle'] and
            confirmation['local_identity']==[state.get(k) for k in ('session_id','generation','frame_id')] and
            confirmation['map_binding']==state.get('map_binding') and
            state['stamp_s']>=confirmation['stamp_s'] and current and stopped)
        if confirmed:
            target=self.request['pose']
            distance=math.hypot(pose[0]-target[0],pose[1]-target[1])
            angle=abs(math.atan2(math.sin(pose[2]-target[2]),math.cos(pose[2]-target[2])))
            confirmed=(distance<=self.settings['goal_position_tolerance_m'] and
                       (not self.request['require_yaw'] or angle<=self.settings['goal_yaw_tolerance_rad']))
        # Missing or stale velocity cannot prove movement; preserve the existing
        # confirmation while requiring a fresh stopped estimate to accept it.
        if confirmation is not None and current and feedback_current and not confirmed:
            if self.handle!=confirmation['handle']:reason='target_changed'
            elif confirmation['local_identity']!=[state.get(k) for k in ('session_id','generation','frame_id')]:reason='localization_identity_changed'
            elif confirmation['map_binding']!=state.get('map_binding'):reason='map_association_changed'
            elif state['stamp_s']<confirmation['stamp_s']:reason='clock_regressed'
            elif not stopped:reason='movement_detected'
            else:
                target=self.request['pose']
                distance=math.hypot(pose[0]-target[0],pose[1]-target[1])
                reason='outside_position_tolerance' if distance>self.settings['goal_position_tolerance_m'] else 'outside_heading_tolerance'
            self.arrival_invalidation=dict(event='manual_arrival_invalidated',reason=reason,stamp_s=state['stamp_s'],
                pose=copy.deepcopy(pose),velocity=copy.deepcopy(velocity),handle=copy.deepcopy(self.handle))
            self.arrival_events.append(copy.deepcopy(self.arrival_invalidation))
            self.confirmation=None
            self.automatic_evidence=None;self.arrival_since=None;self.arrival_last_stamp=None
            confirmation=None
            self.status='running'
        state.update(scope=self.scope,source='runtime_manual_observer',status='arrived' if confirmed else self.status,
            handle=copy.deepcopy(self.handle),stopped=bool(stopped),velocity=copy.deepcopy(velocity),
            manual_arrival_observation=bool(confirmed),manual_confirmation=copy.deepcopy(self.confirmation),
            manual_arrival_evidence=copy.deepcopy(confirmation),
            manual_arrival_invalidation=copy.deepcopy(self.arrival_invalidation),
            proposal_cancelled=self.status=='stopped',execution_feedback=False,physical_acceptance=False,
            route_index=self.route_index,route_count=None if self.route is None else len(self.route['waypoints']),
            route_start_stamp_s=self.route_start_stamp_s,
            route_visits=copy.deepcopy(self.route_visits),
            goal=dict(task_id=self.handle['task_id'],goal_revision=self.handle['goal_revision']) if confirmed else None)
        return state

    def confirm_arrival(self):
        if self.follow_active or self.handle is None or self.status != 'running':
            raise ValueError('No manual target is awaiting confirmation')
        state=self.tick()
        deadline=time.monotonic()+2.
        while (state.get('origin_pose') is None and not state.get('input_stale') and
               state.get('localization_status')=='valid' and time.monotonic()<deadline):
            time.sleep(.05)
            state=self.tick()
        pose=state.get('origin_pose');target=self.request['pose']
        if self.route is not None and (self.route_index!=len(self.route['waypoints'])-1 or
                len(self.route_visits) not in (self.route_index,self.route_index+1)):
            raise ValueError('Not all intermediate waypoints have been observed in order')
        if (state.get('input_stale') or state.get('localization_status')!='valid' or
            state.get('origin_frame_id')!=self.catalog.value['frame_id'] or pose is None or not state.get('stopped')):
            raise ValueError('Current mapped localization and a fresh stopped estimate are required')
        distance=math.hypot(pose[0]-target[0],pose[1]-target[1])
        angle=abs(math.atan2(math.sin(pose[2]-target[2]),math.cos(pose[2]-target[2])))
        if (distance>self.settings['goal_position_tolerance_m'] or
                self.request['require_yaw'] and angle>self.settings['goal_yaw_tolerance_rad']):
            raise ValueError('Outside target tolerance: distance %.2f m, yaw error %.1f degrees' % (distance,math.degrees(angle)))
        if not self._ensure_final_visit(state):
            raise ValueError('A fresh final waypoint observation is required')
        self.confirmation=dict(handle=copy.deepcopy(self.handle),stamp_s=state['stamp_s'],
            pose=copy.deepcopy(pose),local_identity=[state.get(k) for k in ('session_id','generation','frame_id')],
            map_binding=copy.deepcopy(state.get('map_binding')),source='operator_confirmation_with_localization_estimate')
        self.arrival_invalidation=None
        self.status='arrived'
        return copy.deepcopy(self.confirmation)
