"""Operator-driven travel with read-only runtime observations."""
import copy
import math
import time
from .shadow import RuntimeShadowNavigation


class ManualNavigation(RuntimeShadowNavigation):
    scope = 'manual'

    def __init__(self,catalog,client):
        client.require_observation_only()
        from navigation.control.tracking import load_config
        self.settings=load_config()
        # Manual confirmation is a task-level observation, not an automatic
        # control tolerance. Keep the original heading and stopped checks.
        self.settings['goal_position_tolerance_m']=.20
        self.status='idle'
        self.confirmation=None
        self.route=None
        self.route_index=None
        self.route_start_stamp_s=None
        # Route progress is informational; final completion uses the measured trace.
        self.route_progress_tolerance_m=.5
        super().__init__(catalog,client)
        if self.clock_domain != 'runtime_live':
            raise ValueError('Manual navigation requires live observations')

    def start(self,place,handle):
        if self.handle is not None and self.status not in ('arrived','stopped'):
            raise ValueError('An operator target is already active')
        self.handle=copy.deepcopy(handle)
        self.route=self.route_index=self.route_start_stamp_s=None
        requirement=place.get('requirement') or {}
        require_yaw=('required_yaw' in requirement or any(requirement.get(key) for key in
                     ('facing','semantic_conditions','visual_preferences')))
        self.request=dict(place_id=place['place_id'],pose=copy.deepcopy(place['pose']),require_yaw=require_yaw)
        self.confirmation=None
        self.cancel_requested=False
        self.status='running'
        return dict(accepted=True,handle=copy.deepcopy(handle),execution_feedback=False)

    def start_route(self,place,handle):
        ack=self.start(place,handle)
        self.route=copy.deepcopy(place['route'])
        self.route_index=0
        return ack

    def tick(self):
        state=super().tick()
        if (self.route is not None and self.status=='running' and
                self.route_index < len(self.route['waypoints'])-1 and
                not state.get('input_stale') and state.get('localization_status')=='valid' and
                state.get('origin_frame_id')==self.catalog.value['frame_id'] and
                state.get('origin_pose') is not None):
            point=self.route['waypoints'][self.route_index]
            pose=state['origin_pose']
            tolerance=(self.settings['goal_position_tolerance_m'] if self.route_index==0
                       else self.route_progress_tolerance_m)
            if math.hypot(pose[0]-point[0],pose[1]-point[1]) <= tolerance:
                if self.route_index==0:
                    self.route_start_stamp_s=state['origin_stamp_s']
                self.route_index+=1
                state=self.snapshot()
        return state

    def cancel(self,handle):
        if handle != self.handle:
            raise ValueError('Wrong manual navigation target')
        self.confirmation=None
        self.status='stopped'
        self.cancel_requested=True

    def snapshot(self):
        state=super().snapshot()
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
        pose=state.get('origin_pose')
        confirmation=self.confirmation
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
            self.confirmation=None
            self.status='running'
        state.update(scope=self.scope,source='runtime_manual_observer',status='arrived' if confirmed else self.status,
            handle=copy.deepcopy(self.handle),stopped=bool(stopped),velocity=copy.deepcopy(velocity),
            manual_arrival_observation=bool(confirmed),manual_confirmation=copy.deepcopy(self.confirmation),
            proposal_cancelled=self.status=='stopped',execution_feedback=False,physical_acceptance=False,
            route_index=self.route_index,route_count=None if self.route is None else len(self.route['waypoints']),
            route_start_stamp_s=self.route_start_stamp_s,
            goal=dict(task_id=self.handle['task_id'],goal_revision=self.handle['goal_revision']) if confirmed else None)
        return state

    def confirm_arrival(self):
        if self.handle is None or self.status != 'running':
            raise ValueError('No manual target is awaiting confirmation')
        state=self.tick()
        deadline=time.monotonic()+2.
        while (state.get('origin_pose') is None and not state.get('input_stale') and
               state.get('localization_status')=='valid' and time.monotonic()<deadline):
            time.sleep(.05)
            state=self.tick()
        pose=state.get('origin_pose');target=self.request['pose']
        if self.route is not None and self.route_start_stamp_s is None:
            raise ValueError('Route start has not been observed')
        if (state.get('input_stale') or state.get('localization_status')!='valid' or
            state.get('origin_frame_id')!=self.catalog.value['frame_id'] or pose is None or not state.get('stopped')):
            raise ValueError('Current mapped localization and a fresh stopped estimate are required')
        distance=math.hypot(pose[0]-target[0],pose[1]-target[1])
        angle=abs(math.atan2(math.sin(pose[2]-target[2]),math.cos(pose[2]-target[2])))
        if (distance>self.settings['goal_position_tolerance_m'] or
                self.request['require_yaw'] and angle>self.settings['goal_yaw_tolerance_rad']):
            raise ValueError('Outside target tolerance: distance %.2f m, yaw error %.1f degrees' % (distance,math.degrees(angle)))
        self.confirmation=dict(handle=copy.deepcopy(self.handle),stamp_s=state['stamp_s'],
            pose=copy.deepcopy(pose),local_identity=[state.get(k) for k in ('session_id','generation','frame_id')],
            map_binding=copy.deepcopy(state.get('map_binding')),source='operator_confirmation_with_localization_estimate')
        self.status='arrived'
        return copy.deepcopy(self.confirmation)
