"""Feedback-driven seating phase; emits requests, never commands a robot directly."""
import math
import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionLimits:
    position_m: float
    yaw_rad: float
    linear_m_s: float
    angular_rad_s: float
    hold_s: float
    max_gap_s: float
    max_age_s: float

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in vars(self).values()):
            raise ValueError('Explicit positive completion limits are required')


class SeatingStage:
    """One task and one accepted candidate binding. Limits are deployment inputs."""
    terminal={'succeeded','failed','cancelled'}

    def __init__(self,task_id,candidate_set,limits,*,started_s,timeout_s):
        from .inference import validate_candidate_set,binding
        validate_candidate_set(candidate_set)
        if not isinstance(task_id,str) or not task_id.strip():raise ValueError('Task identity required')
        if (type(started_s) not in (int,float) or not math.isfinite(started_s) or started_s<candidate_set['stamp_s'] or
                type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or timeout_s<=0):
            raise ValueError('Explicit task start and positive timeout required')
        self.started_s,self.timeout_s=started_s,timeout_s
        self.task_id,self.candidates,self.limits=task_id,copy.deepcopy(candidate_set),limits
        self.binding=binding(self.candidates)
        self.state,self.goal,self.reason='selecting',None,None
        self.since,self.last_stamp,self.pending=None,None,None
        self.activation_stamp=None
        self.control_active=False

    def result(self,action=None):
        return dict(task_id=self.task_id,state=self.state,reason=self.reason,
                    candidate_binding=self.binding,goal_id=self.goal['id'] if self.goal else None,
                    action=copy.deepcopy(action))

    def select(self,response,candidate_binding):
        from .inference import validate_selection
        if self.state!='selecting' or candidate_binding!=self.binding:raise ValueError('Stale task selection')
        validate_selection(response,self.candidates)
        if response['s']!='v':
            self.state,self.reason='failed','no_suitable_candidate' if response['s']=='n' else 'insufficient_evidence'
            return self.result()
        self.goal=next(c for c in self.candidates['candidates'] if c['id']==response['p'])
        self.state='awaiting_geometry'
        return self.result(dict(operation='validate_goal',goal=self.goal))

    def geometry(self,*,task_id,candidate_binding,feasible,source,stamp_s,now_s):
        if (self.state!='awaiting_geometry' or task_id!=self.task_id or candidate_binding!=self.binding
                or type(feasible) is not bool or not source):raise ValueError('Unbound geometry verification')
        if not self._fresh(stamp_s,now_s) or stamp_s<self.candidates['stamp_s']:raise ValueError('Stale geometry verification')
        if now_s-self.started_s>=self.timeout_s:return self._stop('failed','task_timeout')
        if not feasible:
            self.state,self.reason='failed','geometry_rejected'
            return self.result()
        self.state,self.control_active='approaching',True
        self.activation_stamp=stamp_s
        return self.result(dict(operation='navigate',goal=self.goal,verification_source=source))

    def _fresh(self,stamp,now):
        return (type(stamp) in (int,float) and type(now) in (int,float) and
                math.isfinite(stamp) and math.isfinite(now) and 0<=now-stamp<=self.limits.max_age_s)

    def _stop(self,terminal,reason):
        self.pending,self.reason,self.since=terminal,reason,None
        self.state='stopping' if self.control_active else terminal
        return self.result(dict(operation='stop') if self.control_active else None)

    def cancel(self):
        if self.state in self.terminal:return self.result()
        return self._stop('cancelled','cancelled_by_caller')

    def fail(self,reason):
        if self.state in self.terminal:return self.result()
        return self._stop('failed',reason)

    def tick(self,now_s):
        """Call on the task clock even when telemetry is absent; not a wall-clock thread."""
        if type(now_s) not in (int,float) or not math.isfinite(now_s) or now_s<self.started_s:
            raise ValueError('Invalid task clock')
        if self.state in self.terminal:return self.result()
        if self.state=='stopping':return self.result(dict(operation='stop'))
        if now_s-self.started_s>=self.timeout_s:return self._stop('failed','task_timeout')
        if self.control_active and now_s-(self.last_stamp if self.last_stamp is not None else self.activation_stamp)>self.limits.max_age_s:
            return self._stop('failed','feedback_timeout')
        return self.result()

    def observe(self,feedback,*,now_s):
        if self.state in self.terminal:raise ValueError('Task already ended')
        if not self.control_active:raise ValueError('Navigation has not been verified and requested')
        for key,expected in [('task_id',self.task_id),('candidate_binding',self.binding),
                             ('goal_id',self.goal['id']),('frame_id',self.candidates['frame_id']),
                             ('clock_domain',self.candidates['clock_domain']),
                             ('session_id',self.candidates['session_id']),('generation',self.candidates['generation'])]:
            if feedback.get(key)!=expected:raise ValueError('Feedback identity mismatch: '+key)
        stamp=feedback.get('stamp_s')
        if not self._fresh(stamp,now_s):return self._stop('failed','feedback_stale_or_future')
        if stamp<self.activation_stamp:raise ValueError('Feedback precedes goal verification')
        if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('Late or duplicate feedback')
        gap=self.last_stamp is not None and stamp-self.last_stamp>self.limits.max_gap_s
        self.last_stamp=stamp
        if feedback.get('stopped') is True and self.state=='stopping' and self.pending!='succeeded':
            self.state,self.control_active=self.pending,False
            return self.result()
        if now_s-self.started_s>=self.timeout_s:return self._stop('failed','task_timeout')
        if feedback.get('health')!='valid' or feedback.get('geometry_valid') is not True:
            return self._stop('failed','localization_or_geometry_invalid')
        if feedback.get('navigation_state')=='failed':return self._stop('failed','navigation_failed')
        values=[feedback.get(k) for k in ('x_m','y_m','yaw_rad','linear_m_s','angular_rad_s')]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            return self._stop('failed','missing_motion_or_pose_evidence')
        x,y,yaw,linear,angular=values;gx,gy,gyaw=self.goal['pose']
        position=math.hypot(x-gx,y-gy)<=self.limits.position_m
        orientation=abs(math.atan2(math.sin(yaw-gyaw),math.cos(yaw-gyaw)))<=self.limits.yaw_rad
        ready=(position and orientation and feedback.get('table_relation_valid') is True and
               abs(linear)<=self.limits.linear_m_s and abs(angular)<=self.limits.angular_rad_s and
               feedback.get('navigation_state') in {'arrived','stopped'})
        if self.state=='stopping':
            if self.pending=='succeeded' and (gap or not ready):return self._stop('failed','completion_evidence_lost')
            if feedback.get('stopped') is True:
                self.state,self.control_active=self.pending,False
            return self.result()
        if not ready:
            self.since=None
            self.state='aligning' if position else 'approaching'
            return self.result()
        if self.since is None or gap:self.since=stamp
        self.state='verifying'
        if stamp-self.since>=self.limits.hold_s:return self._stop('succeeded','completion_verified')
        return self.result()

    def observe_navigation(self,request,feedback,*,navigation_state,geometry_valid,table_relation_valid,now_s,simulation=False):
        """Adapt explicit execution feedback, never a command ACK or shadow proposal.

        Navigation arrival and table/clearance checks remain separate evidence.
        Simulation must identify itself and cannot be presented as robot feedback.
        """
        from perception.environment.context import fingerprint
        if (request.get('task_id')!=self.task_id or request.get('candidate_binding')!=self.binding or
                self.goal is None or request.get('candidate_id')!=self.goal['id'] or
                request.get('goal')!=self.goal['pose'] or request.get('target_reference')!='base_link' or
                request.get('frame_id')!=self.candidates['frame_id']):
            raise ValueError('Navigation request does not bind this seating goal')
        if (feedback.get('task_id')!=self.task_id or feedback.get('goal_revision')!=request['goal_revision'] or
                feedback.get('request_binding')!=fingerprint(request) or not feedback.get('source') or
                feedback.get('execution_feedback') is not True or
                feedback.get('scope')!=('simulation' if simulation else 'robot')):
            raise ValueError('Bound execution feedback required; shadow/ACK is not execution')
        if feedback.get('local_identity')!=[self.candidates[k] for k in ('session_id','generation','frame_id')]:
            raise ValueError('Navigation feedback lifecycle changed')
        values=feedback.get('pose',[]);velocity=feedback.get('velocity',[])
        if len(values)!=3 or len(velocity)!=2:raise ValueError('Expected measured pose and velocity')
        row=dict(task_id=self.task_id,candidate_binding=self.binding,goal_id=self.goal['id'],
            **{k:self.candidates[k] for k in ('frame_id','session_id','generation')},
            clock_domain=feedback.get('clock_domain'),stamp_s=feedback.get('stamp_s'),health=feedback.get('status'),
            x_m=values[0],y_m=values[1],yaw_rad=values[2],linear_m_s=velocity[0],angular_rad_s=velocity[1],
            geometry_valid=geometry_valid,table_relation_valid=table_relation_valid,
            navigation_state=navigation_state,stopped=feedback.get('stopped'))
        result=self.observe(row,now_s=now_s)
        return dict(result,feedback_scope='simulation' if simulation else 'robot')
