"""Local navigation-intent IPC. Commands never bypass independent safety."""
import argparse
import base64
from collections import deque
import copy
import json
import math
import os
from pathlib import Path
import socket
import threading
import tempfile
import time
from .transport import Connection, exchange_request


class AgentState:
    def __init__(self):
        self.lock = threading.Lock()
        self.control = self.observation = self.image = self.updated_mono = None
        self.vehicle_context = None
        self.goal_evidence = None
        self.follow = None
        self.follow_observation = None
        self.follow_candidate_history = deque(maxlen=16)
        self.follow_candidates_requested_mono = None
        self.seating=None
        self.follow_enabled=False
        self.environment_epoch=0
        self.environment_fault=None
        self.environment_cutoff=None

    def enable_goal_observations(self):
        with self.lock:
            self.goal_evidence = GoalObservationEvidence()

    def configure_vehicle(self, geometry, motion):
        keys = ('allow_reverse','allow_in_place_rotation','min_turning_radius_m','max_forward_mps','max_reverse_mps',
                'max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2','parameters_assumed')
        with self.lock:
            self.vehicle_context = dict(footprint=geometry.footprint.tolist(),base_from_lidar=geometry.base_from_lidar.tolist(),
                source=copy.deepcopy(geometry.source),assumed=geometry.assumed,
                motion={k:copy.deepcopy(motion.get(k)) for k in keys})

    def invalidate_environment(self, fault, epoch):
        with self.lock:
            self.environment_epoch=epoch
            self.environment_fault=fault.reason
            self.environment_cutoff=([*fault.identity,fault.frame],fault.stamp_ns)
            self.observation=self.image=self.follow_observation=None
            self.follow_candidate_history.clear()
            if self.goal_evidence is not None:
                self.goal_evidence.key=self.goal_evidence.since=self.goal_evidence.last=None
            if self.control is not None:
                self.control.update(map_binding=None,scene_binding=None,agent_mapped_pose=None,
                    goal_observation=None,idle_feedback=None,manual_preview=None,feedback={},
                    reason=fault.reason,planning_status='unavailable',planning_reason=fault.reason,
                    reference_accepted=False,environment_epoch=epoch,environment_fault=fault.reason)

    def _environment_observation_current(self, record):
        if self.environment_fault is not None:return False
        if self.environment_cutoff is None:return True
        identity,cutoff=self.environment_cutoff
        if record.get('local_identity')!=identity:return True
        stamp=record.get('stamp_ns')
        return stamp>cutoff if type(stamp) is int else record.get('stamp_s',-float('inf'))>cutoff/1e9

    def publish_control(self, packet, localization=None, *, map_binding=None, cancelled_request=None, idle_feedback=None,
                        manual_preview=None, agent_mapped_pose=None, environment_epoch=0, environment_fault=None):
        value = dict(request=copy.deepcopy(packet.get('request')), stamp_s=packet['stamp_s'], clock_domain=packet['clock_domain'],
            scene_binding=packet.get('scene_binding'), feedback=copy.deepcopy(packet.get('feedback') or {}),
            reason=packet.get('reason'), reference_accepted=packet.get('result_accepted', False),
            source_packet_binding=packet.get('packet_binding'), scope='shadow_recommendation_not_execution',
            execution_feedback=False, task_completed=False, executable=False)
        value['map_binding'] = copy.deepcopy(map_binding)
        value['cancelled_request'] = copy.deepcopy(cancelled_request)
        value['idle_feedback'] = copy.deepcopy(idle_feedback)
        value['manual_preview'] = copy.deepcopy(manual_preview)
        value['agent_mapped_pose'] = copy.deepcopy(agent_mapped_pose)
        value.update(environment_epoch=environment_epoch,environment_fault=environment_fault)
        proposal=packet.get('proposal') or {}
        value['planning_status']=proposal.get('status')
        value['planning_reason']=proposal.get('reason')
        value['localization'] = None if localization is None else {k: copy.deepcopy(localization.get(k)) for k in
            ('stamp_s', 'session_id', 'generation', 'frame_id', 'health', 'source', 'position', 'quaternion_xyzw')}
        with self.lock:
            if environment_epoch!=self.environment_epoch:return
            self.environment_fault=environment_fault
            if environment_fault is not None:
                value.update(planning_status='unavailable',planning_reason=environment_fault)
            if self.goal_evidence is not None:
                value['goal_observation'] = self.goal_evidence.update(packet,map_binding)
            self.control, self.updated_mono = value, time.monotonic()

    def publish_observation(self, record, image, *, follow_handle=None):
        encoded = None if image is None else base64.b64encode(image).decode()
        if encoded is not None and len(encoded) > 1500000:
            raise ValueError('Agent observation exceeds IPC budget')
        with self.lock:
            if not self._environment_observation_current(record):return
            if follow_handle is None:
                self.observation, self.image = copy.deepcopy(record), encoded
            elif (self.follow and self.follow.get('handle')==follow_handle and
                  self.follow.get('status') in ('starting','running','searching')):
                self.follow_observation=(copy.deepcopy(record),encoded)

    def publish_follow_candidate(self,record,image):
        encoded=None if image is None else base64.b64encode(image).decode()
        if encoded is None or len(encoded)>1500000:raise ValueError('Agent candidate image unavailable or too large')
        with self.lock:
            if self._environment_observation_current(record):
                self.follow_candidate_history.append((copy.deepcopy(record),encoded))

    def follow_candidate(self,observation_id,candidate_binding):
        with self.lock:
            for record,image in reversed(self.follow_candidate_history):
                if (record.get('observation_id')==observation_id and
                        (record.get('follow_candidates') or {}).get('binding')==candidate_binding):
                    return copy.deepcopy(record),image
        return None

    def publish_follow(self, value):
        with self.lock:
            if ((self.follow or {}).get('handle')!=value.get('handle') or
                    value.get('status') not in ('starting','running','searching')):
                self.follow_observation=None
            self.follow=copy.deepcopy(value)

    def snapshot(self, *, with_image=False, with_follow_candidates=False):
        with self.lock:
            if with_follow_candidates:self.follow_candidates_requested_mono=time.monotonic()
            observation,image=self.observation,self.image
            if with_follow_candidates and self.follow_candidate_history and self.control:
                record,encoded=self.follow_candidate_history[-1]
                if (record['clock_domain']==self.control['clock_domain'] and
                        0<=self.control['stamp_s']-record['stamp_s']<=3):
                    observation,image=record,encoded
            if self.follow_observation and self.control:
                record,encoded=self.follow_observation
                if (record['clock_domain']==self.control['clock_domain'] and
                        0<=self.control['stamp_s']-record['stamp_s']<=3):
                    observation,image=record,encoded
            return dict(status='waiting_for_control' if self.control is None else 'available', control=copy.deepcopy(self.control),
                observation=copy.deepcopy(observation), image_base64=image if with_image else None,
                vehicle_context=copy.deepcopy(self.vehicle_context), follow=copy.deepcopy(self.follow),
                seating=None if self.seating is None else self.seating.snapshot(),
                local_skills=dict(follow=self.follow_enabled,seating=self.seating is not None),
                received_age_s=None if self.updated_mono is None else max(0., time.monotonic()-self.updated_mono),
                scope='shadow_recommendation_not_execution', executable=False, execution_feedback=False)

    def follow_candidates_requested(self):
        with self.lock:
            return (self.follow_candidates_requested_mono is not None and
                    time.monotonic()-self.follow_candidates_requested_mono<=2.)


class GoalObservationEvidence:
    """Track fresh localization-based goal criteria, never hardware acceptance."""
    def __init__(self):
        from navigation.control.tracking import load_config
        self.settings = load_config()
        self.key = self.since = self.last = None

    def update(self, packet, map_binding):
        from perception.environment.context import fingerprint
        request, feedback = packet.get('request'), packet.get('feedback') or {}
        diagnostic = packet.get('recorded_goal_diagnostic') or {}
        stamp = feedback.get('stamp_s')
        identity = feedback.get('local_identity')
        ok = (isinstance(request,dict) and feedback.get('status') == 'valid' and
              feedback.get('source') == 'causal_algorithm_poses_not_chassis_feedback' and
              diagnostic.get('request_binding') == fingerprint(request) and
              diagnostic.get('criteria_met') is True and diagnostic.get('stopped') is True and
              type(stamp) in (int,float) and math.isfinite(stamp) and
              type(packet.get('stamp_s')) in (int,float) and
              0 <= packet['stamp_s']-stamp <= self.settings['feedback_max_age_s'] and
              isinstance(identity,list) and len(identity) == 3 and
              (request.get('mode') != 'mapped' or bool((map_binding or {}).get('valid'))))
        if not ok:
            self.key = self.since = self.last = None
            return dict(status='unavailable_or_not_met',execution_feedback=False,physical_acceptance=False)
        key = fingerprint(dict(request=request,identity=identity,clock_domain=packet['clock_domain'],map_binding=map_binding))
        if key != self.key or self.last is not None and (stamp < self.last or stamp-self.last > self.settings['observation_max_age_s']):
            self.key,self.since = key,stamp
        self.last = stamp
        held = stamp-self.since
        return dict(status='criteria_held' if held >= self.settings['goal_dwell_s'] else 'holding',
                    held_s=held,feedback_stamp_s=stamp,clock_domain=packet['clock_domain'],
                    local_identity=copy.deepcopy(identity),request_binding=diagnostic['request_binding'],
                    task_id=request.get('task_id'),goal_revision=request.get('goal_revision'),
                    source=feedback['source'],execution_feedback=False,physical_acceptance=False)


def validate(kind,value):
    from navigation.planning.online import goal_request
    if kind=='goal':return goal_request(value)
    if kind=='cancel':
        if (not isinstance(value,dict) or set(value)!={'task_id','goal_revision'} or not isinstance(value['task_id'],str) or
                type(value['goal_revision']) is not int or value['goal_revision']<0):raise ValueError('Invalid cancellation')
    elif kind=='override':
        if (not isinstance(value,dict) or set(value)!={'manual_override','emergency_stop'} or
                any(type(v) is not bool for v in value.values())):raise ValueError('Override values must be Boolean')
    else:raise ValueError('Only goal, cancel and override are supported')
    return copy.deepcopy(value)


class CommandInbox:
    def __init__(self,path,safety,*,session=None,agent_execution=False,follow_enabled=False):
        self.agent_state=AgentState()
        self.path=Path(path);self.safety=safety;self.session=session;self.pending=deque();self.lock=threading.Lock();self.done=False
        if agent_execution and (session is None or not session.live or not session.execute):
            raise ValueError('Agent execution requires an explicitly enabled live interactive session')
        self.agent_execution=agent_execution
        self.follow_enabled=follow_enabled
        self.agent_state.follow_enabled=follow_enabled
        self.agent_owner=None
        self.manual_preview_owner=None
        self.route_preempted=False
        if self.path.exists():raise FileExistsError(self.path)
        # Linux sockaddr_un limits the address length even before symlink
        # resolution. The CLI resolves this visible link to its private endpoint.
        self.temporary=tempfile.TemporaryDirectory(prefix='social-nav-command-') if len(os.fsencode(self.path))>=104 else None
        self.address=Path(self.temporary.name)/'command.sock' if self.temporary is not None else self.path
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:self.socket.bind(str(self.address));self.address.chmod(0o600)
        except BaseException:
            self.socket.close()
            if self.temporary is not None:self.temporary.cleanup()
            raise
        if self.temporary is not None:self.path.symlink_to(self.address)
        self.socket.listen(2);self.socket.settimeout(.1)
        self.worker=threading.Thread(target=self._work,name='runtime-commands',daemon=True);self.worker.start()

    def offer(self,kind,value):
        value=validate(kind,value)
        with self.lock:
            if len(self.pending)>=64:raise RuntimeError('Command inbox is full')
            self.agent_owner=None
            self.route_preempted=True
            if self.session is not None:self.session.note_command(kind,value)
            self.safety.command(kind,value);self.pending.append((kind,value))

    def manual_preview(self, value):
        """Lease a display-only reference goal, never submit a safety or chassis command."""
        if self.session is None or not self.session.live or self.session.execute:
            raise ValueError('Manual preview requires a live session without hardware output')
        if (not isinstance(value,dict) or set(value)!={'owner','request'} or
                not isinstance(value['owner'],str) or not 1<=len(value['owner'])<=128):
            raise ValueError('Invalid manual preview envelope')
        owner=value['owner'];request=value['request']
        if request is not None:
            request=validate('goal',request)
            if (request.get('mode')!='mapped' or request['request_id'].split(':',1)[0]!=owner):
                raise ValueError('Manual preview must be a bound mapped goal')
        now=time.monotonic()
        with self.lock:
            if (self.agent_owner or {}).get('manual') and request is not None:
                raise ValueError('Manual follow preview is active')
            old=self.manual_preview_owner
            if old is not None and old['expires_mono']<=now:old=self.manual_preview_owner=None
            if old is not None and old['owner']!=owner:
                raise ValueError('Manual preview owned by another session')
            self.manual_preview_owner=(None if request is None else
                dict(owner=owner,request=copy.deepcopy(request),expires_mono=now+3.))
        return dict(accepted=True,executable=False,execution_feedback=False)

    def manual_preview_snapshot(self):
        if self.session is None or not self.session.live or self.session.execute:return None
        with self.lock:
            return copy.deepcopy(self.manual_preview_snapshot_locked())

    def manual_preview_snapshot_locked(self):
        value=self.manual_preview_owner
        if value is not None and value['expires_mono']<=time.monotonic():
            self.manual_preview_owner=None
        return self.manual_preview_owner

    def follow_preview(self, handle, request):
        request=validate('goal',request)
        with self.lock:
            owner=self.agent_owner
            if (self.session is None or not self.session.live or self.session.execute or
                    not owner or not owner.get('manual') or owner['request_id']!=handle['call_id'] or
                    request['mode']!='mapped' or request['task_id']!=owner['internal_task_id'] or
                    request['request_id']!=owner['request_id'] or request['goal_revision']<owner['internal_revision'] or
                    time.monotonic()-owner['heartbeat_mono']>3.):
                raise ValueError('manual_follow_preview_ownership_lost')
            owner['internal_revision']=request['goal_revision']
            self.manual_preview_owner=dict(owner=handle['call_id'],request=copy.deepcopy(request),
                                           expires_mono=owner['heartbeat_mono']+3.)

    def agent_live(self, kind, value):
        """Admit only owned goal/cancel intents into the ordinary safety path."""
        if not self.agent_execution:
            raise ValueError('Agent execution is not enabled')
        value=validate('cancel' if kind=='hold' else kind,value)
        with self.lock:
            if len(self.pending)>=64:raise RuntimeError('Command inbox is full')
            state=self.agent_state.snapshot()
            control=state.get('control') or {}
            active=control.get('request')
            if kind=='goal':
                chassis=self.session.status().get('chassis') or {}
                if chassis.get('state')!='armed' or not chassis.get('status_fresh'):
                    raise ValueError('manual_arming_required')
                if state.get('received_age_s') is None or state['received_age_s']>.5 or not (control.get('map_binding') or {}).get('valid'):
                    raise ValueError('live_navigation_state_unavailable')
                owner=self.agent_owner
                if owner is None:
                    if active is not None:raise ValueError('another_goal_is_active')
                else:
                    evidence=control.get('goal_observation') or {}
                    held=owner.get('held') is True and self._held_goal_stopped(control,owner)
                    active_complete=(not owner.get('held') and isinstance(active,dict) and active.get('request_id')==owner['request_id'] and
                        evidence.get('status')=='criteria_held' and evidence.get('task_id')==owner['task_id'] and
                        evidence.get('goal_revision')==owner['goal_revision'])
                    if (chassis.get('arm_revision')!=owner['arm_revision'] or not (held or active_complete) or
                            value['task_id']!=owner['task_id'] or value['goal_revision']<=owner['goal_revision']):
                        raise ValueError('previous_agent_goal_not_completed_or_preempted')
                if value['mode']!='mapped':raise ValueError('Agent execution requires a mapped goal')
                self.agent_owner=dict(request_id=value['request_id'],task_id=value['task_id'],
                                      goal_revision=value['goal_revision'],arm_revision=chassis.get('arm_revision'))
            else:
                owner=self.agent_owner
                if (owner is None or value['task_id']!=owner['task_id'] or
                        value['goal_revision']!=owner['goal_revision']):
                    raise ValueError('agent_goal_ownership_lost')
                if kind=='hold':
                    chassis=self.session.status().get('chassis') or {}
                    evidence=control.get('goal_observation') or {}
                    if (owner.get('held') or chassis.get('state')!='armed' or not chassis.get('status_fresh') or
                            chassis.get('arm_revision')!=owner['arm_revision'] or
                            state.get('received_age_s') is None or state['received_age_s']>.5 or
                            not (control.get('map_binding') or {}).get('valid') or
                            not isinstance(active,dict) or active.get('request_id')!=owner['request_id'] or
                            evidence.get('status')!='criteria_held' or evidence.get('task_id')!=owner['task_id'] or
                            evidence.get('goal_revision')!=owner['goal_revision']):
                        raise ValueError('agent_goal_not_ready_for_hold')
                    owner['held']=True
                else:self.agent_owner=None
            self.route_preempted=True
            command='cancel' if kind=='hold' else kind
            self.session.note_command(command,value)
            self.safety.command('goal_hold' if kind=='hold' else command,value)
            self.pending.append((command,value))
        return dict(accepted=True,scope='live_goal_intent',execution_feedback=False)

    def agent_live_retarget(self, value):
        """Replace an owned active goal without acquiring or changing its owner."""
        if not self.agent_execution or not isinstance(value,dict) or set(value)!={'previous_revision','goal'}:
            raise ValueError('Invalid Agent retarget request')
        request=validate('goal',value['goal'])
        previous=value['previous_revision']
        with self.lock:
            state=self.agent_state.snapshot();control=state.get('control') or {}
            active=control.get('request') or {};owner=self.agent_owner
            chassis=self.session.status().get('chassis') or {}
            if (len(self.pending)>=64 or not owner or owner.get('kind')=='follow' or owner.get('held') or
                    type(previous) is not int or owner['goal_revision']!=previous or
                    active.get('request_id')!=owner['request_id'] or active.get('goal_revision')!=previous or
                    request['request_id']!=owner['request_id'] or request['task_id']!=owner['task_id'] or
                    request['goal_revision']!=previous+1 or request['mode']!='mapped' or
                    any(request.get(k)!=active.get(k) for k in ('frame_id','target_reference','map_snapshot_id','goal_behavior')) or
                    chassis.get('state')!='armed' or not chassis.get('status_fresh') or
                    chassis.get('arm_revision')!=owner['arm_revision'] or
                    state.get('received_age_s') is None or state['received_age_s']>.5 or
                    not (control.get('map_binding') or {}).get('valid')):
                raise ValueError('agent_retarget_ownership_or_state_invalid')
            owner['goal_revision']=request['goal_revision']
            self.route_preempted=True
            self.session.note_command('goal',request)
            self.safety.command('goal',request)
            self.pending.append(('goal',request))
        return dict(accepted=True,scope='live_goal_intent',execution_feedback=False)

    def agent_live_cancel_owned(self, value):
        if not self.agent_execution or not isinstance(value,dict) or set(value)!={'task_id','call_id'}:
            raise ValueError('Invalid owned cancellation')
        with self.lock:
            owner=self.agent_owner
            if (not owner or owner.get('kind')=='follow' or owner['task_id']!=value['task_id'] or
                    owner['request_id']!=value['call_id'] or len(self.pending)>=64):
                raise ValueError('agent_goal_ownership_lost')
            cancel=dict(task_id=owner['task_id'],goal_revision=owner['goal_revision'])
            self.agent_owner=None;self.route_preempted=True
            self.session.note_command('cancel',cancel);self.safety.command('cancel',cancel)
            self.pending.append(('cancel',cancel))
        return dict(accepted=True,scope='live_goal_intent',execution_feedback=False)

    def agent_follow(self, kind, value, *, live, manual=False):
        """Reserve or stop one follow owner; ACK never means the robot moved."""
        from perception.environment.context import fingerprint
        if not self.follow_enabled:raise ValueError('follow_candidates_not_enabled')
        if manual and (live or self.session is None or not self.session.live or self.session.execute):
            raise ValueError('Manual following requires live observation without hardware output')
        if live and not self.agent_execution:raise ValueError('Agent execution is not enabled')
        if not live and self.session is not None and self.session.execute:
            raise ValueError('Agent shadow commands are disabled when hardware execution is enabled')
        if not isinstance(value,dict):raise ValueError('Invalid follow intent')
        with self.lock:
            if len(self.pending)>=64:raise RuntimeError('Command inbox is full')
            state=self.agent_state.snapshot();control=state.get('control') or {}
            if kind=='start':
                if set(value)!={'handle','intent'}:raise ValueError('Invalid follow start')
                handle,intent=value['handle'],value['intent']
                if (not isinstance(handle,dict) or not isinstance(intent,dict) or
                        set(intent)!={'target_description','reference'} or not isinstance(intent['target_description'],str) or
                        not 1<=len(intent['target_description'].strip())<=2048 or
                        any(not isinstance(handle.get(k),str) or not handle[k] or len(handle[k])>128 for k in ('task_id','goal_id','call_id')) or
                        any(type(handle.get(k)) is not int or handle[k]<0 for k in ('plan_revision','goal_revision'))):
                    raise ValueError('follow_handle_or_intent_invalid')
                ref=intent['reference']
                if (not isinstance(ref,dict) or ref.get('task_id')!=handle['task_id'] or
                        ref.get('task_revision')!=handle['plan_revision'] or ref.get('goal_id')!=handle['goal_id'] or
                        ref.get('description')!=intent['target_description'].strip()):
                    raise ValueError('follow_agent_reference_invalid')
                candidate=self.agent_state.follow_candidate(ref.get('observation_id'),ref.get('candidate_binding'))
                if candidate is None:raise ValueError('follow_agent_observation_unavailable')
                from decision.agent.tools.observations import Observation
                from decision.vlm.follow_target import reference as build_reference
                record,encoded=candidate
                observation=Observation(record,base64.b64decode(encoded,validate=True)).verify()
                checked,_,crop=build_reference(observation,ref.get('candidate_id'),intent['target_description'],
                    task_id=handle['task_id'],revision=handle['plan_revision'],goal_id=handle['goal_id'])
                if checked!=ref:raise ValueError('follow_agent_reference_mismatch')
                if (record.get('clock_domain')!=control.get('clock_domain') or
                        record.get('map_binding')!=control.get('map_binding') or
                        type(record.get('stamp_s')) not in (int,float) or record['stamp_s']>control.get('stamp_s',-1)):
                    raise ValueError('follow_agent_observation_context_changed')
                chassis={} if not live else self.session.status().get('chassis') or {}
                if live and (chassis.get('state')!='armed' or not chassis.get('status_fresh')):
                    raise ValueError('manual_arming_required')
                if (state.get('received_age_s') is None or state['received_age_s']>.5 or
                        live and not (control.get('map_binding') or {}).get('valid')):
                    raise ValueError('live_navigation_state_unavailable')
                owner=self.agent_owner
                if owner is None:
                    if control.get('request') is not None:raise ValueError('another_goal_is_active')
                elif (not live or not owner.get('held') or not self._held_goal_stopped(control,owner) or
                      owner['task_id']!=handle['task_id'] or owner['goal_revision']>=handle['goal_revision'] or
                      owner.get('arm_revision')!=chassis.get('arm_revision')):
                    raise ValueError('previous_agent_goal_not_completed_or_preempted')
                if manual and self.manual_preview_snapshot_locked() is not None:
                    raise ValueError('Manual preview owned by another session')
                self.agent_owner=dict(kind='follow',manual=manual,request_id=handle['call_id'],task_id=handle['task_id'],
                    goal_revision=handle['goal_revision'],internal_task_id='follow:'+handle['call_id'],internal_revision=0,
                    arm_revision=chassis.get('arm_revision'),live=live,heartbeat_mono=time.monotonic())
                pending=dict(copy.deepcopy(value),reference_crop=crop)
                if manual:pending['manual']=True
                self.pending.append(('follow_start',pending))
            elif kind in ('finish','cancel','lost'):
                if (set(value)!=({'call_id','reason'} if kind=='lost' else {'call_id'}) or
                        not isinstance(value['call_id'],str) or kind=='lost' and
                        (not isinstance(value['reason'],str) or not value['reason'] or len(value['reason'])>128)):
                    raise ValueError('Invalid follow stop')
                owner=self.agent_owner
                if (owner is None or owner.get('kind')!='follow' or owner['request_id']!=value['call_id'] or
                        owner.get('manual',False)!=manual or owner['live']!=live or
                        kind in ('finish','lost') and owner.get('held')):
                    raise ValueError('agent_goal_ownership_lost')
                chassis={} if not live else self.session.status().get('chassis') or {}
                hold=bool(kind in ('finish','lost') and live and chassis.get('state')=='armed' and
                    chassis.get('status_fresh') and chassis.get('arm_revision')==owner['arm_revision'] and
                    time.monotonic()-owner['heartbeat_mono']<=3.)
                target=self._follow_stop_locked(owner,hold=hold)
                self.pending.append(('follow_stop',dict(call_id=value['call_id'],target=target,
                    reason=value['reason'] if kind=='lost' else kind,lost=kind=='lost',retained=hold)))
            else:raise ValueError('Unknown follow intent')
            self.route_preempted=True
        return dict(accepted=True,scope='follow_intent',executable=False,execution_feedback=False)

    def _follow_stop_locked(self, owner, *, hold):
        target=dict(task_id=owner['internal_task_id'],goal_revision=owner['internal_revision'])
        if owner.get('manual'):
            self.manual_preview_owner=None
        else:
            self.safety.command('goal_hold' if hold else 'cancel',target)
        if hold:owner['held']=True
        else:self.agent_owner=None
        return target

    def follow_goal(self, handle, request):
        request=validate('goal',request)
        with self.lock:
            owner=self.agent_owner
            if (owner is None or owner.get('manual') or owner.get('kind')!='follow' or owner.get('held') or
                    owner['request_id']!=handle['call_id'] or request['task_id']!=owner['internal_task_id'] or
                    request['request_id']!=owner['request_id'] or request['goal_revision']<owner['internal_revision'] or
                    time.monotonic()-owner['heartbeat_mono']>3.):
                return False
            if owner['live']:
                chassis=self.session.status().get('chassis') or {}
                if (chassis.get('state')!='armed' or not chassis.get('status_fresh') or
                        chassis.get('arm_revision')!=owner['arm_revision']):return False
            self.safety.command('goal',request);owner['internal_revision']=request['goal_revision']
            return True

    def follow_owner_valid(self, handle):
        """Check ownership even when a follow task is waiting without a goal."""
        with self.lock:
            owner=self.agent_owner
            if (owner is None or owner.get('kind')!='follow' or owner.get('held') or
                    owner['request_id']!=handle.get('call_id') or
                    time.monotonic()-owner['heartbeat_mono']>3.):
                return False
            if owner['live']:
                chassis=self.session.status().get('chassis') or {}
                return (chassis.get('state')=='armed' and chassis.get('status_fresh') and
                        chassis.get('arm_revision')==owner['arm_revision'])
            return True

    def pause_follow(self, handle):
        """Stop this owner's current goal while keeping the task and arm state."""
        with self.lock:
            owner=self.agent_owner
            if (owner is None or owner.get('kind')!='follow' or owner.get('held') or
                    owner['request_id']!=handle.get('call_id') or
                    time.monotonic()-owner['heartbeat_mono']>3.):
                raise ValueError('follow_owner_or_heartbeat_lost')
            target=dict(task_id=owner['internal_task_id'],goal_revision=owner['internal_revision'])
            if owner.get('manual'):
                self.manual_preview_owner=None
            elif owner['internal_revision']>0:
                self.safety.command('goal_hold' if owner['live'] else 'cancel',target)
            return target

    def follow_failed(self, handle, *, recoverable=False):
        with self.lock:
            owner=self.agent_owner
            if owner is None or owner.get('kind')!='follow' or owner['request_id']!=handle['call_id'] or owner.get('held'):
                return None
            chassis={} if not owner['live'] else self.session.status().get('chassis') or {}
            hold=bool(recoverable and owner['live'] and chassis.get('state')=='armed' and chassis.get('status_fresh') and
                chassis.get('arm_revision')==owner['arm_revision'] and time.monotonic()-owner['heartbeat_mono']<=3.)
            return dict(target=self._follow_stop_locked(owner,hold=hold),retained=hold)

    @staticmethod
    def _held_goal_stopped(control,owner):
        from navigation.control.tracking import load_config
        from numbers import Real
        cancelled=control.get('cancelled_request') or {}
        feedback=control.get('idle_feedback') or {}
        velocity=feedback.get('velocity')
        stamp=feedback.get('stamp_s');now=control.get('stamp_s')
        settings=load_config()
        return (control.get('request') is None and cancelled.get('task_id')==owner.get('internal_task_id',owner['task_id']) and
            cancelled.get('goal_revision')==owner.get('internal_revision',owner['goal_revision']) and
            (control.get('localization') or {}).get('health')==0 and
            feedback.get('local_identity')==[(control.get('localization') or {}).get(k)
                                            for k in ('session_id','generation','frame_id')] and
            feedback.get('status')=='valid' and feedback.get('source')=='causal_algorithm_poses_not_chassis_feedback' and
            isinstance(stamp,Real) and isinstance(now,Real) and math.isfinite(stamp) and math.isfinite(now) and
            0<=now-stamp<=settings['feedback_max_age_s'] and isinstance(velocity,list) and len(velocity)==2 and
            all(isinstance(v,Real) and math.isfinite(v) for v in velocity) and
            abs(velocity[0])<=settings['stopped_linear_mps'] and abs(velocity[1])<=settings['stopped_angular_rps'])

    def route_command(self,kind,value):
        """A manual intent permanently preempts the optional recorded route."""
        value=validate(kind,value)
        with self.lock:
            if self.route_preempted:return False
            self.safety.command(kind,value)
            return True

    def drain(self):
        with self.lock:items=list(self.pending);self.pending.clear();return items

    def _work(self):
        while not self.done:
            try:peer,_=self.socket.accept()
            except socket.timeout:continue
            except OSError:return
            connection=Connection(peer);peer.settimeout(.5)
            try:
                request=connection.recv()
                if set(request)!={'kind','value'}:raise ValueError('Use kind and value')
                if request['kind'] in ('session_status','playback','arm'):
                    if self.session is None:raise ValueError('Interactive session is not enabled')
                    connection.send(self.session.handle(request['kind'],request['value']));continue
                if request['kind'] in ('agent_state','agent_live_state'):
                    live = request['kind']=='agent_live_state'
                    if live and not self.agent_execution:
                        raise ValueError('Agent execution is not enabled')
                    value=request['value']
                    if (not isinstance(value,dict) or not {'with_image'}<=set(value) or
                            set(value)-{'with_image','follow_call_id','with_follow_candidates'} or
                            type(value['with_image']) is not bool or
                            'with_follow_candidates' in value and type(value['with_follow_candidates']) is not bool or
                            'follow_call_id' in value and (not isinstance(value['follow_call_id'],str) or not value['follow_call_id'])):
                        raise ValueError('State query requires a Boolean with_image')
                    state=self.agent_state.snapshot(with_image=value['with_image'],
                        with_follow_candidates=value.get('with_follow_candidates',False))
                    with self.lock:
                        if (self.agent_owner is not None and self.agent_owner.get('kind')=='follow' and self.agent_owner['live']==live and
                                value.get('follow_call_id')==self.agent_owner['request_id']):
                            self.agent_owner['heartbeat_mono']=time.monotonic()
                        if live:
                            state['agent_owner']=copy.deepcopy(self.agent_owner)
                    if live:
                        state.update(scope='live_observation',chassis=self.session.status().get('chassis'))
                    connection.send(state);continue
                if request['kind']=='agent_manual_preview':
                    connection.send(self.manual_preview(request['value']));continue
                if request['kind']=='agent_seating_select':
                    if self.agent_state.seating is None:raise ValueError('seating_observations_not_enabled')
                    connection.send(self.agent_state.seating.select(request['value']));continue
                if request['kind']=='agent_seating_cancel':
                    if self.agent_state.seating is None:raise ValueError('seating_observations_not_enabled')
                    value=request['value']
                    if not isinstance(value,dict) or set(value)!={'selection_id'}:raise ValueError('Invalid seating cancel')
                    connection.send(self.agent_state.seating.cancel(value['selection_id']));continue
                if request['kind']=='agent_live_retarget':
                    connection.send(self.agent_live_retarget(request['value']));continue
                if request['kind']=='agent_live_cancel_owned':
                    connection.send(self.agent_live_cancel_owned(request['value']));continue
                if request['kind'] in ('agent_live_follow_start','agent_live_follow_finish','agent_live_follow_cancel',
                                       'agent_manual_follow_start','agent_manual_follow_finish','agent_manual_follow_cancel','agent_manual_follow_lost',
                                       'agent_live_follow_lost','agent_shadow_follow_start','agent_shadow_follow_finish',
                                       'agent_shadow_follow_cancel','agent_shadow_follow_lost'):
                    parts=request['kind'].split('_')
                    connection.send(self.agent_follow(parts[-1],request['value'],live=parts[1]=='live',manual=parts[1]=='manual'));continue
                if request['kind'] in ('agent_live_goal','agent_live_cancel','agent_live_hold'):
                    kind={'agent_live_goal':'goal','agent_live_cancel':'cancel','agent_live_hold':'hold'}[request['kind']]
                    connection.send(self.agent_live(kind,request['value']));continue
                if request['kind'] in ('agent_shadow_goal','agent_shadow_cancel'):
                    if self.session is not None and self.session.execute:
                        raise ValueError('Agent shadow commands are disabled when hardware execution is enabled')
                    kind = 'goal' if request['kind']=='agent_shadow_goal' else 'cancel'
                    self.offer(kind,request['value'])
                    connection.send(dict(accepted=True,executable=False,execution_feedback=False));continue
                self.offer(request['kind'],request['value'])
                connection.send(dict(accepted=True,executable=False,execution_feedback=False))
            except Exception as error:
                try:connection.send(dict(accepted=False,error=str(error),executable=False))
                except OSError:pass
            finally:connection.close()

    def close(self):
        self.done=True;self.socket.close();self.worker.join(timeout=2);self.path.unlink(missing_ok=True)
        if self.temporary is not None:self.temporary.cleanup()


def send_request(path,kind,value,*,timeout=2.):
    """Submit one intent; the reply acknowledges receipt, never execution."""
    if kind not in ('session_status','playback','arm'):value=validate(kind,value)
    elif not isinstance(value,dict):raise ValueError('Session commands require an object')
    if not 0<timeout<=10:raise ValueError('Command timeout must be in (0,10] seconds')
    return exchange_request(path,kind,value,timeout=timeout)


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--request',required=True);args=p.parse_args()
    request=json.loads(Path(args.request).read_text())
    print(json.dumps(send_request(args.socket,request['kind'],request['value']),ensure_ascii=False))


if __name__=='__main__':main()
