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
from .transport import Connection


class AgentState:
    def __init__(self):
        self.lock = threading.Lock()
        self.control = self.observation = self.image = self.updated_mono = None
        self.vehicle_context = None
        self.goal_evidence = None

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

    def publish_control(self, packet, localization=None, *, map_binding=None, cancelled_request=None, idle_feedback=None):
        value = dict(request=copy.deepcopy(packet.get('request')), stamp_s=packet['stamp_s'], clock_domain=packet['clock_domain'],
            scene_binding=packet.get('scene_binding'), feedback=copy.deepcopy(packet.get('feedback') or {}),
            reason=packet.get('reason'), reference_accepted=packet.get('result_accepted', False),
            source_packet_binding=packet.get('packet_binding'), scope='shadow_recommendation_not_execution',
            execution_feedback=False, task_completed=False, executable=False)
        value['map_binding'] = copy.deepcopy(map_binding)
        value['cancelled_request'] = copy.deepcopy(cancelled_request)
        value['idle_feedback'] = copy.deepcopy(idle_feedback)
        proposal=packet.get('proposal') or {}
        value['planning_status']=proposal.get('status')
        value['planning_reason']=proposal.get('reason')
        value['localization'] = None if localization is None else {k: copy.deepcopy(localization.get(k)) for k in
            ('stamp_s', 'session_id', 'generation', 'frame_id', 'health', 'source', 'position', 'quaternion_xyzw')}
        with self.lock:
            if self.goal_evidence is not None:
                value['goal_observation'] = self.goal_evidence.update(packet,map_binding)
            self.control, self.updated_mono = value, time.monotonic()

    def publish_observation(self, record, image):
        encoded = None if image is None else base64.b64encode(image).decode()
        if encoded is not None and len(encoded) > 1500000:
            raise ValueError('Agent observation exceeds IPC budget')
        with self.lock:
            self.observation, self.image = copy.deepcopy(record), encoded

    def snapshot(self, *, with_image=False):
        with self.lock:
            return dict(status='waiting_for_control' if self.control is None else 'available', control=copy.deepcopy(self.control),
                observation=copy.deepcopy(self.observation), image_base64=self.image if with_image else None,
                vehicle_context=copy.deepcopy(self.vehicle_context),
                received_age_s=None if self.updated_mono is None else max(0., time.monotonic()-self.updated_mono),
                scope='shadow_recommendation_not_execution', executable=False, execution_feedback=False)


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
    def __init__(self,path,safety,*,session=None,agent_execution=False):
        self.agent_state=AgentState()
        self.path=Path(path);self.safety=safety;self.session=session;self.pending=deque();self.lock=threading.Lock();self.done=False
        if agent_execution and (session is None or not session.live or not session.execute):
            raise ValueError('Agent execution requires an explicitly enabled live interactive session')
        self.agent_execution=agent_execution
        self.agent_owner=None
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

    @staticmethod
    def _held_goal_stopped(control,owner):
        from navigation.control.tracking import load_config
        from numbers import Real
        cancelled=control.get('cancelled_request') or {}
        feedback=control.get('idle_feedback') or {}
        velocity=feedback.get('velocity')
        stamp=feedback.get('stamp_s');now=control.get('stamp_s')
        settings=load_config()
        return (control.get('request') is None and cancelled.get('task_id')==owner['task_id'] and
            cancelled.get('goal_revision')==owner['goal_revision'] and
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
                if request['kind']=='agent_state':
                    value=request['value']
                    if not isinstance(value,dict) or set(value)!={'with_image'} or type(value['with_image']) is not bool:
                        raise ValueError('State query requires a Boolean with_image')
                    connection.send(self.agent_state.snapshot(with_image=value['with_image']));continue
                if request['kind']=='agent_live_state':
                    if not self.agent_execution:raise ValueError('Agent execution is not enabled')
                    value=request['value']
                    if not isinstance(value,dict) or set(value)!={'with_image'} or type(value['with_image']) is not bool:
                        raise ValueError('State query requires a Boolean with_image')
                    state=self.agent_state.snapshot(with_image=value['with_image'])
                    with self.lock:owner=copy.deepcopy(self.agent_owner)
                    state.update(scope='live_observation',chassis=self.session.status().get('chassis'),agent_owner=owner)
                    connection.send(state);continue
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
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(timeout)
    wire=Connection(sock)
    try:
        sock.connect(str(Path(path).resolve()))
        wire.send(dict(kind=kind,value=value))
        return wire.recv()
    finally:wire.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--request',required=True);args=p.parse_args()
    request=json.loads(Path(args.request).read_text())
    print(json.dumps(send_request(args.socket,request['kind'],request['value']),ensure_ascii=False))


if __name__=='__main__':main()
