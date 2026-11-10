"""Local navigation-intent IPC. Commands never bypass independent safety."""
import argparse
import base64
from collections import deque
import copy
import json
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

    def configure_vehicle(self, geometry, motion):
        keys = ('allow_reverse','allow_in_place_rotation','min_turning_radius_m','max_forward_mps','max_reverse_mps',
                'max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2','parameters_assumed')
        with self.lock:
            self.vehicle_context = dict(footprint=geometry.footprint.tolist(),base_from_lidar=geometry.base_from_lidar.tolist(),
                source=copy.deepcopy(geometry.source),assumed=geometry.assumed,
                motion={k:copy.deepcopy(motion.get(k)) for k in keys})

    def publish_control(self, packet, localization=None, *, map_binding=None, cancelled_request=None):
        value = dict(request=copy.deepcopy(packet.get('request')), stamp_s=packet['stamp_s'], clock_domain=packet['clock_domain'],
            scene_binding=packet.get('scene_binding'), feedback=copy.deepcopy(packet.get('feedback') or {}),
            reason=packet.get('reason'), reference_accepted=packet.get('result_accepted', False),
            source_packet_binding=packet.get('packet_binding'), scope='shadow_recommendation_not_execution',
            execution_feedback=False, task_completed=False, executable=False)
        value['map_binding'] = copy.deepcopy(map_binding)
        value['cancelled_request'] = copy.deepcopy(cancelled_request)
        value['localization'] = None if localization is None else {k: copy.deepcopy(localization.get(k)) for k in
            ('stamp_s', 'session_id', 'generation', 'frame_id', 'health', 'source', 'position', 'quaternion_xyzw')}
        with self.lock:
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
    def __init__(self,path,safety,*,session=None):
        self.agent_state=AgentState()
        self.path=Path(path);self.safety=safety;self.session=session;self.pending=deque();self.lock=threading.Lock();self.done=False
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
            self.route_preempted=True
            if self.session is not None:self.session.note_command(kind,value)
            self.safety.command(kind,value);self.pending.append((kind,value))

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
