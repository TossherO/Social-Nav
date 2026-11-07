"""Small interactive-session state beside the existing command and control owners."""
import copy
import json
import socket
import threading
import time


class InteractiveSession:
    def __init__(self, *, live, safety, output, execute=False):
        self.live, self.safety, self.output, self.execute = live, safety, output, execute
        self.lock = threading.RLock()
        self.parent = self.child = None
        if not live:
            self.parent, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.pause_requested = self.paused = self.source_ended = False
        self.epoch = 0
        self.command_generation = 0
        self.preview = self.observation = None
        self.goal_intent = None
        self.offset_s = self.stamp_s = None
        self.playback_epoch = None

    def note_command(self, kind, value):
        with self.lock:
            self.command_generation += 1
            self.preview = None

    def handle(self, kind, value):
        if kind == 'session_status':
            if value != {}:
                raise ValueError('Session status takes an empty object')
            return self.status()
        if kind == 'playback':
            if self.live or set(value) != {'paused'} or type(value['paused']) is not bool:
                raise ValueError('Playback pause is available only for a bag')
            with self.lock:
                if self.source_ended and not value['paused']:
                    raise ValueError('End of bag: restart to seek to another interval')
                self.pause_requested = value['paused']
                self.preview = None
                self.command_generation += 1
                self.parent.send(json.dumps(value).encode())
            return dict(accepted=True, state='pausing' if value['paused'] else 'resuming', executable=False)
        if kind == 'arm':
            if not self.live or not self.execute:
                raise ValueError('This session has no enabled hardware output')
            if set(value) != {'enabled'} or type(value['enabled']) is not bool:
                raise ValueError('Arming requires a Boolean enabled value')
            self.safety.command('arm', value)
            return dict(accepted=True, execution_feedback=False)
        raise ValueError('Unknown interactive session command')

    def update_playback(self, packet):
        with self.lock:
            if packet['epoch'] != self.playback_epoch or packet['paused'] != self.paused:
                self.epoch += 1
                self.preview = None
                self.goal_intent = None
                self.command_generation += 1
            self.playback_epoch = packet['epoch']
            self.paused = self.pause_requested = packet['paused']
            self.source_ended = packet['source_ended']
            self.offset_s, self.stamp_s = packet['offset_s'], packet['stamp_s']

    def publish_observation(self, scene, row, *, goal_intent=None):
        with self.lock:
            self.goal_intent = copy.deepcopy(goal_intent)
            self.observation = dict(localization=copy.deepcopy(row),
                prior=None if scene is None else copy.deepcopy(scene.record['prior']),
                scene_stamp_s=None if scene is None else scene.record['as_of'])

    def publish_preview(self, value, token):
        with self.lock:
            if token != self.command_generation or not self.paused:
                return False
            self.preview = copy.deepcopy(value)
            # One current result; immutable source scenes stay in the runtime.
            (self.output/'preview.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            return True

    def status(self):
        with self.lock:
            chassis=None
            if self.execute:
                try:chassis=json.loads((self.output/'chassis/status.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):pass
                if chassis is not None:
                    stamp=chassis.get('status_mono')
                    age=None if type(stamp) not in (int,float) else time.monotonic()-stamp
                    chassis['status_age_s']=age
                    chassis['status_fresh']=age is not None and 0<=age<=.5
                    if not chassis['status_fresh']:
                        chassis.update(reported_state=chassis.get('state'),state='unavailable',reason='chassis_status_stale')
            return dict(live=self.live, hardware_enabled=self.execute,
                        chassis=chassis,
                        paused=self.paused, pause_requested=self.pause_requested,
                        source_ended=self.source_ended, epoch=self.epoch,
                        stamp_s=self.stamp_s, offset_s=self.offset_s,
                        observation=copy.deepcopy(self.observation), preview=copy.deepcopy(self.preview),
                        goal_intent=copy.deepcopy(self.goal_intent),
                        executable=False, execution_feedback=False)

    def close(self):
        for channel in (self.parent, self.child):
            if channel is not None:
                channel.close()
