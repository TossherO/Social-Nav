"""Independent serial owner for explicitly enabled, supervised live navigation."""
from collections import deque
import json
import multiprocessing as mp
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import select
import signal
import socket
import time
from .chassis_protocol import FeedbackFrames, encode_command_units
from .telemetry import ChassisTelemetry
from .vehicle import controller_settings


class DriveState:
    """Bounded arming handshake and short-lived CPU safety authorizations."""
    def __init__(self, motion):
        self.motion = motion
        self.controller=controller_settings(motion)
        self.state, self.reason = 'disarmed', 'operator_arm_required'
        self.revision = -1
        self.heartbeat = self.phase = None
        self.decision = None
        self.feedback = deque(maxlen=32)

    def stop(self, reason):
        self.state, self.reason = 'disarmed', reason
        self.decision = None

    def receive(self, packet, now):
        sent = packet.get('mono')
        if type(sent) not in (int, float) or not 0 <= now-sent <= .2:
            self.stop('expired_safety_channel'); return
        if self.heartbeat is not None and sent <= self.heartbeat:
            self.stop('reordered_safety_channel'); return
        self.heartbeat = sent
        revision = packet.get('arm_revision')
        if type(revision) is not int or type(packet.get('arm')) is not bool:
            self.stop('invalid_arm_request'); return
        if revision > self.revision:
            self.revision = revision
            self.stop('operator_disarmed')
            if packet['arm']:
                self.state, self.phase = 'checking_stationary', now
                self.feedback.clear()
        elif not packet['arm']:
            self.stop('operator_disarmed')
        self.decision = packet

    def observe(self, row, now):
        self.feedback.append((now, row))
        if not row['candidate_checksum_matches']:
            self.stop('feedback_checksum_mismatch')
        elif any(row['joystick_mps_rps']) or any(row['joystick_axes_raw']):
            self.stop('joystick_takeover')
        elif self.state == 'armed' and row['mode_raw'] != 1:
            self.stop('controller_mode_changed')
        elif self.state == 'armed':
            v,w=row['velocity_mps_rps']
            low=-self.motion['max_reverse_mps'];high=self.motion['max_forward_mps']
            if self.motion.get('controller') is not None:
                low=-self.controller['feedback_max_linear_mps'];high=-low
            if (not low<=v<=high or
                    abs(w)>self.controller['feedback_max_angular_rps']):
                self.stop('observed_velocity_over_limit')

    def command(self, now):
        if self.state == 'disarmed': return (0, 0, 3)
        if self.heartbeat is None or now-self.heartbeat > .2:
            self.stop('safety_watchdog'); return (0, 0, 3)
        fresh = [r for stamp, r in self.feedback if now-stamp <= .25]
        if not fresh:
            if now-self.phase > .25: self.stop('feedback_timeout')
            return (0, 0, 1 if self.state in ('arming', 'armed') else 3)
        if self.state in ('checking_stationary', 'arming'):
            mode = 3 if self.state == 'checking_stationary' else 1
            if any(any(abs(v) > .01 for v in r['velocity_mps_rps']) for r in fresh):
                self.stop('not_stationary'); return (0, 0, 3)
            if now-self.phase > 2.:
                self.stop('arm_confirmation_timeout'); return (0, 0, 3)
            span=now-min(stamp for stamp,_ in self.feedback if now-stamp<=.25)
            if now-self.phase >= .5 and span>=.2 and len(fresh) >= 4 and all(r['mode_raw'] == mode for r in fresh):
                self.state = 'arming' if mode == 3 else 'armed'
                self.reason = None
                self.phase = now
                self.feedback.clear()
                self.decision = None
            return (0, 0, 1 if self.state in ('arming', 'armed') else 3)
        packet = self.decision or {}
        units, expiry = packet.get('units'), packet.get('expires_mono', -1.)
        if not packet.get('allowed') or not isinstance(expiry, (int, float)) or not now < expiry <= self.heartbeat+.2:
            return (0, 0, 1)
        if not isinstance(units, list) or len(units) != 2 or any(type(v) is not int for v in units):
            self.stop('invalid_wire_target'); return (0, 0, 3)
        v, w = units
        reverse = self.motion['max_reverse_mps'] if self.motion['allow_reverse'] else 0.
        if not -reverse*1000 <= v <= self.motion['max_forward_mps']*1000 or abs(w) > self.motion['max_angular_rps']*1000:
            self.stop('wire_target_over_limit'); return (0, 0, 3)
        return (v, w, 1)


def _owner(channel_fd, port, output, motion):
    import fcntl
    import termios
    channel = socket.socket(fileno=channel_fd.detach()); channel.setblocking(False)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    state = DriveState(motion); parser = FeedbackFrames()
    fd = original = None
    output = Path(output); output.mkdir(exist_ok=False)
    recorder = ChassisTelemetry(output)
    last_status = last_write = None
    status_at = 0.
    limit = state.controller['velocity_limit_field']

    def write(command):
        nonlocal last_write
        frame = encode_command_units(*command[:2], mode=command[2], velocity_limit_mm_s=limit)
        started = time.monotonic()
        if os.write(fd, frame) != len(frame): raise RuntimeError('Partial serial write')
        recorder.emit('wire_target', units=list(command), changed=command != last_write,
                      write_started_mono=started, velocity_mps_rps=[x/1000. for x in command[:2]])
        last_write = command

    def status():
        nonlocal last_status, status_at
        row = dict(state=state.state, reason=state.reason, arm_revision=state.revision,
                   execution_feedback=False, task_completed=False)
        if row != last_status:
            last_status = row
            recorder.emit('state', **row)
            status_at = 0.
        now = time.monotonic()
        if now-status_at >= .2:
            recorder.status(dict(row, status_mono=now, telemetry=recorder.summary()))
            status_at = now

    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        # TIOCEXCL prevents later opens; reject an already open port too.
        import subprocess
        owners = subprocess.run(['fuser', str(Path(port).resolve())], capture_output=True, text=True)
        if owners.returncode != 0 or {int(x) for x in owners.stdout.split()} != {os.getpid()}:
            raise RuntimeError('Serial port is already in use or ownership cannot be verified')
        original = termios.tcgetattr(fd)
        attributes = termios.tcgetattr(fd)
        attributes[:4] = [0, 0, termios.CS8 | termios.CREAD | termios.CLOCAL, 0]
        attributes[4:6] = [termios.B19200, termios.B19200]
        attributes[6][termios.VMIN] = 0; attributes[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attributes)
        while not stopping[0]:
            now = time.monotonic()
            ready = select.select([fd, channel], [], [], .02)[0]
            if channel in ready:
                while True:
                    try: packet = json.loads(channel.recv(4096))
                    except BlockingIOError: break
                    if packet.get('close'): stopping[0] = True; break
                    received = time.monotonic()
                    state.receive(packet, received)
                    recorder.emit('authorization', mono=received, packet=packet)
            if fd in ready:
                data = os.read(fd, 4096)
                if not data: raise RuntimeError('Serial input disconnected')
                received = time.monotonic()
                for row in parser.feed(data):
                    state.observe(row, received)
                    recorder.emit('feedback', mono=received, feedback=row,
                                  timestamp_source='host_receive_monotonic',
                                  device_measurement_stamp_s=None)
            write(state.command(time.monotonic())); status()
            # No IPC, GPU operation or telemetry write grants additional validity.
            if state.heartbeat is not None and time.monotonic()-state.heartbeat > 1.:
                raise RuntimeError('Safety process disconnected')
    except Exception as error:
        state.stop(type(error).__name__+': '+str(error)); status()
    finally:
        state.stop('shutdown' if stopping[0] else state.reason)
        if fd is not None and original is not None:
            end = time.monotonic()+1.
            while time.monotonic() < end:
                try: write((0, 0, 3))
                except OSError: break
                time.sleep(.05)
            termios.tcsetattr(fd, termios.TCSANOW, original)
        if fd is not None: os.close(fd)
        try:status()
        except OSError:pass
        recorder.close(); channel.close()


class ChassisProcess:
    def __init__(self, port, output, motion):
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.channel.setblocking(False)
        self.process = mp.get_context('spawn').Process(target=_owner,
            args=(DupFd(child.fileno()), port, output, motion))
        self.process.start(); child.close()

    def close(self):
        try: self.channel.send(b'{"close":true}')
        except OSError: pass
        self.process.join(timeout=2)
        if self.process.is_alive(): self.process.terminate(); self.process.join(timeout=2)
        if self.process.is_alive(): self.process.kill(); self.process.join(timeout=1)
        self.channel.close()
