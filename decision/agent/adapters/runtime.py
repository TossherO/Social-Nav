"""Optional bounded RGB preparation and explicit nonexecuting runtime client."""
import base64
import copy
from pathlib import Path
import queue
import socket
import threading
import numpy as np
from runtime.transport import Connection
from ..tools.observations import Observation


class ObservationWorker:
    def __init__(self, publish, *, metrics=None, geometry=None):
        self.publish, self.metrics, self.geometry = publish, metrics, geometry
        self.queue = queue.Queue(1)
        self.next_stamp = -float('inf')
        self.identity = None
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._work, name='agent-observation', daemon=True)
        self.thread.start()

    def submit(self, scene, pixels, rgb_stamp, rgb_frame, camera_info=None):
        identity = [scene.record[k] for k in ('session_id', 'generation', 'frame_id')]
        stamp = scene.record['as_of']
        if self.error or self.closed or self.queue.full() or identity == self.identity and stamp < self.next_stamp:
            return False
        if pixels is None or rgb_stamp is None or not 0 <= stamp-rgb_stamp <= .15:
            return False
        # Shared producer slots are released after submit: copy only RGB. The
        # sealed SceneInputs keeps the already materialized CPU safety grid.
        image = np.array(pixels, dtype=np.uint8, copy=True)
        if self.metrics:
            self.metrics.transfer('host_copy', image.nbytes, 'agent_rgb_observation')
        self.queue.put_nowait((scene, image, rgb_stamp, rgb_frame, copy.deepcopy(camera_info)))
        self.next_stamp, self.identity = stamp+1., identity
        return True

    def _work(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                scene, image, stamp, frame, camera_info = item
                from ..tools.sensors import sensor_context
                sensors = sensor_context(self.geometry,camera_info,rgb_shape=image.shape,rgb_frame=frame,rgb_stamp=stamp)
                observation = Observation.from_scene(scene, geometry=self.geometry, rgb=dict(pixels=image, stamp_s=stamp, frame_id=frame,
                    source=dict(kind='runtime_shared_host_rgb_copy', observation_id=scene.record['observation_id'])), sensors=sensors,
                    include_local_grid=True)
                self.publish(observation.record, observation.image)
        except Exception as error:
            self.error = type(error).__name__

    def close(self):
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.05)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('Agent observation worker did not stop')
        return dict(error=self.error)


class RuntimeClient:
    def __init__(self, path):
        self.path = Path(path)

    def exchange(self, kind, value):
        if kind not in ('agent_state', 'goal', 'cancel'):
            raise ValueError('Agent runtime client cannot issue override or safety-management commands')
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(.5)
        try:
            sock.connect(str(self.path.resolve()))
            connection = Connection(sock)
            wire_kind = {'goal':'agent_shadow_goal','cancel':'agent_shadow_cancel'}.get(kind,kind)
            connection.send(dict(kind=wire_kind, value=value))
            result = connection.recv()
        finally:
            sock.close()
        if result.get('executable') is not False or result.get('execution_feedback') is True:
            raise ValueError('Agent client only supports a nonexecuting shadow runtime')
        return result

    def snapshot(self, *, with_image=False):
        result = self.exchange('agent_state', dict(with_image=with_image))
        if result.get('scope') != 'shadow_recommendation_not_execution':
            raise ValueError('Unexpected runtime scope')
        return result

    @staticmethod
    def observation(value):
        record, image = value.get('observation'), value.get('image_base64')
        if record is None or record['rgb']['status'] == 'valid' and image is None:
            return None
        return Observation(copy.deepcopy(record), None if image is None else base64.b64decode(image, validate=True)).verify()


class LiveRuntimeClient(RuntimeClient):
    """Explicit live intent channel; it cannot arm or manage safety."""
    def exchange(self, kind, value):
        if kind not in ('agent_state', 'goal', 'cancel', 'hold'):
            raise ValueError('Live Agent cannot arm or manage safety')
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(.5)
        try:
            sock.connect(str(self.path.resolve()))
            connection = Connection(sock)
            wire_kind = {'agent_state':'agent_live_state', 'goal':'agent_live_goal', 'cancel':'agent_live_cancel',
                         'hold':'agent_live_hold'}[kind]
            connection.send(dict(kind=wire_kind, value=value))
            result = connection.recv()
        finally:
            sock.close()
        if result.get('accepted') is False:
            from ..tools.navigation import IntentRejected
            raise IntentRejected(result.get('error', 'Live intent rejected'))
        if result.get('execution_feedback') is True:
            raise ValueError('IPC acknowledgements are not execution evidence')
        return result

    def snapshot(self, *, with_image=False):
        result = self.exchange('agent_state', dict(with_image=with_image))
        if result.get('scope') != 'live_observation':
            raise ValueError('Unexpected live runtime scope')
        return result


class ManualRuntimeClient(RuntimeClient):
    """Read-only observation channel; never send any navigation intent."""
    def exchange(self, kind, value):
        if kind != 'agent_state':
            raise ValueError('Manual Agent client is read-only')
        return super().exchange(kind,value)

    def require_observation_only(self):
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        sock.settimeout(.5)
        try:
            sock.connect(str(self.path.resolve()))
            connection=Connection(sock)
            connection.send(dict(kind='session_status',value={}))
            status=connection.recv()
        finally:
            sock.close()
        if status.get('live') is not True or status.get('hardware_enabled') is not False:
            raise ValueError('Manual Agent requires a live observation session without chassis output')
