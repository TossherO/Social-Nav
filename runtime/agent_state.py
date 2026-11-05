"""Read-only Agent outlet. Querying never refreshes source timestamps."""
import base64
import copy
import threading
import time


class AgentState:
    def __init__(self):
        self.lock = threading.Lock()
        self.control = self.observation = self.image = self.updated_mono = None

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
                received_age_s=None if self.updated_mono is None else max(0., time.monotonic()-self.updated_mono),
                scope='shadow_recommendation_not_execution', executable=False, execution_feedback=False)
