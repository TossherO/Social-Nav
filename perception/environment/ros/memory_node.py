"""Independent bounded RGB worker; local geometry never waits for image indexing."""
from collections import deque, OrderedDict
import json
from pathlib import Path
import threading
import time
import uuid

import yaml

from perception.environment.layers import load_config
from perception.environment.live_memory import PoseBuffer, VisualWriter
from perception.environment.memory import camera_info_record
from perception.environment.context import json_safe
from perception.environment.storage import asset_reference, content_hash, MemoryReader
from perception.mapping_localization.results import file_fingerprint
from perception.mapping_localization.ros.context_node import state_row


class MemoryNode:
    def __init__(self, rospy):
        from sensor_msgs.msg import Image, CameraInfo
        from social_nav_localization.msg import LocalizationState, MapAssociation
        from social_nav_environment.srv import QueryEnvironment
        from std_srvs.srv import Trigger
        from std_msgs.msg import String
        self.rospy = rospy
        self.config = load_config(rospy.get_param('~config'))
        self.settings = self.config['memory']
        self.output = Path(rospy.get_param('~output')).resolve()
        root = Path(__file__).resolve().parents[3]
        if not any(p in self.output.parents for p in ((root/'outputs').resolve(), Path('/tmp'))):
            raise ValueError('Memory output must stay under project outputs or /tmp')
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ('max_image_bytes','queue_size','pose_history_size','max_observations'):
            if type(self.settings[name]) is not int or self.settings[name] <= 0:
                raise ValueError('Memory count/size budgets must be positive integers')
        self.poses = PoseBuffer(self.settings)
        self.queue = deque()
        self.info = deque(maxlen=64)
        self.associations = OrderedDict()
        self.condition = threading.Condition(threading.RLock())
        self.done, self.processing, self.error, self.latest = False, False, None, None
        self.counts = dict(received=0, selected=0, queue_dropped=0, missing_pose=0, missing_camera_info=0, late_pose=0)
        ident = uuid.uuid4().hex
        bag = rospy.get_param('~bag', '')
        calibration = rospy.get_param('~camera_calibration', '')
        self.calibration = yaml.safe_load(Path(calibration).read_text()) if calibration else None
        self.binding = dict(memory_id='memory:'+ident, source_id='ros:'+ident,
            bag=asset_reference(bag) if bag else None, bag_fingerprint=file_fingerprint(bag) if bag else None,
            rgb_topic=self.settings['rgb_topic'], source_search_s=self.settings['source_search_s'],
            calibration_sha256=content_hash(calibration) if calibration else None, calibration=self.calibration,
            source_kind='bag_replay' if bag else 'live_ros', time_scope='causal_observation',
            receipt_time_source='unavailable_in_ros_transport')
        with (self.output/'memory.json').open('x') as stream:
            json.dump(self.binding, stream, ensure_ascii=False, indent=2)
        prefix = self.config['output_topic']+'/memory'
        self.state_pub = rospy.Publisher(prefix+'/state', String, queue_size=1, latch=True)
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self.work, name='visual-memory', daemon=True)
        self.worker.start()
        if not self.ready.wait(10) or self.error:
            raise RuntimeError('Memory index could not start: '+str(self.error))
        self.subscribers = [
            rospy.Subscriber(self.settings['rgb_topic'], Image, self.image, queue_size=2, buff_size=8*1024*1024),
            rospy.Subscriber(self.settings['camera_info_topic'], CameraInfo, self.camera_info, queue_size=4),
            rospy.Subscriber(self.config['localization_topic'], LocalizationState, self.pose, queue_size=64),
            rospy.Subscriber('/localization/map_association', MapAssociation, self.association, queue_size=16)]
        self.services = [rospy.Service(prefix+'/status', Trigger, self.status_service),
                         rospy.Service(prefix+'/query', QueryEnvironment, self.query)]
        self.publish_status()

    def pose(self, message):
        if message.child_frame_id != self.config['sensor_frame']:
            return
        row = state_row(message)
        row.update(stamp_ns=message.header.stamp.to_nsec(), child_frame_id=message.child_frame_id)
        with self.condition:
            if not self.poses.add(row):
                self.counts['late_pose'] += 1
            self.condition.notify()

    def association(self, message):
        from perception.mapping_localization.ros.context_node import association_row
        record = association_row(message)
        record.update(session_id=message.session_id,generation=message.generation)
        with self.condition:
            self.associations[message.header.stamp.to_nsec()] = record
            while len(self.associations) > self.settings['pose_history_size']:
                self.associations.popitem(last=False)

    def camera_info(self, message):
        with self.condition:
            self.info.append(camera_info_record(message))

    def image(self, message):
        with self.condition:
            self.queue.append((message, time.monotonic()))
            self.counts['received'] += 1
            while len(self.queue) > self.settings['queue_size']:
                self.queue.popleft(); self.counts['queue_dropped'] += 1
            self.condition.notify()

    def work(self):
        writer = None
        try:
            writer = VisualWriter(self.output/'observations.sqlite', self.binding, self.settings, self.calibration)
            self.ready.set()
            while True:
                with self.condition:
                    if not self.queue:
                        if self.done:
                            break
                        self.condition.wait(.05)
                        continue
                    image, arrived = self.queue[0]
                    stamp = image.header.stamp.to_nsec()
                    if ((self.poses.last_stamp is None or self.poses.last_stamp < stamp) and
                            time.monotonic()-arrived < self.config['pose_wait_s']):
                        self.condition.wait(.02)
                        continue
                    self.queue.popleft()
                    pose = self.poses.at(stamp)
                    if pose is None:
                        self.counts['missing_pose'] += 1
                        continue
                    candidates = [info for info in self.info if info['stamp_ns'] <= stamp and
                        info['frame_id'] == image.header.frame_id and
                        (info['width'],info['height']) == (image.width,image.height)]
                    intrinsics = max(candidates, key=lambda row: row['stamp_ns']) if candidates else None
                    association = self.associations.get(pose['pose_stamp_ns'], {})
                    if (association.get('session_id'), association.get('generation')) != (pose['session_id'], pose['generation']):
                        association = {}
                    self.processing = True
                # Neither PNG encoding nor SQLite writes hold ROS receive/geometry locks.
                row = writer.append(image, pose, intrinsics=intrinsics, association=association)
                with self.condition:
                    if row is not None:
                        self.latest = row
                        self.counts['selected'] += 1
                        self.counts['missing_camera_info'] += int(intrinsics is None)
                    self.processing = False
                if row is not None:
                    self.publish_status()
        except Exception as error:
            with self.condition:
                self.error = str(error); self.processing = False
            self.rospy.logerr('Visual memory failed: '+str(error))
            self.ready.set()
            self.publish_status()
        finally:
            if writer is not None:
                writer.close()

    def summary(self):
        with self.condition:
            return dict(memory_id=self.binding['memory_id'], error=self.error, counts=dict(self.counts),
                pending=len(self.queue), busy=self.processing,
                latest={key:self.latest[key] for key in ('id','stamp_s','frame_id','session_id','generation')}
                    if self.latest else None)

    def publish_status(self):
        from std_msgs.msg import String
        self.state_pub.publish(String(data=json.dumps(self.summary(), allow_nan=False)))

    def status_service(self, request):
        from std_srvs.srv import TriggerResponse
        return TriggerResponse(success=self.error is None, message=json.dumps(self.summary()))

    def query(self, request):
        from social_nav_environment.srv import QueryEnvironmentResponse
        reader = None
        try:
            if len(request.request_json) > 8192:
                raise ValueError('Memory query exceeds request budget')
            reader = MemoryReader(self.output)
            result = reader.query(json.loads(request.request_json))
            return QueryEnvironmentResponse(success=True, response_json=json.dumps(json_safe(result), allow_nan=False))
        except (ValueError,KeyError,TypeError) as error:
            return QueryEnvironmentResponse(success=False, response_json=json.dumps(dict(error=str(error))))
        finally:
            if reader is not None:
                reader.close()

    def close(self):
        with self.condition:
            self.done = True; self.condition.notify()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            raise RuntimeError('Visual memory worker did not stop')
        with (self.output/'memory_summary.json').open('x') as stream:
            json.dump(self.summary(), stream, ensure_ascii=False, indent=2)


def main():
    import rospy
    rospy.init_node('visual_memory')
    node = MemoryNode(rospy)
    rospy.on_shutdown(node.close)
    rospy.spin()
