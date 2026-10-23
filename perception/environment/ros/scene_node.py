"""Bounded acquisition-time joins and source-bound scene queries for live/bag ROS."""
from collections import deque, OrderedDict
import hashlib
import json
from pathlib import Path
import threading
import time

from perception.environment.context import query_scene
from perception.environment.layers import load_config
from perception.environment.scene import assemble_scene, latest_past, snapshot_from_message, live_scene_status
from perception.environment.live_memory import PoseBuffer
from perception.environment.storage import MemoryReader
from perception.environment.prior import PriorMap, live_prior_status
from perception.sensors.image import rgb_array


class SceneNode:
    def __init__(self, rospy):
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from social_nav_environment.msg import EnvironmentGrid
        from social_nav_localization.msg import LocalizationState, MapAssociation
        from nav_msgs.msg import OccupancyGrid
        from social_nav_environment.srv import QueryEnvironment
        self.rospy,self.String = rospy,String
        self.config = load_config(rospy.get_param('~config'))
        self.settings = self.config['scene']
        for name in ('queue_size','cache_size','history_limit'):
            if type(self.settings[name]) is not int or self.settings[name] <= 0:
                raise ValueError('Scene cache/queue budgets must be positive integers')
        for name in ('join_wait_s','max_rgb_age_s','max_people_age_s','max_people_pose_age_s','max_association_age_s'):
            if not 0 < self.settings[name] < 10:
                raise ValueError('Scene time budgets must be positive and bounded')
        self.output = Path(rospy.get_param('~output')).resolve()
        root = Path(__file__).resolve().parents[3]
        if not any(p in self.output.parents for p in ((root/'outputs').resolve(),Path('/tmp'))):
            raise ValueError('Scene results must stay under project outputs or /tmp')
        self.output.mkdir(parents=True,exist_ok=True)
        self.records = (self.output/'scenes.jsonl').open('x')
        self.clock = 'ros_sim_time' if rospy.get_param('/use_sim_time',False) else 'ros_time'
        self.people_enabled = rospy.get_param('~people_enabled',False)
        self.memory_dir = rospy.get_param('~memory_dir','')
        prior_dir = rospy.get_param('~prior_map','')
        self.prior_map = None
        if prior_dir:
            from perception.mapping_localization.results import map_fingerprint
            self.prior_map = PriorMap(prior_dir,expected_map=map_fingerprint(rospy.get_param('~map_dir')))
        self.association = None
        self.retired_associations = deque(maxlen=32)
        self.displayed_prior = 'uninitialized'
        self.OccupancyGrid = OccupancyGrid
        self.condition = threading.Condition(threading.RLock())
        self.poses = PoseBuffer(self.config['memory'])
        self.queue,self.images,self.people = deque(),deque(maxlen=64),deque(maxlen=64)
        self.cache = OrderedDict()
        self.retired = deque(maxlen=32)
        self.identity,self.last_stamp = None,None
        self.latest = None
        self.done,self.busy,self.error = False,False,None
        self.counts = dict(received=0,published=0,complete=0,queue_dropped=0,rejected_order=0,
                           missing_rgb=0,missing_people=0,invalid=0,prior_valid=0,prior_withdrawals=0)
        prefix = self.settings['output_topic']
        self.pub = rospy.Publisher(prefix,String,queue_size=1)
        self.state_pub = rospy.Publisher(prefix+'/state',String,queue_size=1,latch=True)
        self.prior_pub = rospy.Publisher(prefix+'/prior/occupancy',OccupancyGrid,queue_size=1,latch=True)
        self.subscribers = [rospy.Subscriber(self.config['output_topic']+'/grid',EnvironmentGrid,self.grid,queue_size=2),
            rospy.Subscriber(self.config['memory']['rgb_topic'],Image,self.image,queue_size=2,buff_size=8*1024*1024),
            rospy.Subscriber(self.config['localization_topic'],LocalizationState,self.pose,queue_size=32)]
        if self.prior_map is not None:
            self.subscribers.append(rospy.Subscriber('/localization/map_association',MapAssociation,self.associate,queue_size=8))
        if self.people_enabled:
            self.subscribers.append(rospy.Subscriber(self.settings['people_topic'],String,self.person,queue_size=8))
        self.services = [rospy.Service(prefix+'/query',QueryEnvironment,self.query),
                         rospy.Service(prefix+'/status',Trigger,self.status)]
        self.worker = threading.Thread(target=self.work,name='scene-join',daemon=True)
        self.worker.start()
        self.watchdog = rospy.Timer(rospy.Duration(.25),lambda _:self.publish_status())
        self.publish_status()

    def pose(self, message):
        from perception.mapping_localization.ros.context_node import state_row
        if message.child_frame_id != self.config['sensor_frame']:
            return
        row = state_row(message)
        row.update(stamp_ns=message.header.stamp.to_nsec(),child_frame_id=message.child_frame_id)
        with self.condition:
            self.poses.add(row)
        self.publish_status()

    def grid(self, message):
        with self.condition:
            self.queue.append((message,time.monotonic()))
            self.counts['received'] += 1
            while len(self.queue) > self.settings['queue_size']:
                self.queue.popleft();self.counts['queue_dropped'] += 1
            self.condition.notify()

    def associate(self, message):
        from perception.mapping_localization.ros.context_node import association_row
        row = association_row(message)
        def identity(value):
            return value['local_session_id'],value['local_generation'],value['association_session']
        with self.condition:
            key = identity(row)
            if key in self.retired_associations:
                return
            if self.association is not None:
                old = identity(self.association)
                if key == old and row['stamp_s'] <= self.association['stamp_s']:
                    return
                if key != old:
                    self.retired_associations.append(old)
            self.association = row
        self.publish_status()

    def image(self, message):
        with self.condition:
            self.images.append(dict(stamp_s=message.header.stamp.to_sec(),message=message))
            self.condition.notify()

    def person(self, message):
        try:
            row = json.loads(message.data,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite people JSON')))
            if not isinstance(row,dict) or not isinstance(row.get('stamp_s'),(int,float)):
                raise ValueError('People observations need an acquisition time')
            with self.condition:
                self.people.append(row);self.condition.notify()
        except (ValueError,TypeError) as error:
            self.rospy.logerr_throttle(5,'Scene rejected people input: '+str(error))

    def history(self, meta):
        if not self.memory_dir:
            return [],'not_enabled'
        if not (Path(self.memory_dir)/'observations.sqlite').is_file():
            return [],'not_ready'
        reader = MemoryReader(self.memory_dir)
        try:
            rows = reader.query(dict(operation='query_observations',memory_id=reader.binding['memory_id'],
                session_id=meta['session_id'],generation=meta['generation'],frame=meta['frame_id'],
                as_of=meta['stamp_s'],limit=self.settings['history_limit']))['observations']
            return rows,'valid' if rows else 'empty'
        finally:
            reader.close()

    def work(self):
        try:
            while True:
                with self.condition:
                    if not self.queue:
                        if self.done:
                            break
                        self.condition.wait(.05);continue
                    message,arrived = self.queue[0]
                    stamp = message.header.stamp.to_sec()
                    awaiting_rgb = not any(row['stamp_s'] >= stamp for row in self.images)
                    awaiting_people = self.people_enabled and not any(row['stamp_s'] >= stamp for row in self.people)
                    if not self.done and (awaiting_rgb or awaiting_people) and time.monotonic()-arrived < self.settings['join_wait_s']:
                        self.condition.wait(.02);continue
                    self.queue.popleft()
                    identity = (message.session_id,message.generation,message.header.frame_id)
                    if identity in self.retired or (identity == self.identity and self.last_stamp is not None and stamp <= self.last_stamp):
                        self.counts['rejected_order'] += 1;continue
                    if self.identity is not None and identity != self.identity:
                        self.retired.append(self.identity)
                        self.cache.clear()
                        self.images.clear();self.people.clear()
                        selected,people = None,None
                    else:
                        selected = latest_past(self.images,stamp)
                        people = latest_past(self.people,stamp)
                    self.identity,self.last_stamp = identity,stamp
                    self.busy = True
                snapshot = snapshot_from_message(message)
                rgb = None
                if selected is not None:
                    image = selected['message']
                    rgb = dict(stamp_s=selected['stamp_s'],frame_id=image.header.frame_id,pixels=rgb_array(image),
                        reference=dict(kind='ros_topic',topic=self.config['memory']['rgb_topic'],
                            stamp_ns=image.header.stamp.to_nsec(),frame_id=image.header.frame_id,
                            message_sha256=hashlib.sha256(image.data).hexdigest()))
                observations,memory_status = self.history(snapshot.metadata)
                bundle = assemble_scene(snapshot,clock_domain=self.clock,rgb=rgb,people=people,
                    observations=observations,memory_status=memory_status,settings=self.settings)
                if self.prior_map is not None:
                    bundle = self.prior_map.attach(bundle,self.settings)
                data = json.dumps(bundle.record,ensure_ascii=False,allow_nan=False)
                with self.condition:
                    self.latest = bundle
                    self.cache[bundle.record['observation_id']] = bundle
                    while len(self.cache) > self.settings['cache_size']:
                        self.cache.popitem(last=False)
                    self.counts['published'] += 1
                    self.counts['complete'] += int(bundle.record['health'] == 'valid')
                    self.counts['missing_rgb'] += int(bundle.record['rgb']['status'] != 'valid')
                    self.counts['missing_people'] += int(bundle.record['people']['status'] != 'valid')
                    self.counts['invalid'] += int(bundle.record['health'] == 'invalid')
                    self.counts['prior_valid'] += int(bundle.prior_geometry is not None)
                    self.busy = False
                self.records.write(data+'\n')
                self.pub.publish(self.String(data=data))
                self.publish_status()
        except Exception as error:
            self.error,self.busy = str(error),False
            self.rospy.logerr('Scene worker failed: '+str(error))
            self.publish_status()

    def summary(self):
        with self.condition:
            pose = self.poses.rows[-1] if self.poses.rows else None
            now = self.rospy.Time.now().to_sec()
            return dict(counts=dict(self.counts),pending=len(self.queue),busy=self.busy,error=self.error,
                        latest_id=self.latest.record['observation_id'] if self.latest else None,
                        prior=live_prior_status(self.latest,pose,self.association,now,
                            self.config['consumer']['max_geometry_age_s'],self.settings['max_association_age_s']),
                        current=live_scene_status(self.latest,pose,now,self.config['consumer']['max_geometry_age_s']))

    def publish_status(self):
        with self.condition:
            status = self.summary()
            key = self.latest.record['input_fingerprint'] if status['prior']['ready'] else None
            if key != self.displayed_prior:
                message = self.OccupancyGrid()
                message.header.stamp = self.rospy.Time.now()
                message.header.frame_id = self.latest.record['frame_id'] if self.latest else 'odom'
                message.info.origin.orientation.w = 1.
                if key is None:
                    # Replace the latched display even if an old map TF remains cached.
                    message.info.resolution = 1.
                    message.info.width = message.info.height = 1
                    message.data = [-1]
                    self.counts['prior_withdrawals'] += int(self.displayed_prior != 'uninitialized')
                else:
                    import numpy as np
                    from tf.transformations import quaternion_from_matrix
                    grid = self.latest.prior_geometry
                    meta = grid.metadata
                    local_from_map = np.linalg.inv(np.asarray(meta['map_from_local']))
                    origin = local_from_map @ np.array([*meta['origin_xy'],0.,1.])
                    message.info.origin.position.x,message.info.origin.position.y,message.info.origin.position.z = origin[:3]
                    for name,value in zip(('x','y','z','w'),quaternion_from_matrix(local_from_map)):
                        setattr(message.info.origin.orientation,name,value)
                    message.info.resolution = meta['resolution']
                    message.info.height,message.info.width = meta['shape']
                    message.data = grid.layers['occupancy'].ravel().tolist()
                self.prior_pub.publish(message)
                self.displayed_prior = key
            self.state_pub.publish(self.String(data=json.dumps(status)))

    def status(self, request):
        from std_srvs.srv import TriggerResponse
        return TriggerResponse(success=self.error is None,message=json.dumps(self.summary()))

    def query(self, request):
        from social_nav_environment.srv import QueryEnvironmentResponse
        try:
            if len(request.request_json)>8192:
                raise ValueError('Scene query exceeds request budget')
            payload = json.loads(request.request_json)
            with self.condition:
                bundle = self.cache[payload['observation_id']]
            result = query_scene(bundle,payload,self.config['consumer'])
            return QueryEnvironmentResponse(success=True,response_json=json.dumps(result,allow_nan=False))
        except (KeyError,ValueError,TypeError) as error:
            return QueryEnvironmentResponse(success=False,response_json=json.dumps(dict(error=str(error))))

    def close(self):
        with self.condition:
            self.done = True;self.condition.notify()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            raise RuntimeError('Scene worker did not stop')
        self.records.close()
        if self.latest is not None:
            self.latest.save(self.output)
        (self.output/'summary.json').write_text(json.dumps(self.summary(),indent=2)+'\n')


def main():
    import rospy
    rospy.init_node('perception_scene')
    node = SceneNode(rospy)
    rospy.on_shutdown(node.close)
    rospy.spin()
