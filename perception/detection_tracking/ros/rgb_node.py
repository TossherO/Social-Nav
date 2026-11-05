"""Optional bounded image association transport; independent of the 3D inference thread."""
from collections import deque
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from ..inference.config import ROOT, load_config, project_path
from ..inference.image_detector import ImageDetector
from ..inference.rgb_association import ImageAssociator, causal_geometry, people_in_lidar
from ..inference.rgb_cache import digest, file_digest
from ..inference.ros_inputs import LatestSlot
from perception.sensors.image import rgb_array
from perception.sensors.image_geometry import camera_model


def observation_key(row):
    return tuple(row.get(key) for key in ('session_id','generation','sequence_id','clock_domain','observation_id','stamp_s'))


class RGBAssociationNode:
    def __init__(self):
        import rospy
        import yaml
        from sensor_msgs.msg import Image,CameraInfo
        from std_msgs.msg import String
        from social_nav_people.msg import ImageAssociation,ImageAssociations
        self.rospy,self.String = rospy,String
        self.Image=Image
        self.Association,self.Associations = ImageAssociation,ImageAssociations
        config = load_config(rospy.get_param('~config',None),rospy.get_param('~local_config',None))
        self.config,self.policy = config,config['image_association']
        for field in ('weights','engine'):self.policy[field] = str(project_path(self.policy[field]))
        backend = rospy.get_param('~backend','')
        if backend:
            self.policy['backend'] = backend
            self.policy['precision'] = 'fp16' if backend=='tensorrt' else 'fp32'
        if self.policy['backend']=='tensorrt':self.policy['precision']='fp16'
        calibration = Path(rospy.get_param('~calibration',str(ROOT/'perception/sensors/calibration/front_camera_lidar.yaml')))
        calibration_bytes=calibration.read_bytes()
        calibration_document=yaml.safe_load(calibration_bytes)
        calibrated = calibration_document['lidar_to_camera_optical']
        self.transform = np.asarray(calibrated['matrix'],dtype=float)
        self.lidar_frame,self.camera_frame = calibrated['source_frame'],calibrated['target_frame']
        self.calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
        self.calibration_record = dict(sha256=self.calibration_sha256,matrix=self.transform.tolist(),
                                      source_frame=self.lidar_frame,target_frame=self.camera_frame)
        self.association_implementation = file_digest(project_path('perception/detection_tracking/inference/rgb_association.py'))
        self.pending,self.condition = LatestSlot(),threading.Condition()
        self.images,self.intrinsics,self.history = deque(maxlen=16),deque(maxlen=32),deque(maxlen=8)
        self.latest,self.retired_scopes = None,deque(maxlen=16)
        self.dropped_late = 0
        self.busy,self.processed,self.last_processed_wall,self.last_processed = False,0,None,None
        self.record_publisher = rospy.Publisher(rospy.get_param('~output_topic',config['ros']['output_topic']+'/rgb_associations')+'/records',String,queue_size=64)
        output = rospy.get_param('~output_topic',config['ros']['output_topic']+'/rgb_associations')
        self.publisher = rospy.Publisher(output,ImageAssociations,queue_size=1)
        self.json_publisher = rospy.Publisher(output+'/json',String,queue_size=1)
        self.image_publisher = rospy.Publisher(output+'/image',Image,queue_size=1)
        self.status = rospy.Publisher(output+'/status',String,queue_size=1,latch=True)
        self.status.publish(String(data=json.dumps(dict(state='initializing'))))
        self.detector = ImageDetector(self.policy,rospy.get_param('~device','0'))
        self.detector.detect(np.zeros((640,640,3),dtype=np.uint8))  # Explicit warm-up; never published as evidence.
        self.matcher = ImageAssociator(self.policy)
        self.subscribers = [
            rospy.Subscriber(rospy.get_param('~people_topic',config['ros']['output_topic']+'/json'),String,self.people,queue_size=1),
            rospy.Subscriber(rospy.get_param('~rgb_topic',config['ros']['output_topic']+'/rgb'),Image,self.image,queue_size=2,
                             buff_size=16*1024*1024),
            rospy.Subscriber(rospy.get_param('~camera_info_topic',config['ros']['camera_info_topic']),CameraInfo,self.camera_info,queue_size=4)]
        self.worker = threading.Thread(target=self.run,name='rgb_association',daemon=True)
        self.worker.start()
        rospy.on_shutdown(self.close)
        self.status_timer = rospy.Timer(rospy.Duration(.2),self.status_callback)
        self.status_callback(None)

    def status_callback(self,_):
        policy={key:value for key,value in self.policy.items() if key not in ('weights','engine')}
        self.status.publish(self.String(data=json.dumps(dict(state='ready',busy=self.busy,
            processed=self.processed,last_processed_wall=self.last_processed_wall,
            last_processed=list(self.last_processed) if self.last_processed else None,
            last_received=list(self.latest) if self.latest else None,
            dropped_busy=self.pending.dropped,dropped_late=self.dropped_late,
            detector=self.detector.metadata,association=policy,
            association_implementation=self.association_implementation,
            calibration=self.calibration_record,
            camera_info_topic=self.rospy.get_param('~camera_info_topic',self.config['ros']['camera_info_topic'])),allow_nan=False)))

    def image(self,message):
        with self.condition:
            self.images.append(message)
            self.condition.notify_all()

    def camera_info(self,message):
        with self.condition:
            self.intrinsics.append(message)
            self.condition.notify_all()

    def people(self,message):
        try:
            row = json.loads(message.data)
            if (not row.get('observation_id') or not row.get('session_id') or
                    not isinstance(row.get('generation'),int) or row['generation'] < 0 or
                    not np.isfinite(row['stamp_s'])):
                raise ValueError('Missing finite observation identity')
            key = observation_key(row)
            with self.condition:
                if self.latest is not None:
                    if key[0] == self.latest[0] and key[1] < self.latest[1]:return
                    if key[:4] == self.latest[:4] and key[-1] <= self.latest[-1]:return
                    if key[:4] != self.latest[:4]:
                        if key[:4] in self.retired_scopes:return
                        self.retired_scopes.append(self.latest[:4])
                self.latest = key
            self.pending.put(row)
        except (ValueError,KeyError,TypeError) as error:
            self.rospy.logwarn_throttle(5,'RGB association rejected observation: '+str(error))

    def inputs(self,row):
        desired = row.get('rgb_stamp_s')
        if desired is None:return None,None
        def paired():
            matches = [m for m in self.images if abs(m.header.stamp.to_sec()-desired)<=1e-6]
            if not matches:return None,None
            image = matches[-1]
            infos = [m for m in self.intrinsics if m.header.stamp <= image.header.stamp and
                     m.header.frame_id.lstrip('/')==self.camera_frame.lstrip('/')]
            return image,max(infos,key=lambda m:m.header.stamp) if infos else None
        with self.condition:
            # Image and CameraInfo use separate ROS connections and may arrive in either order.
            self.condition.wait_for(lambda:all(m is not None for m in paired()) or
                                    self.latest!=observation_key(row) or self.rospy.is_shutdown(),.1)
            return paired()

    def run(self):
        while not self.rospy.is_shutdown() and not self.pending.closed:
            row = self.pending.take()
            if row is None:continue
            self.busy = True
            people,image,info,pixels,detections,cost = [],None,None,None,[],None
            try:
                people = people_in_lidar(row)
                image,info = self.inputs(row)
                if row['detection_frame_id'].lstrip('/') != self.lidar_frame.lstrip('/'):
                    raise ValueError('LiDAR frame does not match calibration')
                if image is not None and image.header.frame_id.lstrip('/') != self.camera_frame.lstrip('/'):
                    raise ValueError('RGB optical frame does not match calibration')
                camera = camera_model(info,[image.width,image.height]) if image is not None else None
                delta = row['stamp_s']-image.header.stamp.to_sec() if image is not None else None
                pixels = rgb_array(image) if image is not None else None
                if pixels is not None and camera:detections,cost = self.detector.detect(pixels)
                geometry = causal_geometry(people,row,list(self.history),image.header.stamp.to_sec(),.05) if image else people
                scope = digest([observation_key(row)[:4],self.calibration_sha256])
                result = self.matcher.step(geometry,detections,self.transform,camera,delta,row['stamp_s'],scope,row['health'])
                reason = ''
            except Exception as error:
                result = dict(status='failed',tracks={p['key']:dict(status='failed',rgb_box=None) for p in people})
                reason = str(error)
                self.matcher.previous = {}
            self.history.append(row)
            with self.condition:
                current = self.latest == observation_key(row)
            now = self.rospy.Time.now().to_sec()
            stale = now > 0 and now-row['stamp_s'] > self.config['ros']['max_input_age_s']
            if not current or stale:
                self.dropped_late += 1
            camera = camera_model(info,[image.width,image.height]) if image is not None else None
            image_input = dict(rgb_stamp_ns=image.header.stamp.to_nsec() if image else None,
                image_size=[image.width,image.height] if image else None,
                raw_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest() if pixels is not None else None,
                camera=camera,camera_info_stamp_ns=info.header.stamp.to_nsec() if camera else None,
                camera_frame=self.camera_frame,calibration_sha256=self.calibration_sha256)
            record=dict(**{key:row[key] for key in ('observation_id','sequence_id','session_id','generation','clock_domain','stamp_s')},
                perception_input_fingerprint=row['input_fingerprint'],image_input=image_input,
                detector_identity=digest(self.detector.metadata),detections=detections,detector_s=cost,
                result=result,reason=reason,published=bool(current and not stale))
            # Archive the original observation even if it is too late for the current view.
            self.record_publisher.publish(self.String(data=json.dumps(record,allow_nan=False)))
            self.processed += 1
            self.last_processed = observation_key(row)
            self.last_processed_wall = time.monotonic()
            if current and not stale:self.publish(row,image,info,people,result,reason,pixels)
            self.busy = False
            self.status_callback(None)

    def publish(self,row,image,info,people,result,reason,pixels=None):
        message = self.Associations()
        message.header.stamp = self.rospy.Time.from_sec(row['stamp_s'])
        message.header.frame_id = self.camera_frame
        for field in ('observation_id','sequence_id','session_id','generation','clock_domain'):
            setattr(message,field,row[field])
        message.has_rgb = image is not None
        if image is not None:
            message.rgb_stamp = image.header.stamp
            message.image_width,message.image_height = image.width,image.height
        camera = camera_model(info,[image.width,image.height]) if image is not None else None
        message.has_camera_info = camera is not None
        if message.has_camera_info:
            message.camera_info_stamp = info.header.stamp
            message.camera_model_fingerprint = digest(dict(camera=camera,frame=self.camera_frame,
                                                          stamp_ns=info.header.stamp.to_nsec()))
        message.perception_input_fingerprint = row['input_fingerprint']
        message.calibration_sha256 = self.calibration_sha256
        message.detector_identity = digest(self.detector.metadata)
        message.status,message.reason = result['status'],reason
        message.dropped_busy,message.dropped_late = self.pending.dropped,self.dropped_late
        tracks = {t['track_id']+'/g'+str(t['lifecycle_generation']):t for t in row.get('tracks',[])}
        for person in people:
            track = tracks.get(person['key'])
            if track is None:continue
            record = result['tracks'][person['key']]
            item = self.Association()
            item.track_id,item.lifecycle_generation = track['track_id'],track['lifecycle_generation']
            item.observed,item.confirmed = person['observed'],person['confirmed']
            item.status,item.compensation = record['status'],record.get('compensation','uncompensated')
            item.has_rgb_box = record['status']=='matched'
            if item.has_rgb_box:item.rgb_box = record['rgb_box']
            item.has_detection_score = 'detection_score' in record
            if item.has_detection_score:item.detection_score = record['detection_score']
            item.has_cost = 'cost' in record
            if item.has_cost:item.cost = record['cost']
            message.associations.append(item)
        self.publisher.publish(message)
        if pixels is not None:
            from perception.visualization.people import render_rgb_associations
            rendered=render_rgb_associations(pixels,people,result)
            display=self.Image();display.header=image.header
            display.height,display.width=rendered.shape[:2]
            display.encoding,display.step,display.data='rgb8',rendered.shape[1]*3,rendered.tobytes()
            self.image_publisher.publish(display)
        self.json_publisher.publish(self.String(data=json.dumps(dict(
            **{key:row[key] for key in ('observation_id','sequence_id','session_id','generation','clock_domain','stamp_s')},
            observation_key=observation_key(row),rgb_stamp_s=image.header.stamp.to_sec() if image else None,
            camera_frame=self.camera_frame,image_size=[image.width,image.height] if image else None,
            camera_info_stamp_ns=info.header.stamp.to_nsec() if message.has_camera_info else None,
            camera_model_fingerprint=message.camera_model_fingerprint or None,
            perception_input_fingerprint=message.perception_input_fingerprint,
            calibration_sha256=message.calibration_sha256,detector_identity=message.detector_identity,
            result=result,reason=reason,dropped_busy=self.pending.dropped,dropped_late=self.dropped_late),allow_nan=False)))

    def close(self):
        self.pending.close()
        with self.condition:self.condition.notify_all()


def main():
    import rospy
    rospy.init_node('social_nav_rgb_association')
    node = RGBAssociationNode()
    rospy.spin()
    node.close()


if __name__=='__main__':main()
