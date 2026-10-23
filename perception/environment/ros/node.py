"""Exact-observation cloud/pose join, rolling geometry, ROS queries and display."""
from collections import OrderedDict
import json
from pathlib import Path
import threading
import time

import numpy as np

from perception.mapping_localization.context import pose_matrix
from perception.sensors.pointcloud import pointcloud_fields
from perception.visualization.scans import pointcloud_message
from perception.environment.layers import LocalEnvironment, GridSnapshot, load_config
from perception.environment.context import crop_map, query_geometry, json_safe
from perception.environment.geometry import interpolate_origins


def execute_query(engine, request):
    allowed = {'operation','snapshot_id','frame','bounds','layers','points'}
    if not isinstance(request,dict) or set(request)-allowed or not isinstance(request.get('snapshot_id'),str):
        raise ValueError('A bound snapshot_id and a supported read-only operation are required')
    snapshot = engine.get_snapshot(request['snapshot_id'])
    operation = request.get('operation')
    if operation == 'get_snapshot':
        result = snapshot.metadata
    elif operation == 'crop_map':
        names = request.get('layers',['occupancy'])
        if not isinstance(names,list) or len(names)>8 or any(name not in snapshot.layers for name in names):
            raise ValueError('Unsupported map layer request')
        result = crop_map(snapshot,request['bounds'],frame=request['frame'],layers=names)
    elif operation == 'query_geometry':
        if len(request['points'])>256:
            raise ValueError('Geometry query exceeds the point budget')
        result = query_geometry(snapshot,request['points'],frame=request['frame'])
    else:
        raise ValueError('Unknown environment query operation')
    return json_safe(result)


class EnvironmentNode:
    def __init__(self, rospy):
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        from social_nav_localization.msg import LocalizationState, MapAssociation
        from social_nav_environment.msg import EnvironmentGrid
        from social_nav_environment.srv import QueryEnvironment
        from std_srvs.srv import Trigger
        self.rospy = rospy
        self.config = load_config(rospy.get_param('~config', None))
        self.engine = LocalEnvironment(self.config)
        self.lock = threading.RLock()
        self.engine_lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.clouds,self.poses,self.associations = OrderedDict(),OrderedDict(),OrderedDict()
        self.done = False
        self.processing = False
        self.latest = None
        self.last_scan_stamp = None
        self.scan_durations = {}
        self.status = 'WAITING_FOR_OBSERVATION'
        self.counts = dict(received=0,processed=0,invalid_pose=0,unpaired=0,rate_limited=0,queue_dropped=0,rays_used=0,geometry_errors=0)
        self.timings = []
        self.journal = None
        self.record_scans = rospy.get_param('~record_scans', False)
        prefix = self.config['output_topic']
        self.grid_pub = rospy.Publisher(prefix+'/grid',EnvironmentGrid,queue_size=1)
        self.occ_pub = rospy.Publisher(prefix+'/occupancy',OccupancyGrid,queue_size=1)
        self.cloud_pub = rospy.Publisher(prefix+'/scan',PointCloud2,queue_size=1)
        self.state_pub = rospy.Publisher(prefix+'/state',String,queue_size=1)
        self.subscribers = [rospy.Subscriber(self.config['pointcloud_topic'],PointCloud2,self.cloud,queue_size=2),
            rospy.Subscriber(self.config['localization_topic'],LocalizationState,self.pose,queue_size=32),
            rospy.Subscriber('/localization/map_association',MapAssociation,self.association,queue_size=8)]
        self.services = [rospy.Service(prefix+'/query',QueryEnvironment,self.query),
                         rospy.Service(prefix+'/status',Trigger,self.summary_service)]
        self.worker = threading.Thread(target=self.work,name='environment-geometry',daemon=True)
        self.worker.start()

    def cloud(self,message):
        stamp = message.header.stamp.to_nsec()
        if message.header.frame_id.lstrip('/') != self.config['pointcloud_frame']:
            self.rospy.logerr_throttle(5,'Environment cloud frame differs from explicit sensor alias')
            return
        with self.condition:
            if self.last_scan_stamp is not None and stamp>self.last_scan_stamp:
                self.scan_durations[stamp] = (stamp-self.last_scan_stamp)/1e9
            self.last_scan_stamp = stamp
            self.clouds[stamp] = (message,time.monotonic())
            self.counts['received'] += 1
            while len(self.clouds)>12:
                old,_ = self.clouds.popitem(last=False)
                self.scan_durations.pop(old,None)
                self.counts['queue_dropped'] += 1
            self.condition.notify()

    def pose(self,message):
        if message.child_frame_id != self.config['sensor_frame']:
            return
        with self.condition:
            self.poses[message.header.stamp.to_nsec()] = message
            while len(self.poses)>64:
                self.poses.popitem(last=False)
            self.condition.notify()

    def association(self,message):
        from perception.mapping_localization.ros.context_node import association_row
        with self.lock:
            self.associations[message.header.stamp.to_nsec()] = association_row(message)
            while len(self.associations)>64:
                self.associations.popitem(last=False)

    def work(self):
        while True:
            with self.condition:
                if self.done:
                    return
                if not self.clouds:
                    self.condition.wait(.05)
                    continue
                stamp,(cloud,arrived) = next(iter(self.clouds.items()))
                pose = self.poses.get(stamp)
                if (pose is None or stamp not in self.associations) and time.monotonic()-arrived < self.config['pose_wait_s']:
                    self.condition.wait(.02)
                    continue
                del self.clouds[stamp]
                duration = self.scan_durations.pop(stamp,None)
                if pose is None:
                    self.counts['unpaired'] += 1
                    continue
                association = self.associations.get(stamp,{})
                if (association.get('local_session_id'),association.get('local_generation')) != (pose.session_id,pose.generation):
                    association = {}
                self.processing = True
            # ROS callbacks can keep receiving current observations during geometry work.
            try:
                with self.engine_lock:
                    self.process(stamp,cloud,pose,duration,association)
            finally:
                with self.condition:
                    self.processing = False

    def process(self,stamp,cloud,pose,duration,association):
        if pose.health != pose.VALID:
            self.engine.reset()
            self.status = 'LOCAL_INVALID'
            self.counts['invalid_pose'] += 1
            self.clear_display(pose)
            self.publish_status()
            return
        if (self.engine.last_stamp is not None and pose.session_id == self.engine.identity[0]
                and pose.generation == self.engine.identity[1]
                and stamp/1e9-self.engine.last_stamp < 1/self.config['update_hz']-1e-6):
            self.counts['rate_limited'] += 1
            return
        started = time.monotonic()
        try:
            has_time = any(f.name == 'curvature' for f in cloud.fields)
            fields = pointcloud_fields(cloud,('x','y','z','curvature') if has_time else ('x','y','z'))
            points = np.column_stack([fields[name] for name in ('x','y','z')])
            p,q = pose.pose.position,pose.pose.orientation
            matrix = pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            origins = None
            if has_time:
                with self.lock:
                    samples = []
                    for s,m in reversed(sorted(self.poses.items())):
                        if s>stamp:
                            continue
                        if (m.health != m.VALID or m.session_id != pose.session_id or
                                m.generation != pose.generation or m.header.frame_id != pose.header.frame_id):
                            break
                        samples.append((s/1e9,[m.pose.position.x,m.pose.position.y,m.pose.position.z]))
                    samples.reverse()
                origins = interpolate_origins(fields['curvature'],stamp/1e9,samples,
                                               self.config['max_origin_interpolation_gap_s'])
                duration = float(np.ptp(fields['curvature']))/1000
            snapshot = self.engine.update(points,matrix,pose.header.stamp.to_sec(),(pose.session_id,pose.generation),
                frame=pose.header.frame_id,association=association,scan_duration=duration,ray_origins=origins)
            snapshot.metadata['stamp_ns'] = stamp
            snapshot.metadata.update(localization_source=pose.source,sensor_frame=pose.child_frame_id)
            if self.record_scans:
                from perception.environment.scans import ScanJournal
                if self.journal is None:
                    self.journal = ScanJournal(Path(self.rospy.get_param('~output'))/'scans.sqlite',
                        self.config['scan_recording'], dict(topic=self.config['pointcloud_topic'],
                        sensor_frame=self.config['sensor_frame'], config=self.config,
                        evidence='frontend_accepted_deskewed_points_not_feature_keyframes'))
                self.journal.append(points, matrix, stamp, snapshot.metadata, origins,
                                    fields.get('curvature'))
            self.latest = snapshot
            self.status = snapshot.metadata['geometry_status']
            self.counts['processed'] += 1
            self.counts['rays_used'] += int(snapshot.metadata['rays_used'])
            self.timings.append((time.monotonic()-started)*1000)
            if len(self.timings)>512:
                self.timings.pop(0)
            self.publish(snapshot,cloud,points,matrix)
        except Exception as error:
            self.counts['geometry_errors'] += 1
            self.status = 'GEOMETRY_INVALID: '+str(error)
            self.engine.reset()
            self.clear_display(pose)
            self.rospy.logerr_throttle(5,self.status)
        self.publish_status()

    def clear_display(self,pose):
        if self.latest is None:
            return
        layers = {}
        for key,values in self.latest.layers.items():
            fill = -1 if key == 'occupancy' else False if values.dtype == bool else np.nan
            layers[key] = np.full_like(values,fill)
        meta = dict(self.latest.metadata,stamp_s=pose.header.stamp.to_sec(),
                    snapshot_id='invalid:'+str(pose.header.stamp.to_nsec()),
                    frame_id=pose.header.frame_id,session_id=pose.session_id,generation=pose.generation,
                    geometry_status=self.status,rays_used=False,world_from_lidar=None,association={})
        self.latest = GridSnapshot(meta,layers)
        self.publish(self.latest,None,None,None)

    def publish(self,snapshot,cloud,points,matrix):
        from nav_msgs.msg import OccupancyGrid
        from social_nav_environment.msg import EnvironmentGrid
        meta,layers = snapshot.metadata,snapshot.layers
        output = EnvironmentGrid()
        output.header.stamp = self.rospy.Time.from_sec(meta['stamp_s'])
        output.header.frame_id = meta['frame_id']
        output.snapshot_id,output.session_id,output.generation = meta['snapshot_id'],meta['session_id'],meta['generation']
        output.info.resolution = meta['resolution']
        output.info.height,output.info.width = meta['shape']
        output.info.origin.position.x,output.info.origin.position.y = meta['origin_xy']
        output.info.origin.orientation.w = 1
        output.info.map_load_time = output.header.stamp
        output.rays_used,output.geometry_status = meta['rays_used'],meta['geometry_status']
        output.association_json = json.dumps(meta['association'],allow_nan=False)
        output.metadata_json = json.dumps(meta,allow_nan=False)
        for name,values in layers.items():
            setattr(output,name,values.ravel().tolist())
        self.grid_pub.publish(output)
        self.occ_pub.publish(OccupancyGrid(header=output.header,info=output.info,data=output.occupancy))
        if matrix is None:
            self.cloud_pub.publish(pointcloud_message(np.empty((0,3)),meta['frame_id'],meta['stamp_s']))
        elif self.cloud_pub.get_num_connections():
            finite = points[np.isfinite(points).all(axis=1)]
            finite = finite[::max(1,int(np.ceil(len(finite)/30000)))]
            world = finite @ matrix[:3,:3].T+matrix[:3,3]
            self.cloud_pub.publish(pointcloud_message(world,meta['frame_id'],meta['stamp_s']))

    def summary(self):
        values = np.asarray(self.timings)
        return dict(status=self.status,counts=dict(self.counts),pending=len(self.clouds),busy=self.processing,
                    scan_recording=self.journal.summary() if self.journal is not None else None,
                    timing_scope=('geometry_and_scan_recording_excludes_ros_serialization' if self.record_scans
                                  else 'geometry_update_excludes_ros_serialization'),
                    snapshot_id=self.latest.metadata['snapshot_id'] if self.latest else None,
                    timing_ms=dict(mean=float(values.mean()),p95=float(np.percentile(values,95))) if len(values) else None)

    def publish_status(self):
        from std_msgs.msg import String
        self.state_pub.publish(String(data=json.dumps(self.summary(),allow_nan=False)))

    def summary_service(self,request):
        from std_srvs.srv import TriggerResponse
        with self.lock:
            return TriggerResponse(success=self.counts['geometry_errors']==0,message=json.dumps(self.summary()))

    def query(self,request):
        from social_nav_environment.srv import QueryEnvironmentResponse
        try:
            if len(request.request_json)>8192:
                raise ValueError('Query exceeds the request budget')
            payload = json.loads(request.request_json,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite query')))
            with self.engine_lock:
                result = execute_query(self.engine,payload)
            return QueryEnvironmentResponse(success=True,response_json=json.dumps(result,allow_nan=False))
        except (ValueError,KeyError,TypeError) as error:
            return QueryEnvironmentResponse(success=False,response_json=json.dumps(dict(error=str(error))))

    def close(self):
        with self.condition:
            self.done = True
            self.condition.notify()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            raise RuntimeError('Environment worker did not stop within the shutdown budget')
        if self.journal is not None:
            self.journal.close()
        output = self.rospy.get_param('~output','')
        if output:
            directory = Path(output).resolve()
            root = Path(__file__).resolve().parents[3]
            if not any(p in directory.parents for p in ((root/'outputs').resolve(),Path('/tmp'))):
                raise ValueError('Environment output must be within project outputs or /tmp')
            directory.mkdir(parents=True,exist_ok=True)
            if self.latest is not None:
                self.latest.save(directory/'layers.npz')
            with (directory/'summary.json').open('x') as stream:
                json.dump(dict(self.summary(),config=self.config),stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.write('\n')
        self.engine.close()


def main():
    import rospy
    rospy.init_node('environment')
    node = EnvironmentNode(rospy)
    rospy.on_shutdown(node.close)
    rospy.spin()
