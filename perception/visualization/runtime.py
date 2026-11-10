"""Optional latest-only ROS display; each group retains its own source time."""
from collections import Counter, deque, OrderedDict
import copy
import math
import threading
import time

import numpy as np


GROUPS=('scan','people','map','rgb','navigation')


class DisplayBuffer:
    """At most one owned/pinned pending item per display group."""
    def __init__(self,settings):
        self.settings=dict(settings);self.condition=threading.Condition()
        self.pending={};self.next={};self.identities={};self.superseded=Counter();self.stopping=False

    def offer(self,group,identity,make_item,*,now):
        hz=self.settings[group+'_hz']
        with self.condition:
            if (self.stopping or hz==0 or self.identities.get(group)==identity or
                    now+1e-9<self.next.get(group,-np.inf)):return False
            # Keep a fixed phase: scan jitter must not halve a 10Hz display.
            period=1/hz;deadline=self.next.get(group,now)
            self.next[group]=deadline+(math.floor(max(0.,now-deadline)/period)+1)*period
            item=make_item()
            if group in self.pending:self.superseded[group]+=1
            self.pending[group]=item;self.identities[group]=identity;self.condition.notify()
            return True

    def take(self,timeout=None):
        with self.condition:
            if not self.pending and not self.stopping:self.condition.wait_for(lambda:self.pending or self.stopping,timeout)
            if not self.pending and not self.stopping:return ('idle',None)
            if not self.pending:return None
            group=next(iter(self.pending))
            return group,self.pending.pop(group)

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify_all()


class RuntimeDisplay:
    def __init__(self,settings,geometry,planning,*,asset=None,reference_max_age_s=5.,trajectory_max_age_s=.5,metrics=None):
        import rospy
        from sensor_msgs.msg import PointCloud2,Image
        from nav_msgs.msg import Path,OccupancyGrid
        from visualization_msgs.msg import MarkerArray
        self.rospy=rospy;rospy.init_node('social_nav_runtime_display',disable_signals=True)
        self.map_tf=None
        self._map_tf_seen=OrderedDict()
        if asset is not None:
            from tf2_msgs.msg import TFMessage
            self.map_tf=rospy.Publisher('/tf',TFMessage,queue_size=4)
        self.Image,self.Path,self.OccupancyGrid=Image,Path,OccupancyGrid
        self.publishers={name:rospy.Publisher('/social_nav_runtime_view/'+name,kind,queue_size=1,latch=name in ('path','occupancy'))
            for name,kind in [('scan',PointCloud2),('ground_map',PointCloud2),('rgb',Image),('path',Path),('lidar_path',Path),('reference',Path),('trajectory',Path),('navigation',MarkerArray),('occupancy',OccupancyGrid),('raw_occupancy',OccupancyGrid),('people',MarkerArray)]}
        self.buffer=DisplayBuffer(settings);self.error=None;self.published=Counter()
        self.publish_ms={name:deque(maxlen=512) for name in GROUPS};self.submit_ms=deque(maxlen=512)
        from .navigation import NavigationView
        from .surface import DisplayGrids
        self.navigation_view=NavigationView(geometry,planning,asset=asset,path_z_m=settings['path_z_m'],
            height_mode=settings['height_mode'],ground_plane_radius_m=settings['ground_plane_radius_m'],
            reference_max_age_s=reference_max_age_s,trajectory_max_age_s=trajectory_max_age_s)
        self.grids=DisplayGrids(metrics,asset=asset);self.last_navigation_grid=None
        self.navigation_lock=threading.Lock();self.navigation_result=None;self.last_navigation=None;self.last_view=None
        from .people import PeopleScene
        self.people=PeopleScene()
        self.worker=threading.Thread(target=self._work,name='runtime-display',daemon=True);self.worker.start()

    def submit(self,points,row,people,*,rgb=None,rgb_frame=None,grid=None):
        began=time.monotonic();identity=tuple(row[k] for k in ('session_id','generation','frame_id','stamp_s'))
        stride=max(1,int(np.ceil(len(points)/self.buffer.settings['max_scan_points'])))
        self.buffer.offer('scan',identity,lambda:(np.array(points[::stride,:3],copy=True),dict(row)),now=began)
        self.buffer.offer('people',identity,lambda:(copy.deepcopy(people),row['health']!=0),now=began)
        if grid is not None:
            self.buffer.offer('map',grid.metadata['snapshot_id'],lambda:grid,now=began)
        if rgb is not None:
            self.buffer.offer('rgb',(rgb_frame,people.get('rgb_stamp_s')),
                lambda:(np.array(rgb,copy=True),people.get('rgb_stamp_s'),rgb_frame),now=began)
        with self.navigation_lock:navigation=self.navigation_result
        if navigation is None:self.buffer.offer('navigation',identity,lambda:(dict(row),None,grid),now=began)
        self.submit_ms.append((time.monotonic()-began)*1000)

    def navigation(self,result,reference,request,scene,row,control,reason,*,now,clock_domain,clock_rate,clock_mono,grid=None):
        # Finalized results are replaced, never mutated after publication.
        if self.buffer.settings['navigation_hz']==0:return
        from .navigation import display_packet
        def make_item():
            packet=display_packet(result,reference,request,scene,row,control,reason,
                now=now,clock_domain=clock_domain,clock_rate=clock_rate,clock_mono=clock_mono)
            with self.navigation_lock:self.navigation_result=packet
            return dict(row),packet,grid
        self.buffer.offer('navigation',('control',now),make_item,now=time.monotonic())

    def _header(self,message,frame,stamp):
        message.header.frame_id=frame;message.header.stamp=self.rospy.Time.from_sec(stamp)
        return message

    def _publish_map_transform(self,row):
        if getattr(self,'map_tf',None) is None or row['health']!=0 or not row.get('has_map_pose'):return
        # ROS TF rejects duplicate timestamps for a frame pair even when the
        # duplicate came from another display group or localization session.
        stamp=self.rospy.Time.from_sec(row['stamp_s'])
        key=(row['map_frame'],row['frame_id'],stamp.secs,stamp.nsecs)
        if key in self._map_tf_seen:return
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from perception.mapping_localization.context import pose_matrix
        from perception.mapping_localization.runtime import quaternion
        pose=pose_matrix(row['position'],row['quaternion_xyzw'])
        transform=pose_matrix(row['map_position'],row['map_quaternion_xyzw'])@np.linalg.inv(pose)
        message=TransformStamped();message.header.frame_id=row['map_frame'];message.header.stamp=stamp
        message.child_frame_id=row['frame_id']
        message.transform.translation.x,message.transform.translation.y,message.transform.translation.z=map(float,transform[:3,3])
        q=quaternion(transform[:3,:3])
        message.transform.rotation.x,message.transform.rotation.y,message.transform.rotation.z,message.transform.rotation.w=map(float,q)
        # Use the same measured pose as this display, including newer control poses.
        # This never creates an association or changes a measurement timestamp.
        self.map_tf.publish(TFMessage(transforms=[message]))
        self._map_tf_seen[key]=None
        if len(self._map_tf_seen)>4096:self._map_tf_seen.popitem(last=False)

    def _publish(self,group,item):
        if group=='scan':
            from .scans import pointcloud_message
            from perception.mapping_localization.context import pose_matrix
            points,row=item;valid=row['health']==0
            if valid:
                pose=pose_matrix(row['position'],row['quaternion_xyzw']);points=points@pose[:3,:3].T+pose[:3,3]
                self._publish_map_transform(row)
            else:points=np.empty((0,3))
            frame=row['frame_id']
            self.publishers['scan'].publish(pointcloud_message(points,frame,row['stamp_s']))
        elif group=='people':
            from .people import marker_array
            people,sensor_only=item
            self.publishers['people'].publish(marker_array(self.people.update(people,sensor_only=sensor_only) or []))
        elif group=='map':
            from .surface import ground_cloud
            grid=self.grids.get(item)
            self._publish_occupancy('occupancy',grid)
            if self.publishers['raw_occupancy'].get_num_connections():
                self._publish_occupancy('raw_occupancy',self.grids.get(item,navigation=False))
            self.publishers['ground_map'].publish(ground_cloud(grid))
        elif group=='rgb':
            rgb,stamp,frame=item
            if stamp is None or not frame:raise ValueError('Display RGB needs its real observation time/frame')
            image=self._header(self.Image(),frame,stamp)
            image.height,image.width=rgb.shape[:2];image.encoding='rgb8';image.step=image.width*3;image.data=rgb.tobytes()
            self.publishers['rgb'].publish(image)
        else:
            row,packet,source=item;self.last_navigation=(row,packet);self.last_navigation_grid=self.grids.get(source)
            self._publish_navigation(self.navigation_view.update(row,packet,grid=self.last_navigation_grid))

    def _publish_occupancy(self,name,grid):
        message=self._header(self.OccupancyGrid(),grid.metadata['frame_id'],grid.metadata['stamp_s'])
        message.info.resolution=grid.metadata['resolution'];message.info.height,message.info.width=grid.layers['occupancy'].shape
        message.info.origin.position.x,message.info.origin.position.y=grid.metadata['origin_xy'];message.info.origin.orientation.w=1.
        message.data=grid.layers['occupancy'].reshape(-1).tolist();self.publishers[name].publish(message)

    def _publish_navigation(self,view):
        from geometry_msgs.msg import PoseStamped,Point
        from visualization_msgs.msg import Marker,MarkerArray
        from .surface import line_segments
        self._publish_map_transform(view['localization_row'])
        frame,stamp=view['frame_id'],view['stamp_s']
        for name in ('path','lidar_path','reference','trajectory'):
            if self.publishers[name].get_num_connections()==0:continue
            data=view.get(name);source=stamp
            if name=='path':
                rows=[]
                for t,p,height in view['history']:
                    if not np.isfinite(height):rows=[]
                    else:rows.append((t,[*p[:2],height],[0,0,np.sin(p[2]/2),np.cos(p[2]/2)]))
            elif name=='lidar_path':rows=view['raw_history']
            else:
                source=stamp if data is None else data['stamp_s']
                rows=[]
                if data is not None:
                    for (x,y,yaw),height in zip(data['display_poses'],data['heights']):
                        if not np.isfinite(height):break
                        rows.append((source,[x,y,height],[0,0,np.sin(yaw/2),np.cos(yaw/2)]))
            path=self._header(self.Path(),frame,source)
            for t,position,quaternion in rows:
                pose=self._header(PoseStamped(),frame,t)
                pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=map(float,position)
                pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w=map(float,quaternion)
                path.poses.append(pose)
            self.publishers[name].publish(path)
        if self.publishers['navigation'].get_num_connections()==0:
            self.last_view=view;return
        markers=MarkerArray();clear=self._header(Marker(),frame,stamp);clear.action=Marker.DELETEALL;markers.markers.append(clear)
        def marker(kind,number,color):
            m=self._header(Marker(),frame,stamp);m.ns='navigation';m.id=number;m.type=kind;m.action=Marker.ADD
            m.pose.orientation.w=1.;m.color.r,m.color.g,m.color.b,m.color.a=color;markers.markers.append(m);return m
        for number,name,color,width in [(10,'history',(1.,.63,.16,1.),.04),(11,'reference',(.92,.92,.92,1.),.05),(12,'trajectory',(.31,.86,1.,1.),.06)]:
            data=None if name=='history' else view.get(name)
            points=np.array([[p[0],p[1],height] for _,p,height in view['history']]) if name=='history' else (
                np.empty((0,3)) if data is None else np.c_[data['display_poses'][:,:2],data['heights']])
            segments=line_segments(points)
            if not len(segments):continue
            line=marker(Marker.LINE_LIST,number,color);line.ns=name;line.scale.x=width
            if data is not None:line.header.stamp=self.rospy.Time.from_sec(data['stamp_s'])
            line.points=[Point(*map(float,p)) for p in segments]
        pose=view['pose']
        body_segments=line_segments(view['footprint']) if pose is not None else []
        if len(body_segments):
            body=marker(Marker.LINE_LIST,0,(1.,.63,.16,1.));body.ns='vehicle';body.scale.x=.04
            body.points=[Point(*map(float,p)) for p in body_segments]
        if pose is not None and np.isfinite(view['heading']).all():
            arrow=marker(Marker.ARROW,1,(1.,.63,.16,1.));arrow.ns='vehicle';arrow.scale.x=.09;arrow.scale.y=.16;arrow.scale.z=.2
            arrow.points=[Point(*map(float,p)) for p in view['heading']]
        if view['goal'] is not None and np.isfinite(view['goal_z']):
            goal=marker(Marker.SPHERE,2,(.75,.4,1.,1.));goal.ns='goal';goal.scale.x=goal.scale.y=goal.scale.z=.25
            goal.pose.position=Point(float(view['goal'][0]),float(view['goal'][1]),view['goal_z']+.1)
        color={'ready':(.4,1.,.65,1.),'waiting':(1.,.85,.4,1.),'error':(1.,.35,.3,1.)}[view['level']]
        text=marker(Marker.TEXT_VIEW_FACING,3,color)
        text.ns='status';text.text=view['text'];text.scale.z=.25
        text.pose.position=Point(0. if pose is None else float(pose[0]),0. if pose is None else float(pose[1]),view['text_z'])
        self.publishers['navigation'].publish(markers);self.last_view=view

    def _work(self):
        try:
            while True:
                entry=self.buffer.take(timeout=.1)
                if entry is None:
                    if self.last_navigation is not None:self._publish_navigation(self.navigation_view.update(*self.last_navigation,grid=self.last_navigation_grid,stopped=True))
                    return
                if entry[0]=='idle':
                    if self.last_navigation is not None:
                        view=self.navigation_view.update(*self.last_navigation,grid=self.last_navigation_grid)
                        # No sensor input: clear expired predictions without refreshing source times.
                        if any((view[k] is None)!=(self.last_view[k] is None) for k in ('reference','trajectory','goal')):
                            self._publish_navigation(view)
                    continue
                group,item=entry;began=time.monotonic();self._publish(group,item)
                self.publish_ms[group].append((time.monotonic()-began)*1000);self.published[group]+=1
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def close(self):
        self.buffer.close();self.worker.join(timeout=3)
        percentile=lambda rows:None if not rows else float(np.percentile(rows,95))
        return dict(error=self.error,dropped=sum(self.buffer.superseded.values()),
            superseded=dict(self.buffer.superseded),published=dict(self.published),settings=self.buffer.settings,
            submit_wall_p95_ms=percentile(self.submit_ms),
            publish_wall_p95_ms={k:percentile(v) for k,v in self.publish_ms.items()},
            complete=not self.worker.is_alive() and self.error is None)
