"""Optional latest-only ROS display; each group retains its own source time."""
from collections import Counter, deque, OrderedDict
import copy
import json
import math
import threading
import time

import numpy as np

from navigation.planning.inputs import navigation_grid


GROUPS=('scan','people','map','rgb','navigation')
MAP_BINDING=('map_id','map_frame','association_session','association_generation','map_revision')


class DisplayBuffer:
    """At most one owned/pinned pending item per display group."""
    def __init__(self,settings):
        self.settings=dict(settings);self.condition=threading.Condition()
        self.pending={};self.next={};self.identities={};self.superseded=Counter();self.stopping=False
        self.latest_observation=None

    def observe(self,row,clock,association=None):
        with self.condition:
            self.latest_observation=(dict(row),clock,None if association is None else dict(association))

    def observation(self):
        with self.condition:
            return self.latest_observation

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

    def clear_environment(self,frame,stamp):
        with self.condition:
            if not self.stopping:
                self.pending.pop('map',None)
                self.pending.pop('navigation',None)
                self.pending['map_clear']=(frame,stamp)
                self.identities.pop('map',None)
                self.condition.notify()

    def frozen_basemap(self,paused,snapshot,row=None):
        identity=(paused,None if snapshot is None else (snapshot['reference_binding'],snapshot['job_id']),
                  None if row is None else tuple(row[k] for k in ('session_id','generation','frame_id','stamp_s','health')))
        with self.condition:
            if self.stopping or self.identities.get('frozen_map',(False,None,None))==identity:return
            self.identities['frozen_map']=identity
            self.pending.pop('navigation',None)
            association=None
            if row is not None and self.latest_observation is not None:
                candidate=self.latest_observation[2]
                if candidate is not None and candidate['stamp_s']<=row['stamp_s']:
                    association=candidate
            self.pending['frozen_map']=(paused,snapshot,None if row is None else dict(row),association)
            self.condition.notify()

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify_all()


class RuntimeDisplay:
    def __init__(self,settings,geometry,planning,*,asset=None,reference_max_age_s=5.,trajectory_max_age_s=.5,
                 metrics=None,clock_rate=1.,publish_static=True):
        import rospy
        from sensor_msgs.msg import PointCloud2,Image
        from nav_msgs.msg import Path,OccupancyGrid
        from std_msgs.msg import String
        from visualization_msgs.msg import Marker,MarkerArray
        self.rospy=rospy;rospy.init_node('social_nav_runtime_display',disable_signals=True)
        self.map_tf=None
        self._map_tf_seen=OrderedDict()
        if asset is not None:
            from tf2_msgs.msg import TFMessage
            self.map_tf=rospy.Publisher('/tf',TFMessage,queue_size=4)
        self.Image,self.Path,self.OccupancyGrid=Image,Path,OccupancyGrid
        self.String=String
        latched=('path','occupancy','navigation_basemap','navigation_window','static_basemap','reference_search_map','basemap_source')
        self.publishers={name:rospy.Publisher('/social_nav_runtime_view/'+name,kind,queue_size=1,latch=name in latched)
            for name,kind in [('scan',PointCloud2),('ground_map',PointCloud2),('navigation_basemap',PointCloud2),
                ('navigation_window',Marker),('reference_search_map',PointCloud2),
                ('static_basemap',PointCloud2),('basemap_source',String),('rgb',Image),('path',Path),('lidar_path',Path),
                ('reference',Path),('trajectory',Path),('navigation',MarkerArray),('occupancy',OccupancyGrid),
                ('raw_occupancy',OccupancyGrid),('people',MarkerArray)]}
        self.buffer=DisplayBuffer(settings);self.error=None;self.published=Counter()
        self.publish_ms={name:deque(maxlen=512) for name in (*GROUPS,'map_clear','frozen_map')};self.submit_ms=deque(maxlen=512)
        from .navigation import NavigationView
        from .surface import DisplayGrids
        self.navigation_view=NavigationView(geometry,planning,asset=asset,path_z_m=settings['path_z_m'],
            height_mode=settings['height_mode'],ground_plane_radius_m=settings['ground_plane_radius_m'],
            reference_max_age_s=reference_max_age_s,trajectory_max_age_s=trajectory_max_age_s)
        self.grids=DisplayGrids(metrics,asset=asset);self.last_navigation_grid=None
        self.clock_rate=clock_rate;self.max_grid_age=planning['max_geometry_age_s'];self.publish_static=publish_static
        self.search_surface=None;self.local_surface=None
        self.local_key=None;self.local_frame=None;self.search_key=None;self.basemap_status=None
        self.max_association_age=planning['max_association_age_s']
        self.frozen=False;self.frozen_snapshot=None;self.frozen_row=None;self.frozen_association=None
        self.last_environment_grid=None
        self.navigation_lock=threading.Lock();self.navigation_result=None;self.last_navigation=None;self.last_view=None
        from .people import PeopleScene
        self.people=PeopleScene()
        self.worker=threading.Thread(target=self._work,name='runtime-display',daemon=True);self.worker.start()

    def submit(self,points,row,people,*,rgb=None,rgb_frame=None,grid=None,clock=None,association=None):
        began=time.monotonic();identity=tuple(row[k] for k in ('session_id','generation','frame_id','stamp_s'))
        self.buffer.observe(row,(row['stamp_s'],began) if clock is None else clock,association)
        stride=max(1,int(np.ceil(len(points)/self.buffer.settings['max_scan_points'])))
        self.buffer.offer('scan',identity,lambda:(np.array(points[::stride,:3],copy=True),dict(row)),now=began)
        self.buffer.offer('people',identity,lambda:(copy.deepcopy(people),row['health']!=0),now=began)
        if grid is not None:
            self.environment(grid,now=began)
        if rgb is not None:
            self.buffer.offer('rgb',(rgb_frame,people.get('rgb_stamp_s')),
                lambda:(np.array(rgb,copy=True),people.get('rgb_stamp_s'),rgb_frame),now=began)
        with self.navigation_lock:navigation=self.navigation_result
        if navigation is None:self.buffer.offer('navigation',identity,lambda:(dict(row),None,grid),now=began)
        self.submit_ms.append((time.monotonic()-began)*1000)

    def environment(self,grid,*,now=None):
        self.buffer.offer('map',grid.metadata['snapshot_id'],lambda:grid,now=time.monotonic() if now is None else now)

    def clear_environment(self,frame,stamp):
        self.buffer.clear_environment(frame,stamp)

    def frozen_basemap(self,paused,snapshot=None,*,row=None):
        self.buffer.frozen_basemap(paused,snapshot,row)

    def navigation(self,result,reference,request,scene,row,control,reason,*,now,clock_domain,clock_rate,clock_mono,grid=None,goal_hint=None,search_map=None):
        # Finalized results are replaced, never mutated after publication.
        if self.buffer.settings['navigation_hz']==0:return
        from .navigation import display_packet
        def make_item():
            packet=display_packet(result,reference,request,scene,row,control,reason,
                now=now,clock_domain=clock_domain,clock_rate=clock_rate,clock_mono=clock_mono,goal_hint=goal_hint,search_map=search_map)
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
        if group=='frozen_map':
            was_frozen=self.frozen
            previous_row=self.frozen_row
            self.frozen,self.frozen_snapshot,self.frozen_row,association=item
            if association is not None or previous_row!=self.frozen_row:
                self.frozen_association=association
            if self.last_navigation is not None:
                view=self.navigation_view.update(*self.last_navigation,grid=self.last_navigation_grid,stopped=True)
                if self.frozen:view['text']='BAG PAUSED | see frozen preview'
                self._publish_navigation(view)
                self.last_navigation=None
            if was_frozen and not self.frozen:
                self.last_environment_grid=None
            self._show_basemap()
        elif group=='scan':
            from .scans import pointcloud_message
            from perception.mapping_localization.context import pose_matrix
            points,row=item;valid=row['health']==0
            if valid:
                pose=pose_matrix(row['position'],row['quaternion_xyzw']);points=points@pose[:3,:3].T+pose[:3,3]
                self._publish_map_transform(row)
            else:points=np.empty((0,3))
            frame=row['frame_id']
            self.publishers['scan'].publish(pointcloud_message(points,frame,row['stamp_s']))
            self._show_basemap()
        elif group=='people':
            from .people import marker_array
            people,sensor_only=item
            self.publishers['people'].publish(marker_array(self.people.update(people,sensor_only=sensor_only) or []))
        elif group=='map_clear':
            if self.last_environment_grid is not None:
                from perception.environment.layers import GridSnapshot
                from .surface import coloured_cloud
                frame,stamp=item
                previous=self.last_environment_grid
                grid=GridSnapshot(dict(previous.metadata,frame_id=frame,stamp_s=stamp),
                    dict(occupancy=np.full_like(previous.layers['occupancy'],-1)))
                self._publish_occupancy('occupancy',grid)
                self._publish_occupancy('raw_occupancy',grid)
                self._publish_local_cloud(coloured_cloud([],[],[],[],frame,stamp))
            self.last_environment_grid=None
            self.last_navigation=None
            self.last_navigation_grid=None
            self.last_view=None
            self.frozen_snapshot=None
            self._show_basemap()
        elif group=='map':
            from .surface import ground_cloud
            grid=self.grids.get(item)
            self.last_environment_grid=grid
            self._publish_occupancy('occupancy',grid)
            if self.publishers['raw_occupancy'].get_num_connections():
                self._publish_occupancy('raw_occupancy',self.grids.get(item,navigation=False))
            if self.publishers['ground_map'].get_num_connections():
                self._publish_local_cloud(ground_cloud(grid))
            self._show_basemap()
        elif group=='rgb':
            rgb,stamp,frame=item
            if stamp is None or not frame:raise ValueError('Display RGB needs its real observation time/frame')
            image=self._header(self.Image(),frame,stamp)
            image.height,image.width=rgb.shape[:2];image.encoding='rgb8';image.step=image.width*3;image.data=rgb.tobytes()
            self.publishers['rgb'].publish(image)
        else:
            if self.frozen:return
            row,packet,source=item;self.last_navigation=(row,packet);self.last_navigation_grid=self.grids.get(source)
            view=self.navigation_view.update(row,packet,grid=self.last_navigation_grid)
            self._publish_navigation(view)
            self._show_basemap()

    def _show_basemap(self, *, stopped=False, mono=None):
        from perception.environment.navigation import association_reason
        from .navigation import IDENTITY

        mono=time.monotonic() if mono is None else mono
        observation=self.buffer.observation()
        row,clock,association=observation if observation is not None else (None,None,None)
        now=None;scene=None
        snapshot=self.frozen_snapshot if self.frozen else None
        if self.frozen:
            row=self.frozen_row
            association=self.frozen_association
            now=None if row is None else row['stamp_s']
        elif self.last_navigation is not None:
            nav_row,packet=self.last_navigation
            if row is None or (tuple(nav_row[k] for k in IDENTITY)==tuple(row[k] for k in IDENTITY) and
                               nav_row['stamp_s']>row['stamp_s']):
                row=nav_row
            if packet is not None and row is not None and all(packet['row'][k]==row[k] for k in IDENTITY):
                if clock is None or packet['mono']>=clock[1]:
                    now=packet['now_s']+max(0.,mono-packet['mono'])*packet['clock_rate']
                if self.last_view is not None:
                    snapshot=self.last_view['search_map']
                    scene=packet.get('scene')
        if now is None and clock is not None and not self.frozen:
            stamp,anchor=clock
            now=stamp+max(0.,mono-anchor)*self.clock_rate

        available=not stopped and row is not None and row['health']==0 and now is not None
        asset=self.grids.asset
        static_frame=None if asset is None else navigation_grid(asset).metadata['frame_id']
        mapped=available and self._mapped_observation(row,association,now,asset)
        if snapshot is not None and (
                not mapped or snapshot['frame_id']!=static_frame or
                not 0.<=now-snapshot['stamp_s']<=self.navigation_view.reference_max_age_s or
                (not self.frozen and (scene is None or not 0.<=now-scene['as_of']<=self.max_grid_age or
                                      not self._matching_associations(scene['prior'],association)))):
            snapshot=None

        grid=self.last_environment_grid if available else None
        transform=None
        if grid is not None:
            meta=grid.metadata
            if (any(meta[k]!=row[k] for k in IDENTITY) or
                    meta.get('geometry_status') not in ('OBSERVED','GROUND_UNAVAILABLE') or
                    meta['stamp_s']>row['stamp_s']+1e-6 or
                    not -1e-6<=now-meta['stamp_s']<=self.max_grid_age):
                grid=None
            elif asset is not None:
                if (not mapped or not self._matching_associations(meta.get('association'),association) or
                        association_reason(grid,asset,self.max_association_age) is not None or
                        now-meta['association']['measurement_stamp_s']>self.max_association_age):
                    grid=None
                else:
                    transform=np.asarray(meta['association']['map_from_local'])

        self._publish_basemap_layers(grid,transform,snapshot,static_frame)

    def _mapped_observation(self,row,association,now,asset):
        if asset is None or not association or not row.get('has_map_pose') or not association.get('valid'):
            return False
        meta=asset.snapshot.metadata
        stamp=association.get('stamp_s')
        measured=association.get('measurement_stamp_s')
        return (association.get('status')=='PRIOR_ASSOCIATED' and
                bool(association.get('association_session')) and association.get('association_generation') is not None and
                association.get('map_id')==meta.get('map_id') and
                association.get('map_frame')==row.get('map_frame')==meta['frame_id'] and
                association.get('map_revision')==row.get('map_revision') and
                (association.get('local_session_id'),association.get('local_generation'),association.get('local_frame'))==
                (row['session_id'],row['generation'],row['frame_id']) and
                measured is not None and stamp is not None and measured<=stamp<=row['stamp_s'] and
                0<=now-measured<=self.max_association_age)

    @staticmethod
    def _matching_associations(source,current):
        return bool(source and source.get('valid') and all(source.get(k)==current.get(k) for k in MAP_BINDING))

    def _publish_basemap_layers(self,grid,transform,snapshot,static_frame):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker
        from .surface import NavigationGridSurface,SearchMapSurface,coloured_cloud

        local_key=None if grid is None else tuple(grid.metadata[k] for k in
            ('session_id','generation','frame_id','stamp_s','snapshot_id'))
        if local_key!=self.local_key:
            marker=Marker(ns='navigation_window',id=0,type=Marker.LINE_STRIP,action=Marker.DELETE)
            marker.pose.orientation.w=1.
            if grid is None:
                cloud=coloured_cloud([],[],[],[],self.local_frame,0.)
            else:
                if self.local_surface is None:
                    self.local_surface=NavigationGridSurface(self.grids.asset)
                cloud,boundary=self.local_surface.layer(grid,transform)
                marker.action=Marker.ADD
                marker.scale.x=.045
                marker.color.r,marker.color.g,marker.color.b,marker.color.a=(.35,1.,.86,1.)
                marker.points=[Point(*map(float,p)) for p in boundary]
            marker.header=cloud.header
            self.publishers['navigation_basemap'].publish(cloud)
            self.publishers['navigation_window'].publish(marker)
            self.local_key=local_key;self.local_frame=cloud.header.frame_id

        search_key=None if snapshot is None else (snapshot['reference_binding'],snapshot['job_id'])
        if search_key!=self.search_key:
            if snapshot is None:
                cloud=coloured_cloud([],[],[],[],static_frame,0.)
            else:
                if self.search_surface is None:
                    self.search_surface=SearchMapSurface(self.grids.asset)
                cloud=self.search_surface.cloud(snapshot)
            self.publishers['reference_search_map'].publish(cloud)
            self.search_key=search_key

        status=dict(static_available=static_frame is not None,static_frame=static_frame,local_available=local_key is not None,
            frozen=self.frozen,search_query_id=None if snapshot is None else snapshot['job_id'])
        if status!=self.basemap_status:
            self.publishers['basemap_source'].publish(self.String(data=json.dumps(status)))
            self.basemap_status=status

    def _publish_local_cloud(self,cloud):
        self.publishers['ground_map'].publish(cloud)

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
        for number,name,color,width in [(10,'history',(1.,.61,.24,1.),.06),(11,'reference',(.96,.97,1.,1.),.075),(12,'trajectory',(.25,.9,1.,1.),.10)]:
            data=None if name=='history' else view.get(name)
            points=np.array([[p[0],p[1],height] for _,p,height in view['history']]) if name=='history' else (
                np.empty((0,3)) if data is None else np.c_[data['display_poses'][:,:2],data['heights']])
            curve_frame=frame
            if name=='reference' and data is not None and 'search_xy' in data and self.navigation_view.height_mode=='ground':
                if self.search_surface is None:
                    from .surface import SearchMapSurface
                    self.search_surface=SearchMapSurface(self.grids.asset)
                points=self.search_surface.path_points(data['search_xy'],self.navigation_view.z)
                curve_frame=self.search_surface.frame_id
            segments=line_segments(points)
            if not len(segments):continue
            line=marker(Marker.LINE_LIST,number,color);line.ns=name;line.scale.x=width
            line.header.frame_id=curve_frame
            if data is not None:line.header.stamp=self.rospy.Time.from_sec(data['stamp_s'])
            line.points=[Point(*map(float,p)) for p in segments]
        pose=view['pose']
        body_segments=line_segments(view['footprint']) if pose is not None else []
        if len(body_segments):
            body=marker(Marker.LINE_LIST,0,(1.,.82,.16,1.));body.ns='vehicle';body.scale.x=.065
            body.points=[Point(*map(float,p)) for p in body_segments]
        if pose is not None and np.isfinite(view['heading']).all():
            arrow=marker(Marker.ARROW,1,(1.,.82,.16,1.));arrow.ns='vehicle';arrow.scale.x=.11;arrow.scale.y=.20;arrow.scale.z=.24
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
            if self.publish_static and self.grids.asset is not None:
                from .surface import ground_cloud
                static=ground_cloud(navigation_grid(self.grids.asset),offset=-.015,fill_known=True)
                # A static asset can precede this session's map-to-odometry TF.
                static.header.stamp=self.rospy.Time()
                self.publishers['static_basemap'].publish(static)
            self._show_basemap()
            while True:
                entry=self.buffer.take(timeout=.1)
                if entry is None:
                    if self.last_navigation is not None:self._publish_navigation(self.navigation_view.update(*self.last_navigation,grid=self.last_navigation_grid,stopped=True))
                    self._show_basemap(stopped=True)
                    return
                if entry[0]=='idle':
                    if self.last_navigation is not None and not self.frozen:
                        view=self.navigation_view.update(*self.last_navigation,grid=self.last_navigation_grid)
                        # No sensor input: clear expired predictions without refreshing source times.
                        if any((view[k] is None)!=(self.last_view[k] is None) for k in ('reference','trajectory','goal','search_map')):
                            self._publish_navigation(view)
                    self._show_basemap()
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
