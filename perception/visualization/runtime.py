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
NAVIGATION_HEIGHT_TOPICS=('path','reference','trajectory','navigation')
NAVIGATION_TOPICS=('lidar_path',*NAVIGATION_HEIGHT_TOPICS)


class DisplayConnections:
    """Remember reconnects even when the number of subscribers stays the same."""
    def __init__(self,*,save_peers=False):
        self.epoch=0;self.lock=threading.Lock();self.peers=[] if save_peers else None

    def peer_subscribe(self,topic_name,topic_publish,peer_publish):
        with self.lock:
            self.epoch+=1
            if self.peers is not None:self.peers.append(peer_publish)

    def peer_unsubscribe(self,topic_name,num_peers):
        pass

    def take_peers(self):
        with self.lock:
            if not self.peers:return ()
            peers,self.peers=self.peers,[]
        return peers


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

    def clear_hidden_people(self):
        with self.condition:
            if not self.stopping and 'people' in self.identities:
                self.pending.pop('people',None)
                self.identities.pop('people')
                self.next.pop('people',None)
                self.pending['people_clear']=None
                self.condition.notify()

    def people_clear_pending(self):
        with self.condition:
            return 'people_clear' in self.pending

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
        self._map_tf_missed=OrderedDict()
        self._connections={}
        def publisher(name,topic,kind,*,queue_size=1,latch=False):
            connections=self._connections[name]=DisplayConnections(save_peers=name=='tf')
            return rospy.Publisher(topic,kind,queue_size=queue_size,latch=latch,subscriber_listener=connections)
        if asset is not None:
            from tf2_msgs.msg import TFMessage
            self.map_tf=publisher('tf','/tf',TFMessage,queue_size=4)
        self.Image,self.Path,self.OccupancyGrid=Image,Path,OccupancyGrid
        self.String=String
        latched=('path','occupancy','navigation_basemap','navigation_window','static_basemap','reference_search_map','basemap_source')
        self.publishers={name:publisher(name,'/social_nav_runtime_view/'+name,kind,latch=name in latched)
            for name,kind in [('scan',PointCloud2),('ground_map',PointCloud2),('navigation_basemap',PointCloud2),
                ('navigation_window',Marker),('reference_search_map',PointCloud2),
                ('static_basemap',PointCloud2),('basemap_source',String),('rgb',Image),('path',Path),('lidar_path',Path),
                ('reference',Path),('trajectory',Path),('navigation',MarkerArray),('occupancy',OccupancyGrid),
                ('raw_occupancy',OccupancyGrid),('people',MarkerArray)]}
        self.buffer=DisplayBuffer(settings);self.error=None;self.published=Counter()
        self.publish_ms={name:deque(maxlen=512) for name in (*GROUPS,'map_clear','people_clear','frozen_map')};self.submit_ms=deque(maxlen=512)
        from .navigation import NavigationView
        from .surface import DisplayGrids
        self.navigation_view=NavigationView(geometry,planning,asset=asset,path_z_m=settings['path_z_m'],
            vehicle_clearance_m=settings['vehicle_clearance_m'],
            height_mode=settings['height_mode'],ground_plane_radius_m=settings['ground_plane_radius_m'],
            reference_max_age_s=reference_max_age_s,trajectory_max_age_s=trajectory_max_age_s)
        self.grids=DisplayGrids(metrics,asset=asset);self.last_navigation_grid=None
        self.clock_rate=clock_rate;self.max_grid_age=planning['max_geometry_age_s'];self.publish_static=publish_static
        self.search_surface=None;self.local_surface=None
        self.layer_state={};self.local_frame=None;self.environment_meta=None
        self.last_navigation_source=None;self.navigation_epochs=None
        self.max_association_age=planning['max_association_age_s']
        self.frozen=False;self.frozen_snapshot=None;self.frozen_row=None;self.frozen_association=None
        self.last_environment_grid=None
        self.navigation_lock=threading.Lock();self.navigation_result=None;self.last_navigation=None;self.last_view=None
        from .people import MarkerUpdates,PeopleScene
        self.people=PeopleScene(stable_marker_ids=True)
        self._marker_updates={name:MarkerUpdates() for name in ('people','navigation')}
        self._navigation_marker_view=None
        self.worker=threading.Thread(target=self._work,name='runtime-display',daemon=True);self.worker.start()

    def subscribed(self,*topics):
        return any(name in self.publishers and self.publishers[name].get_num_connections()>0 for name in topics)

    def _connection_epoch(self,name):
        connections=getattr(self,'_connections',{}).get(name)
        return 0 if connections is None else connections.epoch

    def submit(self,points,row,people,*,rgb=None,rgb_frame=None,grid=None,clock=None,association=None):
        began=time.monotonic();identity=tuple(row[k] for k in ('session_id','generation','frame_id','stamp_s'))
        self.buffer.observe(row,(row['stamp_s'],began) if clock is None else clock,association)
        if self.subscribed('scan'):
            stride=max(1,int(np.ceil(len(points)/self.buffer.settings['max_scan_points'])))
            self.buffer.offer('scan',identity,lambda:(np.array(points[::stride,:3],copy=True),dict(row)),now=began)
        if self.subscribed('people'):
            self.buffer.offer('people',identity,lambda:(copy.deepcopy(people),dict(row)),now=began)
        else:
            self.buffer.clear_hidden_people()
        if grid is not None:
            self.environment(grid,now=began)
        if rgb is not None and self.subscribed('rgb'):
            self.buffer.offer('rgb',(rgb_frame,people.get('rgb_stamp_s')),
                lambda:(np.array(rgb,copy=True),people.get('rgb_stamp_s'),rgb_frame),now=began)
        with self.navigation_lock:navigation=self.navigation_result
        if navigation is None:self.buffer.offer('navigation',identity,lambda:(dict(row),None,grid),now=began)
        self.submit_ms.append((time.monotonic()-began)*1000)

    def environment(self,grid,*,now=None,host=None):
        if host is not None:self.grids.remember_host(grid,host)
        self.buffer.offer('map',grid.metadata['snapshot_id'],lambda:grid,now=time.monotonic() if now is None else now)

    def clear_environment(self,frame,stamp):
        self.buffer.clear_environment(frame,stamp)

    def frozen_basemap(self,paused,snapshot=None,*,row=None):
        self.buffer.frozen_basemap(paused,snapshot,row)

    def navigation(self,result,reference,request,scene,row,control,reason,*,now,clock_domain,clock_rate,clock_mono,grid=None,goal_hint=None,search_map=None):
        # Finalized results are replaced, never mutated after publication.
        if self.buffer.settings['navigation_hz']==0:return
        def make_item():
            packet=None
            if self.subscribed(*NAVIGATION_TOPICS,'reference_search_map','basemap_source'):
                from .navigation import display_packet
                packet=display_packet(result,reference,request,scene,row,control,reason,
                    now=now,clock_domain=clock_domain,clock_rate=clock_rate,clock_mono=clock_mono,goal_hint=goal_hint,search_map=search_map)
            with self.navigation_lock:self.navigation_result=True
            return dict(row),packet,grid
        self.buffer.offer('navigation',('control',now),make_item,now=time.monotonic())

    def _header(self,message,frame,stamp):
        message.header.frame_id=frame;message.header.stamp=self.rospy.Time.from_sec(stamp)
        return message

    def _publish_map_transform(self,row):
        if getattr(self,'map_tf',None) is None:return
        connections=getattr(self,'_connections',{}).get('tf')
        peers=[] if connections is None else connections.take_peers()
        if row is None or row['health']!=0 or not row.get('has_map_pose'):return
        # ROS TF rejects duplicate timestamps for a frame pair even when the
        # duplicate came from another display group or localization session.
        stamp=self.rospy.Time.from_sec(row['stamp_s'])
        key=(row['map_frame'],row['frame_id'],stamp.secs,stamp.nsecs)
        lifecycle=(row['session_id'],row['generation'],row.get('map_revision'))
        if connections is not None and not self.map_tf.get_num_connections():
            # Keep measurements without computing transforms so an RViz that
            # first connects during a pause can resolve earlier layer stamps.
            self._map_tf_missed[key]=(lifecycle,row)
            if len(self._map_tf_missed)>4096:self._map_tf_missed.popitem(last=False)
            return
        from tf2_msgs.msg import TFMessage
        if peers:
            # A fresh TF buffer must bracket earlier path and cloud stamps as
            # well as the latest pose. Do not replay into older listeners.
            def eligible(source_key,source_lifecycle):
                elapsed=stamp.secs-source_key[2]+(stamp.nsecs-source_key[3])*1e-9
                return source_lifecycle==lifecycle and source_key[:2]==key[:2] and 0<=elapsed<=9.

            history=[transform for source_key,(source_lifecycle,transform) in self._map_tf_seen.items()
                     if eligible(source_key,source_lifecycle)]
            history.extend(self._map_transform(source_row,self.rospy.Time(source_key[2],source_key[3]))
                           for source_key,(source_lifecycle,source_row) in self._map_tf_missed.items()
                           if source_key!=key and source_key not in self._map_tf_seen and eligible(source_key,source_lifecycle))
            history.sort(key=lambda transform:transform.header.stamp)
            if history:
                # rospy exposes a direct transport publisher through the
                # subscription callback. It uses the same serializer as publish().
                with self.map_tf.impl.publock:
                    for publish in peers:publish(TFMessage(transforms=history))
        if key in self._map_tf_seen:return
        message=self._map_transform(row,stamp)
        self.map_tf.publish(TFMessage(transforms=[message]))
        self._map_tf_seen[key]=(lifecycle,message)
        if len(self._map_tf_seen)>4096:self._map_tf_seen.popitem(last=False)

    def _map_transform(self,row,stamp):
        from geometry_msgs.msg import TransformStamped
        from perception.mapping_localization.runtime import quaternion
        from .surface import map_from_local_pose
        transform=map_from_local_pose(row)
        message=TransformStamped();message.header.frame_id=row['map_frame'];message.header.stamp=stamp
        message.child_frame_id=row['frame_id']
        message.transform.translation.x,message.transform.translation.y,message.transform.translation.z=map(float,transform[:3,3])
        q=quaternion(transform[:3,:3])
        message.transform.rotation.x,message.transform.rotation.y,message.transform.rotation.z,message.transform.rotation.w=map(float,q)
        # Use the same measured pose as this display, including newer control poses.
        # This never creates an association or changes a measurement timestamp.
        return message

    def _publish(self,group,item):
        if group=='frozen_map':
            was_frozen=self.frozen
            previous_row=self.frozen_row
            previous_navigation_row=self._current_row()
            self.frozen,self.frozen_snapshot,self.frozen_row,association=item
            if association is not None or previous_row!=self.frozen_row:
                self.frozen_association=association
            if self.frozen:
                if not was_frozen:self._publish_marker_update('navigation',[],reset=True)
                if self.frozen_row is None:
                    self.last_view=None
                    if (previous_navigation_row is not None and self.buffer.settings['navigation_hz']>0 and
                            self.subscribed(*NAVIGATION_TOPICS)):
                        view=self.navigation_view.update(previous_navigation_row,None,grid=self.last_navigation_grid,stopped=True)
                        view['text']='BAG PAUSED | see frozen preview'
                        self._publish_navigation(view)
                else:
                    self._refresh_navigation(new=True)
            if was_frozen and not self.frozen:
                self.last_environment_grid=None
                self.last_navigation=None
                self.last_navigation_source=None
                self.last_navigation_grid=None
            self._show_basemap()
        elif group=='scan':
            if not self.subscribed('scan'):return
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
            if self.buffer.people_clear_pending():return
            people,row=item
            identity=self.people.identity
            specs=self.people.update(people,sensor_only=row['health']!=0)
            if specs is not None:
                self._publish_map_transform(row)
                self._publish_marker_update('people',specs,reset=self.people.identity!=identity)
        elif group=='people_clear':
            self.people.clear()
            self._publish_marker_update('people',[],reset=True)
        elif group=='map_clear':
            frame,stamp=item
            row=self._current_row() or getattr(self,'_path_latch',(None,False))[0]
            if row is not None:
                self._clear_latched_path(dict(row,frame_id=frame,stamp_s=stamp),force=True)
            if self.environment_meta is not None:
                self.environment_meta=dict(self.environment_meta,frame_id=frame,stamp_s=stamp)
            self.last_environment_grid=None
            self.last_navigation=None
            self.last_navigation_grid=None
            self.last_navigation_source=None
            self.navigation_view.history.clear()
            self.navigation_view.raw_history.clear()
            self._publish_marker_update('navigation',[],reset=True)
            self.last_view=None
            self.frozen_snapshot=None
            self._show_basemap()
        elif group=='map':
            self.last_environment_grid=item
            self._show_basemap()
        elif group=='rgb':
            if not self.subscribed('rgb'):return
            rgb,stamp,frame=item
            if stamp is None or not frame:raise ValueError('Display RGB needs its real observation time/frame')
            image=self._header(self.Image(),frame,stamp)
            image.height,image.width=rgb.shape[:2];image.encoding='rgb8';image.step=image.width*3;image.data=rgb.tobytes()
            self.publishers['rgb'].publish(image)
        else:
            if self.frozen:return
            row,packet,source=item;self.last_navigation=(row,packet);self.last_navigation_source=source
            self._refresh_navigation(new=True)
            self._show_basemap()

    def _refresh_navigation(self,*,new=False):
        if self.buffer.settings['navigation_hz']==0:return
        epochs=tuple(self._connection_epoch(name) for name in NAVIGATION_TOPICS)
        reconnected=epochs!=self.navigation_epochs
        interested=self.subscribed(*NAVIGATION_TOPICS)
        row=self._current_row()
        self._clear_latched_path(row)
        if row is None or (self.last_navigation is None and not self.frozen):
            if interested and reconnected and self.last_view is not None:
                self._publish_navigation(self.last_view)
            self.navigation_epochs=epochs
            return

        packet=None if self.frozen or self.last_navigation is None else self.last_navigation[1]
        diagnostics=packet is not None and self.subscribed('reference_search_map','basemap_source')
        if not interested and not diagnostics:
            if new:self.navigation_view.observe(row)
            self.last_navigation_grid=None
            self.last_view=None
            return
        if self.frozen and not new and not reconnected:return
        if new or reconnected:
            # Plane-mode paths and the search diagnostics need no device terrain.
            source=self.last_navigation_source
            if source is None and self.frozen:source=self.last_environment_grid
            self.last_navigation_grid=None
            if (source is not None and self.subscribed(*NAVIGATION_HEIGHT_TOPICS) and row['health']==0 and
                    self.navigation_view.height_mode=='ground' and
                    all(source.metadata[k]==row[k] for k in ('session_id','generation','frame_id')) and
                    -1e-6<=row['stamp_s']-source.metadata['stamp_s']<=self.max_grid_age):
                self.last_navigation_grid=self.grids.get(source)
        view=self.navigation_view.update(row,packet,grid=self.last_navigation_grid,stopped=self.frozen)
        if self.frozen:view['text']='BAG PAUSED | see frozen preview'
        if new or reconnected or self.last_view is None or any(
                (view[k] is None)!=(self.last_view[k] is None) for k in ('reference','trajectory','goal','search_map')):
            self._publish_navigation(view)
            self.navigation_epochs=epochs

    def _current_row(self):
        if self.frozen:return self.frozen_row
        observation=self.buffer.observation()
        row=None if observation is None else observation[0]
        if self.last_navigation is None:return row
        nav_row,_=self.last_navigation
        if row is None or (all(row[k]==nav_row[k] for k in ('session_id','generation','frame_id')) and
                           nav_row['stamp_s']>row['stamp_s']):
            return nav_row
        return row

    def _clear_latched_path(self,row,*,force=False):
        previous,nonempty=getattr(self,'_path_latch',(None,False))
        if row is None or not nonempty:return
        valid=(row['health']==0 and row['stamp_s']>=previous['stamp_s'] and
            all(row[k]==previous[k] for k in ('session_id','generation','frame_id')))
        superseded=row['stamp_s']>previous['stamp_s'] and not self.subscribed('path')
        if force or not valid or superseded:
            self.publishers['path'].publish(self._header(self.Path(),row['frame_id'],row['stamp_s']))
            self._path_latch=(row,False)

    def _show_basemap(self, *, stopped=False, mono=None):
        layers=('occupancy','raw_occupancy','ground_map','navigation_basemap','navigation_window',
                'reference_search_map','basemap_source')
        if not self.subscribed(*layers) and not any(key is not None for key,_ in self.layer_state.values()):return
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
        if grid is not None:
            meta=grid.metadata
            if (any(meta[k]!=row[k] for k in IDENTITY) or
                    meta.get('geometry_status') not in ('OBSERVED','GROUND_UNAVAILABLE') or
                    meta['stamp_s']>row['stamp_s']+1e-6 or
                    not -1e-6<=now-meta['stamp_s']<=self.max_grid_age):
                grid=None
        self._publish_environment_layers(grid)

        transform=None
        if grid is not None and asset is not None:
            if (not mapped or not self._matching_associations(meta.get('association'),association) or
                    association_reason(grid,asset,self.max_association_age) is not None or
                    now-meta['association']['measurement_stamp_s']>self.max_association_age):
                grid=None
            else:
                transform=np.asarray(meta['association']['map_from_local'])
        self._publish_basemap_layers(grid,transform,snapshot,static_frame)

    def _layer_action(self,name,key,*,latched=False):
        previous=self.layer_state.get(name)
        subscribed=self.subscribed(name)
        if not subscribed and not latched:
            self.layer_state.pop(name,None)
            return None
        if not subscribed and latched and name!='basemap_source' and previous is not None and key!=previous[0]:
            # Replace an obsolete latch once, then build the latest source only
            # if a subscriber asks for it.
            key=None
        # A connected or reconnected ROS peer receives an unchanged latch itself.
        current=(key,0 if latched else self._connection_epoch(name))
        if current==previous:return None
        if key is None and (previous is None or previous[0] is None):return None
        if not subscribed and (not latched or previous is None or
                (key is not None and name!='basemap_source')):return None
        self.layer_state[name]=current
        return 'clear' if key is None else 'publish'

    def _publish_environment_layers(self,source):
        from .surface import coloured_cloud,ground_cloud

        key=None if source is None else source.metadata['snapshot_id']
        if source is not None:self.environment_meta=source.metadata
        occupancy=self._layer_action('occupancy',key,latched=True)
        raw=self._layer_action('raw_occupancy',key)
        ground=self._layer_action('ground_map',key)
        if 'clear' in (occupancy,raw):
            from perception.environment.layers import GridSnapshot
            clear=GridSnapshot(self.environment_meta,dict(occupancy=np.full(self.environment_meta['shape'],-1,np.int8)))
            if occupancy=='clear':self._publish_occupancy('occupancy',clear)
            if raw=='clear':self._publish_occupancy('raw_occupancy',clear)
        if ground=='clear':
            meta=self.environment_meta
            self._publish_local_cloud(coloured_cloud([],[],[],[],meta['frame_id'],meta['stamp_s']))
        display=self.grids.get(source) if ground=='publish' else None
        if occupancy=='publish':self._publish_occupancy('occupancy',display if display is not None else self.grids.get(source,ground=False))
        if raw=='publish':self._publish_occupancy('raw_occupancy',self.grids.get(source,navigation=False,ground=False))
        if ground=='publish':self._publish_local_cloud(ground_cloud(display))

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
        show_cloud=self._layer_action('navigation_basemap',local_key,latched=True)
        show_window=self._layer_action('navigation_window',local_key,latched=True)
        if show_cloud or show_window:
            frame=self.local_frame
            stamp=0.
            cloud=None;boundary=None
            if 'publish' in (show_cloud,show_window):
                if self.local_surface is None:
                    self.local_surface=NavigationGridSurface(self.grids.asset)
                display=self.grids.get(grid)
                frame=grid.metadata['frame_id'] if transform is None else static_frame
                stamp=grid.metadata['stamp_s']
                cloud,boundary=self.local_surface.layer(display,transform,cloud=show_cloud=='publish',boundary=show_window=='publish')
                self.local_frame=frame
            if show_cloud:
                if cloud is None:cloud=coloured_cloud([],[],[],[],frame,stamp)
                self.publishers['navigation_basemap'].publish(cloud)
            if show_window:
                action=Marker.DELETE if boundary is None else Marker.ADD
                marker=self._header(Marker(ns='navigation_window',id=0,type=Marker.LINE_STRIP,action=action),frame,stamp)
                marker.pose.orientation.w=1.
                if boundary is not None:
                    marker.scale.x=.045
                    marker.color.r,marker.color.g,marker.color.b,marker.color.a=(.35,1.,.86,1.)
                    marker.points=[Point(*map(float,p)) for p in boundary]
                self.publishers['navigation_window'].publish(marker)

        search_key=None if snapshot is None else (snapshot['reference_binding'],snapshot['job_id'])
        show_search=self._layer_action('reference_search_map',search_key,latched=True)
        if show_search:
            if show_search=='clear':
                cloud=coloured_cloud([],[],[],[],static_frame,0.)
            else:
                if self.search_surface is None:
                    self.search_surface=SearchMapSurface(self.grids.asset)
                cloud=self.search_surface.cloud(snapshot)
            self.publishers['reference_search_map'].publish(cloud)

        status=dict(static_available=static_frame is not None,static_frame=static_frame,local_available=local_key is not None,
            frozen=self.frozen,search_query_id=None if snapshot is None else snapshot['job_id'])
        if self._layer_action('basemap_source',tuple(status.items()),latched=True):
            self.publishers['basemap_source'].publish(self.String(data=json.dumps(status)))

    def _publish_local_cloud(self,cloud):
        self.publishers['ground_map'].publish(cloud)

    def _publish_occupancy(self,name,grid):
        message=self._header(self.OccupancyGrid(),grid.metadata['frame_id'],grid.metadata['stamp_s'])
        message.info.resolution=grid.metadata['resolution'];message.info.height,message.info.width=grid.layers['occupancy'].shape
        message.info.origin.position.x,message.info.origin.position.y=grid.metadata['origin_xy'];message.info.origin.orientation.w=1.
        message.data=grid.layers['occupancy'].reshape(-1).tolist();self.publishers[name].publish(message)

    def _publish_navigation(self,view):
        from geometry_msgs.msg import PoseStamped
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
            if name=='path':self._path_latch=(view['localization_row'],bool(rows))
        self.last_view=view
        if self.publishers['navigation'].get_num_connections():
            self._publish_navigation_markers(view)

    def _publish_marker_update(self,name,specs,*,reset=False):
        updates=self._marker_updates[name]
        changed,removed=updates.update(specs,reset=reset)
        self._send_marker_update(name,changed,removed)

    def _send_marker_update(self,name,changed=(),removed=(),*,force=False):
        publisher=self.publishers[name]
        if publisher.get_num_connections():
            message=self._marker_updates[name].message(
                self._connection_epoch(name),time.monotonic(),changed,removed,force=force)
            if message is not None:
                publisher.publish(message)

    def _refresh_marker_snapshots(self,*,force=False):
        if (self.last_view is not None and self.last_view is not self._navigation_marker_view and
                self.publishers['navigation'].get_num_connections()):
            self._publish_navigation_markers(self.last_view)
        for name in self._marker_updates:
            if name=='people' and self.buffer.people_clear_pending():continue
            self._send_marker_update(name,force=force)

    def _publish_navigation_markers(self,view):
        from .navigation import IDENTITY
        from .surface import line_segments

        frame,stamp=view['frame_id'],view['stamp_s']
        specs=[]
        def add(namespace,number,kind,color,scale,*,points=(),position=(0.,0.,0.),text='',
                source_frame=frame,source_stamp=stamp):
            specs.append(dict(namespace=namespace,id=number,kind=kind,color=color,scale=scale,
                points=points,position=position,text=text,frame_id=source_frame,stamp_s=source_stamp))
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
            add(name,number,'box',color,(width,0.,0.),points=segments,
                source_frame=curve_frame,source_stamp=stamp if data is None else data['stamp_s'])
        pose=view['pose']
        body_segments=line_segments(view['footprint']) if pose is not None else []
        if len(body_segments):
            add('vehicle',0,'box',(1.,.82,.16,1.),(.065,0.,0.),points=body_segments)
        if pose is not None and np.isfinite(view['heading']).all():
            add('vehicle',1,'arrow',(1.,.82,.16,1.),(.11,.20,.24),points=view['heading'])
        if view['goal'] is not None and np.isfinite(view['goal_z']):
            add('goal',2,'sphere',(.75,.4,1.,1.),(.25,.25,.25),
                position=(view['goal'][0],view['goal'][1],view['goal_z']+.1))
        color={'ready':(.4,1.,.65,1.),'waiting':(1.,.85,.4,1.),'error':(1.,.35,.3,1.)}[view['level']]
        x,y=(0.,0.) if pose is None else pose[:2]
        add('status',3,'text',color,(0.,0.,.25),position=(x,y,view['text_z']),text=view['text'])
        previous=self._navigation_marker_view
        reset=previous is not None and (stamp<previous['stamp_s'] or any(
            previous['localization_row'][key]!=view['localization_row'][key] for key in IDENTITY))
        self._publish_marker_update('navigation',specs,reset=reset)
        self._navigation_marker_view=view

    def _work(self):
        try:
            static_published=False
            while True:
                if not static_published and self.publish_static and self.grids.asset is not None and self.subscribed('static_basemap'):
                    from .surface import ground_cloud
                    static=ground_cloud(navigation_grid(self.grids.asset),offset=-.015,fill_known=True)
                    # A static asset can precede this session's map-to-odometry TF.
                    static.header.stamp=self.rospy.Time()
                    self.publishers['static_basemap'].publish(static)
                    static_published=True
                entry=self.buffer.take(timeout=.1)
                if entry is None:
                    if self.last_navigation is not None and self.subscribed(*NAVIGATION_TOPICS):
                        row=self._current_row()
                        if row is not None:
                            self._publish_navigation(self.navigation_view.update(row,None,grid=self.last_navigation_grid,stopped=True))
                    self._show_basemap(stopped=True)
                    self._refresh_marker_snapshots(force=True)
                    return
                group,item=entry
                if group!='frozen_map':self._publish_map_transform(self._current_row())
                if group=='idle':
                    self._refresh_navigation()
                    self._show_basemap()
                else:
                    began=time.monotonic();self._publish(group,item)
                    self.publish_ms[group].append((time.monotonic()-began)*1000);self.published[group]+=1
                    if group=='frozen_map':self._publish_map_transform(self._current_row())
                    if group not in ('navigation','map','map_clear','frozen_map'):self._show_basemap()
                self._refresh_marker_snapshots()
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def close(self):
        self.buffer.close();self.worker.join(timeout=3)
        percentile=lambda rows:None if not rows else float(np.percentile(rows,95))
        return dict(error=self.error,dropped=sum(self.buffer.superseded.values()),
            superseded=dict(self.buffer.superseded),published=dict(self.published),settings=self.buffer.settings,
            submit_wall_p95_ms=percentile(self.submit_ms),
            publish_wall_p95_ms={k:percentile(v) for k,v in self.publish_ms.items()},
            complete=not self.worker.is_alive() and self.error is None)
