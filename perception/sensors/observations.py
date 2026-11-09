"""Causal LiDAR, IMU and camera observations from bag or live messages."""
from collections import deque
from contextlib import contextmanager, nullcontext
import queue
import numpy as np
from perception.mapping_localization.native import rs16_preprocess
from perception.sensors.pointcloud import pointcloud_fields
from perception.sensors.image import rgb_array


def ros_messages(config,*,stop_event=None,include_rgb=False):
    """Receive sensor topics from ROS without subscribing to derived results."""
    import rospy
    from sensor_msgs.msg import PointCloud2,Imu,Image,CameraInfo
    rospy.init_node('social_nav_sensor_edge',disable_signals=True)
    inbox=queue.Queue(24000);failed=[]
    def callback(message,topic):
        try:inbox.put_nowait((topic,message,None))
        except queue.Full:failed.append('Sensor edge backlog exceeded its budget')
    subscribers=[]
    topics=[(config['common']['lid_topic'],PointCloud2),(config['common']['imu_topic'],Imu)]
    if include_rgb:topics += [('/camera/color/image_raw',Image),('/camera/color/camera_info',CameraInfo)]
    for topic,kind in topics:
        subscribers.append(rospy.Subscriber(topic,kind,callback,callback_args=topic,queue_size=256,
            buff_size=8*1024*1024,tcp_nodelay=True))
    try:
        while not rospy.is_shutdown() and not (stop_event is not None and stop_event.is_set()):
            if failed:raise RuntimeError(failed[0])
            try:yield inbox.get(timeout=.1)
            except queue.Empty:continue
    finally:
        for subscriber in subscribers:subscriber.unregister()


@contextmanager
def _message_source(path,messages,topics):
    if messages is None:
        import rosbag
    with (rosbag.Bag(str(path)) if messages is None else nullcontext(messages)) as bag:
        yield bag.read_messages(topics=topics) if messages is None else bag


class _ScanInterval:
    """Track the requested interval from the first LiDAR measurement."""
    def __init__(self,start,duration):
        self.start=start;self.duration=duration;self.origin=None;self.first=None

    def accept(self,stamp):
        if self.origin is None:self.origin=stamp
        if self.first is None:
            if stamp-self.origin<self.start-1e-6:return False
            self.first=stamp
        if self.duration is not None and stamp-self.first>self.duration:return None
        return True


def selected_messages(path,config,*,start=0.,duration=None,messages=None,image_topics=None):
    """Select a sensor interval, retaining its leading IMU measurements."""
    lidar=config['common']['lid_topic'];imu=config['common']['imu_topic']
    lead=deque();interval=_ScanInterval(start,duration)
    with _message_source(path,messages,[lidar,imu]+list((image_topics or {}).values())) as source:
        for row in source:
            topic,msg,_=row;stamp=msg.header.stamp.to_sec()
            if interval.first is None:
                if topic==imu:
                    lead.append(row)
                    while lead and stamp-lead[0][1].header.stamp.to_sec()>1.:lead.popleft()
                if topic!=lidar:continue
                accepted=interval.accept(stamp)
                if accepted is False:continue
                yield from lead;lead.clear()
                if accepted is None:return
            elif topic==lidar and interval.accept(stamp) is None:return
            yield row


def bag_observations(path,config,*,duration=None,image_topics=None,start=0.,messages=None):
    """Raw scans plus causally collected IMU; no roscore or internal topics.

    Header time is retained. A later IMU arrival only signals that a complete
    earlier scan can be processed; that future sample is never fed to it.
    """
    topic=config['common']['lid_topic'];imu_topic=config['common']['imu_topic']
    offset=config['common'].get('imu_time_offset',0.)
    imu=deque();scans=deque();images=deque(maxlen=16);intrinsics=deque(maxlen=16);last_imu=-np.inf;last_imu_stamp_ns=None
    interval=_ScanInterval(start,duration)
    image_topics=image_topics or {}
    rgb_topic=image_topics.get('rgb');info_topic=image_topics.get('camera_info')
    with _message_source(path,messages,[topic,imu_topic]+list(image_topics.values())) as source:
        for name,msg,_ in source:
            if name==imu_topic:
                t=msg.header.stamp.to_sec()+offset
                stamp_ns=msg.header.stamp.to_nsec()
                if t<=last_imu:
                    caller=(getattr(msg,'_connection_header',None) or {}).get('callerid')
                    raise ValueError('IMU timestamp reversed or duplicated: '
                        f'topic={imu_topic}, previous_stamp_ns={last_imu_stamp_ns}, current_stamp_ns={stamp_ns}, '
                        f'effective_delta_s={t-last_imu:.9f}, header_seq={getattr(msg.header,"seq",None)}, caller={caller}')
                last_imu=t;last_imu_stamp_ns=stamp_ns;a=msg.linear_acceleration;g=msg.angular_velocity
                imu.append((t,a.x,a.y,a.z,g.x,g.y,g.z))
                if interval.first is None:
                    while imu and t-imu[0][0]>1.:imu.popleft()
                if len(imu)>40000:raise RuntimeError('IMU backlog exceeds native input budget')
            elif name==rgb_topic:
                images.append((msg.header.stamp.to_sec(),msg))
            elif name==info_topic:
                intrinsics.append((msg.header.stamp.to_sec(),dict(K=list(msg.K),D=list(msg.D),width=msg.width,height=msg.height,
                    frame_id=msg.header.frame_id,stamp_ns=msg.header.stamp.to_nsec(),distortion_model=msg.distortion_model)))
            elif name==topic:
                stamp=msg.header.stamp.to_sec()
                accepted=interval.accept(stamp)
                if accepted is False:continue
                if accepted is None:return
                fields=pointcloud_fields(msg,['x','y','z','intensity','timestamp'])
                raw=np.column_stack([fields[k] for k in ('x','y','z','intensity')]).astype(np.float32)
                points,begin,end=rs16_preprocess(fields,config,stamp)
                stamp_ns=msg.header.stamp.to_nsec() if config['preprocess'].get('stamp_at_scan_end',True) else round(end*1e9)
                scans.append((raw,points,begin,end,msg.header.frame_id,stamp_ns))
                if len(scans)>32:raise RuntimeError('Scan backlog exceeds native input budget')
            while scans and last_imu>=scans[0][3]:
                raw,points,begin,end,frame,stamp_ns=scans.popleft();samples=[]
                while imu and imu[0][0]<=end:samples.append(imu.popleft())
                image=max((r for r in images if 0<=end-r[0]<=.1),key=lambda r:r[0],default=None)
                camera=max((r for r in intrinsics if image is not None and r[0]<=image[0]),key=lambda r:r[0],default=None)
                yield dict(raw=raw,points=points,begin=begin,end=end,sensor_frame=frame,stamp_ns=stamp_ns,
                    imu=np.asarray(samples,np.float64).reshape(-1,7),rgb=None if image is None else rgb_array(image[1]),
                    rgb_stamp=None if image is None else image[0],rgb_frame=None if image is None else image[1].header.frame_id,
                    rgb_stamp_ns=None if image is None else image[1].header.stamp.to_nsec(),
                    camera_info=None if camera is None else camera[1])
