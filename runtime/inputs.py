"""Bag sensor decoding. Importing this module does not import ROS or start nodes."""
from collections import deque
import numpy as np
from perception.mapping_localization.native import rs16_preprocess
from perception.sensors.pointcloud import pointcloud_fields
from perception.sensors.image import rgb_array


def bag_observations(path,config,*,duration=None,image_topics=None,start=0.,messages=None):
    """Raw scans plus causally collected IMU; no roscore or internal topics.

    Header time is retained. A later IMU arrival only signals that a complete
    earlier scan can be processed; that future sample is never fed to it.
    """
    from contextlib import nullcontext
    if messages is None:import rosbag
    topic=config['common']['lid_topic'];imu_topic=config['common']['imu_topic']
    offset=config['common'].get('imu_time_offset',0.)
    imu=deque();scans=deque();images=deque(maxlen=16);intrinsics=deque(maxlen=16);last_imu=-np.inf;first=None;origin=None
    image_topics=image_topics or {}
    rgb_topic=image_topics.get('rgb');info_topic=image_topics.get('camera_info')
    with (rosbag.Bag(str(path)) if messages is None else nullcontext(messages)) as bag:
        source=bag.read_messages(topics=[topic,imu_topic]+list(image_topics.values())) if messages is None else bag
        for name,msg,_ in source:
            if name==imu_topic:
                t=msg.header.stamp.to_sec()+offset
                if t<=last_imu:raise ValueError('IMU timestamp reversed or duplicated')
                last_imu=t;a=msg.linear_acceleration;g=msg.angular_velocity
                imu.append((t,a.x,a.y,a.z,g.x,g.y,g.z))
                if first is None:
                    while imu and t-imu[0][0]>1.:imu.popleft()
                if len(imu)>40000:raise RuntimeError('IMU backlog exceeds native input budget')
            elif name==rgb_topic:
                images.append((msg.header.stamp.to_sec(),msg))
            elif name==info_topic:
                intrinsics.append((msg.header.stamp.to_sec(),dict(K=list(msg.K),D=list(msg.D),width=msg.width,height=msg.height,
                    frame_id=msg.header.frame_id,stamp_ns=msg.header.stamp.to_nsec(),distortion_model=msg.distortion_model)))
            elif name==topic:
                stamp=msg.header.stamp.to_sec()
                if origin is None:origin=stamp
                if stamp-origin<start-1e-6:continue
                if first is None:first=stamp
                if duration is not None and stamp-first>duration:return
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
