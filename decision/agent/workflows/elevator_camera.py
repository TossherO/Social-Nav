"""Publish independent RGB and base-frame structural LiDAR observations.

Run with the ROS system Python. Each prefixed JSON line keeps the source ROS
timestamp; it is an observation, not evidence of a floor transition.
"""
import base64
import io
import json
import sys
import time
import threading


def image_record(pixels, *, stamp_s, frame_id, sequence, max_edge=1024):
    from PIL import Image
    image=Image.fromarray(pixels)
    if image.mode!='RGB':raise ValueError('Camera image must be RGB')
    image.thumbnail((max_edge,max_edge))
    out=io.BytesIO()
    image.save(out,format='JPEG',quality=85)
    return dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',stamp_s=stamp_s,
                frame_id=frame_id,sequence=sequence,width=image.width,height=image.height,
                mime_type='image/jpeg',image_base64=base64.b64encode(out.getvalue()).decode('ascii'))


def scan_record(message, base_from_lidar, sensor_frame, sequence):
    """Render measured returns in base_link, with +x forward and +y left."""
    import numpy as np
    from PIL import Image, ImageDraw
    from perception.sensors.pointcloud import pointcloud_fields
    if message.header.frame_id.lstrip('/') != sensor_frame.lstrip('/'):
        raise ValueError('Unexpected LiDAR frame')
    fields = pointcloud_fields(message, ('x','y','z'))
    xyz = np.column_stack([fields[k] for k in ('x','y','z')])
    xyz = xyz[np.isfinite(xyz).all(axis=1)] @ base_from_lidar[:3,:3].T + base_from_lidar[:3,3]
    xyz = xyz[(np.abs(xyz[:,:2]).max(axis=1)<8) & (xyz[:,2]>.2) & (xyz[:,2]<2.)]
    if not len(xyz):
        raise ValueError('No local structural returns')
    size, scale = 641, 40
    canvas = Image.new('RGB',(size,size),'white')
    pen = ImageDraw.Draw(canvas)
    for pixel in range(0,size,scale):
        pen.line((pixel,0,pixel,size),fill='#ddd')
        pen.line((0,pixel,size,pixel),fill='#ddd')
    pixels = np.asarray(canvas).copy()
    col, row = np.rint(320-xyz[:,[1,0]]*scale).astype(int).T
    pixels[row,col] = [28,42,57]
    canvas = Image.fromarray(pixels)
    pen = ImageDraw.Draw(canvas)
    pen.polygon([(320,309),(315,326),(325,326)],fill='red')
    pen.text((8,8),'base_link | forward up | left left | grid 1 m',fill='black')
    out = io.BytesIO()
    canvas.save(out,format='PNG')
    return dict(source='local_lidar_returns',clock_domain='ros_time',frame_id=message.header.frame_id,
                display_frame='base_link',stamp_s=message.header.stamp.to_sec(),sequence=sequence,
                base_xy_dm=np.unique(np.rint(xyz[:,:2]*10).astype(np.int16),axis=0).tolist(),
                mime_type='image/png',image_base64=base64.b64encode(out.getvalue()).decode('ascii'))


def main():
    import argparse
    import numpy as np
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, PointCloud2
    from perception.detection_tracking.inference.config import load_config
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-from-lidar',required=True)
    args=parser.parse_args()
    transform=np.asarray(json.loads(args.base_from_lidar),dtype=float)
    sensors=load_config()['ros']

    rospy.init_node('social_nav_elevator_camera',anonymous=True,disable_signals=True)
    bridge=CvBridge()
    last=[0.]
    sequence=[0]
    lock=threading.Lock()
    def publish(prefix,record):
        with lock:
            print(prefix+' '+json.dumps(record,allow_nan=False,separators=(',',':')),flush=True)
    def callback(message):
        now=time.monotonic()
        if now-last[0]<1.:return
        stamp=message.header.stamp.to_sec()
        if stamp<=0 or not 0<=rospy.Time.now().to_sec()-stamp<=2 or not message.header.frame_id:return
        try:
            pixels=bridge.imgmsg_to_cv2(message,desired_encoding='rgb8')
            record=image_record(pixels,stamp_s=stamp,frame_id=message.header.frame_id,sequence=sequence[0])
            publish('FRAME',record)
        except (ValueError,TypeError,RuntimeError) as error:
            print('ERROR '+type(error).__name__,file=sys.stderr,flush=True)
            return
        sequence[0]+=1
        last[0]=now
    scan_last=[0.]
    def scan(message):
        now=time.monotonic()
        if now-scan_last[0]<.5 or not 0<=rospy.Time.now().to_sec()-message.header.stamp.to_sec()<=2:
            return
        scan_last[0]=now
        try:
            publish('SCAN',scan_record(message,transform,sensors['localization_child_frame'],message.header.seq))
        except (ValueError,TypeError) as error:
            rospy.logwarn_throttle(5.,str(error))
    subscriber=rospy.Subscriber(sensors['rgb_topic'],Image,callback,queue_size=1,buff_size=2**24)
    lidar=rospy.Subscriber(sensors['pointcloud_topic'],PointCloud2,scan,queue_size=1,buff_size=4*1024*1024)
    try:rospy.spin()
    finally:
        subscriber.unregister()
        lidar.unregister()


if __name__=='__main__':main()
