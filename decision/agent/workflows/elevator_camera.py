"""Publish camera observations independent of the localization scene.

Run with the ROS system Python. Each prefixed JSON line keeps the source ROS
timestamp; it is an observation, not evidence of a floor transition.
"""
import base64
import io
import json
import sys
import time


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


def main():
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    rospy.init_node('social_nav_elevator_camera',anonymous=True,disable_signals=True)
    bridge=CvBridge()
    last=[0.]
    sequence=[0]
    def callback(message):
        now=time.monotonic()
        if now-last[0]<1.:return
        stamp=message.header.stamp.to_sec()
        if stamp<=0 or not message.header.frame_id:return
        try:
            pixels=bridge.imgmsg_to_cv2(message,desired_encoding='rgb8')
            record=image_record(pixels,stamp_s=stamp,frame_id=message.header.frame_id,sequence=sequence[0])
            print('FRAME '+json.dumps(record,allow_nan=False,separators=(',',':')),flush=True)
        except (ValueError,TypeError,RuntimeError) as error:
            print('ERROR '+type(error).__name__,file=sys.stderr,flush=True)
            return
        sequence[0]+=1
        last[0]=now
    subscriber=rospy.Subscriber('/camera/color/image_raw',Image,callback,queue_size=1,buff_size=2**24)
    try:rospy.spin()
    finally:subscriber.unregister()


if __name__=='__main__':main()
