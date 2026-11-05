"""Optional ROS sensor edge. No perception/navigation result is subscribed."""
import queue


def messages(config,*,stop_event=None,include_rgb=False):
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
