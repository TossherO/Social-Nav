"""Display-only ROS marker bridge. This process never publishes a motion goal."""
import json
import sys
import threading


def main():
    import rospy
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    rospy.init_node('social_nav_manual_agent_route', anonymous=True, disable_signals=True)
    publisher=rospy.Publisher('/social_nav_agent/manual_route', MarkerArray, queue_size=1, latch=True)
    lock=threading.Lock()
    latest=[None]

    def publish(_event=None):
        with lock:
            value=latest[0]
        if value is not None:
            publisher.publish(value)

    timer=rospy.Timer(rospy.Duration(.5),publish)
    try:
        for line in sys.stdin:
            data=json.loads(line)
            frame=data['frame_id']
            clear=Marker();clear.header.frame_id=frame;clear.action=Marker.DELETEALL
            markers=[clear]
            points=data.get('points',[])
            index=data.get('route_index')
            for i,point in enumerate(points):
                if point[2] is None:
                    continue
                dot=Marker();dot.header.frame_id=frame;dot.ns='manual_route';dot.id=i
                dot.type=Marker.SPHERE;dot.action=Marker.ADD;dot.pose.orientation.w=1.
                dot.pose.position=Point(*point);dot.scale.x=dot.scale.y=dot.scale.z=.18
                dot.color.a=1.;dot.color.r=1. if index==i else .15;dot.color.g=.7;dot.color.b=1.
                markers.append(dot)
                label=Marker();label.header.frame_id=frame;label.ns='manual_route_labels';label.id=i
                label.type=Marker.TEXT_VIEW_FACING;label.action=Marker.ADD;label.pose.orientation.w=1.
                label.pose.position=Point(point[0],point[1],point[2]+.3);label.scale.z=.28
                label.color.r=label.color.g=label.color.b=label.color.a=1.;label.text=str(i+1)
                markers.append(label)
            lines=Marker();lines.header.frame_id=frame;lines.ns='manual_route_line';lines.id=0
            lines.type=Marker.LINE_LIST;lines.action=Marker.ADD;lines.pose.orientation.w=1.
            lines.scale.x=.06;lines.color.r=.2;lines.color.g=.7;lines.color.b=1.;lines.color.a=1.
            for a,b in zip(points,points[1:]):
                if a[2] is not None and b[2] is not None:
                    lines.points.extend([Point(*a),Point(*b)])
            markers.append(lines)
            for marker in markers[1:]:marker.lifetime=rospy.Duration(2.)
            with lock:latest[0]=MarkerArray(markers=markers)
            publish()
    finally:
        timer.shutdown()
        with lock:
            value=latest[0]
        if value is not None:
            clear=Marker();clear.action=Marker.DELETEALL
            clear.header.frame_id=value.markers[0].header.frame_id
            publisher.publish(MarkerArray(markers=[clear]))


if __name__=='__main__':
    main()
