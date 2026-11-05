"""RViz goal/cancel intents for the existing runtime; no chassis publisher."""
import argparse
import copy
import json
import math
import threading
import uuid


class ManualGoals:
    def __init__(self,send,*,task_id,frame='odom',map_snapshot_id=None):
        if not isinstance(task_id,str) or not task_id:raise ValueError('A task_id is required')
        if frame not in ('map','odom'):raise ValueError('Select map or odom; implicit transforms are not supported')
        if frame=='map' and not map_snapshot_id:raise ValueError('Map goals need a verified static environment package')
        self.send,self.task_id,self.frame,self.map_snapshot_id=send,task_id,frame,map_snapshot_id
        self.revision=0;self.last_submitted=None;self.lock=threading.Lock()

    def goal(self,*,frame,position,quaternion,stamp_s):
        values=[*position,*quaternion,stamp_s]
        if len(position)!=3 or len(quaternion)!=4 or not all(math.isfinite(x) for x in values) or stamp_s<0:
            raise ValueError('Expected a finite stamped PoseStamped')
        if frame!=self.frame:raise ValueError('Goal frame differs from selected frame; no implicit transform')
        x,y,z,w=quaternion
        if abs(sum(v*v for v in quaternion)-1)>1e-3 or abs(x)>1e-5 or abs(y)>1e-5:
            raise ValueError('Expected a normalized planar goal quaternion')
        yaw=math.atan2(2*w*z,1-2*z*z)
        with self.lock:
            self.revision+=1
            request=dict(request_id='rviz-'+uuid.uuid4().hex,task_id=self.task_id,goal_revision=self.revision,
                mode='mapped' if self.frame=='map' else 'local',frame_id=self.frame,target_reference='base_link',
                goal=[position[0],position[1],yaw],algorithm='hybrid_astar',goal_behavior='stop',
                motion_intent='forward_preferred',source='rviz_2d_nav_goal',source_intent_stamp_s=stamp_s)
            if self.frame=='map':request['map_snapshot_id']=self.map_snapshot_id
            # An acknowledgement can be lost after the runtime has queued the
            # goal. A later cancel must cover that revision as well as older ones.
            self.last_submitted=copy.deepcopy(request)
            reply=self.send('goal',request)
            return dict(reply=reply,request=request,executable=False,execution_feedback=False,task_completed=False)

    def cancel(self):
        with self.lock:
            if self.last_submitted is None:raise ValueError('This bridge has no submitted goal to cancel')
            value={k:self.last_submitted[k] for k in ('task_id','goal_revision')}
            reply=self.send('cancel',value)
            if reply.get('accepted') is True:self.last_submitted=None
            return dict(reply=reply,request=value,executable=False,execution_feedback=False,task_completed=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',required=True,help='commands.sock of the selected running runtime')
    p.add_argument('--frame',choices=('map','odom'),default='odom')
    p.add_argument('--static-map',help='Verified static environment package; required for map goals')
    p.add_argument('--task-id',default='manual-'+uuid.uuid4().hex)
    p.add_argument('--goal-topic',default='/move_base_simple/goal')
    p.add_argument('--cancel-topic',default='/social_nav/manual_goal/cancel')
    p.add_argument('--state-topic',default='/social_nav/manual_goal/state')
    p.add_argument('--map-topic',default='/social_nav/manual_goal/map',help='Read-only static occupancy for RViz map-goal selection')
    args=p.parse_args(argv)
    if (args.frame=='map')!=bool(args.static_map):p.error('--static-map is required exactly for --frame map')
    snapshot=None;asset=None
    if args.static_map:
        from perception.environment.storage import StaticEnvironmentMap
        asset=StaticEnvironmentMap(args.static_map)
        if asset.manifest['frame_id']!=args.frame:p.error('Static environment frame differs from selected goal frame')
        snapshot=asset.manifest['snapshot_id']
    from runtime.command import send_request
    bridge=ManualGoals(lambda kind,value:send_request(args.socket,kind,value),task_id=args.task_id,
        frame=args.frame,map_snapshot_id=snapshot)
    import rospy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import Bool,String
    rospy.init_node('social_nav_manual_goal',disable_signals=False)
    publisher=rospy.Publisher(args.state_topic,String,queue_size=8,latch=True)
    map_publisher=None
    if asset is not None:
        grid=asset.snapshot;metadata=grid.metadata
        message=OccupancyGrid();message.header.frame_id=metadata['frame_id']
        message.header.stamp=rospy.Time.from_sec(metadata['stamp_s'])
        message.info.resolution=metadata['resolution'];message.info.height,message.info.width=grid.layers['occupancy'].shape
        message.info.origin.position.x,message.info.origin.position.y=metadata['origin_xy']
        message.info.origin.orientation.w=1.;message.data=grid.layers['occupancy'].reshape(-1).tolist()
        map_publisher=rospy.Publisher(args.map_topic,OccupancyGrid,queue_size=1,latch=True)
        map_publisher.publish(message);asset.close()

    def handle(action):
        try:result=action()
        except (ValueError,TypeError,KeyError,OSError,EOFError) as error:
            result=dict(error=str(error),executable=False,execution_feedback=False,task_completed=False)
            rospy.logwarn('Manual navigation intent rejected: %s',error)
        publisher.publish(String(data=json.dumps(result,ensure_ascii=False,allow_nan=False)))

    def goal(message):
        pose=message.pose
        handle(lambda:bridge.goal(frame=message.header.frame_id,position=[pose.position.x,pose.position.y,pose.position.z],
            quaternion=[pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w],stamp_s=message.header.stamp.to_sec()))

    subscribers=[rospy.Subscriber(args.goal_topic,PoseStamped,goal,queue_size=1),
        rospy.Subscriber(args.cancel_topic,Bool,lambda m:handle(bridge.cancel) if m.data else None,queue_size=1)]
    rospy.loginfo('RViz intents use frame %s and runtime socket %s; acknowledgements do not mean motion',args.frame,args.socket)
    rospy.spin()
    for subscriber in subscribers:subscriber.unregister()


if __name__=='__main__':main()
