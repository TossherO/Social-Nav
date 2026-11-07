"""RViz goal/cancel intents for the existing runtime; no chassis publisher."""
import copy
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
        return self._submit([position[0],position[1],yaw],stamp_s,'rviz_2d_nav_goal')

    def point(self,*,frame,position,stamp_s):
        if (frame!=self.frame or len(position)!=3 or not all(math.isfinite(x) for x in [*position,stamp_s]) or stamp_s<0):
            raise ValueError('Expected a finite point in the selected frame')
        return self._submit(list(position[:2]),stamp_s,'rviz_clicked_point')

    def _submit(self,goal,stamp_s,source):
        with self.lock:
            self.revision+=1
            request=dict(request_id='rviz-'+uuid.uuid4().hex,task_id=self.task_id,goal_revision=self.revision,
                mode='mapped' if self.frame=='map' else 'local',frame_id=self.frame,target_reference='base_link',
                goal=goal,algorithm='hybrid_astar',goal_behavior='stop',
                motion_intent='forward_preferred',source=source,source_intent_stamp_s=stamp_s)
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
