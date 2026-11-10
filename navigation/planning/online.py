"""Causal ROS geometry/people planning bridge; no chassis publisher."""
from collections import deque
import copy
import json
import math
import threading
import time
import numpy as np
from perception.environment.context import fingerprint
from perception.environment.scene import assemble_scene,snapshot_from_message
from .motion import estimate_motion
from .continuity import goal_binding


def planning_scene(snapshot,*,clock_domain,settings,people=None):
    scene=assemble_scene(snapshot,clock_domain=clock_domain,settings=settings,people=people)
    scene.record['scope']='people_ros_planning' if people is not None else 'geometry_only_ros_planning'
    scene.record['input_fingerprint']=fingerprint({k:v for k,v in scene.record.items() if k!='input_fingerprint'})
    # assemble_scene has just bound these bytes; consumers verify at their own
    # acceptance boundary, rather than hashing all layers twice during assembly.
    return scene


def goal_request(value):
    """Validate the wire envelope; the existing planner checks geometry and frames."""
    if not isinstance(value,dict):raise ValueError('Goal must be a JSON object')
    for key in ('request_id','task_id','frame_id'):
        if not isinstance(value.get(key),str) or not value[key]:raise ValueError('Missing '+key)
    if type(value.get('goal_revision')) is not int or value['goal_revision']<0:raise ValueError('Invalid goal_revision')
    if value.get('mode') not in ('mapped','local') or value.get('target_reference')!='base_link':
        raise ValueError('Online goals require mapped/local mode and base_link reference')
    point=np.asarray(value.get('goal'),float)
    if point.shape not in ((2,),(3,)) or not np.isfinite(point).all():raise ValueError('Invalid goal')
    if value['mode']=='mapped' and not value.get('map_snapshot_id'):raise ValueError('Missing map_snapshot_id')
    return copy.deepcopy(value)


class PlanningBridge:
    """Bounded callback inbox; exactly one owner calls step and owns the runtime."""
    def __init__(self,runtime,geometry,motion,scene_settings,*,asset=None,clock_domain):
        self.runtime=runtime;self.geometry=geometry;self.motion=motion;self.scene_settings=scene_settings
        self.asset=asset;self.clock_domain=clock_domain;self.lock=threading.Lock()
        self.rows=deque(maxlen=256);self.commands=deque();self.pending=None;self.barrier=None
        self.request=None;self.scene=None;self.last_submission=None;self.last_clock=None
        self.reason='waiting_for_goal';self.command_error=None
        self.people=deque(maxlen=64);self.people_changed=False;self.snapshot=None
        self.scene_selection=None

    def push_people(self,row):
        if not isinstance(row,dict) or not np.isfinite(row.get('stamp_s',np.nan)):raise ValueError('Invalid people envelope')
        row=copy.deepcopy(row);row['clock_domain']=self.clock_domain
        with self.lock:
            self.people.append(row);self.people_changed=True

    def push_state(self,row):
        with self.lock:
            if self.rows:
                old=self.rows[-1]
                if (tuple(row[k] for k in ('session_id','generation','frame_id'))!=
                        tuple(old[k] for k in ('session_id','generation','frame_id')) or row['stamp_s']<old['stamp_s']):
                    self.barrier='localization_reset';self.pending=None;self.rows.clear()
            if row['health']!=0:self.barrier=self.barrier or 'current_localization_invalid'
            self.rows.append(copy.deepcopy(row))

    def push_grid(self,message):
        with self.lock:self.pending=message

    def push_command(self,kind,value):
        with self.lock:
            if len(self.commands)>=32:
                self.commands.clear();self.barrier='command_queue_overflow'
            self.commands.append((kind,copy.deepcopy(value)))

    def _invalidate(self,reason,*,drop_goal=False):
        if self.scene is not None or self.runtime.latest is not None or reason in ('seek','localization_reset','command_queue_overflow'):
            self.runtime.invalidate(reason)
        self.scene=None;self.snapshot=None;self.scene_selection=None;self.last_submission=None;self.reason=reason
        if drop_goal:self.request=None

    def step(self,now):
        with self.lock:
            rows=list(self.rows);commands=list(self.commands);self.commands.clear()
            pending=self.pending;self.pending=None;barrier=self.barrier;self.barrier=None
            people=list(self.people);changed=self.people_changed;self.people_changed=False
        if self.last_clock is not None and now<self.last_clock-1e-6:
            barrier='seek';pending=None
            with self.lock:self.rows.clear();self.people.clear()
        self.last_clock=now
        if barrier:
            self._invalidate(barrier,drop_goal=barrier in ('seek','localization_reset','command_queue_overflow'))
            if barrier in ('seek','localization_reset'):commands=[]
            if barrier in ('seek','localization_reset'):
                people=[]
                with self.lock:self.people.clear()
        for kind,value in commands:
            try:
                if kind=='goal':
                    request=goal_request(value)
                    # Reject late/changed revisions before replacing an active goal.
                    if self.request is not None and request['task_id']==self.request['task_id']:
                        old=self.request
                        if request['goal_revision']<old['goal_revision']:raise ValueError('Old goal revision')
                        if (request['goal_revision']==old['goal_revision'] and
                                goal_binding(request)!=goal_binding(old)):
                            raise ValueError('Changed goal requires new goal_revision')
                    changed=self.request is None or goal_binding(request)!=goal_binding(self.request)
                    self.request=request;self.last_submission=None
                    if changed:self.runtime.current=None
                elif kind=='cancel':
                    if (not isinstance(value,dict) or not isinstance(value.get('task_id'),str) or
                            not value['task_id'] or type(value.get('goal_revision')) is not int or value['goal_revision']<0):
                        raise ValueError('Cancel must bind task_id and integer goal_revision')
                    if self.request is not None and all(value.get(k)==self.request[k] for k in ('task_id','goal_revision')):
                        self.runtime.cancel(self.request);self.request=None;self.scene=None;self.last_submission=None;self.reason='cancel'
                else:raise ValueError('Unknown command')
                self.command_error=None
            except (ValueError,TypeError) as error:self.command_error=str(error)
        if pending is not None or changed and self.snapshot is not None:
            try:
                snapshot=snapshot_from_message(pending) if pending is not None else self.snapshot
                past=[p for p in people if p['stamp_s']<=snapshot.metadata['stamp_s']]
                source=max(past,key=lambda p:p['stamp_s']) if past else None
                selection=(id(snapshot),fingerprint(source))
                if selection!=self.scene_selection:
                    scene=planning_scene(snapshot,clock_domain=self.clock_domain,settings=self.scene_settings,people=source)
                    self.snapshot=snapshot;self.scene_selection=selection
                    if self.scene is None or self.scene.record['input_fingerprint']!=scene.record['input_fingerprint']:
                        self.scene=scene;self.last_submission=None
            except (ValueError,KeyError,TypeError) as error:self._invalidate('invalid_environment_grid: '+str(error))
        latest=rows[-1] if rows else None
        if self.scene is not None:
            record=self.scene.record
            reason=None
            if latest is None or latest['health']!=0:reason='current_localization_invalid'
            elif [latest[k] for k in ('session_id','generation','frame_id')]!=[record[k] for k in ('session_id','generation','frame_id')]:
                reason='localization_lifecycle_mismatch'
            elif not 0<=now-latest['stamp_s']<=self.motion['max_result_age_s']:reason='localization_future_or_stale'
            if reason:self._invalidate(reason)
        if self.request is not None and self.scene is not None:
            key=(fingerprint(self.request),self.scene.record['input_fingerprint'])
            if key!=self.last_submission:
                state=estimate_motion(rows,self.scene,self.geometry,self.motion)
                self.runtime.update(self.request,self.scene,state,self.asset,now=now)
                # A grid callback can precede the corresponding localization callback.
                if state['status']=='valid':self.last_submission=key
        status=self.runtime.tick(now=now,clock_domain=self.clock_domain)
        if self.request is None or self.scene is None:
            status=dict(status='unavailable',reason=self.reason,result=None,executable=False,task_completed=False)
        status.update(scope='causal_ros_planning',goal=self.request,command_error=self.command_error,
            people_status=None if self.scene is None else self.scene.record['people']['status'])
        return status


def trajectory_path(result,rospy):
    """Path stamps retain acquisition + relative trajectory time, never publish time."""
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    path=Path()
    if result is None:return path
    binding=result['input_binding'];path.header.frame_id=binding['frame_id']
    path.header.stamp=rospy.Time.from_sec(binding['as_of'])
    for t,x,y,yaw in result['trajectory']['points']:
        pose=PoseStamped();pose.header.frame_id=path.header.frame_id
        pose.header.stamp=rospy.Time.from_sec(binding['as_of']+t)
        pose.pose.position.x=x;pose.pose.position.y=y
        pose.pose.orientation.z=math.sin(yaw/2);pose.pose.orientation.w=math.cos(yaw/2)
        path.poses.append(pose)
    return path


def run(args,planning,motion,geometry,settings):
    import rospy
    from std_msgs.msg import String
    from nav_msgs.msg import Path
    from social_nav_environment.msg import EnvironmentGrid
    from rospy.numpy_msg import numpy_msg
    from social_nav_localization.msg import LocalizationState
    from perception.environment.layers import load_config
    from perception.environment.storage import StaticEnvironmentMap
    from perception.mapping_localization.ros.context_node import state_row
    from .reference import ReferenceRuntime
    rospy.init_node('social_nav_planning',disable_signals=False)
    clock='ros_sim_time' if rospy.get_param('/use_sim_time',False) else 'ros_time'
    asset=StaticEnvironmentMap(args.static_map) if args.static_map else None
    with ReferenceRuntime(planning,motion,geometry,settings) as runtime:
        bridge=PlanningBridge(runtime,geometry,motion,load_config(args.environment_config)['scene'],asset=asset,clock_domain=clock)
        state_pub=rospy.Publisher('/navigation/planning/state',String,queue_size=1,latch=True)
        path_pub=rospy.Publisher('/navigation/planning/trajectory',Path,queue_size=1,latch=True)
        shadow=None
        from navigation.control.shadow import DwaShadowController,load_config as shadow_config
        from navigation.control.dwa import load_config as dwa_config
        direct_control=DwaShadowController(planning,motion,geometry,asset=asset,
            settings=shadow_config(args.shadow_config),dwa_settings=dwa_config(args.dwa_config))
        if args.shadow:
            from navigation.control.shadow import TOPICS,publisher
            shadow=direct_control
            observation_pub=publisher(rospy,TOPICS['observation'],String,queue_size=1)
            proposal_pub=publisher(rospy,TOPICS['proposal'],String,queue_size=1)
        def command(kind,message):
            try:bridge.push_command(kind,json.loads(message.data))
            except (ValueError,TypeError) as error:rospy.logwarn('Invalid planning command: %s',error)
        def people(message):
            try:bridge.push_people(json.loads(message.data))
            except (ValueError,TypeError) as error:rospy.logwarn('Invalid people input: %s',error)
        subscribers=[rospy.Subscriber(args.grid_topic,numpy_msg(EnvironmentGrid),bridge.push_grid,queue_size=1,buff_size=16*1024*1024),
            rospy.Subscriber(args.state_topic,LocalizationState,lambda m:bridge.push_state(state_row(m)),queue_size=128),
            rospy.Subscriber(args.people_topic,String,people,queue_size=16),
            rospy.Subscriber('/navigation/planning/goal',String,lambda m:command('goal',m),queue_size=16),
            rospy.Subscriber('/navigation/planning/cancel',String,lambda m:command('cancel',m),queue_size=16)]
        previous=None;last_publish=0.;last_shadow=-float('inf');last_observation=None;events=[];last_pose=None;blocked_since=None;direct_reason=None
        try:
            while not rospy.is_shutdown():
                began=time.perf_counter();now=rospy.Time.now().to_sec()
                status=bridge.step(now);result=status['result']
                mono=time.monotonic();packet=None
                reference=result
                now=rospy.Time.now().to_sec()
                changed=direct_control.accept(reference,bridge.request,bridge.scene,now=now,clock_domain=clock)
                with bridge.lock:rows=list(bridge.rows)
                pose_key=None if not rows else (rows[-1]['stamp_s'],rows[-1]['health'])
                if changed or pose_key!=last_pose or mono-last_shadow>=direct_control.settings['period_s']:
                    packet=direct_control.step(bridge.request,bridge.scene,rows,reference,now=now,clock_domain=clock,mono=mono)
                    last_shadow=mono;last_pose=pose_key
                    p=packet.get('proposal') or {}
                    stopping=(p.get('status')=='candidate' and p.get('velocity',[0])[0]<.03 and
                        not (packet.get('recorded_goal_diagnostic') or {}).get('reached_xy',False))
                    if p.get('reason')=='no_admissible_velocity' or stopping:
                        if blocked_since is None:blocked_since=mono
                        if mono-blocked_since>=1.:runtime.request_replan()
                    else:blocked_since=None
                result=direct_control.controller.result
                if bridge.request is None or bridge.scene is None or (result is not None and
                        not 0<=now-result['input_binding']['as_of']<=motion['state_max_age_s']):result=None
                status.update(result=result,status='control_ready' if result else 'unavailable',
                    reference_binding=None if reference is None else reference['binding'])
                if packet is not None:direct_reason=packet['reason'] or (packet.get('proposal') or {}).get('reason')
                status['reason']=direct_reason
                events.extend(runtime.drain_events());events=events[-256:]
                key=(None if result is None else result['result_binding'],status['reason'],fingerprint(status['goal']),status['command_error'])
                if shadow is not None and packet is not None:
                    scene=bridge.scene
                    if scene is not None and scene.record['input_fingerprint']!=last_observation:
                        observation_pub.publish(String(data=json.dumps(scene.record,ensure_ascii=False,allow_nan=False)))
                        last_observation=scene.record['input_fingerprint']
                    with bridge.lock:rows=list(bridge.rows)
                    proposal_pub.publish(String(data=json.dumps(packet,ensure_ascii=False,allow_nan=False)));last_shadow=mono
                if key!=previous or mono-last_publish>=.2:
                    path_pub.publish(trajectory_path(result,rospy))
                    # The compact state includes full candidate timing and input binding,
                    # without copying rejected candidates or the global map into ROS JSON.
                    wire={k:v for k,v in status.items() if k!='result'}
                    wire['worker_events']=events;events=[];wire['owner_compute_ms']=(time.perf_counter()-began)*1000
                    wire['result']=None if result is None else {k:result[k] for k in
                        ('result_binding','input_binding','request_binding','configuration_binding','local_goal','trajectory','control_mode',
                         'candidate_count','ranked_candidates','valid_candidates','certification_attempts','unattempted_candidates','selection_stopped_early','rejections','timing_ms') if k in result}
                    state_pub.publish(String(data=json.dumps(wire,ensure_ascii=False,allow_nan=False)))
                    previous=key;last_publish=mono
                time.sleep(.01)  # Wall time: a paused /clock must still revoke stale paths.
        finally:
            if shadow is not None:
                packet=shadow.step(None,None,[],None,now=rospy.Time.now().to_sec(),clock_domain=clock,reason='node_shutdown')
                proposal_pub.publish(String(data=json.dumps(packet,allow_nan=False)))
            path_pub.publish(Path())
            state_pub.publish(String(data=json.dumps(dict(status='unavailable',reason='node_shutdown',executable=False,task_completed=False))))
            for subscriber in subscribers:subscriber.unregister()
