"""Independent ROS shadow gate: private TwistStamped only, never chassis control."""
import argparse
from collections import OrderedDict,deque
import copy
import json
import threading
import time
import numpy as np
from perception.environment.context import fingerprint
from perception.environment.scene import SceneInputs,snapshot_from_message
from navigation.control.shadow import ClockWatch,TOPICS,load_config,observed_feedback,publisher
from navigation.planning.online import goal_request,planning_scene
from navigation.planning.continuity import goal_binding,context_binding
from .guard import LocalSafety


def chassis_authorization(result, *, arm, emergency, manual, active_goal, clock_domain):
    """Convert one completed safety decision to a bounded chassis authorization."""
    allowed=bool(result and result['state']=='allowed' and result.get('completion_freshness_checked'))
    units=[] if result is None else result.get('wire_velocity_units',[])
    # A clipped/changed target needs its own exact wire check.
    allowed=allowed and len(units)==2 and [x/1000. for x in units]==result['velocity']
    if emergency or manual or active_goal is None:allowed=False
    if result is not None and result.get('request')!=active_goal:allowed=False
    completed=time.monotonic() if result is None else result['decision_complete_mono']
    packet=dict(mono=time.monotonic(),arm=arm['enabled'],arm_revision=arm['revision'],
        allowed=allowed,units=units,expires_mono=completed+min(.15,(result or {}).get('remaining_validity_s',0.)))
    feedback=(result or {}).get('feedback') or {}
    packet['diagnostics']=dict(reason=(result or {}).get('reason'),
        safety_sequence=(result or {}).get('sequence_id'),decision_complete_mono=completed,
        decision_stamp_s=(result or {}).get('stamp_s'),
        trajectory_binding=(result or {}).get('trajectory_binding'),
        trajectory_as_of=(result or {}).get('trajectory_as_of'),clock_domain=clock_domain,
        pose_velocity_estimate={key:feedback.get(key) for key in ('source','stamp_s','velocity')})
    return packet


def _association_remaining(scene, now):
    """Bound authorization by the original association when static support is used."""
    proof=scene.geometry.metadata.get('navigation_fusion')
    if proof is None:
        return None
    source=scene.geometry.layers['navigation_source']
    if not ((source==3)|(source==10)).any():return None
    measured=scene.record['prior'].get('measurement_stamp_s')
    limit=proof['max_association_age_s']
    if measured is None or not np.isfinite(measured) or not np.isfinite(limit):
        raise ValueError('association_expired_during_gate')
    age=now-measured
    remaining=limit-age
    if age<0 or remaining<=0:
        raise ValueError('association_expired_during_gate')
    return remaining


class ObservationInbox:
    def __init__(self,limit):
        self.limit=limit;self.grids=OrderedDict();self.records=OrderedDict();self.latest=None;self.cached=None
        self.people=deque(maxlen=64);self.current_key=None;self.current_scene=None

    def _put(self,table,key,value):
        table[key]=value
        while len(table)>self.limit:table.popitem(last=False)

    def push_grid(self,snapshot):
        m=snapshot.metadata
        if self.latest is not None:
            old=self.latest.metadata
            if ([m[k] for k in ('session_id','generation','frame_id')]!=[old[k] for k in ('session_id','generation','frame_id')] or m['stamp_s']<old['stamp_s']):
                self.grids.clear();self.records.clear();self.cached=None
                self.current_key=self.current_scene=None;self.people.clear()
        self.latest=snapshot;self._put(self.grids,m['snapshot_id'],snapshot)

    def push_record(self,record):
        key=record['input_fingerprint']
        if key!=fingerprint({k:v for k,v in record.items() if k!='input_fingerprint'}):raise ValueError('shadow_record_binding_invalid')
        self._put(self.records,key,copy.deepcopy(record))

    def source(self,binding):
        """Capture references under the caller's lock; verification is outside it."""
        record=self.records.get(binding)
        if record is None:raise ValueError('waiting_for_bound_observation')
        # Preserve the verified original snapshot if a later grid reuses its ID;
        # current geometry is always joined and checked separately.
        if self.cached is not None and self.cached.record is record:return self.cached
        snapshot=self.grids.get(record['geometry']['snapshot_id'])
        if snapshot is None:raise ValueError('source_grid_not_received')
        return SceneInputs(record,snapshot,None)

    def push_people(self,people,clock):
        people=copy.deepcopy(people);people['clock_domain']=clock
        self.people.append((people,fingerprint(people)))

    def selection(self):
        if self.latest is None:raise ValueError('current_grid_unavailable')
        past=[p for p in self.people if p[0]['stamp_s']<=self.latest.metadata['stamp_s']]
        selected=max(past,key=lambda p:p[0]['stamp_s']) if past else (None,None)
        # Repeated identifiers do not certify unchanged bytes. Rebuild for each
        # newly received immutable snapshot, even if a producer reuses its ID.
        key=(id(self.latest),selected[1])
        return key,selected[0]

class ShadowSafety:
    def __init__(self,planning,motion,geometry,*,clock_domain,settings=None,asset=None,scene_settings=None,execution=False):
        if execution and clock_domain!='runtime_live':raise ValueError('Hardware safety requires live measurements')
        self.execution=execution
        self.settings=settings or load_config();self.clock_domain=clock_domain;self.motion=motion;self.geometry=geometry
        self.guard=LocalSafety(geometry,motion,planning['safety_margin_m'],people_settings=planning['people'],shadow=not execution,execution=execution,asset=asset)
        self.lock=threading.RLock();self.inbox=ObservationInbox(self.settings['inbox_size']);self.rows=deque(maxlen=256)
        self.packet=None;self.packet_received=self.grid_received=None;self.active_goal=None;self.cancelled={}
        self.manual=False;self.emergency=False;self.sequence=0;self.last_packet=None;self.barrier=None
        self.wake=threading.Event()
        self.clock=ClockWatch(self.settings['clock_stall_s'])
        self.asset=asset
        if scene_settings is None:
            from perception.environment.layers import load_config as environment_config
            scene_settings=environment_config()['scene']
        self.scene_settings=scene_settings

    def push_grid(self,snapshot,*,mono=None):
        with self.lock:
            self.inbox.push_grid(snapshot);self.grid_received=time.monotonic() if mono is None else mono

    def push_record(self,record):
        with self.lock:self.inbox.push_record(record)

    def push_people(self,people):
        with self.lock:self.inbox.push_people(people,self.clock_domain)

    def push_state(self,row):
        with self.lock:
            previous=self.rows[-1] if self.rows else None
            changed=(previous is None or row['stamp_s']!=previous['stamp_s'] or
                any(row.get(k)!=previous.get(k) for k in ('health','status','has_map_pose','map_frame','map_revision')) or
                any(row[k]!=previous[k] for k in ('session_id','generation','frame_id')))
            if self.rows and (row['stamp_s']<self.rows[-1]['stamp_s'] or
                    [row[k] for k in ('session_id','generation','frame_id')]!=[self.rows[-1][k] for k in ('session_id','generation','frame_id')]):
                self.rows.clear();self.active_goal=None;self.barrier='localization_reset'
            self.rows.append(copy.deepcopy(row))
            # Live execution must recertify promptly when a new pose arrives,
            # without changing the age limit or refreshing its source timestamp.
            if self.execution and changed and (self.active_goal is not None or self.barrier):
                self.wake.set()

    def _source_scene(self,binding):
        with self.lock:
            scene=self.inbox.source(binding)
            if scene is self.inbox.cached:return scene
        scene.verify()
        with self.lock:
            current=self.inbox.source(binding)
            if current.record is not scene.record or current.geometry is not scene.geometry:
                raise ValueError('source_observation_changed_during_join')
            self.inbox.cached=scene
        return scene

    def _current_scene(self):
        with self.lock:
            key,people=self.inbox.selection();snapshot=self.inbox.latest
            scene=self.inbox.current_scene if key==self.inbox.current_key else None
        if scene is None:
            scene=planning_scene(snapshot,clock_domain=self.clock_domain,settings=self.scene_settings,people=people)
        with self.lock:
            if self.inbox.selection()[0]==key:
                self.inbox.current_key,self.inbox.current_scene=key,scene
            # A newer selection is handled by the gate's bounded recheck below.
            rows=list(self.rows)
        return scene,key,rows

    def command(self,kind,value):
        with self.lock:
            if kind=='goal':
                request=goal_request(value)
                if request['goal_revision']<=self.cancelled.get(request['task_id'],-1):raise ValueError('cancelled_goal_revision')
                old=self.active_goal
                if old is not None and old['task_id']==request['task_id']:
                    if request['goal_revision']<old['goal_revision']:raise ValueError('old_goal_revision')
                    if request['goal_revision']==old['goal_revision'] and goal_binding(request)!=goal_binding(old):raise ValueError('changed_goal_without_revision')
                self.active_goal=request
            elif kind=='cancel':
                if not isinstance(value,dict) or not isinstance(value.get('task_id'),str) or type(value.get('goal_revision')) is not int or value['goal_revision']<0:
                    raise ValueError('invalid_shadow_cancel')
                self.cancelled[value['task_id']]=max(value['goal_revision'],self.cancelled.get(value['task_id'],-1))
                if (self.active_goal is not None and value['task_id']==self.active_goal['task_id'] and
                        value['goal_revision']>=self.active_goal['goal_revision']):self.active_goal=None
            elif kind=='override':
                if set(value)!={'manual_override','emergency_stop'} or any(type(v) is not bool for v in value.values()):raise ValueError('invalid_shadow_override')
                self.manual,self.emergency=value['manual_override'],value['emergency_stop']
            else:raise ValueError('unknown_shadow_command')
            self.wake.set()

    def push_packet(self,packet,*,mono=None):
        if packet.get('packet_binding')!=fingerprint({k:v for k,v in packet.items() if k!='packet_binding'}):raise ValueError('shadow_packet_binding_invalid')
        if packet.get('executable') is not False or packet.get('task_completed') is not False:raise ValueError('invalid_shadow_scope')
        with self.lock:
            key=(packet['producer_id'],packet['sequence_id'])
            if self.last_packet is not None and key[0]==self.last_packet[0] and key[1]<=self.last_packet[1]:raise ValueError('late_shadow_packet')
            if self.packet is not None and packet['stamp_s']<self.packet['stamp_s']:raise ValueError('older_shadow_producer')
            def event_key(value):
                if value is None:return None
                p=value.get('proposal') or {}
                return value.get('reason'),p.get('status'),p.get('trajectory_binding')
            changed=event_key(self.packet)!=event_key(packet)
            self.last_packet=key;self.packet=copy.deepcopy(packet);self.packet_received=time.monotonic() if mono is None else mono
            # Identical heartbeats refresh the watchdog but do not race the
            # periodic tick into two checks of the same proposal every period.
            if changed:self.wake.set()

    def tick(self,now,*,mono=None,clock_now=None):
        mono=time.monotonic() if mono is None else mono;began=time.perf_counter();self.sequence+=1
        out=dict(sequence_id=self.sequence,stamp_s=now,clock_domain=self.clock_domain,velocity=[0.,0.],
            state='stop_request',reason=None,scope='shadow_hypothetical_actuator_response',
            executable=False,task_completed=False,execution_feedback=False,timing_ms={})
        try:
            with self.lock:
                reason=self.clock.update(now,mono)
                if reason=='clock_reversed':self.active_goal=None;self.packet=None;self.last_packet=None;self.rows.clear()
                if reason:raise ValueError(reason)
                if self.barrier:
                    reason=self.barrier;self.barrier=None;raise ValueError(reason)
                if self.emergency:raise ValueError('emergency_stop')
                if self.manual:raise ValueError('manual_override')
                if self.active_goal is None:raise ValueError('no_active_goal')
                if self.packet is None or mono-self.packet_received>self.settings['packet_timeout_s']:raise ValueError('proposal_timeout')
                packet=self.packet
                if packet['clock_domain']!=self.clock_domain or not 0<=now-packet['stamp_s']<=self.settings['packet_timeout_s']:
                    raise ValueError('proposal_clock_or_age_invalid')
                if packet['reason']:raise ValueError('controller_'+packet['reason'])
                if fingerprint(packet['request'])!=fingerprint(self.active_goal):raise ValueError('active_goal_mismatch')
                if self.grid_received is None or mono-self.grid_received>self.settings['grid_timeout_s']:raise ValueError('grid_wall_timeout')
                original=packet['proposal']
                if original is not None and original.get('control_mode')=='dwa':
                    out['control_diagnostics']={k:original.get(k) for k in ('control_mode','candidate_count','ranked_candidates','valid_candidates','certification_attempts','unattempted_candidates','selection_stopped_early','rejections','timing_ms','compute_ms')}
                out.update(controller_reason=None if original is None else original.get('reason'),controller_ms=packet['compute_ms'],
                    proposed_velocity=None if original is None else original.get('velocity'),
                    proposal_receive_delay_ms=(now-packet['stamp_s'])*1000)
                # No moving proposal means a zero recommendation regardless of
                # geometry. Avoid hashing/assembling full maps just to reject it.
                if original is None or original.get('status')!='candidate':raise ValueError('planner_unavailable')
                request=copy.deepcopy(self.active_goal)
            t=time.perf_counter();source_scene=self._source_scene(packet['scene_binding'])
            out['timing_ms']['source_join']=(time.perf_counter()-t)*1000
            for attempt in range(2):
                check_now=now if clock_now is None else clock_now()
                if check_now<now:raise ValueError('clock_reversed')
                t=time.perf_counter();scene,selection,rows=self._current_scene()
                if context_binding(request,source_scene,self.asset)!=context_binding(request,scene,self.asset):raise ValueError('shadow_context_changed')
                out['timing_ms']['observation_join']=out['timing_ms'].get('observation_join',0.)+(time.perf_counter()-t)*1000
                proposal=copy.deepcopy(original)
                if self.execution and proposal.get('status')=='candidate':
                    # Check the exact quantized wire target, retaining original packet provenance.
                    out['wire_velocity_units']=[int(float(x)*1000.) for x in proposal['velocity']]
                    proposal['velocity']=[x/1000. for x in out['wire_velocity_units']]
                proposal['gate_input_scene_binding']=proposal['scene_binding'];proposal['scene_binding']=scene.record['input_fingerprint']
                t=time.perf_counter();feedback=observed_feedback(rows,scene,self.geometry,self.motion,now=check_now)
                out['timing_ms']['feedback']=out['timing_ms'].get('feedback',0.)+(time.perf_counter()-t)*1000
                out.update(scene_binding=scene.record['input_fingerprint'],as_of=scene.record['as_of'],request_binding=fingerprint(request),
                    proposal_scene_binding=packet['scene_binding'],current_observation_revalidated=True,revalidation_attempts=attempt+1,
                    trajectory_source_scene_binding=proposal.get('source_scene_binding'),
                    request=request,proposal_binding=packet['packet_binding'],trajectory_binding=proposal.get('trajectory_binding'),
                    trajectory_as_of=proposal.get('trajectory_as_of'),feedback=feedback,people_status=scene.record['people']['status'])
                t=time.perf_counter();gate=self.guard.gate(proposal,request,scene,feedback,now=check_now,clock_domain=self.clock_domain,dt=self.settings['period_s'])
                out['timing_ms']['independent_gate']=out['timing_ms'].get('independent_gate',0.)+(time.perf_counter()-t)*1000
                out.update(velocity=gate['velocity'],state=gate['state'],reason=gate['reason'],
                    check=gate.get('check'),people_check=gate.get('people_check'),
                    motion_uncertainty_margin_m=gate.get('motion_uncertainty_margin_m'),motion_sweep=gate.get('motion_sweep'),
                    velocity_filter=gate.get('velocity_filter'),dynamic_geometry=gate.get('dynamic_geometry'))
                with self.lock:
                    if self.emergency:raise ValueError('emergency_stop')
                    if self.manual:raise ValueError('manual_override')
                    if self.active_goal is None or fingerprint(self.active_goal)!=fingerprint(request):raise ValueError('goal_changed_during_gate')
                    latest=self.rows[-1] if self.rows else None
                    if (latest is None or latest['health']!=0 or
                            [latest[k] for k in ('session_id','generation','frame_id')]!=feedback['local_identity']):
                        raise ValueError('localization_changed_during_gate')
                    changed=self.inbox.selection()[0]!=selection
                if gate['state']!='allowed' or not changed:break
                if attempt or time.perf_counter()-began>=self.settings['period_s']:
                    raise ValueError('observation_changed_during_gate')
            if out['state']=='allowed' and clock_now is not None:
                final_now=clock_now()
                if not now<=final_now or not 0<=final_now-proposal['trajectory_as_of']<=self.guard.settings['max_input_age_s']:
                    raise ValueError('proposal_expired_during_gate')
                if not 0<=final_now-packet['stamp_s']<=self.settings['packet_timeout_s']:raise ValueError('proposal_clock_or_age_invalid')
                if not 0<=final_now-feedback['stamp_s']<=self.guard.settings['feedback_max_age_s']:
                    raise ValueError('feedback_expired_during_gate')
                if not 0<=final_now-scene.record['as_of']<=self.guard.settings['max_input_age_s']:
                    raise ValueError('geometry_expired_during_gate')
                association_remaining=_association_remaining(scene,final_now)
                out['remaining_trajectory_validity_s']=self.guard.settings['max_input_age_s']-(final_now-proposal['trajectory_as_of'])
                out['remaining_validity_s']=min(out['remaining_trajectory_validity_s'],
                    self.guard.settings['feedback_max_age_s']-(final_now-feedback['stamp_s']),
                    self.guard.settings['max_input_age_s']-(final_now-scene.record['as_of']),
                    self.settings['packet_timeout_s']-(final_now-packet['stamp_s']))
                if association_remaining is not None:
                    out['remaining_association_validity_s']=association_remaining
                    out['remaining_validity_s']=min(out['remaining_validity_s'],association_remaining)
                out['completion_freshness_checked']=True
            if time.perf_counter()-began>self.settings['period_s']:raise ValueError('gate_deadline_exceeded')
        except (ValueError,KeyError,TypeError,RuntimeError) as error:
            self.guard.last_target=np.zeros(2);out.update(velocity=[0.,0.],state='stop_request',reason=str(error))
        out['timing_ms']['total']=(time.perf_counter()-began)*1000
        return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--geometry',choices=['historical'],required=True)
    parser.add_argument('--config');parser.add_argument('--motion-config');parser.add_argument('--motion-profile');parser.add_argument('--shadow-config')
    parser.add_argument('--static-map');parser.add_argument('--environment-config')
    parser.add_argument('--grid-topic',default='/perception/environment/grid');parser.add_argument('--state-topic',default='/localization/state')
    parser.add_argument('--people-topic',default='/perception/people/json')
    parser.add_argument('--collision-backend',choices=['cpu','cuda'])
    args=parser.parse_args()
    from navigation.planning.inputs import load_config as planning_config
    from navigation.planning.motion import load_motion_config
    from navigation.planning.footprint import VehicleGeometry
    from perception.environment.storage import StaticEnvironmentMap
    from perception.environment.layers import load_config as environment_config
    import rospy
    from std_msgs.msg import String
    from geometry_msgs.msg import TwistStamped
    from social_nav_environment.msg import EnvironmentGrid
    from rospy.numpy_msg import numpy_msg
    from social_nav_localization.msg import LocalizationState
    from perception.mapping_localization.ros.context_node import state_row
    planning=planning_config(args.config);motion=load_motion_config(args.motion_config,profile=args.motion_profile);settings=load_config(args.shadow_config)
    if args.collision_backend:motion['collision_backend']=args.collision_backend
    geometry=VehicleGeometry.historical(planning['geometry_reference']);rospy.init_node('social_nav_shadow_safety')
    clock='ros_sim_time' if rospy.get_param('/use_sim_time',False) else 'ros_time'
    asset=StaticEnvironmentMap(args.static_map) if args.static_map else None
    safety=ShadowSafety(planning,motion,geometry,clock_domain=clock,settings=settings,asset=asset,scene_settings=environment_config(args.environment_config)['scene'])
    state_pub=publisher(rospy,TOPICS['state'],String,queue_size=1)
    velocity_pub=publisher(rospy,TOPICS['velocity'],TwistStamped,queue_size=1)
    def consume(callback,value):
        try:callback(value)
        except (ValueError,KeyError,TypeError) as error:rospy.logwarn_throttle(2.,'Rejected shadow input: %s',error)
    def decode(callback,message):consume(lambda value:callback(json.loads(value)),message.data)
    subscribers=[rospy.Subscriber(args.grid_topic,numpy_msg(EnvironmentGrid),lambda m:consume(safety.push_grid,snapshot_from_message(m)),queue_size=1,buff_size=16*1024*1024),
        rospy.Subscriber(args.state_topic,LocalizationState,lambda m:consume(safety.push_state,state_row(m)),queue_size=128),
        rospy.Subscriber(TOPICS['observation'],String,lambda m:decode(safety.push_record,m),queue_size=4),
        rospy.Subscriber(TOPICS['proposal'],String,lambda m:decode(safety.push_packet,m),queue_size=1),
        rospy.Subscriber(args.people_topic,String,lambda m:decode(safety.push_people,m),queue_size=32),
        rospy.Subscriber('/navigation/planning/goal',String,lambda m:decode(lambda v:safety.command('goal',v),m),queue_size=16),
        rospy.Subscriber('/navigation/planning/cancel',String,lambda m:decode(lambda v:safety.command('cancel',v),m),queue_size=16),
        rospy.Subscriber(TOPICS['override'],String,lambda m:decode(lambda v:safety.command('override',v),m),queue_size=8)]
    def emit(row):
        message=TwistStamped();message.header.stamp=rospy.Time.from_sec(row['stamp_s']);message.header.seq=row['sequence_id']
        # Commands use longitudinal body speed, not odom-frame X velocity.
        message.header.frame_id='base_link'
        message.twist.linear.x,message.twist.angular.z=row['velocity'];velocity_pub.publish(message)
        state_pub.publish(String(data=json.dumps(row,ensure_ascii=False,allow_nan=False)))
    shutdown_sent=False
    def stop():
        nonlocal shutdown_sent
        if shutdown_sent:return
        shutdown_sent=True
        emit(dict(sequence_id=safety.sequence+1,stamp_s=rospy.Time.now().to_sec(),velocity=[0.,0.],state='stop_request',
            reason='node_shutdown',clock_domain=clock,executable=False,task_completed=False,execution_feedback=False,
            scope='ROS_shadow_hypothetical_actuator_response'))
        time.sleep(.05)  # Best effort flush while rospy transports are still open.
    rospy.on_shutdown(stop)
    deadline=time.monotonic()
    try:
        while not rospy.is_shutdown():
            safety.wake.wait(timeout=max(0.,deadline-time.monotonic()));safety.wake.clear()
            mono=time.monotonic()
            emit(safety.tick(rospy.Time.now().to_sec(),mono=mono,clock_now=lambda:rospy.Time.now().to_sec()))
            deadline=mono+settings['period_s']
    finally:
        stop()
        for subscriber in subscribers:subscriber.unregister()


if __name__=='__main__':main()
