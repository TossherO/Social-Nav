"""Decision/runtime ownership, time, identity and navigation adapter boundaries."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from runtime.decision import DecisionBuffer, DecisionRecorder
from decision.vlm.observations import freeze_window, frame
from decision.vlm.navigation import DecisionRequest
from decision.vlm.annotation.rules import digest
from decision.vlm.seating.inputs import observation_clock, bind_scene_observation
from decision.vlm.seating.geometry import check_navigation
from decision.vlm.seating.stage import SeatingStage, CompletionLimits
from navigation.tasks import TaskPolicy
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from perception.environment.context import fingerprint
from perception.environment.scene import assemble_scene
from perception.environment.layers import load_config as environment_config
from tests.test_environment_scene import person
from tests.test_navigation_planning import grid
from tests.test_navigation_prediction import track


def observation(stamp,generation=0):
    snapshot=grid(generation=generation);snapshot.metadata['stamp_s']=stamp
    row=person(stamp)
    row.update(observation_id=str(round(stamp*1e9)),pose_health='valid',pose_source='localization',
        pose_stamp_s=stamp,timestamp_source='sensor_header',world_from_lidar=snapshot.metadata['world_from_lidar'],
        localization_binding=dict(session_id='test',generation=generation,frame_id='odom',source='fast_lio'))
    row['tracks']=[dict(track(4,2,(.5,0,0),stamp=stamp),lifecycle_generation=7,state='observed')]
    row['covariance_order']=['x','y','z','dx','dy','dz','yaw','vx','vy','vz']
    row['input_fingerprint']=fingerprint(row)
    arrays=dict(raw=np.array([[1.,0.,0.,3.],[2.,1.,1.,5.]],np.float32),rgb=np.full((48,64,3),120,np.uint8))
    association={k:row[k] for k in ('session_id','generation','sequence_id','clock_domain','observation_id','stamp_s')}
    association.update(perception_input_fingerprint=row['input_fingerprint'],
        image_input=dict(raw_rgb_sha256=hashlib.sha256(arrays['rgb'].tobytes()).hexdigest(),
            rgb_stamp_ns=round((stamp-.02)*1e9),image_size=[64,48]),
        result=dict(status='complete',tracks={'a/g7':dict(status='matched',rgb_box=[20,5,35,40])}))
    scene=assemble_scene(snapshot,clock_domain='synthetic',settings=environment_config()['scene'],people=row)
    return row,arrays,association,scene


def candidate_set(scene):
    r=scene.record
    return dict(instruction='在桌边就座',observation_id=r['observation_id'],stamp_s=r['as_of'],
        **{k:r[k] for k in ('frame_id','clock_domain','session_id','generation')},revision=0,
        candidates=[dict(id=0,pose=[4.,2.,0.],motion='forward',geometry='unknown')])


class DecisionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.planning=load_config()

    def fill(self,buffer):
        for i in range(12):
            row,arrays,rgb,scene=observation(10+i/10)
            buffer.push(row,arrays,rgb,scene=scene)
        return row,arrays,rgb,scene

    def test_window_lease_survives_input_reuse_and_localization_reset(self):
        buffer=DecisionBuffer(max_frames=12,slot_bytes=32768)
        try:
            row,arrays,rgb,scene=self.fill(buffer);lease=buffer.acquire()
            self.assertIsNotNone(lease);self.assertIsNone(buffer.acquire())
            arrays['rgb'][:]=0
            for i in range(12):
                r,a,b,s=observation(12+i/10,generation=1);buffer.push(r,a,b,scene=s)
            with buffer.consume(lease) as read:
                value=read(lease['entries'][-1]['buffer'])
                self.assertEqual(int(value['rgb'][0,0,0]),120)
                self.assertFalse(value['rgb'].flags.writeable)
                del value
            fresh=buffer.acquire()
            self.assertIsNotNone(fresh)
            with buffer.consume(fresh):pass
        finally:buffer.close()

    def test_real_renderer_and_pid_map_freeze_without_annotation_or_model(self):
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            _,_,_,scene=self.fill(buffer);lease=buffer.acquire()
            with tempfile.TemporaryDirectory() as temporary,buffer.consume(lease) as read:
                result=freeze_window(lease['entries'],lease['history'],read,Path(temporary)/'input',extent=[-4,4,-4,4],view_binding={})
                self.assertEqual(result['fields']['dt'],[-1.,-.5,0.])
                self.assertEqual(result['pid_to_track'],{'0':['tracker_session',7,'a']})
                self.assertEqual(len(result['images']),3)
                policy=TaskPolicy('follow','follow',self.planning,self.geometry);gate=DecisionRequest(policy)
                request=gate.begin(scene,window=result,now=11.1)
                with self.assertRaises(ValueError):gate.begin(scene,window=result,now=11.1)
                accepted=gate.accept(request,'{"id":0,"state":"visible"}',scene=scene,now=11.1)
                self.assertEqual(accepted['operation'],'plan');self.assertFalse(accepted['executable'])
                with self.assertRaises(ValueError):gate.accept(request,'{"id":0,"state":"visible"}',scene=scene,now=11.1)
        finally:buffer.close()

    def test_missing_history_and_future_rgb_are_not_substituted(self):
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            row,arrays,rgb,scene=observation(10)
            buffer.push(row,arrays,rgb,scene=scene);self.assertIsNone(buffer.acquire())
            self.fill(buffer);lease=buffer.acquire()
            with buffer.consume(lease) as read:
                entry=copy.deepcopy(lease['entries'][-1])
                entry['association']['image_input']['rgb_stamp_ns']+=100_000_000
                with self.assertRaises(ValueError):frame(entry,read(entry['buffer']))
        finally:buffer.close()

    def test_async_recorder_drains_and_records_zero_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder=DecisionRecorder(Path(temporary)/'inputs')
            for i in range(13):
                row,arrays,rgb,scene=observation(10+i/10);recorder.submit(row,arrays,rgb,scene=scene)
            result=recorder.close()
            self.assertEqual(result['written'],1);self.assertEqual(result['model_requests'],0)

    def test_rgb_and_pose_loss_do_not_reuse_the_previous_window(self):
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            self.fill(buffer)
            row,arrays,rgb,scene=observation(11.2)
            rgb['image_input']['rgb_stamp_ns']-=1_000_000_000
            buffer.push(row,arrays,rgb,scene=scene)
            self.assertIsNone(buffer.acquire())
            row,arrays,rgb,_=observation(11.3);row['pose_health']='invalid'
            self.assertFalse(buffer.push(row,arrays,rgb))
            self.assertIsNone(buffer.acquire())
        finally:buffer.close()

    def test_follow_retains_identity_across_two_second_cycle_but_revokes_gaps(self):
        row,_,_,scene=observation(10);policy=TaskPolicy('follow','follow',self.planning,self.geometry)
        ticket=policy.observe(scene,pid_to_track={0:['tracker_session',7,'a']})
        policy.respond(ticket,dict(id=0,state='visible'),scene=scene,now=10.)
        for i in range(1,24):
            _,_,_,later=observation(10+i/10)
            if i==20:policy.observe(later,pid_to_track={0:['tracker_session',7,'a']})
            result=policy.update(later,now=10+i/10)
            self.assertEqual(result['operation'],'plan')
            self.assertEqual(result['request']['decision_source'],ticket['binding'])
        self.assertEqual(policy.last_visual,10.)
        _,_,_,later=observation(13.)
        self.assertEqual(policy.update(later,now=13.)['reason'],'follow_continuity_lost')
        _,_,_,later=observation(13.1)
        self.assertEqual(policy.update(later,now=13.1)['operation'],'stop')

    def test_response_deadline_and_cancellation_remain_strict(self):
        _,_,_,scene=observation(10);policy=TaskPolicy('seat','seating',self.planning,self.geometry);gate=DecisionRequest(policy)
        envelope=gate.begin(scene,candidates=candidate_set(scene),now=10)
        with self.assertRaises(ValueError):gate.accept(envelope,'{"p":0,"s":"v"}',scene=scene,now=12.1)
        self.assertEqual(gate.last_response['raw_response'],'{"p":0,"s":"v"}')
        self.assertIsNone(policy.request)
        envelope=gate.begin(scene,candidates=candidate_set(scene),now=10)
        gate.cancel()
        with self.assertRaises(ValueError):gate.accept(envelope,dict(p=0,s='v'),scene=scene,now=10)

    def test_clock_and_scene_binding_do_not_retag_frozen_sources(self):
        self.assertEqual(observation_clock(dict(runtime={},backend='native_roll_fast_lio',source='bag'),[]),'runtime_bag')
        with self.assertRaises(ValueError):observation_clock(dict(clock_domain='runtime_bag'),[],explicit='ros_sim_time')
        with self.assertRaises(ValueError):observation_clock({},[])
        _,_,_,scene=observation(10);r=scene.record
        raw=dict(observation_id='raw',stamp_s=10.,source={},world_from_lidar=r['localization']['world_from_lidar'],
            **{k:r[k] for k in ('frame_id','session_id','generation','clock_domain')})
        bound=bind_scene_observation(raw,scene)
        self.assertEqual(raw['observation_id'],'raw');self.assertEqual(bound['observation_id'],r['observation_id'])
        with self.assertRaises(ValueError):bind_scene_observation(dict(raw,clock_domain='ros_sim_time'),scene)

    def test_seating_preflight_keeps_candidates_unknown_and_blocks_reverse(self):
        _,_,_,scene=observation(10);data=candidate_set(scene)
        data['candidates'][0]['pose']=[2.5,2.,0.]
        data['candidates'].append(dict(id=1,pose=[1.,2.,0.],motion='reverse',geometry='unknown'))
        before=copy.deepcopy(data)
        result=check_navigation(data,scene,self.planning,load_motion_config(),self.geometry,now=10.)
        self.assertEqual(result['candidates'][0]['status'],'path_found',result['candidates'][0])
        self.assertEqual(result['candidates'][1]['status'],'not_planned')
        self.assertEqual(data,before);self.assertFalse(result['executable'])

    def test_seating_feedback_rejects_shadow_and_wrong_revision(self):
        _,_,_,scene=observation(10);data=candidate_set(scene)
        stage=SeatingStage('seat',data,CompletionLimits(.15,.15,.02,.03,1.,.3,.5),started_s=10.,timeout_s=30.)
        policy=TaskPolicy('seat','seating',self.planning,self.geometry)
        ticket=policy.observe(scene,candidates=data)
        request=policy.respond(ticket,dict(p=0,s='v'),scene=scene,now=10.)['request']
        stage.select(dict(p=0,s='v'),stage.binding)
        stage.geometry(task_id='seat',candidate_binding=stage.binding,feasible=True,source='synthetic',stamp_s=10.,now_s=10.)
        feedback=dict(task_id='seat',goal_revision=request['goal_revision'],request_binding=fingerprint(request),
            source='synthetic_actuator',execution_feedback=True,scope='simulation',local_identity=['test',0,'odom'],
            pose=[4.,2.,0.],velocity=[0.,0.],clock_domain='synthetic',stamp_s=10.,status='valid',stopped=False)
        options=dict(navigation_state='arrived',geometry_valid=True,table_relation_valid=True,now_s=10.,simulation=True)
        with self.assertRaises(ValueError):stage.observe_navigation(request,dict(feedback,scope='shadow'),**options)
        with self.assertRaises(ValueError):stage.observe_navigation(request,dict(feedback,goal_revision=99),**options)
        result=stage.observe_navigation(request,feedback,**options)
        self.assertEqual(result['state'],'verifying');self.assertEqual(result['feedback_scope'],'simulation')


if __name__=='__main__':unittest.main()
