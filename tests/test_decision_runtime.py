"""Decision/runtime ownership, time, identity and navigation adapter boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from runtime.decision import DecisionBuffer, DecisionRecorder
from decision.vlm.observations import freeze_window, frame
from decision.vlm.navigation import DecisionRequest
from decision.vlm.annotation.rules import digest
from decision.vlm.t3.inputs import observation_clock, bind_scene_observation
from decision.vlm.t3.geometry import check_navigation
from decision.vlm.t3.stage import SeatingStage, CompletionLimits
from navigation.tasks import TaskPolicy
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from perception.environment.context import fingerprint
from tests.decision_fixtures import candidate_set, observation


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

    def test_pose_only_scan_restores_causal_rgb_compensation(self):
        from perception.detection_tracking.inference.rgb_association import causal_geometry,people_in_lidar
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            for i in range(13):
                row,arrays,association,scene=observation(round(10+i*.1,1))
                published=buffer.push(row,arrays,association,scene=scene if i%2==0 else None,
                                      keep_association_history=True)
                self.assertEqual(published,i%2==0)
            lease=buffer.acquire()
            with buffer.consume(lease):
                self.assertEqual(len(lease['history']),7)
                self.assertEqual(len(lease['association_history']),13)
                rgb_stamp=11.13
                current=people_in_lidar(row)
                sparse=causal_geometry(current,row,[r['result'] for r in lease['history']],rgb_stamp,.05)
                dense=causal_geometry(current,row,[r['result'] for r in lease['association_history']],rgb_stamp,.05)
                self.assertNotIn('compensation',sparse[0])
                self.assertEqual(dense[0]['compensation'],'causal_track_prediction')
                self.assertEqual(dense[0]['geometry_stamp_s'],11.1)
            restarted,*_=observation(12.,generation=1)
            self.assertFalse(buffer.push(restarted,{},None,keep_association_history=True))
            self.assertIsNone(buffer.acquire())
            self.assertEqual(len(buffer.association_history),1)
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
            with self.assertRaisesRegex(ValueError,'structured T2'):
                    gate.begin(scene,window=result,now=11.1)
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

    def test_seating_preflight_keeps_candidates_unknown_and_respects_reverse_policy(self):
        _,_,_,scene=observation(10);data=candidate_set(scene)
        data['candidates'][0]['pose']=[2.5,2.,0.]
        data['candidates'].append(dict(id=1,pose=[1.,2.,0.],motion='reverse',geometry='unknown'))
        before=copy.deepcopy(data)
        motion=load_motion_config();motion['allow_reverse']=False
        result=check_navigation(data,scene,self.planning,motion,self.geometry,now=10.)
        self.assertEqual(result['candidates'][0]['status'],'path_found',result['candidates'][0])
        self.assertEqual(result['candidates'][1]['status'],'not_planned')
        self.assertEqual(data,before);self.assertFalse(result['executable'])
        motion['allow_reverse']=True
        allowed=check_navigation(data,scene,self.planning,motion,self.geometry,now=10.)
        self.assertEqual(allowed['candidates'][1]['status'],'path_found')
        self.assertEqual(data,before);self.assertFalse(allowed['executable'])

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
