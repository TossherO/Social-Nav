"""Follow re-identification retains the fixed reference and requires a stopped owner."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from decision.agent import ROOT
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.observations import Observation
from decision.agent.tools.places import Catalog
from decision.vlm.follow_target import candidates
from perception.environment.context import fingerprint
from perception.environment.layers import load_config
from perception.environment.scene import assemble_scene
from tests.test_agent import ScriptedFeedback
from tests.test_decision_runtime import observation
from tests.navigation_fixtures import asset,scene_at
from navigation.planning.footprint import VehicleGeometry
import numpy as np


def candidate_observation(stamp,track_id='a',*,associated=True):
    row,arrays,_,scene=observation(stamp)
    row['tracks'][0]['track_id']=track_id
    row['input_fingerprint']=fingerprint({k:v for k,v in row.items() if k!='input_fingerprint'})
    scene=assemble_scene(scene.geometry,clock_domain='synthetic',people=row,settings=load_config()['scene'])
    detections=[dict(box=[20.,5.,35.,40.],score=.9,class_id=0)]
    association=dict(status='complete',tracks={track_id+'/g7':dict(status='matched' if associated else 'unmatched',
        rgb_box=[20.,5.,35.,40.],detection_index=0 if associated else None)})
    value=candidates(scene,arrays['rgb'],stamp-.02,detections,association,detector=dict(backend='test'))
    return Observation.from_scene(scene,rgb=dict(pixels=arrays['rgb'],stamp_s=stamp-.02,frame_id='camera'),
                                  follow_candidates=value)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(Path(self.tmp.name)/'session.sqlite')
        self.nav=ScriptedFeedback();self.nav.clock_domain='synthetic'
        self.nav.state.update(clock_domain='synthetic',session_id='test',generation=0,frame_id='odom')
        self.obs=candidate_observation(10.)
        self.nav.observe=lambda:self.obs
        catalog=copy.deepcopy(Catalog.load(ROOT/'decision/agent/configs/simulation.json').value)
        catalog['scope']='development_prior';catalog['map_snapshot_id']='static-a'
        self.session=AgentSession('跟随红衣人物到门口',Catalog(catalog),self.nav,self.store)
        self.session.call('record_task',goals=[dict(goal_id='g',kind='skill',stage_type='follow',target_id='',description='跟随红衣人物到门口')])
        self.session.call('observe_scene')
        self.session.call('select_follow_target',_context=self.session.context(),goal_id='g',
            observation_id=self.obs.record['observation_id'],candidate_binding=self.obs.record['follow_candidates']['binding'],
            target_description='红衣人物',match='unique',candidate_id=0)
        self.original=copy.deepcopy(self.session.follow_reference)
        self.session.follow_tracking_reference=copy.deepcopy(self.original)
        handle=dict(task_id=self.session.state['task_id'],goal_id='g',plan_revision=1,goal_revision=1,call_id='old')
        last=dict(identity=['tracker_session',7,'a'],stamp_s=10.,position_xy_m=[4.,2.],
                  velocity_xy_mps=[.2,0.],clock_domain='synthetic',local_identity=['test',0,'odom'])
        last['binding']=fingerprint(last)
        self.nav.scope='live';self.nav.state.update(scope='live',status='failed',stop_observation=True,stopped=True,
            chassis_state='armed',chassis_status_fresh=True,localization_status='valid',input_stale=False,
            follow_status=dict(status='failed',handle=handle,reason='follow_track_lost',recovery_available=True,
                               last_evidence=last))
        self.session.skill.record.update(stage_type='follow',goal_id='g',status='failed',handle=None,goal_revision=1)

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def recover(self,match='unique',candidate_id=0):
        self.session._applying_context=self.session.context()
        try:
            return self.session._accept_follow_recovery(goal_id='g',reference_id=self.original['reference_id'],
                observation_id=self.obs.record['observation_id'],candidate_binding=self.obs.record['follow_candidates']['binding'],
                match=match,candidate_id=candidate_id,evidence='合成T2输出')
        finally:self.session._applying_context=None

    def set_current(self,*,associated=True):
        self.obs=candidate_observation(10.5,'b',associated=associated)
        self.nav.state['stamp_s']=10.5
        self.session.call('observe_scene')

    def test_new_track_is_recorded_and_revalidated_by_runtime_on_start(self):
        self.set_current();result=self.recover()
        self.assertEqual(result['status'],'selected')
        record=result['reference']
        self.assertEqual(record['previous_identity'],['tracker_session',7,'a'])
        self.assertEqual(record['track']['identity'],['tracker_session',7,'b'])
        self.assertEqual(record['original_reference_id'],self.original['reference_id'])
        self.assertEqual(self.store.read_follow_recovery(record['reference_id'],self.session.state['task_id'],1,'g')[0],record)
        self.assertEqual(self.store.read_follow_reference(self.session.state['task_id'],1,'g')[0],self.original)
        self.nav.check_route=lambda _:dict(proposal_allowed=True)
        calls=[]
        self.nav.start_follow=lambda ref,handle:(calls.append(copy.deepcopy(ref)) or dict(accepted=True,handle=handle))
        self.assertEqual(self.session.call('start_follow',goal_id='g')['status'],'accepted')
        self.assertEqual(calls[0]['track']['identity'],['tracker_session',7,'b'])
        self.assertFalse(result['current_track_valid'])

    def test_ambiguous_and_two_dimensional_matches_do_not_resume(self):
        self.set_current();self.assertEqual(self.recover('ambiguous',-1)['status'],'ambiguous')
        self.set_current(associated=False);self.assertEqual(self.recover()['status'],'visual_only')
        self.assertIsNone(self.session.follow_recovery_reference)
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertEqual(self.session.call('start_follow',goal_id='g')['reason'],
                         'follow_reidentification_required')

    def test_new_loss_clears_previous_recovered_identity(self):
        self.session.follow_recovery_reference=copy.deepcopy(self.original)
        self.session._remember_follow_loss('g')
        self.assertIsNone(self.session.follow_recovery_reference)
        self.assertEqual(self.session.call('start_follow',goal_id='g')['reason'],
                         'follow_reidentification_required')

    def test_loop_exit_cancels_retained_follow_owner(self):
        from decision.agent.core.loop import AgentLoop
        self.nav.handle=copy.deepcopy(self.nav.state['follow_status']['handle'])
        self.session.state['status']='waiting_follow_recovery'
        actor=AgentLoop(self.session,lambda *_:None)
        try:
            with patch.object(actor,'tick'),patch.object(actor,'stop') as stop:
                actor.run(driver=type('Driver',(),{'poll':lambda *_:'exit'})())
            stop.assert_called_once_with(reason='exit')
        finally:actor.close()

    def test_stop_and_original_reference_are_mandatory(self):
        self.set_current();self.nav.state['stop_observation']=False
        self.assertEqual(self.recover()['reason'],'follow_stop_not_confirmed')
        self.nav.state['stop_observation']=True
        self.session._applying_context=self.session.context()
        result=self.session._accept_follow_recovery(goal_id='g',reference_id='wrong',observation_id=self.obs.record['observation_id'],
            candidate_binding=self.obs.record['follow_candidates']['binding'],match='unique',candidate_id=0,evidence='红衣')
        self.session._applying_context=None
        self.assertEqual(result['reason'],'follow_loss_not_ready_for_recovery')

    def test_async_t2_response_is_preserved_and_bound_before_selection(self):
        self.set_current();self.nav.t=.5
        def infer(reference,crop,obs):
            value=obs.record['follow_candidates'];raw=dict(id=0,match='unique')
            return dict(status='complete',reference_id=reference['reference_id'],observation_id=value['observation_id'],
                candidate_binding=value['binding'],backend='remote_sglang_dspark',scene_binding=value['scene_binding'],
                clock_domain=value['clock_domain'],source_stamp_s=value['stamp_s'],rgb_stamp_s=value['rgb_stamp_s'],
                raw_content=json.dumps(raw),response=raw)
        with patch('decision.vlm.recovery.run',side_effect=infer):
            result=self.session.call('request_follow_recovery',goal_id='g',reference_id=self.original['reference_id'])
            self.assertEqual(result['status'],'pending')
            self.session.follow_recovery_worker.thread.join(2);self.session.tick()
        self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.session.follow_recovery_reference['track']['identity'],['tracker_session',7,'b'])
        row=self.store.db.execute("SELECT value FROM events WHERE kind='follow_recovery_raw' ORDER BY seq DESC LIMIT 1").fetchone()
        self.assertEqual(json.loads(row[0])['result']['raw_content'],json.dumps(dict(id=0,match='unique')))

    def test_cancelled_t2_response_cannot_restore_follow(self):
        self.set_current();self.nav.t=.5;gate=threading.Event()
        def infer(reference,crop,obs):
            gate.wait(2);value=obs.record['follow_candidates'];raw=dict(id=0,match='unique')
            return dict(status='complete',reference_id=reference['reference_id'],observation_id=value['observation_id'],
                candidate_binding=value['binding'],backend='remote_sglang_dspark',scene_binding=value['scene_binding'],
                clock_domain=value['clock_domain'],source_stamp_s=value['stamp_s'],rgb_stamp_s=value['rgb_stamp_s'],
                raw_content=json.dumps(raw),response=raw)
        try:
            with patch('decision.vlm.recovery.run',side_effect=infer):
                self.session.call('request_follow_recovery',goal_id='g',reference_id=self.original['reference_id'])
                self.session.event('cancel');gate.set();self.session.follow_recovery_worker.thread.join(2);self.session.tick()
            self.assertEqual(self.session.state['status'],'cancelled')
            self.assertIsNone(self.session.follow_recovery_reference)
            row=self.store.db.execute("SELECT value FROM events WHERE kind='follow_recovery_raw' ORDER BY seq DESC LIMIT 1").fetchone()
            self.assertTrue(json.loads(row[0])['stale'])
        finally:gate.set()

    def test_unavailable_t2_does_not_replay_identical_input(self):
        self.set_current();self.nav.t=.5
        with patch('decision.vlm.recovery.run',return_value=dict(status='unavailable',reason='remote_dspark_unavailable')):
            self.session.call('request_follow_recovery',goal_id='g',reference_id=self.original['reference_id'])
            self.session.follow_recovery_worker.thread.join(2);self.session.tick()
        result=self.session.call('request_follow_recovery',goal_id='g',reference_id=self.original['reference_id'])
        self.assertEqual(result['reason'],'follow_recovery_input_already_attempted')

    def test_bounded_search_then_recovery_preserves_original_loss(self):
        self.set_current();grid=asset().snapshot;grid.metadata['frame_id']=self.session.catalog.value['frame_id']
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        inputs=dict(global_grid=grid,scene=scene_at(10.5),pose=[2.,2.,0.],local_pose=[2.,2.,0.],
                    stamp_s=10.5,clock_domain='synthetic',geometry=geometry)
        self.nav.spatial_inputs=lambda:inputs
        self.nav.check_route=lambda _:dict(proposal_allowed=True,reference_found=False,status='available')
        prepared=self.session.call('prepare_follow_search',goal_id='g')
        self.assertEqual(prepared['status'],'available')
        chosen=next(v for v in prepared['candidate_set']['candidates'] if v['kind']=='rotate')
        binding=prepared['candidate_set']['binding']
        result=self.session.call('execute_follow_search',_context=self.session.context(),goal_id='g',
                                 candidate_binding=binding,candidate_id=chosen['id'])
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(self.session.follow_search['attempts'],1)
        self.assertIsNone(self.session.follow_search['candidate_binding'])
        handle=copy.deepcopy(self.session.skill.record['handle'])
        self.session.skill.record.update(status='held',handle=handle)
        self.nav.state.update(handle=handle,status='stopped',stop_observation=True,stopped=True,follow_status=None)
        self.assertEqual(self.session.call('observe_scene')['status'],'available')
        result=self.recover()
        self.assertEqual(result['status'],'selected')
        self.assertEqual(result['reference']['loss_handle']['call_id'],'old')
        self.assertEqual(result['reference']['track']['identity'],['tracker_session',7,'b'])

    def test_search_rechecks_changed_unknown_space_before_submitting(self):
        self.set_current();static=asset().snapshot;static.metadata['frame_id']=self.session.catalog.value['frame_id']
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        inputs=dict(global_grid=static,scene=scene_at(10.5),pose=[2.,2.,0.],local_pose=[2.,2.,0.],
                    stamp_s=10.5,clock_domain='synthetic',geometry=geometry)
        self.nav.spatial_inputs=lambda:inputs
        prepared=self.session.call('prepare_follow_search',goal_id='g')
        chosen=next(v for v in prepared['candidate_set']['candidates'] if v['kind']=='approach')
        cells=np.zeros((60,60),np.int8);cells[20,25]=-1
        from perception.environment.layers import GridSnapshot
        source=GridSnapshot(copy.deepcopy(inputs['scene'].geometry.metadata),dict(occupancy=cells))
        inputs['scene']=assemble_scene(source,clock_domain='synthetic',settings=load_config()['scene'])
        result=self.session.call('execute_follow_search',_context=self.session.context(),goal_id='g',
            candidate_binding=prepared['candidate_set']['binding'],candidate_id=chosen['id'])
        self.assertEqual(result['reason'],'follow_search_current_sweep_not_free')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_wait_search_attempts_are_bounded(self):
        self.set_current();self.nav.t=.5
        static=asset().snapshot;static.metadata['frame_id']=self.session.catalog.value['frame_id']
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.nav.spatial_inputs=lambda:dict(global_grid=static,scene=scene_at(self.nav.state['stamp_s']),
            pose=[2.,2.,0.],local_pose=[2.,2.,0.],stamp_s=self.nav.state['stamp_s'],clock_domain='synthetic',geometry=geometry)
        for i in range(3):
            prepared=self.session.call('prepare_follow_search',goal_id='g')
            self.assertEqual(prepared['status'],'available')
            result=self.session.call('execute_follow_search',_context=self.session.context(),goal_id='g',
                candidate_binding=prepared['candidate_set']['binding'],candidate_id=0)
            self.assertEqual(result['status'],'waiting')
            self.obs=candidate_observation(10.6+i*.1,'b')
            self.session.tick()
            self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.session.call('prepare_follow_search',goal_id='g')['status'],'exhausted')


if __name__=='__main__':unittest.main()
