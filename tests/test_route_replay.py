"""Recorded legs may supply goals, never floor confirmation or execution success."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from navigation.planning.inputs import load_config
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.route_replay import RouteReplay
from perception.mapping_localization.results import file_fingerprint
from perception.environment.scene import assemble_scene
from perception.environment.layers import load_config as environment_config
from tests.test_navigation_planning import grid,asset
from tests.test_navigation_route import reference


def scene_at(stamp,*,associated=True,generation=0,x=2.):
    snapshot=grid(generation=generation);snapshot.metadata.update(stamp_s=stamp,snapshot_id=str((stamp,x,generation)))
    snapshot.metadata['world_from_lidar'][0][3]=x
    snapshot.metadata['association']=dict(valid=associated,map_id='map-a',map_frame='map',local_frame='odom',
        local_session_id='test',local_generation=generation,stamp_s=stamp,measurement_stamp_s=stamp-.1,
        map_from_local=np.eye(4).tolist(),association_session='association',association_generation=0)
    return assemble_scene(snapshot,clock_domain='synthetic',settings=environment_config()['scene'])


class RouteReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.directory=Path(self.tmp.name)
        self.bag=self.directory/'source.bag';self.bag.write_bytes(b'synthetic reference identity, no ROS bag decoding')
        self.record=reference();self.record['source']=dict(bag_fingerprint=file_fingerprint(self.bag),
            source_start_s=0.,not_blind_evaluation=True)
        self.record['segments'][0].update(source=dict(start_offset_s=9,end_offset_s=12))
        self.record['segments'][1].update(source=dict(start_offset_s=12,end_offset_s=20))
        self.path=self.directory/'route.json'
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.settings=dict(lookahead_m=3.,min_update_s=2.,goal_shift_m=1.)

    def replay(self):
        self.path.write_text(json.dumps(self.record))
        return RouteReplay(self.path,self.bag,'f1',self.geometry,load_config(),self.settings,asset())

    def step(self,replay,stamp,**kwargs):
        scene=scene_at(stamp,**kwargs)
        state={k:scene.record[k] for k in ('session_id','generation','frame_id')};state.update(stamp_s=stamp,health=0)
        return replay.step(scene,state,now=stamp,clock_domain='synthetic')

    def test_mapped_to_local_uses_only_a_pre_exit_anchor(self):
        replay=self.replay();first=self.step(replay,10.)
        self.assertEqual(first[0],'goal');self.assertEqual(first[1]['mode'],'mapped')
        self.step(replay,11.8)
        action=self.step(replay,12.1,associated=False)
        self.assertEqual(action[1]['mode'],'local')
        self.assertEqual(replay.anchors['door'].stamp_s,11.8)
        self.assertEqual(action[1]['route_anchor_observation'],replay.anchors['door'].observation_binding)
        json.dumps(replay.drain_events(),allow_nan=False)

    def test_missing_anchor_does_not_create_a_local_goal(self):
        replay=self.replay()
        self.assertIsNone(self.step(replay,12.1,associated=False))
        self.assertIsNone(replay.current);self.assertFalse(replay.anchors)

    def test_map_loss_cancels_and_recovery_uses_a_new_revision(self):
        replay=self.replay();old=self.step(replay,10.)[1]
        cancel=self.step(replay,10.3,associated=False)
        self.assertEqual(cancel,('cancel',dict(task_id=old['task_id'],goal_revision=old['goal_revision'])))
        new=self.step(replay,10.6)[1]
        self.assertGreater(new['goal_revision'],old['goal_revision'])

    def test_short_age_outage_preserves_goal_but_input_is_still_rejected(self):
        from navigation.planning.inputs import scene_status
        replay=self.replay();old=self.step(replay,10.)[1]
        scene=scene_at(10.1)
        state=dict(stamp_s=10.7,health=0,session_id='test',generation=0,frame_id='odom')
        self.assertEqual(scene_status(scene,10.7,'synthetic',load_config()),'geometry_future_or_stale')
        self.assertIsNone(replay.step(scene,state,now=10.7,clock_domain='synthetic'))
        self.assertEqual(replay.current,old);self.assertEqual(replay.cancels,0)
        self.assertIsNone(self.step(replay,10.8));self.assertEqual(replay.current,old)
        # A real external cancellation must still revoke the route.
        replay.disable('manual');self.assertEqual(self.step(replay,10.9)[0],'cancel')

    def test_floor_end_waits_and_localization_reset_stays_disabled(self):
        replay=self.replay();self.step(replay,10.)
        self.assertEqual(self.step(replay,21.)[0],'cancel')
        self.assertEqual(replay.summary()['status'][1],'waiting_for_external_confirmation')
        self.assertFalse(replay.summary()['task_completed'])
        replay=self.replay();self.step(replay,10.)
        self.assertEqual(self.step(replay,10.5,generation=1)[0],'cancel')
        self.assertIsNone(self.step(replay,11.,generation=1))
        self.assertEqual(replay.disabled,'localization_reset_requires_route_restart')

    def test_goal_hysteresis_does_not_restart_search_on_each_scan(self):
        self.record['segments'][0].update(points_xy=[[2,2],[12,2]],source=dict(start_offset_s=9,end_offset_s=30))
        self.record['segments'][1]['source']=dict(start_offset_s=30,end_offset_s=40)
        replay=self.replay();old=self.step(replay,10.)[1]
        self.assertIsNone(self.step(replay,10.3,x=2.2))
        self.assertIsNone(self.step(replay,11.5,x=3.1))
        new=self.step(replay,12.1,x=3.3)[1]
        self.assertGreater(new['goal_revision'],old['goal_revision'])
        self.assertTrue(new['request_id'].endswith(':'+str(new['goal_revision'])))

    def test_source_identity_and_overlapping_windows_are_rejected(self):
        self.record['source']['bag_fingerprint']['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'same bag'):self.replay()
        self.record['source']['bag_fingerprint']=file_fingerprint(self.bag)
        self.record['segments'][1]['source']['start_offset_s']=11
        with self.assertRaisesRegex(ValueError,'nonoverlapping'):self.replay()

    def test_manual_preemption_cannot_resume_from_another_observation(self):
        replay=self.replay();self.step(replay,10.);replay.disable('manual')
        self.assertEqual(self.step(replay,10.2)[0],'cancel')
        self.assertIsNone(self.step(replay,11.))
        self.assertEqual(replay.disabled,'manual')
