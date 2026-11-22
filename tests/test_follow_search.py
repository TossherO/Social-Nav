"""Bounded search candidates never treat unknown geometry as free."""
import unittest
import math
import numpy as np

from decision.osnet.reid import ReidPolicy
from decision.osnet.search import FollowSearch
from navigation.planning.footprint import VehicleGeometry
from navigation.search import candidates,check_pose,bounds_status,current_search_pose,follow_candidates
from perception.environment.context import fingerprint
from tests.navigation_fixtures import asset,bundle,scene_at


class FollowSearchGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.inputs=dict(global_grid=asset().snapshot,scene=bundle(),pose=[2.,2.,0.],local_pose=[2.,2.,0.],
                         stamp_s=10.,clock_domain='synthetic',geometry=self.geometry)
        self.last=dict(identity=['tracker',1,'old'],stamp_s=9.9,clock_domain='synthetic',
                       local_identity=['test',0,'odom'],position_xy_m=[4.,2.],velocity_xy_mps=[.2,0.])
        self.last['binding']=fingerprint(self.last)

    def test_candidates_include_only_short_checked_poses(self):
        result=candidates(self.inputs,self.last,now=10.,clock_domain='synthetic',
                          local_identity=['test',0,'odom'],anchor=[2.,2.])
        self.assertEqual(result[0]['kind'],'wait')
        self.assertEqual({x['kind'] for x in result},{'wait','rotate','approach'})
        self.assertTrue(all(x.get('check',{}).get('valid',True) for x in result))
        for item in result[1:]:self.assertLessEqual(np.linalg.norm(np.asarray(item['pose'][:2])-[2.,2.]),.8)

    def test_unknown_local_cells_block_approach_and_stale_geometry_is_refused(self):
        cells=np.zeros((60,60),np.int8);cells[20,25]=-1
        self.inputs['scene']=bundle(cells)
        result=candidates(self.inputs,self.last,now=10.,clock_domain='synthetic',
                          local_identity=['test',0,'odom'],anchor=[2.,2.])
        self.assertNotIn('approach',[x['kind'] for x in result])
        with self.assertRaisesRegex(ValueError,'unavailable'):
            candidates(self.inputs,self.last,now=10.6,clock_domain='synthetic',
                       local_identity=['test',0,'odom'],anchor=[2.,2.])

    def test_foreign_history_cannot_suggest_approach(self):
        result=candidates(self.inputs,self.last,now=10.,clock_domain='synthetic',
                          local_identity=['other',0,'odom'],anchor=[2.,2.])
        self.assertNotIn('approach',[x['kind'] for x in result])
        with self.assertRaisesRegex(ValueError,'budget'):
            check_pose(self.inputs,[4.,2.,0.],now=10.,clock_domain='synthetic',anchor=[2.,2.])

    def test_search_is_rebased_and_passed_approach_is_rejected(self):
        approach=dict(kind='approach',pose=[2.8,2.,0.],last_observed_pose=[4.,2.,0.])
        np.testing.assert_allclose(current_search_pose(approach,[2.4,2.,0.]),[3.2,2.,0.])
        with self.assertRaisesRegex(ValueError,'search_approach_already_passed'):
            current_search_pose(approach,[3.,2.,0.])
        self.assertEqual(current_search_pose(dict(kind='rotate',pose=[2.,2.,.5]),[3.,2.,0.]),[3.,2.,.5])
        behind=dict(self.last,position_xy_m=[1.,2.])
        behind['binding']=fingerprint({k:v for k,v in behind.items() if k!='binding'})
        options=candidates(self.inputs,behind,now=10.,clock_domain='synthetic',
                           local_identity=['test',0,'odom'],anchor=[2.,2.])
        self.assertNotIn('approach',[option['kind'] for option in options])

    def test_t2_always_gets_wait_and_map_bound_choices_when_available(self):
        unknown=follow_candidates(self.inputs['scene'],None,self.geometry,self.last)
        self.assertEqual(unknown['candidates'],[dict(id=0,kind='wait',pose=None)])
        scene=self.inputs['scene'];static=asset()
        result=follow_candidates(scene,static,self.geometry,self.last)
        self.assertEqual(result['scene_binding'],scene.record['input_fingerprint'])
        self.assertEqual(result['candidates'][0]['kind'],'wait')
        self.assertEqual(result['binding'],fingerprint({k:v for k,v in result.items() if k!='binding'}))

    def test_independent_search_radius_checks_current_mapped_motion(self):
        from navigation.planning.inputs import load_config
        from navigation.planning.continuity import goal_binding
        static=asset();request=dict(task_id='search',goal_revision=1,mode='mapped',frame_id='map',
            target_reference='base_link',goal=[2.5,2.,0.],map_snapshot_id='static-a',
            search_bounds=dict(anchor_xy_m=[2.,2.],radius_m=1.5,map_snapshot_id='static-a',loss_binding='loss'))
        self.assertIsNone(bounds_status(request,bundle(),static,load_config(),[[2.,2.,0.],[3.4,2.,0.]],margin=.05,geometry=self.geometry))
        self.assertEqual(bounds_status(request,bundle(),static,load_config(),[[2.,2.,0.],[3.48,2.,0.]],margin=.05,geometry=self.geometry),
                         'follow_search_radius_exceeded')
        modified=dict(request,search_bounds=dict(request['search_bounds'],radius_m=1.))
        self.assertNotEqual(goal_binding(request),goal_binding(modified))
        self.assertEqual(bounds_status(request,bundle(associated=False),static,load_config(),[[2.,2.,0.]],margin=0.,geometry=self.geometry),
                         'follow_search_map_association_invalid')

    def test_safety_rejects_motion_outside_search_radius(self):
        from navigation.planning.motion import load_motion_config
        from navigation.safety.guard import LocalSafety
        static=asset();scene=bundle();motion=load_motion_config();motion['collision_backend']='cpu'
        request=dict(task_id='search',request_id='r',goal_revision=1,mode='mapped',frame_id='map',
            target_reference='base_link',goal=[2.5,2.,0.],map_snapshot_id='static-a',
            search_bounds=dict(anchor_xy_m=[2.,2.],radius_m=.1,map_snapshot_id='static-a',loss_binding='loss'))
        feedback=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',
                      local_identity=['test',0,'odom'],pose=[2.,2.,0.],velocity=[0.,0.])
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.36,0.],request_binding=fingerprint(request),
                      scene_binding=scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic')
        gate=LocalSafety(self.geometry,motion,.05,simulation=True,asset=static)
        result=gate.gate(proposal,request,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['reason'],'follow_search_radius_exceeded')
        self.assertEqual(result['velocity'],[0.,0.])


class LocalFollowLossTests(unittest.TestCase):
    identity = ['tracker', 1, 'a']
    lifecycle = 7

    def setUp(self):
        self.selector = FollowSearch()
        self.last = None

    def observe(self, stamp, *, reason='no_detections', box=None, position=None, score=None,
                association=None, observed=False, confirmed=False, recovered=False):
        scene = scene_at(stamp)
        scene_binding = scene.record['input_fingerprint']
        stamp_ns = round((stamp - .02) * 1e9)
        candidates = [] if box is None else [dict(id=0, box=box, track=dict(support_stamp_s=stamp))]
        source = dict(clock_domain='synthetic', local_identity=['test', 0, 'odom'], stamp_s=stamp,
                      rgb_stamp_ns=stamp_ns, image_size=[256, 192], scene_binding=scene_binding,
                      association_status='complete', candidates=candidates)
        decision = dict(status='unconfirmed', reason=reason, score=score)
        if confirmed:
            decision.update(status='recovered' if recovered else 'same', reason='confirmed',
                            candidate_id=0, identity=self.identity, lifecycle_generation=self.lifecycle)
        loss = self.selector.observe(source, decision, last=self.last, positions={0: position}, pose_xy=[0., 0.],
            observed={(tuple(self.identity), self.lifecycle)} if observed else set(),
            association_tracks={} if association is None else {'a/g7': association}, policy=ReidPolicy())
        if confirmed:
            self.last = dict(clock_domain='synthetic', local_identity=['test', 0, 'odom'], stamp_s=stamp,
                             scene_binding=scene_binding, identity=self.identity, position_xy_m=position,
                             lifecycle_generation=self.lifecycle)
            self.last['binding'] = fingerprint(self.last)
        return loss

    def edge_history(self):
        self.observe(10., confirmed=True, recovered=True, box=[115, 50, 155, 140], position=[1., 0.])
        self.observe(10.3, confirmed=True, box=[195, 50, 235, 140], position=[1., .4])
        self.observe(10.6, confirmed=True, box=[223, 50, 256, 140], position=[1., .8])

    def choose(self, stamp, *, pose=None, renewal=False, different_scene=False):
        scene = scene_at(stamp)
        options = dict(clock_domain='synthetic', source_stamp_s=stamp,
                       scene_binding=scene.record['input_fingerprint'], candidates=[dict(id=0, kind='wait', pose=None)])
        if pose is not None:
            candidate = dict(id=1, kind='approach', pose=pose, check=dict(valid=True,
                             clock_domain='synthetic', source_stamp_s=stamp))
            if renewal:
                candidate['continuation_id'] = self.selector.continuation['intent_id']
            options['candidates'].append(candidate)
        options['binding'] = fingerprint(options)
        return self.selector.choose(options, last=self.last, scene=scene_at(stamp + .1) if different_scene else scene)

    def test_edge_departure_waits_then_approaches_once_and_renews_same_target(self):
        self.edge_history()
        self.assertFalse(self.observe(10.8)['search_ready'])
        self.assertFalse(self.observe(11.3)['search_ready'])
        loss = self.observe(11.8)
        self.assertEqual(loss['reason'], 'left_image_edge')
        self.assertTrue(loss['search_ready'])
        self.assertIsNone(loss['departure_yaw_rad'])
        with self.assertRaisesRegex(ValueError, 'observation_mismatch'):
            self.choose(11.8, different_scene=True)
        self.assertEqual(self.choose(11.8)['kind'], 'wait')
        self.assertFalse(self.selector.attempted)
        point = self.last['position_xy_m']
        self.assertEqual(self.choose(11.8, pose=point)['pose'], point)
        original = self.selector.continuation.copy()
        self.assertTrue(self.observe(12.)['search_ready'])
        refreshed = self.choose(12., pose=point, renewal=True)
        self.assertEqual(refreshed['continuation_id'], original['intent_id'])
        self.assertEqual(self.selector.continuation, original)
        self.assertEqual(self.choose(12.)['kind'], 'wait')
        self.assertIsNone(self.selector.continuation)
        for stamp in (12.2, 12.7, 13.2, 13.7):
            self.assertFalse(self.observe(stamp)['search_ready'])

    def test_occlusion_missing_frames_and_an_unconfirmed_reappearance_interrupt_loss(self):
        self.edge_history()
        self.observe(10.8)
        self.assertEqual(self.observe(11., reason='below_similarity', score=.2, box=[220, 40, 256, 150])['state'],
                         'occluded')
        self.assertEqual(self.observe(11.3)['loss_count'], 1)
        self.assertEqual(self.observe(11.8)['loss_count'], 2)
        self.assertEqual(self.observe(12.4)['reason'], 'loss_onset_unobserved')
        self.assertEqual(self.observe(12.6, reason='awaiting_confirmation')['state'], 'recovering')
        self.assertEqual(self.observe(12.7)['loss_count'], 0)
        self.assertEqual(self.observe(12.7, reason='source_not_new')['loss_count'], 0)
        self.assertEqual(self.observe(13., reason='awaiting_confirmation')['state'], 'recovering')
        self.assertFalse(self.observe(13.3)['search_ready'])
        self.observe(13.5, confirmed=True, recovered=True, box=[225, 50, 256, 140], position=[1., .85])
        self.assertEqual(self.observe(13.6)['state'], 'unknown')

    def test_corner_direction_requires_two_actual_outbound_displacements(self):
        for stamp, x_box, position in [(10., 110, [1., 0.]), (10.2, 114, [1.3, 0.]),
                                       (10.4, 145, [1.3, .2]), (10.6, 170, [1.3, .4])]:
            self.observe(stamp, confirmed=True, recovered=stamp == 10., box=[x_box, 50, x_box + 40, 140],
                         position=position)
        loss = self.observe(10.7)
        self.assertEqual(loss['reason'], 'turned_toward_image_side')
        self.assertAlmostEqual(loss['departure_yaw_rad'], math.pi / 2)
        evidence = loss['departure_evidence']
        self.assertEqual([item['support_stamp_s'] for item in evidence], [10., 10.2, 10.4, 10.6])
        self.assertEqual(evidence[-1]['scene_binding'], self.last['scene_binding'])
        self.assertEqual(evidence[-1]['position_xy_m'], self.last['position_xy_m'])

    def test_corner_with_only_one_outbound_displacement_targets_position_only(self):
        self.observe(10., confirmed=True, recovered=True, box=[100, 50, 140, 140], position=[1., 0.])
        self.observe(10.2, confirmed=True, box=[105, 50, 145, 140], position=[1.3, 0.])
        self.observe(10.4, confirmed=True, box=[166, 50, 206, 140], position=[1.3, .35])
        loss = self.observe(10.7)
        self.assertEqual(loss['reason'], 'turned_toward_image_side')
        self.assertIsNone(loss['departure_yaw_rad'])
        self.assertIsNone(loss['departure_evidence'])

    def test_receding_track_can_be_lost_to_small_fragments_or_empty_image(self):
        for stamp, height, position in [(10., 120, [1., 0.]), (10.2, 75, [1.22, 0.]), (10.4, 42, [1.42, 0.])]:
            self.observe(stamp, confirmed=True, recovered=stamp == 10., box=[105, 48, 145, 48 + height],
                         position=position)
        small = self.observe(10.8, reason='appearance_low_quality', box=[115, 55, 128, 82])
        self.assertEqual(small['reason'], 'receding_in_view')
        self.assertEqual(small['loss_count'], 1)
        self.assertFalse(self.observe(11.3)['search_ready'])
        self.assertTrue(self.observe(11.8)['search_ready'])

    def test_single_range_or_size_change_and_low_score_alone_do_not_prove_loss(self):
        for stamp, height, position in [(10., 120, [1., 0.]), (10.2, 120, [1., 0.]), (10.4, 42, [1.5, 0.])]:
            self.observe(stamp, confirmed=True, recovered=stamp == 10., box=[105, 48, 145, 48 + height],
                         position=position)
        self.assertEqual(self.observe(10.8, reason='appearance_low_quality', box=[115, 55, 128, 82])['state'], 'occluded')
        self.assertEqual(self.observe(11., observed=True)['state'], 'holding')
        self.assertEqual(self.observe(11.3, reason='below_similarity', score=.1, box=[10, 35, 50, 110])['state'], 'unknown')
        for stamp in (11.5, 12., 12.5, 13.):
            self.assertEqual(self.observe(stamp)['state'], 'unknown')


if __name__=='__main__':unittest.main()
