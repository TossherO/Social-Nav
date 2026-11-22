"""Causal planning blockage, route progress and recovery stop requirements."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from decision.agent import CONFIG_DIR
from decision.agent.adapters.manual import ManualNavigation
from decision.agent.tools.navigation import NavigationSkill
from decision.agent.tools.places import Catalog
from decision.agent.workflows.manual_reference import reference_request
from navigation.planning.continuity import goal_binding, reference_envelope
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.planner import Planner
from navigation.planning.reference import ReferenceRuntime
from runtime.command import AgentState
from runtime.control import control_planning_observation
from tests.navigation_fixtures import FakeWorker, bundle, request, scene_at


class BlockageFeedbackTests(unittest.TestCase):
    def test_reference_distinguishes_an_occupied_destination_from_a_waypoint(self):
        cells = np.zeros((60, 60), np.int8)
        cells[40, 40] = 100
        scene = bundle(cells)
        geometry = VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]], np.eye(4), source='test')
        for behavior, expected in (('stop','destination_occupied'), ('pass_through','route_blocked')):
            with self.subTest(behavior=behavior):
                with ReferenceRuntime(load_config(),load_motion_config(),geometry,worker_factory=FakeWorker) as runtime:
                    goal = request('local',goal_behavior=behavior)
                    runtime.update(goal,scene,None,now=10.)
                    runtime.tick(now=10.,clock_domain='synthetic')
                    result = Planner(runtime.planning,geometry).query(goal,scene=scene,now=10.,clock_domain='synthetic')
                    self.assertEqual(result['reason'],'goal_occupied')
                    runtime.worker.reply(reference_envelope(result,goal,scene,None))
                    outcome = runtime.tick(now=10.,clock_domain='synthetic')['planning_observation']
                    self.assertEqual(outcome['kind'],expected)
                    self.assertEqual(outcome['blocked_position'],[4.,4.])
                    self.assertEqual(outcome['scene_binding'],scene.record['input_fingerprint'])
                    self.assertEqual(outcome['source_stamp_s'],10.)
                    runtime.update(dict(goal,request_id='replacement'),scene,None,now=10.)
                    self.assertIsNone(runtime.status(now=10.,clock_domain='synthetic')['planning_observation'])

    def test_local_control_preserves_geometry_stamp_and_inconclusive_status(self):
        scene = scene_at(10.)
        intent = request('local',goal_behavior='stop')
        packet = dict(request=intent,stamp_s=10.2,clock_domain='synthetic',feedback=dict(status='valid'),
                      scene_binding=scene.record['input_fingerprint'])
        for reason, status in (('no_admissible_velocity','blocked'), ('certification_time_budget_exceeded','unknown')):
            packet['proposal']=dict(status='unavailable',reason=reason,unattempted_candidates=0)
            observation=control_planning_observation(packet,scene,.5)
            self.assertEqual(observation['status'],status)
            self.assertEqual(observation['source_stamp_s'],10.)
            state=AgentState()
            state.publish_control(packet,control_observation=observation)
            self.assertEqual(state.snapshot()['control']['control_observation'],observation)

    def setUp(self):
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value)
        value.update(scope='development_prior',map_snapshot_id='static',localization_map_id='map-id')
        for place in value['places']:
            place['status']='manual_annotation'
        self.catalog=Catalog(value)
        self.binding=dict(valid=True,map_id='map-id',map_frame='sim_odom')
        self.feedback=dict(status='valid',source='causal_algorithm_poses_not_chassis_feedback',
            local_identity=['s',0,'odom'],stamp_s=10.,pose=[4.,4.,0.],velocity=[0.,0.])
        self.control=dict(clock_domain='runtime_live',stamp_s=10.,request=None,feedback={},
            idle_feedback=self.feedback,localization=dict(session_id='s',generation=0,frame_id='odom',health=0),
            map_binding=self.binding)
        self.state=dict(control=self.control,received_age_s=.01,vehicle_context=None,observation=dict(
            local_identity=['s',0,'odom'],clock_domain='runtime_live',map_binding=self.binding,
            map_base_pose=[4.,4.,0.],stamp_s=10.))
        client=SimpleNamespace(require_observation_only=lambda:None,snapshot=lambda **_:self.state)
        self.nav=ManualNavigation(self.catalog,client)
        self.events=[]
        self.skill=NavigationSkill(self.nav,self.catalog,'task',lambda name,_:self.events.append(name))

    def start(self, route=None):
        place=dict(goal_id='goal',place_id='elevator',frame_id='sim_odom',pose=[6.,4.],
            view_id='view',task_goal_binding='binding',requirement=dict(kind='point',target_id='elevator'))
        if route:
            place['route']=route
        self.assertEqual(self.skill.start(place,1)['status'],'accepted')

    def sample(self, now, source_stamp, *, status='blocked', kind='destination_occupied', request_override=None,
               preview_reason=None):
        handle=self.nav.handle
        expected=reference_request(handle,self.nav.request['pose'],self.nav.route,self.nav.route_index,self.catalog)
        identity=request_override or expected
        observation=dict(source='reference',status=status,kind=kind,source_stamp_s=source_stamp,
            valid_until_s=source_stamp+5.,reason='goal_occupied' if status=='blocked' else None,
            goal_binding=goal_binding(identity), clock_domain='runtime_live',local_identity=['s',0,'odom'],
            map_binding=self.binding,scene_binding=str(source_stamp), **{k:identity[k] for k in
            ('request_id','task_id','goal_revision')})
        self.control.update(stamp_s=now,manual_preview=dict(owner=handle['call_id'],request_id=identity['request_id'],
            goal_revision=identity['goal_revision'],status='unavailable',reason=preview_reason,
            planning_observation=observation if preview_reason is None else None))
        self.feedback['stamp_s']=self.state['observation']['stamp_s']=now
        return self.skill.tick()

    def test_same_route_blockage_across_background_replanning(self):
        self.start(dict(waypoints=[[5.,4.],[6.,4.]],final_yaw=None))
        for stamp, reason in ((10.,'waiting_for_background_reference'),(11.1,'reference_observation_changed'),
                              (12.2,'waiting_for_background_reference')):
            self.assertFalse(self.sample(stamp+.1,stamp,kind='route_blocked'))
            self.assertFalse(self.sample(stamp+.2,stamp,preview_reason=reason))
            self.assertFalse(self.sample(stamp+.3,stamp,kind='route_blocked'))  # Same scan republished.
        self.assertFalse(self.sample(13.1,12.2,preview_reason='waiting_for_background_reference'))
        self.assertTrue(self.sample(13.3,13.2,kind='route_blocked'))
        proof=self.skill.record['blockage']
        self.assertEqual((proof['first_source_stamp_s'],proof['source_stamp_s'],proof['observation_count']),
                         (10.,13.2,4))
        self.assertFalse(self.sample(13.4,13.2,preview_reason='waiting_for_background_reference'))
        self.assertFalse(proof['current'])
        self.assertFalse(self.sample(14.4,14.3,kind='route_blocked'))
        self.assertEqual(self.events.count('navigation_blocked'),1)

    def test_pending_reference_expires_or_is_interrupted_by_other_feedback(self):
        self.start(dict(waypoints=[[5.,4.],[6.,4.]],final_yaw=None))
        def blocked(stamp):
            return self.sample(stamp,stamp,kind='route_blocked')

        self.assertFalse(blocked(10.1))
        self.assertFalse(self.sample(15.2,15.2,preview_reason='waiting_for_background_reference'))
        self.assertFalse(blocked(15.3))  # Previous scan expired while waiting.
        self.assertFalse(self.sample(17.,17.,status='clear'))
        self.assertFalse(blocked(18.4))
        self.assertFalse(self.sample(20.,20.,preview_reason='observation_unavailable'))
        self.assertFalse(blocked(21.5))
        wrong=reference_request(self.nav.handle,[6.,4.],self.nav.route,1,self.catalog)
        self.assertFalse(self.sample(24.,24.,preview_reason='waiting_for_background_reference',request_override=wrong))
        self.assertFalse(blocked(24.6))
        self.assertTrue(blocked(27.7))
        self.assertEqual(self.skill.record['blockage']['first_source_stamp_s'],24.6)

    def test_duplicate_source_and_unknown_never_trigger_then_new_failure_recovers(self):
        self.start()
        self.assertFalse(self.sample(10.1,10.))
        self.assertFalse(self.sample(13.2,10.))
        self.assertFalse(self.sample(13.3,13.3,status='unknown'))
        self.assertFalse(self.sample(13.4,13.4))
        self.assertTrue(self.sample(16.5,16.5))
        blockage=self.skill.record['blockage']
        self.assertEqual(blockage['first_source_stamp_s'],13.4)
        self.assertEqual(blockage['kind'],'destination_occupied')
        self.assertFalse(self.sample(16.6,16.5))
        self.assertEqual(self.events.count('navigation_blocked'),1)
        self.assertFalse(self.sample(16.7,16.7,status='unknown'))
        self.assertFalse(self.skill.record['blockage']['current'])
        self.assertTrue(self.sample(16.8,16.8,status='clear'))
        self.assertEqual(self.skill.record['status'],'running')
        self.assertIsNone(self.skill.record['blockage'])

    def test_changed_route_segment_drops_an_old_planner_report(self):
        self.start(dict(waypoints=[[5.,4.],[6.,4.]],final_yaw=None))
        old=reference_request(self.nav.handle,[6.,4.],self.nav.route,0,self.catalog)
        self.assertFalse(self.sample(10.1,10.1,kind='route_blocked'))
        self.assertTrue(self.sample(13.2,13.2,kind='route_blocked'))
        self.assertEqual(self.skill.record['blockage']['route_index'],0)
        self.nav.route_index=1
        self.assertTrue(self.sample(13.3,13.3,kind='route_blocked',request_override=old))
        self.assertIsNone(self.nav.snapshot()['planning_observation'])
        self.assertIsNone(self.skill.record['blockage'])

    def test_manual_detour_checks_matching_reference_path_even_with_free_waypoints(self):
        self.start(dict(waypoints=[[5.,4.],[6.,4.]],final_yaw=None))
        self.nav.detour_check=dict(position=[4.,5.],frame_id='sim_odom',tolerance_m=.2)
        self.sample(10.1,10.1,status='clear',kind=None)
        preview=self.control['manual_preview']
        preview['planning_observation']['result_binding']='new-path'
        preview.update(reference_frame='sim_odom',path_result_binding='old-path',
                       reference_path=[[4.,4.],[4.,5.],[5.,4.]])
        self.assertEqual(self.nav.snapshot()['planning_observation']['status'],'clear')
        preview['path_result_binding']='new-path'
        self.assertEqual(self.nav.snapshot()['planning_observation']['reason'],'detour_preview_crosses_prior_blockage')
        self.assertEqual(preview['planning_observation']['status'],'clear')
        preview.update(path_result_binding='different-path',reference_path=[[4.,4.],[5.,4.]])
        preview['planning_observation']['result_binding']='different-path'
        self.assertEqual(self.nav.snapshot()['planning_observation']['status'],'clear')

    def test_gaps_or_inconclusive_results_break_sustained_evidence(self):
        self.start()
        self.assertFalse(self.sample(10.1,10.1))
        self.assertFalse(self.sample(16.2,16.2))
        self.assertTrue(self.sample(19.3,19.3))
        self.assertFalse(self.sample(19.4,19.3,status='unknown'))
        self.assertFalse(self.sample(19.5,19.3))
        self.assertFalse(self.sample(19.6,19.6))
        self.assertFalse(self.skill.record['blockage']['current'])
        self.assertFalse(self.sample(22.7,22.7))
        self.assertTrue(self.skill.record['blockage']['current'])
        self.assertEqual(self.events.count('navigation_blocked'),1)
        self.assertEqual(self.skill.record['blockage']['first_source_stamp_s'],19.6)
        self.assertFalse(self.sample(30.,30.))
        self.assertFalse(self.skill.record['blockage']['current'])

    def test_recovery_cancel_requires_matching_fresh_stop(self):
        self.start()
        handle=self.nav.handle
        self.skill.cancel()
        self.skill.record['recovery_cancel_stamp_s']=10.
        self.control['stamp_s']=self.state['observation']['stamp_s']=10.2
        self.assertFalse(self.skill.tick())  # Stop estimate predates the cancellation sample.
        self.control['stamp_s']=self.state['observation']['stamp_s']=10.3
        self.feedback.update(stamp_s=10.3,velocity=[.2,0.])
        self.assertFalse(self.skill.tick())
        self.control['stamp_s']=self.state['observation']['stamp_s']=10.4
        self.feedback.update(stamp_s=10.4,velocity=[0.,0.])
        self.assertTrue(self.skill.tick())
        self.assertEqual(self.skill.record['status'],'cancelled')
        self.assertEqual(self.skill.record['recovery_stop']['handle'],handle)
        self.assertEqual(self.skill.record['recovery_stop']['stop_measurement_stamp_s'],10.4)


if __name__=='__main__':
    unittest.main()
