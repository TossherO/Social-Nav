"""Continuous seating updates retain task ownership and stop on invalid evidence."""
import copy
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from decision.agent.core.seating import (
    footprint_approved, observation_history, poll_observation_arrival, poll_selection, poll_updates,
)
from runtime.command import CommandInbox


class SeatingUpdateTests(unittest.TestCase):
    def test_round_selection_exposes_clear_unknown_and_blocked_footprints(self):
        import numpy as np
        from unittest.mock import patch
        from runtime.seating import SeatingWorker

        binding = (['s', 0, 'odom'], 'runtime_live', dict(map_id='map'))
        scene = SimpleNamespace(record=dict(as_of=10., frame_id='odom', clock_domain='runtime_live',
            observation_id='scan', session_id='s', generation=0, prior=dict(map_from_local=np.eye(4))))
        frame = (scene, SimpleNamespace(record={'input_fingerprint':'rgb', 'rgb':{'stamp_s':10.}}),
                 None, None, None, None)
        worker = object.__new__(SeatingWorker)
        worker.lock = threading.Lock()
        worker.closed, worker.epoch, worker.sequence, worker.last_candidate = False, 1, 0, None
        worker.seating = dict(table_clearance_m=.1, manual_round_deadline_s=8., manual_round_candidate_count=3)
        worker.planning, worker.motion = dict(safety_margin_m=0., people={}), dict(min_turning_radius_m=.5, allow_reverse=True)
        worker.geometry, worker.asset = SimpleNamespace(footprint=[[.8,.3]], assumed=False), None
        worker._latest, worker._binding, worker._source_now = Mock(return_value=frame), Mock(return_value=binding), Mock()
        worker._map_from_local = Mock(return_value=np.eye(3))
        worker._view_search = Mock(return_value={})
        value = dict(selection_id='selection', instruction='停靠',
            round_table=dict(id='table', map_id='map', center_xy_m=[0.,0.], radius_m=.35),
            manual_observation=dict(after_s=0., visited_xy_m=[]))
        statuses = iter(('free','unknown','occupied'))
        with patch('navigation.planning.reference.planning_config', return_value={}), \
             patch('navigation.planning.planner.Planner'), \
             patch('perception.environment.navigation.navigation_geometry'), \
             patch('runtime.seating.round_parking_approach',
                   side_effect=lambda _, geometry, pose, margin, allowance: (pose, next(statuses), 0.)), \
             patch('runtime.seating.SeatingDecision') as decision:
            decision.return_value.infer.return_value = dict(status='unavailable', reason='test_input_captured')
            with self.assertRaisesRegex(ValueError, 'test_input_captured'):
                worker._choose(value, binding, frame, None, 1, {})
        candidates = decision.return_value.seating.call_args.args[1]['candidates']
        self.assertEqual([candidate['geometry'] for candidate in candidates], ['feasible','unknown','blocked'])

    def test_round_selection_routes_the_displayed_backoff_after_map_transform_changes(self):
        import numpy as np
        from unittest.mock import patch
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.planner import se2, transform_poses
        from perception.environment.layers import GridSnapshot
        from runtime.seating import SeatingWorker

        geometry = VehicleGeometry([[-.28,-.33],[.82,-.33],[.82,.33],[-.28,.33]],
                                   np.eye(4), source='test', assumed=False)
        lidar = np.eye(4)
        lidar[:3, 3] = [.2, -.3, 1.]
        prior = dict(valid=True, map_id='map', map_frame='map', local_session_id='s',
                     local_generation=0, local_frame='odom')
        grid = GridSnapshot(dict(origin_xy=[-5.,-5.],resolution=.1),
                            dict(occupancy=np.zeros((100,100), dtype=np.int8)))
        source_transform = np.eye(4)
        source_transform[:3, :3] = [[np.cos(.10), 0., np.sin(.10)], [0., 1., 0.],
                                    [-np.sin(.10), 0., np.cos(.10)]]
        source_transform[:3, 3] = [.4, -.1, .1]
        current_transform = np.eye(4)
        current_transform[:2, :2] = se2([0.,0.,.6])[:2,:2]
        current_transform[:3, 3] = [-.3, .2, .1]
        record = dict(frame_id='odom', clock_domain='runtime_live', observation_id='scan', session_id='s',
                      generation=0, localization=dict(world_from_lidar=lidar))
        source = SimpleNamespace(record=dict(record, as_of=10., prior=dict(prior,map_from_local=source_transform)),
                                 geometry=grid)
        current = SimpleNamespace(record=dict(record, as_of=10.3, prior=dict(prior,map_from_local=current_transform)),
                                  geometry=grid)
        observation = SimpleNamespace(record=dict(input_fingerprint='source-rgb',rgb={'stamp_s':9.95}))
        frame = (source, observation, None, None, None, None)
        worker = object.__new__(SeatingWorker)
        worker.lock, worker.frames = threading.Lock(), deque()
        worker.closed, worker.epoch, worker.sequence, worker.last_candidate = False, 1, 0, None
        worker.geometry, worker.asset = geometry, None
        worker.seating = dict(table_clearance_m=.1, manual_round_deadline_s=8.,
                              manual_round_candidate_count=1, manual_round_front_allowance_m=.2)
        worker.planning = dict(safety_margin_m=0., people={})
        worker.motion = dict(min_turning_radius_m=.5, allow_reverse=True)
        worker._latest = Mock(side_effect=AssertionError('First model call must use the selected observation'))
        worker._source_now = Mock(return_value=10.4)
        worker._view_search = Mock(return_value={})
        value = dict(selection_id='selection', observation_binding='request-rgb', instruction='停靠',
            round_table=dict(id='table', map_id='map', center_xy_m=[3.,1.], radius_m=.35),
            manual_observation=dict(after_s=8., visited_xy_m=[]))
        worker.active = (value,frame,None,1)

        def allow_backoff(_, geometry, pose, margin, allowance):
            yaw = pose[2]
            return [pose[0]-allowance*np.cos(yaw),pose[1]-allowance*np.sin(yaw),yaw], 'free', allowance

        with patch('navigation.planning.reference.planning_config', return_value={}), \
             patch('navigation.planning.planner.Planner') as planner, \
             patch('perception.environment.navigation.navigation_geometry',return_value=grid), \
             patch('runtime.seating.round_parking_approach',side_effect=allow_backoff) as approach, \
             patch('runtime.seating.SeatingDecision') as decision:
            planner.return_value.query.return_value = dict(status='path_found', current_path_valid=True)
            model = decision.return_value

            def infer(_):
                worker.update_scene(current, None)
                return dict(status='complete',timing={'first_output_text_s':.1,'stream_complete_s':.2})

            def accept(*args, **kwargs):
                candidate = model.seating.call_args.args[1]['candidates'][0]
                return dict(operation='plan',request=dict(goal=candidate['pose'],candidate_id=candidate['id'],
                                                         candidate_binding='test-candidate-binding'))

            model.infer.side_effect = infer
            model.accept.side_effect = accept
            worker._run(value,frame,None,1)

        source_se2 = se2(geometry.pose(source_transform @ lidar)) @ np.linalg.inv(se2(geometry.pose(lidar)))
        current_se2 = se2(geometry.pose(current_transform @ lidar)) @ np.linalg.inv(se2(geometry.pose(lidar)))
        drawn = model.seating.call_args.args[1]['candidates'][0]
        self.assertEqual(drawn['geometry'],'feasible')
        self.assertEqual(worker.result['status'],'selected')
        self.assertEqual(worker.result['observation_binding'],'request-rgb')
        self.assertEqual(worker.result['source_observation_binding'],'source-rgb')
        self.assertEqual(worker.result['attempts'][0]['timing'],{'first_output_text_s':.1,'stream_complete_s':.2})
        self.assertEqual(worker.result['geometry_stamp_s'],10.3)
        self.assertEqual(worker.result['front_backoff_m'],.2)
        self.assertAlmostEqual(worker.result['mapped_round_table']['target_gap_m'],.3)
        np.testing.assert_allclose(worker.result['pose'],[3.-.35-.82-.1-.2,1.,0.],atol=1e-12)
        np.testing.assert_allclose(transform_poses([drawn['pose']],source_se2)[0],worker.result['pose'],atol=1e-12)
        queried = planner.return_value.query.call_args
        self.assertIs(queried.kwargs['scene'],current)
        np.testing.assert_allclose(transform_poses([queried.args[0]['goal']],current_se2)[0],
                                   worker.result['pose'],atol=1e-12)
        approach.assert_called_once()

    def test_manual_round_initial_request_requires_its_rgb_binding(self):
        from runtime.seating import SeatingWorker

        worker = object.__new__(SeatingWorker)
        worker.lock = threading.Lock()
        worker.closed, worker.active, worker.thread = False, None, None
        observation = SimpleNamespace(record=dict(input_fingerprint='other-rgb',rgb={'stamp_s':10.}))
        scene = SimpleNamespace(record={'as_of':10.})
        worker.frames = deque([(scene,observation,None,None,None,None)])
        value = dict(selection_id='selection', observation_binding='missing-rgb', instruction='停靠',
            round_table=dict(id='table', map_id='map', center_xy_m=[3.,1.], radius_m=.35),
            manual_observation=dict(after_s=0., visited_xy_m=[]))
        with self.assertRaisesRegex(ValueError,'seating_rgb_observation_expired'):
            worker.select(value)
        value['manual_observation']['after_s'] = 10.
        self.assertEqual(worker.select(value)['error'],'seating_waiting_for_new_view')

    def test_rejected_observation_is_available_to_next_selection(self):
        import numpy as np
        from decision.agent.tools.navigation import spatial_context
        from runtime.seating import SeatingWorker

        binding = dict(map_frame='map')
        feedback = dict(stamp_s=10., clock_domain='runtime_live', session_id='s', generation=0,
                        frame_id='odom', map_binding=binding)
        pending = dict(goal_id='g1', revision=1, context=spatial_context(feedback),
                       selection_id='selection', observation_binding='frame', manual_observation={})
        result = dict(status='observation', pose=[1., 2., .3], selection_id='selection',
                      observation_binding='frame', local_identity=pending['context']['local_identity'],
                      map_binding=binding, clock_domain='runtime_live', geometry_stamp_s=9.9)
        goal = {}
        checked = dict(status='available', static_footprint={'status':'free'},
                       local_footprint={'status':'occupied'})
        session = SimpleNamespace(seating_pending=pending, seating_active=None,
            state=dict(revision=1), task_goals=SimpleNamespace(current=lambda _:goal),
            navigation=SimpleNamespace(scope='manual', snapshot=lambda:feedback,
                value={'seating':result}, client=SimpleNamespace(exchange=Mock())),
            catalog=SimpleNamespace(value={'frame_id':'map'}), spatial_view=None,
            prepare_current_images=Mock(), observation_stop=Mock(return_value=True),
            check_candidate=Mock(return_value=checked), skill=Mock(),
            conversation=SimpleNamespace(notice=Mock()), _save=Mock())

        poll_selection(session)

        session.skill.start.assert_not_called()
        rejected = observation_history(goal, 10.)['rejected_views']
        self.assertEqual(rejected, [
            dict(pose=[1., 2., .3], reason='seating_view_footprint_static_free_local_occupied')])
        self.assertEqual(session._save.call_args.args[1]['footprint_check']['local'], 'occupied')
        worker = object.__new__(SeatingWorker)
        worker.geometry = SimpleNamespace(pose=lambda _:np.zeros(3))
        worker._map_from_local = Mock(return_value=np.eye(3))
        scene = SimpleNamespace(record={'localization':{'world_from_lidar':np.eye(4)}})
        search = dict(bounds_xy_m=[0.,0.,3.,3.], visited_xy_m=[])
        _, reason, _ = worker._view_route(Mock(), {'goal':[1.1,2.,1.2]}, scene, None, {}, search, rejected)
        self.assertEqual(reason, 'seating_view_repeats_rejected_pose')

    def test_view_pose_stays_in_source_map_when_current_transform_changes(self):
        import numpy as np
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.planner import se2, transform_poses
        from runtime.seating import SeatingWorker

        worker = object.__new__(SeatingWorker)
        worker.geometry = VehicleGeometry([[-.2,-.2],[.6,-.2],[.6,.2],[-.2,.2]],
                                          np.eye(4),source='test')
        worker.asset, worker.planning = None, {'safety_margin_m':.05}
        worker._source_now = Mock(return_value=10.)
        worker._route = Mock(return_value=(None,False))
        source_se2, current_se2 = se2([1.,2.,.5]), se2([-2.,3.,-.7])
        current = SimpleNamespace(record={'localization':{'world_from_lidar':np.eye(4)}})
        source = object()
        worker._map_from_local = Mock(side_effect=lambda scene: current_se2 if scene is current else source_se2)
        table = dict(center_xy_m=[-4.,-4.],radius_m=.35)
        chosen = {'goal':[1.,2.,.2]}
        search = dict(bounds_xy_m=[0.,0.,3.,3.],visited_xy_m=[])

        pose,reason,revised = worker._view_route(Mock(),chosen,current,None,table,search,[],source_scene=source)

        self.assertIsNone(reason)
        self.assertFalse(revised)
        np.testing.assert_allclose(pose,transform_poses([chosen['goal']],source_se2)[0])
        routed = worker._route.call_args.args
        self.assertIs(routed[2],current)
        np.testing.assert_allclose(transform_poses([routed[1]['goal']],current_se2)[0],pose)

    def test_manual_round_schema_accepts_sixteen_headings_and_excludes_blocked(self):
        from math import pi
        from jsonschema import ValidationError
        from decision.vlm.project_config import load_config
        from decision.vlm.t3.geometry import round_candidates
        from decision.vlm.t3.inference import validate_selection

        table = dict(id='table',map_id='map',center_xy_m=[0.,0.],radius_m=.35)
        count = load_config()['tasks']['seating']['manual_round_candidate_count']
        poses = round_candidates(table,[[.8,.3]],.1,count=count)
        self.assertEqual([p['id'] for p in poses],list(range(16)))
        self.assertAlmostEqual(poses[1]['pose'][2]-poses[0]['pose'][2], pi/8)
        candidates = dict(instruction='停靠',observation_id='scan',stamp_s=1.,frame_id='odom',
                          clock_domain='runtime_live',session_id='s',generation=0,revision=0,
                          round_table=table,candidates=poses)
        validate_selection(dict(s='v',p=15),candidates)
        poses[15]['geometry'] = 'blocked'
        with self.assertRaises(ValidationError):
            validate_selection(dict(s='v',p=15),candidates)
        del candidates['round_table']
        with self.assertRaisesRegex(ValueError,'At most 8'):
            validate_selection(dict(s='v',p=7),candidates)

    def test_round_parking_front_overlap_moves_goal_back_but_checks_full_body(self):
        import numpy as np
        from decision.vlm.t3.geometry import round_parking_approach
        from navigation.planning.footprint import VehicleGeometry
        from perception.environment.layers import GridSnapshot

        geometry = VehicleGeometry([[-.28,-.33],[.82,-.33],[.82,.33],[-.28,.33]],
                                   np.eye(4),source='test')
        metadata = dict(resolution=.05,origin_xy=[0.,0.])
        cells = np.zeros((100,100),dtype=np.int8)
        pose = [2.,2.,0.]
        cells[40,55] = 100

        def evaluate():
            grid = GridSnapshot(metadata,dict(occupancy=cells))
            return round_parking_approach(grid,geometry,pose,.05,.20)

        self.assertEqual(evaluate(),([1.8,2.,0.],'free',.20))
        cells[40,31] = 100  # The backoff would move into this cell.
        self.assertEqual(evaluate(),(pose,'occupied',0.))
        cells[40,31] = 0
        cells[40,48] = 100  # Occupancy outside the front allowance.
        self.assertEqual(evaluate(),(pose,'occupied',0.))

    def test_observation_path_uses_static_asset_and_checks_beyond_first_unknown(self):
        import numpy as np
        from unittest.mock import patch
        from navigation.planning.footprint import VehicleGeometry
        from perception.environment.layers import GridSnapshot
        from runtime.seating import SeatingWorker

        worker = object.__new__(SeatingWorker)
        worker.geometry = VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],
                                          np.eye(4),source='test')
        worker.planning = dict(safety_margin_m=.05,max_path_samples=10000)
        worker.asset = SimpleNamespace(manifest={'snapshot_id':'static'})
        cells = np.zeros((30,40),dtype=np.int8)
        cells[9:12,15:18] = -1
        grid = GridSnapshot(dict(origin_xy=[-1.,-1.],resolution=.1),dict(occupancy=cells))
        scene = SimpleNamespace(geometry=grid,record=dict(prior=dict(map_frame='map',map_from_local=np.eye(4)),
                                 localization={'world_from_lidar':np.eye(4)},clock_domain='test'))
        planner = Mock()
        planner.query.return_value = dict(status='path_found',path=[[0.,0.,0.],[1.8,0.,0.]],
                                          reference_overlay={'status':'valid'},local_check={'status':'unknown'},
                                          current_path_valid=False,conditional_reference=False)
        request = dict(goal=[1.8,0.,0.],mode='local',frame_id='odom')
        with patch('perception.environment.navigation.navigation_geometry',return_value=grid):
            self.assertIsNone(worker._mapped_view_route(planner,request,scene,10.,[1.8,0.,0.]))
            self.assertIs(planner.query.call_args.kwargs['asset'],worker.asset)
            self.assertEqual(planner.query.call_args.args[0]['mode'],'mapped')
            cells[10,23] = 100
            self.assertEqual(worker._mapped_view_route(planner,request,scene,10.,[1.8,0.,0.]),
                             'seating_view_current_path_occupied')
        session = SimpleNamespace(navigation=SimpleNamespace(scope='manual'))
        checked = dict(status='available',static_footprint={'status':'free'},local_footprint={'status':'unknown'})
        verdict = dict(status='observation',mapped_reference_verified=True)
        self.assertTrue(footprint_approved(session,{'manual_observation':{}},verdict,checked))
        self.assertFalse(footprint_approved(session,{'manual_observation':{}},dict(verdict,status='selected'),checked))

    def test_round_parking_uses_local_approach_for_mapped_pose(self):
        import numpy as np
        from perception.environment.layers import GridSnapshot
        from navigation.planning.footprint import VehicleGeometry
        from runtime.seating import SeatingWorker
        from unittest.mock import patch

        geometry = VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.eye(4),source='test')
        cells = np.zeros((30,40),dtype=np.int8)
        meta = dict(origin_xy=[-1.,-1.],resolution=.1,snapshot_id='test-grid')
        live = GridSnapshot(meta,dict(occupancy=cells))
        prior_cells = cells.copy()
        prior_cells[10,16] = 100
        prior = GridSnapshot(meta,dict(occupancy=prior_cells))
        scene = SimpleNamespace(geometry=live, record=dict(prior={'map_from_local':np.eye(4),'map_frame':'map'},
                                 localization={'world_from_lidar':np.eye(4)},clock_domain='test'))
        planner = Mock()
        valid = dict(status='path_found',current_path_valid=True,reference_overlay={'status':'valid'})
        planner.query.return_value = valid
        worker = object.__new__(SeatingWorker)
        worker.asset=SimpleNamespace(snapshot=prior,manifest={'snapshot_id':'static'})
        worker.geometry=geometry
        worker.planning=dict(safety_margin_m=0.,max_path_samples=10000)
        table=dict(center_xy_m=[1.5,0.],radius_m=.35)
        goal = {'goal':[.65,0.,0.]}
        with patch('perception.environment.navigation.navigation_geometry', return_value=live):
            self.assertEqual(worker._route(planner,goal,scene,10.5,manual_table=table),(None,True))
            request = planner.query.call_args.args[0]
            self.assertEqual(request['mode'],'mapped')
            self.assertTrue(request['use_local_approach'])
            planner.query.return_value = dict(valid,current_path_valid=False,local_check={'status':'occupied'})
            self.assertEqual(worker._route(planner,goal,scene,10.5,manual_table=table),
                             ('seating_local_path_occupied',False))

    def test_manual_round_static_override_requires_local_free_and_bound_result(self):
        session = SimpleNamespace(navigation=SimpleNamespace(scope='manual'))
        pending = {'manual_observation': {}}
        result = {'conditional_prior_cleared': True}
        checked = dict(status='available', static_footprint={'status':'occupied'}, local_footprint={'status':'free'})
        self.assertTrue(footprint_approved(session,pending,result,checked))
        self.assertFalse(footprint_approved(session,pending,{},checked))
        session.navigation.scope = 'live'
        self.assertFalse(footprint_approved(session,pending,result,checked))
        session.navigation.scope = 'manual'
        checked['local_footprint']['status'] = 'unknown'
        self.assertFalse(footprint_approved(session,pending,result,checked))

    def test_manual_viewpoint_arrival_resumes_seating_with_post_stop_source_time(self):
        from decision.agent.tools.navigation import spatial_context

        handle = {'call_id':'view'}
        feedback = dict(stamp_s=10.,clock_domain='runtime_live',session_id='s',generation=0,
                        frame_id='odom',manual_arrival_observation=True)
        active = dict(handle=handle,revision=1,context=spatial_context(feedback),
                      instruction='桌边停靠',round_table={'id':'table'},observation_binding='first',
                      goal_id='g1',observation_pose=[1.,2.,.3])
        goal = dict(seating_views=[[1.,2.]],selected=None)
        exchange = Mock(side_effect=[{'accepted':False,'error':'seating_waiting_for_new_view'},
                                     {}])
        session = SimpleNamespace(skill=SimpleNamespace(record=dict(handle=None,completed=[{'handle':handle}],
                                  connected=True)),navigation=SimpleNamespace(snapshot=lambda:feedback,
                                  client=SimpleNamespace(exchange=exchange)),
                                  state=dict(status='waiting_navigation',revision=1),
                                  observation_stop=lambda _:True,seating_active=active,seating_pending=None,
                                  task_goals=SimpleNamespace(current=lambda _:goal),_save=Mock(),stop=Mock())
        poll_observation_arrival(session,active)
        self.assertIsNone(session.seating_pending)
        feedback['stamp_s'] = 10.2
        exchange.side_effect = lambda _,payload:dict(accepted=True,selection_id=payload['selection_id'])
        poll_observation_arrival(session,active)
        self.assertEqual(session.seating_pending['manual_observation'],
                         dict(after_s=10.,visited_xy_m=[[1.,2.]]))
        self.assertEqual(session.state['status'],'waiting_seating')
        self.assertIsNone(goal['selected'])
        self.assertIsNone(session.seating_active)

    def test_manual_viewpoint_blockage_preserves_goal_or_waits_for_fresh_stop(self):
        from decision.agent.tools.navigation import spatial_context

        for obstruction in ('start_footprint_occupied','returned_path_occupied'):
            with self.subTest(obstruction=obstruction):
                handle = {'call_id':'view'}
                feedback = dict(stamp_s=10.,clock_domain='runtime_live',session_id='s',generation=0,
                                frame_id='odom',manual_arrival_observation=False)
                active = dict(handle=handle,revision=1,context=spatial_context(feedback),
                              instruction='桌边停靠',round_table={'id':'table'},observation_binding='first',
                              goal_id='g1',observation_pose=[1.,2.,.3])
                goal = dict(seating_views=[[1.,2.]],selected=None)
                observation = Mock(return_value=dict(status='blocked',reason=obstruction))
                skill = SimpleNamespace(record=dict(handle=handle,status='running',connected=True,completed=[]),
                                        current_planning_observation=observation,
                                        blockage_matches_feedback=lambda *_:True,cancel=Mock())
                exchange = Mock(side_effect=lambda _,payload:dict(accepted=True,selection_id=payload['selection_id']))
                session = SimpleNamespace(skill=skill,navigation=SimpleNamespace(snapshot=lambda:feedback,
                    client=SimpleNamespace(exchange=exchange)),state=dict(status='waiting_navigation',revision=1),
                    observation_stop=Mock(return_value=False),seating_active=active,seating_pending=None,
                    task_goals=SimpleNamespace(current=lambda _:goal),_save=Mock(),stop=Mock())

                poll_observation_arrival(session,active)
                skill.cancel.assert_not_called()
                self.assertIs(session.seating_active,active)
                if obstruction=='start_footprint_occupied':
                    self.assertEqual(session.state['reason'],'seating_observation_reference_waiting')
                skill.record.update(status='blocked',blockage={'current':True})
                feedback['stamp_s'] = 13.
                poll_observation_arrival(session,active)
                if obstruction=='start_footprint_occupied':
                    self.assertEqual(session.state['reason'],'seating_observation_start_occupied')
                    observation.return_value = None
                    poll_observation_arrival(session,active)
                    self.assertEqual(session.state['reason'],'seating_observation_start_occupied')
                    skill.record.update(status='running',blockage=None)
                    observation.return_value = dict(status='clear',reason=None)
                    poll_observation_arrival(session,active)
                    self.assertIsNone(session.state['reason'])
                    self.assertEqual(skill.record['handle'],handle)
                    skill.cancel.assert_not_called()
                    exchange.assert_not_called()
                else:
                    skill.cancel.assert_called_once_with(reason='seating_observation_path_blocked')
                    self.assertEqual(skill.record['recovery_cancel_stamp_s'],13.)
                    poll_observation_arrival(session,active)
                    exchange.assert_not_called()
                    skill.record.update(status='cancelled',handle=None,recovery_stop={'handle':handle})
                    feedback['stamp_s'] = 14.
                    poll_observation_arrival(session,active)
                    exchange.assert_not_called()
                    session.observation_stop.return_value = True
                    feedback['stamp_s'] = 14.1
                    poll_observation_arrival(session,active)
                    self.assertEqual(session.seating_pending['manual_observation'],
                                     dict(after_s=14.1,visited_xy_m=[[1.,2.]]))
                    self.assertEqual(session.state['status'],'waiting_seating')
                    self.assertEqual(session._save.call_args.args[0],'seating_observation_reselected')
                session.stop.assert_not_called()

    def test_manual_retarget_updates_preview_and_clears_arrival_evidence(self):
        from decision.agent.adapters.manual import ManualNavigation
        nav=object.__new__(ManualNavigation)
        old=dict(call_id='call',goal_revision=1);new=dict(call_id='call',goal_revision=2)
        nav.handle=copy.deepcopy(old);nav.status='running';nav.follow_active=False
        nav.request=dict(pose=[1.,2.,0.],require_yaw=True)
        nav.confirmation={'old':True};nav.automatic_evidence={'old':True}
        nav.arrival_since=nav.arrival_last_stamp=5.
        nav.retarget(dict(pose=[2.,2.,1.]),old,new)
        self.assertEqual(nav.request['pose'],[2.,2.,1.])
        self.assertEqual(nav.handle,new)
        self.assertIsNone(nav.confirmation);self.assertIsNone(nav.automatic_evidence)
        with self.assertRaisesRegex(ValueError,'no longer active'):
            nav.retarget(dict(pose=[3.,2.,1.]),old,new)

    def test_live_retarget_requires_same_owner_and_fresh_armed_state(self):
        inbox=object.__new__(CommandInbox)
        inbox.agent_execution=True;inbox.lock=threading.Lock();inbox.pending=deque()
        inbox.agent_owner=dict(request_id='call',task_id='task',goal_revision=1,arm_revision=4)
        goal=dict(request_id='call',task_id='task',goal_revision=1,mode='mapped',frame_id='map',
                  target_reference='base_link',goal=[1.,2.,0.],map_snapshot_id='map-a')
        control=dict(request=copy.deepcopy(goal),map_binding=dict(valid=True))
        inbox.agent_state=SimpleNamespace(snapshot=lambda:dict(control=control,received_age_s=.1))
        inbox.session=SimpleNamespace(status=lambda:dict(chassis=dict(state='armed',status_fresh=True,arm_revision=4)),
                                      note_command=Mock())
        events=[];inbox.safety=SimpleNamespace(command=lambda *v:events.append(v))
        next_goal=dict(goal,goal_revision=2,goal=[1.2,2.,0.])
        with self.assertRaisesRegex(ValueError,'ownership_or_state'):
            inbox.agent_live_retarget(dict(previous_revision=1,goal=dict(next_goal,task_id='other')))
        self.assertEqual(events,[])
        self.assertTrue(inbox.agent_live_retarget(dict(previous_revision=1,goal=next_goal))['accepted'])
        self.assertEqual(events[-1],('goal',next_goal))
        with self.assertRaisesRegex(ValueError,'ownership_or_state'):
            inbox.agent_live_retarget(dict(previous_revision=1,goal=next_goal))
        inbox.agent_live_cancel_owned(dict(task_id='task',call_id='call'))
        self.assertEqual(events[-1],('cancel',dict(task_id='task',goal_revision=2)))

    def test_update_applies_new_pose_only_for_current_stage(self):
        context=dict(clock_domain='live',local_identity=['session',1,'odom'],map_binding=dict(valid=True,map_id='m',map_frame='map'))
        handle=dict(task_id='task',goal_id='seat',call_id='call',goal_revision=1)
        active=dict(selection_id='selection',goal_id='seat',revision=1,context=context,
                    observation_binding='original',sequence=1,handle=handle)
        result=dict(selection_id='selection',sequence=2,status='selected',observation_binding='original',
            local_identity=context['local_identity'],clock_domain='live',map_binding=context['map_binding'],
            geometry_stamp_s=10.,pose=[2.,1.,0.],mapped_edge={})
        feedback=dict(stamp_s=10.,clock_domain='live',session_id='session',generation=1,frame_id='odom',map_binding=context['map_binding'])
        goal=dict(binding='binding',requirement={},selected=dict(pose=[1.,1.,0.],handle=handle))
        skill=SimpleNamespace(record=dict(handle=handle,status='running'),
            retarget=Mock(return_value=dict(status='accepted',handle=dict(handle,goal_revision=2))))
        nav=SimpleNamespace(value={'seating':result},snapshot=lambda:feedback,client=SimpleNamespace(exchange=Mock()))
        session=SimpleNamespace(seating_active=active,seating_pending=None,state=dict(status='waiting_navigation',revision=1),
            skill=skill,navigation=nav,task_goals=SimpleNamespace(current=lambda _:goal),catalog=SimpleNamespace(value={'frame_id':'map'}),
            spatial_view=SimpleNamespace(record={'view_id':'view'}),prepare_current_images=lambda:None,
            check_candidate=lambda *_:dict(status='available',static_footprint=dict(status='free'),local_footprint=dict(status='free')),
            observation=None,_save=Mock(),stop=Mock())
        poll_updates(session)
        self.assertEqual(skill.retarget.call_count,1)
        self.assertEqual(goal['selected']['pose'],[2.,1.,0.])
        self.assertEqual(session.seating_active['handle']['goal_revision'],2)
        # A delayed response from an old task cannot replace the active goal.
        result['sequence']=3;result['local_identity']=['other',1,'odom']
        skill.record['handle']=session.seating_active['handle']
        poll_updates(session)
        self.assertEqual(skill.retarget.call_count,1)
        session.stop.assert_called_once_with('needs_clarification','seating_proposal_expired_or_context_changed')

    def test_hold_confirmation_retains_result_then_rechecks_after_stop(self):
        context=dict(clock_domain='live',local_identity=['session',1,'odom'],map_binding=dict(valid=True,map_id='m',map_frame='map'))
        handle=dict(task_id='task',goal_id='seat',call_id='call',goal_revision=1)
        active=dict(selection_id='selection',goal_id='seat',revision=1,context=context,
                    observation_binding='original',sequence=1,handle=handle)
        result=dict(selection_id='selection',sequence=2,status='selected',observation_binding='original',
            local_identity=context['local_identity'],clock_domain='live',map_binding=context['map_binding'],
            geometry_stamp_s=10.,pose=[2.,1.,0.],mapped_edge={})
        feedback=dict(stamp_s=10.,clock_domain='live',session_id='session',generation=1,frame_id='odom',map_binding=context['map_binding'])
        goal=dict(binding='binding',requirement={},selected=dict(pose=[1.,1.,0.],handle=handle))
        skill=SimpleNamespace(record=dict(handle=handle,status='holding',completed=[]),
            start=Mock(return_value=dict(status='accepted',handle=dict(handle,call_id='new',goal_revision=2))))
        nav=SimpleNamespace(scope='live',value={'seating':result},snapshot=lambda:feedback,client=SimpleNamespace(exchange=Mock()))
        session=SimpleNamespace(seating_active=active,seating_pending=None,state=dict(status='waiting_navigation',revision=1),
            skill=skill,navigation=nav,task_goals=SimpleNamespace(current=lambda _:goal),catalog=SimpleNamespace(value={'frame_id':'map'}),
            spatial_view=SimpleNamespace(record={'view_id':'view'}),prepare_current_images=lambda:None,
            check_candidate=lambda *_:dict(status='available',static_footprint=dict(status='free'),local_footprint=dict(status='free')),
            observation=None,_save=Mock(),stop=Mock())
        poll_updates(session)
        self.assertEqual(active['sequence'],1)
        skill.start.assert_not_called()
        skill.record['status']='held'
        poll_updates(session)
        self.assertEqual(active['sequence'],2)
        skill.start.assert_called_once()
        self.assertEqual(goal['selected']['handle']['call_id'],'new')


if __name__=='__main__':unittest.main()
