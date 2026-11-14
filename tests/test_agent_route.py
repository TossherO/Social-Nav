"""Agent-selected routes and circuit evidence, using synthetic feedback only."""
import copy
import io
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
from decision.agent import CONFIG_DIR,ROOT
from decision.agent.core.route import circuit_evidence,circuit_plan_diagnostics,path_outside_region,static_route_line_diagnostics
from decision.agent.core.loop import AgentLoop
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.core.tasks import centroid
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback,query_map,reply


class RouteFeedback(ScriptedFeedback):
    def start_route(self,place,handle):
        self.handle=copy.deepcopy(handle)
        self.points=iter(place['route']['waypoints'])
        self.final_yaw=place['route']['final_yaw']
        self.start_stamp=None
        self.state.update(handle=copy.deepcopy(handle),status='running',stopped=False)
        return dict(accepted=True,handle=copy.deepcopy(handle))

    def tick(self):
        self.t+=.11;self.state['stamp_s']=10+self.t
        if self.state['status']=='running':
            point=next(self.points,None)
            if point is None:
                self.state.update(status='arrived',stopped=True,goal=dict(task_completed=True,
                    task_id=self.handle['task_id'],goal_revision=self.handle['goal_revision']))
            else:
                self.state['pose']=[*point,self.final_yaw]
                if self.start_stamp is None:self.start_stamp=self.state['stamp_s']
                self.state['route_start_stamp_s']=self.start_stamp
        return self.snapshot()


class AgentRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.temp.name)/'session.sqlite')
        self.catalog=Catalog.load(CONFIG_DIR/'simulation.json')
        self.nav=RouteFeedback()
        from perception.environment.layers import GridSnapshot
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,pose=self.nav.state['pose'],
            stamp_s=self.nav.state['stamp_s'],clock_domain=self.nav.clock_domain)
        self.session=AgentSession('绕区域顺时针一圈',self.catalog,self.nav,self.store)

    def tearDown(self):
        self.store.close();self.temp.cleanup()

    def points(self,direction='clockwise'):
        polygon=self.catalog.regions['lobby_area']['polygon'];center=centroid(polygon)
        radius=max(np.linalg.norm(np.asarray(v)-center) for v in polygon)+2
        sign=-1 if direction=='clockwise' else 1
        return [[float(center[0]+radius*math.cos(sign*2*math.pi*i/8)),
                 float(center[1]+radius*math.sin(sign*2*math.pi*i/8))] for i in range(9)]

    def test_circuit_executes_one_plan_and_requires_measured_closed_trace(self):
        self.assertEqual(self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',
            target_id='lobby_area',direction='clockwise',description='绕一圈')])['status'],'recorded')
        view=query_map(self.session)
        ack=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                              waypoints=self.points(),final_yaw=0.)
        self.assertEqual(ack['status'],'accepted')
        self.assertFalse(ack['execution_completed'])
        for _ in range(10):self.session.tick()
        outcome=self.session.call('check_goal',goal_id='loop')
        self.assertEqual(outcome['status'],'completed')
        self.assertFalse(outcome['real_robot_success'])
        self.assertTrue(outcome['checks']['circuit_met'])

    def test_delayed_route_rechecks_current_view_and_preserves_original_call(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.))
        original=copy.deepcopy(call);self.nav.state['stamp_s']=14.
        self.assertFalse(self.session.valid_context(context));self.assertTrue(self.session.valid_expired_context(context))
        result=self.session.revalidate_delayed_route(call,context)
        self.assertEqual(result['status'],'accepted');self.assertEqual(call,original)
        self.assertEqual(result['revalidation']['model_view_id'],view)
        self.assertEqual(result['revalidation']['model_source_stamp_s'],10.)
        self.assertEqual(result['revalidation']['checked_source_stamp_s'],14.)
        self.assertNotEqual(result['revalidation']['checked_view_id'],view)
        self.assertEqual(self.session.skill.record['handle']['view_id'],result['revalidation']['checked_view_id'])

    def test_delayed_route_refuses_when_pose_changed(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        self.nav.state.update(stamp_s=14.,pose=[5.,4.,0.])
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.))
        self.assertEqual(self.session.revalidate_delayed_route(call,context)['reason'],'robot_pose_changed_since_model_input')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_delayed_route_refuses_heading_or_localization_identity_change(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.))
        self.nav.state.update(stamp_s=14.,pose=[4.,4.,.5])
        self.assertEqual(self.session.revalidate_delayed_route(call,context)['reason'],'robot_pose_changed_since_model_input')
        view=self.session.spatial_view.record['view_id'];context=self.session.context()
        call['arguments']['view_id']=view
        self.nav.state.update(stamp_s=18.,generation=1)
        self.assertEqual(self.session.revalidate_delayed_route(call,context)['status'],'refused')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_delayed_route_rejects_old_evidence_and_other_execution_scopes(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.))
        self.nav.state['stamp_s']=41.
        self.assertEqual(self.session.revalidate_delayed_route(call,context)['status'],'refused')
        self.nav.state['stamp_s']=14.;self.nav.scope='live'
        self.assertEqual(self.session.revalidate_delayed_route(call,context)['status'],'refused')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_delayed_invalid_route_returns_current_geometry_feedback(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session);context=self.session.context();points=self.points()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=[points[0],centroid(self.catalog.regions['lobby_area']['polygon']).tolist(),points[4]],final_yaw=0.))
        self.nav.state['stamp_s']=14.;result=self.session.revalidate_delayed_route(call,context)
        self.assertEqual(result['reason'],'circuit_waypoint_inside_region')
        self.assertIn('waypoint_inside_region',result['diagnostics']['issues'])
        self.assertEqual(self.session.route_preview['view_id'],result['revalidation']['checked_view_id'])
        self.assertIsNone(self.session.skill.record['handle'])

    def test_route_feedback_reports_actual_direction_even_if_region_intersected(self):
        polygon=self.catalog.regions['lobby_area']['polygon'];points=self.points('clockwise')
        result=circuit_plan_diagnostics(points,polygon,'counterclockwise',.2)
        self.assertEqual(result['actual_direction'],'clockwise')
        self.assertIn('direction_opposite_to_requested',result['issues'])
        points[2]=centroid(polygon).tolist()
        result=circuit_plan_diagnostics(points,polygon,'counterclockwise',.2)
        self.assertIn('waypoint_inside_region',result['issues'])

    def test_repeated_failed_route_waits_for_user_instead_of_calling_model_32_times(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        self.nav.tick=lambda:self.nav.snapshot()
        points=self.points();bad=[points[0],centroid(self.catalog.regions['lobby_area']['polygon']).tolist(),points[4]]
        def work(_payload,_observation,view):
            self.nav.state['stamp_s']+=4.
            return reply('navigate_route',goal_id='loop',view_id=view.record['view_id'],
                         orientation_source='autonomous',waypoints=bad,final_yaw=0.)
        actor=AgentLoop(self.session,work)
        try:
            for _ in range(6):
                actor.tick()
                if actor.model.thread is not None:actor.model.thread.join(2.)
                if self.session.state['status']=='needs_clarification':break
            self.assertEqual(actor.requests,3)
            self.assertEqual(self.session.state['reason'],'repeated_route_refusal')
            self.assertEqual(self.store.db.execute("select count(*) from events where kind='route_repeated_refusal'").fetchone()[0],1)
            self.assertIsNone(self.session.skill.record['handle'])
        finally:actor.close()

    def test_repeated_refusal_guard_does_not_stop_an_active_navigation_handle(self):
        actor=AgentLoop(self.session,lambda *_:None)
        self.session.skill.record['handle']={'call_id':'active'}
        value=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id='view',waypoints=[[1,1],[2,2]],
                   orientation_source='autonomous',final_yaw=0.))
        try:
            for _ in range(4):
                self.assertFalse(actor.repeated_route_refusal(value,dict(status='refused',reason='cancel_and_confirm_existing_call_first')))
            self.assertEqual(self.session.skill.record['handle'],{'call_id':'active'})
            self.assertEqual(self.store.db.execute("select count(*) from events where kind='route_repeated_refusal'").fetchone()[0],0)
        finally:actor.close()

    def test_circuit_adds_closure_and_rejects_waypoint_inside_region(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session);points=self.points()
        bad=[points[0],centroid(self.catalog.regions['lobby_area']['polygon']).tolist(),points[4]]
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
            waypoints=bad,final_yaw=0.)
        self.assertEqual(result['reason'],'circuit_waypoint_inside_region')
        self.assertEqual(result['diagnostics']['inside_waypoint_indices'],[1])
        self.assertEqual(result['diagnostics']['scope'],'waypoint_order_only_not_planned_path_or_execution_check')
        state=self.session.prompt()['current_state']
        self.assertEqual(state['last_tool_feedback']['diagnostics']['inside_waypoint_indices'],[1])
        self.assertEqual(self.session.spatial_view.record['route_preview']['waypoints'],bad)
        pixels=np.asarray(Image.open(io.BytesIO(self.session.spatial_view.images[1])))
        self.assertTrue(np.any(np.all(pixels==[207,48,48],axis=2)))
        self.assertIsNone(self.session.skill.record['handle'])
        view=self.session.spatial_view.record['view_id']
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
            waypoints=[points[0],points[3],points[6]])
        self.assertEqual(result['status'],'accepted')
        route=self.session.task_goals.current('loop')['selected']['route']
        self.assertEqual(route['waypoints'],[points[0],points[3],points[6],points[0]])
        self.assertIsNone(route['final_yaw'])

    def test_circuit_order_does_not_reject_imaginary_crossing_chords(self):
        polygon=[[-1,-1],[1,-1],[1,1],[-1,1]]
        points=[[2,0],[-1,-2],[-1,2]]
        self.assertFalse(path_outside_region(points+[points[0]],polygon))
        result=circuit_plan_diagnostics(points,polygon,'clockwise',.2)
        self.assertIsNone(result['reason'])
        self.assertTrue(result['closure_added'])

    def test_manual_region_check_uses_matching_fresh_reference_only(self):
        from decision.agent.workflows.manual_reference import region_check
        handle={'call_id':'test'}
        route={'region_polygon':[[-1,-1],[1,-1],[1,1],[-1,1]]}
        preview=dict(owner='test',request_id='test:1',reference_frame='map',status='reference_ready',
                     reference_stamp_s=10.,reference_path=[[-2,0],[2,0]])
        self.assertEqual(region_check(preview,route,1,handle,'map',11.),'crosses')
        self.assertEqual(region_check(preview,route,0,handle,'map',11.),'not_applicable')
        self.assertEqual(region_check(preview,route,1,handle,'map',14.),'unavailable')
        self.assertEqual(region_check(dict(preview,request_id='old'),route,1,handle,'map',11.),'unavailable')
        self.assertEqual(region_check(dict(preview,reference_path=None),route,1,handle,'map',11.),'unavailable')
        self.assertEqual(region_check(dict(preview,reference_path=[[-2,2],[2,2]]),route,1,handle,'map',11.),'clear')

    def test_manual_preview_exposes_only_matching_reference(self):
        from runtime.control import manual_preview_path
        request=dict(task_id='t',goal_revision=1,goal=[1.,2.],mode='mapped',frame_id='map')
        reference=dict(result=dict(frame_id='map',request=request,path=[[0,0,0],[1,2,0]]))
        self.assertEqual(manual_preview_path(reference,request),[[0.,0.],[1.,2.]])
        self.assertIsNone(manual_preview_path(reference,dict(request,goal=[2.,2.])))
        self.assertIsNone(manual_preview_path(dict(result=dict(reference['result'],frame_id='odom')),request))
        self.assertIsNone(manual_preview_path(dict(result=dict(reference['result'],path=[[0,0]]*1025)),request))

    def test_manual_region_conflict_withdraws_suggestion_and_reports(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        from decision.agent.workflows.manual import cancel_region_conflict
        session=SimpleNamespace(store=Mock(),conversation=Mock(),skill=Mock())
        reference=Mock()
        with patch('decision.agent.workflows.manual.emit') as emit:
            cancel_region_conflict(session,reference,'goal',1)
        session.skill.cancel.assert_called_once_with(reason='planned_route_intersects_task_region')
        reference.set.assert_called_once_with(None,None)
        self.assertIn('手柄停稳',emit.call_args.args[0]['instruction'])
        session.conversation.notice.assert_called_once()

    def test_waypoint_checks_reject_occupied_and_unknown_without_current_evidence(self):
        from decision.agent.core.route import waypoint_occupancy_diagnostics
        from perception.environment.layers import GridSnapshot
        metadata=dict(frame_id='map',shape=[10,10],resolution=1.,origin_xy=[0.,0.],snapshot_id='test')
        cells=np.zeros((10,10),np.int8);cells[2,2]=100;cells[4,4]=-1
        grid=GridSnapshot(metadata,dict(occupancy=cells))
        result=waypoint_occupancy_diagnostics(dict(global_grid=grid),[[2.5,2.5],[4.5,4.5],[6.5,6.5]],0.)
        self.assertEqual([item['reason'] for item in result['waypoints']],
                         ['waypoint_statically_occupied','waypoint_unknown',None])
        self.assertFalse(result['full_footprint_checked'])

    def test_optional_final_yaw_is_not_added_to_reference(self):
        from decision.agent.workflows.manual_reference import reference_request
        catalog=type('Catalog',(),{'value':dict(frame_id='map',map_snapshot_id='map1')})()
        handle=dict(call_id='c',task_id='t',goal_revision=1)
        route=dict(waypoints=[[1,1],[2,2]],final_yaw=None)
        result=reference_request(handle,[2,2,1],route,1,catalog,require_yaw=False)
        self.assertEqual(result['goal'],[2,2])
        self.assertEqual(result['goal_behavior'],'stop')

    def test_static_line_feedback_distinguishes_occupied_unknown_and_outside(self):
        from perception.environment.layers import GridSnapshot
        cells=np.zeros((10,10),np.int8);cells[0,4]=100;cells[0,7]=-1
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[10,10],resolution=.1,
                               origin_xy=[0.,0.],snapshot_id='unit-static'),dict(occupancy=cells))
        result=static_route_line_diagnostics(grid,[[.05,.05],[.85,.05],[1.2,.05]])
        self.assertGreater(result['segments'][0]['occupied_samples'],0)
        self.assertGreater(result['segments'][0]['unknown_samples'],0)
        self.assertGreater(result['segments'][1]['outside_samples'],0)
        self.assertEqual(result['scope'],'static_centerline_only_not_vehicle_or_actual_planned_path')

    def test_route_rejects_changed_source_instead_of_using_old_view(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        inputs=self.nav.spatial_inputs();self.nav.spatial_inputs=lambda:inputs
        view=query_map(self.session);inputs['stamp_s']+=.1
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.)
        self.assertEqual(result['reason'],'spatial_source_changed')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_unspecified_direction_is_agent_choice_not_user_requirement(self):
        goal=dict(goal_id='loop',kind='circuit',target_id='lobby_area',description='绕一圈')
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'recorded')
        view=query_map(self.session)
        args=dict(goal_id='loop',view_id=view,orientation_source='autonomous',waypoints=self.points(),final_yaw=0.)
        self.assertEqual(self.session.call('navigate_route',**args)['reason'],'choose_a_circuit_direction')
        self.assertEqual(self.session.call('navigate_route',direction='clockwise',**args)['status'],'accepted')
        selected=self.session.task_goals.current('loop')['selected']
        self.assertNotIn('direction',selected['requirement'])
        self.assertEqual(selected['route']['direction_source'],'autonomous')

    def test_explicit_direction_cannot_be_changed_by_route(self):
        goal=dict(goal_id='loop',kind='circuit',target_id='lobby_area',direction='clockwise',description='顺时针绕')
        self.session.call('record_task',goals=[goal]);view=query_map(self.session)
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.,direction='counterclockwise')
        self.assertEqual(result['reason'],'route_direction_conflicts_with_user_requirement')
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points('counterclockwise'),final_yaw=0.)
        self.assertEqual(result['reason'],'proposed_route_does_not_make_requested_circuit')

    def test_direction_and_gaps_use_actual_trace_not_proposal(self):
        points=self.points();samples=[dict(stamp_s=i*.1,pose=[*p,0.]) for i,p in enumerate(points)]
        polygon=self.catalog.regions['lobby_area']['polygon'];segments=[dict(samples=samples)]
        self.assertEqual(circuit_evidence(segments,polygon,'clockwise',0.,.8,.12)['status'],'met')
        self.assertEqual(circuit_evidence(segments,polygon,'counterclockwise',0.,.8,.12)['status'],'not_met')
        self.assertEqual(circuit_evidence([dict(samples=samples[:3]),dict(samples=samples[3:])],polygon,
                                         'clockwise',0.,.8,.12)['status'],'unproven')
        self.assertFalse(path_outside_region([points[0],points[4]],polygon))

    def test_circuit_plan_rejects_reverse_winding_hidden_by_net_one_circle(self):
        from decision.agent.core.route import circuit_plan_diagnostics
        polygon=[[-1,-1],[1,-1],[1,1],[-1,1]]
        angles=np.r_[np.linspace(0,-4*math.pi,17),np.linspace(-4*math.pi,-2*math.pi,9)[1:]]
        points=np.column_stack((3*np.cos(angles),3*np.sin(angles)))
        result=circuit_plan_diagnostics(points,polygon,'clockwise',.12)
        self.assertEqual(result['reason'],'proposed_route_excessive_reverse_winding')
        self.assertAlmostEqual(result['signed_winding_rad'],2*math.pi)
        self.assertAlmostEqual(result['reverse_winding_rad'],2*math.pi)

    def test_adapter_without_route_support_is_explicitly_refused(self):
        self.nav.start_route=None
        # A backend must expose a callable, not merely an attribute.
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        view=query_map(self.session)
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.)
        self.assertEqual(result['reason'],'route_execution_not_supported_by_adapter')

    def test_static_view_without_current_pose_cannot_submit_route(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                                                     direction='clockwise',description='绕一圈')])
        inputs=self.nav.spatial_inputs()
        self.nav.spatial_inputs=lambda:dict(inputs,pose=None,stamp_s=None)
        view=query_map(self.session)
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.)
        self.assertEqual(result['reason'],'query_a_current_spatial_view_first')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_user_checkpoints_are_one_task_and_passed_in_order_without_model_calls(self):
        goal=dict(goal_id='visit',kind='sequence',target_id='',description='依次经过电梯口和走廊远端',
                  checkpoints=['elevator','lobby_entrance','corridor_wait'])
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'recorded')
        view=query_map(self.session)
        points=[[5.,4.],[6.,4.],[8.,4.]]
        wrong=self.session.call('navigate_route',goal_id='visit',view_id=view,orientation_source='autonomous',
                                waypoints=[points[1],points[0],points[2]],final_yaw=0.)
        self.assertEqual(wrong['reason'],'route_does_not_include_ordered_user_checkpoints')
        ack=self.session.call('navigate_route',goal_id='visit',view_id=view,orientation_source='autonomous',
                              waypoints=points,final_yaw=0.)
        self.assertEqual(ack['status'],'accepted')
        for _ in range(4):self.session.tick()
        outcome=self.session.call('check_goal',goal_id='visit')
        self.assertEqual(outcome['status'],'completed')
        self.assertTrue(outcome['checks']['sequence_met'])

    def test_sequence_uses_measured_order_and_does_not_accept_skipped_checkpoint(self):
        from decision.agent.core.route import sequence_evidence
        checkpoints=['elevator','lobby_entrance','corridor_wait']
        samples=[dict(stamp_s=i*.2,pose=p) for i,p in enumerate([[5.,4.,0.],[8.,4.,0.],[6.,4.,0.],[8.,4.,0.]])]
        result=sequence_evidence([dict(samples=samples)],checkpoints,self.session.task_goals.checkpoint_met,0.,.6)
        self.assertEqual(result['status'],'met')
        result=sequence_evidence([dict(samples=samples[:2])],checkpoints,self.session.task_goals.checkpoint_met,0.,.2)
        self.assertEqual(result['status'],'not_met')

    def test_route_cancel_never_completes_requirement(self):
        goal=dict(goal_id='visit',kind='sequence',target_id='',description='经过电梯口和大厅门口',
                  checkpoints=['elevator','lobby_entrance'])
        self.session.call('record_task',goals=[goal]);view=query_map(self.session)
        self.session.call('navigate_route',goal_id='visit',view_id=view,orientation_source='autonomous',
                          waypoints=[[5.,4.],[6.,4.]],final_yaw=0.)
        self.session.tick()
        self.assertEqual(self.session.call('cancel_navigation')['status'],'stopping')
        self.nav.stopped();self.session.tick()
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertFalse(self.session.task_goals.finished())
        self.assertEqual(self.session.call('check_goal',goal_id='visit')['status'],'unproven')

    def test_simulation_stops_if_new_reference_cuts_through_region(self):
        from decision.agent.adapters.simulation import SimulationNavigation
        nav=SimulationNavigation()
        nav.pose=np.array([4.,4.,0.])
        nav.start_route(dict(route=dict(waypoints=[[4.,4.],[8.,4.]],final_yaw=0.,
            region_polygon=[[5.,3.],[7.,3.],[7.,5.],[5.,5.]])),
            dict(task_id='t',call_id='c',goal_revision=1))
        nav.route_index=1;nav._route_request()
        nav.planner.query=lambda *args,**kwargs:dict(status='path_found',path=[[4.,4.,0.],[8.,4.,0.]])
        with patch('navigation.planning.continuity.reference_envelope',return_value=None):
            state=nav.tick()
        self.assertEqual(state['stop_reason'],'planned_route_intersects_task_region')
        self.assertEqual(state['status'],'stopped')
        np.testing.assert_allclose(state['pose'],[4.,4.,0.])

    @unittest.skipUnless((ROOT/'outputs/runtime/build/planning/libsocial_nav_pose_search.so').is_file(),
                         'runtime CPU planner is not available')
    def test_real_simulation_passes_intermediate_point_then_stops_at_final(self):
        from decision.agent.adapters.simulation import SimulationNavigation
        nav=SimulationNavigation()
        place=dict(pose=[5.,4.,0.],route=dict(waypoints=[[4.5,4.],[5.,4.]],final_yaw=0.,region_polygon=None))
        self.assertTrue(nav.check_route(place)['reference_found'])
        nav.start_route(place,dict(task_id='t',call_id='c',goal_revision=1))
        transition=None
        for _ in range(180):
            state=nav.tick()
            if state['route_index']==1 and transition is None:transition=state
            if state['status'] in ('arrived','stopped','failed'):break
        self.assertIsNotNone(transition)
        self.assertFalse(transition['stopped'])
        self.assertEqual(state['status'],'arrived')
        self.assertTrue(state['stopped'])
