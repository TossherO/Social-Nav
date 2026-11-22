"""Agent-selected routes and ordered evidence, using synthetic feedback only."""
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
from decision.agent.core.route import record_route_visit,validate_route_progress
from decision.agent.core.loop import AgentLoop
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback,query_map,reply


class RouteFeedback(ScriptedFeedback):
    def start_route(self,place,handle):
        self.handle=copy.deepcopy(handle)
        self.route=copy.deepcopy(place['route'])
        self.visits=[]
        self.points=iter(self.route['waypoints'])
        self.final_yaw=place['route']['final_yaw']
        self.start_stamp=None
        self.state.update(handle=copy.deepcopy(handle),status='running',stopped=False,goal=None,
                          route_count=len(self.route['waypoints']),route_index=0,route_visits=[],route_start_stamp_s=None,
                          planning_observation=None)
        return dict(accepted=True,handle=copy.deepcopy(handle))

    def tick(self):
        self.t+=.11;self.state['stamp_s']=10+self.t
        self.state.update(stop_measurement_stamp_s=self.state['stamp_s'],localization_status='valid')
        if self.state['status']=='running':
            point=next(self.points,None)
            if point is None:
                self.state.update(status='arrived',stopped=True,goal=dict(task_completed=True,
                    task_id=self.handle['task_id'],goal_revision=self.handle['goal_revision']))
            else:
                self.state['pose']=[*point,self.final_yaw or 0.]
                measured=dict(self.state,origin_pose=self.state['pose'],origin_frame_id='sim_odom',
                              origin_stamp_s=self.state['stamp_s'])
                record_route_visit(self.route,self.visits,self.handle,measured,len(self.visits),.2)
                self.state['route_visits']=copy.deepcopy(self.visits)
                self.state['route_index']=min(len(self.visits),len(self.route['waypoints'])-1)
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
        polygon=self.catalog.regions['lobby_area']['polygon'];center=np.mean(polygon,axis=0)
        radius=max(np.linalg.norm(np.asarray(v)-center) for v in polygon)+2
        sign=-1 if direction=='clockwise' else 1
        return [[float(center[0]+radius*math.cos(sign*2*math.pi*i/8)),
                 float(center[1]+radius*math.sin(sign*2*math.pi*i/8))] for i in range(9)]

    def test_route_visits_require_order_freshness_and_matching_context(self):
        route=dict(waypoints=[[4.,4.],[5.,4.],[4.,4.]])
        handle=dict(call_id='test')
        visits=[]
        state=dict(origin_pose=[4.,4.,0.],origin_frame_id='sim_odom',stamp_s=10.,origin_stamp_s=10.,
                   clock_domain='simulation',session_id='unit',generation=0,frame_id='sim_odom',
                   source='synthetic_differential_drive_plant')
        self.assertFalse(record_route_visit(route,visits,handle,state,1,.5))
        without_measurement_time = {key:value for key,value in state.items() if key != 'origin_stamp_s'}
        self.assertFalse(record_route_visit(route,visits,handle,without_measurement_time,0,.2))
        self.assertTrue(record_route_visit(route,visits,handle,state,0,.2))
        state.update(origin_pose=[5.,4.,0.],stamp_s=12.,origin_stamp_s=10.)
        self.assertFalse(record_route_visit(route,visits,handle,state,1,.5))
        state['origin_stamp_s']=12.
        self.assertTrue(record_route_visit(route,visits,handle,state,1,.5))
        state.update(origin_pose=[4.,4.,0.],stamp_s=13.,origin_stamp_s=13.)
        self.assertTrue(record_route_visit(route,visits,handle,state,2,.2))
        proof=dict(route_visits=visits,handle=handle,frame_id='sim_odom',stamp_s=13.,source=state['source'],
                   context=dict(clock_domain='simulation',local_identity=['unit',0,'sim_odom'],map_binding=None))
        self.assertEqual(validate_route_progress(route,proof)['status'],'met')
        altered=copy.deepcopy(proof);altered['route_visits'][1]['local_identity'][1]=1
        self.assertEqual(validate_route_progress(route,altered)['status'],'unproven')
        altered=copy.deepcopy(proof);altered['route_visits'].pop(1)
        self.assertEqual(validate_route_progress(route,altered)['status'],'unproven')

    def test_long_trajectory_keeps_older_measured_samples(self):
        from decision.agent.core.trajectory import Trajectory
        trajectory=Trajectory(capacity=64)
        for i in range(500):
            state=dict(origin_pose=[i*.01,0.,0.],origin_frame_id='map',stamp_s=i*.11,
                origin_stamp_s=i*.11,clock_domain='simulation',session_id='test',generation=0,frame_id='odom')
            trajectory.update(state,'map')
        segments=trajectory.snapshot(frame_id='map',clock_domain='simulation',now=100.)
        samples=segments[0]['samples']
        self.assertEqual(samples[0]['stamp_s'],0.)
        self.assertGreater(samples[-1]['stamp_s'],50.)
        self.assertLessEqual(len(samples),64)
        self.assertFalse(trajectory.truncated)

    def test_delayed_route_rechecks_current_view_and_preserves_original_call(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证'))
        original=copy.deepcopy(call);self.nav.state['stamp_s']=14.
        self.assertTrue(self.session.valid_context(context))
        result=self.session.revalidate_route(call,context)
        self.assertEqual(result['status'],'accepted');self.assertEqual(call,original)
        self.assertEqual(result['revalidation']['model_view_id'],view)
        self.assertEqual(result['revalidation']['model_source_stamp_s'],10.)
        self.assertEqual(result['revalidation']['checked_source_stamp_s'],14.)
        self.assertNotEqual(result['revalidation']['checked_view_id'],view)
        self.assertEqual(self.session.skill.record['handle']['view_id'],result['revalidation']['checked_view_id'])

    def test_delayed_route_refuses_when_pose_changed(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        self.nav.state.update(stamp_s=14.,pose=[5.,4.,0.])
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证'))
        self.assertEqual(self.session.revalidate_route(call,context)['reason'],'robot_pose_changed_since_model_input')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_delayed_route_refuses_heading_or_localization_identity_change(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证'))
        self.nav.state.update(stamp_s=14.,pose=[4.,4.,.5])
        self.assertEqual(self.session.revalidate_route(call,context)['reason'],'robot_pose_changed_since_model_input')
        view=self.session.spatial_view.record['view_id'];context=self.session.context()
        call['arguments']['view_id']=view
        self.nav.state.update(stamp_s=18.,generation=1)
        self.assertEqual(self.session.revalidate_route(call,context)['status'],'refused')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_route_revalidates_after_long_model_response(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证'))
        self.nav.state['stamp_s']=100.
        self.assertEqual(self.session.revalidate_route(call,context)['status'],'accepted')
        self.assertIsNotNone(self.session.skill.record['handle'])

    def test_route_remains_unsupported_in_live_scope(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',description='绕一圈')])
        view=query_map(self.session);context=self.session.context()
        self.nav.scope='live'
        call=dict(tool='navigate_route',arguments=dict(goal_id='loop',view_id=view,orientation_source='autonomous',
                  waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证'))
        self.assertEqual(self.session.revalidate_route(call,context)['reason'],'route_execution_not_supported_by_adapter')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_repeated_failed_route_waits_for_user_instead_of_calling_model_32_times(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        self.nav.tick=lambda:self.nav.snapshot()
        points=self.points();bad=[points[0],[-1.,-1.],points[4]]
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
                self.assertFalse(actor.repeated_refusal(value,dict(status='refused',reason='cancel_and_confirm_existing_call_first')))
            self.assertEqual(self.session.skill.record['handle'],{'call_id':'active'})
            self.assertEqual(self.store.db.execute("select count(*) from events where kind='route_repeated_refusal'").fetchone()[0],0)
        finally:actor.close()

    def test_repeated_navigation_refusal_ignores_view_id_and_waits_after_three(self):
        actor=AgentLoop(self.session,lambda *_:None)
        try:
            value=dict(tool='navigate',arguments=dict(goal_id='goal',pose=[5.,4.,0.],view_id='v1',orientation_source='autonomous'))
            refusal=dict(status='refused',reason='current_stage_does_not_support_navigation')
            self.assertFalse(actor.repeated_refusal(value,refusal))
            value['arguments']['view_id']='v2'
            self.assertFalse(actor.repeated_refusal(value,refusal))
            value['arguments']['view_id']='v3'
            self.assertTrue(actor.repeated_refusal(value,refusal))
            self.assertEqual(self.store.db.execute("select count(*) from events where kind='navigation_repeated_refusal'").fetchone()[0],1)
            self.session.skill.record['handle']={'call_id':'active'}
            self.assertFalse(actor.repeated_refusal(value,refusal))
        finally:actor.close()

    def test_many_waypoints_request_review_but_allow_an_explained_route(self):
        self.session.call('record_task',goals=[dict(goal_id='route',kind='route',target_id='',description='经过多处')])
        view=query_map(self.session)
        points=[[4.,4.],[5.,4.],[6.,4.],[7.,4.],[8.,4.],[9.,4.]]
        args=dict(goal_id='route',view_id=view,orientation_source='autonomous',waypoints=points)
        result=self.session.call('navigate_route',**args)
        self.assertEqual(result['reason'],'route_waypoints_review_required')
        self.assertEqual(result['distinct_waypoints'],6)
        self.assertIsNone(self.session.skill.record['handle'])
        result=self.session.call('navigate_route',**args,waypoint_reason='用户指定依次经过这些地点')
        self.assertEqual(result['status'],'accepted',result)
        route=self.session.task_goals.current('route')['selected']['route']
        self.assertEqual(route['waypoints'],points)
        self.assertEqual(route['waypoint_reason'],'用户指定依次经过这些地点')

    def test_three_waypoints_are_not_automatically_closed(self):
        self.session.call('record_task',goals=[dict(goal_id='route',kind='route',target_id='',description='按顺序经过')])
        view=query_map(self.session);points=[[4.,4.],[5.,4.],[6.,4.]]
        result=self.session.call('navigate_route',goal_id='route',view_id=view,orientation_source='autonomous',waypoints=points)
        self.assertEqual(result['status'],'accepted',result)
        self.assertEqual(self.session.task_goals.current('route')['selected']['route']['waypoints'],points)

    def test_four_waypoints_request_review_before_route_is_accepted(self):
        self.session.call('record_task',goals=[dict(goal_id='route',kind='route',target_id='',description='按顺序经过')])
        view=query_map(self.session);points=[[4.,4.],[5.,4.],[6.,4.],[7.,4.]]
        result=self.session.call('navigate_route',goal_id='route',view_id=view,orientation_source='autonomous',waypoints=points)
        self.assertEqual(result['reason'],'route_waypoints_review_required')
        self.assertEqual(result['distinct_waypoints'],4)
        self.assertIsNone(self.session.skill.record['handle'])

    def test_waypoint_reason_never_bypasses_occupied_check(self):
        from perception.environment.layers import GridSnapshot
        occupancy=np.zeros((100,140),np.int8);occupancy[40,90]=100
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='occupied'),
                          dict(occupancy=occupancy))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,pose=self.nav.state['pose'],
            stamp_s=self.nav.state['stamp_s'],clock_domain=self.nav.clock_domain)
        self.session.call('record_task',goals=[dict(goal_id='route',kind='route',target_id='',description='经过多处')])
        view=query_map(self.session);points=[[4.,4.],[5.,4.],[6.,4.],[7.,4.],[8.,4.],[9.05,4.05]]
        result=self.session.call('navigate_route',goal_id='route',view_id=view,orientation_source='autonomous',
                                 waypoints=points,waypoint_reason='需要这些位置')
        self.assertEqual(result['reason'],'waypoint_statically_occupied')
        self.assertEqual(result['distinct_waypoints'],6)
        self.assertIn('三个',result['guidance'])
        self.assertIsNone(self.session.skill.record['handle'])

    def test_manual_preview_exposes_only_matching_reference(self):
        from runtime.control import manual_preview_path
        request=dict(task_id='t',goal_revision=1,goal=[1.,2.],mode='mapped',frame_id='map')
        reference=dict(result=dict(frame_id='map',request=request,path=[[0,0,0],[1,2,0]]))
        self.assertEqual(manual_preview_path(reference,request),[[0.,0.],[1.,2.]])
        self.assertIsNone(manual_preview_path(reference,dict(request,goal=[2.,2.])))
        self.assertIsNone(manual_preview_path(dict(result=dict(reference['result'],frame_id='odom')),request))
        self.assertIsNone(manual_preview_path(dict(result=dict(reference['result'],path=[[0,0]]*1025)),request))

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

    def test_route_rejects_changed_source_instead_of_using_old_view(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        inputs=self.nav.spatial_inputs();self.nav.spatial_inputs=lambda:inputs
        view=query_map(self.session);inputs['stamp_s']+=.1
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证')
        self.assertEqual(result['reason'],'spatial_source_changed')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_adapter_without_route_support_is_explicitly_refused(self):
        self.nav.start_route=None
        # A backend must expose a callable, not merely an attribute.
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        view=query_map(self.session)
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证')
        self.assertEqual(result['reason'],'route_execution_not_supported_by_adapter')

    def test_static_view_without_current_pose_cannot_submit_route(self):
        self.session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                                                     description='绕一圈')])
        inputs=self.nav.spatial_inputs()
        self.nav.spatial_inputs=lambda:dict(inputs,pose=None,stamp_s=None)
        view=query_map(self.session)
        result=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                                 waypoints=self.points(),final_yaw=0.,waypoint_reason='合成测试需要逐段验证')
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
        extra=[points[0],[5.5,4.],points[1],[6.5,4.],[7.,4.],points[2]]
        review=self.session.call('navigate_route',goal_id='visit',view_id=view,orientation_source='autonomous',
                                 waypoints=extra,final_yaw=0.)
        self.assertEqual(review['reason'],'route_waypoints_review_required')
        self.assertIn('保留用户指定的全部检查点',review['explanation'])
        ack=self.session.call('navigate_route',goal_id='visit',view_id=view,orientation_source='autonomous',
                              waypoints=points,final_yaw=0.)
        self.assertEqual(ack['status'],'accepted')
        for _ in range(4):self.session.tick()
        outcome=self.session.call('check_goal',goal_id='visit')
        self.assertEqual(outcome['status'],'completed')
        self.assertEqual(outcome['checks']['route_progress']['waypoint_count'],3)

    def test_legacy_circuit_uses_generic_route_evidence_without_automatic_completion(self):
        # A historical task may still contain circuit metadata. Passing ordered
        # waypoints makes evidence available; it does not complete the user's
        # semantic requirement or add a separate geometric circuit gate.
        goal=dict(goal_id='loop',kind='route',target_id='lobby_area',description='绕区域一圈',
                  circuit=dict(turns=1,direction='clockwise'))
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'recorded')
        view=query_map(self.session)
        ack=self.session.call('navigate_route',goal_id='loop',view_id=view,orientation_source='autonomous',
                              waypoints=[[5.,4.],[6.,4.],[8.,4.]],final_yaw=0.)
        self.assertEqual(ack['status'],'accepted')
        for _ in range(4):self.session.tick()
        from decision.agent.core.completion import stage_evidence
        stage=self.session.task_goals.current('loop')
        facts,_=stage_evidence(self.session,stage)
        self.assertEqual(facts['status'],'available')
        self.assertEqual(facts['checks']['route_progress']['waypoint_count'],3)
        self.assertNotIn('circuit',facts['checks'])
        self.assertEqual(stage['status'],'pending')

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

    def test_blocked_route_replacement_keeps_endpoint_and_uses_only_new_handle_for_route_progress(self):
        goal=dict(goal_id='visit',kind='sequence',target_id='',description='依次经过电梯口和大厅门口',
                  checkpoints=['elevator','lobby_entrance','corridor_wait'],required_yaw=.1)
        self.session.call('record_task',goals=[goal])
        args=dict(goal_id='visit',orientation_source='user',waypoints=[[5.,4.],[6.,4.],[8.,4.]])
        accepted=self.session.call('navigate_route',view_id=query_map(self.session),**args)
        old_handle=accepted['handle']
        self.session.tick()
        planning=dict(status='blocked',kind='route_blocked',reason='test_blocked_reference',source='reference',
                      goal_binding='unit-route-leg',source_stamp_s=self.nav.state['stamp_s'],
                      valid_until_s=self.nav.state['stamp_s']+5.,blocked_position=[5.5,4.])
        self.nav.state.update(planning_observation=planning,route_index=1)
        self.session.skill.record.update(status='blocked',blockage=dict(planning,handle=old_handle,route_index=1,current=True,
            first_source_stamp_s=planning['source_stamp_s']-3.,observation_count=2))
        self.assertEqual(self.session.call('cancel_navigation')['status'],'stopping')
        stage=self.session.task_goals.current('visit')
        self.assertEqual(self.session.navigation_replacement_gate(stage)['reason'],'awaiting_confirmed_stop_after_blockage')
        self.nav.stopped();self.session.tick()
        archived=stage['route_attempts'][0]
        self.assertEqual(archived['handle'],old_handle)
        self.assertEqual([visit['waypoint'] for visit in archived['visits']],[[5.,4.]])
        self.assertTrue(archived['measured_trajectory'])
        view=query_map(self.session)
        self.assertEqual(self.session.call('navigate_route',view_id=view,**args)['reason'],
                         'route_replacement_repeats_blocked_segment')
        self.assertEqual(self.session.call('navigate_route',view_id=view,**dict(args,waypoints=[[6.,4.],[8.,4.]]))['reason'],
                         'route_replacement_repeats_blocked_segment')
        alternative=[[5.5,5.],[6.,4.],[8.,4.]]
        self.assertEqual(self.session.call('navigate_route',view_id=view,**dict(args,waypoints=alternative,final_yaw=.2))['reason'],
                         'route_replacement_changed_required_final_yaw')
        self.assertEqual(self.session.call('navigate_route',view_id=view,**dict(args,waypoints=[[5.5,5.],[6.,4.],[8.,4.1]]))['reason'],
                         'route_replacement_changed_final_position')
        self.assertEqual(self.session.call('navigate_route',view_id=view,**dict(args,waypoints=[[5.5,4.],*alternative]))['reason'],
                         'route_replacement_uses_blocked_position')
        self.nav.check_route=lambda _:dict(reference_found=False,status='no_path',reason='candidate_preview_failed')
        refusal=self.session.call('navigate_route',view_id=view,**dict(args,waypoints=alternative))
        self.assertEqual(refusal['reason'],'candidate_preview_failed')
        self.assertEqual(stage['selected']['handle'],old_handle)
        self.nav.check_route=RouteFeedback.check_route.__get__(self.nav)
        accepted=self.session.call('navigate_route',view_id=view,**dict(args,waypoints=alternative))
        self.assertEqual(accepted['status'],'accepted',accepted)
        self.assertNotEqual(accepted['handle'],old_handle)
        self.assertEqual(stage['selected']['route']['final_yaw'],.1)
        self.assertEqual(self.nav.state['route_visits'],[])
        for _ in range(4):self.session.tick()
        result=self.session.call('check_goal',goal_id='visit')
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['checks']['sequence_checkpoint_progress'],dict(confirmed_count=3,required_count=3))
        visits=result['checks']['route_progress']['visits']
        self.assertEqual([visit['waypoint'] for visit in visits],alternative)
        self.assertTrue(all(visit['handle']==accepted['handle'] for visit in visits))
        self.assertEqual(len(stage['route_attempts']),1)

    def test_ordinary_cancellation_cannot_change_a_route(self):
        self.session.call('record_task',goals=[dict(goal_id='route',kind='route',target_id='',description='按顺序经过')])
        args=dict(goal_id='route',orientation_source='autonomous',waypoints=[[5.,4.],[6.,4.]])
        self.assertEqual(self.session.call('navigate_route',view_id=query_map(self.session),**args)['status'],'accepted')
        self.session.call('cancel_navigation')
        self.nav.stopped();self.session.tick()
        view=query_map(self.session)
        changed=self.session.call('navigate_route',view_id=view,**dict(args,waypoints=[[5.,5.],[6.,4.]]))
        self.assertEqual(changed['reason'],'route_replacement_requires_confirmed_blockage')
        self.assertEqual(self.session.call('navigate_route',view_id=view,**args)['status'],'accepted')

    def test_simulation_replacement_preflight_compares_actual_first_leg_to_blocked_position(self):
        from types import SimpleNamespace
        from decision.agent.adapters.simulation import SimulationNavigation
        nav=SimulationNavigation.__new__(SimulationNavigation)
        nav.t=0.;nav.pose=[4.,4.,0.];nav.geometry=None
        nav.world=SimpleNamespace(scene=lambda *_:None)
        plan=dict(status='path_found',reason=None,frame_id='sim_odom',path=[[4.,4.,0.],[6.,4.,0.]])
        nav.planner=SimpleNamespace(query=lambda *_args,**_kwargs:plan)
        place=dict(route=dict(waypoints=[[6.,4.],[8.,4.]]),route_blocked_position_check=
                   dict(position=[5.,4.],frame_id='sim_odom',tolerance_m=.2))
        result=nav.check_route(place)
        self.assertFalse(result['reference_found'])
        self.assertEqual(result['reason'],'route_replacement_preview_crosses_blocked_position')
        plan['path']=[[4.,4.,0.],[4.,5.,0.],[6.,5.,0.],[6.,4.,0.]]
        result=nav.check_route(place)
        self.assertTrue(result['reference_found'])
        self.assertEqual(result['blocked_position_check']['status'],'clear')
        self.assertFalse(result['remaining_route_legs_checked'])

    @unittest.skipUnless((ROOT/'outputs/runtime/build/planning/libsocial_nav_pose_search.so').is_file(),
                         'runtime CPU planner is not available')
    def test_real_simulation_passes_intermediate_point_then_stops_at_final(self):
        from decision.agent.adapters.simulation import SimulationNavigation
        nav=SimulationNavigation()
        place=dict(pose=[5.,4.,0.],route=dict(waypoints=[[4.5,4.],[5.,4.]],final_yaw=0.))
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
