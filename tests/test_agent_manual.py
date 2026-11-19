"""Operator-driven Agent checks use synthetic observations only."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from decision.agent import CONFIG_DIR
from decision.agent.adapters.manual import ManualNavigation
from decision.agent.adapters.runtime import ManualRuntimeClient
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from perception.environment.layers import GridSnapshot
from tests.test_agent import point_goal
from decision.agent.core.tasks import centroid
from decision.agent.workflows.manual import RouteDisplay,display_final_yaw,display_points
from runtime.control import agent_mapped_pose


class ManualTests(unittest.TestCase):
    def setUp(self):
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value)
        value.update(scope='development_prior',map_snapshot_id='static',localization_map_id='map-id')
        for place in value['places']:place['status']='manual_annotation'
        self.catalog=Catalog(value)
        self.binding=dict(valid=True,map_id='map-id',map_frame='sim_odom')
        self.feedback=dict(status='valid',source='causal_algorithm_poses_not_chassis_feedback',
            local_identity=['s',0,'odom'],stamp_s=10.,pose=[4.,4.,0.],velocity=[0.,0.])
        self.control=dict(clock_domain='runtime_live',stamp_s=10.,request=None,feedback={},idle_feedback=self.feedback,
            localization=dict(session_id='s',generation=0,frame_id='odom',health=0),map_binding=self.binding)
        self.obs=dict(local_identity=['s',0,'odom'],clock_domain='runtime_live',map_binding=self.binding,
            map_base_pose=[4.,4.,0.],stamp_s=10.)
        self.state=dict(control=self.control,observation=self.obs,received_age_s=.01,vehicle_context=None)
        self.client=SimpleNamespace(require_observation_only=lambda:None,snapshot=lambda **_:self.state)
        self.nav=ManualNavigation(self.catalog,self.client)

    def update(self,pose,stamp,velocity=(0.,0.)):
        self.control['stamp_s']=self.feedback['stamp_s']=self.obs['stamp_s']=stamp
        self.obs['map_base_pose']=list(pose)
        self.feedback['velocity']=list(velocity)

    def test_manual_client_refuses_motion_commands(self):
        client=ManualRuntimeClient('/unused')
        for kind in ('goal','cancel','hold','arm'):
            with self.assertRaisesRegex(ValueError,'read-only'):
                client.exchange(kind,{})

    def test_follow_waits_for_matching_status_and_current_measured_stop(self):
        self.client.exchange = MagicMock(return_value=dict(accepted=True))
        handle = dict(task_id='task', goal_revision=1, call_id='follow')
        self.nav.start_follow(dict(target_description='人物'), handle)
        self.state['follow'] = dict(handle=handle, status='running')
        self.update([4.,4.,0.], 10.1)
        self.assertEqual(self.nav.tick()['pose'], self.feedback['pose'])
        self.nav.finish_follow(handle)
        self.state['follow'] = dict(handle=dict(handle, call_id='old'), status='stopped')
        self.assertFalse(self.nav.tick()['stop_observation'])
        self.state['follow'] = dict(handle=handle, status='stopped')
        self.update([4.,4.,0.], 10.2, velocity=(.2,0.))
        self.assertFalse(self.nav.tick()['stop_observation'])
        self.update([4.,4.,0.], 10.3)
        state = self.nav.tick()
        self.assertTrue(state['stop_observation'])
        self.assertEqual(state['stop_measurement_stamp_s'], 10.3)
        self.assertEqual([call.args[0] for call in self.client.exchange.call_args_list],
                         ['follow_start','follow_finish'])
        self.state['follow'] = dict(handle=handle, status='preempted')
        interrupted = self.nav.tick()
        self.assertEqual(interrupted['status'], 'preempted')
        self.assertFalse(interrupted['stop_observation'])

    def test_automatic_arrival_uses_fresh_stopped_localization_and_dwell(self):
        self.nav.automatic_arrival=True
        handle=dict(task_id='task',goal_revision=1,call_id='first')
        self.nav.start(dict(place_id='elevator',pose=[5.,4.,0.],requirement={}),handle)
        self.update([5.,4.,0.],10.1,velocity=(.2,0.));self.nav.tick()
        self.update([5.,4.,0.],10.2);self.nav.tick()
        self.update([5.,4.,0.],10.6);self.assertFalse(self.nav.tick()['manual_arrival_observation'])
        self.update([5.,4.,0.],10.8);state=self.nav.tick()
        self.assertTrue(state['manual_arrival_observation'])
        self.assertIsNone(state['manual_confirmation'])
        self.assertEqual(state['manual_arrival_evidence']['source'],'automatic_localization_and_stop_estimate')
        self.update([5.5,4.,0.],10.9);self.assertFalse(self.nav.tick()['manual_arrival_observation'])

    def test_automatic_arrival_restarts_dwell_after_stale_or_motion(self):
        self.nav.automatic_arrival=True
        handle=dict(task_id='task',goal_revision=1,call_id='first')
        self.nav.start(dict(place_id='elevator',pose=[5.,4.,0.],requirement={}),handle)
        self.update([5.,4.,0.],10.1);self.nav.tick()
        self.update([5.,4.,0.],10.4,velocity=(.2,0.));self.nav.tick()
        self.update([5.,4.,0.],10.8);self.assertFalse(self.nav.tick()['manual_arrival_observation'])
        self.update([5.,4.,0.],11.2);self.assertFalse(self.nav.tick()['manual_arrival_observation'])
        self.update([5.,4.,0.],11.4);self.assertTrue(self.nav.tick()['manual_arrival_observation'])

    def test_automatic_arrival_with_slow_rgb_and_fresh_mapped_localization(self):
        self.nav.automatic_arrival=True
        self.nav.start(dict(place_id='elevator',pose=[5.,4.,0.],requirement={}),
                       dict(task_id='task',goal_revision=1,call_id='first'))
        for stamp in (10.1,10.3,10.5,10.7):
            self.control['stamp_s']=self.feedback['stamp_s']=stamp
            self.control['agent_mapped_pose']=dict(pose=[5.,4.,0.],stamp_s=stamp,
                clock_domain='runtime_live',local_identity=['s',0,'odom'],map_binding=self.binding)
            state=self.nav.tick()
        self.assertTrue(state['manual_arrival_observation'])
        self.assertEqual(state['origin_stamp_s'],10.7)
        self.assertEqual(self.obs['stamp_s'],10.)

    def test_manual_map_pose_never_falls_back_after_loss_or_identity_change(self):
        self.nav.automatic_arrival=True
        self.nav.start(dict(place_id='elevator',pose=[4.,4.,0.],requirement={}),
                       dict(task_id='task',goal_revision=1,call_id='first'))
        mapped=dict(pose=[4.,4.,0.],stamp_s=10.,clock_domain='runtime_live',
                    local_identity=['s',0,'odom'],map_binding=self.binding)
        self.control['agent_mapped_pose']=mapped
        self.assertIsNotNone(self.nav.tick()['origin_pose'])
        for change in (None,dict(mapped,local_identity=['other',0,'odom']),
                       dict(mapped,map_binding=dict(self.binding,map_id='wrong')),
                       dict(mapped,stamp_s=9.0),dict(mapped,clock_domain='runtime_bag')):
            self.control['agent_mapped_pose']=change
            state=self.nav.tick()
            self.assertIsNone(state['origin_pose'])
            self.assertFalse(state['manual_arrival_observation'])

    def test_live_mapped_pose_retains_source_time_and_rejects_expired_association(self):
        prior=dict(self.binding,local_session_id='s',local_generation=0,local_frame='odom',
                   measurement_stamp_s=10.,map_from_local=np.eye(4).tolist())
        scene=SimpleNamespace(record=dict(session_id='s',generation=0,frame_id='odom',
            clock_domain='runtime_live',localization={'status':'valid'},prior=prior))
        row=dict(session_id='s',generation=0,frame_id='odom',health=0,stamp_s=10.1,
                 position=[5.,4.,0.],quaternion_xyzw=[0.,0.,0.,1.])
        result=agent_mapped_pose(scene,row,self.nav.geometry,10.2,.5,'runtime_live')
        self.assertEqual(result['stamp_s'],10.1)
        self.assertEqual(result['association_stamp_s'],10.)
        np.testing.assert_allclose(result['pose'],self.nav.geometry.pose(np.array(
            [[1,0,0,5],[0,1,0,4],[0,0,1,0],[0,0,0,1]])))
        self.assertIsNone(agent_mapped_pose(scene,row,self.nav.geometry,10.51,.5,'runtime_live'))
        self.assertIsNone(agent_mapped_pose(scene,dict(row,generation=1),self.nav.geometry,10.2,.5,'runtime_live'))
        self.assertIsNone(agent_mapped_pose(scene,dict(row,health=1),self.nav.geometry,10.2,.5,'runtime_live'))
        self.assertIsNone(agent_mapped_pose(scene,dict(row,stamp_s=9.99),self.nav.geometry,10.2,.5,'runtime_live'))

    def test_runtime_state_delivers_fresh_map_pose_for_automatic_arrival(self):
        from runtime.command import AgentState
        state=AgentState()
        binding=dict(valid=True,map_id='map-id',map_frame='sim_odom',map_revision=0,
                     association_session='s:map',association_generation=0)
        prior=dict(binding,local_session_id='s',local_generation=0,local_frame='odom',
                   measurement_stamp_s=10.,map_from_local=np.eye(4).tolist())
        scene=SimpleNamespace(record=dict(session_id='s',generation=0,frame_id='odom',
            clock_domain='runtime_live',localization={'status':'valid'},prior=prior))
        row=dict(session_id='s',generation=0,frame_id='odom',health=0,stamp_s=10.,
                 position=[5.,4.,0.],quaternion_xyzw=[0.,0.,0.,1.])
        target=self.nav.geometry.pose(np.array([[1,0,0,5],[0,1,0,4],[0,0,1,0],[0,0,0,1]])).tolist()
        client=SimpleNamespace(require_observation_only=lambda:None,snapshot=lambda **_:state.snapshot())
        def publish(stamp):
            row['stamp_s']=stamp;prior['measurement_stamp_s']=stamp
            feedback=dict(self.feedback,stamp_s=stamp)
            pose=agent_mapped_pose(scene,row,self.nav.geometry,stamp,.5,'runtime_live')
            state.publish_control(dict(stamp_s=stamp,clock_domain='runtime_live',request=None,feedback={}),
                row,map_binding=binding,idle_feedback=feedback,agent_mapped_pose=pose)
        publish(10.)
        nav=ManualNavigation(self.catalog,client,automatic_arrival=True)
        nav.start(dict(place_id='elevator',pose=target,requirement={}),
                  dict(task_id='task',goal_revision=1,call_id='first'))
        for stamp in (10.1,10.3,10.5,10.7):
            publish(stamp);result=nav.tick()
        self.assertTrue(result['manual_arrival_observation'])
        self.assertEqual(result['manual_arrival_evidence']['stamp_s'],10.7)

    def test_automatic_arrival_can_complete_a_task_without_operator_confirmation(self):
        self.nav.automatic_arrival=True
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('到电梯',self.catalog,self.nav,store)
                session.call('record_task',goals=[point_goal('elevator')]);session.call('query_map')
                pose=self.catalog.places['elevator']['pose']
                result=session.call('navigate',goal_id='elevator',pose=pose,
                    view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')
                self.assertEqual(result['status'],'accepted')
                self.update(pose,10.2);session.tick()
                self.update(pose,10.5);session.tick()
                self.update(pose,10.8);session.tick()
                result=session.call('check_goal',goal_id='elevator')
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(result['evidence_kind'],'automatic_localization_and_stop_estimate')
                self.assertEqual(session.call('finish_task',status='succeeded',evidence='到达')['status'],'succeeded')
            finally:store.close()

    def test_two_manual_goals_advance_without_operator_confirmation(self):
        self.nav.automatic_arrival=True
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('依次到两个位置',self.catalog,self.nav,store)
                session.call('record_task',goals=[point_goal('elevator'),point_goal('lobby_entrance')])
                for index,target in enumerate(('elevator','lobby_entrance')):
                    session.call('query_map')
                    pose=self.catalog.places[target]['pose']
                    result=session.call('navigate',goal_id=target,pose=pose,
                        view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')
                    self.assertEqual(result['status'],'accepted',result)
                    for offset in (.2,.5,.8):
                        self.update(pose,10.+index+offset);session.tick()
                    result=session.call('check_goal',goal_id=target)
                    self.assertEqual(result['status'],'completed',result)
                    self.assertEqual(result['evidence_kind'],'automatic_localization_and_stop_estimate')
                self.assertEqual(session.call('finish_task',status='succeeded',evidence='到达')['status'],'succeeded')
            finally:store.close()

    def test_completed_route_reacquires_arrival_after_motion_without_duplicate_visit(self):
        self.nav.automatic_arrival=True
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('依次经过三个点',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='route',kind='route',target_id='',description='依次经过')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='route',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=[[4.,4.],[5.,4.],[6.,4.]])
                self.assertEqual(result['status'],'accepted',result)
                for stamp,pose in [(10.2,[4.,4.,0.]),(10.4,[5.,4.,0.]),(10.6,[6.,4.,0.]),
                                   (10.9,[6.,4.,0.]),(11.2,[6.,4.,0.])]:
                    self.update(pose,stamp);session.tick()
                self.assertEqual(session.skill.record['status'],'succeeded')
                original=copy.deepcopy(self.nav.route_visits)
                self.assertEqual(len(original),3)
                self.update([6.02,4.,0.],11.3,velocity=(.1,0.));session.tick()
                self.assertFalse(self.nav.snapshot()['manual_arrival_observation'])
                self.assertEqual(self.nav.arrival_invalidation['reason'],'movement_detected')
                self.assertEqual(store.db.execute("select count(*) from events where kind='manual_arrival_invalidated'").fetchone()[0],1)
                for stamp in (11.4,11.7,12.):
                    self.update([6.02,4.,0.],stamp);session.tick()
                self.assertTrue(self.nav.snapshot()['manual_arrival_observation'])
                self.assertEqual(self.nav.route_visits,original)
                self.assertEqual(store.db.execute("select count(*) from events where kind='manual_arrival_recovered'").fetchone()[0],1)
                self.assertEqual(session.call('check_goal',goal_id='route')['status'],'completed')
                saved=session.task_goals.goals[0]['completion']['proof']
                self.assertAlmostEqual(saved['stamp_s'],11.2)
                self.assertAlmostEqual(saved['current_manual_arrival_evidence']['stamp_s'],12.)
            finally:store.close()

    def test_reviewed_target_can_be_replaced_after_later_evidence_invalidation(self):
        self.nav.automatic_arrival=True
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('依次到两个位置',self.catalog,self.nav,store)
                session.call('record_task',goals=[point_goal('elevator'),point_goal('lobby_entrance')]);session.call('query_map')
                pose=self.catalog.places['elevator']['pose']
                self.assertEqual(session.call('navigate',goal_id='elevator',pose=pose,
                    view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')['status'],'accepted')
                for stamp in (10.2,10.5,10.8):self.update(pose,stamp);session.tick()
                self.assertEqual(session.call('check_goal',goal_id='elevator')['status'],'completed')
                self.update(pose,10.9,velocity=(.1,0.));session.tick()
                self.assertEqual(self.nav.status,'running')
                self.update(pose,11.);session.tick();session.call('query_map')
                result=session.call('navigate',goal_id='lobby_entrance',pose=self.catalog.places['lobby_entrance']['pose'],
                    view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')
                self.assertEqual(result['status'],'accepted',result)
            finally:store.close()

    def test_manual_confirmation_reuses_final_visit_after_automatic_evidence_is_lost(self):
        self.nav.automatic_arrival=True
        handle=dict(task_id='task',goal_revision=1,call_id='route')
        place=dict(place_id='goal',pose=[6.,4.,0.],requirement={},
                   route=dict(waypoints=[[4.,4.],[5.,4.],[6.,4.]],final_yaw=None))
        self.nav.start_route(place,handle)
        for stamp,pose in [(10.2,[4.,4.,0.]),(10.4,[5.,4.,0.]),(10.6,[6.,4.,0.]),
                           (10.9,[6.,4.,0.]),(11.2,[6.,4.,0.])]:
            self.update(pose,stamp);self.nav.tick()
        self.assertTrue(self.nav.snapshot()['manual_arrival_observation'])
        visits=copy.deepcopy(self.nav.route_visits)
        self.update([6.,4.,0.],11.3,velocity=(.1,0.));self.nav.tick()
        self.update([6.,4.,0.],11.4)
        confirmation=self.nav.confirm_arrival()
        self.assertEqual(confirmation['source'],'operator_confirmation_with_localization_estimate')
        self.assertTrue(self.nav.snapshot()['manual_arrival_observation'])
        self.assertEqual(self.nav.route_visits,visits)

    def test_recovery_does_not_reuse_waypoint_evidence_with_changed_identity(self):
        self.nav.automatic_arrival=True
        handle=dict(task_id='task',goal_revision=1,call_id='route')
        place=dict(place_id='goal',pose=[6.,4.,0.],requirement={},
                   route=dict(waypoints=[[4.,4.],[5.,4.],[6.,4.]],final_yaw=None))
        self.nav.start_route(place,handle)
        for stamp,pose in [(10.2,[4.,4.,0.]),(10.4,[5.,4.,0.]),(10.6,[6.,4.,0.]),
                           (10.9,[6.,4.,0.]),(11.2,[6.,4.,0.])]:
            self.update(pose,stamp);self.nav.tick()
        self.update([6.,4.,0.],11.3,velocity=(.1,0.));self.nav.tick()
        self.nav.route_visits[1]['handle']=dict(handle,call_id='wrong')
        for stamp in (11.4,11.7,12.):self.update([6.,4.,0.],stamp);state=self.nav.tick()
        self.assertFalse(state['manual_arrival_observation'])
        self.assertEqual(len(self.nav.route_visits),3)
        with self.assertRaisesRegex(ValueError,'final waypoint'):self.nav.confirm_arrival()

    def test_manual_reconfirmation_resumes_review_without_replacing_original_proof(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session,pose=self.prepared_arrival(store)
                original=copy.deepcopy(session.skill.record['completed'][-1])
                session.stop('needs_clarification','请核对到达状态')
                self.update(pose,10.4,velocity=(.1,0.));session.tick()
                self.update(pose,10.5);session.event('arrived')
                self.assertEqual(session.state['status'],'ready')
                result=session.call('check_goal',goal_id='elevator')
                self.assertEqual(result['status'],'completed')
                self.assertEqual(result['evidence_kind'],'operator_confirmation_and_localization_estimate')
                proof=session.task_goals.goals[0]['completion']['proof']
                self.assertEqual(proof['stamp_s'],original['stamp_s'])
                self.assertEqual(proof['current_manual_arrival_evidence']['stamp_s'],10.5)
                self.assertEqual(session.skill.record['completed'][-1],original)
            finally:store.close()

    def test_route_marker_heights_are_source_marked(self):
        grid=GridSnapshot(dict(frame_id='map',shape=[2,2],resolution=1.,origin_xy=[0.,0.],snapshot_id='static'),
            dict(occupancy=np.zeros((2,2),np.int8),ground_z=np.array([[2.,0.],[0.,4.]],np.float32),
                 ground_support=np.array([[True,False],[False,True]])))
        points,approx=display_points(grid,[[.5,.5],[1.5,.5],[9.,9.]])
        self.assertAlmostEqual(points[0][2],2.12,places=6)
        self.assertAlmostEqual(points[1][2],3.12,places=6)
        self.assertEqual(approx,[1,2])

    def test_manual_goal_heading_reaches_rviz_and_route_refresh_preserves_it(self):
        from decision.agent.workflows import manual_rviz
        single=[1.,2.,math.pi/2]
        route=dict(waypoints=[[1.,2.],[3.,4.]],final_yaw=-math.pi/2)
        self.assertEqual(display_final_yaw(single,None),math.pi/2)
        self.assertEqual(display_final_yaw(single,route),-math.pi/2)
        self.assertIsNone(display_final_yaw(single,dict(route,final_yaw=None)))
        display=RouteDisplay('map');display.process=MagicMock()
        display.send([[1.,2.,.3]],None,display_final_yaw(single,None))
        display.send([[1.,2.,.3],[3.,4.,.5]],1,display_final_yaw(single,route))
        display.send([[1.,2.,.3],[3.,4.,.5]],1,None)
        lines=''.join(call.args[0] for call in display.process.stdin.write.call_args_list)

        class Marker:
            ARROW=0;SPHERE=2;LINE_LIST=5;TEXT_VIEW_FACING=9;ADD=0;DELETEALL=3
            def __init__(self):
                self.header=SimpleNamespace(frame_id='')
                self.pose=SimpleNamespace(position=None,orientation=SimpleNamespace(x=0.,y=0.,z=0.,w=0.))
                self.scale=SimpleNamespace(x=0.,y=0.,z=0.)
                self.color=SimpleNamespace(r=0.,g=0.,b=0.,a=0.)
                self.points=[]
        published=[]
        publisher=SimpleNamespace(publish=published.append)
        rospy=SimpleNamespace(init_node=lambda *a,**k:None,Publisher=lambda *a,**k:publisher,
                              Timer=lambda *a,**k:SimpleNamespace(shutdown=lambda:None),Duration=lambda x:x)
        point=lambda x,y,z:SimpleNamespace(x=x,y=y,z=z)
        modules={'rospy':rospy,'geometry_msgs':SimpleNamespace(msg=SimpleNamespace(Point=point)),
                 'geometry_msgs.msg':SimpleNamespace(Point=point),
                 'visualization_msgs':SimpleNamespace(msg=SimpleNamespace(Marker=Marker,MarkerArray=None)),
                 'visualization_msgs.msg':SimpleNamespace(Marker=Marker,MarkerArray=lambda markers:SimpleNamespace(markers=markers))}
        with patch.dict('sys.modules',modules),patch('sys.stdin',io.StringIO(lines)):
            manual_rviz.main()
        arrows=[[m for m in msg.markers if getattr(m,'ns',None)=='manual_goal_heading'] for msg in published[:3]]
        self.assertEqual([len(items) for items in arrows],[1,1,0])
        self.assertAlmostEqual(arrows[0][0].pose.orientation.z,math.sin(math.pi/4))
        self.assertAlmostEqual(arrows[1][0].pose.orientation.z,math.sin(-math.pi/4))
        self.assertEqual((arrows[1][0].pose.position.x,arrows[1][0].pose.position.y),(3.,4.))
        self.assertAlmostEqual(arrows[1][0].pose.position.z,.62)
        self.assertIsNone(manual_rviz.heading_geometry([[1.,2.,None]],0.))

    def test_manual_client_refuses_runtime_with_hardware_output(self):
        client=ManualRuntimeClient('/unused')
        with patch('decision.agent.adapters.runtime.exchange_request') as exchange:
            for status in (dict(live=True,hardware_enabled=True),dict(live=False,hardware_enabled=False),{}):
                exchange.return_value=status
                with self.assertRaisesRegex(ValueError,'without chassis output'):
                    client.require_observation_only()
            exchange.return_value=dict(live=True,hardware_enabled=False)
            client.require_observation_only()
            exchange.assert_called_with(client.path,'session_status',{},timeout=.5)

    def test_confirmation_requires_current_position_heading_and_stopped_estimate(self):
        handle=dict(task_id='task',goal_revision=1,call_id='first')
        self.nav.start(dict(place_id='elevator',pose=[5.,4.,0.],requirement=dict(required_yaw=0.)),handle)
        with self.assertRaisesRegex(ValueError,'Outside target tolerance'):self.nav.confirm_arrival()
        self.update([5.,4.,0.],10.1,velocity=(.2,0.))
        with self.assertRaisesRegex(ValueError,'stopped estimate'):self.nav.confirm_arrival()
        self.update([5.,4.,.3],10.2)
        with self.assertRaisesRegex(ValueError,'Outside target tolerance'):self.nav.confirm_arrival()
        self.update([4.84,4.,0.],10.3)
        self.nav.confirm_arrival()
        self.assertTrue(self.nav.snapshot()['manual_arrival_observation'])
        self.update([5.5,4.,0.],11.)
        self.assertFalse(self.nav.snapshot()['manual_arrival_observation'])

    def prepared_arrival(self, store):
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        session=AgentSession('到电梯',self.catalog,self.nav,store)
        session.call('record_task',goals=[point_goal('elevator')])
        session.call('query_map')
        pose=self.catalog.places['elevator']['pose']
        result=session.call('navigate',goal_id='elevator',pose=pose,
            view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')
        self.assertEqual(result['status'],'accepted',result)
        self.update(pose,10.2);session.tick();session.event('arrived')
        self.update(pose,10.3);session.tick()
        return session,pose

    def test_goal_check_waits_for_fresh_manual_observation_without_another_tool_call(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session,pose=self.prepared_arrival(store)
                self.obs['stamp_s']=9.
                original_tick=self.nav.tick
                def refresh():
                    self.update(pose,10.4)
                    return original_tick()
                with patch.object(self.nav,'tick',side_effect=refresh) as tick:
                    result=session.call('check_goal',goal_id='elevator')
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(tick.call_count,1)
                event=json.loads(store.db.execute("SELECT value FROM events WHERE kind='manual_goal_evidence_wait_finished'").fetchone()[0])
                self.assertTrue(event['ready'])
                self.assertEqual(event['attempts'],1)
            finally:store.close()

    def test_goal_check_timeout_reports_missing_evidence_and_never_marks_completion(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session,_=self.prepared_arrival(store)
                self.obs['stamp_s']=9.
                with patch('decision.agent.workflows.scripted_review.time.monotonic',side_effect=[0.,0.,0.,2.,2.]), \
                     patch('decision.agent.workflows.scripted_review.time.sleep'):
                    result=session.call('check_goal',goal_id='elevator')
                self.assertEqual(result['status'],'unproven',result)
                self.assertIn('mapped_pose_unavailable',result['details'])
                self.assertIn('mapped_pose_out_of_date',result['details'])
                self.assertNotEqual(session.task_goals.goals[0]['status'],'completed')
                event=json.loads(store.db.execute("SELECT value FROM events WHERE kind='manual_goal_evidence_unproven'").fetchone()[0])
                self.assertEqual(event['issues'],result['details'])
            finally:store.close()

    def test_goal_check_waits_for_fresh_stopped_estimate_without_forgetting_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session,pose=self.prepared_arrival(store)
                self.feedback['stamp_s']=9.
                snapshot=self.nav.snapshot()
                self.assertFalse(snapshot['stopped'])
                self.assertIsNotNone(snapshot['manual_confirmation'])
                original_tick=self.nav.tick
                def refresh():
                    self.update(pose,10.4)
                    return original_tick()
                with patch.object(self.nav,'tick',side_effect=refresh) as tick:
                    result=session.call('check_goal',goal_id='elevator')
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(tick.call_count,1)
            finally:store.close()

    def test_goal_check_never_reuses_confirmation_after_movement(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session,pose=self.prepared_arrival(store)
                self.obs['stamp_s']=9.
                original_tick=self.nav.tick
                def moved():
                    self.update(pose,10.4,velocity=(.2,0.))
                    return original_tick()
                with patch.object(self.nav,'tick',side_effect=moved) as tick:
                    result=session.call('check_goal',goal_id='elevator')
                self.assertEqual(result['status'],'unproven',result)
                self.assertIn('manual_arrival_evidence_missing_or_mismatched',result['details'])
                self.assertIn('stopped_estimate_unavailable',result['details'])
                self.assertEqual(tick.call_count,1)
                self.assertNotEqual(session.task_goals.goals[0]['status'],'completed')
            finally:store.close()

    def test_two_goals_require_independent_operator_confirmations(self):
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('two stops',self.catalog,self.nav,store)
                session.call('record_task',goals=[point_goal('elevator'),point_goal('lobby_entrance')])
                session.call('query_map')
                pose=self.catalog.places['elevator']['pose']
                result=session.call('navigate',goal_id='elevator',pose=pose,
                    view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')
                self.assertEqual(result['status'],'accepted',result)
                pose=[pose[0],pose[1],pose[2]+.4]
                self.update(pose,10.2)
                session.tick()
                self.assertEqual(session.skill.record['completed'],[])
                session.event('arrived')
                self.update(pose,10.3);session.tick()
                self.assertEqual(session.call('check_goal',goal_id='elevator')['status'],'completed')
                session.call('query_map')
                pose2=self.catalog.places['lobby_entrance']['pose']
                result=session.call('navigate',goal_id='lobby_entrance',pose=pose2,
                    view_id=session.spatial_view.record['view_id'],orientation_source='autonomous')
                self.assertEqual(result['status'],'accepted',result)
                self.update(pose2,10.5);session.tick()
                self.assertEqual(len(session.skill.record['completed']),1)
                session.event('arrived')
                self.update(pose2,10.6);session.tick()
                self.assertEqual(session.call('check_goal',goal_id='lobby_entrance')['status'],'completed')
                self.assertEqual(session.call('finish_task',status='succeeded',evidence='manual test')['status'],'succeeded')
                self.assertEqual(session.skill.record['completed'][-1]['evidence_kind'],
                                 'operator_confirmation_and_localization_estimate')
            finally:store.close()

    def test_manual_circuit_requires_measured_route_and_operator_confirmation(self):
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[200,200],resolution=.1,origin_xy=[-5.,-5.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((200,200),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        polygon=self.catalog.regions['lobby_area']['polygon'];center=centroid(polygon)
        radius=max(np.linalg.norm(np.asarray(v)-center) for v in polygon)+1
        points=[[float(center[0]+radius*math.cos(-2*math.pi*i/8)),
                 float(center[1]+radius*math.sin(-2*math.pi*i/8))] for i in range(9)]
        self.update([*points[0],0.],10.)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('绕一圈',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                    description='绕一圈')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='loop',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=points,final_yaw=0.,waypoint_reason='合成测试需要逐段验证')
                self.assertEqual(result['status'],'accepted',result)
                with self.assertRaisesRegex(ValueError,'intermediate waypoints'):session.event('arrived')
                for i,point in enumerate(points):
                    self.update([*point,0.],10.2+i*.2)
                    session.tick()
                self.assertEqual(self.nav.route_index,len(points)-1)
                self.assertEqual(session.skill.record['completed'],[])
                session.event('arrived')
                self.update([*points[-1],0.],12.1);session.tick()
                outcome=session.call('check_goal',goal_id='loop')
                self.assertEqual(outcome['status'],'completed',outcome)
                self.assertEqual(outcome['checks']['route_progress']['status'],'met')
                self.assertEqual(outcome['checks']['route_progress']['source'],'ordered_measured_waypoint_visits_not_semantic_completion')
                self.assertFalse(outcome['real_robot_success'])
            finally:store.close()

    def test_manual_circuit_automatically_completes_from_measured_trace(self):
        self.nav.automatic_arrival=True
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[200,200],resolution=.1,origin_xy=[-5.,-5.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((200,200),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        polygon=self.catalog.regions['lobby_area']['polygon'];center=centroid(polygon)
        radius=max(np.linalg.norm(np.asarray(v)-center) for v in polygon)+1
        points=[[float(center[0]+radius*math.cos(-2*math.pi*i/8)),
                 float(center[1]+radius*math.sin(-2*math.pi*i/8))] for i in range(9)]
        self.update([*points[0],0.],10.)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('绕一圈',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                    description='绕一圈')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='loop',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=points,final_yaw=0.,waypoint_reason='合成测试需要逐段验证')
                self.assertEqual(result['status'],'accepted',result)
                for i,point in enumerate(points):
                    self.update([*point,0.],10.2+i*.2);session.tick()
                for stamp in (12.,12.2,12.4):
                    self.update([*points[-1],0.],stamp);session.tick()
                outcome=session.call('check_goal',goal_id='loop')
                self.assertEqual(outcome['status'],'completed',outcome)
                self.assertEqual(outcome['checks']['route_progress']['status'],'met')
                self.assertEqual(outcome['checks']['route_progress']['source'],'ordered_measured_waypoint_visits_not_semantic_completion')
                self.assertEqual(outcome['evidence_kind'],'automatic_localization_and_stop_estimate')
            finally:store.close()

    def test_manual_observation_requires_fresh_stopped_estimate(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('观察门',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='inspect',kind='skill',stage_type='observation',
                    target_id='',description='观察门')])
                self.update([4.,4.,0.],10.1,velocity=(.2,0.))
                self.assertEqual(session.call('check_goal',goal_id='inspect')['reason'],'observation_requires_confirmed_stop')
                self.update([4.,4.,0.],10.2)
                self.assertEqual(session.call('check_goal',goal_id='inspect')['reason'],'read_current_rgb_after_stage_entry')
                self.feedback['stamp_s']=9.
                self.assertEqual(session.call('check_goal',goal_id='inspect')['reason'],'observation_requires_confirmed_stop')
            finally:store.close()

    def test_manual_sequence_uses_measured_order(self):
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[100,140],resolution=.1,origin_xy=[0.,0.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((100,140),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('依次经过',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='visit',kind='sequence',target_id='',
                    checkpoints=['elevator','lobby_entrance'],description='依次经过')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='visit',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=[[5.,4.],[6.,4.]],final_yaw=0.)
                self.assertEqual(result['status'],'accepted',result)
                for stamp,pose in [(10.2,[5.,4.,0.]),(10.4,[5.5,4.,0.]),(10.6,[6.,4.,0.])]:
                    self.update(pose,stamp);session.tick()
                session.event('arrived')
                self.update([6.,4.,0.],10.8);session.tick()
                outcome=session.call('check_goal',goal_id='visit')
                self.assertEqual(outcome['status'],'completed',outcome)
                self.assertEqual(outcome['checks']['route_progress']['waypoint_count'],2)
            finally:store.close()

    def test_manual_observation_completion_allows_finish(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('观察门',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='inspect',kind='skill',stage_type='observation',
                    target_id='',description='观察门')])
                self.update([4.,4.,0.],10.2)
                self.control['feedback']['pose']=[4.,4.,0.]
                session.observation=SimpleNamespace(record=dict(observation_id='frame',clock_domain='runtime_live',
                    rgb=dict(stamp_s=10.1),base_pose=[4.,4.,0.],input_fingerprint='source'))
                session._applying_context=dict(observation_id='frame')
                with patch.object(session,'_current_rgb_context',return_value=True):
                    from decision.agent.workflows.scripted_review import _check_observation_stage
                    result=_check_observation_stage(session,'inspect',session.task_goals.current('inspect'),
                        self.nav.snapshot(),'frame','satisfied','看到门')
                self.assertEqual(result['status'],'completed',result)
                self.assertEqual(session.finish_task('succeeded','观察完成')['status'],'succeeded')
            finally:store.close()

    def test_incomplete_manual_circuit_does_not_complete(self):
        grid=GridSnapshot(dict(frame_id='sim_odom',shape=[200,200],resolution=.1,origin_xy=[-5.,-5.],snapshot_id='unit-static'),
                          dict(occupancy=np.zeros((200,200),np.int8)))
        self.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,stamp_s=self.control['stamp_s'],
            clock_domain='runtime_live',pose=list(self.obs['map_base_pose']),geometry=self.nav.geometry)
        polygon=self.catalog.regions['lobby_area']['polygon'];center=centroid(polygon)
        radius=max(np.linalg.norm(np.asarray(v)-center) for v in polygon)+1
        points=[[float(center[0]+radius*math.cos(-2*math.pi*i/8)),
                 float(center[1]+radius*math.sin(-2*math.pi*i/8))] for i in range(9)]
        self.update([*points[0],0.],10.)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('绕一圈',self.catalog,self.nav,store)
                session.call('record_task',goals=[dict(goal_id='loop',kind='route',target_id='lobby_area',
                    description='绕一圈')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='loop',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=points,final_yaw=0.,waypoint_reason='合成测试需要逐段验证')
                self.assertEqual(result['status'],'accepted',result)
                for i,point in enumerate([points[0],points[1],points[0]]):
                    self.update([*point,0.],10.2+i*.2);session.tick()
                with self.assertRaisesRegex(ValueError,'intermediate waypoints'):session.event('arrived')
                self.assertEqual(session.skill.record['status'],'running')
                self.assertEqual(session.skill.record['completed'],[])
            finally:store.close()


if __name__=='__main__':unittest.main()
