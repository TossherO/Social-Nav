"""Operator-driven Agent checks use synthetic observations only."""
import copy
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
from decision.agent.workflows.manual import display_points


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

    def test_route_marker_heights_are_source_marked(self):
        grid=GridSnapshot(dict(frame_id='map',shape=[2,2],resolution=1.,origin_xy=[0.,0.],snapshot_id='static'),
            dict(occupancy=np.zeros((2,2),np.int8),ground_z=np.array([[2.,0.],[0.,4.]],np.float32),
                 ground_support=np.array([[True,False],[False,True]])))
        points,approx=display_points(grid,[[.5,.5],[1.5,.5],[9.,9.]])
        self.assertAlmostEqual(points[0][2],2.12,places=6)
        self.assertAlmostEqual(points[1][2],3.12,places=6)
        self.assertEqual(approx,[1,2])

    def test_manual_client_refuses_runtime_with_hardware_output(self):
        client=ManualRuntimeClient('/unused')
        wire=MagicMock()
        with patch('decision.agent.adapters.runtime.socket.socket') as sock, \
                patch('decision.agent.adapters.runtime.Connection',return_value=wire):
            for status in (dict(live=True,hardware_enabled=True),dict(live=False,hardware_enabled=False),{}):
                wire.recv.return_value=status
                with self.assertRaisesRegex(ValueError,'without chassis output'):
                    client.require_observation_only()
            wire.recv.return_value=dict(live=True,hardware_enabled=False)
            client.require_observation_only()
            wire.send.assert_called_with(dict(kind='session_status',value={}))

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
                with patch('decision.agent.core.session.time.monotonic',side_effect=[0.,0.,0.,2.,2.]), \
                     patch('decision.agent.core.session.time.sleep'):
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
                self.assertIn('operator_confirmation_missing_or_mismatched',result['details'])
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
                session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                    direction='clockwise',description='绕一圈')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='loop',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=points,final_yaw=0.)
                self.assertEqual(result['status'],'accepted',result)
                with self.assertRaisesRegex(ValueError,'start has not been observed'):session.event('arrived')
                for i,point in enumerate(points):
                    self.update([*point,0.],10.2+i*.2)
                    session.tick()
                self.assertEqual(self.nav.route_index,len(points)-1)
                self.assertEqual(session.skill.record['completed'],[])
                session.event('arrived')
                self.update([*points[-1],0.],12.1);session.tick()
                outcome=session.call('check_goal',goal_id='loop')
                self.assertEqual(outcome['status'],'completed',outcome)
                self.assertTrue(outcome['checks']['circuit_met'])
                self.assertEqual(outcome['checks']['circuit_evidence']['source'],'measured_localization_not_planned_route')
                self.assertFalse(outcome['real_robot_success'])
                row=store.db.execute("SELECT value FROM events WHERE kind='manual_route_trace'").fetchone()
                saved=json.loads(row[0])
                self.assertEqual(saved['result']['status'],'met')
                self.assertGreaterEqual(len(saved['segments'][0]['samples']),3)
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
                self.assertEqual([m['checkpoint'] for m in outcome['checks']['sequence_evidence']['matched']],
                                 ['elevator','lobby_entrance'])
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
                    result=session._check_observation_stage('inspect',session.task_goals.current('inspect'),
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
                session.call('record_task',goals=[dict(goal_id='loop',kind='circuit',target_id='lobby_area',
                    direction='clockwise',description='绕一圈')])
                session.call('query_map')
                result=session.call('navigate_route',goal_id='loop',view_id=session.spatial_view.record['view_id'],
                    orientation_source='autonomous',waypoints=points,final_yaw=0.)
                self.assertEqual(result['status'],'accepted',result)
                for i,point in enumerate([points[0],points[1],points[0]]):
                    self.update([*point,0.],10.2+i*.2);session.tick()
                with self.assertRaisesRegex(ValueError,'Measured route not proven'):session.event('arrived')
                self.assertEqual(session.skill.record['status'],'running')
                self.assertEqual(session.skill.record['completed'],[])
            finally:store.close()


if __name__=='__main__':unittest.main()
