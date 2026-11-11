"""Operator-driven Agent checks use synthetic observations only."""
import copy
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


if __name__=='__main__':unittest.main()
