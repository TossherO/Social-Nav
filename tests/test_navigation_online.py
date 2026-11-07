"""Wire goals and ROS bridge revocation; synthetic geometry, no robot execution."""
import unittest
from unittest.mock import patch
import numpy as np
from perception.environment.layers import load_config as environment_config
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.reference import ReferenceRuntime,load_config as reference_config
from navigation.planning.online import PlanningBridge,goal_request
from tests.navigation_fixtures import FakeWorker,scene_at
from tests.test_navigation_planning import request


class OnlineTests(unittest.TestCase):
    def setUp(self):
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.runtime=ReferenceRuntime(load_config(),load_motion_config(),geometry,reference_config(),worker_factory=FakeWorker)
        self.bridge=PlanningBridge(self.runtime,geometry,load_motion_config(),environment_config()['scene'],clock_domain='synthetic')
        self.patch=patch('navigation.planning.online.snapshot_from_message',side_effect=lambda m:m);self.patch.start()
        self.req=request('local',goal=[4.,2.]);self.poses()

    def tearDown(self):self.runtime.close();self.patch.stop()

    def test_people_join_is_latest_past_and_late_delivery_does_not_advance_grid_cutoff(self):
        from tests.test_environment_scene import person
        from tests.test_navigation_planning import grid
        def source(stamp):
            value=person(stamp);value['localization_binding']=dict(session_id='test',generation=0,frame_id='odom')
            return value
        self.bridge.push_people(source(10.1));self.bridge.push_people(source(9.9));self.bridge.push_grid(grid())
        self.bridge.step(10.)
        scene=self.bridge.scene
        self.assertEqual(scene.record['people']['data']['stamp_s'],9.9)
        self.bridge.push_people(source(10.));self.bridge.step(10.)
        self.assertEqual(self.bridge.scene.record['as_of'],10.)
        self.assertEqual(self.bridge.scene.record['people']['data']['stamp_s'],10.)
        self.assertNotEqual(scene.record['input_fingerprint'],self.bridge.scene.record['input_fingerprint'])

    def poses(self,generation=0,health=0):
        for t in [9.7,9.8,9.9,10.]:
            self.bridge.push_state(dict(stamp_s=t,health=health,session_id='test',generation=generation,frame_id='odom',
                position=[2.,2.,0.],quaternion_xyzw=[0.,0.,0.,1.]))

    def submit(self):
        self.bridge.push_command('goal',self.req);self.bridge.push_grid(scene_at(10.).geometry)
        return self.bridge.step(10.)

    def test_geometry_only_scene_has_own_binding_and_real_motion_source(self):
        self.submit();item=self.runtime.latest
        self.assertIsNotNone(item);item['scene'].verify()
        self.assertEqual(item['scene'].record['scope'],'geometry_only_ros_planning')
        from navigation.control.shadow import observed_feedback
        feedback=observed_feedback(list(self.bridge.rows),item['scene'],self.bridge.geometry,self.bridge.motion,now=10.)
        self.assertEqual(feedback['status'],'valid');self.assertFalse(feedback['execution_feedback'])
        self.assertEqual(item['scene'].record['rgb']['status'],'missing')

    def test_cancel_is_bound_and_old_cancel_cannot_clear_new_revision(self):
        self.submit();self.bridge.push_command('goal',dict(self.req,goal_revision=1));self.bridge.step(10.)
        self.bridge.push_command('cancel',dict(task_id='t',goal_revision=True));status=self.bridge.step(10.)
        self.assertIsNotNone(status['command_error']);self.assertIsNotNone(self.bridge.request)
        self.bridge.push_command('cancel',dict(task_id='t',goal_revision=0));self.bridge.step(10.)
        self.assertEqual(self.bridge.request['goal_revision'],1)
        self.bridge.push_command('cancel',dict(task_id='t',goal_revision=1));status=self.bridge.step(10.)
        self.assertEqual(status['reason'],'cancel');self.assertIsNone(self.bridge.request)
        self.bridge.push_command('goal',dict(self.req,goal_revision=1));self.bridge.push_grid(scene_at(10.).geometry)
        self.assertIn('cancelled_goal_revision',self.bridge.step(10.)['reason'])

    def test_changed_goal_requires_revision_and_invalid_wire_preserves_current_goal(self):
        self.submit();self.bridge.push_command('goal',dict(self.req,goal=[3.,2.]));status=self.bridge.step(10.)
        self.assertIn('new goal_revision',status['command_error']);self.assertEqual(self.bridge.request,self.req)
        for value in [[],{},dict(self.req,goal=[float('nan'),0]),dict(self.req,goal_revision=True)]:
            with self.assertRaises((ValueError,TypeError)):goal_request(value)

    def test_invalid_pose_immediately_revokes_previous_scene(self):
        self.submit();self.bridge.push_state(dict(self.bridge.rows[-1],health=2,stamp_s=10.1))
        status=self.bridge.step(10.1);self.assertIsNone(self.runtime.latest)
        self.assertIsNone(status['result']);self.assertEqual(status['reason'],'current_localization_invalid')

    def test_reset_drops_goal_and_requires_explicit_rebinding(self):
        self.submit();self.poses(generation=1);self.bridge.step(10.)
        self.assertIsNone(self.bridge.request);self.assertIsNone(self.bridge.scene)

    def test_clock_seek_cannot_keep_old_local_coordinates(self):
        self.submit();status=self.bridge.step(5.)
        self.assertEqual(status['reason'],'seek');self.assertIsNone(self.bridge.request)

    def test_seek_after_cancel_resets_observation_order_and_new_goal_can_resume(self):
        self.submit();self.bridge.push_command('cancel',dict(task_id='t',goal_revision=0));self.bridge.step(10.)
        self.bridge.step(5.)
        self.poses();self.bridge.push_command('goal',dict(self.req,goal_revision=1))
        self.bridge.push_grid(scene_at(10.).geometry);self.bridge.step(10.)
        self.assertIsNotNone(self.runtime.latest)

    def test_cancel_before_first_grid_still_latches_the_target_revision(self):
        self.bridge.push_command('goal',self.req);self.bridge.step(10.)
        self.bridge.push_command('cancel',dict(task_id='t',goal_revision=0));self.bridge.step(10.)
        self.bridge.push_command('goal',self.req);self.bridge.push_grid(scene_at(10.).geometry)
        self.assertIn('cancelled_goal_revision',self.bridge.step(10.)['reason'])

    def test_malformed_grid_revokes_previous_result(self):
        self.submit()
        with patch('navigation.planning.online.snapshot_from_message',side_effect=ValueError('bad metadata')):
            self.bridge.push_grid(object());status=self.bridge.step(10.)
        self.assertIsNone(status['result']);self.assertIn('invalid_environment_grid',status['reason'])

    def test_callback_buffers_are_bounded_and_overflow_revokes_work(self):
        for i in range(400):self.bridge.push_state(dict(self.bridge.rows[-1],stamp_s=10.+i*.01))
        self.assertEqual(len(self.bridge.rows),256)
        for i in range(50):self.bridge.push_command('cancel',dict(task_id='other',goal_revision=i))
        self.assertLessEqual(len(self.bridge.commands),32)
        self.assertEqual(self.bridge.step(14.)['reason'],'command_queue_overflow')


if __name__=='__main__':unittest.main()
