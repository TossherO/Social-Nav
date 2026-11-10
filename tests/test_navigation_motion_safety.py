"""Finite-horizon execution checks and their limitations, with synthetic inputs."""
import unittest
import numpy as np
from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry
from navigation.safety.guard import LocalSafety, predicted_trajectory
from navigation.safety.config import load_safety_config
from perception.environment.context import fingerprint
from tests.navigation_fixtures import bundle, request


class MotionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.motion=load_motion_config(profile='trial')
        self.motion.update(parameters_assumed=False,collision_backend='cpu',collision_library=None,
            execution_max_longitudinal_speed_mps=.8,execution_max_angular_rps=.4)
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),
            source='synthetic',assumed=False)
        self.feedback=dict(status='valid',source='synthetic',pose=[2.,2.,0.],velocity=[0.,0.],
            stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'])
        self.req=request('local')

    def gate(self,scene):
        guard=LocalSafety(self.geometry,self.motion,.05,execution=True)
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.2,0.],
            request_binding=fingerprint(self.req),scene_binding=scene.record['input_fingerprint'],
            trajectory_as_of=10.,trajectory_binding='synthetic')
        return guard.gate(proposal,self.req,scene,self.feedback,now=10.,clock_domain='synthetic',dt=.1)

    def test_free_prediction_is_allowed_and_obstacle_is_rejected(self):
        result=self.gate(bundle())
        self.assertEqual(result['state'],'allowed',result)
        self.assertAlmostEqual(result['motion_sweep']['points'][-1][0],load_safety_config()['prediction_s'])
        cells=np.zeros((60,60),np.int8);cells[20,24]=100
        blocked=self.gate(bundle(cells))
        self.assertEqual(blocked['reason'],'motion_sweep_occupied')
        self.assertEqual(blocked['velocity'],[0.,0.])

    def test_execution_requires_feedback_bounds(self):
        for key in ('execution_max_longitudinal_speed_mps','execution_max_angular_rps'):
            motion=dict(self.motion);motion.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):
                LocalSafety(self.geometry,motion,.05,execution=True)

    def test_execution_rejects_feedback_over_speed_bound(self):
        self.feedback['velocity']=[.81,0.]
        result=self.gate(bundle())
        self.assertEqual(result['reason'],'feedback_exceeds_execution_speed_bound')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_prediction_does_not_claim_a_complete_stop(self):
        settings=load_safety_config()
        path=predicted_trajectory(self.feedback,[.2,0.],self.motion,settings)
        self.assertAlmostEqual(path[-1,0],settings['prediction_s'])
        self.assertGreater(path[-1,1],path[0,1])


if __name__=='__main__':unittest.main()
