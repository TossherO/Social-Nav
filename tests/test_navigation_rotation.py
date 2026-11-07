"""Rotation references, bounded control and independent swept-footprint checks."""
import unittest
import numpy as np
from navigation.planning.planner import Planner
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry,swept_check,footprint_status
from navigation.planning.continuity import reference_envelope
from navigation.control.dwa import DwaController
from navigation.safety.guard import LocalSafety
from navigation.kinematics import braking_path
from tests.test_navigation_planning import bundle,request


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config();self.cfg['algorithm']='hybrid_astar'
        self.motion=load_motion_config()
        self.geometry=VehicleGeometry([[-.45,-.2],[.45,-.2],[.45,.2],[-.45,.2]],np.eye(4),source='synthetic')
        self.scene=bundle();self.req=request('local',goal=[2.,2.,np.pi/2],goal_behavior='stop')
        self.feedback=dict(status='valid',source='synthetic_plant',pose=[2.,2.,0.],velocity=[0.,0.],stamp_s=10.,
                           clock_domain='synthetic',local_identity=['test',0,'odom'])

    def reference(self):
        result=Planner(self.cfg,self.geometry,require_actual_start=True).query(self.req,scene=self.scene,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        return reference_envelope(result,self.req,self.scene,None)

    def test_native_same_position_goal_rotates_without_translation(self):
        ref=self.reference();path=np.asarray(ref['result']['path'])
        np.testing.assert_allclose(path[:,:2],np.broadcast_to([2.,2.],path[:,:2].shape))
        self.assertAlmostEqual(path[-1,2],np.pi/2)
        self.assertEqual(swept_check(self.scene.geometry,self.geometry,path,.05,10000)['status'],'free')

    def test_dwa_and_safety_allow_bounded_zero_linear_command(self):
        controller=DwaController(self.cfg,self.motion,self.geometry)
        result=controller.command(self.req,self.scene,self.feedback,self.reference(),now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'candidate',result)
        self.assertEqual(result['velocity'][0],0.);self.assertGreater(result['velocity'][1],0.)
        self.assertLessEqual(result['velocity'][1],self.motion['max_angular_acceleration_rps2']*.1+1e-9)
        gate=LocalSafety(self.geometry,self.motion,.05,simulation=True)
        allowed=gate.gate(result,self.req,self.scene,self.feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(allowed['state'],'allowed',allowed)
        self.assertFalse(allowed['executable'])

    def test_unknown_between_free_endpoint_footprints_blocks_rotation(self):
        cells=np.zeros((60,60),np.int8);cells[23,23]=-1
        scene=bundle(cells);a=[2.,2.,0.];b=[2.,2.,np.pi/2]
        self.assertEqual(footprint_status(scene.geometry,self.geometry,a,0.),'free')
        self.assertEqual(footprint_status(scene.geometry,self.geometry,b,0.),'free')
        check=swept_check(scene.geometry,self.geometry,np.array([a,b]),0.,10000)
        self.assertEqual(check['status'],'unknown',check)

    def test_rotation_braking_has_no_translation_and_stops(self):
        poses=braking_path(np.array([2.,2.,0.]),np.array([0.,.3]),self.motion,.2,timed=True)
        np.testing.assert_allclose(poses[:,1:3],np.broadcast_to([2.,2.],poses[:,1:3].shape))
        expected=.3*.2+.3**2/(2*self.motion['max_angular_acceleration_rps2'])
        self.assertAlmostEqual(poses[-1,3],expected)

    def test_delayed_feedback_simulation_reaches_yaw_then_stops(self):
        from navigation.simulation import run_case
        result=run_case('rotation',self.cfg,self.motion,self.geometry,max_s=15.,plant_delay_s=.2,plant_acceleration_scale=.75)
        self.assertTrue(result['completed'],result['safety_counts'])
        self.assertFalse(result['collision'])
        np.testing.assert_allclose(result['final_pose'][:2],[2.,2.],atol=1e-10)

    def test_long_rotation_preserves_route_direction(self):
        import copy
        from perception.environment.context import fingerprint
        self.req=dict(self.req,goal=[2.,2.,-np.pi/2]);req=self.req
        result=copy.deepcopy(self.reference()['result'])
        angles=np.linspace(0,3*np.pi/2,61)
        result['path']=np.c_[np.full(len(angles),2.),np.full(len(angles),2.),angles].tolist()
        result.pop('result_binding');result['result_binding']=fingerprint(result)
        ref=reference_envelope(result,req,self.scene,None)
        controller=DwaController(self.cfg,self.motion,self.geometry)
        output=controller.command(req,self.scene,self.feedback,ref,now=10.,clock_domain='synthetic')
        self.assertEqual(output['status'],'candidate',output)
        self.assertGreater(output['velocity'][1],0.)

    def test_gpu_rotation_matches_cpu_decision_and_trajectory(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        from perception.environment.device import DeviceGridSnapshot
        from navigation.control.device import DeviceDwaBatch
        ref=self.reference();cpu=DwaController(self.cfg,self.motion,self.geometry)
        batch=DeviceDwaBatch();batch.bind(DeviceGridSnapshot.seal(self.scene.geometry.metadata,
            {k:torch.tensor(v,device='cuda') for k,v in self.scene.geometry.layers.items()}))
        gpu=DwaController(self.cfg,self.motion,self.geometry,device_batch=batch)
        a=cpu.command(self.req,self.scene,self.feedback,ref,now=10.,clock_domain='synthetic')
        b=gpu.command(self.req,self.scene,self.feedback,ref,now=10.,clock_domain='synthetic')
        self.assertEqual(a['status'],'candidate',a);self.assertEqual(b['status'],'candidate',b)
        np.testing.assert_allclose(a['velocity'],b['velocity'],atol=1e-10)
        np.testing.assert_allclose(cpu.result['trajectory']['points'],gpu.result['trajectory']['points'],atol=1e-9)


if __name__=='__main__':unittest.main()
