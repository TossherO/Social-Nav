"""Reverse reference, signed control, direction changes and independent stopping."""
import copy
import unittest
import numpy as np

from navigation.kinematics import velocity_step,turning_residual
from navigation.control.dwa import DwaController
from navigation.planning.continuity import reference_envelope
from navigation.planning.planner import Planner
from navigation.safety.guard import LocalSafety
from perception.environment.context import fingerprint
from tests import test_navigation_dwa as fixtures
from tests.navigation_fixtures import bundle


class ReverseTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.DwaTests();self.f.setUp()
        self.motion=dict(self.f.motion,allow_reverse=True)
        self.cfg=copy.deepcopy(self.f.cfg);self.cfg['motion_constraints']['allow_reverse']=True
        self.request=dict(self.f.req,goal=[1.,2.,0.])
        self.reference=self.reference_for(self.request)

    def reference_for(self,request):
        result=Planner(self.cfg,self.f.geometry,require_actual_start=True).query(
            request,scene=self.f.scene,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        return reference_envelope(result,request,self.f.scene,None)

    def test_reverse_reference_drives_backwards_with_bounded_speed(self):
        self.assertLess(self.reference['result']['path_length_m'],1.01)
        controller=DwaController(self.cfg,self.motion,self.f.geometry)
        for v in (0.,-.1,-.25):
            feedback=dict(self.f.feedback,velocity=[v,0.])
            result=controller.command(self.request,self.f.scene,feedback,self.reference,now=10.,clock_domain='synthetic')
            self.assertEqual(result['status'],'candidate',result)
            self.assertLess(result['velocity'][0],0.)
            self.assertGreaterEqual(result['velocity'][0],-self.motion['max_reverse_mps'])
            self.assertEqual(result['reference_motion']['direction'],-1)
            self.assertEqual(feedback['velocity'],[v,0.])

    def test_forward_only_policy_rejects_reverse_reference(self):
        controller=DwaController(self.cfg,dict(self.motion,allow_reverse=False),self.f.geometry)
        result=controller.command(self.request,self.f.scene,self.f.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(result['reason'],'reference_requires_reverse')

    def test_opposite_cones_brake_to_zero_before_reversing_scalar_and_batch(self):
        initial=np.array([[.1,.08],[.01,-.005],[-.1,.08],[-.01,-.005]])
        targets=np.array([[-.15,.08],[-.15,.08],[.3,.08],[.3,.08]])
        velocity=initial.copy();stopped=np.zeros(4,bool)
        for _ in range(30):
            after=velocity_step(velocity,targets,self.motion,.1)
            scalar=np.stack([velocity_step(v,t,self.motion,.1) for v,t in zip(velocity,targets)])
            np.testing.assert_allclose(after,scalar,atol=1e-14)
            self.assertTrue(np.all(turning_residual(after,self.motion)>=-1e-10))
            self.assertTrue(np.all(abs(after-velocity)<=.1*np.array([
                self.motion['max_acceleration_mps2'],self.motion['max_angular_acceleration_rps2']])+1e-12))
            self.assertTrue(np.all((after[:,0]*initial[:,0]>=0)|stopped))
            stopped|=(abs(after)<1e-12).all(1);velocity=after
        self.assertTrue(stopped.all());np.testing.assert_allclose(velocity,targets,atol=1e-12)

    def test_cusp_stops_before_tracking_reverse_section(self):
        request=dict(self.f.req,goal=[2.1,2.,0.]);reference=self.reference_for(request)
        result=copy.deepcopy(reference['result']);result['path']=[[2.,2.,0.],[2.5,2.,0.],[2.1,2.,0.]]
        result.pop('result_binding');result['result_binding']=fingerprint(result)
        reference=reference_envelope(result,request,self.f.scene,None)
        controller=DwaController(self.cfg,self.motion,self.f.geometry)
        a=controller.command(request,self.f.scene,self.f.feedback,reference,now=10.,clock_domain='synthetic')
        self.assertEqual(a['reference_motion'],dict(direction=1,section=0,stop_at_cusp=True))
        moving=dict(self.f.feedback,pose=[2.5,2.,0.],velocity=[.1,0.])
        b=controller.command(request,self.f.scene,moving,reference,now=10.,clock_domain='synthetic')
        self.assertEqual(b['status'],'candidate',b);self.assertEqual(b['velocity'],[0.,0.])
        self.assertTrue(b['reference_motion']['stop_at_cusp'])
        c=controller.command(request,self.f.scene,dict(moving,velocity=[0.,0.]),reference,now=10.,clock_domain='synthetic')
        self.assertEqual(c['status'],'candidate',c);self.assertLess(c['velocity'][0],0.)
        self.assertEqual(c['reference_motion'],dict(direction=-1,section=1,stop_at_cusp=False))

    def test_independent_safety_requires_rear_free_stopping_space(self):
        feedback=dict(self.f.feedback,velocity=[-.2,0.])
        for value in (0,-1,100):
            cells=np.zeros((60,60),np.int8);cells[18:23,16]=value;scene=bundle(cells)
            guard=LocalSafety(self.f.geometry,self.motion,self.cfg['safety_margin_m'],shadow=True)
            proposal=dict(status='candidate',control_mode='dwa',velocity=[-.2,0.],trajectory_as_of=10.,
                request_binding=fingerprint(self.request),scene_binding=scene.record['input_fingerprint'],trajectory_binding='synthetic')
            result=guard.gate(proposal,self.request,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
            self.assertEqual(result['state'],'allowed' if value==0 else 'stop_request',result)
            if value:self.assertIn(result['reason'],('stopping_sweep_unknown','stopping_sweep_occupied'))

    def test_reverse_gpu_matches_cpu_including_opposite_pending_command(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        from perception.environment.device import DeviceGridSnapshot
        from navigation.control.device import DeviceDwaBatch
        batch=DeviceDwaBatch();batch.bind(DeviceGridSnapshot.seal(self.f.scene.geometry.metadata,
            {k:torch.tensor(v,device='cuda') for k,v in self.f.scene.geometry.layers.items()}))
        for velocity,pending in (([0.,0.],None),([-.15,-.08],None),([.01,.005],[-.15,.08])):
            feedback=dict(self.f.feedback,velocity=velocity)
            cpu=DwaController(self.cfg,self.motion,self.f.geometry)
            gpu=DwaController(self.cfg,self.motion,self.f.geometry,device_batch=batch)
            a=cpu.command(self.request,self.f.scene,feedback,self.reference,now=10.,clock_domain='synthetic',pending=pending)
            with torch.cuda.stream(torch.cuda.Stream(priority=-1)):
                b=gpu.command(self.request,self.f.scene,feedback,self.reference,now=10.,clock_domain='synthetic',pending=pending)
            self.assertEqual(a['status'],'candidate',a);self.assertEqual(a['status'],b['status'],b)
            np.testing.assert_allclose(a['velocity'],b['velocity'],atol=1e-10)
            np.testing.assert_allclose(cpu.result['trajectory']['points'],gpu.result['trajectory']['points'],atol=1e-10)


if __name__=='__main__':unittest.main()
