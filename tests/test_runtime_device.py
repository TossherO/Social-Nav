"""Explicit CUDA parity for the device output boundary, not repeated pipeline tests."""
from pathlib import Path
import unittest
import numpy as np
from perception.environment.layers import LocalEnvironmentMap,load_config
from perception.environment.occupancy import OccupancyTree
from runtime.metrics import Metrics


class DeviceBoundaryTests(unittest.TestCase):

    def test_runtime_environment_library_uses_the_fixed_integer_height_projection(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        from tests.test_environment_projection import height_column
        library=Path(__file__).resolve().parents[1]/'outputs/runtime/build/environment/libsocial_nav_occupancy_cuda.so'
        for settings in (dict(),dict(missing=(8,)),dict(occupied=True)):
            expected,actual=height_column('cuda',library=library,**settings)
            np.testing.assert_array_equal(actual['occupancy'],expected['occupancy'])
            np.testing.assert_array_equal(actual['coverage'],expected['coverage'])
        _,complete=height_column('cuda',library=library)
        self.assertEqual(complete['coverage'][1,10],1.)

    def test_fused_navigation_keeps_raw_device_evidence_and_matches_cpu_control(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        from tests.test_environment_navigation import observation, prior_asset, scene
        from tests.test_navigation_dwa import DwaTests
        from perception.environment.navigation import NAVIGATION_DTYPES
        from perception.environment.support import NavigationSupport, finalize
        from tests.test_environment_support import points
        from perception.environment.device import DeviceGridSnapshot
        from navigation.control.device import DeviceDwaBatch
        from navigation.control.dwa import DwaController
        from navigation.planning.planner import Planner
        from navigation.planning.continuity import reference_envelope
        fixture=DwaTests();fixture.setUp();prior=prior_asset()
        cells=np.zeros((60,60),np.int8);cells[18:23,18:23]=-1
        config=load_config();fusion=NavigationSupport(config);device_fusion=NavigationSupport(config)
        fusion.ground.settings['ground_min_observations']=device_fusion.ground.settings['ground_min_observations']=1
        raw=observation(cells);fused=fusion.apply(raw,prior,observation=points(raw));current=scene(fused)
        original=DeviceGridSnapshot.seal(raw.metadata,{k:torch.tensor(v,device='cuda') for k,v in raw.layers.items()})
        derived=device_fusion.apply(original,prior,observation=points(raw))
        host=finalize(derived.materialize(consumer='test_navigation'))
        derived=DeviceGridSnapshot.seal(host.metadata,derived.consume())
        self.assertEqual(original.consume()['occupancy'][20,20].item(),-1)
        self.assertEqual(derived.consume()['navigation_occupancy'][20,20].item(),0)
        for name in NAVIGATION_DTYPES:
            np.testing.assert_array_equal(derived.materialize(consumer='test_navigation',fields=[name]).layers[name],fused.layers[name])
        result=Planner(fixture.cfg,fixture.geometry).query(fixture.req,scene=current,asset=prior,now=10.,clock_domain='synthetic')
        reference=reference_envelope(result,fixture.req,current,prior)
        cpu=DwaController(fixture.cfg,fixture.motion,fixture.geometry,asset=prior)
        batch=DeviceDwaBatch();batch.bind(derived)
        motion=dict(fixture.motion,collision_backend='cuda',collision_library=str(Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_collision_cuda.so'))
        gpu=DwaController(fixture.cfg,motion,fixture.geometry,asset=prior,device_batch=batch)
        a=cpu.command(fixture.req,current,fixture.feedback,reference,now=10.,clock_domain='synthetic')
        with torch.cuda.stream(torch.cuda.Stream(priority=-1)):
            b=gpu.command(fixture.req,current,fixture.feedback,reference,now=10.,clock_domain='synthetic')
        self.assertEqual(a['status'],'candidate',a);self.assertEqual(b['status'],a['status'],b)
        np.testing.assert_allclose(a['velocity'],b['velocity'],atol=1e-10)

    def test_dwa_device_batch_preserves_cpu_decisions_and_braking_samples(self):
        import torch
        from tests.test_navigation_dwa import DwaTests
        from perception.environment.device import DeviceGridSnapshot
        from navigation.control.device import DeviceDwaBatch
        from navigation.control.dwa import DwaController
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        fixture=DwaTests();fixture.setUp();metrics=Metrics()
        from tests.test_navigation_prediction import tracked_scene,track
        from navigation.planning.people import PeopleModel
        from navigation.planning.continuity import reference_envelope
        from navigation.planning.planner import Planner
        from perception.environment.context import fingerprint
        batch=DeviceDwaBatch(metrics=metrics)
        for velocity in ([0.,0.],[.4,.2],[.95,0.]):
            feedback=dict(fixture.feedback,velocity=velocity)
            snapshot=DeviceGridSnapshot.seal(fixture.scene.geometry.metadata,
                {k:torch.tensor(v,device='cuda') for k,v in fixture.scene.geometry.layers.items()})
            batch.bind(snapshot)
            cpu=DwaController(fixture.cfg,fixture.motion,fixture.geometry)
            motion=dict(fixture.motion,collision_backend='cuda',collision_library=str(Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_collision_cuda.so'))
            gpu=DwaController(fixture.cfg,motion,fixture.geometry,device_batch=batch)
            a=cpu.command(fixture.req,fixture.scene,feedback,fixture.reference,now=10.,clock_domain='synthetic')
            with torch.cuda.stream(torch.cuda.Stream(priority=-1)):
                b=gpu.command(fixture.req,fixture.scene,feedback,fixture.reference,now=10.,clock_domain='synthetic')
            self.assertEqual(b['status'],a['status'],b)
            np.testing.assert_allclose(b['velocity'],a['velocity'],atol=1e-10)
            if a['status']=='candidate':
                np.testing.assert_allclose(gpu.result['trajectory']['points'],cpu.result['trajectory']['points'],atol=1e-10)
        scene=tracked_scene([track(3.5,3.5,id='a'),track(4.5,3.5,id='b')])
        model=PeopleModel(scene,fixture.cfg['people'])
        social=dict(source='synthetic',stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'],groups=[[list(k) for k in model.ids]])
        social['binding']=fingerprint(social);request=dict(fixture.req,social=social)
        reference=reference_envelope(Planner(fixture.cfg,fixture.geometry).query(request,scene=scene,now=10.,clock_domain='synthetic'),request,scene,None)
        snapshot=DeviceGridSnapshot.seal(scene.geometry.metadata,{k:torch.tensor(v,device='cuda') for k,v in scene.geometry.layers.items()})
        batch.bind(snapshot)
        cpu=DwaController(fixture.cfg,fixture.motion,fixture.geometry)
        gpu=DwaController(fixture.cfg,motion,fixture.geometry,device_batch=batch)
        a=cpu.command(request,scene,fixture.feedback,reference,now=10.,clock_domain='synthetic')
        self.assertEqual(a['status'],'candidate',a)
        with torch.cuda.stream(torch.cuda.Stream(priority=-1)):
            b=gpu.command(request,scene,fixture.feedback,reference,now=10.,clock_domain='synthetic')
        self.assertEqual(a['status'],b['status']);np.testing.assert_allclose(a['velocity'],b['velocity'],atol=1e-10)
        self.assertGreater(metrics.summary()['transfer_bytes']['h2d:dwa_groups'],0)
        self.assertGreater(metrics.summary()['transfer_bytes']['d2h:dwa_candidate_summaries'],0)

    def test_device_projection_and_explicit_partial_export(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        library=Path(__file__).resolve().parents[1]/'outputs/runtime/build/environment/libsocial_nav_occupancy_cuda.so'
        config=load_config();config['compute'].update(backend='cuda',device=0)
        make=lambda resolution:OccupancyTree(resolution,library=library,backend='cuda')
        host=LocalEnvironmentMap(config,tree_factory=make);device=LocalEnvironmentMap(config,tree_factory=make)
        self.addCleanup(host.close);self.addCleanup(device.close)
        x,y=np.meshgrid(np.linspace(-3,3,40),np.linspace(-3,3,40));cloud=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        metrics=Metrics()
        for i in range(5):
            expected=host.update(cloud,np.eye(4),1+i*.1,('device-test',0),scan_duration=.1)
            actual=device.update(cloud,np.eye(4),1+i*.1,('device-test',0),scan_duration=.1,device='cuda:0',metrics=metrics)
        cpu=actual.materialize(consumer='test_parity',metrics=metrics)
        self.assertEqual(set(cpu.layers),set(expected.layers))
        for k,v in expected.layers.items():
            self.assertEqual(cpu.layers[k].dtype,v.dtype)
            np.testing.assert_allclose(cpu.layers[k],v,atol=3e-7,rtol=3e-7,err_msg=k)
        partial=actual.materialize(consumer='safety',fields=['occupancy','person_background_free'],metrics=metrics)
        self.assertEqual(set(partial.layers),{'occupancy','person_background_free'})
        self.assertNotIn('d2h:environment_host_layers',metrics.summary()['transfer_bytes'])
        self.assertFalse(partial.layers['occupancy'].flags.writeable)
