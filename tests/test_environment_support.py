"""Lightweight support invariants and independent CPU/CUDA agreement."""
import copy
import unittest
import numpy as np

from perception.environment.layers import load_config,GridSnapshot,project_evidence
from perception.environment.support import NavigationSupport,finalize
from perception.environment.navigation import navigation_geometry
from tests.test_environment_navigation import observation,prior_asset,scene


def points(raw,xyz=(),moving=()):
    m=raw.metadata
    return dict(stamp_s=m['stamp_s'],identity=(m['session_id'],m['generation'],m['frame_id']),
                endpoints=np.asarray(xyz,np.float64).reshape(-1,3),transient_mask=np.asarray(moving,np.uint8))


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config();self.support=NavigationSupport(self.config);self.prior=prior_asset()

    def raw(self,stamp=10.):return observation(np.full((60,60),-1,np.int8),stamp=stamp)

    def apply(self,raw=None,xyz=(),moving=()):
        raw=self.raw() if raw is None else raw
        return self.support.apply(raw,self.prior,observation=points(raw,xyz,moving))

    def test_actual_weak_rays_are_free_but_missing_height_is_unknown(self):
        floor=np.full((1,1),-.87);config=self.config
        z=(np.arange(-8,9)+.5)*.1
        voxels=np.c_[np.full(17,.05),np.full(17,.05),z,np.full(17,.4),np.zeros(17)]
        for v,expected in ((voxels,0),(voxels[:-1],-1),(np.empty((0,5)),-1)):
            result=project_evidence(v,floor,np.ones((1,1),bool),[0,0],config,10.)
            self.assertEqual(result['occupancy'][0,0],expected)

    def test_static_free_preserves_raw_and_original_time_and_verifies(self):
        raw=self.raw();result=self.apply(raw)
        self.assertIs(raw.layers['occupancy'],result.layers['occupancy'])
        self.assertEqual(result.layers['navigation_occupancy'][20,20],0)
        self.assertEqual(result.layers['navigation_source_stamp'][20,20],1.)
        navigation_geometry(scene(result),asset=self.prior,check_asset=True,now=10.)
        with self.assertRaises(ValueError):navigation_geometry(scene(result),asset=self.prior,check_asset=True,now=11.)
        self.prior.snapshot.layers['occupancy'][20,20]=100
        with self.assertRaises(ValueError):navigation_geometry(scene(result),asset=self.prior,check_asset=True,now=10.)

    def test_no_prior_unknown_prior_wrong_floor_and_boundary_remain_unknown(self):
        raw=self.raw();empty=self.support.apply(raw,None)
        self.assertNotIn('navigation_occupancy',empty.layers)
        self.assertIs(empty.layers,raw.layers)
        self.prior.snapshot.layers['ground_z'][:]=.85
        self.assertNotIn('navigation_occupancy',self.apply().layers)
        self.prior=prior_asset();self.prior.snapshot.layers['occupancy'][20,21]=-1
        raw=self.raw(10.2);raw.metadata['association']['map_from_local'][0][3]=.001
        result=self.apply(raw)
        self.assertEqual(result.layers.get('navigation_occupancy',result.layers['occupancy'])[20,20],-1)

    def test_static_floor_survives_missing_or_noisy_current_ground(self):
        self.apply()
        raw=self.raw(10.2);raw.metadata['ground_tracking']=dict(status='unavailable')
        self.assertEqual(self.apply(raw).layers['navigation_occupancy'][20,20],0)
        raw=self.raw(10.4);raw.metadata['ground_tracking']['plane_world'][2]+=.1
        self.assertEqual(self.apply(raw).layers['navigation_occupancy'][20,20],0)
        raw=self.raw(10.6);raw.metadata['association']['valid']=False
        self.assertNotIn('navigation_occupancy',self.apply(raw).layers)

    def test_ground_missing_endpoints_overlay_and_ordinary_protection(self):
        hit=[[2.05,2.05,-.3]]
        a=self.apply(self.raw(),hit,[0]);self.assertEqual(a.layers['navigation_occupancy'][20,20],100)
        self.apply(self.raw(10.2),hit,[1])
        b=self.apply(self.raw(10.5));self.assertEqual(b.layers['navigation_occupancy'][20,20],100)
        navigation_geometry(scene(b),asset=self.prior,check_asset=True,now=10.5)
        c=self.apply(self.raw(12.1));self.assertEqual(c.layers['navigation_occupancy'][20,20],0)
        self.assertEqual(c.layers['navigation_source_stamp'][20,20],1.)
        self.apply(self.raw(12.2),hit,[1])
        self.assertEqual(self.apply(self.raw(12.3)).layers['navigation_occupancy'][20,20],100)
        self.assertEqual(self.apply(self.raw(12.5)).layers['navigation_occupancy'][20,20],0)

    def test_raw_occupied_always_wins_and_reset_drops_endpoint_history(self):
        raw=self.raw();raw.layers['occupancy'][20,20]=100
        result=self.apply(raw);self.assertEqual(result.layers['navigation_source'][20,20],2)
        self.apply(self.raw(10.2),[[2.05,2.05,-.3]],[0]);self.support.reset()
        self.assertEqual(self.apply(self.raw(10.3)).layers['navigation_occupancy'][20,20],0)

    def test_local_ground_conflicts_veto_prior_and_survive_independent_capsule_verification(self):
        from perception.environment.navigation import safety_capsule
        raw=self.raw();raw.layers['ground_conflict']=np.zeros((60,60),bool)
        raw.layers['ground_conflict'][20,20]=True;raw.layers['occupancy'][20,21]=100
        raw.layers['ground_conflict'][20,21]=True
        result=self.apply(raw);nav=result.layers['navigation_occupancy']
        self.assertEqual(nav[20,20],-1);self.assertEqual(nav[20,21],100)
        self.assertEqual(nav[20,22],0)
        navigation_geometry(scene(safety_capsule(result)),asset=self.prior,check_asset=True,now=10.)
        lost=copy.deepcopy(result);lost.layers.pop('ground_conflict')
        with self.assertRaises(ValueError):navigation_geometry(scene(lost),now=10.)

    def test_optional_trace_records_confirmation_and_reset_without_changing_maps(self):
        from runtime.metrics import Metrics
        import json
        traced=NavigationSupport(self.config)
        metrics=Metrics(trace_path='unused_trace_path_no_write')
        for stamp,valid in ((10.,True),(10.2,True),(10.4,False)):
            raw=self.raw(stamp);raw.metadata['association']['valid']=valid
            expected=self.support.apply(raw,self.prior,observation=points(raw))
            actual=traced.apply(raw,self.prior,observation=points(raw),metrics=metrics)
            self.assertEqual(actual.metadata,expected.metadata)
            self.assertEqual(actual.layers.keys(),expected.layers.keys())
            for key in actual.layers:np.testing.assert_equal(actual.layers[key],expected.layers[key])
        rows=list(metrics.trace_rows);json.dumps(rows,allow_nan=False)
        self.assertEqual(len(rows),3);self.assertEqual(rows[0]['measurement_stamp_s'],10.)
        self.assertEqual(rows[0]['diagnostics']['floor_source'],'selected_static_map')
        self.assertEqual(rows[-1]['diagnostics']['reset_reason'],'association_unavailable')


class SupportCudaTests(unittest.TestCase):
    def test_native_integer_columns_and_ordinary_hit_survive_later_person(self):
        import torch
        from perception.environment.occupancy import OccupancyTree
        from tests.test_environment_projection import height_column
        from tests.test_environment_near_field import sample
        from perception.environment.near_field import NearFieldFreeMemory
        from perception.environment.cuda_view import native_view
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        config=load_config();hit=np.array([[.55,.05,.5]])
        for backend in ('cpu','cuda'):
            for ground in (-.87,-.8701):
                for layers in height_column(backend,ground=ground):
                    self.assertEqual(layers['coverage'][1,10],1.)
            tree=OccupancyTree(.1,backend=backend);self.addCleanup(tree.close)
            memory=NearFieldFreeMemory(config,dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
            states=[];overlay=[]
            for stamp,cloud,moving in ((1.,hit,[0]),(1.1,hit,[1]),(1.5,np.empty((0,3)),[]),(3.1,np.empty((0,3)),[])):
                tree.update(cloud,[0,0,.84],[-4,-4,-2,4,4,3],stamp,config,rays=False,
                    transient_mask=moving,transient_ttl=.2,read=False)
                states.append(bool((tree.snapshot(stamp)[:,3]>=config['occupied_probability']).any()))
                raw=sample(stamp);raw.layers['ground_support'][40,45]=True
                if backend=='cpu':view=memory.update(raw,cloud,transient_mask=moving).layers
                else:view,_=native_view(tree,raw.layers['ground_z'],raw.layers['ground_support'],[-4,-4],config,stamp,cloud,
                    source_layers=raw.layers,metadata=raw.metadata,policy=memory,transient_mask=moving)
                overlay.append(int(view['occupancy'][40,45]))
            self.assertEqual(states,[True,True,True,False]);self.assertEqual(overlay,[100,100,100,-1])

    def test_rotation_overlap_missing_ground_roll_and_streams_match_cpu(self):
        import torch
        from scipy.spatial.transform import Rotation
        from perception.environment.device import DeviceGridSnapshot
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        config=load_config();prior=prior_asset();prior.snapshot.layers['occupancy'][10:12,20:23]=100
        prior.snapshot.layers['occupancy'][25:27,30:33]=-1
        for angles in ((0,0,0),(.02,-.015,.31)):
            host=NavigationSupport(config);gpu=NavigationSupport(config)
            transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix()
            stream=torch.cuda.Stream()
            for j in range(6):
                raw=observation(np.full((60,60),-1,np.int8),stamp=10.+.2*j,origin=(.1*j,0.))
                raw.metadata['association']['map_from_local']=transform.tolist()
                raw.layers['ground_conflict']=np.zeros(raw.layers['occupancy'].shape,bool)
                raw.layers['ground_conflict'][20:23,20:25]=True
                xyz=[[2.05,2.05,-.3]] if j<2 else []
                obs=points(raw,xyz,[j] if xyz else [])
                expected=host.apply(raw,prior,observation=obs)
                with torch.cuda.stream(stream):
                    device=DeviceGridSnapshot.seal(raw.metadata,{k:torch.from_numpy(v).to('cuda:0') for k,v in raw.layers.items()})
                    result=gpu.apply(device,prior,observation=obs)
                actual=finalize(result.materialize(consumer='support_test'))
                for name in ('navigation_occupancy','navigation_source','navigation_source_stamp'):
                    np.testing.assert_equal(actual.layers[name],expected.layers[name],err_msg=name)
                navigation_geometry(scene(actual),asset=prior,check_asset=True,now=raw.metadata['stamp_s'])
