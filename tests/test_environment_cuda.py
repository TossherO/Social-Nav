"""Explicit CUDA evidence parity: rays, lifetimes, rolling bounds and layer semantics."""
import unittest
import numpy as np

from perception.environment.occupancy import OccupancyTree
from perception.environment.layers import load_config, project_evidence


def ordered(voxels):
    return voxels[np.lexsort(voxels[:, :3].T)]


class CudaEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('Explicit environment CUDA tests require a visible GPU')

    def engines(self, resolution=.15):
        engines = [OccupancyTree(resolution,backend=backend) for backend in ('cpu','cuda')]
        for engine in engines:self.addCleanup(engine.close)
        return engines

    def test_suspended_history_matches_cpu_and_preserves_obstacle_updates(self):
        from tests.test_environment_near_field import suspended_history_sequence,person_background_sequence
        cpu=suspended_history_sequence('cpu');gpu=suspended_history_sequence('cuda')
        for (a,am),(b,bm) in zip(cpu,gpu):
            for key in a:np.testing.assert_allclose(a[key],b[key],atol=3e-7,rtol=2e-7,err_msg=key)
            for key in ('cached_cells','restored_cells','history_suspended','temporary_obstacle_cells'):
                self.assertEqual(am[key],bm[key],key)
        for layers in person_background_sequence('cuda',suspended=True):
            self.assertFalse(layers['person_background_free'].any())

    def test_integer_height_coverage_preserves_complete_missing_and_occupied_columns(self):
        from tests.test_environment_projection import height_column
        cases = [dict(), dict(ground=-.8701), dict(shift=10), dict(missing=(8,)),
                 dict(missing=(0,)), dict(missing=(16,)), dict(occupied=True), dict(endpoint=True),
                 dict(ground=np.nan), dict(supported=False),
                 dict(resolution=.15, ground=.075, minimum=.15, maximum=.6, keys=range(1, 5))]
        for dtype in (np.float32, np.float64):
            for case in cases:
                with self.subTest(dtype=dtype, case=case):
                    cpu_reference, cpu = height_column('cpu', dtype=dtype, **case)
                    gpu_reference, gpu = height_column('cuda', dtype=dtype, **case)
                    for layers in (cpu, gpu_reference, gpu):
                        for field in ('coverage', 'occupancy'):
                            np.testing.assert_array_equal(layers[field], cpu_reference[field])
                    if case.get('occupied') or case.get('endpoint'):
                        self.assertEqual(gpu['occupancy'][1, 10], 100)
                    elif 'missing' in case or not case.get('supported', True) or np.isnan(case.get('ground', 0)):
                        self.assertEqual(gpu['occupancy'][1, 10], -1)
                    else:
                        self.assertEqual(gpu['coverage'][1, 10], 1.)
                        self.assertEqual(gpu['occupancy'][1, 10], 0)

    def test_exclusive_person_background_matches_cpu_including_mixed_and_unknown_cells(self):
        from tests.test_environment_near_field import person_background_sequence
        cpu,gpu=person_background_sequence('cpu'),person_background_sequence('cuda')
        for a,b in zip(cpu,gpu):
            self.assertEqual(set(a),set(b))
            for name in a:np.testing.assert_allclose(a[name],b[name],atol=3e-7,rtol=2e-7,err_msg=name)
        self.assertTrue(gpu[0]['person_background_free'][40,45])
        self.assertFalse(gpu[0]['person_background_free'][40,50])
        self.assertFalse(gpu[1]['person_background_free'][40,45])
        self.assertFalse(gpu[2]['person_background_free'][40,45])

    def test_ground_reference_recovery_withdraws_cached_views_on_both_backends(self):
        from perception.environment.layers import LocalEnvironmentMap
        maps=[]
        for backend in ['cpu','cuda']:
            config=load_config();config['compute']['backend']=backend
            model=LocalEnvironmentMap(config);self.addCleanup(model.close);maps.append(model)
        x,y=np.meshgrid(np.linspace(-3,3,30),np.linspace(-3,3,30))
        floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        for i in range(11):
            if i==5:
                for model in maps:model.ground_tracker.reference=.54
            views=[m.update(floor,np.eye(4),1+i*.1,('synthetic',0),scan_duration=.1) for m in maps]
            for key in views[0].layers:
                np.testing.assert_allclose(views[0].layers[key],views[1].layers[key],atol=3e-7,rtol=2e-7,err_msg=key)
            if i==5:
                self.assertFalse(views[1].layers['ground_support'].any())
                self.assertFalse(views[1].layers['near_field_free'].any())
        self.assertTrue(views[1].layers['ground_support'].any())
        self.assertAlmostEqual(maps[1].ground_tracker.reference,.85,places=2)

    def test_hit_invalidation_requires_post_hit_free_on_both_backends(self):
        from tests.test_environment_near_field import column_sequence
        self.assertEqual(column_sequence('cpu'),[0,100,100,-1,0])
        self.assertEqual(column_sequence('cuda'),[0,100,100,-1,0])

    def test_near_field_history_and_obstacle_lifetimes_match_cpu_reference(self):
        from perception.environment.near_field import NearFieldFreeMemory
        from perception.environment.cuda_view import native_view
        from tests.test_environment_near_field import sample
        config=dict(load_config(),hit_invalidates_free=True)
        settings=dict(remember_outside_ring=True,temporary_obstacle_overlay=True)
        cpu=NearFieldFreeMemory(config,settings);gpu=NearFieldFreeMemory(config,settings)
        engine=OccupancyTree(.1,backend='cuda');self.addCleanup(engine.close)
        observations=[(sample(1,state=0),[],None),(sample(2),[[.55,.05,.5]],[1]),
            (sample(2.5),[],None),(sample(3),[],None),(sample(4,origin_x=-3.9),[],None),
            (sample(5,state=0,identity=1),[],None),(sample(6,identity=1),[[.55,.05,.5]],[0]),
            (sample(7,identity=1),[],None),(sample(9,identity=1),[],None),
            (sample(4,identity=1),[],None)]
        from scipy.spatial.transform import Rotation
        for i,angles in enumerate(((0,0,3.13),(.18,-.16,-2.9),(.4,.1,.5),(-.11,.2,1.5))):
            source=sample(10+i,state=0,identity=2)
            transform=np.asarray(source.metadata['world_from_lidar'])
            transform[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix()
            source.metadata['world_from_lidar']=transform.tolist()
            observations.append((source,[],None))
        for index,(source,points,moving) in enumerate(observations):
            points=np.asarray(points).reshape(-1,3)
            a=cpu.update(source,points,transient_mask=moving)
            m=source.metadata;raw=source.layers
            layers,description=native_view(engine,raw['ground_z'],raw['ground_support'],m['origin_xy'],gpu.config,
                m['stamp_s'],points,source_layers=raw,metadata=m,policy=gpu,transient_mask=moving)
            self.assertEqual(set(a.layers),set(layers))
            for name in a.layers:
                np.testing.assert_allclose(a.layers[name],layers[name],atol=3e-7,rtol=2e-7,err_msg=name)
            for name in ('restored_cells','cached_cells','temporary_obstacle_cells','ground_conflict_cells'):
                self.assertEqual(a.metadata['near_field_memory'][name],description[name],name)
            if index==3:self.assertEqual(layers['occupancy'][40,45],0)  # person .8s, not ordinary 2s

    def test_cuda_exact_distance_fields_include_border_and_no_obstacle_cases(self):
        from perception.environment.cuda_view import native_view
        from perception.environment.layers import finish_evidence
        config=load_config();engine=OccupancyTree(.1,backend='cuda');self.addCleanup(engine.close)
        random=np.random.default_rng(37)
        for cells in [np.zeros((11,17),np.int8),np.full((11,17),-1,np.int8),
                      random.choice([-1,0,100],(11,17)).astype(np.int8)]:
            floor=np.zeros(cells.shape,np.float32);support=np.ones(cells.shape,bool)
            expected=finish_evidence(cells,np.ones(cells.shape,bool),np.ones(cells.shape),np.zeros(cells.shape,np.float32),
                np.full(cells.shape,np.nan),np.full(cells.shape,np.nan),floor,support,.1)
            actual,_=native_view(engine,floor,support,[0,0],dict(config,resolution=.1),1.,np.empty((0,3)),source_layers=expected)
            for name in expected:np.testing.assert_allclose(actual[name],expected[name],atol=3e-7,rtol=2e-7,err_msg=name)

    def test_moving_rays_duplicate_hits_and_rolling_evidence_match(self):
        config=load_config();config.update(resolution=.15)
        engines=self.engines()
        random=np.random.default_rng(31)
        cloud=random.uniform([-2.8,-2.8,-1.8],[2.8,2.8,1.8],(160,3)).astype(np.float32)
        cloud=np.r_[cloud,cloud[:30],[[1.05,0,0],[2.05,0,0]]].astype(np.float32)
        for step,(stamp,shift) in enumerate(((1.,0),(1.2,0),(1.5,.3),(2.,6.),(10.,6.))):
            points=cloud.copy();points[:,0]+=shift
            origins=np.zeros_like(points);origins[:,0]=shift+np.linspace(0,.05,len(points))
            mask=np.arange(len(points))%7==0
            results=[engine.update(points,[shift,0,0],[-3+shift,-3,-3,3+shift,3,3],stamp,config,
                ray_origins=origins,transient_mask=mask,transient_ttl=.8) for engine in engines]
            np.testing.assert_allclose(ordered(results[1]),ordered(results[0]),atol=2e-7,rtol=2e-7)

    def test_native_cpu_and_cuda_layers_match_numpy_reference(self):
        config=load_config();config.update(resolution=.15,min_obstacle_height_m=.15,max_height_m=.6)
        engines=self.engines()
        z=.225+np.arange(3)*.15
        points=np.c_[np.repeat(2.05,3),np.repeat(.05,3),z].astype(np.float32)
        endpoints=np.r_[points,[[.225,.225,.225]]].astype(float)
        for stamp in (1.,1.2,1.4):
            for engine in engines:engine.update(points,[0,0,0],[-3,-3,-3,3,3,3],stamp,config)
        for dtype in (np.float32,np.float64):
            floor=np.zeros((40,40),dtype);floor[0,:]=np.nan
            support=np.isfinite(floor)
            expected=project_evidence(engines[0].snapshot(1.4),floor,support,[-3,-3],config,1.4,endpoints=endpoints)
            for engine in engines:
                actual=engine.project(floor,support,[-3,-3],config,1.4,endpoints=endpoints)
                for name in expected:
                    np.testing.assert_allclose(actual[name],expected[name],atol=2e-7,err_msg=name)

    def test_transient_expiry_empty_bounds_and_voxel_budget(self):
        config=load_config()
        for engine in self.engines():
            bounds=[-3,-3,-3,3,3,3]
            engine.update([[1.05,0,0],[2.05,0,0]],[0,0,0],bounds,1.,config,rays=False,
                transient_mask=[1,0],transient_ttl=.8)
            result=engine.update(np.empty((0,3)),[0,0,0],bounds,2.,config,rays=False,
                transient_mask=[],transient_ttl=.8)
            self.assertEqual(len(result),1);self.assertGreater(result[0,0],2.)
            result=engine.update(np.empty((0,3)),[10,0,0],[7,-3,-3,13,3,3],2.1,config,rays=False)
            self.assertEqual(len(result),0)
            with self.assertRaisesRegex(RuntimeError,'budget exceeded'):
                engine.update([[8,0,0],[9,0,0]],[10,0,0],[7,-3,-3,13,3,3],3.,
                              dict(config,max_voxels=1),rays=False)
            self.assertEqual(len(engine.snapshot(3.)),0)


if __name__=='__main__':unittest.main()
