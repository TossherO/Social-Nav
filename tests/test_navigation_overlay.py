"""Reference-only evidence projection, with independent polygon overlap checks."""
import unittest
from pathlib import Path
import numpy as np

from perception.environment.layers import GridSnapshot
from navigation.planning.overlay import reference_overlay,overlap_states
from navigation.planning.footprint import VehicleGeometry,polygon_cells
from navigation.planning.inputs import load_config
from navigation.planning.planner import Planner
from tests.navigation_fixtures import asset,bundle,request

POSE_LIBRARY=Path('outputs/runtime/build/planning/libsocial_nav_pose_search.so')

class ReferenceOverlayTests(unittest.TestCase):
    def setUp(self):
        self.scene=bundle();self.local=self.scene.geometry
        cells=np.zeros((60,60),np.int8);cells[20,20]=-1;cells[30,30]=100
        self.prior=asset(cells).snapshot

    def apply(self,local=None,transform=None):
        return reference_overlay(self.prior,local or self.local,self.local.layers['occupancy'],
            np.eye(3) if transform is None else transform,self.scene)

    def test_existing_local_free_complements_unknown_without_mutating_evidence(self):
        original=self.prior.layers['occupancy'].copy()
        result,report=self.apply()
        self.assertEqual(result.layers['occupancy'][20,20],0)
        self.assertEqual(result.layers['occupancy'][30,30],100)
        self.assertEqual(report['unknown_to_free'],1)
        self.assertEqual(report['source_as_of'],self.scene.record['as_of'])
        self.assertEqual(report['source_scene'],self.scene.record['input_fingerprint'])
        self.assertFalse(report['persistent']);self.assertNotIn('last_observed',result.layers)
        self.assertFalse(result.layers['occupancy'].flags.writeable)
        np.testing.assert_array_equal(self.prior.layers['occupancy'],original)

    def test_static_derived_free_cannot_recursively_supply_prior_unknowns(self):
        self.local.layers['occupancy'][20,20]=-1
        derived=GridSnapshot(self.local.metadata,dict(occupancy=np.zeros((60,60),np.int8)))
        result,_=self.apply(derived)
        self.assertEqual(result.layers['occupancy'][20,20],-1)

    def test_current_endpoint_overlay_vetoes_free_raw_background(self):
        cells=self.local.layers['occupancy'].copy();cells[20,20]=100
        result,_=self.apply(GridSnapshot(self.local.metadata,dict(occupancy=cells)))
        self.assertEqual(result.layers['occupancy'][20,20],100)

    def test_free_centre_does_not_hide_unknown_or_occupied_corner(self):
        transform=np.eye(3);transform[0,2]=.03
        for value in (-1,100):
            self.local.layers['occupancy'][20,21]=value
            result,_=self.apply(transform=transform)
            self.assertEqual(result.layers['occupancy'][20,20],100 if value==100 else -1)

    def test_live_obstacle_overrides_free_and_conditional_prior_then_withdraws(self):
        from perception.environment.changes import reference_grid
        conditional=np.zeros((60,60),bool);conditional[25,25]=True
        self.prior.layers['occupancy'][25,25]=100
        prior=GridSnapshot(self.prior.metadata,dict(occupancy=self.prior.layers['occupancy'],conditional_obstacle=conditional))
        original=prior.layers['occupancy'].copy()
        self.local.layers['occupancy'][25,25]=100;self.local.layers['occupancy'][25,26]=100
        result,report=reference_overlay(prior,self.local,self.local.layers['occupancy'],np.eye(3),self.scene)
        search=reference_grid(result,geometry_only=True)
        self.assertEqual(search.layers['occupancy'][25,25],100)
        self.assertEqual(search.layers['occupancy'][25,26],100)
        self.assertEqual(report['current_occupied_added'],1)
        np.testing.assert_array_equal(prior.layers['occupancy'],original)
        self.assertTrue(prior.layers['conditional_obstacle'][25,25])
        self.local.layers['occupancy'][25,25:27]=0
        later,_=reference_overlay(prior,self.local,self.local.layers['occupancy'],np.eye(3),self.scene)
        self.assertEqual(reference_grid(later,geometry_only=True).layers['occupancy'][25,25],0)
        self.assertEqual(later.layers['occupancy'][25,26],0)

    def test_next_query_withdraws_unavailable_support_and_keeps_static_asset(self):
        result,_=self.apply();self.assertEqual(result.layers['occupancy'][20,20],0)
        self.local.layers['occupancy'][:]=-1
        later,_=self.apply();self.assertEqual(later.layers['occupancy'][20,20],-1)
        self.assertEqual(self.prior.layers['occupancy'][20,20],-1)

    def test_outside_local_coverage_and_large_tilt_cannot_supply_free(self):
        transform=np.eye(3);transform[0,2]=4.
        result,_=self.apply(transform=transform)
        self.assertEqual(result.layers['occupancy'][20,20],-1)
        c,s=np.cos(.2),np.sin(.2);tilt=np.eye(4);tilt[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
        self.scene.record['prior']['map_from_local']=tilt.tolist()
        result,report=self.apply();self.assertEqual(result.layers['occupancy'][20,20],-1)
        self.assertEqual(report['reason'],'map_local_tilt_exceeds_planar_limit')

    def test_vector_projection_matches_independent_polygon_intersections(self):
        rng=np.random.default_rng(1105)
        cells=rng.choice([0,-1,100],size=(60,60),p=[.85,.08,.07]).astype(np.int8)
        self.local.layers['occupancy'][:]=cells
        y,x=np.indices(cells.shape);centres=np.c_[x.ravel()+.5,y.ravel()+.5]*.1
        points=rng.uniform(1,5,(20,2));corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*.07
        for angle in (.17,.8,1.4):
            c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
            free,hit=overlap_states(points,rotation,.14,self.local,cells)
            for i,p in enumerate(points):
                keep=polygon_cells(corners@rotation+p,centres,.05)
                values=cells.ravel()[keep]
                self.assertEqual(free[i],bool(np.all(values==0)))
                self.assertEqual(hit[i],bool(np.any(values==100)))

    def test_obstacle_projection_matches_all_cell_overlaps_when_rotated(self):
        cells=np.zeros((60,60),np.int8);cells[15:18,20]=100;cells[36,40]=100
        self.local.layers['occupancy'][:]=cells
        self.prior.layers['occupancy'][:]=0
        angle=.63;c,s=np.cos(angle),np.sin(angle)
        transform=np.array([[c,-s,1.3],[s,c,-.6],[0.,0.,1.]])
        result,_=self.apply(transform=transform)
        y,x=np.indices((60,60));xy=np.c_[x.ravel()+.5,y.ravel()+.5]*.1
        points=xy@transform[:2,:2].T+transform[:2,2]
        _,expected=overlap_states(points,transform[:2,:2].T,.1,self.local,cells)
        np.testing.assert_array_equal(result.layers['occupancy']==100,expected.reshape(60,60))

    @unittest.skipUnless(POSE_LIBRARY.is_file(),'Native pose search library not built')
    def test_mapped_search_and_current_check_accept_the_same_known_free_start(self):
        cfg=load_config();geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        cfg['algorithm']='hybrid_astar'
        cfg['pose_search']['native_library']='outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        source=asset();source.snapshot.layers['occupancy'][20,20]=-1
        planner=Planner(cfg,geometry)
        result=planner.query(request('mapped',goal=[4.,2.]),scene=self.scene,asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['local_start_footprint'],'free')
        self.assertEqual(result['reference_overlay']['unknown_to_free'],1)
        self.assertEqual(source.snapshot.layers['occupancy'][20,20],-1)
        self.assertTrue(result['current_path_valid'])

    @unittest.skipUnless(POSE_LIBRARY.is_file(),'Native pose search library not built')
    def test_mapped_old_path_is_replanned_around_new_obstacle(self):
        from navigation.planning.footprint import swept_check
        cfg=load_config();cfg['algorithm']='hybrid_astar'
        cfg['pose_search']['native_library']='outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        planner=Planner(cfg,geometry);source=asset();req=request('mapped',goal=[4.5,2.])
        first=planner.query(req,scene=bundle(),asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(first['status'],'path_found',first)
        cells=np.zeros((60,60),np.int8);cells[17:23,30]=100
        second=planner.query(req,scene=bundle(cells),asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(second['status'],'path_found',second)
        self.assertNotIn('reference_continuity',second)
        self.assertGreater(second['reference_overlay']['current_occupied_added'],0)
        self.assertEqual(swept_check(bundle(cells).geometry,geometry,second['path'],cfg['safety_margin_m'],cfg['max_path_samples'],prefix=False)['status'],'free')

    @unittest.skipUnless(POSE_LIBRARY.is_file(),'Native pose search library not built')
    def test_new_full_barrier_does_not_reuse_old_route(self):
        cfg=load_config();cfg['algorithm']='hybrid_astar'
        cfg['pose_search']['native_library']='outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        planner=Planner(cfg,geometry);source=asset();req=request('mapped',goal=[4.5,2.])
        first=planner.query(req,scene=bundle(),asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(first['status'],'path_found',first)
        cells=np.zeros((60,60),np.int8);cells[:,30]=100
        blocked=planner.query(req,scene=bundle(cells),asset=source,now=10.,clock_domain='synthetic')
        self.assertNotEqual(blocked['status'],'path_found')
        self.assertNotIn('reference_continuity',blocked)

    def test_sustained_control_stall_requests_replan_without_mistaking_turns(self):
        from navigation.planning.reference import ReferenceRuntime
        from navigation.planning.motion import load_motion_config
        from tests.navigation_fixtures import FakeWorker
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        with ReferenceRuntime(load_config(),load_motion_config(),geometry,worker_factory=FakeWorker) as runtime:
            turn=dict(proposal=dict(status='candidate',velocity=[0.,.2]))
            runtime.note_control(turn,monotonic_s=1.);runtime.note_control(turn,monotonic_s=3.)
            self.assertFalse(runtime.need_replan)
            stop=dict(proposal=dict(status='candidate',velocity=[0.,0.]))
            runtime.note_control(stop,monotonic_s=4.);runtime.note_control(stop,monotonic_s=4.9)
            self.assertFalse(runtime.need_replan)
            runtime.note_control(stop,monotonic_s=5.);self.assertTrue(runtime.need_replan)
            runtime.note_control(turn,monotonic_s=5.1);self.assertIsNone(runtime.control_blocked_since)

    @unittest.skipUnless(POSE_LIBRARY.is_file(),'Native pose search library not built')
    def test_no_path_withdraws_previous_reference(self):
        from navigation.planning.reference import ReferenceRuntime
        from navigation.planning.motion import load_motion_config
        from navigation.planning.continuity import reference_envelope
        from perception.environment.context import fingerprint
        from tests.navigation_fixtures import FakeWorker
        cfg=load_config();cfg['algorithm']='hybrid_astar'
        cfg['pose_search']['native_library']='outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        scene=bundle();req=request('mapped',goal=[4.5,2.]);source=asset()
        result=Planner(cfg,geometry).query(req,scene=scene,asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        with ReferenceRuntime(cfg,load_motion_config(),geometry,worker_factory=FakeWorker) as runtime:
            runtime.update(req,scene,None,source,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
            runtime.worker.reply(reference_envelope(result,req,scene,source));runtime.tick(now=10.,clock_domain='synthetic')
            self.assertIsNotNone(runtime.reference)
            runtime.worker.send(dict(job_id=2,epoch=runtime.epoch,context=runtime.context,request=req,scene=scene,sent_mono=0.))
            result.update(status='no_path',reason='returned_path_no_curvature_route',path=[])
            result.pop('result_binding');result['result_binding']=fingerprint(result)
            runtime.worker.reply(reference_envelope(result,req,scene,source))
            status=runtime.tick(now=10.,clock_domain='synthetic')
            self.assertIsNone(runtime.reference);self.assertIsNone(runtime.current)
            self.assertEqual(status['reason'],'reference_has_no_path')


if __name__=='__main__':unittest.main()
