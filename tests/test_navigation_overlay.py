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

    def test_reference_remembers_hit_until_new_local_free_even_without_an_active_goal(self):
        from navigation.planning.reference import ReferenceRuntime
        from navigation.planning.motion import load_motion_config
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config as environment_config
        from tests.navigation_fixtures import FakeWorker

        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        source=asset()
        target=(25,25)
        index=np.ravel_multi_index(target,(60,60))
        def observation(stamp,value,measured,*,restored=False,generation=0):
            local=bundle(generation=generation).geometry
            local.metadata['stamp_s']=stamp
            local.metadata['association'].update(stamp_s=stamp,measurement_stamp_s=stamp-.05)
            local.layers['occupancy'][target]=value
            local.layers['last_observed']=np.full((60,60),measured)
            local.layers['near_field_free']=np.zeros((60,60),bool)
            local.layers['near_field_free'][target]=restored
            return assemble_scene(local,clock_domain='synthetic',settings=environment_config()['scene'])

        with ReferenceRuntime(load_config(),load_motion_config(),geometry,worker_factory=FakeWorker) as runtime:
            for tick,value,measured,restored,blocked in [
                (10.,100,10.,False,True), (13.,-1,13.,False,True),
                (14.,0,9.,False,True), (15.,0,15.,True,True),
                (16.,0,16.,False,False), (17.,-1,17.,False,False),
                (18.,100,18.,False,True)]:
                runtime.observe(observation(tick,value,measured,restored=restored),source,now=tick)
                self.assertEqual(index in runtime.obstacles.blocked,blocked)
            hidden=observation(18.5,-1,18.5)
            runtime.observe(hidden,source,now=18.5)
            self.assertTrue(runtime.update(request('mapped'),hidden,None,source,now=18.5))
            runtime.tick(now=18.5,clock_domain='synthetic')
            self.assertIn(index,runtime.worker.job['reference_obstacles'])
            runtime.observe(observation(19.,-1,19.,generation=1),source,now=19.)
            self.assertFalse(runtime.obstacles.blocked)
            self.assertEqual(source.snapshot.layers['occupancy'][target],0)

    def test_work_cache_keeps_door_evidence_and_rechecks_each_observation(self):
        from types import SimpleNamespace
        from navigation.planning.costmap import build_costmap
        from navigation.planning.overlay import ReferenceObstacles
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config as environment_config

        cfg=load_config()
        cfg['algorithm']='dijkstra'
        cfg['pose_search']['enabled']=False
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        backend=SimpleNamespace(search=lambda *args: (np.empty((0,2),int),0.))
        planner=Planner(cfg,geometry,backend)
        source=asset()
        ledger=ReferenceObstacles()
        door=(27,27)
        req=request('mapped')

        def observation(stamp,value,generation=0):
            local=bundle(generation=generation).geometry
            local.metadata['stamp_s']=stamp
            local.metadata['association'].update(stamp_s=stamp,measurement_stamp_s=stamp-.05)
            local.layers['occupancy'][door]=value
            local.layers['last_observed']=np.full((60,60),stamp)
            return assemble_scene(local,clock_domain='synthetic',settings=environment_config()['scene'])

        def query(scene,request=req):
            stamp=scene.record['as_of']
            return planner.query(request,scene=scene,asset=source,now=stamp,clock_domain='synthetic',
                reference_obstacles=ledger.blocked)

        for stamp,value,blocked,cost_reused in [
            (10.,0,False,False),(11.,100,True,False),(12.,-1,True,True),
            (13.,0,False,False),(14.,0,False,True)
        ]:
            with self.subTest(stamp=stamp):
                scene=observation(stamp,value)
                ledger.update(scene,source.snapshot,np.eye(3),('map',0))
                result=query(scene)
                self.assertFalse(result['reference_work_reused'])
                self.assertEqual(result['costmap_reused'],cost_reused)
                self.assertEqual(result['input_binding']['as_of'],stamp)
                search_grid=planner._reference_work[3]
                self.assertEqual(search_grid.layers['occupancy'][door],100 if blocked else 0)
                costs,clearance=build_costmap(search_grid,geometry.inscribed_radius,planner.config,return_clearance=True)
                np.testing.assert_array_equal(planner._costs,costs)
                np.testing.assert_array_equal(planner._clearance,clearance)

                # A caller may annotate its reply. The private grid retains its proof.
                result['reference_overlay']['source_scene']='caller_annotation'
                repeated=query(scene)
                self.assertTrue(repeated['reference_work_reused'])
                self.assertTrue(repeated['costmap_reused'])
                self.assertEqual(repeated['reference_overlay']['source_scene'],scene.record['input_fingerprint'])
                if blocked:
                    rejected=query(scene,dict(req,goal=[2.75,2.75,0.]))
                    self.assertEqual(rejected['reason'],'goal_footprint_occupied')
                    self.assertNotIn('costmap_comparison',rejected['timing_ms'])

        changed_session=query(observation(15.,0,generation=1))
        self.assertFalse(changed_session['reference_work_reused'])
        self.assertFalse(changed_session['costmap_reused'])
        planner.reset()
        after_seek=query(observation(15.,0,generation=1))
        self.assertFalse(after_seek['reference_work_reused'])
        self.assertFalse(after_seek['costmap_reused'])
        source=asset()
        after_map_reload=query(observation(15.,0,generation=1))
        self.assertFalse(after_map_reload['reference_work_reused'])
        self.assertFalse(after_map_reload['costmap_reused'])
        revised_at_same_time=observation(15.,100,generation=1)
        at_same_time=query(revised_at_same_time)
        self.assertFalse(at_same_time['reference_work_reused'])
        self.assertFalse(at_same_time['costmap_reused'])
        self.assertEqual(planner._reference_work[3].layers['occupancy'][door],100)
        stale=planner.query(req,scene=revised_at_same_time,asset=source,now=16.,clock_domain='synthetic')
        self.assertEqual(stale['reason'],'geometry_future_or_stale')
        self.assertNotIn('reference_work_reused',stale)

        # Measured free removes the conditional cost even when the optimistic
        # reference occupancy remains free on both sides of the update.
        from tests.test_environment_changes import evidence
        source=asset()
        static_cells=np.zeros((60,60),np.int8)
        static_cells[door]=100
        source.snapshot.layers.update(evidence(static_cells).layers)
        source.snapshot.layers['conditional_obstacle']=static_cells==100
        query(observation(16.,-1))
        before_grid=planner._reference_work[3]
        before_costs=planner._costs
        self.assertTrue(before_grid.layers['conditional_obstacle'][door])
        local=observation(17.,0).geometry
        local.layers.update(evidence(local.layers['occupancy'],17.).layers)
        confirmed=assemble_scene(local,clock_domain='synthetic',settings=environment_config()['scene'])
        after=query(confirmed)
        self.assertFalse(after['costmap_reused'])
        after_grid=planner._reference_work[3]
        np.testing.assert_array_equal(before_grid.layers['occupancy'],after_grid.layers['occupancy'])
        self.assertFalse(after_grid.layers['conditional_obstacle'][door])
        self.assertTrue(np.any(before_costs!=planner._costs))

    def test_conditional_cost_disk_matches_distance_threshold(self):
        from scipy.ndimage import distance_transform_edt
        from unittest.mock import patch
        from navigation.planning.costmap import build_costmap

        occupancy=np.zeros((55,63),np.int8)
        occupancy[26,32]=100
        conditional=np.zeros_like(occupancy,bool)
        conditional[10,0]=conditional[8,7]=conditional[27,32]=True
        grid=asset(occupancy).snapshot
        risk=GridSnapshot(grid.metadata,dict(grid.layers,conditional_obstacle=conditional))
        cfg=load_config()
        resolution=grid.metadata['resolution']
        distance=distance_transform_edt(~conditional)*resolution
        for radius in (.2,.25,.45,.67):
            with self.subTest(radius=radius):
                expected,clearance=build_costmap(grid,radius,cfg,return_clearance=True)
                near=(expected<253)&(distance <= radius+cfg['safety_margin_m']+resolution)
                expected[near]=np.maximum(expected[near],cfg['conditional_obstacle_cost'])
                costs,result_clearance=build_costmap(risk,radius,cfg,return_clearance=True)
                np.testing.assert_array_equal(costs,expected)
                np.testing.assert_array_equal(result_clearance,clearance)
                with patch('navigation.planning.costmap.cv2',None):
                    np.testing.assert_array_equal(build_costmap(risk,radius,cfg),expected)

    @unittest.skipUnless(POSE_LIBRARY.is_file(),'Native pose search library not built')
    def test_follow_reference_uses_static_free_beyond_local_observation(self):
        cfg=load_config()
        cfg['algorithm']='hybrid_astar'
        cfg['pose_search']['native_library']='outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        cells=np.zeros((60,60),np.int8)
        cells[:,35:]=-1
        scene=bundle(cells)
        source=asset()
        req=request('local',goal=[4.5,2.],completion_policy='continuous')
        planner=Planner(cfg,geometry)
        result=planner.query(req,scene=scene,asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['frame_id'],'odom')
        self.assertEqual(result['reference_search_frame'],'map')
        np.testing.assert_allclose(result['path'][-1][:2],req['goal'])
        self.assertEqual(result['local_check']['status'],'unknown')
        self.assertFalse(result['current_path_valid'])
        blocked=np.ravel_multi_index((20,45),(60,60))
        result=planner.query(req,scene=scene,asset=source,now=10.,clock_domain='synthetic',
                             reference_obstacles=[int(blocked)])
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['reference_overlay']['retained_occupied_cells'],1)
        self.assertEqual(result['request'],req)
        self.assertEqual(result['goal'],req['goal'])
        adjusted=result['follow_goal_adjustment']['selected_goal']
        self.assertNotEqual(adjusted,req['goal'])
        np.testing.assert_allclose(result['path'][-1][:2],adjusted)
        self.assertLessEqual(np.linalg.norm(np.asarray(adjusted)-req['goal']),.8)

    @unittest.skipUnless(POSE_LIBRARY.is_file(),'Native pose search library not built')
    def test_continuous_follow_keeps_inflight_and_adopts_exact_origin_goal(self):
        from navigation.planning.continuity import reference_envelope
        from navigation.planning.reference import ReferenceRuntime
        from navigation.planning.motion import load_motion_config
        from tests.navigation_fixtures import FakeWorker,scene_at

        cfg=load_config()
        cfg['pose_search']['native_library']=str(POSE_LIBRARY)
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        person=('call',('tracker',7,'a'),1)
        initial=request('local',goal=[4.,2.],goal_revision=1,completion_policy='continuous',valid_until_s=10.3)
        latest=dict(initial,goal=[4.4,2.],goal_revision=3,valid_until_s=10.9)
        first,second=scene_at(10.),scene_at(10.4)
        with ReferenceRuntime(cfg,load_motion_config(),geometry,worker_factory=FakeWorker) as owner:
            planner=Planner(owner.planning,geometry)
            owner.update(initial,first,None,now=10.,follow_key=person)
            owner.tick(now=10.,clock_domain='synthetic')
            owner.update(latest,second,None,now=10.4,follow_key=person)
            self.assertEqual(owner.worker.restarts,0)
            self.assertEqual(owner.worker.job['request'],initial)
            path=planner.query(initial,scene=first,now=10.,clock_domain='synthetic')
            self.assertEqual(path['status'],'path_found',path['reason'])
            owner.worker.reply(reference_envelope(path,initial,first,None))
            owner.last_sent=-np.inf
            status=owner.tick(now=10.4,clock_domain='synthetic')
            self.assertEqual(status['request'],dict(initial,valid_until_s=10.9))
            self.assertEqual(status['result']['result']['request'],initial)
            self.assertEqual(status['result']['as_of'],10.)
            self.assertEqual(owner.worker.job['request'],latest)
            status=owner.tick(now=10.4,clock_domain='synthetic')
            self.assertEqual(status['request']['goal_revision'],1)
            path=planner.query(latest,scene=second,now=10.4,clock_domain='synthetic')
            from perception.environment.context import fingerprint
            negative=dict(path,status='no_path',reason='goal_occupied',path=[])
            negative.pop('result_binding')
            negative['result_binding']=fingerprint(negative)
            owner.worker.reply(reference_envelope(negative,latest,second,None))
            status=owner.tick(now=10.4,clock_domain='synthetic')
            self.assertEqual(status['request']['goal_revision'],1)
            owner.next_retry=owner.last_sent=-np.inf
            owner.tick(now=10.4,clock_domain='synthetic')
            owner.worker.reply(reference_envelope(path,latest,second,None))
            status=owner.tick(now=10.4,clock_domain='synthetic')
            self.assertEqual(status['request'],latest)
            self.assertEqual(status['result']['as_of'],10.4)

            turned=dict(latest,goal=[1.9,2.],goal_revision=4)
            owner.update(turned,second,None,now=10.4,follow_key=person)
            self.assertIsNone(owner.tick(now=10.4,clock_domain='synthetic')['result'])
            owner.update(dict(latest,goal_revision=5),second,None,now=10.4,
                         follow_key=('call',('tracker',8,'b'),1))
            self.assertIsNone(owner.reference)
            owner.cancel()
            self.assertIsNone(owner.tick(now=10.4,clock_domain='synthetic')['request'])

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
