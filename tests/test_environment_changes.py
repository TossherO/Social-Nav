"""Changing occupancy must not turn absent observations into free-space truth."""
import unittest
from pathlib import Path
import numpy as np

from perception.environment.changes import PriorRevisions, conditional_obstacles, reference_grid
from perception.environment.layers import GridSnapshot, load_config, finish_evidence
from perception.environment.occupancy import OccupancyTree
from navigation.planning.planner import Planner
from navigation.planning.inputs import load_config as planning_config
from navigation.planning.footprint import VehicleGeometry
from tests.navigation_fixtures import asset, bundle, request, scene_at


def evidence(cells, stamp=10., generation=0):
    shape=cells.shape; floor=np.zeros(shape,np.float32); nan=np.full(shape,np.nan)
    layers=finish_evidence(cells.copy(),cells!=-1,np.full(shape,stamp),
        (cells==0).astype(np.float32),nan.copy(),nan.copy(),floor,np.ones(shape,bool),.1)
    pose=np.eye(4);pose[:2,3]=[2.,2.]
    return GridSnapshot(dict(snapshot_id='local:'+str(stamp),session_id='test',generation=generation,
        stamp_s=stamp,resolution=.1,origin_xy=[0.,0.],shape=list(shape),world_from_lidar=pose.tolist(),frame_id='odom'),layers)


class ChangeTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config()
        self.association=dict(map_from_local=np.eye(4).tolist(),association_session='a',association_generation=0)

    def test_confirmed_free_keeps_its_measurement_and_new_hits_revoke_it(self):
        from perception.environment.free_memory import ConfirmedFreeMemory, LAYER

        memory=ConfirmedFreeMemory()
        source=asset()
        def snapshot(stamp, value, *, source_stamp=None, x=0., generation=0, inherited=False):
            cells=np.full((60,60),-1,np.int8)
            ix=30-round(x/.1)
            if 0<=ix<60:
                cells[30,ix]=value
            grid=evidence(cells,stamp,generation)
            grid.metadata.update(origin_xy=[x,0.],navigation_fusion={'test':True},association=dict(
                self.association,valid=True,map_id='map-a',map_frame='map'))
            grid.layers.update(navigation_occupancy=cells,near_field_free=np.full(cells.shape,inherited))
            if source_stamp is not None:
                grid.layers['last_observed'][:]=source_stamp
            return grid

        first=memory.update(snapshot(10.,0),source)
        memory.update(snapshot(30.,-1,x=4.),source)
        returned=memory.update(snapshot(40.,-1),source)
        self.assertEqual(returned.layers[LAYER][30,30],10.)
        hit=memory.update(snapshot(41.,100),source)
        self.assertFalse(np.isfinite(hit.layers[LAYER][30,30]))
        stale=memory.update(snapshot(42.,0,source_stamp=40.5),source)
        self.assertFalse(np.isfinite(stale.layers[LAYER][30,30]))
        inherited=memory.update(snapshot(43.,0,inherited=True),source)
        self.assertFalse(np.isfinite(inherited.layers[LAYER][30,30]))
        cleared=memory.update(snapshot(44.,0),source)
        self.assertEqual(cleared.layers[LAYER][30,30],44.)
        self.assertEqual(first.layers[LAYER][30,30],10.)
        reset=memory.update(snapshot(45.,-1,generation=1),source)
        self.assertFalse(np.isfinite(reset.layers[LAYER][30,30]))

    def test_confirmed_free_rejects_old_measurements_after_a_raw_hit_without_fusion(self):
        from perception.environment.free_memory import ConfirmedFreeMemory, LAYER

        def snapshot(stamp, cell, *, fusion=True, measured=None):
            cells = np.full((60, 60), -1, np.int8)
            cells[30, 30] = cell
            grid = evidence(cells, stamp)
            grid.metadata['association'] = dict(self.association, valid=True, map_id='map-a', map_frame='map')
            if fusion:
                grid.metadata['navigation_fusion'] = {'test': True}
                grid.layers['navigation_occupancy'] = cells
            if measured is not None:
                grid.layers['last_observed'][30, 30] = measured
            return grid

        memory, source = ConfirmedFreeMemory(), asset()
        first = memory.update(snapshot(10., 0), source)
        self.assertEqual(first.layers[LAYER][30, 30], 10.)
        hit = memory.update(snapshot(11., 100, fusion=False), source)
        self.assertNotIn(LAYER, hit.layers)
        repeated_stamp = memory.update(snapshot(11., 0, measured=10.5), source)
        self.assertEqual(repeated_stamp.layers[LAYER][30, 30], -np.inf)
        recovered = memory.update(snapshot(12., -1), source)
        self.assertEqual(recovered.layers[LAYER][30, 30], -np.inf)
        stale = memory.update(snapshot(13., 0, measured=10.5), source)
        self.assertEqual(stale.layers[LAYER][30, 30], -np.inf)
        fresh = memory.update(snapshot(14., 0), source)
        self.assertEqual(fresh.layers[LAYER][30, 30], 14.)

    def test_working_map_applies_remembered_free_to_unknown_and_occupied(self):
        cells=np.zeros((60,60),np.int8);cells[30,30]=-1;cells[34,34]=100
        prior=evidence(cells)
        local=evidence(np.full_like(cells,-1),30.)
        ledger=np.full(cells.shape,10.)
        state=PriorRevisions()
        out=state.apply(prior,local,self.association,scene_binding='remembered',confirmed_free=ledger)
        for y,x in ((30,30),(34,34)):
            self.assertEqual(out.layers['occupancy'][y,x],0)
            self.assertEqual(out.layers['revised_free_stamp'][y,x],10.)
        ledger[30,30]=ledger[34,34]=-np.inf
        out=state.apply(prior,local,self.association,scene_binding='revoked',confirmed_free=ledger)
        self.assertEqual(out.layers['occupancy'][30,30],-1)
        self.assertEqual(out.layers['occupancy'][34,34],100)

    def test_working_map_revokes_partial_occupied_overlap_after_window_moves_away(self):
        cells = np.zeros((60, 60), np.int8)
        cells[30, 30] = cells[34, 30] = -1
        prior = evidence(cells)
        unknown = np.full_like(cells, -1)
        missing = np.full(cells.shape, -np.inf)
        for fusion in (True, False):
            with self.subTest(fusion=fusion):
                state = PriorRevisions()
                first = state.apply(prior, evidence(unknown, 10.), self.association,
                                    scene_binding='first', confirmed_free=np.full(cells.shape, 10.))
                self.assertEqual(first.layers['occupancy'][30, 30], 0)
                self.assertEqual(first.layers['occupancy'][34, 30], 0)

                obstacle = unknown.copy()
                obstacle[30, 0] = 100
                partial = evidence(obstacle, 11.)
                partial.metadata['origin_xy'] = [3.05, 0.]
                state.apply(prior, partial, self.association, scene_binding='partial', overlay_current=False,
                            confirmed_free=missing if fusion else None)

                obstacle = unknown.copy()
                obstacle[34, 0] = 100
                touching = evidence(obstacle, 12.)
                touching.metadata['origin_xy'] = [3.1, 0.]
                state.apply(prior, touching, self.association, scene_binding='touching', overlay_current=False,
                            confirmed_free=missing if fusion else None)

                moved = evidence(unknown, 13.)
                moved.metadata['origin_xy'] = [4., 0.]
                result = state.apply(prior, moved, self.association, scene_binding='moved',
                                     confirmed_free=missing)
                self.assertEqual(result.layers['occupancy'][30, 30], -1)
                self.assertTrue(np.isnan(result.layers['revised_free_stamp'][30, 30]))
                self.assertEqual(result.layers['occupancy'][34, 30], 0)
                self.assertEqual(result.layers['revised_free_stamp'][34, 30], 10.)

    def test_ordinary_obstacle_expires_but_free_rays_remain_and_observed_wall_refreshes(self):
        engine=OccupancyTree(.1);self.addCleanup(engine.close)
        bounds=[-3,-3,-3,3,3,3]; cloud=np.array([[1.05,.05,.55]],np.float32)
        for t in (1.,2.,3.,4.):
            voxels=engine.update(cloud,[0,.05,.55],bounds,t,self.config)
            self.assertTrue((voxels[:,3]>=self.config['occupied_probability']).any())
        voxels=engine.update(np.empty((0,3)),[0,.05,.55],bounds,6.1,self.config)
        self.assertFalse((voxels[:,3]>=self.config['occupied_probability']).any())
        self.assertTrue((voxels[:,3]<=self.config['free_probability']).any())
        self.assertFalse((voxels[:,0]>1).any()) # obstacle withdrawn, not marked free

    def test_cleared_prior_stays_revised_while_new_obstacles_are_never_promoted(self):
        cells=np.zeros((60,60),np.int8);cells[20:25,30:35]=100
        prior=evidence(cells);prior.metadata.update(snapshot_id='static',frame_id='map')
        original=prior.layers['occupancy'].copy();state=PriorRevisions()
        current=evidence(np.zeros_like(cells))
        revised=state.apply(prior,current,self.association,scene_binding='first')
        self.assertEqual(revised.metadata['prior_revision']['free_corrections'],25)
        self.assertTrue((revised.layers['occupancy'][20:25,30:35]==0).all())
        occupied=np.full_like(cells,-1);occupied[22,32]=100
        during=state.apply(prior,evidence(occupied,11.),self.association,scene_binding='during')
        self.assertEqual(during.layers['occupancy'][22,32],100)
        self.assertFalse(during.layers['conditional_obstacle'][22,32])
        absent=state.apply(prior,evidence(np.full_like(cells,-1),14.),self.association,scene_binding='absent')
        self.assertEqual(absent.layers['occupancy'][22,32],0)
        np.testing.assert_array_equal(prior.layers['occupancy'],original)
        self.assertEqual(absent.layers['revised_free_stamp'][22,32],10.)

    def test_historical_local_free_cannot_become_new_evidence_to_clear_prior(self):
        cells=np.zeros((60,60),np.int8);cells[20:25,30:35]=100
        prior=evidence(cells);prior.metadata.update(snapshot_id='static',frame_id='map')
        current=evidence(np.zeros_like(cells));current.layers['near_field_free']=np.ones(cells.shape,bool)
        revised=PriorRevisions().apply(prior,current,self.association,scene_binding='historical_only')
        self.assertEqual(revised.metadata['prior_revision']['free_corrections'],0)
        self.assertTrue((revised.layers['occupancy'][20:25,30:35]==100).all())

    def test_unknown_sparse_or_rotated_boundary_cannot_clear_historical_obstacles(self):
        cells=np.full((60,60),100,np.int8);prior=evidence(cells)
        current=evidence(np.full_like(cells,-1));current.layers['occupancy'][20,20]=0
        # One isolated free pixel does not cover an intersecting rotated target cell.
        angle=.3;self.association['map_from_local']=[[np.cos(angle),-np.sin(angle),0,0],
            [np.sin(angle),np.cos(angle),0,0],[0,0,1,0],[0,0,0,1]]
        result=PriorRevisions().apply(prior,current,self.association,scene_binding='sparse')
        self.assertEqual(result.metadata['prior_revision']['free_corrections'],0)

    def test_lifecycle_and_backward_seek_discard_free_corrections(self):
        cells=np.zeros((60,60),np.int8);cells[22,32]=100;prior=evidence(cells)
        state=PriorRevisions();free=evidence(np.zeros_like(cells))
        state.apply(prior,free,self.association,scene_binding='free')
        for stamp,generation in ((9.,0),(12.,1)):
            result=state.apply(prior,evidence(np.full_like(cells,-1),stamp,generation),
                               self.association,scene_binding='reset')
            self.assertEqual(result.layers['occupancy'][22,32],100)

    def test_reloading_another_asset_cannot_reuse_old_corrections_by_snapshot_name(self):
        cells=np.zeros((60,60),np.int8);cells[22,32]=100
        prior=evidence(cells);state=PriorRevisions()
        state.apply(prior,evidence(np.zeros_like(cells)),self.association,scene_binding='free')
        replacement=evidence(cells) # same declared name, independently loaded asset
        result=state.apply(replacement,evidence(np.full_like(cells,-1),11.),self.association,scene_binding='new_asset')
        self.assertEqual(result.layers['occupancy'][22,32],100)

    def test_small_prior_obstacle_is_conditional_but_wall_and_unknown_are_not(self):
        cells=np.zeros((60,60),np.int8);cells[22:24,28:30]=100;cells[40,:]=100;cells[0,:]=-1
        prior=evidence(cells)
        prior.layers['conditional_obstacle']=conditional_obstacles(prior.layers,self.config)
        optimistic=reference_grid(prior)
        self.assertTrue((optimistic.layers['occupancy'][22:24,28:30]==0).all())
        self.assertTrue((optimistic.layers['occupancy'][40,:]==100).all())
        self.assertTrue((optimistic.layers['occupancy'][0,:]==-1).all())
        self.assertTrue((prior.layers['occupancy'][22:24,28:30]==100).all())

    def test_conditional_global_route_is_not_reported_as_current_collision_free(self):
        cells=np.zeros((60,60),np.int8);cells[29:32,30]=100
        source=asset(cells);source.snapshot.layers['conditional_obstacle']=cells==100
        geometry=VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.eye(4),source='synthetic')
        cfg=planning_config();cfg['conditional_obstacle_cost']=1
        result=Planner(cfg,geometry).query(request(start=[2.,3.,0],goal=[4.,3.]),asset=source)
        self.assertEqual(result['status'],'path_found',result)
        self.assertTrue(result['conditional_reference'])
        self.assertFalse(result['current_path_valid']);self.assertTrue(result['needs_observation'])

    def test_assumed_free_reference_requires_observation_and_revision_preserves_provenance(self):
        cells=np.zeros((60,60),np.int8)
        source=asset(cells)
        source.snapshot.metadata.update(free_assumption='no_filtered_endpoint_in_cell',
                                        reference_policy='unobserved_cells_assumed_free')
        geometry=VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.eye(4),source='synthetic')
        config=planning_config();config['algorithm']='hybrid_astar'
        config['pose_search']['native_library']=str(Path(__file__).resolve().parents[1]/
            'outputs/runtime/build/planning/libsocial_nav_pose_search.so')
        result=Planner(config,geometry).query(request(start=[2.,3.,0],goal=[4.,3.]),asset=source)
        self.assertEqual(result['status'],'path_found',result)
        self.assertTrue(result['unobserved_free_assumed'])
        self.assertEqual(result['reference_policy'],'unobserved_cells_assumed_free')
        self.assertFalse(result['current_path_valid']);self.assertTrue(result['needs_observation'])

        prior=evidence(cells);prior.layers['assumed_free']=np.ones(cells.shape,bool)
        current=cells.copy();current[20,20]=100
        revised=PriorRevisions().apply(prior,evidence(current),self.association,scene_binding='observed')
        self.assertEqual(revised.layers['occupancy'][20,20],100)
        self.assertFalse(revised.layers['assumed_free'][20,20])
        self.assertTrue(revised.layers['assumed_free'][10,10])

    def test_manual_free_map_is_a_static_reference_and_mapped_checks_still_apply(self):
        cells = np.zeros((60, 60), np.int8)
        source = asset(cells)
        manual = np.zeros(cells.shape, bool); manual[50, 50] = True
        source.snapshot.layers['manual_edit'] = manual
        # Older edited packages have a per-cell mask but no aggregate metadata.
        geometry = VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]], np.eye(4), source='synthetic')
        planner = Planner(planning_config(), geometry)
        static = planner.query(request(), asset=source)
        self.assertEqual(static['status'], 'path_found', static)
        self.assertTrue(static['manual_free_present'])
        self.assertTrue(static['unobserved_free_assumed'])
        self.assertEqual(static['reference_policy'], 'unobserved_cells_assumed_free')
        self.assertFalse(static['current_path_valid'])
        self.assertTrue(static['global_reference_only'])
        self.assertTrue(static['needs_observation'])

        current = planner.query(request('mapped'), scene=bundle(), asset=source, now=10., clock_domain='synthetic')
        self.assertEqual(current['status'], 'path_found', current)
        self.assertTrue(current['manual_free_present'])
        self.assertTrue(current['current_path_valid'])
        self.assertEqual(current['local_check']['status'], 'free')

        # A manual occupied or unknown cell is not an assumed free cell.
        cells[50, 50] = 100
        result = planner.query(request(), asset=source)
        self.assertFalse(result['manual_free_present'])
        self.assertFalse(result['unobserved_free_assumed'])

    def test_distinct_correction_histories_have_distinct_bindings(self):
        cells=np.zeros((60,60),np.int8);cells[20,20]=cells[40,40]=100;prior=evidence(cells)
        snapshots=[]
        for y,x in ((20,20),(40,40)):
            local=np.full_like(cells,-1);local[y-4:y+5,x-4:x+5]=0
            state=PriorRevisions();state.apply(prior,evidence(local),self.association,scene_binding='first')
            snapshots.append(state.apply(prior,evidence(np.full_like(cells,-1),11.),
                                        self.association,scene_binding='same_last_scene'))
        self.assertNotEqual(snapshots[0].metadata['snapshot_id'],snapshots[1].metadata['snapshot_id'])

    def test_revision_uses_measured_time_and_cropped_views_keep_the_ledger(self):
        cells=np.zeros((60,60),np.int8);cells[20,20]=cells[40,40]=100;prior=evidence(cells)
        current=evidence(np.zeros_like(cells));current.layers['last_observed'][:]=5.
        state=PriorRevisions()
        result=state.apply(prior,current,self.association,scene_binding='observed',bounds=[1,1,3,3])
        self.assertEqual(result.metadata['shape'],[20,20])
        self.assertEqual(result.layers['revised_free_stamp'][10,10],5.)
        self.assertEqual(result.layers['last_observed'][10,10],5.)
        later=state.apply(prior,evidence(np.full_like(cells,-1),11.),self.association,scene_binding='later')
        self.assertEqual(later.layers['occupancy'][40,40],0)

    def test_global_background_and_current_collision_evidence_remain_distinct(self):
        cells=np.zeros((60,60),np.int8);prior=evidence(cells)
        local=cells.copy();local[20,20]=100;current=evidence(local)
        state=PriorRevisions()
        view=state.apply(prior,current,self.association,scene_binding='occupied')
        background=state.apply(prior,current,self.association,scene_binding='occupied',overlay_current=False)
        self.assertEqual(view.layers['occupancy'][20,20],100)
        self.assertEqual(background.layers['occupancy'][20,20],0)
        self.assertEqual(current.layers['occupancy'][20,20],100)

    def test_global_cache_reuses_geometry_but_rechecks_current_obstacles(self):
        cfg=planning_config();geometry=VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.eye(4),source='synthetic')
        planner=Planner(cfg,geometry);source=asset()
        first=planner.query(request('mapped'),scene=scene_at(9.9),asset=source,now=9.9,clock_domain='synthetic')
        second=planner.query(request('mapped'),scene=bundle(),asset=source,now=10.,clock_domain='synthetic')
        self.assertEqual(first['status'],'path_found',first)
        self.assertFalse(first['costmap_reused'])
        self.assertEqual(second['status'],'path_found',second)
        self.assertNotEqual(first['input_binding']['scene'],second['input_binding']['scene'])
        self.assertTrue(second['costmap_reused'])
        self.assertTrue(second['current_path_valid'])

        cells=np.zeros((60,60),np.int8);cells[20,20]=100
        blocked=planner.query(request('mapped'),scene=bundle(cells),asset=source,now=10.,clock_domain='synthetic')
        self.assertNotEqual(second['input_binding']['scene'],blocked['input_binding']['scene'])
        self.assertEqual(blocked['status'],'no_path',blocked)
        self.assertEqual(blocked['reason'],'start_footprint_occupied')
        self.assertFalse(blocked['current_path_valid'])
        self.assertEqual(blocked['local_start_footprint'],'occupied')


if __name__=='__main__':unittest.main()
