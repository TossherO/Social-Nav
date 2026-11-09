"""Geometry safety, real upstream searches, and stale-result/lifecycle boundaries."""
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from perception.environment.layers import load_config as environment_config
from perception.environment.scene import assemble_scene
from navigation.planning.inputs import load_config
from navigation.planning.footprint import VehicleGeometry, footprint_status, swept_check, path_poses, compact_poses
from navigation.planning.planner import Planner, PlanningSession
from tests.navigation_fixtures import grid, bundle, asset, request


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.geometry = VehicleGeometry([[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]], np.eye(4), source='synthetic')
        self.planner = Planner(self.config, self.geometry)

    def query(self, req=None, **kwargs):
        return self.planner.query(req or request(), **kwargs)

    def test_retained_reference_is_rechecked_and_never_claims_the_old_start_is_actual(self):
        # Use a forward-facing reference so this case exercises tail reuse,
        # independent of the grid searcher's initial cell-centre heading.
        req=request('local',goal=[4.5,2.],algorithm='hybrid_astar')
        first=self.query(req,scene=bundle(),now=10.,clock_domain='synthetic')
        self.assertEqual(first['status'],'path_found',first)
        current=grid();current.metadata['world_from_lidar'][0][3]=2.06
        current.metadata['snapshot_id']='local:new'
        scene=assemble_scene(current,clock_domain='synthetic',settings=environment_config()['scene'])
        second=self.query(req,scene=scene,now=10.,clock_domain='synthetic')
        self.assertEqual(second['status'],'path_found',second)
        continuity=second['reference_continuity']
        self.assertEqual(continuity['origin_result'],first['result_binding'])
        self.assertEqual(continuity['current_scene_binding'],scene.record['input_fingerprint'])
        self.assertFalse(continuity['start_is_actual']);self.assertFalse(second['current_path_valid'])
        self.assertTrue(second['global_reference_only']);self.assertEqual(second['start'][0],2.06)
        earlier=grid();earlier.metadata['stamp_s']=9.9
        old_scene=assemble_scene(earlier,clock_domain='synthetic',settings=environment_config()['scene'])
        reversed_time=self.query(req,scene=old_scene,now=9.9,clock_domain='synthetic')
        self.assertNotIn('reference_continuity',reversed_time)
        changed=self.query(dict(req,goal_revision=1,goal=[4.,3.]),scene=scene,now=10.,clock_domain='synthetic')
        self.assertNotIn('reference_continuity',changed)

    def test_retained_reference_does_not_clear_new_unknown_or_occupied_cells(self):
        req=request('local',goal=[4.5,2.])
        self.query(req,scene=bundle(),now=10.,clock_domain='synthetic')
        cells=np.zeros((60,60),np.int8);cells[:,30]=-1
        r=self.query(req,scene=bundle(cells),now=10.,clock_domain='synthetic')
        self.assertNotIn('reference_continuity',r)
        self.assertNotEqual(r['status'],'path_found');self.assertEqual(cells[20,30],-1)

    def test_historical_lidar_is_not_the_base_origin(self):
        geometry = VehicleGeometry.historical(self.config['geometry_reference'])
        pose = geometry.pose(np.eye(4))
        np.testing.assert_allclose(pose, [-.67, -.33, 0.])
        self.assertGreater(geometry.radius, .8)

    def test_footprint_checks_unknown_occupied_exterior_and_rotation_sweep(self):
        cells = np.zeros((60, 60), np.int8); cells[20, 22] = -1
        self.assertEqual(footprint_status(grid(cells), self.geometry, [2.1, 2.1, 0], .01), 'unknown')
        cells[20, 22] = 100
        self.assertEqual(footprint_status(grid(cells), self.geometry, [2.1, 2.1, 0], .01), 'occupied')
        self.assertEqual(footprint_status(grid(), self.geometry, [0, 0, 0], .01), 'outside')
        long = VehicleGeometry([[-.7, -.12], [.7, -.12], [.7, .12], [-.7, .12]], np.eye(4), source='synthetic')
        cells = np.zeros((60, 60), np.int8); cells[24, 24] = 100
        checked = swept_check(grid(cells), long, np.array([[2, 2, 0], [2, 2, np.pi/2]]), .01, 2000)
        self.assertEqual(checked['status'], 'occupied')

    def test_unknown_start_is_not_cleared_and_input_is_not_modified(self):
        cells = np.zeros((60, 60), np.int8); cells[20, 20] = -1; before = cells.copy()
        result = self.query(asset=asset(cells))
        self.assertEqual(result['reason'], 'start_footprint_unknown')
        np.testing.assert_array_equal(cells, before)

    def test_unknown_xy_goal_returns_serializable_unavailability(self):
        cells=np.zeros((60,60),np.int8);cells[40,40]=-1
        result=self.query(asset=asset(cells))
        self.assertEqual(result['reason'],'goal_unknown')
        self.assertIs(result['needs_observation'],True)
        json.dumps(result,allow_nan=False)

    @unittest.skipUnless(Path(load_config()['native_library']).is_file(), 'Build official Noetic bridge to run native search')
    def test_both_real_searches_path_checks_and_disconnected_unknown_barrier(self):
        cells = np.zeros((60, 60), np.int8); cells[30, :40] = 100
        for algorithm in ('dijkstra', 'astar'):
            result = self.query(request(algorithm=algorithm), asset=asset(cells))
            self.assertEqual(result['status'], 'path_found', result)
            self.assertEqual(result['footprint_check']['status'], 'free')
            self.assertFalse(result['executable'])
            self.assertEqual(result['path'][0][:2], [2., 2.])
            self.assertEqual(result['path'][-1][:2], [4., 4.])
            cells[30, :] = -1
            self.planner._cost_binding = None
            self.assertIn(self.query(request(algorithm=algorithm), asset=asset(cells))['status'], ('no_path','unavailable'))
            cells[30, :] = 0; cells[30, :40] = 100

    @unittest.skipUnless(Path(load_config()['native_library']).is_file(), 'Native bridge absent')
    def test_current_unknown_withdraws_local_validity_but_keeps_static_reference(self):
        cells = np.zeros((60, 60), np.int8); cells[30, :] = -1
        result = self.query(request('mapped'), scene=bundle(cells), asset=asset(), now=10., clock_domain='synthetic')
        self.assertEqual(result['status'], 'path_found')
        self.assertFalse(result['current_path_valid'])
        self.assertEqual(result['local_check']['status'], 'unknown')
        self.assertGreater(result['local_check']['valid_distance_m'], 0.)

    def test_scene_requires_past_fresh_pose_and_same_clock_and_map(self):
        for now, clock, expected in [(9.9, 'synthetic', 'geometry_future_or_stale'),
                                      (11., 'synthetic', 'geometry_future_or_stale'),
                                      (10., 'wrong', 'clock_domain_mismatch')]:
            result = self.query(request('mapped'), scene=bundle(), asset=asset(), now=now, clock_domain=clock)
            self.assertEqual(result['reason'], expected)
        result = self.query(request('mapped'), scene=bundle(associated=False), asset=asset(), now=10., clock_domain='synthetic')
        self.assertEqual(result['reason'], 'association_unavailable')
        wrong = asset(); wrong.snapshot.metadata['map_id'] = 'other'
        self.assertEqual(self.query(request('mapped'), scene=bundle(), asset=wrong, now=10., clock_domain='synthetic')['reason'], 'association_map_mismatch')

    def test_metadata_tampering_and_nonfinite_goal_fail_closed(self):
        scene = bundle(); scene.record['as_of'] = 12.
        self.assertEqual(self.query(request('local'), scene=scene, now=12., clock_domain='synthetic')['status'], 'invalid_input')
        # Nonfinite values must be rejected before JSON binding, with a clear exception.
        with self.assertRaises(ValueError): self.query(request(goal=[np.nan, 2.]), asset=asset())

    def test_cancellation_seek_revisions_and_late_result_cannot_restore_path(self):
        session = PlanningSession(); req = request()
        first = session.begin(req)
        result = self.query(req, asset=asset())
        session.begin(request(request_id='new'))
        self.assertFalse(session.accept(first, result))
        session.invalidate('seek'); self.assertFalse(session.accept(first, result))
        session.begin(req)
        with self.assertRaises(ValueError): session.begin(request(goal=[3., 3.]))
        session.begin(request(goal_revision=1, goal=[3., 3.]))
        with self.assertRaises(ValueError): session.begin(req)

    def test_localization_reset_requires_new_goal_revision(self):
        session = PlanningSession(); req = request('local')
        session.begin(req, bundle())
        with self.assertRaises(ValueError): session.begin(req, bundle(generation=1))
        session.begin(request('local', goal_revision=1), bundle(generation=1))
        with self.assertRaises(ValueError): session.begin(request('local', goal_revision=2), bundle())
        session.invalidate('seek')
        session.begin(req, bundle())

    def test_swept_check_budget_and_single_point_path(self):
        self.assertEqual(swept_check(grid(), self.geometry, path_poses([[2, 2]], 0.), .01, 100)['status'], 'free')
        self.assertEqual(swept_check(grid(), self.geometry, path_poses([[2, 2], [4, 4]], 0.), .01, 1)['status'], 'budget_exceeded')

    @unittest.skipUnless(Path(load_config()['native_library']).is_file(), 'Native bridge absent')
    def test_result_acceptance_binds_map_and_detects_changed_results(self):
        session = PlanningSession(); req = request(); original = asset()
        ticket = session.begin(req, asset=original)
        result = self.query(req, asset=original)
        self.assertTrue(session.accept(ticket, result))
        self.assertIsNotNone(session.current)
        changed = copy.deepcopy(result); changed['path'][0][0] += 1
        self.assertFalse(session.accept(ticket, changed))
        ticket = session.begin(req, asset=asset(np.ones((60,60),np.int8)*100))
        self.assertFalse(session.accept(ticket, result))

    def test_adaptive_sweep_preserves_real_margin_in_a_tight_straight_passage(self):
        cells=np.zeros((60,60),np.int8);cells[:,18]=100
        # Actual gap is 0.055m, enough for the requested 0.05m margin.
        poses=np.array([[2.155,2.,np.pi/2],[2.155,4.,np.pi/2]])
        self.assertEqual(swept_check(grid(cells),self.geometry,poses,.05,20000)['status'],'free')
        cells[:,19]=100
        self.assertEqual(swept_check(grid(cells),self.geometry,poses,.05,20000)['status'],'occupied')

    def test_translation_checks_the_interior_and_preserves_a_safe_prefix(self):
        cells=np.zeros((60,60),np.int8);poses=np.array([[1.,2.5,0.],[5.,2.5,0.]])
        self.assertEqual(swept_check(grid(cells),self.geometry,poses,.01,3)['status'],'free')
        cells[24:27,30]=100
        result=swept_check(grid(cells),self.geometry,poses,.01,2000)
        self.assertEqual(result['status'],'occupied')
        self.assertGreater(result['valid_distance_m'],1.)
        self.assertLess(result['valid_distance_m'],1.8)
        # Compaction must preserve a reverse segment and an in-place rotation.
        original=np.array([[1,2,0],[2,2,0],[3,2,0],[2,2,0],[2,2,1.]])
        np.testing.assert_allclose(compact_poses(original),original[[0,2,3,4]])

    @unittest.skipUnless(Path(load_config()['native_library']).is_file(), 'Native bridge absent')
    def test_local_replanning_keeps_global_goal_and_refuses_an_unknown_barrier(self):
        cells=np.zeros((60,60),np.int8);cells[18:23,35:38]=100
        self.planner.config['local_replanning']['lookahead_m']=4.
        req=request('mapped',goal=[5.2,2.])
        scene=bundle(cells)
        result=self.query(req,scene=scene,asset=asset(),now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['local_check']['status'],'occupied')
        self.assertEqual(result['local_replan']['status'],'path_found',result)
        self.assertEqual(len(result['local_replan']['request']['goal']),2)
        self.assertEqual(result['request']['goal'],req['goal'])
        current=result['current_local_path']
        self.assertEqual(current['source'],'local_replan')
        self.assertFalse(current['complete']);self.assertFalse(result['current_path_valid'])
        self.assertEqual(swept_check(scene.geometry,self.geometry,current['poses'],self.config['safety_margin_m'],10000)['status'],'free')
        config=copy.deepcopy(self.config);config['local_replanning']['lookahead_m']=5.
        hard=Planner(config,self.geometry).query(request('mapped',goal=[5.2,2.,0.]),scene=scene,asset=asset(),
                                               now=10.,clock_domain='synthetic')
        self.assertEqual(len(hard['local_replan']['request']['goal']),3)
        self.assertAlmostEqual(hard['local_replan']['request']['goal'][2],0.)
        cells[:,30]=-1
        scene=bundle(cells)
        blocked=self.query(req,scene=scene,asset=asset(),now=10.,clock_domain='synthetic')
        self.assertNotEqual(blocked['local_replan']['status'],'path_found')
        self.assertEqual(blocked['current_local_path']['source'],'static_reference_prefix')

    @unittest.skipUnless(Path(load_config()['native_library']).is_file(), 'Native bridge absent')
    def test_full_footprint_search_recovers_narrow_routes_without_clearing_unknown(self):
        geometry=VehicleGeometry.historical(self.config['geometry_reference'])
        cells=np.full((60,60),-1,np.int8);cells[20:30,5:55]=0
        req=request(start=[1.5,2.5,0.],goal=[4.,2.5,0.])
        result=Planner(self.config,geometry).query(req,asset=asset(cells))
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['search_backend'],'bounded_curvature_hybrid_astar')
        np.testing.assert_allclose(result['path'][0],req['start'])
        np.testing.assert_allclose(result['path'][-1],req['goal'])
        self.assertEqual(result['footprint_check']['status'],'free')
        moved=Planner(self.config,geometry)
        moved.query(req,asset=asset(cells))
        second=dict(req,start=[1.55,2.5,0.])
        next_result=moved.query(second,asset=asset(cells))
        self.assertEqual(next_result['status'],'path_found',next_result)
        self.assertTrue(next_result['search_attempts'][-1]['geometry_cache_reused'])
        np.testing.assert_allclose(next_result['path'][0],second['start'])
        legacy=copy.deepcopy(self.config);legacy['search_geometry']='circle';legacy['pose_search']['enabled']=False
        self.assertEqual(Planner(legacy,geometry).query(req,asset=asset(cells))['status'],'no_path')
        cells[20:30,30]=-1
        self.assertNotEqual(Planner(self.config,geometry).query(req,asset=asset(cells))['status'],'path_found')


class CausalSceneReplayTests(unittest.TestCase):
    def test_invalid_pose_clears_history_even_with_the_same_local_session(self):
        from perception.environment.replay import replay_scans
        state=[]
        class Evidence:
            def __init__(self,config):pass
            def reset(self):state.clear()
            def update(self,points,transform,stamp,identity,**kwargs):
                state.append(stamp)
                snapshot=grid();snapshot.metadata['stamp_s']=stamp
                snapshot.metadata['retained_observations']=list(state)
                return snapshot
            def close(self):pass
        rows=[dict(stamp_s=t,health=2 if t==9.9 else 0,session_id='same',generation=0,frame_id='odom',
                   child_frame_id='lidar',position=[0,0,0],quaternion_xyzw=[0,0,0,1])
              for t in [9.7,9.8,9.9,10.,10.1]]
        db=sqlite3.connect(':memory:');self.addCleanup(db.close)
        db.execute('CREATE TABLE scans(stamp REAL PRIMARY KEY,payload BLOB)')
        for stamp in [9.8,9.9,10.1]:
            buffer=io.BytesIO();np.savez(buffer,points=np.array([[2.,0,0]]),times=np.array([stamp]),frame=np.asarray('lidar'))
            db.execute('INSERT INTO scans VALUES (?,?)',(stamp,buffer.getvalue()))
        with patch('perception.environment.replay.LocalEnvironmentMap',Evidence):
            result=list(replay_scans(db,rows,{'algorithm_parameters':{'preprocess':{'blind':1.}}},
                                     environment_config(),9.7,10.1))
        self.assertIsNone(result[1][0])
        self.assertEqual(result[-1][0].metadata['retained_observations'],[10.1])
        self.assertEqual(result[-1][2]['invalid_scans'],1)

    def test_raw_deskew_uses_measured_past_poses_and_excludes_future_firings(self):
        from perception.environment.replay import rebuild_scenes
        from scipy.spatial.transform import Rotation
        from perception.environment.scene import SceneInputs
        captured=[]
        class Evidence:
            def __init__(self,config):pass
            def reset(self):pass
            def update(self,points,transform,stamp,identity,**kwargs):
                captured.append((points.copy(),kwargs['ray_origins'].copy()))
                result=grid();result.metadata['world_from_lidar']=transform.tolist()
                return result
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            rows=[]
            for t,x,yaw in [(9.8,1.8,0.),(9.9,1.9,np.pi/4),(10.,2.,np.pi/2),(10.1,100.,np.pi)]:
                rows.append(dict(stamp_s=t,health=0,session_id='test',generation=0,frame_id='odom',child_frame_id='lidar',
                                 position=[x,2.,0.],quaternion_xyzw=Rotation.from_euler('z',yaw).as_quat().tolist()))
            (run/'localization.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (run/'run.json').write_text(json.dumps(dict(success=True,source='bag',bag='synthetic-not-opened',
                playback=dict(bag_fingerprint=dict(kind='synthetic')),algorithm_parameters=dict(preprocess=dict(blind=1.)))))
            original=grid();transform=np.asarray(original.metadata['world_from_lidar'])
            transform[:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix();original.metadata['world_from_lidar']=transform.tolist()
            scene=assemble_scene(original,clock_domain='synthetic',settings=environment_config()['scene']);scene.save(root/'scene')
            def raw(command,**kwargs):
                db=sqlite3.connect(command[-1]);db.execute('CREATE TABLE scans(stamp REAL PRIMARY KEY,payload BLOB)')
                stream=io.BytesIO();np.savez(stream,points=np.array([[2.,0,0],[2.,0,0],[.2,0,0]]),
                    times=np.array([9.9,10.01,9.9]),frame=np.asarray('lidar'))
                db.execute('INSERT INTO scans VALUES (?,?)',(10.,stream.getvalue()));db.commit();db.close()
            with patch('perception.environment.replay.subprocess.run',side_effect=raw),patch('perception.environment.replay.LocalEnvironmentMap',Evidence):
                rebuild_scenes(run,[root/'scene'],root/'out')
            self.assertEqual(len(captured),1)
            np.testing.assert_allclose(captured[0][0],[[np.sqrt(2),.1-np.sqrt(2),0]],atol=1e-6)
            np.testing.assert_allclose(captured[0][1],[[1.9,2.,0]],atol=1e-6)
            rebuilt=SceneInputs.load(root/'out/scene')
            self.assertEqual(rebuilt.record['derivation']['future_firings_excluded'],1)
            self.assertNotEqual(rebuilt.record['input_fingerprint'],scene.record['input_fingerprint'])


if __name__ == '__main__': unittest.main()
