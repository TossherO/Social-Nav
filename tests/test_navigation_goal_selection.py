"""Endpoint proofs and bounded route-owned alternatives, with no map relabeling."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from perception.environment.context import fingerprint
from navigation.planning.footprint import VehicleGeometry,goal_obstruction,footprint_status
from navigation.planning.inputs import load_config
from navigation.planning.planner import Planner
from navigation.planning.reference import ReferenceRuntime
from navigation.planning.continuity import reference_envelope
from tests.navigation_fixtures import grid,bundle,request
from tests import test_route_replay as route_fixtures
from tests import test_navigation_dwa as dwa_fixtures
from tests.navigation_fixtures import FakeWorker


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config();self.cfg['algorithm']='hybrid_astar'
        self.geometry=VehicleGeometry([[-.3,-.2],[.7,-.2],[.7,.2],[-.3,.2]],np.eye(4),source='synthetic')

    def test_xy_blocked_disk_rejects_before_costmap_and_search(self):
        for value in (-1,100):
            cells=np.zeros((60,60),np.int8);cells[20,41]=value
            with patch('navigation.planning.costmap.build_costmap',side_effect=AssertionError('Unnecessary full-grid work')):
                result=Planner(self.cfg,self.geometry).query(request('local',goal=[4.,2.]),scene=bundle(cells),now=10.,clock_domain='synthetic')
            self.assertEqual(result['reason'],'goal_footprint_'+('unknown' if value==-1 else 'occupied'))
            self.assertNotIn('search_attempts',result);self.assertEqual(cells[20,41],value)

    def test_failure_of_one_heading_cannot_reject_free_heading_goal(self):
        cells=np.zeros((60,60),np.int8);cells[20,25]=100;snapshot=grid(cells)
        self.assertEqual(goal_obstruction(snapshot,self.geometry,[2.,2.,0.],.05),'occupied')
        self.assertIsNone(goal_obstruction(snapshot,self.geometry,[2.,2.],.05))
        self.assertEqual(footprint_status(snapshot,self.geometry,[2.,2.,np.pi/2],.05),'free')

    def test_exact_goal_position_is_not_the_cell_centre(self):
        cells=np.zeros((60,60),np.int8);cells[20,23]=100;snapshot=grid(cells)
        self.assertIsNone(goal_obstruction(snapshot,self.geometry,[2.01,2.05],.05))
        self.assertEqual(goal_obstruction(snapshot,self.geometry,[2.09,2.05],.05),'occupied')

    def test_offset_base_and_map_edge_do_not_invent_an_inner_disk(self):
        offset=VehicleGeometry(self.geometry.footprint+[1.,0.],np.eye(4),source='offset')
        self.assertIsNone(goal_obstruction(grid(),offset,[0.,2.],.05))
        self.assertEqual(goal_obstruction(grid(),self.geometry,[.1,2.],.05),'outside')

    def test_route_budget_never_increases_global_budget_and_detour_is_explicit(self):
        base=request('local',goal=[4.,2.],route_binding='synthetic',route_segment_id='segment')
        planner=Planner(self.cfg,self.geometry,require_actual_start=True)
        selection=dict(search_budget_s=.05,max_path_length_m=1.)
        result=planner.query(dict(base,route_goal_selection=selection),scene=bundle(),now=10.,clock_domain='synthetic')
        self.assertEqual(result['reason'],'route_candidate_detour');self.assertEqual(result['path'],[])
        result=planner.query(dict(base,route_goal_selection=dict(selection,search_budget_s=2.)),scene=bundle(),now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'invalid_input')
        self.assertEqual(planner.config['pose_search']['max_search_s'],.8)


class RouteSelectionTests(unittest.TestCase):
    def setUp(self):
        self.f=route_fixtures.RouteReplayTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.record['segments'][0].update(points_xy=[[2,2],[10,2]],source=dict(start_offset_s=9,end_offset_s=30))
        self.f.record['segments'][1]['source']=dict(start_offset_s=30,end_offset_s=40)

    def failure(self,req):
        return dict(request_binding=fingerprint(req),reason='returned_path_search_budget_exceeded',as_of=10.,job_id=1)

    def step(self,replay,stamp,*,failure=None,**kw):
        scene=route_fixtures.scene_at(stamp,**kw)
        state=dict(stamp_s=stamp,health=0,session_id='test',generation=0,frame_id='odom')
        return replay.step(scene,state,now=stamp,clock_domain='synthetic',failure=failure)

    def test_failures_try_finite_shorter_goals_with_new_revisions_and_shared_budget(self):
        replay=self.f.replay();req=self.step(replay,10.)[1];requests=[req]
        for i in range(5):
            req=self.step(replay,10.1+i*.1,failure=self.failure(req))[1];requests.append(req)
        self.assertEqual([r['goal'][0] for r in requests],[5.,4.5,4.,3.5,3.,2.5])
        self.assertEqual([r['goal_revision'] for r in requests],list(range(6)))
        self.assertAlmostEqual(sum(r['route_goal_selection']['search_budget_s'] for r in requests),.8)
        self.assertIsNone(self.step(replay,10.7,failure=self.failure(req)))
        self.assertIsNone(self.step(replay,13.)) # A short goal does not oscillate back to the far one.
        self.assertFalse(replay.summary()['task_completed'])

    def test_late_failure_for_old_goal_is_ignored(self):
        replay=self.f.replay();first=self.step(replay,10.)[1]
        second=self.step(replay,10.1,failure=self.failure(first))[1]
        self.assertIsNone(self.step(replay,10.2,failure=self.failure(first)))
        self.assertEqual(replay.current,second)

    def test_restored_endpoints_do_not_restart_exhausted_alternatives(self):
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config as environment_config
        replay=self.f.replay();req=None
        for i in range(6):
            stamp=10.+i*.1;scene=route_fixtures.scene_at(stamp)
            scene.geometry.layers['occupancy'][:,42:]=-1
            scene=assemble_scene(scene.geometry,clock_domain='synthetic',settings=environment_config()['scene'])
            state=dict(stamp_s=stamp,health=0,session_id='test',generation=0,frame_id='odom')
            action=replay.step(scene,state,now=stamp,clock_domain='synthetic',failure=None if req is None else self.failure(req))
            if action is not None:req=action[1]
        self.assertFalse(replay.pending_goals)
        self.assertIsNone(self.step(replay,13.,failure=self.failure(req)))
        self.assertEqual(replay.current,req)
        self.assertIsNone(self.step(replay,16.,failure=self.failure(req)))
        # Physical progress, rather than a map timestamp, advances the route.
        advanced=self.step(replay,16.1,x=5.1)[1]
        self.assertGreater(advanced['goal'][0],5.1)
        self.assertGreater(advanced['goal_revision'],req['goal_revision'])

    def test_stale_scene_cannot_switch_a_failed_goal(self):
        replay=self.f.replay();req=self.step(replay,10.)[1]
        scene=route_fixtures.scene_at(10.)
        state=dict(stamp_s=11.,health=0,session_id='test',generation=0,frame_id='odom')
        self.assertIsNone(replay.step(scene,state,now=11.,clock_domain='synthetic',failure=self.failure(req)))
        self.assertEqual(replay.current,req);self.assertEqual(replay.cancels,0)

    def test_stop_endpoint_is_not_replaced_by_an_intermediate_goal(self):
        self.f.record['segments'][0].update(points_xy=[[2,2],[5,2]],goal_behavior='stop')
        replay=self.f.replay();req=self.step(replay,10.)[1]
        self.assertEqual(req['goal_behavior'],'stop');self.assertFalse(replay.pending_goals)
        self.assertIsNone(self.step(replay,10.1,failure=self.failure(req)))
        self.assertEqual(replay.current['goal'],[5.,2.])

    def test_passed_intermediate_goal_advances_without_waiting_for_hysteresis(self):
        replay=self.f.replay();req=self.step(replay,10.)[1]
        new=self.step(replay,10.1,x=5.1)[1]
        self.assertGreater(new['goal'][0],5.1);self.assertGreater(new['goal_revision'],req['goal_revision'])

    def test_reference_failure_handoff_keeps_original_request_and_observation(self):
        f=dwa_fixtures.DwaTests();f.setUp()
        runtime=ReferenceRuntime(f.cfg,f.motion,f.geometry,worker_factory=FakeWorker);self.addCleanup(runtime.close)
        runtime.update(f.req,f.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
        failed=copy.deepcopy(f.reference['result']);failed.update(status='no_path',reason='goal_footprint_occupied',path=[])
        failed.pop('result_binding');failed['result_binding']=fingerprint(failed)
        runtime.worker.reply(reference_envelope(failed,f.req,f.scene,None));runtime.tick(now=10.,clock_domain='synthetic')
        self.assertEqual(runtime.failure['request_binding'],fingerprint(f.req));self.assertEqual(runtime.failure['as_of'],10.)
        self.assertEqual(runtime.failure['reason'],'goal_footprint_occupied')
        # Retry the same request after backoff, using a fresh observation. The
        # route layer need not manufacture a new goal revision to recover.
        new_scene=route_fixtures.scene_at(10.1)
        runtime.update(f.req,new_scene,None,now=10.1)
        with patch('navigation.planning.reference.time.monotonic',return_value=runtime.next_retry+.01):
            runtime.latest['received_mono']=runtime.next_retry
            runtime.tick(now=10.1,clock_domain='synthetic')
        self.assertEqual(runtime.worker.job['request'],f.req)
        self.assertEqual(runtime.worker.job['scene'].record['as_of'],10.1)
        runtime.invalidate('goal_changed');self.assertIsNone(runtime.failure)


if __name__=='__main__':unittest.main()
