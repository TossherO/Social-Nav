"""A floor's fixed planning raster and projected obstacle indices travel together."""
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import yaml

from navigation.planning.continuity import local_goal
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import cell_of, load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.planner import Planner, PlanningSession, limit_local_grid
from navigation.planning.reference import ReferenceRuntime
from navigation.planning.runtime import ReferenceQuery, Worker, decode_display_map
from perception.environment.layers import GridSnapshot
from perception.environment.storage import StaticEnvironmentMap, content_hash
from tests.navigation_fixtures import FakeWorker, bundle, request


class NavigationWorkExtentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config['algorithm'] = 'hybrid_astar'
        self.geometry = VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='test')
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        cells = np.full((100,110), -1, np.int8)
        cells[30:61,30:61] = 0
        metadata = dict(shape=list(cells.shape), resolution=.1, origin_xy=[-2.,-2.],
                        frame_id='map', map_id='map-a', snapshot_id='static-a')
        layer = self.directory/'layers.npz'
        GridSnapshot(metadata,dict(occupancy=cells)).save(layer)
        manifest = dict(kind='static_environment', time_scope='offline_final_prior',
                        snapshot_id='static-a', frame_id='map', layers_sha256=content_hash(layer))
        (self.directory/'manifest.yaml').write_text(yaml.safe_dump(manifest))
        self.asset = StaticEnvironmentMap(self.directory)
        self.scene = bundle()

    def test_outside_goals_and_starts_cannot_be_replaced_by_local_or_follow_plans(self):
        planner = Planner(self.config,self.geometry)
        valid = request('mapped',goal=[3.5,2.,0.])
        first = planner.query(valid,scene=self.scene,asset=self.asset,now=10.,clock_domain='synthetic')
        self.assertEqual(first['status'],'path_found',first['reason'])
        self.assertEqual(first['path'][-1],[3.5,2.,0.])
        cases = (
            (request('static',goal=[5.3,2.]), 'goal_outside'),
            (request('static',start=[-.3,2.,0.],goal=[3.5,2.]), 'start_outside'),
            (request('mapped',goal=[5.3,2.,0.]), 'goal_outside'),
            (request('mapped',goal=[5.1,2.]), 'goal_outside'),
            (request('mapped',goal=[5.,2.]), 'goal_footprint_outside'),
            (request('mapped',goal=[5.3,2.,0.],goal_behavior='stop',use_local_approach=True), 'goal_outside'),
            (request('local',goal=[5.3,2.],completion_policy='continuous'), 'goal_outside'),
        )
        for goal, reason in cases:
            with self.subTest(goal=goal):
                scene = None if goal['mode']=='static' else self.scene
                result = planner.query(goal,asset=self.asset,scene=scene,now=10.,clock_domain='synthetic')
                self.assertEqual((result['status'],result['reason'],result['path']),('no_path',reason,[]))
                self.assertNotIn('follow_goal_adjustment',result)
                self.assertNotIn('reference_continuity',result)
        local = planner.query(request('local',goal=[5.3,2.]),asset=self.asset,
                              scene=self.scene,now=10.,clock_domain='synthetic')
        self.assertEqual(local['status'],'path_found',local['reason'])
        with self.assertRaisesRegex(ValueError,'goal_outside'):
            local_goal(cases[-1][0],self.scene,self.geometry,self.asset,self.config)

    def test_retained_cells_and_costs_belong_to_the_floor_and_window(self):
        cells = np.zeros((60,60),np.int8)
        cells[30,31] = cells[20,55] = 100
        scene = bundle(cells)
        with ReferenceRuntime(self.config,load_motion_config(),self.geometry,worker_factory=FakeWorker) as runtime:
            runtime.observe(scene,self.asset,now=10.)
            blocked = runtime._retained_obstacles(scene,self.asset)
            self.assertEqual(len(blocked),1)
            grid = self.asset.navigation_snapshot
            x,y = cell_of(grid,[3.15,3.05])
            self.assertEqual(blocked,(int(y*grid.metadata['shape'][1]+x),))
            larger = StaticEnvironmentMap(self.directory,navigation_window=(10,10,80,80))
            self.assertEqual(runtime._retained_obstacles(scene,larger),())
            self.assertEqual(runtime._retained_obstacles(bundle(generation=1),self.asset),())
            planner = Planner(self.config,self.geometry)
            goal = request('mapped',goal=[3.5,2.,0.])

            def plan(asset, obstacles):
                return planner.query(goal,asset=asset,scene=scene,now=10.,clock_domain='synthetic',
                                     reference_obstacles=obstacles)

            a = plan(self.asset,blocked)
            b = plan(self.asset,blocked)
            self.assertFalse(a['reference_work_reused'])
            self.assertTrue(b['reference_work_reused'])
            self.assertTrue(b['costmap_reused'])
            c = plan(larger,())
            self.assertFalse(c['reference_work_reused'])
            self.assertFalse(c['costmap_reused'])
            self.assertNotIn('reference_continuity',c)
            self.assertNotEqual(a['input_binding']['navigation_grid'],c['input_binding']['navigation_grid'])
            session = PlanningSession()
            self.assertFalse(session.accept(session.begin(goal,scene,larger),a))

    def test_mapped_approach_projects_retained_cells_without_changing_local_evidence(self):
        grid = self.asset.navigation_snapshot
        x,y = cell_of(grid,[3.15,3.05])
        blocked = (int(y*grid.metadata['shape'][1]+x),)
        local = self.scene.geometry
        transform = np.array([[0.,-1.,5.],[1.,0.,1.],[0.,0.,1.]])
        constrained = limit_local_grid(local,grid,transform,blocked)
        x,y = cell_of(local,[2.05,1.85])
        self.assertEqual(constrained.layers['occupancy'][y,x],100)
        self.assertEqual(constrained.layers['occupancy'][20,50],100)
        self.assertFalse(local.layers['occupancy'].any())
        self.assertNotEqual(constrained.metadata['snapshot_id'],local.metadata['snapshot_id'])
        self.assertEqual(constrained.metadata['work_extent']['work_snapshot_id'],grid.metadata['snapshot_id'])

    def test_new_outside_follow_goal_clears_a_compatible_active_reference(self):
        goal = request('local',goal=[4.7,2.],completion_policy='continuous',valid_until_s=11.)
        with ReferenceRuntime(self.config,load_motion_config(),self.geometry,worker_factory=FakeWorker) as runtime:
            self.assertTrue(runtime.update(goal,self.scene,None,self.asset,now=10.,follow_key='track',clock_rate=.001))
            runtime.tick(now=10.,clock_domain='synthetic')
            query = ReferenceQuery(runtime.planning,self.geometry)
            active = query.query(runtime.worker.job,self.scene,self.asset)
            runtime.worker.reply(active)
            self.assertEqual(runtime.tick(now=10.,clock_domain='synthetic')['status'],'reference_ready')
            outside = dict(goal,goal_revision=1,goal=[5.15,2.])
            self.assertFalse(runtime.update(outside,self.scene,None,self.asset,now=10.,follow_key='track'))
            status = runtime.tick(now=10.,clock_domain='synthetic')
            self.assertEqual(status['reason'],'goal_outside')
            self.assertIsNone(status['result'])

    def test_worker_reloads_same_source_with_new_window_and_keeps_cell_indices_aligned(self):
        worker = Worker(self.config,self.geometry)
        self.addCleanup(worker.stop)
        deadline = time.monotonic()+10.
        while not worker.ready:
            worker.poll()
            if time.monotonic()>deadline:self.fail('Reference worker did not start')
            time.sleep(.01)
        seen = []
        for job_id, asset in enumerate((self.asset,StaticEnvironmentMap(self.directory,navigation_window=(10,10,80,80)))):
            point = [3.15,3.05]
            grid = asset.navigation_snapshot
            x,y = cell_of(grid,point)
            job = dict(job_id=job_id,epoch=0,request=request('mapped',goal=[3.5,2.,0.]),scene=self.scene,
                       asset=asset,sent_mono=time.monotonic(),now=10.,clock_rate=.001,display_enabled=True,
                       reference_obstacles=(int(y*grid.metadata['shape'][1]+x),))
            worker.send(job)
            deadline = time.monotonic()+5.
            reply = None
            while reply is None:
                reply = worker.poll()
                if time.monotonic()>deadline:self.fail('Reference worker did not reply')
                time.sleep(.01)
            reply = reply[1]
            self.assertIsNone(reply['error'])
            result = reply['result']['result']
            self.assertEqual(result['status'],'path_found',result['reason'])
            self.assertFalse(result['reference_work_reused'])
            display = reply['display_map']
            self.assertEqual(display['origin_xy'],tuple(grid.metadata['origin_xy']))
            occupancy = decode_display_map(display)[0]
            self.assertEqual(occupancy.shape,tuple(grid.metadata['shape']))
            self.assertEqual(occupancy[y,x],100)
            seen.append((int(x),int(y)))
        self.assertNotEqual(*seen)
