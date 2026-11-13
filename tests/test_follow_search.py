"""Bounded search candidates never treat unknown geometry as free."""
import unittest
import numpy as np

from navigation.planning.footprint import VehicleGeometry
from navigation.search import candidates,check_pose,bounds_status
from perception.environment.context import fingerprint
from tests.navigation_fixtures import asset,bundle


class FollowSearchGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.inputs=dict(global_grid=asset().snapshot,scene=bundle(),pose=[2.,2.,0.],local_pose=[2.,2.,0.],
                         stamp_s=10.,clock_domain='synthetic',geometry=self.geometry)
        self.last=dict(identity=['tracker',1,'old'],stamp_s=9.9,clock_domain='synthetic',
                       local_identity=['test',0,'odom'],position_xy_m=[4.,2.],velocity_xy_mps=[.2,0.])
        self.last['binding']=fingerprint(self.last)

    def test_candidates_include_only_short_checked_poses(self):
        result=candidates(self.inputs,self.last,now=10.,clock_domain='synthetic',
                          local_identity=['test',0,'odom'],anchor=[2.,2.])
        self.assertEqual(result[0]['kind'],'wait')
        self.assertEqual({x['kind'] for x in result},{'wait','rotate','approach'})
        self.assertTrue(all(x.get('check',{}).get('valid',True) for x in result))
        for item in result[1:]:self.assertLessEqual(np.linalg.norm(np.asarray(item['pose'][:2])-[2.,2.]),.8)

    def test_unknown_local_cells_block_approach_and_stale_geometry_is_refused(self):
        cells=np.zeros((60,60),np.int8);cells[20,25]=-1
        self.inputs['scene']=bundle(cells)
        result=candidates(self.inputs,self.last,now=10.,clock_domain='synthetic',
                          local_identity=['test',0,'odom'],anchor=[2.,2.])
        self.assertNotIn('approach',[x['kind'] for x in result])
        with self.assertRaisesRegex(ValueError,'unavailable'):
            candidates(self.inputs,self.last,now=10.6,clock_domain='synthetic',
                       local_identity=['test',0,'odom'],anchor=[2.,2.])

    def test_foreign_history_cannot_suggest_approach(self):
        result=candidates(self.inputs,self.last,now=10.,clock_domain='synthetic',
                          local_identity=['other',0,'odom'],anchor=[2.,2.])
        self.assertNotIn('approach',[x['kind'] for x in result])
        with self.assertRaisesRegex(ValueError,'budget'):
            check_pose(self.inputs,[4.,2.,0.],now=10.,clock_domain='synthetic',anchor=[2.,2.])

    def test_independent_search_radius_checks_current_mapped_motion(self):
        from navigation.planning.inputs import load_config
        from navigation.planning.continuity import goal_binding
        static=asset();request=dict(task_id='search',goal_revision=1,mode='mapped',frame_id='map',
            target_reference='base_link',goal=[2.5,2.,0.],map_snapshot_id='static-a',
            search_bounds=dict(anchor_xy_m=[2.,2.],radius_m=1.5,map_snapshot_id='static-a',loss_binding='loss'))
        self.assertIsNone(bounds_status(request,bundle(),static,load_config(),[[2.,2.,0.],[3.4,2.,0.]],margin=.05,geometry=self.geometry))
        self.assertEqual(bounds_status(request,bundle(),static,load_config(),[[2.,2.,0.],[3.48,2.,0.]],margin=.05,geometry=self.geometry),
                         'follow_search_radius_exceeded')
        modified=dict(request,search_bounds=dict(request['search_bounds'],radius_m=1.))
        self.assertNotEqual(goal_binding(request),goal_binding(modified))
        self.assertEqual(bounds_status(request,bundle(associated=False),static,load_config(),[[2.,2.,0.]],margin=0.,geometry=self.geometry),
                         'follow_search_map_association_invalid')

    def test_safety_rejects_motion_outside_search_radius(self):
        from navigation.planning.motion import load_motion_config
        from navigation.safety.guard import LocalSafety
        static=asset();scene=bundle();motion=load_motion_config();motion['collision_backend']='cpu'
        request=dict(task_id='search',request_id='r',goal_revision=1,mode='mapped',frame_id='map',
            target_reference='base_link',goal=[2.5,2.,0.],map_snapshot_id='static-a',
            search_bounds=dict(anchor_xy_m=[2.,2.],radius_m=.1,map_snapshot_id='static-a',loss_binding='loss'))
        feedback=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',
                      local_identity=['test',0,'odom'],pose=[2.,2.,0.],velocity=[0.,0.])
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.36,0.],request_binding=fingerprint(request),
                      scene_binding=scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic')
        gate=LocalSafety(self.geometry,motion,.05,simulation=True,asset=static)
        result=gate.gate(proposal,request,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['reason'],'follow_search_radius_exceeded')
        self.assertEqual(result['velocity'],[0.,0.])


if __name__=='__main__':unittest.main()
