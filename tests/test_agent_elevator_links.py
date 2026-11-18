"""User-confirmed elevator correspondence is a hint, not a localization."""
import unittest
from types import SimpleNamespace
import numpy as np

from decision.agent.core.elevator_links import candidate_links,fit_floor_transform,transform_seed_pose,capture_source_pose,destination_lidar_seed


def catalog(floor, positions):
    return SimpleNamespace(value={'floor_id':floor},fingerprint=floor+'-catalog',places={k:dict(pose=v) for k,v in positions.items()})


class ElevatorLinkTests(unittest.TestCase):
    def setUp(self):
        self.floor2=catalog('floor2',{'e1':[0,0],'e2':[3,0]})
        self.floor1=catalog('floor1',{'d1':[10,10],'d2':[13,10]})
        self.links=[dict(floor2='e1',floor1='d1'),dict(floor2='e2',floor1='d2')]

    def test_unique_is_only_a_hint_and_keeps_destination_coordinates(self):
        result=candidate_links(self.links,self.floor2,self.floor1,[.2,0.,0.],max_distance_m=2.,ambiguity_m=.5)
        self.assertEqual(result['status'],'unique_candidate')
        self.assertEqual(result['candidates'][0]['destination_reference_xy'],[10,10])
        self.assertFalse(result['verified_elevator'])
        self.assertFalse(result['verified_pose'])

    def test_ambiguous_and_missing_are_not_assumed(self):
        self.assertEqual(candidate_links(self.links,self.floor2,self.floor1,[1.5,0.,0.],
                                         max_distance_m=3.,ambiguity_m=.2)['status'],'ambiguous')
        self.assertEqual(candidate_links(self.links,self.floor2,self.floor1,[8.,0.,0.],
                                         max_distance_m=2.,ambiguity_m=.2)['status'],'unavailable')

    def test_floor_transform_is_a_seed_and_preserves_orientation(self):
        result=fit_floor_transform(self.links,self.floor2,self.floor1)
        self.assertLess(result['max_residual_m'],1e-6)
        self.assertFalse(result['verified_localization'])
        pose=transform_seed_pose(result,[1.,0.,.25])
        self.assertAlmostEqual(pose[0],11.)
        self.assertAlmostEqual(pose[1],10.)
        self.assertAlmostEqual(pose[2],.25)
        with self.assertRaises(ValueError):fit_floor_transform(self.links[:1],self.floor2,self.floor1)

    def test_rotation_and_inverse_use_floor_coordinates(self):
        a=catalog('floor2',{'e1':[0.,0.],'e2':[3.,0.],'e3':[0.,2.]})
        b=catalog('floor1',{'d1':[10.,5.],'d2':[10.,8.],'d3':[8.,5.]})
        links=[dict(floor2='e'+str(i),floor1='d'+str(i)) for i in (1,2,3)]
        forward=fit_floor_transform(links,a,b);back=fit_floor_transform(links,b,a)
        pose=[1.,.5,.3]
        mapped=transform_seed_pose(forward,pose)
        self.assertAlmostEqual(mapped[0],9.5);self.assertAlmostEqual(mapped[1],6.)
        restored=transform_seed_pose(back,mapped)
        np.testing.assert_allclose(restored,pose,atol=1e-8)

    def test_source_pose_is_bound_and_lidar_offset_is_applied(self):
        self.floor2.value.update(localization_map_id='source-map',frame_id='map')
        state=dict(localization_status='valid',input_stale=False,origin_pose=[1.,0.,0.],origin_frame_id='map',
                   stamp_s=10.1,origin_stamp_s=10.,clock_domain='runtime_live',session_id='s',generation=0,frame_id='odom',
                   map_binding=dict(valid=True,map_id='source-map',map_frame='map'))
        source=capture_source_pose(state,self.floor2)
        transform=fit_floor_transform(self.links,self.floor2,self.floor1)
        offset=np.eye(4);offset[0,3]=.67;offset[1,3]=.33
        seed=destination_lidar_seed(transform,source,offset)
        self.assertAlmostEqual(seed['lidar_xy_yaw'][0],11.67)
        self.assertAlmostEqual(seed['lidar_xy_yaw'][1],10.33)
        self.assertFalse(seed['verified_localization'])
        state['map_binding']['map_id']='wrong'
        with self.assertRaises(ValueError):capture_source_pose(state,self.floor2)

    def test_live_handoff_seed_rejects_a_different_destination_map(self):
        from unittest.mock import patch
        from runtime.interactive import handoff_initial_pose
        hint=dict(pose=dict(quaternion_xyzw=[0.,0.,0.,1.]))
        asset=SimpleNamespace(snapshot=SimpleNamespace(metadata={'map_id':'target-map'}))
        value=dict(floor_id='floor1',lidar_xy_yaw=[2.,3.,.4],verified_localization=False,
                   floor_transform=dict(to_floor='floor1',to_map_id='target-map'))
        with patch('runtime.interactive.selected_initial_height',return_value=dict(lidar_z_m=1.2)):
            self.assertEqual(handoff_initial_pose(value,asset,hint),[0.,0.,.4,2.,3.,1.2])
            value['floor_transform']['to_map_id']='another-map'
            with self.assertRaises(ValueError):handoff_initial_pose(value,asset,hint)

    def test_handoff_seed_is_rejected_for_execution_and_bag_before_runtime_start(self):
        import contextlib,io
        from runtime.interactive import main
        for args in (['--bag','test.bag','--agent-observations'],
                     ['--live','--agent-observations','--execute','--vehicle-profile','x','--chassis-port','x']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(args+['--output','unused','--handoff-seed','unused.json'])
