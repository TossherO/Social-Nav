"""Destination-floor scan provenance, geometry and observation-only runtime."""
import math
import unittest
from types import SimpleNamespace

import numpy as np
from decision.agent.core.elevator_links import destination_seed,standard_destination_seed


class ElevatorSeedTests(unittest.TestCase):
    def test_base_to_lidar_uses_selected_destination_pose_and_post_exit_scan(self):
        transition=SimpleNamespace(status='awaiting_destination_localization',task_id='t',goal_id='lift',
            from_floor='floor2',to_floor='floor1',destination_map_id='target-map',last_observation_stamp_s=10.)
        scan=dict(source='local_lidar_returns',clock_domain='ros_time',frame_id='lidar',stamp_s=11.)
        offset=np.eye(4);offset[:2,3]=[.67,.33]
        seed=destination_seed(transition,[2.,3.,math.pi/2],scan,offset)
        np.testing.assert_allclose(seed['lidar_xy_yaw'],[1.67,3.67,math.pi/2])
        self.assertFalse(seed['verified_localization'])
        with self.assertRaises(ValueError):
            destination_seed(transition,[2.,3.,0.],dict(scan,stamp_s=9.),offset)

    def test_live_handoff_seed_rejects_a_different_destination_map(self):
        from unittest.mock import patch
        from runtime.interactive import handoff_initial_pose
        hint=dict(pose=dict(quaternion_xyzw=[0.,0.,0.,1.]))
        asset=SimpleNamespace(snapshot=SimpleNamespace(metadata={'map_id':'target-map'}))
        value=dict(floor_id='floor1',lidar_xy_yaw=[2.,3.,.4],verified_localization=False,
                   source='agent_destination_scan_and_map',scan_frame='lidar',scan_stamp_s=11.,exit_stamp_s=10.,
                   destination=dict(to_floor='floor1',to_map_id='target-map'))
        with patch('runtime.interactive.selected_initial_height',return_value=dict(lidar_z_m=1.2)):
            self.assertEqual(handoff_initial_pose(value,asset,hint),[0.,0.,.4,2.,3.,1.2])
            value['destination']['to_map_id']='another-map'
            with self.assertRaises(ValueError):handoff_initial_pose(value,asset,hint)

    def test_standard_seed_preserves_all_six_rviz_values_and_checks_map_and_time(self):
        from runtime.interactive import handoff_initial_pose
        transition=SimpleNamespace(status='awaiting_destination_localization',task_id='t',goal_id='lift',
            from_floor='floor2',to_floor='floor1',destination_map_id='target-map',last_observation_stamp_s=10.)
        standard=dict(map_id='target-map',frame_id='map',reference='rslidar',pose_rpy_xyz=[.01,-.02,.3,1.,2.,-.14])
        scan=dict(source='local_lidar_returns',clock_domain='ros_time',frame_id='rslidar',stamp_s=11.)
        seed=standard_destination_seed(transition,standard,scan)
        asset=SimpleNamespace(snapshot=SimpleNamespace(metadata={'map_id':'target-map'}))
        self.assertEqual(handoff_initial_pose(seed,asset,{}),standard['pose_rpy_xyz'])
        with self.assertRaises(ValueError):standard_destination_seed(transition,dict(standard,map_id='wrong'),scan)
        with self.assertRaises(ValueError):standard_destination_seed(transition,standard,dict(scan,stamp_s=10.))
        seed['lidar_xy_yaw'][0]+=1
        with self.assertRaises(ValueError):handoff_initial_pose(seed,asset,{})

    def test_handoff_seed_is_rejected_for_execution_and_bag_before_runtime_start(self):
        import contextlib,io
        from runtime.interactive import main
        for args in (['--bag','test.bag','--agent-observations'],
                     ['--live','--agent-observations','--execute','--vehicle-profile','x','--chassis-port','x']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(args+['--output','unused','--handoff-seed','unused.json'])
