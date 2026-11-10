"""Vehicle-specific motion and serial fields; synthetic profiles only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from navigation.control.chassis import DriveState
from navigation.control.chassis_protocol import decode_feedback,encode_command_units
from navigation.control.vehicle import controller_settings,load_vehicle_profile
from navigation.planning.motion import load_motion_config
from navigation.planning.inputs import load_config as planning_config
from navigation.safety.config import load_safety_config
from tests.chassis_fixtures import frame


class VehicleExecutionConfigTests(unittest.TestCase):
    def setUp(self):
        self.motion=load_motion_config(profile='trial')
        # Keep the synthetic low-speed and braking boundary independent of
        # changes to the actual wheelchair trial profile.
        self.motion.update(max_forward_mps=.1,max_reverse_mps=.05,max_angular_rps=.2)
        self.motion['max_acceleration_mps2']=.1
        self.motion['max_angular_acceleration_rps2']=.15

    def profile(self):
        p=yaml.safe_load(Path('navigation/control/vehicle.example.yaml').read_text())
        p.update(vehicle_id='synthetic_only',protocol='af01_12_bf01_32_payload_sum_mode1_3')
        p['verification']=dict(source='synthetic unit test; no physical acceptance',
            geometry_confirmed=True,motion_confirmed=True,protocol_confirmed=True)
        p['geometry']=dict(footprint=[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],base_from_lidar=np.eye(4).tolist())
        p['measured_limits']={k:self.motion[k] for k in ('max_forward_mps','max_reverse_mps','max_angular_rps',
            'max_acceleration_mps2','max_angular_acceleration_rps2')}
        p['controller']=dict(velocity_limit_field=1500,feedback_max_linear_mps=.15,feedback_max_angular_rps=.3)
        p['execution_max_longitudinal_speed_mps']=.10
        return p

    def load(self,p):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vehicle.yaml';path.write_text(yaml.safe_dump(p))
            return load_vehicle_profile(path,self.motion,require_execution=True)

    def test_safety_and_dwa_share_prediction_horizon(self):
        from navigation.control.dwa import DwaController
        from navigation.safety.guard import LocalSafety
        geometry,motion,_=self.load(self.profile())
        planning=planning_config()
        dwa=DwaController(planning,motion,geometry,execution=True)
        guard=LocalSafety(geometry,motion,planning['safety_margin_m'],execution=True)
        self.assertEqual(dwa.settings['prediction_s'],guard.settings['prediction_s'])
        self.assertEqual(dwa.settings['sample_s'],guard.settings['sample_s'])
        self.assertEqual(guard.settings,load_safety_config())

    def test_wheelchair_trial_profile_loads_for_execution(self):
        trial=load_motion_config(profile='trial')
        _,selected,profile=load_vehicle_profile('configs/wheelchair_vehicle.yaml',trial,require_execution=True)
        self.assertEqual(selected['execution_max_longitudinal_speed_mps'],profile['execution_max_longitudinal_speed_mps'])
        self.assertEqual(selected['execution_max_angular_rps'],profile['measured_limits']['max_angular_rps'])

    def test_legacy_speed_field_migrates_without_changing_its_value(self):
        p=self.profile();p['execution_max_planar_speed_mps']=p.pop('execution_max_longitudinal_speed_mps')
        _,motion,_=self.load(p)
        self.assertEqual(motion['execution_max_longitudinal_speed_mps'],.1)
        p['execution_max_longitudinal_speed_mps']=.2
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.load(p)

    def test_unconfirmed_profile_and_invalid_limits_are_rejected(self):
        for field in ('motion_confirmed','protocol_confirmed'):
            p=self.profile();p['verification'][field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):self.load(p)
        for value in (None,0.,float('nan'),True,.09):
            p=self.profile();p['execution_max_longitudinal_speed_mps']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.load(p)
        p=self.profile();p['measured_limits']['max_forward_mps']=.15
        with self.assertRaises(ValueError):self.load(p)

    def test_real_execution_needs_explicit_valid_controller_fields(self):
        for value in (None,{},dict(velocity_limit_field=0,feedback_max_linear_mps=.15,feedback_max_angular_rps=.3),
                dict(velocity_limit_field=1500,feedback_max_linear_mps=.01,feedback_max_angular_rps=.3)):
            p=self.profile();p['controller']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.load(p)

    def test_wire_limit_is_distinct_from_host_cap_and_watchdog_is_unchanged(self):
        _,motion,_=self.load(self.profile())
        state=DriveState(motion);state.state='armed';state.phase=0.;state.heartbeat=10.
        state.decision=dict(allowed=True,units=[101,0],expires_mono=10.1)
        state.observe(decode_feedback(frame(mode=1,velocity=.12,angular=.2)),10.)
        self.assertEqual(state.state,'armed')
        self.assertEqual(state.command(10.),(0,0,3));self.assertEqual(state.reason,'wire_target_over_limit')
        state=DriveState(motion);state.state='armed';state.phase=0.;state.heartbeat=10.
        state.observe(decode_feedback(frame(mode=1)),10.)
        self.assertEqual(state.command(10.201),(0,0,3));self.assertEqual(state.reason,'safety_watchdog')
        packet=encode_command_units(100,0,mode=1,velocity_limit_mm_s=controller_settings(motion)['velocity_limit_field'])
        self.assertEqual(int.from_bytes(packet[7:9],byteorder='big',signed=True),1500)

    def test_feedback_bounds_and_legacy_reverse_limit_are_still_enforced(self):
        _,motion,_=self.load(self.profile())
        for row in (frame(velocity=.151),frame(angular=.301)):
            state=DriveState(motion);state.state='armed';state.observe(decode_feedback(row),10.)
            self.assertEqual(state.reason,'observed_velocity_over_limit')
        state=DriveState(self.motion);state.state='armed'
        state.observe(decode_feedback(frame(velocity=-.06)),10.)
        self.assertEqual(state.reason,'observed_velocity_over_limit')


if __name__=='__main__':unittest.main()
