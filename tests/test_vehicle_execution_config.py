"""Vehicle-specific stopping and serial fields; synthetic profiles only."""
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
from navigation.safety.config import load_stopping_config
from tests.test_ground_probe import frame


class VehicleExecutionConfigTests(unittest.TestCase):
    def setUp(self):
        self.motion=load_motion_config('navigation/planning/configs/motion_trial.yaml')
        # These tests exercise a synthetic 0.1 m/s² braking boundary, independent
        # of the current wheelchair trial acceleration setting.
        self.motion['max_acceleration_mps2']=.1
        self.motion['max_angular_acceleration_rps2']=.15
        self.motion['brake_deceleration_scale']=.75

    def profile(self):
        p=yaml.safe_load(Path('navigation/control/vehicle.example.yaml').read_text())
        p.update(vehicle_id='synthetic_only',protocol='af01_12_bf01_32_payload_sum_mode1_3')
        p['verification']=dict(source='synthetic unit test; no physical acceptance',
            geometry_confirmed=True,motion_confirmed=True,protocol_confirmed=True)
        p['geometry']=dict(footprint=[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],base_from_lidar=np.eye(4).tolist())
        p['measured_limits']={k:self.motion[k] for k in ('max_forward_mps','max_reverse_mps','max_angular_rps',
            'max_acceleration_mps2','max_angular_acceleration_rps2')}
        p['measured_limits'].update(minimum_braking_mps2=.1,minimum_angular_braking_rps2=.2,
            maximum_actuator_delay_s=.7,firmware_watchdog_s=.8)
        p['stopping_model']=dict(reaction_s=1.,actuator_delay_s=.8)
        p['controller']=dict(velocity_limit_field=1500,feedback_max_linear_mps=.15,feedback_max_angular_rps=.3)
        p['execution_response_bounds']=dict(max_planar_speed_mps=.10,max_stop_time_s=1.2)
        return p

    def load(self,p):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vehicle.yaml';path.write_text(yaml.safe_dump(p))
            return load_vehicle_profile(path,self.motion,require_execution=True)

    def test_profile_stopping_model_is_shared_by_dwa_and_independent_safety(self):
        from navigation.control.dwa import DwaController
        from navigation.safety.guard import LocalSafety
        geometry,motion,_=self.load(self.profile())
        motion.update(collision_backend='cpu',collision_library=None)
        planning=planning_config()
        dwa=DwaController(planning,motion,geometry)
        guard=LocalSafety(geometry,motion,planning['safety_margin_m'],people_settings=planning['people'],execution=True)
        self.assertEqual(dwa.safety_settings,guard.settings)
        self.assertEqual(guard.settings['reaction_s'],1.)
        self.assertEqual(guard.settings['actuator_delay_s'],.8)
        self.assertEqual(guard.settings['feedback_max_age_s'],.15)

    def test_profile_budgets_must_cover_declared_hardware_response(self):
        for key,value in (('reaction_s',.7),('actuator_delay_s',.6)):
            p=self.profile();p['stopping_model'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.load(p)

    def test_exact_decimal_braking_boundary_passes_and_lower_limits_are_rejected(self):
        p=self.profile()
        p['measured_limits'].update(minimum_braking_mps2=.075,minimum_angular_braking_rps2=.1125)
        self.load(p)
        for key,value in (('minimum_braking_mps2',.07499999999999999),
                          ('minimum_angular_braking_rps2',.11249999999999999)):
            candidate=self.profile()
            candidate['measured_limits'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Stopping model exceeds guaranteed braking'):
                self.load(candidate)

    def test_model_override_cannot_change_freshness_or_reduce_defaults(self):
        default=load_stopping_config()
        for override in ({'reaction_s':.1},{'actuator_delay_s':float('nan')},
                {'reaction_s':True},{'feedback_max_age_s':1.},{'unknown':1.},[]):
            with self.subTest(override=override),self.assertRaises(ValueError):
                load_stopping_config(dict(stopping_model=override))
        self.assertEqual(default['reaction_s'],.2)
        self.assertEqual(load_stopping_config(self.motion),default)

    def test_unconfirmed_profile_is_still_rejected_with_larger_budgets(self):
        for field in ('motion_confirmed','protocol_confirmed'):
            p=self.profile();p['verification'][field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):self.load(p)
        p=self.profile();p['measured_limits']['firmware_watchdog_s']=None
        with self.assertRaises(ValueError):self.load(p)

    def test_real_execution_needs_explicit_valid_controller_fields(self):
        for value in (None,{},dict(velocity_limit_field=0,feedback_max_linear_mps=.15,feedback_max_angular_rps=.3),
                dict(velocity_limit_field=1500,feedback_max_linear_mps=.01,feedback_max_angular_rps=.3)):
            p=self.profile();p['controller']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.load(p)

    def test_execution_requires_independent_valid_response_bounds(self):
        for bounds in (None, {}, dict(max_planar_speed_mps=.09,max_stop_time_s=1.),
                dict(max_planar_speed_mps=.1,max_stop_time_s=0.),
                dict(max_planar_speed_mps=.1,max_stop_time_s=float('nan')),
                dict(max_planar_speed_mps=.1,max_stop_time_s=True)):
            p=self.profile();p['execution_response_bounds']=bounds
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):self.load(p)
        p=self.profile();p['measured_limits']['max_forward_mps']=.15
        with self.assertRaisesRegex(ValueError,'declared measured vehicle speed'):self.load(p)

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
