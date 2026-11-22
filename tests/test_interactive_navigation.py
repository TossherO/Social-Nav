"""Interactive bag clocks, frozen planning and driver guards; no real hardware."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
import numpy as np
import yaml

from navigation.control.chassis import DriveState
from navigation.control.chassis_protocol import decode_feedback
from navigation.control.vehicle import load_vehicle_profile
from navigation.planning.manual_goal import ManualGoals
from navigation.planning.motion import load_motion_config
from runtime.playback import BagPlayback
from runtime.session import InteractiveSession
from tests.chassis_fixtures import frame


class PlaybackTests(unittest.TestCase):
    def test_initial_pose_keyboard_adjustments_are_local_and_bounded(self):
        from runtime.interactive import adjust_initial_pose
        pose=[.1,.2,np.pi/2,2.,3.,4.]
        np.testing.assert_allclose(adjust_initial_pose(pose,'w'),[.1,.2,np.pi/2,2.,3.05,4.],atol=1e-12)
        np.testing.assert_allclose(adjust_initial_pose(pose,'a'),[.1,.2,np.pi/2,1.95,3.,4.],atol=1e-12)
        self.assertAlmostEqual(adjust_initial_pose(pose,'q')[2],np.pi/2+np.pi/36)
        self.assertEqual(pose,[.1,.2,np.pi/2,2.,3.,4.])
        with self.assertRaises(ValueError):adjust_initial_pose(pose,'invalid')

    def test_live_localization_gravity_override_is_forwarded(self):
        from runtime.interactive import parser,runtime_command
        args=parser().parse_args(['--live','--output','/tmp/example','--roll-gravity-deg','6'])
        config=dict(map='map',static_map='prior',coverage='coverage',motion_config='motion')
        command=runtime_command(args,config,[0.,0.,0.,0.,0.,0.])
        self.assertEqual(command[command.index('--roll-gravity-deg')+1],'6.0')

    def test_local_window_override_reaches_runtime(self):
        from runtime.interactive import parser,runtime_command
        args=parser().parse_args(['--live','--output','/tmp/example','--environment-window-m','12'])
        config=dict(map='map',static_map='prior',coverage='coverage',motion_config='motion')
        command=runtime_command(args,config,[0.,0.,0.,0.,0.,0.])
        index=command.index('--environment-window-m')
        self.assertEqual(command[index+1],'12.0')

    def test_follow_backend_is_frozen_for_floor_handoff(self):
        from runtime.interactive import parser,runtime_command
        args=parser().parse_args(['--live','--output','/tmp/example','--agent-observations',
                                  '--agent-follow-candidates'])
        config=dict(map='map',static_map='prior',coverage='coverage',motion_config='motion')
        with patch('decision.vlm.t2.hosted.follow_backend',return_value='hosted'):
            command=runtime_command(args,config,[0.]*6)
        self.assertEqual(command[command.index('--t2-backend')+1],'hosted')

    def test_reid_weights_are_frozen_for_floor_handoff_without_jpeg(self):
        from runtime.interactive import parser,runtime_command
        config=dict(map='map',static_map='prior',coverage='coverage',motion_config='motion')
        with tempfile.TemporaryDirectory(prefix='local reid ') as directory:
            weights=Path(directory)/'osnet.engine';weights.touch()
            args=parser().parse_args(['--live','--output','/tmp/example','--agent-observations',
                '--agent-follow-candidates','--t2-backend','reid','--t2-reid-weights',str(weights),
                '--t2-jpeg-quality','80'])
            with self.assertRaisesRegex(ValueError,r'\.engine\.json metadata file'):
                runtime_command(args,config,[0.]*6)
            weights.with_suffix('.engine.json').write_text('{}')
            command=runtime_command(args,config,[0.]*6)
            self.assertEqual(command[command.index('--t2-reid-weights')+1],str(weights))
            self.assertNotIn('--t2-jpeg-quality',command)
            args.t2_raw_rgb=True
            with self.assertRaisesRegex(ValueError,'conflicts with --t2-raw-rgb'):
                runtime_command(args,config,[0.]*6)

    def test_display_goal_follows_processed_intent_and_clears_on_cancel_or_replay_change(self):
        session=InteractiveSession(live=True,safety=None,output=Path('/tmp'))
        goal=dict(request_id='g',frame_id='map',goal=[1.,2.,.3])
        session.publish_observation(None,{},goal_intent=goal)
        goal['goal'][0]=99.
        self.assertEqual(session.status()['goal_intent']['goal'],[1.,2.,.3])
        session.publish_observation(None,{})
        self.assertIsNone(session.status()['goal_intent'])
        session.publish_observation(None,{},goal_intent=goal)
        session.update_playback(dict(paused=True,epoch=1,source_ended=False,stamp_s=10.,offset_s=1.))
        self.assertIsNone(session.status()['goal_intent'])

    def test_bag_pose_file_checks_source_and_never_initializes_live_input(self):
        from tests.test_initial_pose import InitialPoseTests
        from runtime.interactive import bag_initial_pose
        fixture=InitialPoseTests();fixture.setUp()
        try:
            config=dict(map=str(fixture.map))
            np.testing.assert_allclose(bag_initial_pose(fixture.path,config,bag=fixture.bag,start=0.),[.1,-.2,1.2,1,2,3])
            with self.assertRaises(ValueError):bag_initial_pose(fixture.path,config,bag=fixture.bag,start=1.)
            with self.assertRaises(ValueError):bag_initial_pose(fixture.path,config,bag=fixture.bag,start=0.,live=True)
            fixture.bag.write_bytes(b'different bag')
            with self.assertRaises(ValueError):bag_initial_pose(fixture.path,config,bag=fixture.bag,start=0.)
        finally:fixture.temporary.cleanup()

    def test_unassociated_localization_has_visible_reason_even_while_paused(self):
        from runtime.interactive import session_notice
        self.assertEqual(session_notice(None)[0],'waiting_for_initial_pose')
        self.assertEqual(session_notice(None,selected=True)[0],'initializing')
        status=dict(paused=True,observation=dict(localization=dict(health=0,has_map_pose=False),prior=dict(valid=False)))
        stage,text,error=session_notice(status,selected=True)
        self.assertEqual(stage,'map_unlocalized');self.assertTrue(error)
        self.assertIn('PAUSED',text);self.assertIn('initial pose',text)
        status['observation']['localization']['has_map_pose']=True
        self.assertEqual(session_notice(status,selected=True)[0],'map_association_unavailable')
        status['observation']['prior']['valid']=True
        self.assertEqual(session_notice(status,selected=True)[0],'paused_ready')

    def test_pause_holds_source_time_and_resume_reanchors_wall_clock(self):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        playback = BagPlayback(child, rate=1., pause_at=1.)
        playback.start(100.); playback.last = 101.
        before = playback.anchor; stopped = threading.Event(); rows = []
        worker = threading.Thread(target=playback.boundary, args=(rows.append, stopped))
        try:
            worker.start(); time.sleep(.25)
            self.assertTrue(rows and rows[-1]['paused'])
            parent.send(b'{"paused":false}'); worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            self.assertFalse(rows[-1]['paused']); self.assertEqual(rows[-1]['stamp_s'],101.)
            self.assertGreater(playback.anchor-before,.25)
            self.assertEqual(playback.last,101.)
        finally: stopped.set(); worker.join(timeout=1); parent.close(); playback.close()

    def test_stale_preview_and_bag_arming_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            session = InteractiveSession(live=False,safety=None,output=Path(folder))
            try:
                session.update_playback(dict(paused=True,epoch=1,source_ended=False,stamp_s=12.,offset_s=2.))
                old = session.command_generation
                session.note_command('goal',{})
                self.assertFalse(session.publish_preview({'old':True},old))
                with self.assertRaises(ValueError): session.handle('arm',{'enabled':True})
            finally: session.close()

    def test_async_chassis_status_cannot_display_stale_armed_as_current(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'chassis').mkdir()
            status=root/'chassis/status.json'
            session=InteractiveSession(live=True,safety=None,output=root,execute=True)
            try:
                status.write_text(json.dumps(dict(state='armed',reason=None,status_mono=10.)))
                with patch('runtime.session.time.monotonic',return_value=10.1):
                    self.assertEqual(session.status()['chassis']['state'],'armed')
                with patch('runtime.session.time.monotonic',return_value=10.6):
                    row=session.status()['chassis']
                    self.assertEqual(row['state'],'unavailable')
                    self.assertEqual(row['reported_state'],'armed')
                    self.assertEqual(row['reason'],'chassis_status_stale')
                self.assertEqual(json.loads(status.read_text())['state'],'armed')
            finally:session.close()

    def test_xy_goal_does_not_invent_heading(self):
        values=[]
        bridge=ManualGoals(lambda kind,value: values.append(value) or {'accepted':True},task_id='point-test',frame='map',map_snapshot_id='map')
        bridge.point(frame='map',position=[1.,2.,3.],stamp_s=10.)
        self.assertEqual(values[0]['goal'],[1.,2.])
        self.assertEqual(values[0]['source_intent_stamp_s'],10.)

    def test_manual_pose_keeps_only_height_and_tilt_from_map_hint(self):
        from runtime.interactive import selected_initial_pose
        hint=dict(map=dict(frame_id='map'),pose=dict(position_m=[9.,8.,-.7],quaternion_xyzw=[0.,0.,0.,1.]))
        pose=selected_initial_pose('map',[1.,2.,0.],[0.,0.,0.,1.],hint)
        self.assertEqual(pose,[0.,0.,0.,1.,2.,-.7])
        with self.assertRaises(ValueError):selected_initial_pose('odom',[1.,2.,0.],[0.,0.,0.,1.],hint)

    def test_frozen_query_uses_causal_motion_and_preserves_scene(self):
        from tests.test_navigation_shadow import ShadowTests
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config
        from navigation.planning.frozen import preview
        fixture=ShadowTests();fixture.setUp()
        scene=assemble_scene(fixture.scene.geometry,clock_domain='runtime_bag',settings=load_config()['scene'])
        original=copy.deepcopy(scene.record)
        future=dict(fixture.rows[-1],stamp_s=20.,position=[100.,100.,0.])
        result=preview(fixture.req,scene,[*fixture.rows,future],fixture.controller.controller,fixture.config)
        self.assertTrue(result['reference_available'],result)
        self.assertTrue(result['generated'],result)
        self.assertEqual(result['feedback']['pose'],[2.,2.,0.])
        self.assertEqual(result['as_of'],10.);self.assertEqual(scene.record,original)
        self.assertFalse(result['executable']);self.assertFalse(result['task_completed'])
        with self.assertRaises(ValueError):preview(fixture.req,fixture.scene,fixture.rows,fixture.controller.controller,fixture.config)

    def test_frozen_display_map_follows_accepted_preview_and_stays_out_of_session_json(self):
        from navigation.planning.runtime import decode_display_map
        from perception.environment.layers import load_config
        from perception.environment.scene import assemble_scene
        from runtime.control import ControlLoop
        from tests.navigation_fixtures import asset, bundle, request
        from tests.test_navigation_shadow import ShadowTests

        fixture=ShadowTests();fixture.setUp()
        local=np.zeros((60,60),np.int8);local[45,44]=100
        scene=assemble_scene(bundle(local).geometry,clock_domain='runtime_bag',settings=load_config()['scene'])
        static=np.zeros_like(local);static[45,44:46]=100
        prior=asset(static)
        conditional=np.zeros_like(static,dtype=bool);conditional[45,44:46]=True
        prior.snapshot.layers['conditional_obstacle']=conditional
        goal=request('mapped',goal=[4.,2.],valid_until_s=12.)
        loop=ControlLoop.__new__(ControlLoop)
        loop.preview_token=None;loop.scene=scene;loop.device=object();loop.request=goal;loop.asset=prior
        loop.display=Mock();loop.batch=Mock();loop.lock=threading.Lock();loop.environment_epoch=0
        loop.controller=fixture.controller;loop.rows=fixture.rows;loop.settings=dict(control_period_s=.1)
        loop.reference=SimpleNamespace(planning=fixture.config,display_enabled=True);loop.metrics=None
        loop._search_scene=lambda scene:scene
        with tempfile.TemporaryDirectory() as folder:
            session=loop.session=InteractiveSession(live=False,safety=None,output=Path(folder))
            self.addCleanup(session.close)
            session.update_playback(dict(paused=True,epoch=1,source_ended=False,stamp_s=10.,offset_s=2.))
            token=session.command_generation
            with patch('perception.visualization.surface.DisplayGrids'), \
                 patch('perception.visualization.interactive.preview_geometry',return_value={}) as geometry:
                loop._preview_frozen(token,0)
                clear,display=loop.display.frozen_basemap.call_args_list
                self.assertEqual(call(True),clear)
                snapshot=display.args[1]
                saved=json.loads((Path(folder)/'preview.json').read_text())
                self.assertEqual(saved,session.preview)
                self.assertEqual(saved['reference']['binding'],snapshot['reference_binding'])
                self.assertEqual(saved['reference']['result']['result_binding'],snapshot['result_binding'])
                self.assertNotIn('display_map',saved)
                self.assertNotIn('cells',saved['display'])
                cells,conditional=decode_display_map(snapshot)
                self.assertEqual(100,cells[45,44]);self.assertFalse(conditional[45,44])
                self.assertEqual(0,cells[45,45]);self.assertTrue(conditional[45,45])
                loop._preview_frozen(token,0)
                self.assertEqual(2,loop.display.frozen_basemap.call_count)

                # A new command arrives between the planning result and publication.
                session.note_command('goal',goal)
                geometry.side_effect=lambda *args:session.note_command('cancel',{}) or {}
                loop._preview_frozen(session.command_generation,0)
                self.assertEqual(call(True),loop.display.frozen_basemap.call_args)
                self.assertIsNone(session.preview)
                loop.request=None
                loop._preview_frozen(session.command_generation,0)
                self.assertEqual(call(True),loop.display.frozen_basemap.call_args)

                # An environment fault arriving during the next query rejects its snapshot.
                loop.request=goal
                session.note_command('goal',goal)
                geometry.side_effect=lambda *args:setattr(loop,'environment_epoch',1) or {}
                loop._preview_frozen(session.command_generation,0)
                self.assertEqual(call(True),loop.display.frozen_basemap.call_args)
                self.assertIsNone(session.preview)

                loop.scene=None
                loop._preview_frozen(session.command_generation,1)
                self.assertEqual('observation_unavailable',session.preview['reason'])
                self.assertEqual(call(True),loop.display.frozen_basemap.call_args)
                loop.scene=scene
                resumed=dict(paused=False,epoch=2,source_ended=False,stamp_s=10.,offset_s=2.)
                geometry.side_effect=lambda *args:session.update_playback(resumed) or {}
                loop._preview_frozen(session.command_generation,1)
                self.assertEqual(call(True),loop.display.frozen_basemap.call_args)
                self.assertIsNone(session.preview)

            loop.display=None;loop.preview_token=None
            session.update_playback(dict(resumed,paused=True,epoch=3))
            with patch('navigation.planning.runtime.encode_display_map') as encode:
                loop._preview_frozen(session.command_generation,1)
            encode.assert_not_called()
            self.assertTrue(session.preview['reference_available'])


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.motion=load_motion_config();self.state=DriveState(self.motion)

    def packet(self,stamp,revision=1,**values):
        return dict(mono=stamp,arm=True,arm_revision=revision,allowed=True,units=[20,0],expires_mono=stamp+.1,**values)

    def arm(self):
        for i in range(65):
            now=10.+i*.02
            self.state.receive(self.packet(now),now)
            mode=3 if self.state.state=='checking_stationary' else 1
            self.state.observe(decode_feedback(frame(mode=mode)),now)
            self.state.command(now)
        self.assertEqual(self.state.state,'armed')
        return now

    def test_only_armed_fresh_authorization_moves_and_expiry_stops(self):
        self.assertEqual(self.state.command(10.),(0,0,3))
        now=self.arm()
        self.assertEqual(self.state.command(now),(20,0,1))
        self.assertEqual(self.state.command(now+.11),(0,0,1))
        self.assertEqual(self.state.command(now+.21),(0,0,3))
        self.assertEqual(self.state.reason,'safety_watchdog')

    def test_takeover_latches_until_new_operator_revision(self):
        now=self.arm()
        feedback=decode_feedback(frame());feedback['joystick_axes_raw']=[1,0]
        self.state.observe(feedback,now)
        self.state.receive(self.packet(now+.01),now+.01)
        self.assertEqual(self.state.command(now+.01),(0,0,3))
        self.assertEqual(self.state.reason,'joystick_takeover')
        self.state.receive(self.packet(now+.02,revision=2),now+.02)
        self.assertEqual(self.state.state,'checking_stationary')

    def test_stale_feedback_checksum_and_speed_limits_stop(self):
        now=self.arm()
        bad=self.packet(now+.02);bad['units']=[5000,0]
        self.state.receive(bad,now+.02)
        self.assertEqual(self.state.command(now+.02),(0,0,3))
        self.assertEqual(self.state.reason,'wire_target_over_limit')
        self.state=DriveState(self.motion);now=self.arm()
        bad=decode_feedback(frame());bad['candidate_checksum_matches']=False
        self.state.observe(bad,now)
        self.assertEqual(self.state.command(now),(0,0,3))
        self.state=DriveState(self.motion);now=self.arm()
        self.state.receive(self.packet(now+.3),now+.3)
        self.assertEqual(self.state.command(now+.3),(0,0,3))
        self.assertEqual(self.state.reason,'feedback_timeout')

    def test_unconfirmed_vehicle_template_cannot_execute(self):
        with self.assertRaises(ValueError):
            load_vehicle_profile('navigation/control/vehicle.example.yaml',self.motion,require_execution=True)

    def test_measured_profile_cannot_understate_motion_limits(self):
        profile=yaml.safe_load(Path('navigation/control/vehicle.example.yaml').read_text())
        profile.update(vehicle_id='synthetic_fixture',protocol='af01_12_bf01_32_payload_sum_mode1_3')
        profile['controller']=dict(velocity_limit_field=1500,feedback_max_linear_mps=1.5,feedback_max_angular_rps=1.5)
        profile['execution_max_longitudinal_speed_mps']=2.
        profile['verification']=dict(source='synthetic unit test, not a real vehicle',geometry_confirmed=True,motion_confirmed=True,protocol_confirmed=True)
        profile['geometry']=dict(footprint=[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],base_from_lidar=np.eye(4).tolist())
        limits=profile['measured_limits']
        for name in ('max_forward_mps','max_reverse_mps','max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2'):
            limits[name]=self.motion[name]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vehicle.yaml';path.write_text(yaml.safe_dump(profile))
            geometry,motion,_=load_vehicle_profile(path,self.motion,require_execution=True)
            self.assertFalse(geometry.assumed);self.assertFalse(motion['parameters_assumed'])
            for key,value in (('max_forward_mps',.01),('max_angular_rps',.01)):
                bad=copy.deepcopy(profile);bad['measured_limits'][key]=value;path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):load_vehicle_profile(path,self.motion,require_execution=True)

    def test_execution_gate_checks_quantized_target_and_original_clock(self):
        from tests.test_navigation_shadow import ShadowTests
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config
        from navigation.planning.planner import Planner
        from navigation.planning.continuity import reference_envelope
        from navigation.control.shadow import DwaShadowController
        from navigation.safety.shadow import ShadowSafety
        fixture=ShadowTests();fixture.setUp()
        fixture.geometry.assumed=False;motion=dict(fixture.motion,parameters_assumed=False,
            execution_max_longitudinal_speed_mps=fixture.motion['max_forward_mps'],
            execution_max_angular_rps=fixture.motion['max_angular_rps'])
        scene=assemble_scene(fixture.scene.geometry,clock_domain='runtime_live',settings=load_config()['scene'])
        result=Planner(fixture.config,fixture.geometry,require_actual_start=True).query(fixture.req,scene=scene,now=10.,clock_domain='runtime_live')
        controller=DwaShadowController(fixture.config,motion,fixture.geometry)
        packet=controller.step(fixture.req,scene,fixture.rows,reference_envelope(result,fixture.req,scene,None),now=10.,clock_domain='runtime_live',mono=0.)
        original=copy.deepcopy(packet)
        safety=ShadowSafety(fixture.config,motion,fixture.geometry,clock_domain='runtime_live',execution=True)
        safety.command('goal',fixture.req);safety.push_grid(scene.geometry,mono=0.);safety.push_record(scene.record)
        for row in fixture.rows:safety.push_state(row)
        safety.push_packet(packet,mono=0.)
        result=safety.tick(10.,mono=0.,clock_now=lambda:10.)
        self.assertEqual(result['state'],'allowed',result)
        self.assertEqual(result['check']['status'],'free')
        self.assertAlmostEqual(result['motion_sweep']['points'][-1][0], safety.guard.settings['prediction_s'])
        self.assertEqual([x/1000 for x in result['wire_velocity_units']],result['velocity'])
        self.assertLessEqual(result['remaining_validity_s'],.15)
        self.assertEqual(packet,original)

    def test_bag_execution_is_rejected_before_gpu_or_serial_access(self):
        from runtime.__main__ import main
        with patch('sys.argv',['runtime','--bag','x.bag','--output','/tmp/unused','--execute']),self.assertRaises(ValueError):main()

    def test_serial_owner_handshake_reverse_spin_and_parent_stall(self):
        import os
        import pty
        import select
        import struct
        import tty
        from navigation.control.chassis import ChassisProcess
        master,slave=pty.openpty();tty.setraw(slave);port=os.ttyname(slave);os.close(slave)
        os.set_blocking(master,False);stop=threading.Event();writes=[];wire_limits=[]
        self.motion['controller']=dict(velocity_limit_field=1500,feedback_max_linear_mps=1.5,feedback_max_angular_rps=1.5)
        def controller():
            pending=bytearray();mode=3
            while not stop.wait(.015):
                if select.select([master],[],[],0)[0]:
                    try:pending.extend(os.read(master,4096))
                    except OSError:pass
                while len(pending)>=12:
                    start=pending.find(b'\xaf\x01')
                    if start<0:pending.clear();break
                    del pending[:start]
                    if len(pending)<12:break
                    packet=bytes(pending[:12]);del pending[:12]
                    _,_,v,w,mode,limit,error,led,checksum=struct.unpack('>2B2hBhBBB',packet)
                    writes.append((v,w,mode))
                    wire_limits.append(limit)
                try:os.write(master,frame(mode=mode))
                except OSError:pass
        emulator=threading.Thread(target=controller);emulator.start();driver=None
        try:
            with tempfile.TemporaryDirectory() as folder:
                output=Path(folder)/'chassis';driver=ChassisProcess(port,output,self.motion)
                began=time.monotonic()
                while time.monotonic()-began<3.:
                    now=time.monotonic()
                    packet=self.packet(now);packet['units']=[-20,30]
                    driver.channel.send(json.dumps(packet).encode());time.sleep(.025)
                self.assertIn((-20,30,1),writes)
                self.assertTrue(wire_limits);self.assertEqual(set(wire_limits),{1500})
                # No more parent traffic: independent watchdog must restore zero/mode3.
                time.sleep(.4)
                self.assertEqual(writes[-1],(0,0,3))
                status=json.loads((output/'status.json').read_text())
                self.assertEqual(status['reason'],'safety_watchdog')
                driver.close();driver=None
                rows=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
                self.assertTrue(any(row['kind']=='feedback' and row['device_measurement_stamp_s'] is None for row in rows))
                self.assertTrue(any(row['kind']=='authorization' for row in rows))
                self.assertTrue(any(row['kind']=='wire_target' and row['units']==[-20,30,1] for row in rows))
                self.assertTrue(any(row['kind']=='state' and row['reason']=='safety_watchdog' for row in rows))
                self.assertTrue(json.loads((output/'telemetry_summary.json').read_text())['complete'])
        finally:
            if driver is not None:driver.close()
            stop.set();emulator.join(timeout=1);os.close(master)

