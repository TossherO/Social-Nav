"""CPU safety keeps stopping when the main computation stops sending packets."""
import socket
import time
import unittest
from runtime.safety import SafetyProcess,SafetyTelemetry
from runtime.transport import Connection


def delayed_gate_run(*arguments):
    """Child-only instrumentation; production safety has no test hooks."""
    import os
    from pathlib import Path
    from navigation.safety.shadow import ShadowSafety
    import runtime.safety as module
    original=ShadowSafety.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs);gate=self.guard.gate;first=[True]
        def delayed(*a,**k):
            if first[0]:
                first[0]=False;Path(os.environ['SOCIAL_NAV_TEST_GATE_STARTED']).touch();time.sleep(.04)
            return gate(*a,**k)
        self.guard.gate=delayed
    ShadowSafety.__init__=initialize
    module._run(*arguments)


class SafetyIsolationTests(unittest.TestCase):
    def test_fresh_live_pose_promptly_rechecks_without_extending_its_lifetime(self):
        import copy
        import json
        from unittest.mock import patch
        from tests.test_navigation_shadow import ShadowTests
        from perception.environment.scene import assemble_scene
        from navigation.control.shadow import DwaShadowController
        from navigation.planning.planner import Planner
        from navigation.planning.continuity import reference_envelope

        class Alive:
            def is_alive(self):return True

        class FakeChassis:
            def __init__(self,*_):
                self.channel,self.remote=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
                self.remote.setblocking(False);self.process=Alive()
            def close(self):self.channel.close();self.remote.close()

        fixture=ShadowTests();fixture.setUp();fixture.geometry.assumed=False
        motion=dict(fixture.motion,parameters_assumed=False,
            execution_response_bounds=dict(max_planar_speed_mps=fixture.motion['max_forward_mps'],max_stop_time_s=.1),
            execution_max_angular_rps=fixture.motion['max_angular_rps'])
        scene=assemble_scene(fixture.scene.geometry,clock_domain='runtime_live',settings=fixture.safety.scene_settings)
        planner=Planner(fixture.config,fixture.geometry,require_actual_start=True)
        reference=reference_envelope(planner.query(fixture.req,scene=scene,now=10.,clock_domain='runtime_live'),fixture.req,scene,None)
        packet=DwaShadowController(fixture.config,motion,fixture.geometry).step(
            fixture.req,scene,fixture.rows,reference,now=10.,clock_domain='runtime_live',mono=0.)
        settings=dict(fixture.safety.settings,period_s=.2)
        incoming,producer=socket.socketpair();sender=Connection(producer)
        with patch('navigation.control.chassis.ChassisProcess',FakeChassis):
            safety=SafetyProcess(incoming,fixture.config,motion,fixture.geometry,
                clock_domain='runtime_live',settings=settings,chassis_port='synthetic-only',chassis_output='unused')
        incoming.close()
        try:
            deadline=time.monotonic()+10.
            while not safety.ready:
                safety.poll()
                if time.monotonic()>deadline:self.fail('Safety startup timed out')
                time.sleep(.005)
            safety.command('goal',fixture.req);safety.command('arm',dict(enabled=True));safety.scene(scene)
            origin=time.monotonic()
            for row in fixture.rows:
                sender.send(dict(kind='state',state=row,clock_s=10.,clock_mono=origin,clock_rate=1.))
            safety.packet(packet)
            def next_authorization(predicate,timeout):
                deadline=time.monotonic()+timeout
                while time.monotonic()<deadline:
                    try:
                        item=json.loads(safety.chassis.remote.recv(65536))
                        if predicate(item):return item
                    except BlockingIOError:time.sleep(.001)
                self.fail('Expected authorization was not observed')
            first=next_authorization(lambda x:x.get('allowed'),1.)
            time.sleep(.02)
            # Empty any earlier datagrams before sending a genuinely new pose.
            try:
                while True:safety.chassis.remote.recv(65536)
            except BlockingIOError:pass
            receipt=time.monotonic();stamp=10.+receipt-origin
            sender.send(dict(kind='state',state=dict(fixture.rows[-1],stamp_s=stamp),
                             clock_s=stamp,clock_mono=receipt,clock_rate=1.))
            fresh=next_authorization(lambda x:x.get('allowed') and
                (x.get('diagnostics') or {}).get('pose_velocity_estimate',{}).get('stamp_s')==stamp,.14)
            self.assertLess(fresh['mono']-receipt,.14)
            self.assertLessEqual(fresh['expires_mono'],receipt+.15+.005)
            # The source pose then stops: the previous authorization cannot be
            # kept alive by heartbeats or by the scheduler wake.
            stale=next_authorization(lambda x:not x.get('allowed') and x['mono']>fresh['expires_mono'],.5)
            self.assertFalse(stale['allowed'])
        finally:sender.close();safety.close()

    def test_map_received_during_gate_is_rechecked_before_emitting_a_velocity(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import os
        import copy
        from tests.test_navigation_shadow import ShadowTests
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import GridSnapshot,load_config
        fixture=ShadowTests();fixture.setUp()
        incoming,producer=socket.socketpair();sender=Connection(producer)
        with tempfile.TemporaryDirectory() as temporary:
            marker=Path(temporary)/'started'
            with patch.dict(os.environ,SOCIAL_NAV_TEST_GATE_STARTED=str(marker)),patch('runtime.safety._run',delayed_gate_run):
                safety=SafetyProcess(incoming,fixture.config,fixture.motion,fixture.geometry,clock_domain='synthetic');incoming.close()
                try:
                    deadline=time.monotonic()+10
                    while not safety.ready:
                        safety.poll()
                        if time.monotonic()>deadline:self.fail('Safety startup timed out')
                        time.sleep(.005)
                    safety.command('goal',fixture.req);safety.scene(fixture.scene)
                    for row in fixture.rows:
                        mono=time.monotonic();sender.send(dict(kind='state',state=row,clock_s=10.,clock_mono=mono,clock_rate=1.))
                    safety.packet(fixture.packet)
                    deadline=time.monotonic()+5
                    while not marker.exists():
                        if time.monotonic()>deadline:self.fail('Safety never entered the full gate')
                        time.sleep(.001)
                    metadata=copy.deepcopy(fixture.scene.geometry.metadata);metadata.update(stamp_s=10.01,snapshot_id='changed-during-check')
                    layers={k:v.copy() for k,v in fixture.scene.geometry.layers.items()};layers['occupancy'][20,20]=100
                    scene=assemble_scene(GridSnapshot(metadata,layers),clock_domain='synthetic',settings=load_config()['scene'])
                    safety.scene(scene);time.sleep(.12)
                    outputs=safety.poll();self.assertTrue(outputs)
                    self.assertFalse(any(r['state']=='allowed' for r in outputs),outputs)
                    self.assertTrue(any(r.get('as_of')==10.01 for r in outputs),outputs)
                finally:sender.close();safety.close()

    def test_main_stall_does_not_block_independent_stop_decisions(self):
        from tests.test_navigation_shadow import ShadowTests
        fixture=ShadowTests();fixture.setUp()
        from tests.test_navigation_prediction import tracked_scene,track
        from navigation.planning.planner import Planner
        from navigation.planning.continuity import reference_envelope
        fixture.scene=tracked_scene([track(5.,5.)])
        planner=Planner(fixture.config,fixture.geometry,require_actual_start=True)
        result=reference_envelope(planner.query(fixture.req,scene=fixture.scene,now=10.,clock_domain='synthetic'),fixture.req,fixture.scene,None)
        fixture.packet=fixture.controller.step(fixture.req,fixture.scene,fixture.rows,result,now=10.,clock_domain='synthetic',mono=0.)
        incoming,producer=socket.socketpair();sender=Connection(producer)
        safety=SafetyProcess(incoming,fixture.config,fixture.motion,fixture.geometry,clock_domain='synthetic');incoming.close()
        recorder=None;received=[]
        try:
            deadline=time.monotonic()+10
            while not safety.ready:
                safety.poll()
                if time.monotonic()>deadline:self.fail('Safety process did not start')
                time.sleep(.01)
            recorder=SafetyTelemetry(safety,lambda _,row:received.append(row))
            safety.command('goal',fixture.req);safety.scene(fixture.scene)
            for row in fixture.rows:
                mono=time.monotonic();sender.send(dict(kind='state',state=row,clock_s=10.,clock_mono=mono,clock_rate=1.))
            safety.packet(fixture.packet)
            # No parent work or packet refresh during this watchdog window.
            time.sleep(.45)
            rows=list(received);self.assertTrue(rows)
            self.assertTrue(any(r['state']=='allowed' for r in rows),rows)
            self.assertTrue(any(r['reason']=='proposal_timeout' and r['velocity']==[0.,0.] for r in rows),rows)
            self.assertTrue(safety.process.is_alive())
            self.assertTrue(all(not r['executable'] and not r['task_completed'] for r in rows))
            safety.command('goal',dict(fixture.req,goal_revision=fixture.req['goal_revision']-1))
            time.sleep(.1)
            self.assertTrue(safety.process.is_alive())
            self.assertTrue(any((r['reason'] or '').startswith('input_rejected:') for r in received))
        finally:
            if recorder is not None:recorder.close()
            sender.close();safety.close()
