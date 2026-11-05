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
