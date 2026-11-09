"""Causal observations, asynchronous cancellation and nonexecuting boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
import numpy as np
from decision.agent.core.async_model import AsyncModel
from decision.agent import ROOT
from decision.agent.tools.places import Catalog
from decision.agent.core.loop import AgentLoop
from decision.agent.tools.observations import Observation, ObservationSet
from decision.agent.adapters.runtime import ObservationWorker, RuntimeClient
from decision.agent.core.store import Store
from decision.agent.core.session import AgentSession
from decision.agent.workflows.drivers import SavedObservationDriver, SimulationDriver
from decision.agent.workflows.replay import ActionReplay
from perception.environment.layers import load_config
from perception.environment.scene import assemble_scene
from runtime.command import AgentState, CommandInbox
from tests.test_environment_consumers import snapshot
from tests.test_agent import ScriptedFeedback, reply, navigate, action


class ObservationLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root/'session.sqlite')
        self.scene = assemble_scene(snapshot(), clock_domain='synthetic', settings=load_config()['scene'])
        self.observation = Observation.from_scene(self.scene, rgb=dict(pixels=np.zeros((20, 30, 3), np.uint8),
            stamp_s=9.95, frame_id='camera', source=dict(kind='unit_fixture')), source=dict(scene='fixture', expected='not_for_model'))
        self.catalog = Catalog.load(ROOT/'decision/agent/configs/simulation.json')
        self.nav = ScriptedFeedback()
        self.nav.clock_domain = 'synthetic'
        self.nav.state.update(clock_domain='synthetic', session_id='local', frame_id='odom')
        self.nav.observe = lambda: self.observation
        self.tools = AgentSession('unit task', self.catalog, self.nav, self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_rgb_future_or_stale_rejected_before_model(self):
        for stamp in (10.01, 9.7):
            with self.assertRaises(ValueError):
                Observation.from_scene(self.scene, rgb=dict(pixels=np.zeros((20, 30, 3), np.uint8), stamp_s=stamp, frame_id='camera'))

    def test_observation_assets_and_metadata_bound_and_labels_excluded(self):
        self.observation.verify()
        with self.assertRaises(ValueError):
            Observation(self.observation.record, b'changed').verify()
        self.assertNotIn('source', self.observation.prompt())
        self.assertNotIn('not_for_model', json.dumps(self.observation.prompt()))
        path = self.root/'observations.sqlite'
        db = ObservationSet(path, create=True)
        db.add(self.observation)
        self.assertEqual(db.read(1).record, self.observation.record)
        db.close()

    def test_missing_people_count_is_unknown_including_legacy_frozen_input(self):
        from perception.environment.context import fingerprint
        self.assertIsNone(self.observation.record['people']['count'])
        record = copy.deepcopy(self.observation.record)
        record['people']['count'] = 0
        record['input_fingerprint'] = fingerprint({k: v for k, v in record.items() if k != 'input_fingerprint'})
        old = Observation(record, self.observation.image)
        self.assertIsNone(old.prompt()['people']['count'])
        self.assertEqual(old.record['people']['count'], 0)

    def test_async_cancel_returns_without_wait_and_late_result_is_invalidated(self):
        worker = AsyncModel()
        gate = threading.Event()
        worker.submit('plan', {'task_id': 'old'}, lambda: gate.wait(1))
        began = time.monotonic()
        worker.invalidate()
        self.assertLess(time.monotonic()-began, .1)
        gate.set()
        worker.thread.join(1)
        self.assertTrue(worker.poll()['stale'])
        worker.close()

    def test_async_queue_is_bounded(self):
        worker = AsyncModel()
        gate = threading.Event()
        worker.submit('plan', {}, lambda: gate.wait(1))
        with self.assertRaises(RuntimeError):
            worker.submit('plan', {}, lambda: {})
        gate.set()
        worker.thread.join(1)
        worker.poll()
        worker.close()

    def test_observation_worker_copies_only_rgb_before_slot_release(self):
        published = []
        worker = ObservationWorker(lambda record, image: published.append(Observation(record, image)))
        pixels = np.zeros((20, 30, 3), np.uint8)
        self.assertTrue(worker.submit(self.scene, pixels, 9.95, 'camera'))
        pixels[:] = 255
        worker.close()
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].record['rgb']['pixel_sha256'], self.observation.record['rgb']['pixel_sha256'])

    def test_runtime_state_reads_do_not_update_source_time_or_claim_execution(self):
        state = AgentState()
        state.publish_control(dict(stamp_s=10., clock_domain='runtime_bag', feedback={}, request=None))
        first, second = state.snapshot(), state.snapshot()
        self.assertEqual(first['control']['stamp_s'], second['control']['stamp_s'])
        self.assertFalse(second['execution_feedback'])

    def test_agent_runtime_client_rejects_safety_override_before_connection(self):
        client = RuntimeClient(self.root/'does-not-exist.sock')
        with self.assertRaises(ValueError):
            client.exchange('override', dict(emergency_stop=False, manual_override=False))

    def test_invalid_runtime_localization_cannot_offer_shadow_goal(self):
        from decision.agent.adapters.shadow import RuntimeShadowNavigation
        value = dict(control=dict(clock_domain='runtime_bag', stamp_s=10., feedback={}, request=None,
                                  localization=dict(session_id='s', generation=0, frame_id='odom', health=1)), received_age_s=.01)
        nav = RuntimeShadowNavigation(self.catalog, SimpleNamespace(snapshot=lambda **kw: value))
        result = nav.check_route(dict(pose=[0, 0]))
        self.assertFalse(result['proposal_allowed'])

    def test_runtime_query_ipc_is_read_only_and_preserves_observation_binding(self):
        sent = []
        inbox = CommandInbox(self.root/'commands.sock', SimpleNamespace(command=lambda *args: sent.append(args)))
        try:
            inbox.agent_state.publish_control(dict(stamp_s=10., clock_domain='synthetic', feedback={}, request=None))
            inbox.agent_state.publish_observation(self.observation.record, self.observation.image)
            client = RuntimeClient(self.root/'commands.sock')
            metadata = client.snapshot()
            self.assertIsNone(metadata['image_base64'])
            full = client.snapshot(with_image=True)
            self.assertEqual(client.observation(full).record, self.observation.record)
            self.assertEqual(sent, [])
            self.assertEqual(inbox.drain(), [])
        finally:
            inbox.close()

    def run_two_actions(self, work):
        actor = AgentLoop(self.tools, work)
        actor.tick()
        actor.model.thread.join(1)
        actor.tick()
        if actor.model.thread is not None:
            actor.model.thread.join(1)
        actor.tick()
        return actor

    def test_first_model_turn_receives_current_rgb_without_redundant_observe_call(self):
        received = []
        def work(payload, observation, spatial_view=None):
            received.append((payload, observation))
            return reply('observe_scene') if observation is None else reply('ask_user', question='请确认入口。')
        actor = self.run_two_actions(work)
        try:
            self.assertEqual(actor.requests, 1)
            self.assertEqual(self.tools.state['status'], 'needs_clarification')
            self.assertEqual(received[0][1].image, self.observation.image)
            self.assertFalse(any(m['role']=='assistant' for m in received[0][0]['messages']))
            self.assertEqual(self.tools.memory.snapshot()[0]['semantic']['kind'], 'historical_model_tool_choice')
        finally:
            actor.close()

    def test_user_cancel_does_not_wait_for_model_or_apply_late_tool(self):
        gate = threading.Event()
        actor = AgentLoop(self.tools, lambda *_: (gate.wait(1), reply('navigate', goal_id='elevator', pose=[5,4,0], view_id='not_applied', orientation_source='autonomous'))[1])
        try:
            actor.tick()
            actor.event('cancel')
            self.assertEqual(self.tools.state['status'], 'cancelled')
            gate.set(); actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(self.tools.skill.record['attempts'], [])
            self.assertEqual(actor.requests, 1)
        finally:
            gate.set()
            actor.close()

    def test_expired_image_is_not_refreshed_by_query(self):
        self.tools.call('observe_scene')
        context = self.tools.context()
        self.nav.state['stamp_s'] = 14.
        self.assertFalse(self.tools.valid_context(context))
        self.assertEqual(context['source_stamp_s'], 10.)

    def test_localization_reset_revokes_spatial_memory_but_keeps_semantics(self):
        self.tools.call('observe_scene')
        context = self.tools.context()
        self.tools.memory.remember(dict(verdict='supported', evidence='doorway'), context, place_id='elevator',
                                   clock_domain='synthetic', map_binding=None)
        self.nav.state['generation'] = 1
        self.tools.tick()
        saved = self.store.load()['evidence_memory'][0]
        self.assertFalse(saved['spatial_valid'])
        self.assertEqual(saved['semantic']['evidence'], 'doorway')
        self.assertFalse(self.tools.valid_context(context))
        self.assertEqual(self.tools.state['status'], 'needs_clarification')

    def test_waiting_for_observation_does_not_poll_the_model(self):
        from perception.environment.context import fingerprint
        def work(payload, observation, spatial_view=None):
            return reply('observe_scene') if observation is None else reply('wait_for_event', event='observation')
        actor = self.run_two_actions(work)
        try:
            self.assertEqual(self.tools.state['status'], 'waiting_observation')
            for _ in range(5):
                actor.tick()
            self.assertEqual(actor.requests, 1)
            record = copy.deepcopy(self.observation.record)
            record['stamp_s'] = 10.05
            record['input_fingerprint'] = fingerprint({k:v for k,v in record.items() if k != 'input_fingerprint'})
            fresh = Observation(record, self.observation.image).verify()
            self.nav.observe = lambda: fresh
            actor.tick()
            self.assertEqual(actor.requests, 2)
        finally:
            actor.close()

    def test_model_error_is_not_automatically_retried(self):
        def fail(*args):
            raise RuntimeError('explicit model failure')
        actor = AgentLoop(self.tools, fail)
        try:
            actor.tick(); actor.model.thread.join(1); actor.tick()
            for _ in range(4):
                actor.tick()
            self.assertEqual(actor.requests, 1)
            self.assertEqual(self.tools.state['status'], 'needs_clarification')
        finally:
            actor.close()

    def test_idle_map_change_prevents_model_request(self):
        self.nav.state['map_binding'] = dict(valid=False)
        actor = AgentLoop(self.tools, lambda *_: reply('observe_scene'))
        try:
            actor.tick()
            self.assertEqual(actor.requests, 0)
            self.assertTrue(self.tools.skill.record['context_lost'])
        finally:
            actor.close()

    def test_run_processes_actions_until_user_answer_is_needed(self):
        actor = AgentLoop(self.tools, ActionReplay([
            action('lookup_places', name=''), action('ask_user', question='请选择下一步。')]))
        try:
            state = actor.run(timeout_s=1.)
            self.assertEqual(state['status'], 'needs_clarification')
            self.assertEqual(actor.exit_reason, 'needs_clarification')
            self.assertEqual(actor.requests, 2)
            self.assertEqual(len(state['history']), 2)
        finally:
            actor.close()

    def test_run_console_can_cancel_while_model_is_pending(self):
        gate = threading.Event()
        actor = AgentLoop(self.tools, lambda *_: (gate.wait(2), reply('navigate', goal_id='elevator', pose=[5,4,0], view_id='not_applied', orientation_source='autonomous'))[1])
        def poll(actor, emit):
            if actor.model.busy:
                actor.event('cancel')
        try:
            state = actor.run(console=SimpleNamespace(poll=poll), timeout_s=1.)
            self.assertEqual(state['status'], 'cancelled')
            self.assertTrue(actor.model.thread.is_alive())
            gate.set(); actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(self.tools.skill.record['attempts'], [])
            self.assertEqual(actor.requests, 1)
        finally:
            gate.set()
            actor.model.thread.join(1)
            actor.close()

    def test_run_wall_budget_invalidates_pending_work_without_retry(self):
        gate = threading.Event()
        actor = AgentLoop(self.tools, lambda *_: (gate.wait(2), reply('navigate', goal_id='elevator', pose=[5,4,0], view_id='not_applied', orientation_source='autonomous'))[1])
        try:
            state = actor.run(timeout_s=.03)
            self.assertEqual(actor.exit_reason, 'wall_time_limit')
            self.assertEqual(state['status'], 'cancelled')
            gate.set(); actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(self.tools.skill.record['attempts'], [])
            self.assertEqual(actor.requests, 1)
        finally:
            gate.set()
            actor.model.thread.join(1)
            actor.close()

    def test_run_error_cancels_and_confirms_an_active_navigation(self):
        navigate(self.tools, 'elevator')
        original_tick = self.nav.tick
        def tick():
            if self.nav.state['status'] == 'stopping':
                self.nav.stopped()
            return original_tick()
        self.nav.tick = tick
        actor = AgentLoop(self.tools, lambda *_: reply('wait_for_event', event='navigation'))
        def fail(actor):
            raise RuntimeError('explicit driver failure')
        try:
            with self.assertRaisesRegex(RuntimeError, 'explicit driver failure'):
                actor.run(driver=SimpleNamespace(poll=fail))
            self.assertEqual(actor.exit_reason, 'run_error')
            self.assertEqual(self.tools.state['status'], 'cancelled')
            self.assertIsNone(self.tools.skill.record['handle'])
            self.assertEqual(self.tools.skill.record['completed'], [])
        finally:
            if actor.model.thread is not None:
                actor.model.thread.join(1)
            actor.close()

    def test_run_simulation_limit_is_supplied_by_driver(self):
        self.nav.t = 50.
        actor = AgentLoop(self.tools, lambda *_: reply('wait_for_event', event='observation'))
        try:
            actor.run(driver=SimulationDriver(self.nav, max_s=.3, fast=True), timeout_s=1.)
            self.assertEqual(actor.exit_reason, 'simulation_time_limit')
            self.assertGreaterEqual(self.nav.t, 50.3)
        finally:
            actor.close()

    def test_shutdown_retains_unconfirmed_cancel_without_resending(self):
        navigate(self.tools, 'elevator')
        calls = []
        cancel = self.nav.cancel
        def tracked_cancel(handle):
            calls.append(handle)
            return cancel(handle)
        self.nav.cancel = tracked_cancel
        self.tools.event('cancel')
        self.nav.tick = lambda: (_ for _ in ()).throw(ConnectionError('offline'))
        actor = AgentLoop(self.tools, lambda *_: reply('observe_scene'))
        try:
            actor.stop(timeout_s=.01)
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.tools.state['status'], 'stopping')
            self.assertEqual(self.tools.state['stop_target'], 'cancelled')
            self.assertIsNotNone(self.tools.skill.record['handle'])
            self.assertEqual(self.store.load()['status'], 'stopping')
        finally:
            actor.close()

    def test_saved_driver_freezes_input_during_request_and_advances_on_wait(self):
        from perception.environment.context import fingerprint
        fresh_record = copy.deepcopy(self.observation.record)
        fresh_record.update(observation_id='next', stamp_s=10.1)
        fresh_record['rgb'].update(stamp_s=10.05, age_s=.05)
        fresh_record['input_fingerprint'] = fingerprint({k:v for k,v in fresh_record.items() if k != 'input_fingerprint'})
        fresh = Observation(fresh_record, self.observation.image).verify()
        self.nav.current = self.observation
        self.nav.update = lambda value: setattr(self.nav, 'current', value)
        self.nav.observe = lambda: self.nav.current
        reads, calls = [], []
        def read(sequence):
            reads.append(sequence)
            return fresh
        def work(payload, observation, spatial_view=None):
            calls.append(self.nav.current.record['observation_id'])
            return (reply('wait_for_event', event='observation') if len(calls) == 1 else
                    reply('ask_user', question='已读取下一观察。'))
        actor = AgentLoop(self.tools, work)
        def inspect(actor):
            if actor.requests == 1:
                self.assertEqual(reads, [])
        try:
            driver = SavedObservationDriver(self.nav, SimpleNamespace(read=read), [1, 2])
            state = actor.run(driver=driver, on_tick=inspect, timeout_s=1.)
            self.assertEqual(state['status'], 'needs_clarification')
            self.assertEqual(reads, [2])
            self.assertEqual(calls, [self.observation.record['observation_id'], 'next'])
        finally:
            actor.close()

    def test_saved_input_exhaustion_does_not_reissue_model_request(self):
        actor = AgentLoop(self.tools, lambda *_: reply('wait_for_event', event='observation'))
        try:
            driver = SavedObservationDriver(self.nav, SimpleNamespace(), [1])
            state = actor.run(driver=driver, timeout_s=1.)
            self.assertEqual(actor.exit_reason, 'input_exhausted')
            self.assertEqual(state['status'], 'waiting_observation')
            self.assertEqual(actor.requests, 1)
        finally:
            actor.close()


if __name__ == '__main__':
    unittest.main()
