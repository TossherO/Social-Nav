"""Task origins, transport uncertainty and runtime cancellation receipts."""
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from decision.agent import ROOT
from decision.agent.tools.places import Catalog
from decision.agent.workflows.interaction import dispatch
from decision.agent.core.loop import AgentLoop
from decision.agent.core.memory import EvidenceMemory
from decision.agent.adapters.shadow import RuntimeShadowNavigation, SavedShadowNavigation
from decision.agent.tools.observations import ObservationSet
from decision.agent.core.store import Store
from decision.agent.core.session import AgentSession
from tests.test_agent import ScriptedFeedback, action


class AgentReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'session.sqlite')
        self.catalog = Catalog.load(ROOT/'decision/agent/configs/simulation.json')
        self.nav = ScriptedFeedback()

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def tools(self):
        return AgentSession('返回本次起点', self.catalog, self.nav, self.store)

    def test_measured_start_is_task_local_and_preserves_explicit_fixture_heading(self):
        self.nav.state.update(origin_pose=[2.,3.,.4], origin_frame_id='sim_odom')
        tools = self.tools()
        origin = tools.state['task_start']
        self.assertEqual(origin['pose'][:2], [2.,3.])
        self.assertEqual(origin['pose'][2], self.catalog.places['mission_start']['pose'][2])
        self.assertEqual(origin['task_origin']['measured_pose'][2], .4)
        self.assertEqual(self.catalog.places['mission_start']['pose'][:2], [4,4])

    @unittest.skipUnless((ROOT/'data/datasets/wheelchair_agent/floor1/observations.sqlite').exists(),'Local observation fixture unavailable')
    def test_saved_start_is_from_matching_map_and_shared_asset_stays_unbound(self):
        catalog = Catalog.load(ROOT/'decision/agent/configs/floor1.json')
        dataset = ObservationSet(ROOT/'data/datasets/wheelchair_agent/floor1/observations.sqlite')
        try:
            r = dataset.read(1)
            tools = AgentSession('返回本次起点', catalog, SavedShadowNavigation(catalog,r), self.store)
            self.assertEqual(tools.state['task_start']['pose'], r.record['map_base_pose'])
            self.assertIsNone(catalog.places['mission_start']['pose'])
        finally:
            dataset.close()

    def test_disconnect_during_cancel_does_not_resend_or_overwrite_user_intent(self):
        tools = self.tools()
        tools.call('navigate', place_id='elevator')
        calls = []
        def cancel(handle):
            calls.append(handle)
            raise ConnectionError('ACK unknown')
        self.nav.cancel = cancel
        tools.event('cancel')
        self.nav.tick = lambda: (_ for _ in ()).throw(ConnectionError('offline'))
        tools.tick(); tools.tick()
        self.assertEqual(len(calls),1)
        self.assertEqual(tools.state['status'],'stopping')
        self.assertEqual(tools.state['stop_target'],'cancelled')
        self.assertIsNotNone(self.store.load()['navigation']['handle'])

    def test_goal_ack_loss_is_not_resent_or_successful(self):
        tools = self.tools()
        calls=[]
        def start(place,handle):
            calls.append(handle)
            self.nav.state.update(handle=copy.deepcopy(handle),status='running',stopped=False)
            raise TimeoutError('ACK lost')
        self.nav.start=start
        result=tools.call('navigate',place_id='elevator')
        tools.tick(); tools.tick()
        self.assertEqual(result['status'],'uncertain')
        self.assertEqual(len(calls),1)
        self.assertEqual(tools.skill.record['status'],'stopping')
        self.assertEqual(tools.skill.record['completed'],[])

    def test_reset_finishes_cancel_without_completing_task(self):
        tools=self.tools()
        tools.call('navigate',place_id='elevator')
        self.nav.state['generation']=1
        tools.tick()
        self.assertEqual(tools.state['status'],'stopping')
        self.nav.stopped(); tools.tick()
        self.assertEqual(tools.state['status'],'needs_clarification')
        self.assertEqual(tools.skill.record['completed'],[])
        self.assertIsNone(tools.skill.record['handle'])

    def test_answer_survives_tool_history_eviction_and_replace_clears_it(self):
        tools=self.tools()
        actor=AgentLoop(tools,lambda *_:action('observe_scene'))
        try:
            tools.call('ask_user',question='哪个入口？')
            dispatch('answer 玻璃门内侧',actor)
            for _ in range(30):
                tools.call('get_navigation_state')
            self.assertEqual(tools.prompt()['clarifications'][0]['answer'],'玻璃门内侧')
            self.assertEqual(tools.state['instruction'],'返回本次起点')
            dispatch('replace 改去大厅中央',actor)
            self.assertEqual(tools.prompt()['clarifications'],[])
        finally:
            actor.close()

    def runtime(self):
        value=dict(control=dict(clock_domain='runtime_bag',stamp_s=10.,request=None,feedback={},
                    localization={'session_id':'s','generation':0,'frame_id':'odom','health':0}),received_age_s=.01)
        client=SimpleNamespace(snapshot=lambda **kw:value,exchange=lambda *a:{'accepted':True})
        return RuntimeShadowNavigation(self.catalog,client),value,client

    def test_runtime_cancel_requires_processed_identity(self):
        nav,value,_=self.runtime()
        handle=dict(task_id='task',goal_revision=2,call_id='call')
        nav.handle=handle;nav.cancel(handle)
        self.assertFalse(nav.snapshot()['proposal_cancelled'])
        value['control']['cancelled_request']=dict(task_id='other',goal_revision=2)
        self.assertFalse(nav.snapshot()['proposal_cancelled'])
        value['control']['cancelled_request']=dict(task_id='task',goal_revision=2)
        self.assertTrue(nav.snapshot()['proposal_cancelled'])
        self.assertFalse(nav.snapshot()['execution_feedback'])

    def test_runtime_disconnect_hides_cached_image_and_map_context_is_independent(self):
        nav,value,client=self.runtime()
        value['control']['map_binding']=dict(valid=True,map_id='map')
        self.assertEqual(nav.snapshot()['map_binding'],value['control']['map_binding'])
        self.assertFalse(nav.check_route({'pose':[1,2]})['proposal_allowed'])
        client.snapshot=lambda **kw:(_ for _ in ()).throw(ConnectionError('outage'))
        with self.assertRaises(ConnectionError):
            nav.tick()
        self.assertTrue(nav.snapshot()['input_stale'])
        self.assertIsNone(nav.observe())


class MemoryTests(unittest.TestCase):
    def test_bounded_causal_recall_preserves_contradictions_without_reviving_space(self):
        memory=EvidenceMemory(capacity=2)
        context=dict(task_id='task',observation_id='o',observation_binding='input',source_stamp_s=10.,local_identity=['s',0,'odom'])
        for stamp,verdict in ((9.,'unknown'),(10.,'supported'),(11.,'contradicted')):
            memory.remember(dict(verdict=verdict),dict(context,source_stamp_s=stamp,observation_id=str(stamp)),
                            place_id='entrance',clock_domain='bag',map_binding={'map_id':'map'})
        self.assertEqual(len(memory.snapshot()),2)
        self.assertEqual(len(memory.recall(place_ids=['entrance'],now=10.,clock_domain='bag')),1)
        memory.invalidate(local_identity=['s',1,'odom'],clock_domain='bag',map_binding={'map_id':'map'})
        memory.invalidate(local_identity=['s',0,'odom'],clock_domain='bag',map_binding={'map_id':'map'})
        recalled=memory.recall(place_ids=['entrance'],now=12.,clock_domain='bag')
        self.assertEqual([e['semantic']['verdict'] for e in recalled],['supported','contradicted'])
        self.assertTrue(all(not e['spatial_valid'] and not e['proves_arrival'] for e in recalled))


if __name__=='__main__':
    unittest.main()
