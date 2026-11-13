"""Registration tool policy and native responses, using only mock services."""
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from decision.agent import CONFIG_DIR
from decision.agent.core.loop import AgentLoop
from decision.agent.core.model import ApiError,Model,REGISTRATION_TOOLS
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Budget,Store
from decision.agent.tools.framework import Reply
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback,point_goal,reply
from tests.test_agent_conversation import wire


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.store=Store(self.root/'session.sqlite');self.budget=Budget(self.root/'budget.sqlite')
        self.config=self.root/'private.json';self.config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
        self.session=AgentSession('先到电梯，再到大厅',Catalog.load(CONFIG_DIR/'simulation.json'),ScriptedFeedback(),self.store)

    def tearDown(self):
        self.budget.close();self.store.close();self.temp.cleanup()

    def client(self,handler):
        return Model(self.config,self.budget,self.store,transport=httpx.MockTransport(handler))

    def test_initial_phase_offers_registration_tools_then_restores_execution_tools(self):
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,text=wire(reply('lookup_places',name='电梯').message['tool_calls']))
        client=self.client(handler)
        self.assertEqual(client.decide(self.session.prompt()).action['tool'],'lookup_places')
        first=requests[-1]
        self.assertEqual(first['tool_choice'],'required')
        self.assertEqual({t['function']['name'] for t in first['tools']},set(REGISTRATION_TOOLS))
        self.assertNotIn('navigate',{t['function']['name'] for t in first['tools']})
        self.assertIn('任务登记阶段',first['messages'][0]['content'])
        self.session.call('record_task',goals=[point_goal('elevator')])
        client.decide(self.session.prompt())
        second=requests[-1]
        self.assertEqual(second['tool_choice'],'auto')
        self.assertIn('navigate',{t['function']['name'] for t in second['tools']})

    def test_unoffered_tool_is_preserved_but_never_accepted_or_retried(self):
        native=reply('get_navigation_state').message['tool_calls']
        calls=[]
        def handler(request):
            calls.append(request);return httpx.Response(200,text=wire(native))
        with self.assertRaises(ApiError):self.client(handler).decide(self.session.prompt())
        self.assertEqual(len(calls),1)
        record=json.loads(self.store.db.execute("SELECT value FROM events WHERE kind='api_response'").fetchone()[0])
        self.assertEqual(record['tool_calls'],native)
        self.assertEqual(self.session.task_goals.goals,[])

    def test_plain_text_is_visible_but_identified_as_unregistered(self):
        raw='建议按航点前往，并声称已经完成。'
        actor=AgentLoop(self.session,lambda *_:Reply.parse(dict(role='assistant',content=raw)))
        try:
            actor.tick();actor.model.thread.join(1);actor.tick()
            self.assertEqual(self.session.state['status'],'needs_clarification')
            self.assertEqual(self.session.state['reason'],'model_text_without_task_registration')
            self.assertEqual(self.session.state['assistant_text'],raw)
            self.assertFalse(self.session.task_goals.goals)
            event=json.loads(self.store.db.execute("SELECT value FROM events WHERE kind='assistant_message'").fetchone()[0])
            self.assertEqual(event['message']['content'],raw)
            self.assertFalse(event['automatically_retried'])
        finally:actor.close()

    def test_clarifying_tool_is_still_available(self):
        response=reply('ask_user',question='请选择哪个入口？')
        result=self.client(lambda _:httpx.Response(200,text=wire(response.message['tool_calls']))).decide(self.session.prompt())
        self.assertEqual(result.action['tool'],'ask_user')
        self.assertEqual(self.session.call('ask_user',**result.action['arguments'])['status'],'waiting_user_answer')


if __name__=='__main__':unittest.main()
