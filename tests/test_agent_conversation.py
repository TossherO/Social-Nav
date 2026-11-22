"""Native tool protocol, causal media and bounded dialogue; no live API."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from decision.agent import CONFIG_DIR
from decision.agent.core.conversation import Conversation
from decision.agent.core.loop import AgentLoop
from decision.agent.core.model import ApiError, Model, ModelTemporarilyUnavailable, cache_messages, cache_usage
from decision.agent.core.model_projection import ModelProjection
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Budget, Store
from decision.agent.tools.framework import Reply
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback, action, navigate, point_goal, query_map, reply


def wire(calls=None, text=None, fragments=None):
    chunks = fragments or [dict(content=text,tool_calls=[] if calls is None else
                               [dict(call,index=i) for i,call in enumerate(calls)])]
    finish = 'tool_calls' if calls else 'stop'
    values = [dict(model=Model.model,choices=[dict(delta=chunk,finish_reason=None)]) for chunk in chunks]
    values += [dict(model=Model.model,choices=[dict(delta={},finish_reason=finish)]),
               dict(choices=[],usage=dict(prompt_tokens=100,completion_tokens=20))]
    return ''.join('data: '+json.dumps(value)+'\n\n' for value in values)+'data: [DONE]\n\n'


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root/'session.sqlite')
        self.budget = Budget(self.root/'budget.sqlite')
        self.nav = ScriptedFeedback()
        self.session = AgentSession('到电梯口',Catalog.load(CONFIG_DIR/'simulation.json'),self.nav,self.store)
        self.config = self.root/'private.json'
        self.config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='unit-secret')))

    def tearDown(self):
        self.budget.close();self.store.close();self.temp.cleanup()

    def client(self, handler):
        return Model(self.config,self.budget,self.store,transport=httpx.MockTransport(handler))

    def test_registration_retry_waits_and_rebuilds_request(self):
        requests=[]
        def work(payload, _observation, _view):
            requests.append(payload)
            if len(requests)==1:
                raise ModelTemporarilyUnavailable()
            return reply('record_task',goals=[point_goal('elevator')])
        actor=AgentLoop(self.session,work)
        with patch('decision.agent.core.loop.time') as clock:
            clock.monotonic.return_value=100.
            actor.tick()
            actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(actor.state['reason'],'model_temporarily_unavailable')
            clock.monotonic.return_value=104.9
            actor.tick()
            self.assertEqual(len(requests),1)
            clock.monotonic.return_value=105.
            actor.tick()
            actor.model.thread.join(1)
            actor.tick()
            self.assertIsNone(actor.state['reason'])
            self.assertEqual(self.session.task_goals.goals[0]['requirement']['target_id'],'elevator')
            self.assertIsNot(requests[0],requests[1])
        actor.close()

    def test_cancel_during_network_retry_wait_prevents_another_request(self):
        def work(*_args):
            raise ModelTemporarilyUnavailable()
        actor=AgentLoop(self.session,work)
        with patch('decision.agent.core.loop.time') as clock:
            clock.monotonic.return_value=100.
            actor.tick()
            actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(actor.state['reason'],'model_temporarily_unavailable')
            actor.event('cancel')
            clock.monotonic.return_value=106.
            actor.tick()
            self.assertEqual(actor.state['status'],'cancelled')
            self.assertEqual(actor.requests,1)
        actor.close()

    def test_network_retry_backs_off_and_obeys_longer_service_delay(self):
        failures=iter([None,None,None,None,None,75.,None])
        def work(*_args):
            raise ModelTemporarilyUnavailable(retry_after_s=next(failures))
        actor=AgentLoop(self.session,work)
        with patch('decision.agent.core.loop.time') as clock:
            now=100.
            clock.monotonic.return_value=now
            actor.tick()
            for count,delay in enumerate((5.,10.,20.,40.,60.,75.,60.),start=1):
                actor.model.thread.join(1)
                actor.tick()
                self.assertEqual(actor.model_retry_at,now+delay)
                clock.monotonic.return_value=now+delay-.1
                actor.tick()
                self.assertEqual(actor.requests,count)
                now+=delay
                clock.monotonic.return_value=now
                if count<7:
                    actor.tick()
        actor.close()

    def test_default_history_retains_more_then_trims_old_complete_groups(self):
        events=[]
        conversation=Conversation('原任务',{},lambda kind,value:events.append((kind,value)))
        for i in range(20):conversation.notice('test',index=i,text='x'*900)
        self.assertEqual(len(conversation.exchanges),20)
        self.assertFalse(any(kind=='conversation_trimmed' for kind,_ in events))
        for i in range(20,33):conversation.notice('test',index=i,text='x'*900)
        self.assertEqual(len(conversation.exchanges),32)
        self.assertEqual(json.loads(conversation.exchanges[0][0]['content'])['index'],1)
        self.assertEqual(conversation.messages()[0]['content'],'原任务')
        self.assertTrue(any(kind=='conversation_trimmed' for kind,_ in events))

    def test_history_size_limit_still_trims_old_groups(self):
        conversation=Conversation('原任务',{},lambda *_:None)
        conversation.notice('test',index=1,text='x'*25000)
        conversation.notice('test',index=2,text='x'*25000)
        self.assertEqual(len(conversation.exchanges),1)
        self.assertEqual(json.loads(conversation.exchanges[0][0]['content'])['index'],2)

    def test_fragmented_native_arguments_and_separate_tools_preserve_original_reply(self):
        call = reply('lookup_places',name='电梯口').message['tool_calls'][0]
        raw = call['function']['arguments']
        parts = [dict(content='先查询地点。',tool_calls=[dict(index=0,id=call['id'],type='function',
                  function=dict(name='lookup_places',arguments=raw[:5]))]),
                 dict(tool_calls=[dict(index=0,id=call['id'],type='function',function=dict(arguments=raw[5:]))])]
        requests=[]
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,text=wire([call],fragments=parts))
        response = self.client(handle).decide(self.session.prompt())
        self.assertEqual(response.action,action('lookup_places',name='电梯口'))
        self.assertEqual(response.message['tool_calls'][0],call)
        self.assertEqual(response.message['content'],'先查询地点。')
        body=requests[0]
        self.assertIn('tools',body);self.assertNotIn('response_format',body)
        self.assertFalse(body['parallel_tool_calls'])
        self.assertNotIn('tools',json.loads(body['messages'][-1]['content'][0]['text'])['state'])

    def test_native_tool_result_pair_is_sent_on_the_next_request(self):
        seen=[]
        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200,text=wire(reply('lookup_places',name='').message['tool_calls']))
        client=self.client(handle)
        first=client.decide(self.session.prompt())
        value=self.session.call(first.action['tool'],**first.action['arguments'])
        self.session.conversation.commit(first,value)
        client.decide(self.session.prompt())
        messages=seen[1]['messages']
        assistant=next(m for m in messages if m['role']=='assistant')
        result=next(m for m in messages if m['role']=='tool')
        self.assertEqual(assistant,first.message)
        self.assertEqual(result['tool_call_id'],first.message['tool_calls'][0]['id'])
        self.assertEqual(json.loads(result['content']),value)

    def test_explicit_cache_marks_stable_prefixes_and_preserves_fresh_tail(self):
        original=self.session.prompt()['messages']
        native=reply('lookup_places',name='').message
        original += [native,dict(role='tool',tool_call_id=native['tool_calls'][0]['id'],content='{"status":"available"}'),
                     dict(role='user',content=[dict(type='text',text='current state'),
                          dict(type='image_url',image_url=dict(url='data:image/png;base64,unit'))])]
        frozen=copy.deepcopy(original)
        first=cache_messages('fixed system',original,'explicit')
        self.assertEqual(original,frozen)
        marks=[i for i,m in enumerate(first) if isinstance(m.get('content'),list)
               and any('cache_control' in p for p in m['content'])]
        self.assertEqual(marks,[0,2,len(first)-2])
        self.assertEqual(first[-1],original[-1])
        self.assertEqual(first[-2]['tool_call_id'],native['tool_calls'][0]['id'])
        self.assertEqual(json.loads(first[-2]['content'][0]['text']),dict(status='available'))
        changed=copy.deepcopy(original);changed[-1]['content'][0]['text']='new current state'
        second=cache_messages('fixed system',changed,'explicit')
        self.assertEqual(first[:-1],second[:-1])
        self.assertNotEqual(first[-1],second[-1])
        self.assertNotIn('cache_control',json.dumps(cache_messages('fixed system',original,'implicit')))

    def test_missing_cache_usage_remains_unknown(self):
        stats=cache_usage(dict(prompt_tokens=100),'explicit')
        self.assertIsNone(stats['cached_tokens'])
        self.assertIsNone(stats['hit_fraction'])
        self.assertIsNone(stats['cache_creation_input_tokens'])
        stats=cache_usage(dict(prompt_tokens=100,prompt_tokens_details=dict(cached_tokens=60,cache_creation_input_tokens=20)),'explicit')
        self.assertEqual(stats['hit_fraction'],.6)
        self.assertEqual(stats['cache_creation_input_tokens'],20)

    def test_model_stream_has_idle_read_timeout(self):
        seen=[]
        def handle(request):
            seen.append(request.extensions.get('timeout'))
            return httpx.Response(200,text=wire(reply('lookup_places',name='').message['tool_calls']))
        self.client(handle).decide(self.session.prompt())
        self.assertEqual(seen[0]['read'],120.)

    def test_metadata_is_referenced_only_while_full_tool_result_is_retained(self):
        view_id=query_map(self.session)
        view=self.session.spatial_view
        original=copy.deepcopy(view.record)
        call=reply('query_map')
        self.session.conversation.commit(call,dict(status='available',view=view.record,target_candidates_generated=False))
        seen=[]
        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200,text=wire(reply('lookup_places',name='').message['tool_calls']))
        client=self.client(handle)
        client.decide(self.session.prompt(),spatial_view=view)
        self.session.conversation.max_exchanges=1
        self.session.conversation.notice('test_trim_trigger')
        client.decide(self.session.prompt(),spatial_view=view)
        def context(body):
            return next(json.loads(p['text']) for p in body['messages'][-1]['content'] if p['type']=='text'
                        and p['text'].startswith('{') and json.loads(p['text']).get('source')=='current_spatial_view')
        referenced=context(seen[0]);full=context(seen[1])
        self.assertEqual(referenced['details_in_tool'],'query_map')
        projection=ModelProjection(self.session.model_reference_names)
        self.assertEqual(referenced['view']['view_id'],projection.reference('view',view_id))
        self.assertNotIn('annotations',referenced['view'])
        self.assertNotIn('details_in_tool',full)
        self.assertEqual(full['view']['view_id'],projection.reference('view',original['view_id']))
        self.assertEqual(full['view']['annotation_reference']['reference_map_id'],projection.reference('map',self.session.reference_map.record['reference_map_id']))
        self.assertNotIn('annotations',full['view'])
        self.assertEqual(view.record,original)
        self.assertLess(len(json.dumps(referenced)),len(json.dumps(full)))
        for body in seen:
            self.assertEqual(sum(p['type']=='image_url' for p in body['messages'][-1]['content']),2)

    def test_multiple_tool_calls_are_recorded_and_rejected_without_execution_or_retry(self):
        calls=[reply('observe_scene').message['tool_calls'][0],reply('get_navigation_state').message['tool_calls'][0]]
        seen=[]
        def handle(request):
            seen.append(request)
            return httpx.Response(200,text=wire(calls))
        with self.assertRaises(ApiError):self.client(handle).decide(self.session.prompt())
        self.assertEqual(len(seen),1)
        saved=json.loads(self.store.db.execute("SELECT value FROM events WHERE kind='api_response'").fetchone()[0])
        self.assertEqual(saved['tool_calls'],calls)
        self.assertEqual(self.session.conversation.snapshot()['consumed_tool_calls'],0)

    def test_text_only_reply_waits_for_user_without_claiming_success_or_parsing_action_json(self):
        for text in ('任务完成。',json.dumps(action('get_navigation_state'))):
            response=Reply.parse(dict(role='assistant',content=text))
            actor=AgentLoop(self.session,lambda *_:response)
            try:
                actor.tick();actor.model.thread.join(1);actor.tick()
                self.assertEqual(self.session.state['status'],'needs_clarification')
                self.assertEqual(self.session.state['assistant_text'],text)
                self.assertFalse(self.session.state['completed_revisions'])
                self.assertEqual(self.session.conversation.messages()[-1]['content'],text)
                self.assertEqual(actor.requests,1)
            finally:actor.close()
            self.session.event('answer','继续检查')

    def test_arrival_text_corrected_once_using_current_evidence(self):
        self.assertEqual(navigate(self.session,'elevator')['status'],'accepted')
        self.session.call('wait_for_event',event='navigation')
        self.session.tick()
        self.nav.arrive()
        evidence=[]
        def work(payload,*_):
            state=payload['current_state']
            evidence.append(state['completion_evidence'])
            if len(evidence)==1:
                return Reply.parse(dict(role='assistant',content='已经到达。'))
            self.assertIn('原生工具',state['decision_feedback'])
            return reply('complete_subtask',goal_id='elevator',evidence='到达电梯口且已停稳')
        actor=AgentLoop(self.session,work)
        try:
            actor.tick();actor.model.thread.join(2);actor.tick()
            self.assertEqual(self.session.state['status'],'ready')
            self.nav.t+=.1
            actor.tick();actor.model.thread.join(2);actor.tick()
            self.assertEqual(self.session.state['status'],'succeeded')
            self.assertEqual([row['missing'] for row in evidence],[[],[]])
            self.assertGreater(evidence[1]['measurement_stamp_s'],evidence[0]['measurement_stamp_s'])
            self.assertEqual(actor.requests,2)
            self.assertFalse(self.session.state['clarifications'])
        finally:actor.close()

    def test_repeated_execution_text_pauses_without_parsing_or_requesting_clarification(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        raw=json.dumps(action('complete_subtask',goal_id='elevator',evidence='到达'))
        actor=AgentLoop(self.session,lambda *_:Reply.parse(dict(role='assistant',content=raw)))
        try:
            for _ in range(8):
                actor.tick()
                if actor.model.thread:actor.model.thread.join(2)
            self.assertEqual(actor.requests,2)
            self.assertEqual(self.session.state['status'],'paused')
            self.assertEqual(self.session.state['reason'],'model_action_missing')
            self.assertEqual(self.session.task_goals.goals[0]['status'],'pending')
            rows=self.store.db.execute("SELECT value FROM events WHERE kind='model_action_missing' ORDER BY seq")
            logged=[json.loads(row[0]) for row in rows]
            self.assertEqual([r['message']['content'] for r in logged],[raw,raw])
            self.assertEqual([r['correction_requested'] for r in logged],[True,False])
        finally:actor.close()

    def test_accepted_goal_waits_for_feedback_before_next_model_request(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        query_map(self.session)
        calls=[]
        def work(_payload,_observation,view):
            calls.append(1)
            if len(calls)==1:
                return reply('navigate',goal_id='elevator',pose=[5,4,0],
                             view_id=view.record['view_id'],orientation_source='autonomous')
            return reply('check_goal',goal_id='elevator',review_result='satisfied',review_evidence='已到达目标并停稳')
        actor=AgentLoop(self.session,work)
        try:
            actor.tick();actor.model.thread.join(1);actor.tick()
            self.assertEqual(self.session.state['status'],'waiting_navigation')
            self.assertEqual(self.nav.state['status'],'running')
            for _ in range(5):actor.tick()
            self.assertEqual(len(calls),1)
            self.assertEqual(self.session.skill.record['completed'],[])
            self.nav.arrive()
            actor.tick();actor.model.thread.join(1)
            self.assertEqual(len(calls),2)
            actor.tick()
            self.assertEqual(self.session.task_goals.goals[0]['status'],'completed')
        finally:
            actor.close()

    def test_user_cancel_still_cancels_while_waiting_for_navigation(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        query_map(self.session)
        actor=AgentLoop(self.session,lambda _payload,_observation,view:reply('navigate',goal_id='elevator',
                        pose=[5,4,0],view_id=view.record['view_id'],orientation_source='autonomous'))
        try:
            actor.tick();actor.model.thread.join(1);actor.tick()
            self.assertEqual(self.session.state['status'],'waiting_navigation')
            actor.event('cancel')
            self.assertEqual(self.nav.state['status'],'stopping')
            self.nav.stopped();actor.tick()
            self.assertEqual(self.session.state['status'],'cancelled')
            self.assertFalse(self.session.state['completed_revisions'])
        finally:
            actor.close()

    def test_reused_call_identity_is_rejected_before_second_tool_execution(self):
        response=reply('get_navigation_state')
        actor=AgentLoop(self.session,lambda *_:response)
        try:
            actor.tick();actor.model.thread.join(1);actor.tick()
            actor.model.thread.join(1);actor.tick()
            self.assertEqual(self.session.state['status'],'needs_clarification')
            count=self.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='tool_intent'").fetchone()[0]
            self.assertEqual(count,1)
        finally:actor.close()

    def test_trim_keeps_whole_pairs_original_task_and_authoritative_requirements(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        self.session.conversation.max_exchanges=2
        for _ in range(5):
            self.session.conversation.commit(reply('get_navigation_state'),dict(status='idle'))
        prompt=self.session.prompt()
        self.assertEqual(prompt['messages'][0]['content'],'到电梯口')
        self.assertEqual(json.loads(prompt['messages'][2]['content'])['source'],'registered_task_plan')
        pairs=prompt['messages'][3:]
        self.assertEqual([m['role'] for m in pairs],['assistant','tool','assistant','tool'])
        for a,b in zip(pairs[::2],pairs[1::2]):self.assertEqual(a['tool_calls'][0]['id'],b['tool_call_id'])
        self.assertEqual(prompt['current_state']['task_goals'][0],dict(goal_id='elevator',status='pending'))
        self.assertEqual(json.loads(prompt['messages'][2]['content'])['requirements'][0]['target_id'],'elevator')
        self.assertNotIn('history',prompt)
        self.assertGreater(self.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='conversation_exchange'").fetchone()[0],2)

    def test_replace_starts_new_dialogue_and_keeps_old_events(self):
        self.session.conversation.commit(reply('lookup_places',name=''),dict(status='available'))
        self.session.event('replace','到大厅中央')
        self.assertEqual(self.session.conversation.messages()[0]['content'],'到大厅中央')
        self.assertFalse(any(m['role']=='assistant' for m in self.session.conversation.messages()))
        self.assertEqual(self.session.state['revision'],2)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='conversation_exchange'").fetchone()[0],1)

    def test_maps_are_current_attachments_and_not_retained_as_history_pixels(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        view_id=query_map(self.session)
        seen=[]
        outputs=[reply('navigate',goal_id='elevator',pose=[5,4,0],view_id=view_id,orientation_source='autonomous'),
                 reply('wait_for_event',event='navigation')]
        def handle(request):
            body=json.loads(request.content);seen.append(body)
            selected=outputs[len(seen)-1].message['tool_calls']
            if len(seen)==1:
                selected=copy.deepcopy(selected)
                args=json.loads(selected[0]['function']['arguments'])
                state=json.loads(body['messages'][-1]['content'][0]['text'])['state']
                args['view_id']=state['visual_inputs']['maps']['view_id']
                selected[0]['function']['arguments']=json.dumps(args)
            return httpx.Response(200,text=wire(selected))
        client=self.client(handle)
        first=client.decide(self.session.prompt(),spatial_view=self.session.spatial_view)
        self.assertEqual(first.action['arguments']['view_id'],view_id)
        result=self.session.call(first.action['tool'],**first.action['arguments'])
        self.assertEqual(result['status'],'accepted')
        self.session.conversation.commit(first,result)
        client.decide(self.session.prompt(),spatial_view=self.session.spatial_view)
        def images(body):
            return [part for m in body['messages'] if isinstance(m.get('content'),list)
                    for part in m['content'] if part['type']=='image_url']
        self.assertEqual(len(images(seen[0])),3)
        self.assertEqual(len(images(seen[1])),3)
        self.assertEqual(seen[0]['messages'][1],seen[1]['messages'][1])
        self.assertEqual(sum(p['type']=='image_url' for p in seen[1]['messages'][-1]['content']),2)
        self.assertNotIn('data:image',json.dumps(self.session.conversation.messages()))
        self.assertEqual(self.session.spatial_view.record['source_stamp_s'],self.nav.state['stamp_s'])

    def test_retained_tool_alias_stays_readable_but_cannot_act_on_a_changed_map_view(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        query_map(self.session)
        old = self.session.prompt()
        old_view = self.session.spatial_view
        old_id = old_view.record['view_id']
        alias = ModelProjection(self.session.model_reference_names).reference('view',old_id)
        self.assertEqual(alias,'v1')
        call = reply('check_candidate',pose=[5.,4.,0.],view_id=alias)
        seen = []

        def handle(request):
            seen.append(json.loads(request.content))
            tool_call = call.message['tool_calls'][0]
            return httpx.Response(200,text=wire([dict(tool_call,id=tool_call['id']+str(len(seen)))]))

        client=self.client(handle)
        accepted=client.decide(old,spatial_view=old_view)
        self.assertEqual(accepted.action['arguments']['view_id'],old_id)
        # Keep the old alias in the assistant arguments alone.
        self.session.conversation.commit(accepted,dict(status='available'))
        self.nav.state['stamp_s']+=.5
        query_map(self.session)
        current=self.session.prompt()
        self.assertNotEqual(self.session.spatial_view.record['view_id'],old_id)
        refused=client.decide(current,spatial_view=self.session.spatial_view)
        self.assertIsNone(refused.action)
        self.assertEqual(refused.argument_error['reason'],'invalid_tool_reference')
        previous,replayed = [m for m in seen[1]['messages'] if m['role'] in ('assistant','tool')]
        self.assertEqual(previous,accepted.message)
        self.assertEqual(previous['tool_calls'][0]['id'],replayed['tool_call_id'])
        state=json.loads(seen[1]['messages'][-1]['content'][0]['text'])['state']
        self.assertEqual(state['visual_inputs']['maps']['view_id'],'v2')
        logged = [json.loads(row[0]) for row in self.store.db.execute(
            "SELECT value FROM events WHERE kind='api_request' ORDER BY seq")]
        self.assertEqual(logged[0]['model_references']['v1'],['view',old_id])
        self.assertEqual(logged[1]['model_references']['v1'],['view',old_id])
        self.assertEqual(logged[1]['model_references']['v2'],['view',self.session.spatial_view.record['view_id']])
        for body in seen:
            self.assertNotIn('model_references',body)
            self.assertNotIn(old_id,json.dumps(body))


if __name__=='__main__':unittest.main()
