"""Offline checks for manual GUI data and identity-bound commands."""
import base64
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from decision.agent.core.store import Store
from decision.agent.ui_data import ContextTimeline,connect_readonly,event_detail,event_summary,message_detail,poll_session,request_context
from decision.agent.workflows.interaction import dispatch_structured


class ManualUiTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.session=SimpleNamespace(state=dict(task_id='task',revision=2),
            navigation=SimpleNamespace(scope='manual'),
            task_goals=SimpleNamespace(goals=[dict(status='pending',requirement=dict(goal_id='first'))]),
            skill=SimpleNamespace(record=dict(handle=dict(call_id='call'))))
        self.actor=SimpleNamespace(session=self.session,event=lambda name,text:self.calls.append((name,text)))

    def command(self,**changes):
        value=dict(command_id='id',name='arrived',task_id='task',revision=2,
                   goal_id='first',call_id='call',text=None)
        value.update(changes)
        return json.dumps(value,ensure_ascii=False)

    def test_command_checks_task_goal_handle_and_never_replays(self):
        used=set()
        bad=dispatch_structured(self.command(call_id='old'),self.actor,used)
        self.assertEqual(bad['reason'],'task_or_goal_changed');self.assertEqual(self.calls,[])
        self.assertEqual(dispatch_structured(self.command(),self.actor,used)['reason'],'duplicate_command')
        ok=dispatch_structured(self.command(command_id='new'),self.actor,used)
        self.assertTrue(ok['accepted']);self.assertEqual(self.calls,[('arrived',None)])
        self.session.navigation.scope='live'
        bad=dispatch_structured(self.command(command_id='live'),self.actor,used)
        self.assertFalse(bad['accepted']);self.assertEqual(len(self.calls),1)

    def test_answer_is_single_structured_event(self):
        result=dispatch_structured(self.command(name='answer',text='第一行\n第二行'),self.actor,set())
        self.assertTrue(result['accepted']);self.assertEqual(self.calls,[('answer','第一行\n第二行')])

    def test_readonly_history_omits_images_until_detail_and_redacts_secret(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.sqlite';store=Store(path)
            store.save(dict(task_id='task'), 'task_created', {})
            image=b'example image data'
            data='data:image/png;base64,'+base64.b64encode(image).decode()
            store.event('api_request',dict(call_id='call',model='test',observation_id='obs',
                request=dict(messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=data))])],
                             authorization='secret')))
            db=connect_readonly(path)
            try:
                state,events=poll_session(db,0)
                self.assertEqual(state['task_id'],'task')
                self.assertNotIn('base64',json.dumps(events))
                kind,detail,images=event_detail(db,events[-1]['seq'])
                self.assertEqual(kind,'api_request');self.assertEqual(images[0][1],image)
                self.assertEqual(detail['request']['authorization'],'[已隐藏]')
                self.assertNotIn(data,json.dumps(detail))
                with self.assertRaises(sqlite3.OperationalError):db.execute('DELETE FROM events')
            finally:db.close();store.close()

    def test_arrival_evidence_diagnostics_are_visible_in_history(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.sqlite';store=Store(path)
            store.save(dict(task_id='task'),'task_created',{})
            store.event('manual_goal_evidence_unproven',dict(goal_id='first',issues=['mapped_pose_unavailable']))
            db=connect_readonly(path)
            try:
                _,events=poll_session(db,0)
                diagnostic=events[-1]
                self.assertEqual(diagnostic['kind'],'manual_goal_evidence_unproven')
                self.assertIn('mapped_pose_unavailable',event_summary(diagnostic))
                self.assertEqual(event_detail(db,diagnostic['seq'])[1]['goal_id'],'first')
            finally:db.close();store.close()

    def test_actual_request_messages_keep_all_images_and_tool_arguments(self):
        image='data:image/png;base64,'+base64.b64encode(b'image').decode()
        messages=[dict(role='system',content='规则'),dict(role='user',content=[
            dict(type='text',text='查看场景'),dict(type='image_url',image_url=dict(url=image)),
            dict(type='image_url',image_url=dict(url=image))]),
            dict(role='assistant',content=None,tool_calls=[dict(id='call',type='function',
                 function=dict(name='lookup_places',arguments='{"name":"门"}'))]),
            dict(role='tool',tool_call_id='call',content='{"status":"available"}')]
        value=dict(request=dict(messages=messages,tools=[dict(function=dict(name='lookup_places'))]))
        context=request_context(value)
        self.assertEqual([m['role'] for m in context['messages']],['system','user','assistant','tool'])
        self.assertEqual(context['messages'][1]['images'],[0,1])
        self.assertNotIn(image,json.dumps(context))
        self.assertEqual(context['tools'],['lookup_places'])
        self.assertIn('lookup_places',context['messages'][2]['summary'])
        self.assertEqual(message_detail(messages[2])['工具调用参数'][0]['arguments'],{'name':'门'})
        self.assertEqual(messages[2]['tool_calls'][0]['function']['arguments'],'{"name":"门"}')
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite');store.event('api_request',value)
            db=connect_readonly(Path(root)/'session.sqlite')
            try:
                _,events=poll_session(db,0)
                self.assertEqual(events[0]['context'],context)
                _,detail,images=event_detail(db,events[0]['seq'])
                self.assertEqual(len(images),2)
                self.assertEqual(detail['request']['messages'][2]['tool_calls'][0]['function']['arguments'],
                                 '{"name":"门"}')
            finally:db.close();store.close()

    def test_tool_summary_removes_trailing_blank_lines_without_changing_message(self):
        message=dict(role='assistant',content='准备查询。\n\n',tool_calls=[dict(id='call',type='function',
            function=dict(name='lookup_places',arguments='{}'))])
        context=request_context(dict(request=dict(messages=[message])))
        self.assertEqual(context['messages'][0]['summary'],'准备查询。 lookup_places')
        self.assertEqual(message['content'],'准备查询。\n\n')
        self.assertEqual(message_detail(message)['原始消息']['content'],'准备查询。\n\n')

    def test_system_summary_uses_one_line_without_altering_original(self):
        message=dict(role='system',content='第一段规则。\n\n第二段规则。\n')
        context=request_context(dict(request=dict(messages=[message])))
        self.assertEqual(context['messages'][0]['summary'],'第一段规则。 第二段规则。')
        self.assertEqual(message_detail(message)['content'],'第一段规则。\n\n第二段规则。\n')

    def test_timeline_appends_once_and_marks_removed_context(self):
        timeline=ContextTimeline()
        system=dict(role='system',content='规则')
        user=dict(role='user',content='导航')
        state1=dict(role='user',content='当前观察 1')
        state2=dict(role='user',content='当前观察 2')
        assistant=dict(role='assistant',content=None,tool_calls=[dict(id='a',type='function',
            function=dict(name='lookup_places',arguments='{}'))])
        result=dict(role='tool',tool_call_id='a',content='找到地点')
        def context(messages):return request_context(dict(request=dict(messages=messages)))
        timeline.request(context([system,user,state1]),1)
        timeline.response(assistant,2)
        timeline.exchange([assistant,result],3)
        self.assertEqual(len(timeline.entries),5)
        timeline.request(context([system,user,assistant,result,state2]),4)
        self.assertEqual(len(timeline.entries),6)
        self.assertEqual(sum(x['role']=='assistant' for x in timeline.entries),1)
        self.assertFalse(timeline.entries[2]['active'])
        self.assertTrue(all(not x['pending'] for x in timeline.entries if x['active']))
        timeline.request(context([system,user,state2]),5)
        self.assertEqual(len(timeline.entries),6)
        self.assertFalse(timeline.entries[3]['active'])
        self.assertFalse(timeline.entries[4]['active'])

    def test_timeline_keeps_repeated_occurrences_distinct(self):
        timeline=ContextTimeline()
        message=dict(role='user',content='相同消息')
        context=request_context(dict(request=dict(messages=[message,message])))
        timeline.request(context,1)
        self.assertEqual(len(timeline.entries),2)
        timeline.request(request_context(dict(request=dict(messages=[message]))),2)
        self.assertEqual(sum(x['active'] for x in timeline.entries),1)
        timeline.request(context,3)
        self.assertEqual(len(timeline.entries),3)
        self.assertEqual(sum(x['active'] for x in timeline.entries),2)

    def test_tool_projection_updates_one_entry_without_losing_original_identity(self):
        timeline=ContextTimeline()
        assistant=dict(role='assistant',content=None,tool_calls=[dict(id='call-1',type='function',
            function=dict(name='query_map',arguments='{}'))])
        tool=dict(role='tool',tool_call_id='call-1',content='{"view":{"large":"full"}}')
        projected=dict(tool,content='{"view":{"reference":"map"}}')
        def context(messages):return request_context(dict(request=dict(messages=messages)))
        timeline.request(context([dict(role='system',content='规则')]),1)
        timeline.response(assistant,2)
        timeline.exchange([assistant,tool],3)
        timeline.request(context([dict(role='system',content='规则'),assistant,projected]),4)
        self.assertEqual(len(timeline.entries),3)
        entry=timeline.entries[-1]
        self.assertTrue(entry['active'])
        self.assertTrue(entry['representation_changed'])
        self.assertEqual((entry['kind'],entry['seq'],entry['index']),('api_request',4,2))

    def test_context_view_shows_one_row_and_grays_removed_message(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            shared=dict(role='system',content='规则')
            removed=dict(role='assistant',content='历史答案')
            first=dict(seq=1,kind='api_request',wall_s=0.,value=dict(model='test'),
                       context=request_context(dict(request=dict(messages=[shared,removed]))))
            second=dict(seq=2,kind='api_request',wall_s=1.,value=dict(model='test'),
                        context=request_context(dict(request=dict(messages=[shared]))))
            window.update_session(None,[first,second])
            self.assertEqual(window.context_tree.topLevelItemCount(),2)
            self.assertEqual(window.context_tree.topLevelItem(0).foreground(0).color().name(),'#7755aa')
            self.assertEqual(window.context_tree.topLevelItem(1).foreground(0).color().name(),'#9aa3b0')
        finally:
            window.close()
            app.processEvents()


if __name__=='__main__':unittest.main()
