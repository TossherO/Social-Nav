"""Offline checks for manual GUI data and identity-bound commands."""
import base64
import json
import sqlite3
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from decision.agent.core.store import Store
from decision.agent.ui_data import ContextTimeline,connect_readonly,display_value,event_detail,event_summary,message_detail,poll_session,request_context,select_detail
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

    def test_elevator_requests_and_replies_are_visible_then_cleared(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtCore, QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            image='data:image/jpeg;base64,'+base64.b64encode(b'image').decode()
            request=dict(messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=image))])])
            window.show_elevator_transient(dict(kind='request',call_id='call',request=request))
            self.assertIn('1 条消息 · 1 张图片',window.request_summary.text())
            item=window.transient_items[0][1]
            window.show_elevator_transient(dict(kind='response',call_id='call',message=dict(content='尚未完成')))
            self.assertEqual(item.childCount(),1)
            self.assertIn('尚未完成',item.child(0).text(2))
            seq=item.data(0,QtCore.Qt.UserRole)
            self.assertEqual(window.transient_requests[seq]['response']['content'],'尚未完成')
            window.clear_elevator_transient()
            self.assertFalse(window.transient_requests)
            self.assertFalse(window.transient_calls)
            self.assertEqual(window.current_tree.topLevelItemCount(),0)
            self.assertEqual(window.context_tree.indexOfTopLevelItem(item),-1)
            self.assertFalse(window.request_button.isEnabled())
        finally:
            window.close();app.processEvents()

    def test_planning_thinking_selection_is_passed_to_each_new_task(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as root:
            socket=Path(root)/'commands.sock';socket.touch()
            catalog=Path(root)/'catalog.json';catalog.write_text('{}')
            window=ManualWindow(socket=str(socket),catalog=str(catalog))
            try:
                self.assertEqual(window.planning_thinking.currentData(),'low')
                window.instruction.setText('测试任务')
                with patch.object(window,'set_reader',return_value=True), patch.object(window.process,'start') as start:
                    for mode in ('off','low','medium'):
                        window.planning_thinking.setCurrentIndex(window.planning_thinking.findData(mode))
                        window.start_task()
                        args=start.call_args.args[1]
                        self.assertEqual(args[args.index('--thinking-mode')+1],'planning')
                        self.assertEqual(args[args.index('--planning-thinking-mode')+1],mode)
            finally:
                window.close();app.processEvents()

    def test_optional_task_bag_is_only_enabled_for_live_manual_session(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as root:
            socket=Path(root)/'commands.sock';socket.touch()
            catalog=Path(root)/'catalog.json';catalog.write_text('{}')
            window=ManualWindow(socket=str(socket),catalog=str(catalog))
            try:
                window.instruction.setText('测试任务')
                with patch.object(window,'set_reader',return_value=True),patch.object(window.process,'start') as start:
                    window.start_task();self.assertNotIn('--record-bag',start.call_args.args[1])
                    window.record_bag.setChecked(True);window.start_task()
                    self.assertIn('--record-bag',start.call_args.args[1])
            finally:window.close();app.processEvents()
            replay=ManualWindow(socket=str(socket),catalog=str(catalog),replay=True)
            try:
                replay.instruction.setText('测试任务');replay.record_bag.setChecked(True)
                with patch.object(replay,'set_reader',return_value=True),patch.object(replay.process,'start') as start:
                    replay.start_task();self.assertNotIn('--record-bag',start.call_args.args[1])
            finally:replay.close();app.processEvents()

    def test_bag_finalization_status_stays_visible_after_agent_exits(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as root:
            window=ManualWindow()
            try:
                window.session_path=Path(root)/'session.sqlite'
                status=Path(root)/'bag_recording_1.json'
                status.write_text(json.dumps(dict(status='finalizing',bag='/tmp/session/navigation.bag')))
                window.refresh_recording()
                self.assertIn('正在关闭与校验',window.recording_label.text())
                status.write_text(json.dumps(dict(status='saved',bag='/tmp/session/navigation.bag')))
                window.refresh_recording()
                self.assertIn('已保存',window.recording_label.text())
                self.assertIn('/tmp/session/navigation.bag',window.recording_label.text())
            finally:window.close();app.processEvents()

    def test_replay_uses_guarded_shadow_and_disables_manual_confirmation(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as root:
            socket=Path(root)/'commands.sock';socket.touch()
            catalog=Path(root)/'catalog.json';catalog.write_text('{}')
            window=ManualWindow(socket=str(socket),catalog=str(catalog),replay=True)
            try:
                window.instruction.setText('跟随前方行人')
                with patch.object(window,'set_reader',return_value=True), patch.object(window.process,'start') as start:
                    window.start_task();args=start.call_args.args[1]
                self.assertIn('shadow',args);self.assertIn('--require-bag',args);self.assertNotIn('manual',args)
            finally:
                window.close();app.processEvents()

    def test_shadow_gui_rejects_arrival_but_permits_cancel(self):
        self.session.navigation.scope='shadow'
        used=set()
        self.assertEqual(dispatch_structured(self.command(),self.actor,used)['reason'],'unavailable_in_shadow')
        result=dispatch_structured(self.command(command_id='cancel',name='cancel'),self.actor,used)
        self.assertTrue(result['accepted'])
        self.assertEqual(self.calls,[('cancel',None)])

    def test_replay_guard_requires_bag_and_no_hardware(self):
        from decision.agent.workflows.shadow import require_bag_session
        with patch('runtime.transport.exchange_request') as exchange:
            for status in (dict(live=True,hardware_enabled=False),dict(live=False,hardware_enabled=True),{}):
                exchange.return_value=status
                with self.assertRaises(ValueError):require_bag_session('commands.sock')
            exchange.return_value=dict(live=False,hardware_enabled=False)
            require_bag_session('commands.sock')

    def test_reuse_preview_locks_instruction_and_passes_checked_source(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.sqlite';path.touch()
            socket=Path(root)/'commands.sock';socket.touch()
            catalog=Path(root)/'catalog.json';catalog.write_text('{}')
            plan=dict(instruction='先到门，再到电梯',initial_floor='floor2',digest='selection-digest',clarifications=[],
                      goals=[dict(goal_id='one',kind='point',target_id='door',description='到门')])
            window=ManualWindow(socket=str(socket),catalog=str(catalog))
            try:
                with patch.object(QtWidgets.QFileDialog,'getOpenFileName',return_value=(str(path),'')), \
                     patch('decision.agent.core.reuse.load_registered_task',return_value=plan), \
                     patch.object(QtWidgets.QDialog,'exec_',return_value=QtWidgets.QDialog.Accepted):
                    window.choose_reuse()
                self.assertTrue(window.instruction.isReadOnly());self.assertEqual(window.instruction.text(),plan['instruction'])
                self.assertFalse(window.planning_thinking.isEnabled())
                with patch.object(window,'set_reader',return_value=True),patch.object(window.process,'start') as start:
                    window.start_task();args=start.call_args.args[1]
                    self.assertEqual(args[args.index('--reuse-session')+1],str(path))
                    self.assertEqual(args[args.index('--reuse-digest')+1],plan['digest'])
                window.clear_reuse();self.assertFalse(window.instruction.isReadOnly())
                self.assertTrue(window.planning_thinking.isEnabled())
            finally:window.close();app.processEvents()

    def test_planning_thinking_does_not_extend_to_execution(self):
        from decision.agent.workflows.common import make_loop
        args=SimpleNamespace(actions=None,api_config='unused',cache_mode='implicit',thinking_mode='planning',
                             planning_thinking_mode='medium',max_requests=32)
        with patch('decision.agent.core.model.api_work',return_value=None) as api:
            loop=make_loop(args,SimpleNamespace(),Path('/unused'))
            for mode in ('off','low','medium'):
                args.planning_thinking_mode=mode
                loop.work(dict(current_state=dict(requirements_registered=False)),None,None)
                self.assertEqual(api.call_args.kwargs['thinking_mode'],mode)
                loop.work(dict(current_state=dict(requirements_registered=True)),None,None)
                self.assertEqual(api.call_args.kwargs['thinking_mode'],'off')
            args.thinking_mode='low'
            loop.work(dict(current_state=dict(requirements_registered=True)),None,None)
            self.assertEqual(api.call_args.kwargs['thinking_mode'],'low')

    def test_late_elevator_message_does_not_leak_into_a_new_ride(self):
        from decision.agent.workflows.common import make_loop
        args=SimpleNamespace(actions=None,api_config='unused',cache_mode='implicit',thinking_mode='off',max_requests=32)
        old=SimpleNamespace(status='monitoring_ride')
        session=SimpleNamespace(elevator=SimpleNamespace(transition=old))
        received=[]
        with patch('decision.agent.core.model.api_work',return_value=None) as api:
            loop=make_loop(args,session,Path('/unused'),transient_sink=received.append)
            loop.work(dict(current_state=dict(requirements_registered=True)),None,None)
            sink=api.call_args.kwargs['transient_sink']
            sink(dict(kind='current'))
            old.status='awaiting_destination_localization'
            sink(dict(kind='previous_phase'))
            session.elevator.transition=SimpleNamespace(status='monitoring_ride')
            sink(dict(kind='late'))
        self.assertEqual(received,[dict(kind='current',phase='monitoring_ride')])

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

    def test_skip_command_is_bound_to_the_current_goal_and_manual_scope(self):
        used=set()
        self.assertFalse(dispatch_structured(self.command(name='skip',goal_id='old'),self.actor,used)['accepted'])
        self.assertTrue(dispatch_structured(self.command(command_id='skip',name='skip'),self.actor,used)['accepted'])
        self.assertEqual(self.calls,[('skip',None)])
        self.session.navigation.scope='live'
        self.assertFalse(dispatch_structured(self.command(command_id='live',name='skip'),self.actor,used)['accepted'])

    def test_skip_button_requires_confirmation_and_submits_current_identity(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            window.status=dict(task_id='task',revision=2,goal_id='route',goal_description='绕圈',call_id='call',status='waiting_navigation')
            window.status_received=time.monotonic()
            with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.No), patch.object(window.process,'write') as write:
                window.send_command('skip');write.assert_not_called()
            with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes), patch.object(window.process,'write',return_value=1) as write:
                window.send_command('skip')
                value=json.loads(write.call_args.args[0])
                self.assertEqual((value['name'],value['task_id'],value['revision'],value['goal_id'],value['call_id']),
                                 ('skip','task',2,'route','call'))
        finally:
            window.close();app.processEvents()

    def test_answer_is_single_structured_event(self):
        result=dispatch_structured(self.command(name='answer',text='第一行\n第二行'),self.actor,set())
        self.assertTrue(result['accepted']);self.assertEqual(self.calls,[('answer','第一行\n第二行')])

    def test_completed_navigation_can_accept_identity_bound_arrival_reconfirmation(self):
        selected=dict(call_id='old-call',task_id='task')
        self.session.navigation.handle=selected
        self.session.skill.record.update(handle=None,completed=[dict(goal_id='first',handle=selected)])
        self.session.task_goals.goals[0]['selected']=dict(handle=selected)
        value=self.command(call_id='old-call')
        self.assertTrue(dispatch_structured(value,self.actor,set())['accepted'])
        self.assertEqual(self.calls,[('arrived',None)])
        self.session.navigation.handle=dict(call_id='different')
        self.assertFalse(dispatch_structured(self.command(command_id='other',call_id='old-call'),self.actor,set())['accepted'])

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
                self.assertIn('地图定位不可用',event_summary(diagnostic))
                self.assertEqual(event_detail(db,diagnostic['seq'])[1]['goal_id'],'first')
            finally:db.close();store.close()

    def test_repeated_route_refusal_is_visible_to_operator(self):
        event=dict(kind='route_repeated_refusal',value=dict(count=3,reason='proposed_route_intersects_task_region'))
        self.assertIn('3 次',event_summary(event))
        self.assertIn('路线穿过目标区域',event_summary(event))

    def test_actual_request_messages_keep_all_images_and_tool_arguments(self):
        image='data:image/png;base64,'+base64.b64encode(b'image').decode()
        messages=[dict(role='system',content='规则'),dict(role='user',content=[
            dict(type='text',text='查看场景'),dict(type='image_url',image_url=dict(url=image,detail='high')),
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
            store.event('api_request',value)
            db=connect_readonly(Path(root)/'session.sqlite')
            try:
                self.assertEqual(db.execute('select count(*) from request_images').fetchone()[0],1)
                _,events=poll_session(db,0)
                self.assertEqual([event['context'] for event in events],[context,context])
                _,detail,images=event_detail(db,events[0]['seq'])
                self.assertEqual(len(images),2)
                self.assertEqual(detail['request']['messages'][2]['tool_calls'][0]['function']['arguments'],
                                 '{"name":"门"}')
            finally:db.close();store.close()

    def test_detail_separates_message_images_from_entire_request(self):
        def image(value):return 'data:image/png;base64,'+base64.b64encode(value).decode()
        messages=[dict(role='system',content='规则'),
                  dict(role='user',content=[dict(type='image_url',image_url=dict(url=image(b'map')))]),
                  dict(role='user',content=[dict(type='image_url',image_url=dict(url=image(b'rgb'))),
                                             dict(type='image_url',image_url=dict(url=image(b'local')))])]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.sqlite';store=Store(path)
            store.event('api_request',dict(request=dict(messages=messages)))
            db=connect_readonly(path)
            try:
                kind,value,images=event_detail(db,1)
                _,all_images,_=select_detail(kind,value,images)
                self.assertEqual([x[1] for x in all_images],[b'map',b'rgb',b'local'])
                _,system_images,_=select_detail(kind,value,images,0)
                self.assertEqual(system_images,[])
                _,map_images,_=select_detail(kind,value,images,1)
                self.assertEqual([x[1] for x in map_images],[b'map'])
                _,current_images,chosen=select_detail(kind,value,images,2,2)
                self.assertEqual([x[1] for x in current_images],[b'rgb',b'local'])
                self.assertEqual(chosen,1)
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
            self.assertEqual(window.context_tree.topLevelItemCount(),4)
            self.assertEqual(window.context_tree.topLevelItem(1).foreground(0).color().name(),'#7755aa')
            self.assertEqual(window.context_tree.topLevelItem(2).foreground(0).color().name(),'#9aa3b0')
            self.assertIn('历史',window.context_tree.topLevelItem(2).text(1))
            self.assertEqual(window.current_tree.topLevelItemCount(),1)
            self.assertEqual(window.current_tree.topLevelItem(0).text(2),'规则')
            self.assertIn('1 条消息',window.request_summary.text())
        finally:
            window.close()
            app.processEvents()

    def test_floor_preset_and_live_task_monitor(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtCore,QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow(floor='floor2')
        try:
            self.assertTrue(window.catalog.text().endswith('floor2_catalog.json'))
            self.assertIn('floor2_current',window.catalog.text())
            if not Path(window.catalog.text()).is_file():
                self.assertIn('尚无对应语义标注',window.notice.text())
            window.socket.setText('/old/socket')
            window.floor.setCurrentIndex(window.floor.findData('floor1'))
            self.assertTrue(window.catalog.text().endswith('floor1_catalog.json'))
            self.assertEqual(window.socket.text(),'')
            window.status=dict(status='waiting_navigation',navigation_status='running',localization_status='valid',
                input_stale=False,origin_pose=[0.,0.,0.],floor_id='floor1',goal_description='到停靠点',
                goal_distance_m=.1,goal_tolerance_m=.2,arrival_dwell_s=.3,arrival_dwell_required_s=.5,
                goal_yaw_error_rad=.1,goal_yaw_tolerance_rad=.2,stopped=True,stage_type='follow',
                follow_status=dict(status='running'),model_busy=True,
                monitor=dict(focus='关注入口',conditional_review=dict(region_id='region',region_state='inside',phase='reviewing')))
            window.annotation_names={'region':'中转等候区'}
            window.status_received=time.monotonic()
            with patch.object(window.process,'state',return_value=QtCore.QProcess.Running):window.refresh_status()
            value=window.task_monitor.toPlainText()
            for expected in ('floor1','到停靠点','0.10 m','连续停稳','朝向偏差','关注入口','中转等候区','区域内','模型判断中','running','完成时区域检查：未登记'):
                self.assertIn(expected,value)
        finally:
            window.close();app.processEvents()

    def test_monitor_scroll_and_selection_survive_frequent_updates(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtCore,QtGui,QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow();window.show();window.settings_toggle.setChecked(False)
        try:
            window.status=dict(status='waiting_navigation',navigation_status='running',localization_status='valid',
                input_stale=False,origin_pose=[0.,0.,0.],floor_id='floor2',goal_id='goal',goal_description='前往目标',
                goal_distance_m=.2,goal_tolerance_m=.2,stopped=True,
                monitor=dict(focus='核对任务',last_review=dict(result='unknown',evidence='观察依据很长。'*250)))
            window.status_received=time.monotonic()
            with patch.object(window.process,'state',return_value=QtCore.QProcess.Running):
                window.refresh_status();app.processEvents()
                bar=window.task_monitor.verticalScrollBar()
                self.assertGreater(bar.maximum(),10)
                cursor=window.task_monitor.textCursor()
                cursor.setPosition(5);cursor.setPosition(12,QtGui.QTextCursor.KeepAnchor)
                window.task_monitor.setTextCursor(cursor)
                bar.setValue(bar.maximum()//2)
                previous=bar.value();height=window.task_monitor.height()
                self.assertGreater(previous,0)
                window.status['goal_distance_m']=.19
                window.status['monitor']['last_review']['evidence']+='额外的观察结果。'
                window.refresh_status();app.processEvents()
                self.assertEqual(bar.value(),previous)
                self.assertEqual((window.task_monitor.textCursor().anchor(),window.task_monitor.textCursor().position()),(5,12))
                self.assertEqual(window.task_monitor.height(),height)
                self.assertIn('0.19 m',window.task_monitor.toPlainText())
            self.assertEqual(window.left_sections.count(),3)
            self.assertGreater(window.task_monitor.maximumHeight(),window.task_monitor.minimumHeight())
        finally:
            window.close();app.processEvents()

    def test_registered_goal_requirements_and_agent_choice_are_visible(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            goals=[dict(status='pending',requirement=dict(goal_id='g1',kind='circuit',target_id='area',description='绕圆形区域一圈'),
                        selected=dict(route=dict(direction='clockwise',direction_source='autonomous'))),
                   dict(status='pending',requirement=dict(goal_id='g2',kind='region',target_id='area',
                        description='进入区域',semantic_conditions=['确认能够看到电梯']))]
            state=dict(instruction='绕圈后进入区域',task_goals=goals,navigation=dict(goal_id='g1',status='running'))
            event=dict(seq=1,wall_s=0.,kind='reference_map_prepared',value=dict(annotations=dict(
                regions=[dict(region_id='area',name='圆形区域')],points=[])))
            window.update_session(state,[event])
            text=window.goal_details.toPlainText()
            self.assertIn('要求：绕圆形区域一圈',text)
            self.assertIn('目标：圆形区域',text)
            self.assertIn('用户要求方向：未指定',text)
            self.assertIn('执行方案方向：顺时针（Agent 选择）',text)
            self.assertEqual(window.goals.topLevelItem(0).text(1),'进行中')
            window.goals.setCurrentItem(window.goals.topLevelItem(1))
            self.assertIn('确认能够看到电梯',window.goal_details.toPlainText())
        finally:
            window.close();app.processEvents()

    def test_current_request_keeps_actual_order_after_system_prompt_changes(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            first=[dict(role='system',content='登记规则'),dict(role='user',content='任务')]
            second=[dict(role='system',content='导航规则'),dict(role='user',content='任务')]
            events=[dict(seq=i,kind='api_request',wall_s=float(i),value={},
                         context=request_context(dict(request=dict(messages=messages))))
                    for i,messages in enumerate((first,second),1)]
            window.update_session(None,events)
            self.assertEqual([window.current_tree.topLevelItem(i).text(2) for i in range(2)],['导航规则','任务'])
            self.assertIn('历史',window.context_tree.topLevelItem(1).text(1))
            self.assertEqual(window.latest_request_seq,2)
        finally:
            window.close();app.processEvents()

    def test_status_height_is_stable_and_target_preview_keeps_original_image(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtGui,QtWidgets
        from decision.agent.manual_ui import ImageDialog,ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            height=window.status_label.height()
            window.status_label.setPlainText('人工驾驶 '+('很长的状态信息 '*200))
            app.processEvents()
            self.assertEqual(window.status_label.height(),height)
            self.assertTrue(window.status_label.isReadOnly())
            with tempfile.TemporaryDirectory() as root:
                path=Path(root)/'target.png';image=QtGui.QPixmap(1400,900)
                image.fill(QtGui.QColor('blue'));self.assertTrue(image.save(str(path)))
                window.session_path=Path(root)/'session.sqlite'
                window.show_target(dict(goal_id='g',target_map_xy_yaw=[1,2,0],position_tolerance_m=.3,
                                        target_image=str(path)))
                self.assertTrue(window.target_zoom.isEnabled())
                self.assertEqual(window.target_pixmap.size(),image.size())
                dialog=ImageDialog(window.target_pixmap,'目标示意图',window)
                try:
                    self.assertEqual(dialog.view.image_item.pixmap().size(),image.size())
                    before=dialog.view.transform().m11();dialog.view.zoom(1.25)
                    self.assertGreater(dialog.view.transform().m11(),before)
                    dialog.view.resetTransform();self.assertEqual(dialog.view.transform().m11(),1.)
                finally:dialog.close()
                window.show_target(dict(goal_id='new',target_map_xy_yaw=[1,2,0],position_tolerance_m=.3))
                self.assertFalse(window.target_zoom.isEnabled());self.assertIsNone(window.target_pixmap)
        finally:window.close();app.processEvents()

    def test_settings_can_collapse_and_history_event_panes_resize(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            window.show();app.processEvents()
            self.assertTrue(window.settings_panel.isVisible())
            before=window.context_tabs.height()
            window.settings_toggle.setChecked(False);app.processEvents()
            self.assertFalse(window.settings_panel.isVisible())
            self.assertGreater(window.context_tabs.height(),before)
            self.assertEqual(window.right_sections.count(),2)
            window.right_sections.setSizes([250,350]);app.processEvents()
            self.assertGreater(window.operation_events.height(),180)
            window.settings_toggle.setChecked(True);app.processEvents()
            self.assertTrue(window.settings_panel.isVisible())
        finally:window.close();app.processEvents()

    def test_detail_image_zoom_uses_selected_original(self):
        import os
        from unittest.mock import patch
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtCore,QtGui,QtWidgets
        from decision.agent.manual_ui import DetailDialog
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        image=QtGui.QImage(1200,800,QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('red'))
        data=QtCore.QByteArray();buffer=QtCore.QBuffer(data);buffer.open(QtCore.QIODevice.WriteOnly)
        self.assertTrue(image.save(buffer,'PNG'));buffer.close()
        dialog=DetailDialog('api_request',{},[('地图',bytes(data))],message_only=True)
        try:
            button=next(x for x in dialog.findChildren(QtWidgets.QPushButton) if x.text()=='放大查看选中图片')
            self.assertTrue(button.isEnabled())
            with patch('decision.agent.manual_ui.ImageDialog') as viewer:
                button.click()
                self.assertEqual(viewer.call_args.args[0].size(),image.size())
                viewer.return_value.exec_.assert_called_once()
        finally:dialog.close();app.processEvents()

    def test_detail_notice_uses_one_line_and_gui_hides_internal_identifiers(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtWidgets
        from decision.agent.manual_ui import DetailDialog
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        identity='a'*64
        data=dict(task_id=identity,image_sha256=identity,view_id='view-123456abcdef',
                  content=json.dumps(dict(call_id=identity,pose=[1,2,3])))
        visible=display_value(data)
        self.assertNotIn(identity,json.dumps(visible));self.assertEqual(visible['view_id'],'view-123456abcdef')
        self.assertEqual(data['image_sha256'],identity)
        dialog=DetailDialog('api_request',data,[],message_only=True)
        try:
            dialog.show();app.processEvents()
            label=dialog.findChildren(QtWidgets.QLabel)[0]
            self.assertLess(label.height(),80)
            self.assertGreater(dialog.findChild(QtWidgets.QSplitter).height(),500)
            self.assertNotIn(identity,dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText())
        finally:dialog.close();app.processEvents()

    def test_window_shows_arrival_diagnostics_separately_from_model_context(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5 import QtCore,QtWidgets
        from decision.agent.manual_ui import ManualWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window=ManualWindow()
        try:
            event=dict(seq=42,kind='manual_goal_evidence_unproven',wall_s=0.,
                       value=dict(goal_id='first',issues=['mapped_pose_unavailable']))
            window.update_session(None,[event])
            self.assertEqual(window.context_tree.topLevelItemCount(),0)
            self.assertEqual(window.operation_events.topLevelItemCount(),1)
            item=window.operation_events.topLevelItem(0)
            self.assertIn('地图定位不可用',item.text(2))
            self.assertEqual(item.data(0,QtCore.Qt.UserRole),42)
        finally:
            window.close();app.processEvents()


if __name__=='__main__':unittest.main()
