"""Offline checks for manual GUI data and identity-bound commands."""
import base64
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from decision.agent.core.store import Store
from decision.agent.ui_data import connect_readonly,event_detail,event_summary,poll_session
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


if __name__=='__main__':unittest.main()
