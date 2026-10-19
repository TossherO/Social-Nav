"""Qt integration: asynchronous saves, scope changes and keyboard behavior."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import time
import unittest

from PyQt5 import QtCore, QtTest, QtWidgets

from decision.vlm.annotation.app import Window
from decision.vlm.annotation.backend import LocalBackend
from decision.vlm.annotation.store import Store, new_event
from tests.test_vlm_annotation import fixture


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.settings = tempfile.TemporaryDirectory()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, cls.settings.name)

    @classmethod
    def tearDownClass(cls):
        cls.settings.cleanup()

    def wait(self, predicate, seconds=10):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.app.processEvents()
            if predicate():
                return
            QtTest.QTest.qWait(10)
        self.fail('Qt操作超时：'+self.window.message.text())

    def idle(self):
        self.wait(lambda: self.window.active_job is None and not self.window.jobs)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/'project'
        fixture(self.path)
        store = Store(self.path)
        store.writer('seed', acquire=True)
        event = new_event(store.sources()[0], 0, 3, '测试事件')
        event['split'] = 'development'
        store.commit(dict(project_id=store.identity, writer_id='seed', operation_id='seed', expected_revision=0, event=event))
        store.writer('seed', release=True)
        self.window = Window(lambda: LocalBackend(self.path))
        self.window.show()
        self.wait(lambda: self.window.view_data is not None)
        self.window.goto(20)
        self.wait(lambda: self.window.view_data is not None and self.window.view_data['frame']['id']=='source/20')
        self.idle()

    def tearDown(self):
        self.idle()
        self.window.close()
        self.wait(lambda: self.window.final_close)
        self.app.processEvents()
        self.temporary.cleanup()

    def test_group_review_and_rapid_edits_keep_latest_state(self):
        w = self.window
        w.pick('track:A', False)
        w.pick('track:B', True)
        w.apply_group()
        w.modify_label(lambda label: label.update(note='first'))
        w.modify_label(lambda label: label.update(note='latest'))
        self.idle()
        self.assertFalse(w.paused, w.message.text())
        self.assertEqual(w.label()['note'], 'latest')
        w.mode.setCurrentIndex(1)
        self.wait(lambda: w.view_data is not None)
        w.mark('complete')
        self.idle()
        result = Store(self.path).events()[0]['document']['t1']['source/20']
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['note'], 'latest')
        self.assertEqual(result['groups'][0]['relation'], 'w')

    def test_browse_does_not_create_samples_and_text_keeps_wasd(self):
        w = self.window
        w.people.setFocus()
        QtTest.QTest.keyClick(w.people, QtCore.Qt.Key_D)
        self.wait(lambda: w.view_data is not None and w.view_data['frame']['id']=='source/21')
        self.assertEqual(w.event['t1'], {})
        w.reviewer.setFocus()
        w.reviewer.setText('')
        QtTest.QTest.keyClicks(w.reviewer, 'wasd')
        self.assertEqual(w.reviewer.text(), 'wasd')
        self.assertEqual(w.frame_index, 21)
        w.people.setFocus()
        QtTest.QTest.keyClick(w.people, QtCore.Qt.Key_2)
        self.idle()
        self.assertEqual(w.event['t1']['source/21']['status'], 'skipped')
        self.assertEqual(w.frame_index, 21)

    def test_t2_evidence_page_saves_independently_of_t1(self):
        w = self.window
        def seed(event):
            event['persons']['target-person'] = dict(name='目标A')
            event['links'].append(dict(key='track:A', person_id='target-person', start_ns=1700000000000000000,
                                       end_ns=1700000003000000000))
            event['targets']['task'] = dict(id='task', description='灰色衣服的人', person_id='target-person', absent=False,
                                            reference_frame='source/10', samples={})
        w.mutate(seed)
        self.idle()
        w.refresh_targets()
        w.tabs.setCurrentIndex(1)
        w.pick('track:A', False)
        w.set_true_key()
        w.set_candidates()
        for field, value in (('visibility', 'clear'), ('coverage', 'represented'), ('match', 'unique')):
            widget = w.t2_controls[field]
            widget.setCurrentIndex(widget.findData(value))
        self.idle()
        self.assertFalse(w.paused, w.message.text())
        w.mode.setCurrentIndex(1)
        self.wait(lambda:w.view_data is not None)
        w.mark('complete')
        self.idle()
        event = Store(self.path).events()[0]['document']
        self.assertEqual(event['t1'], {})
        self.assertEqual(event['targets']['task']['samples']['source/20']['status'], 'complete')

    def test_interval_sampling_does_not_build_every_model_window(self):
        from unittest.mock import patch
        w = self.window
        with patch.object(w.backend,'window',side_effect=AssertionError('bulk sampling should stay lightweight')):
            w.batch_samples()
            self.idle()
        self.assertFalse(w.paused,w.message.text())
        self.assertEqual(set(w.event['t1']),{'source/0','source/20'})
        self.assertTrue(all(label['binding'] is None for label in w.event['t1'].values()))


if __name__ == '__main__':
    unittest.main()
