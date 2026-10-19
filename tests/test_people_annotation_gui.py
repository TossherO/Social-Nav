"""Run with the Qt interpreter; synthetic data only, no ROS/model/GPU required."""
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5 import QtCore, QtWidgets, QtTest
    from perception.detection_tracking.annotation.app import Window
except ImportError:
    Window = None

from perception.detection_tracking.annotation.backend import LocalBackend
from perception.detection_tracking.annotation.data import Store, blank_review, dumps, edit_review


BOUNDS = [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5]
BOXES = [[3., 1., .6, .6, .6, 1.7, 0., .85]]


class SlowBackend(LocalBackend):
    """Real SQLite with bounded synthetic asset I/O and controllable write failure."""
    def __init__(self, project):
        super().__init__(project)
        self.delay, self.fail, self.calls = .06, False, []

    def audit(self, operation):
        self.calls.append((operation, threading.get_ident()))

    def prepare(self, progress, detections=None):
        self.audit('prepare')
        time.sleep(self.delay)
        return super().prepare(progress, detections)

    def read_frame(self, ident):
        self.audit('read')
        time.sleep(.02)
        store = Store(self.project)
        try:
            frame = store.frame(ident)
            points = np.array([[0., 0., .2, 1.], [3., 1., .5, 1.]], dtype=np.float32)
            return frame, points, np.zeros((12, 16, 3), dtype=np.uint8), frame['stamp_ns'], ''
        finally:
            store.close()

    def save_review(self, *args):
        self.audit('review')
        time.sleep(self.delay)
        if self.fail:
            raise OSError('test write failure')
        return super().save_review(*args)

    def set_meta(self, *args):
        self.audit('preference')
        return super().set_meta(*args)

    def save_position(self, *args):
        self.audit('position')
        return super().save_position(*args)

    def save_group(self, *args):
        self.audit('group')
        return super().save_group(*args)

    def reset_source_reviews(self, *args):
        self.audit('reset')
        time.sleep(self.delay)
        return super().reset_source_reviews(*args)

    def close(self):
        self.audit('close')
        return super().close()


@unittest.skipIf(Window is None, 'requires the Qt interpreter')
class PeopleAnnotationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def wait_for(self, condition, timeout=5.):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            QtTest.QTest.qWait(5)
        self.assertTrue(condition(), 'Qt operation did not finish')

    def idle(self):
        return self.window.write_task is None and not self.window.writes and not self.window.loading

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        project = Path(self.temp.name) / 'project'
        store = Store(project, create=True, settings={}, effective={'DATA_CONFIG': {'POINT_CLOUD_RANGE': BOUNDS}})
        try:
            for source in ('a', 'b'):
                with store.connection:
                    store.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)',
                        (source, source + '.bag', '{}', '/points', '', 'capture', source, 'train', 1))
                    for ordinal in range(3):
                        store.connection.execute('INSERT INTO frames '
                            '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,fingerprint,review) '
                            'VALUES (?,?,?,?,?,?,?,?,?)', (source + '/' + str(ordinal), source, ordinal,
                            1000000000 + ordinal * 100000000, 2000000000 + ordinal * 100000000, 0,
                            'rslidar', 'bound', dumps(blank_review())))
                for ordinal in range(3):
                    store.save_candidates(source + '/' + str(ordinal), BOXES, {'model': 'test'}, 'bound')
        finally:
            store.close()
        self.backend = SlowBackend(project)
        self.warnings = []
        warning = patch.object(QtWidgets.QMessageBox, 'warning', side_effect=lambda *args: self.warnings.append(args[-1]))
        warning.start()
        self.addCleanup(warning.stop)
        self.window = Window(SimpleNamespace(project=project, detections=None, sample_interval=None,
            start=0., duration=None, sequence=None, cloud_renderer='software'), backend=self.backend)
        self.addCleanup(self.close_window)
        self.window.show()
        self.wait_for(lambda: self.window.frame is not None and self.idle())

    def close_window(self):
        if not self.window.backend_closed:
            self.backend.fail = False
            if self.window.save_error:
                self.window.retry_save()
                self.wait_for(self.idle)
            self.window.close()
            self.wait_for(lambda: self.window.backend_closed)

    def stored(self, ident):
        store = Store(self.backend.project)
        try:
            return store.frame(ident)
        finally:
            store.close()

    def test_async_edits_navigation_and_close_preserve_order(self):
        window = self.window
        self.backend.delay = .18
        ticks = []
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        timer.start(10)
        try:
            window.select(('candidate', '0'))
            QtTest.QTest.keyClick(window.canvas, QtCore.Qt.Key_Q)
            self.assertEqual(len(window.review['centers']), 1)
            window.note.setText('note before navigation')
            window.save_note()
            window.undo()
            self.assertEqual(window.note.text(), '')
            window.redo()
            window.navigate(1, False)
            self.assertEqual(window.frame['id'], 'a/0')
            self.assertIn('保存中', window.save_notice.text())
            self.wait_for(lambda: window.frame['id'] == 'a/1' and self.idle())
            saved = self.stored('a/0')['review']
            self.assertEqual(saved['note'], 'note before navigation')
            self.assertEqual(len(saved['centers']), 1)
            self.assertGreater(len(ticks), 5, 'GUI event loop blocked during storage')
            window.mark_status('complete')
            window.close()
            self.wait_for(lambda: window.backend_closed)
            self.assertEqual(self.stored('a/1')['review']['status'], 'complete')
        finally:
            timer.stop()
        self.assertTrue(self.backend.calls)
        self.assertTrue(all(ident != threading.get_ident() for _, ident in self.backend.calls))
        self.assertFalse(self.warnings)

    def test_save_failure_retains_draft_retry_and_conflict_requires_explicit_reload(self):
        window = self.window
        self.backend.fail = True
        window.change('add', xy=[0., 0.])
        window.change('add', xy=[.4, .4])
        window.navigate(1, False)
        self.wait_for(lambda: window.save_error is not None and self.idle())
        self.assertEqual(window.frame['id'], 'a/0')
        self.assertEqual(len(window.review['centers']), 2)
        self.assertEqual(self.stored('a/0')['review']['centers'], [])
        self.backend.fail = False
        window.retry_save()
        self.wait_for(self.idle)
        self.assertIsNone(window.save_error)
        self.assertEqual(len(self.stored('a/0')['review']['centers']), 2)
        store = Store(self.backend.project)
        try:
            other = edit_review(store.frame('a/0')['review'], 'note', BOUNDS, note='other writer')
            store.save_review('a/0', other)
        finally:
            store.close()
        window.change('add', xy=[.8, .8])
        self.wait_for(lambda: window.save_error is not None and self.idle())
        self.assertEqual(len(window.review['centers']), 3)
        self.assertEqual(self.stored('a/0')['review']['note'], 'other writer')
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            window.reload_unsaved()
        self.wait_for(self.idle)
        self.assertIsNone(window.save_error)
        self.assertEqual(len(window.review['centers']), 2)
        self.assertEqual(window.note.text(), 'other writer')

    def test_sequence_reset_and_preferences_stay_on_backend(self):
        window = self.window
        window.change('add', xy=[0., 0.])
        window.activate_sequence('b')
        self.wait_for(lambda: window.source_id == 'b' and window.frame is not None and self.idle())
        window.change('add', xy=[.1, .1])
        window.interval.setValue(.5)
        window.point_scale.setValue(2.)
        window.root_group.setText('new capture')
        window.save_group()
        self.wait_for(self.idle)
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            window.clear_sequence_marks()
        self.wait_for(self.idle)
        self.assertEqual(self.stored('a/0')['review']['status'], 'draft')
        self.assertEqual(self.stored('b/0')['review']['status'], 'unreviewed')
        self.assertEqual(self.stored('b/0')['candidates'], BOXES)
        store = Store(self.backend.project)
        try:
            self.assertEqual(store.meta('point_scale'), 2.)
            self.assertEqual(store.meta('sample_intervals')['b'], .5)
            self.assertEqual(store.source('b')['root_group'], 'new capture')
        finally:
            store.close()
        self.assertFalse(self.warnings)


if __name__ == '__main__':
    unittest.main()
