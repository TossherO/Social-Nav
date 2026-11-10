"""Qt editor interaction checks; run with a display or QT_QPA_PLATFORM=offscreen."""
import os
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backend_bases import MouseEvent

from tests import test_manual_map_edit as fixtures
from perception.environment.storage import StaticEnvironmentMap


@unittest.skipUnless(os.environ.get('DISPLAY') or os.environ.get('QT_QPA_PLATFORM') == 'offscreen',
                     'Qt requires a display or QT_QPA_PLATFORM=offscreen')
class MapEditorInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest('PyQt5 is unavailable') from error
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from perception.visualization.map_editor import make_figure
        from PyQt5 import QtWidgets
        import matplotlib.pyplot as plt
        self.fixture = fixtures.ManualMapEditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.static_map()
        self.output = self.fixture.root/'ui_output'
        self.fig = make_figure(self.source, self.output)
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.axes[0]
        self.bar = self.fig.canvas.manager.window.findChild(QtWidgets.QToolBar, 'manual_map_editor')
        self.QtWidgets = QtWidgets
        self.fig.canvas.draw()

    def event(self, name, xy, *, button=None, step=0):
        pixel = self.ax.transData.transform(xy)
        event = MouseEvent(name, self.fig.canvas, *pixel, button=button, step=step)
        self.fig.canvas.callbacks.process(name, event)
        return event

    def test_pointer_zoom_pan_and_continuous_brush(self):
        canvas, ax = self.fig.canvas, self.ax
        self.event('motion_notify_event', (.25, .25))
        self.assertTrue(all(circle.get_visible() for circle in ax.patches))
        radius = self.bar.findChild(self.QtWidgets.QDoubleSpinBox, 'brush_radius')
        radius.setValue(.1)
        self.assertTrue(all(circle.get_radius() == .1 for circle in ax.patches))
        before = np.asarray(ax.images[0].get_array()).copy()
        left, right = ax.get_xlim()
        fraction = (.25-left)/(right-left)
        self.event('scroll_event', (.25, .25), button='up', step=1)
        xlim = ax.get_xlim()
        self.assertAlmostEqual(xlim[1]-xlim[0], (right-left)/1.2)
        self.assertAlmostEqual((.25-xlim[0])/(xlim[1]-xlim[0]), fraction, places=2)
        canvas.draw()
        self.event('button_press_event', (.25, .25), button=3)
        self.assertFalse(any(circle.get_visible() for circle in ax.patches))
        self.event('motion_notify_event', (.45, .25), button=3)
        self.event('button_release_event', (.45, .25), button=3)
        self.assertLess(ax.get_xlim()[0], xlim[0])
        np.testing.assert_array_equal(ax.images[0].get_array(), before)
        self.bar.findChild(self.QtWidgets.QAction, 'reset_view').trigger()
        canvas.draw()
        state = self.bar.findChild(self.QtWidgets.QComboBox, 'edit_state')
        state.setCurrentIndex(state.findData(-1))
        self.event('button_press_event', (.1, .1), button=1)
        self.event('motion_notify_event', (.8, .1), button=1)
        self.event('button_release_event', (.8, .1), button=1)
        with patch('PyQt5.QtWidgets.QMessageBox.information'), patch('PyQt5.QtWidgets.QMessageBox.warning') as warning:
            self.bar.findChild(self.QtWidgets.QAction, 'edit_save').trigger()
            self.assertFalse(warning.called, warning.call_args)
        snapshot = StaticEnvironmentMap(self.output).snapshot
        origin = np.asarray(snapshot.metadata['origin_xy'])
        cell = np.floor((np.array([.45, .1])-origin)/snapshot.metadata['resolution']).astype(int)
        self.assertEqual(snapshot.layers['occupancy'][cell[1], cell[0]], -1)
        self.assertTrue(snapshot.layers['manual_edit'][cell[1], cell[0]])

    def test_canvas_fills_window_without_stretching_or_hiding_map(self):
        asset = StaticEnvironmentMap(self.source).snapshot
        origin = np.asarray(asset.metadata['origin_xy'])
        extent = origin + np.asarray(asset.metadata['shape'][::-1])*asset.metadata['resolution']
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.assertGreater(self.ax.get_position().width, .7)
        self.assertGreater(self.ax.get_position().height, .7)
        self.assertLessEqual(xlim[0], origin[0]); self.assertGreaterEqual(xlim[1], extent[0])
        self.assertLessEqual(ylim[0], origin[1]); self.assertGreaterEqual(ylim[1], extent[1])
        self.assertAlmostEqual(self.ax.bbox.width/(xlim[1]-xlim[0]),
                               self.ax.bbox.height/(ylim[1]-ylim[0]))

    def test_cursor_hidden_in_polygon_mode_and_view_reset(self):
        from matplotlib.backend_bases import KeyEvent
        self.event('motion_notify_event', (.2, .2))
        mode = self.bar.findChild(self.QtWidgets.QComboBox, 'edit_mode')
        mode.setCurrentIndex(mode.findData('polygon'))
        self.assertFalse(any(circle.get_visible() for circle in self.ax.patches))
        original = self.ax.get_xlim()
        self.event('scroll_event', (.2, .2), button='up', step=1)
        self.bar.findChild(self.QtWidgets.QAction, 'reset_view').trigger()
        np.testing.assert_allclose(self.ax.get_xlim(), original)
        self.fig.canvas.draw()
        for xy in [(.01, .01), (.4, .01), (.4, .4), (.01, .4)]:
            self.event('button_press_event', xy, button=1)
        self.fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', self.fig.canvas, key='enter'))
        with patch('PyQt5.QtWidgets.QMessageBox.information'), patch('PyQt5.QtWidgets.QMessageBox.warning') as warning:
            self.bar.findChild(self.QtWidgets.QAction, 'edit_save').trigger()
            self.assertFalse(warning.called, warning.call_args)
        self.assertTrue(StaticEnvironmentMap(self.output).snapshot.layers['manual_edit'].any())

    def test_localization_selection_after_zoom_and_pan(self):
        from matplotlib.backend_bases import KeyEvent
        from perception.mapping_localization.manual_edit import read_xyzi
        from perception.visualization.map_editor import make_figure
        import matplotlib.pyplot as plt
        source = self.fixture.roll_map()
        output = self.fixture.root/'roll_ui_output'
        plt.close(self.fig)
        self.fig = make_figure(source, output, localization=True)
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.axes[0]
        self.ax.set_xlim(-1, 1); self.ax.set_ylim(0, 2)
        self.fig.canvas.draw()
        bar = self.fig.canvas.manager.window.findChild(self.QtWidgets.QToolBar, 'manual_map_editor')
        self.event('scroll_event', (0, 1), button='up', step=1)
        self.fig.canvas.draw()
        self.event('button_press_event', (0, 1), button=2)
        self.event('motion_notify_event', (.1, 1), button=2)
        self.event('button_release_event', (.1, 1), button=2)
        self.fig.canvas.draw()
        bar.findChild(self.QtWidgets.QDoubleSpinBox, 'z_min').setValue(.4)
        bar.findChild(self.QtWidgets.QDoubleSpinBox, 'z_max').setValue(.6)
        for xy in [(-.2, .8), (.2, .8), (.2, 1.2), (-.2, 1.2)]:
            self.event('button_press_event', xy, button=1)
        self.fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', self.fig.canvas, key='enter'))
        with patch('PyQt5.QtWidgets.QMessageBox.information'), patch('PyQt5.QtWidgets.QMessageBox.warning') as warning:
            bar.findChild(self.QtWidgets.QAction, 'edit_save').trigger()
            self.assertFalse(warning.called, warning.call_args)
        np.testing.assert_array_equal(read_xyzi(output/'keyframes/surf1.pcd'), [[2, 0, .5, 4]])


if __name__ == '__main__':
    unittest.main()
