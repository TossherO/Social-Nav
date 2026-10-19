"""Buffered OpenGL point view with an explicit Qt software fallback; no model/ROS imports."""
import ctypes
import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .appearance import BACKGROUND, HEIGHT_RGB


SOFTWARE_PALETTE = np.stack([np.interp(np.linspace(0, 1, 24), np.linspace(0, 1, len(HEIGHT_RGB)), channel)
                             for channel in HEIGHT_RGB.T], axis=-1)


def view_matrices(width, height, focus, yaw, elevation, distance):
    az, el = np.radians([yaw, elevation])
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    eye = np.asarray(focus) + distance * direction
    forward = -direction
    up = np.array([0., 1., 0.]) if abs(elevation) > 89 else np.array([0., 0., 1.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4)
    view[:3, :3] = np.array([right, up, -forward])
    view[:3, 3] = -view[:3, :3] @ eye
    near, far = .03, max(500., distance * 10)
    f = 1 / math.tan(math.radians(45) / 2)
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1] = f / (max(width, 1) / max(height, 1)), f
    projection[2, 2], projection[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    projection[3, 2] = -1
    return projection.astype(np.float32), view.astype(np.float32)


class Camera:
    def setup(self):
        self.focus = np.array([4.8, 0., .5])
        self.yaw, self.elevation, self.distance = -60., 35., 32.
        self.points = np.empty((0, 6), dtype=np.float32)
        self.lines = np.empty((0, 6), dtype=np.float32)
        self.reference_vertices = 0
        self.point_scale = 1.
        self.dirty, self.key, self.drag = True, None, None
        self.setMinimumSize(240, 220)

    def scene(self, points, colors, lines, line_colors, focus, distance, key, reference_vertices=0):
        self.points = np.ascontiguousarray(np.c_[points, colors], dtype=np.float32)
        self.lines = np.ascontiguousarray(np.c_[lines, line_colors], dtype=np.float32)
        self.reference_vertices = reference_vertices
        if key != self.key:
            self.focus, self.distance = np.asarray(focus, dtype=float), float(distance)
            self.key = key
        self.dirty = True
        self.update()

    def camera(self, elevation, yaw):
        self.elevation, self.yaw = float(elevation), float(yaw)
        self.update()

    def set_point_scale(self, scale):
        self.point_scale = float(scale)
        self.update()  # Reuse scene buffers; only the rasterized point size changes.

    def mousePressEvent(self, event):
        self.drag = event.pos()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseMoveEvent(self, event):
        if self.drag is None:
            return
        delta = event.pos() - self.drag
        self.drag = event.pos()
        if event.buttons() & QtCore.Qt.MiddleButton or event.modifiers() & QtCore.Qt.ShiftModifier:
            _, view = view_matrices(self.width(), self.height(), self.focus, self.yaw, self.elevation, self.distance)
            self.focus += (view[0, :3] * -delta.x() + view[1, :3] * delta.y()) * self.distance / max(1, self.height())
        elif event.buttons() & QtCore.Qt.RightButton:
            self.distance = np.clip(self.distance * math.exp(delta.y() * .01), .2, 300)
        else:
            self.yaw -= delta.x() * .5
            self.elevation = np.clip(self.elevation + delta.y() * .5, -89.9, 89.9)
        self.update()

    def wheelEvent(self, event):
        self.distance = np.clip(self.distance * 1.15 ** (-event.angleDelta().y() / 120), .2, 300)
        self.update()
        event.accept()


class SoftwareCloud(Camera, QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup()

    def project(self, points):
        projection, view = view_matrices(self.width(), self.height(), self.focus, self.yaw, self.elevation, self.distance)
        clip = np.c_[points, np.ones(len(points))] @ (projection @ view).T
        valid = clip[:, 3] > .03
        w = np.maximum(clip[:, 3], .0001)
        screen = clip[:, :2] / w[:, None]
        valid &= (np.abs(screen) < 1.1).all(axis=1)
        screen = (screen * [1, -1] + 1) * [self.width() / 2, self.height() / 2]
        return screen, valid

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(BACKGROUND))
        line_screen, line_valid = self.project(self.lines[:, :3])

        def draw_lines(start, end, width):
            for i in range(start, end, 2):
                if line_valid[i] and line_valid[i + 1]:
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(*map(float, self.lines[i, 3:6])), width))
                    painter.drawLine(QtCore.QPointF(*map(float, line_screen[i])), QtCore.QPointF(*map(float, line_screen[i + 1])))

        draw_lines(0, self.reference_vertices, 1.)
        data = self.points[::max(1, math.ceil(len(self.points) / 8000))]
        screen, valid = self.project(data[:, :3])
        # Color buckets keep fallback drawing batched instead of one draw call per point.
        if len(data):
            # Green alone cannot distinguish cyan from lavender on this height ramp.
            buckets = np.argmin(np.sum((data[:, None, 3:6] - SOFTWARE_PALETTE) ** 2, axis=2), axis=1)
            for bucket, color in enumerate(SOFTWARE_PALETTE):
                indices = np.flatnonzero(valid & (buckets == bucket))
                if len(indices):
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(*map(float, color)), 2.4 * self.point_scale))
                    painter.drawPoints(QtGui.QPolygonF([QtCore.QPointF(*map(float, p)) for p in screen[indices]]))
        draw_lines(self.reference_vertices, len(self.lines), 1.8)
        painter.end()


class GLCloud(Camera, QtWidgets.QOpenGLWidget):
    failed = QtCore.pyqtSignal(str)
    initialized = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        surface = QtGui.QSurfaceFormat()
        surface.setVersion(2, 1)
        surface.setProfile(QtGui.QSurfaceFormat.NoProfile)
        surface.setDepthBufferSize(24)
        self.setFormat(surface)
        self.setup()
        self.buffers, self.gl, self.renderer = [], None, ''

    def initializeGL(self):
        try:
            from OpenGL import GL
            self.gl = GL
            self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
            self.buffers = list(GL.glGenBuffers(2))
            GL.glClearColor(*QtGui.QColor(BACKGROUND).getRgbF())
            GL.glEnable(GL.GL_DEPTH_TEST)
            self.context().aboutToBeDestroyed.connect(self.release_buffers)
            self.initialized.emit(self.renderer)
        except Exception as error:
            self.failed.emit(str(error))

    def release_buffers(self):
        if self.buffers:
            self.makeCurrent()
            self.gl.glDeleteBuffers(len(self.buffers), self.buffers)
            self.buffers = []
            self.doneCurrent()

    def paintGL(self):
        if self.gl is None or not self.buffers:
            return
        gl = self.gl
        try:
            gl.glViewport(0, 0, round(self.width() * self.devicePixelRatioF()), round(self.height() * self.devicePixelRatioF()))
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            projection, view = view_matrices(self.width(), self.height(), self.focus, self.yaw, self.elevation, self.distance)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadMatrixf(np.ascontiguousarray(projection.T))
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadMatrixf(np.ascontiguousarray(view.T))
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            for buffer, data, mode in zip(self.buffers, (self.points, self.lines), (gl.GL_POINTS, gl.GL_LINES)):
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
                if self.dirty:
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
                gl.glVertexPointer(3, gl.GL_FLOAT, 24, ctypes.c_void_p(0))
                gl.glColorPointer(3, gl.GL_FLOAT, 24, ctypes.c_void_p(12))
                gl.glPointSize(2.5 * self.point_scale * self.devicePixelRatioF())
                gl.glLineWidth(1.8)
                gl.glDrawArrays(mode, 0, len(data))
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            self.dirty = False
        except Exception as error:
            self.failed.emit(str(error))


class CloudView(QtWidgets.QWidget):
    mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, software=False):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        software = software or QtWidgets.QApplication.platformName() in ('offscreen', 'minimal')
        self.canvas = SoftwareCloud(self) if software else GLCloud(self)
        self.layout.addWidget(self.canvas, 1)
        self.mode_label = QtWidgets.QLabel('软件绘图 · 显示抽样' if software else 'OpenGL 点云绘图', self)
        self.mode_label.setStyleSheet('color: #8095a5;')
        self.mode_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.layout.addWidget(self.mode_label)
        if isinstance(self.canvas, GLCloud):
            self.canvas.failed.connect(self.fallback, QtCore.Qt.QueuedConnection)
            self.canvas.initialized.connect(lambda renderer: self.mode_label.setText('OpenGL · ' + renderer))
            QtCore.QTimer.singleShot(1500, self.check_context)

    def check_context(self):
        if isinstance(self.canvas, GLCloud) and self.isVisible() and not self.canvas.isValid():
            self.fallback('OpenGL 上下文不可用')

    def fallback(self, reason):
        if not isinstance(self.canvas, GLCloud):
            return
        old = self.canvas
        self.canvas = SoftwareCloud(self)
        for name in ('points', 'lines', 'reference_vertices', 'point_scale', 'focus', 'yaw', 'elevation', 'distance', 'key'):
            setattr(self.canvas, name, getattr(old, name))
        self.layout.replaceWidget(old, self.canvas)
        self.layout.setStretch(0, 1)
        old.hide()
        old.deleteLater()
        self.mode_label.setText('软件绘图 · 显示抽样 · ' + reason)
        self.mode_changed.emit('三维视图使用软件绘图：' + reason)

    def scene(self, *args):
        self.canvas.scene(*args)

    def camera(self, elevation, yaw):
        self.canvas.camera(elevation, yaw)

    def set_point_scale(self, scale):
        self.canvas.set_point_scale(scale)
