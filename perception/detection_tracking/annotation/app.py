"""Qt center editor with cached bag I/O, blitted BEV and buffered 3D; no Torch/ROS master."""
import copy
from collections import deque
import json
import math
import os
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data import (ANNOTATION_STATES, FRAME_STATUS_NAMES, annotation_step,
                   complete_review, edit_review, empty_review,
                   pending_candidate_ids, resolve_sequence)
from .backend import LocalBackend
from .cloud_view import CloudView
from .appearance import BACKGROUND, HEIGHT_COLORS, height_colors, ANNOTATION_COLORS, review_display_status
from .timeline import BagTimeline


COLORS = dict(pending='#ffd166', accepted='#38df9c', rejected='#f36878', reference='#93adcc')
NAMES = dict(pending='待复核', accepted='已确认', rejected='已拒绝', reference='仅供参考')


def box_corners(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [dx / 2, dy / 2]
    c, s = math.cos(yaw), math.sin(yaw)
    xy = local @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.vstack([np.c_[xy, np.full(4, z - dz / 2)], np.c_[xy, np.full(4, z + dz / 2)]])


class Job(QtCore.QThread):
    result = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function, self.cancelled = function, False

    def report(self, text):
        if self.cancelled:
            raise RuntimeError('操作已取消')
        self.progress.emit(str(text))

    def run(self):
        try:
            value = self.function(self.report)
            if not self.cancelled:
                self.result.emit(value)
        except Exception as error:
            if not self.cancelled:
                self.error.emit(type(error).__name__ + ': ' + str(error))

class ImageView(QtWidgets.QLabel):
    """Keep one source pixmap and rescale on resize without redecoding the bag."""
    def __init__(self):
        super().__init__('无匹配 RGB')
        self.original = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(240, 135)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet('background: #172029; color: #d8e2ec;')

    def show_image(self, image):
        if image is None:
            self.original = None
            self.setText('扫描前 100 ms 内无可用 RGB；可继续点云标注。')
            self.setToolTip('')
            return
        data = np.ascontiguousarray(image)
        h, w = data.shape[:2]
        qimage = QtGui.QImage(data.data, w, h, data.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.original = QtGui.QPixmap.fromImage(qimage)
        self.fit_image()

    def fit_image(self):
        if self.original is not None:
            self.setPixmap(self.original.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image()


class Window(QtWidgets.QMainWindow):
    def __init__(self, args, backend=None):
        super().__init__()
        self.args = args
        self.backend = backend if backend is not None else LocalBackend(args.project)
        self.remote = bool(getattr(self.backend, 'is_remote', False))
        self.synchronizing, self.remote_refresh_seen, self.remote_exit_confirmed = False, 0, False
        self.remote_pending_seen = 0
        self.remote_refresh_requested = 0
        self.edit_controls = []
        self.bounds, self.metadata = None, {}
        self.writes, self.write_task = deque(), None
        self.save_error, self.pending_navigation = None, None
        self.close_requested, self.backend_closed, self.resetting = False, False, False
        self.jobs, self.rows, self.index, self.frame = [], [], -1, None
        self.points, self.image, self.image_ns = None, None, None
        self.review, self.selection = None, None
        self.undo_stack, self.redo_stack = [], []
        self.loading, self.closing, self.adding, self.dragging = False, False, False, False
        self.selection_artist = None
        self.xy_binding = None
        self.view_key, self.drag_position = None, None
        self.source_id, self.rows_by_source, self.progress = None, {}, {}
        self.all_rows_by_source, self.row_by_id = {}, {}
        self.sample_intervals = {}
        self.visited_sources = set()
        self.requested_frame = None
        self.prefetch_task = None
        self.bev_background, self.point_key, self.aux_key, self.rgb_key = None, None, None, None
        self.text_artists = []
        self.setWindowTitle('行人中心复核 · ' + (str(getattr(args, 'project_id', '远程')) if self.remote else str(args.project)))
        self.resize(1500, 920)
        self.build_ui()
        self.centralWidget().setEnabled(False)
        self.focus_changed(None, None)
        QtCore.QTimer.singleShot(0, self.prepare)
        if self.remote:
            self.sync_timer = QtCore.QTimer(self)
            self.sync_timer.timeout.connect(self.poll_remote)
            self.sync_timer.start(500)

    def message(self, value):
        self.statusBar().showMessage(str(value))

    def error(self, value):
        if not self.closing:
            self.message(value)
            QtWidgets.QMessageBox.warning(self, '操作未完成', str(value))

    def job(self, function, done):
        task = Job(function, self)
        self.jobs.append(task)
        task.progress.connect(self.message)
        task.error.connect(self.error)
        task.result.connect(lambda value: done(value) if not self.closing else None)
        task.finished.connect(lambda: self.finish_job(task))
        task.start()
        return task

    def finish_job(self, task):
        if task in self.jobs:
            self.jobs.remove(task)
        task.deleteLater()

    def queue_write(self, function, done=None, kind='settings', key=None):
        # One writer preserves read-modify-write preferences and review CAS order.
        # Coalesce queued preferences, never annotation edits or destructive actions.
        if key is not None:
            self.writes = deque(item for item in self.writes if item['key'] != key)
        self.writes.append(dict(function=function, done=done, kind=kind, key=key))
        self.start_write()
        self.update_save_notice()

    def start_write(self):
        if self.write_task is not None or not self.writes:
            return
        entry = self.writes.popleft()
        task = Job(lambda progress: entry['function'](), self)
        task.entry = entry
        self.write_task = task
        self.jobs.append(task)
        task.result.connect(lambda value: entry['done'](value) if entry['done'] else None)
        task.error.connect(lambda value: self.write_failed(entry, value))
        task.finished.connect(lambda: self.write_finished(task))
        task.start()

    def write_failed(self, entry, value):
        self.close_requested = False
        self.pending_navigation = None
        if entry['kind'] == 'review':
            # Later CAS writes depend on this one. Keep the latest working draft
            # in memory, retain the last acknowledged baseline, and require retry
            # or explicit discard. No silent reload can erase a person's edits.
            self.writes = deque(item for item in self.writes if item['kind'] != 'review')
            self.save_error = value
        if entry['kind'] == 'reset':
            self.resetting = False
            self.centralWidget().setEnabled(True)
            self.focus_changed(None, QtWidgets.QApplication.focusWidget())
        self.update_save_notice()
        self.error(('标注未保存，当前修改仍保留在窗口中。\n' if entry['kind'] == 'review' else '写入未完成。\n') + value)

    def write_finished(self, task):
        self.write_task = None
        self.finish_job(task)
        self.start_write()
        self.update_save_notice()
        if self.write_task is not None:
            return
        if self.save_error:
            return
        if task.entry['kind'] == 'review':
            message = '已自动保存'
            if self.remote:
                message = '已在本机暂存，后台同步中' if self.backend.status()['pending'] else '服务器已保存'
            self.message(message)
        if self.close_requested:
            QtCore.QTimer.singleShot(0, self.close)
        elif self.pending_navigation is not None:
            action, self.pending_navigation = self.pending_navigation, None
            action()

    def update_save_notice(self):
        pending = sum(item['kind'] == 'review' for item in self.writes)
        pending += int(self.write_task is not None and self.write_task.entry['kind'] == 'review')
        self.retry_button.setVisible(bool(self.save_error))
        self.reload_button.setVisible(bool(self.save_error))
        if self.save_error:
            self.save_notice.setText('未保存 · 请重试；冲突时可放弃本机未保存修改并重读。关闭前须处理。')
            self.save_notice.setStyleSheet('color: #a32c3b;')
        elif pending:
            self.save_notice.setText(('正在本机暂存' if self.remote else '保存中') + ' · %d 项；切帧和退出会等待保存完成。' % pending)
            self.save_notice.setStyleSheet('color: #866116;')
        elif self.remote:
            state = self.backend.status()
            if state['conflicts']:
                text, color = '同步冲突 · 本机草稿保留，请处理冲突。', '#a32c3b'
            elif state['pending']:
                text, color = '已在本机暂存 · 待同步 %d 项' % state['pending'], '#866116'
                if state['read_only']:
                    text += ' · 只读时不发送'
            elif state['read_only']:
                text, color = '只读浏览', '#526678'
            else:
                text, color = ('服务器已保存' if self.frame is not None else '远程工程已连接'), '#28714c'
            if not state['online']:
                text += ' · 离线缓存'
            if state['error']:
                text += '\n' + state['error']
            if state['refresh_serial'] > self.remote_refresh_seen:
                text += '\n子包状态已改变；刷新完成前暂停编辑，可点击同步／重连。'
            self.save_notice.setText(text)
            self.save_notice.setStyleSheet('color: ' + color + ';')
            self.conflict_button.setVisible(bool(state['conflicts']))
        else:
            self.save_notice.setText('已保存到本机工程' if self.frame is not None else '')
            self.save_notice.setStyleSheet('color: #28714c;')

    def save_metadata(self, key, value):
        value = copy.deepcopy(value)
        self.metadata[key] = value
        self.queue_write(lambda: self.backend.set_meta(key, value), key=key)

    def after_saved(self, action):
        if self.synchronizing:
            self.message('正在刷新远程状态；可继续操作视图。')
            return False
        if not self.loading and not self.resetting:
            self.save_note()
        if self.save_error:
            self.message('当前修改未保存；请先重试保存，或明确放弃未保存修改并重读。')
            return False
        if self.write_task is not None or self.writes or self.resetting:
            self.pending_navigation = action
            self.message('等待保存完成后继续；可继续操作视图。')
            return False
        return True

    def editable(self):
        if self.resetting or self.synchronizing or self.closing:
            return False
        if not self.remote:
            return True
        state = self.backend.status()
        return (not self.backend.read_only and not state['conflicts'] and not state['reset_pending'] and
                state['refresh_serial'] <= self.remote_refresh_seen)

    def poll_remote(self):
        if self.closing:
            return
        self.update_save_notice()
        enabled = self.editable() and not self.loading and self.frame is not None
        for widget in self.edit_controls + list(self.state_buttons.values()):
            widget.setEnabled(enabled)
        movable = bool(enabled and self.selection and self.selection[0] == 'center')
        for widget in (self.xvalue, self.yvalue, self.move_button):
            widget.setEnabled(movable)
        self.note.setReadOnly(not enabled)
        state = self.backend.status()
        if self.remote_pending_seen and not state['pending'] and not self.save_error:
            self.message('服务器已保存')
        self.remote_pending_seen = state['pending']
        self.connection_caption.setText('远程数据 · ' + ('在线' if state['online'] else '离线缓存') +
                                        (' · 只读' if self.backend.read_only else ' · 编辑在本机暂存后同步'))
        if (state['refresh_serial'] > self.remote_refresh_requested and not self.loading and not self.synchronizing and
                self.write_task is None and not self.writes and not self.save_error):
            self.remote_refresh_requested = state['refresh_serial']
            self.refresh_remote()

    def refresh_remote(self, resolve=False):
        if self.synchronizing or self.loading or self.closing:
            return
        if not self.after_saved(lambda: self.refresh_remote(resolve)):
            return
        if resolve:
            reply = QtWidgets.QMessageBox.question(self, '处理同步冲突',
                '将发生冲突的子包中全部未同步编辑保留到本机冲突存档，然后读取服务器结果？\n'
                '不会覆盖服务器标注；存档可通过export-drafts导出。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        self.synchronizing = True
        self.sync_button.setEnabled(False)
        self.poll_remote()
        function = self.backend.resolve_conflicts if resolve else self.backend.synchronize
        task = self.job(lambda progress: function(), self.remote_refreshed)
        task.error.connect(lambda _: self.remote_refresh_failed())

    def remote_refresh_failed(self):
        self.synchronizing = False
        self.sync_button.setEnabled(True)
        self.poll_remote()

    def remote_refreshed(self, snapshot):
        self.synchronizing = False
        self.sync_button.setEnabled(True)
        self.remote_refresh_seen = self.backend.status()['refresh_serial']
        self.remote_refresh_requested = self.remote_refresh_seen
        # Preferences belong to this client; a refresh must not revert a point
        # scale or browser position changed while its HTTP request was in flight.
        snapshot['metadata'] = copy.deepcopy(self.metadata)
        self.clear_frame()
        self.source_id = None
        self.undo_stack, self.redo_stack = [], []
        self.refresh(snapshot)
        self.poll_remote()

    def queue_review(self, review, expected):
        ident = self.frame['id']
        review, expected = copy.deepcopy(review), copy.deepcopy(expected)

        def saved(frame):
            if self.frame is not None and self.frame['id'] == ident:
                self.frame = frame  # Acknowledged baseline; keep newer working edits.

        self.queue_write(lambda: self.backend.save_review(ident, review, expected), saved, kind='review')

    def retry_save(self):
        if self.save_error and self.frame is not None:
            self.save_note()
            self.save_error = None
            self.queue_review(self.review, self.frame['review'])

    def reload_unsaved(self):
        if not self.save_error or self.frame is None:
            return
        reply = QtWidgets.QMessageBox.question(self, '放弃未保存修改',
            '放弃当前帧尚未保存的全部修改，并重新读取工程中的结果？\n已保存的结果保留。',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.save_error, self.pending_navigation = None, None
        self.review = copy.deepcopy(self.frame['review'])
        self.note.setText(self.review.get('note', ''))
        self.undo_stack, self.redo_stack = [], []
        self.sync_status(self.frame)
        self.update_save_notice()
        self.load(self.index)

    def button(self, layout, label, action):
        widget = QtWidgets.QPushButton(label)
        widget.clicked.connect(action)
        layout.addWidget(widget)
        return widget

    def build_ui(self):
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        sequence = QtWidgets.QHBoxLayout()
        sequence.addWidget(QtWidgets.QLabel('当前子 bag'))
        self.sequence = QtWidgets.QComboBox()
        self.sequence.currentIndexChanged.connect(self.sequence_changed)
        sequence.addWidget(self.sequence, 1)
        self.button(sequence, '上一包 [PgUp]', lambda: self.change_sequence(-1))
        self.button(sequence, '下一包 [PgDn]', lambda: self.change_sequence(1))
        self.connection_caption = QtWidgets.QLabel('缓存标注 · 不使用检测 GPU')
        sequence.addWidget(self.connection_caption)
        layout.addLayout(sequence)
        self.caption = QtWidgets.QLabel('正在读取工程……')
        self.caption.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.caption)
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter, 1)
        central = QtWidgets.QWidget()
        views = QtWidgets.QVBoxLayout(central)
        views.setContentsMargins(0, 0, 0, 0)
        views.addWidget(QtWidgets.QLabel('BEV · 可编辑人工 XY · 点色表示高度'))
        self.figure = Figure(figsize=(8, 7), facecolor=BACKGROUND)
        self.figure.subplots_adjust(left=.13, right=.97, bottom=.11, top=.98)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        views.addWidget(self.toolbar)
        views.addWidget(self.canvas, 3)
        auxiliary = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.cloud_view = CloudView(self, software=getattr(self.args, 'cloud_renderer', 'auto') == 'software')
        self.cloud_view.mode_changed.connect(self.message)
        scene = QtWidgets.QWidget()
        scene_layout = QtWidgets.QVBoxLayout(scene)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        self.cloud_caption = QtWidgets.QLabel('三维点云 · 全局')
        self.cloud_caption.setWordWrap(True)
        scene_layout.addWidget(self.cloud_caption)
        cameras = QtWidgets.QHBoxLayout()
        self.button(cameras, '俯视 [T]', lambda: self.camera(90, -90))
        self.button(cameras, '斜视 [V]', lambda: self.camera(35, -60))
        self.local_view = QtWidgets.QCheckBox('选中局部 [L]')
        self.local_view.setChecked(True)
        self.local_view.toggled.connect(lambda _: self.render_aux())
        cameras.addWidget(self.local_view)
        self.button(cameras, '复位 [F]', self.reset_view)
        scene_layout.addLayout(cameras)
        scene_layout.addWidget(self.cloud_view, 1)
        self.grid_caption = QtWidgets.QLabel('')
        self.grid_caption.setWordWrap(True)
        self.grid_caption.setStyleSheet('color: #526678; font-size: 11px;')
        scene_layout.addWidget(self.grid_caption)
        scene.setToolTip('三维点云：左键旋转、中键／Shift 拖动平移、滚轮／右键缩放；选择目标后查看局部原始点。')
        rgb_panel = QtWidgets.QWidget()
        rgb_layout = QtWidgets.QVBoxLayout(rgb_panel)
        rgb_layout.setContentsMargins(0, 0, 0, 0)
        rgb_layout.addWidget(QtWidgets.QLabel('原始 RGB（不投影）'))
        self.rgb = ImageView()
        rgb_layout.addWidget(self.rgb, 1)
        auxiliary.addWidget(scene)
        auxiliary.addWidget(rgb_panel)
        auxiliary.setSizes([450, 280])
        splitter.addWidget(central)
        splitter.addWidget(auxiliary)
        side = QtWidgets.QWidget()
        side.setMinimumWidth(330)
        side.setMaximumWidth(430)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(panel)
        controls.setContentsMargins(0, 0, 4, 0)
        controls.setSpacing(5)
        scroll.setWidget(panel)
        side_layout.addWidget(scroll, 1)
        legend = QtWidgets.QLabel('黄框：待复核　绿十字：人工 XY\n灰框：模型参考　红框：已拒绝　白：选中')
        legend.setStyleSheet('font-size: 12px; color: #394d60;')
        controls.addWidget(legend)
        self.group_toggle = QtWidgets.QToolButton()
        self.group_toggle.setText('分组与序列管理 ▸')
        self.group_toggle.setCheckable(True)
        controls.addWidget(self.group_toggle)
        self.group_panel = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout(self.group_panel)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group = QtWidgets.QFormLayout()
        self.root_group, self.location_group = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.root_group.setToolTip('来自同一原始录制的切片使用相同值，避免相邻片段跨训练／验证集；不影响当前子包的标注。')
        self.location_group.setToolTip('不同录制若重复经过同一地点，使用相同值，避免重复场景跨训练／验证集。')
        self.split = QtWidgets.QComboBox()
        self.split.addItems(['unassigned', 'train', 'val', 'holdout'])
        group.addRow('原始采集组', self.root_group)
        group.addRow('重复地点组', self.location_group)
        group.addRow('数据划分', self.split)
        group_layout.addLayout(group)
        self.edit_controls.append(self.button(group_layout, '保存当前 bag 分组', self.save_group))
        clear_button = self.button(group_layout, '清空当前序列所有标记…', self.clear_sequence_marks)
        self.edit_controls.extend([clear_button, self.root_group, self.location_group, self.split])
        clear_button.setToolTip(
            '需要确认：清空当前子包所有帧的状态、人工中心、候选决定和备注，恢复原检测建议；保留其他子包与检测缓存。')
        self.group_panel.setVisible(False)
        self.group_toggle.toggled.connect(self.group_panel.setVisible)
        self.group_toggle.toggled.connect(lambda checked: self.group_toggle.setText('分组与序列管理 ' + ('▾' if checked else '▸')))
        controls.addWidget(self.group_panel)
        row = QtWidgets.QHBoxLayout()
        self.score = QtWidgets.QDoubleSpinBox()
        self.score.setRange(0, 1)
        self.score.setSingleStep(.05)
        self.score.setValue(.1)
        self.score.setMinimumWidth(65)
        self.score.setMaximumWidth(90)
        row.addWidget(QtWidgets.QLabel('显示阈值'))
        row.addWidget(self.score)
        self.rejected = QtWidgets.QCheckBox('显示已拒绝 [H]')
        self.rejected.setToolTip('显示被人工拒绝的红色建议框，便于回查和重新确认；仍受显示阈值限制，不改变标注。')
        row.addWidget(self.rejected)
        controls.addLayout(row)
        height = QtWidgets.QHBoxLayout()
        self.zlow, self.zhigh = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for widget in (self.zlow, self.zhigh):
            widget.setMinimumWidth(75)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            widget.setRange(-100, 100)
            widget.setSingleStep(.25)
        height.addWidget(QtWidgets.QLabel('显示 Z'))
        height.addWidget(self.zlow)
        height.addWidget(self.zhigh)
        controls.addLayout(height)
        point_size = QtWidgets.QHBoxLayout()
        point_size.addWidget(QtWidgets.QLabel('点大小'))
        self.point_scale = QtWidgets.QDoubleSpinBox()
        self.point_scale.setRange(.5, 4.)
        self.point_scale.setDecimals(1)
        self.point_scale.setSingleStep(.1)
        self.point_scale.setSuffix(' ×')
        self.point_scale.setValue(1.)
        self.point_scale.setMinimumWidth(80)
        self.point_scale.setMaximumWidth(110)
        self.point_scale.setToolTip('同时调整 BEV 与三维点云的显示点径；只改变显示大小，不改变原始数据、候选或标注。工程记住设置。')
        point_size.addWidget(self.point_scale)
        self.button(point_size, '恢复 1×', lambda: self.point_scale.setValue(1.))
        point_size.addStretch(1)
        controls.addLayout(point_size)
        self.cloud_view.set_point_scale(self.point_scale.value())
        self.point_scale.valueChanged.connect(self.resize_points)
        self.listing = QtWidgets.QListWidget()
        self.listing.setMinimumHeight(155)
        self.listing.setStyleSheet('QListWidget { background: #172029; color: #d8e2ec; } '
                                   'QListWidget::item:selected { background: #33495d; }')
        self.listing.itemClicked.connect(self.list_clicked)
        controls.addWidget(self.listing, 1)
        operations = QtWidgets.QGridLayout()
        buttons = [('确认候选 [Q]', self.accept), ('删除／拒绝 [R]', self.delete),
                   ('补漏点 [E]', self.add_mode), ('确认可见候选 [B]', self.accept_remaining),
                   ('恢复未核验 [C]', self.reset_candidate),
                   ('撤销 [Ctrl+Z]', self.undo), ('重做 [Ctrl+Y]', self.redo)]
        for index, (label, action) in enumerate(buttons):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(action)
            self.edit_controls.append(button)
            operations.addWidget(button, index // 2, index % 2)
        controls.addLayout(operations)
        values = QtWidgets.QGridLayout()
        self.xvalue, self.yvalue = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for widget in (self.xvalue, self.yvalue):
            widget.setMinimumWidth(85)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            widget.setRange(-10000, 10000)
            widget.setDecimals(3)
            widget.setSingleStep(.05)
        values.addWidget(QtWidgets.QLabel('X / m'), 0, 0)
        values.addWidget(self.xvalue, 0, 1)
        values.addWidget(QtWidgets.QLabel('Y / m'), 0, 2)
        values.addWidget(self.yvalue, 0, 3)
        self.move_button = QtWidgets.QPushButton('移动选中中心')
        self.move_button.clicked.connect(self.move_numeric)
        values.addWidget(self.move_button, 1, 0, 1, 4)
        self.move_button.setToolTip('将选中的人工中心移动到左侧 X、Y 坐标；候选需先确认，可撤销。')
        self.move_button.setEnabled(False)
        controls.addLayout(values)
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText('可选备注；修改后自动保存')
        self.note.editingFinished.connect(self.save_note)
        controls.addWidget(self.note)
        self.save_notice = QtWidgets.QLabel('')
        self.save_notice.setWordWrap(True)
        side_layout.addWidget(self.save_notice)
        recovery = QtWidgets.QHBoxLayout()
        self.retry_button = self.button(recovery, '重试保存', self.retry_save)
        self.reload_button = self.button(recovery, '放弃未保存修改并重读…', self.reload_unsaved)
        self.retry_button.hide()
        self.reload_button.hide()
        side_layout.addLayout(recovery)
        if self.remote:
            remote_actions = QtWidgets.QHBoxLayout()
            self.sync_button = self.button(remote_actions, '同步／重连', lambda: self.refresh_remote(False))
            self.conflict_button = self.button(remote_actions, '处理同步冲突…', lambda: self.refresh_remote(True))
            self.conflict_button.hide()
            side_layout.addLayout(remote_actions)
        help_row = QtWidgets.QHBoxLayout()
        help_row.addWidget(QtWidgets.QLabel('修改自动保存 · 默认导出待标和已标帧'), 1)
        self.button(help_row, '快捷键 [F1]', self.show_shortcuts)
        controls.addLayout(help_row)
        states = QtWidgets.QHBoxLayout()
        states.addWidget(QtWidgets.QLabel('标记为'))
        self.state_buttons = {}
        self.state_buttons['draft'] = self.button(states, '待标／普通 [1]', lambda: self.mark_status('draft'))
        self.state_buttons['draft'].setToolTip('撤销跳过／已标状态；有人工核验或BEV点则为待标，否则退回普通帧。')
        self.state_buttons['skipped'] = self.button(states, '跳过 [2]', lambda: self.mark_status('skipped'))
        self.state_buttons['skipped'].setToolTip('保留当前内容，标记为跳过，不用于训练；不跳帧。')
        self.complete_button = self.button(states, '已标 [3]', lambda: self.mark_status('complete'))
        self.state_buttons['complete'] = self.complete_button
        self.complete_button.setToolTip('直接接受当前阈值内的待复核候选并保存为已标；空中心表示无人。请先补齐漏检，隐藏待复核项须先处理。不弹确认、不跳帧。')
        side_layout.addLayout(states)
        self.empty_button = QtWidgets.QPushButton('标记无人／空标注 [4]')
        self.empty_button.clicked.connect(lambda: self.mark_status('empty'))
        self.empty_button.setToolTip(
            '明确确认检测范围内整帧无人：拒绝范围内全部候选（含隐藏低分项），清除已确认中心及人工补点，'
            '保存为已标（无人）。不弹确认、不跳帧，可用Ctrl+Z撤销。显示筛选不缩小确认范围。')
        self.edit_controls.append(self.empty_button)
        side_layout.addWidget(self.empty_button)
        hint = QtWidgets.QLabel('状态不跳帧 · W / S 切换标注帧')
        hint.setStyleSheet('color: #526678; font-size: 12px;')
        side_layout.addWidget(hint)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([630, 490, 370])
        navigation = QtWidgets.QHBoxLayout()
        self.button(navigation, '上一原始帧 [A]', lambda: self.navigate(-1, False))
        self.button(navigation, '下一原始帧 [D]', lambda: self.navigate(1, False))
        for direction, label in [(-1, '上一标注帧 [W]'), (1, '下一标注帧 [S]')]:
            self.button(navigation, label, lambda checked=False, step=direction: self.navigate(step, True)).setToolTip(
                '优先复用方向内一个步长范围内最近的标注帧（含已标／跳过）；否则按步长浏览，不改变任何帧的状态。')
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(.01, 3600)
        self.interval.setValue(self.args.sample_interval or 1.)
        self.interval.valueChanged.connect(self.save_interval)
        navigation.addWidget(QtWidgets.QLabel('采样步长 s'))
        navigation.addWidget(self.interval)
        self.seek = QtWidgets.QDoubleSpinBox()
        self.seek.setRange(0, 1000000)
        self.seek.setDecimals(3)
        navigation.addWidget(QtWidgets.QLabel('当前包秒数'))
        navigation.addWidget(self.seek)
        self.button(navigation, '跳转', self.seek_time)
        layout.addLayout(navigation)
        self.timeline = BagTimeline(self)
        self.timeline.seek.connect(self.seek_frame)
        layout.addWidget(self.timeline)
        for widget in (self.score, self.zlow, self.zhigh):
            widget.valueChanged.connect(lambda _: self.render())
        self.rejected.toggled.connect(lambda _: self.render())
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('motion_notify_event', self.motion)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('scroll_event', self.scroll)
        self.canvas.mpl_connect('draw_event', self.background_drawn)
        self.canvas.mpl_connect('resize_event', lambda _: self.invalidate_background())
        self.axes.callbacks.connect('xlim_changed', lambda _: self.invalidate_background())
        self.axes.callbacks.connect('ylim_changed', lambda _: self.invalidate_background())
        actions = [
            ('1', lambda: self.mark_status('draft'), '撤销完成标记：待标／普通'),
            ('2', lambda: self.mark_status('skipped'), '标记为跳过'),
            ('3', lambda: self.mark_status('complete'), '标记为已标'),
            ('4', lambda: self.mark_status('empty'), '标记无人／空标注，拒绝全部候选并清空中心'),
            ('W', lambda: self.navigate(-1, True), '上一标注帧'),
            ('S', lambda: self.navigate(1, True), '下一标注帧'),
            ('A', lambda: self.navigate(-1, False), '上一原始帧'),
            ('D', lambda: self.navigate(1, False), '下一原始帧'),
            ('Up', lambda: self.select_list_item(-1), '选择上一候选／中心'),
            ('Down', lambda: self.select_list_item(1), '选择下一候选／中心'),
            ('PgUp', lambda: self.change_sequence(-1), '上一子包'),
            ('PgDown', lambda: self.change_sequence(1), '下一子包'),
            ('E', self.add_mode, '补漏点'), ('Escape', self.cancel_selection, '取消补点／拖动／选择'),
            ('R', self.delete, '拒绝候选／删除中心'), ('Q', self.accept, '确认选中候选'),
            ('C', self.reset_candidate, '恢复选中候选为未核验'),
            ('B', self.accept_remaining, '确认可见待复核候选'),
            ('Ctrl+Z', self.undo, '撤销'), ('Ctrl+Y', self.redo, '重做'),
            ('H', self.rejected.toggle, '显示／隐藏已拒绝候选'),
            ('L', self.local_view.toggle, '局部／全局点云'),
            ('T', lambda: self.camera(90, -90), '点云俯视'),
            ('V', lambda: self.camera(35, -60), '点云斜视'), ('F', self.reset_view, '复位视图'),
            ('F1', self.show_shortcuts, '快捷键帮助'),
        ]
        self.shortcuts = []
        self.shortcut_help = '\n'.join('%s：%s' % (key, name) for key, _, name in actions)
        for key, action, _ in actions:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=action)
            shortcut.setAutoRepeat(key in ('W', 'S', 'A', 'D', 'Up', 'Down'))
            self.shortcuts.append(shortcut)
        QtWidgets.QApplication.instance().focusChanged.connect(self.focus_changed)

    def focus_changed(self, old, current):
        editing = False
        while current is not None and current is not self:
            if isinstance(current, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
                editing = True
                break
            current = current.parentWidget()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(not editing and self.bounds is not None and not self.resetting and not self.closing)

    def show_shortcuts(self):
        QtWidgets.QMessageBox.information(self, '标注快捷键', self.shortcut_help + '\n\n输入框内保留原有编辑行为；标记状态不自动跳帧。')

    def prepare(self):
        cache = self.args.detections
        self.job(lambda progress: self.backend.prepare(progress, cache), self.refresh)

    def refresh(self, snapshot):
        self.metadata = snapshot['metadata']
        self.sample_intervals = self.metadata.get('sample_intervals') or {}
        first = self.bounds is None
        if not first and self.bounds != snapshot['bounds']:
            raise ValueError('工程检测范围已改变，请关闭窗口后重新打开')
        self.bounds = snapshot['bounds']
        if first:
            self.setup_bev()
        for widget, value in ((self.zlow, self.bounds[2]), (self.zhigh, self.bounds[5]),
                              (self.point_scale, self.metadata.get('point_scale') or 1.)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.cloud_artist.set_sizes([2.4 * self.point_scale.value() ** 2])
        self.cloud_view.set_point_scale(self.point_scale.value())
        self.centralWidget().setEnabled(True)
        self.focus_changed(None, QtWidgets.QApplication.focusWidget())
        sources = snapshot['sources']
        self.sources = {s['id']: s for s in sources}
        self.rows_by_source = {s['id']: [] for s in sources}
        self.all_rows_by_source = {s['id']: [] for s in sources}
        self.row_by_id = {}
        self.beginnings = {}
        self.progress = {s['id']: dict(total=0, unreviewed=0, draft=0, complete=0, skipped=0) for s in sources}
        for row in sorted(snapshot['rows'], key=lambda r: r['ordinal']):
            ident = row['source_id']
            beginning = self.beginnings.setdefault(ident, row['record_ns'])
            review = json.loads(row['review'])
            row['status'] = review['status']
            row['display_status'] = review_display_status(review)
            self.all_rows_by_source[ident].append(row)
            self.row_by_id[row['id']] = row
            self.progress[ident]['total'] += 1
            self.progress[ident][row['status']] += 1
            elapsed = (row['record_ns'] - beginning) / 1e9
            if self.args.start <= elapsed and (self.args.duration is None or elapsed < self.args.start + self.args.duration):
                self.rows_by_source[ident].append(row)
        self.sequence.blockSignals(True)
        self.sequence.clear()
        for source in sources:
            if self.rows_by_source[source['id']]:
                self.sequence.addItem(self.sequence_label(source['id']), source['id'])
                self.sequence.setItemData(self.sequence.count() - 1, source['path'], QtCore.Qt.ToolTipRole)
        self.sequence.blockSignals(False)
        if not self.sequence.count():
            self.error('指定时间范围内没有扫描')
            return
        last = self.metadata.get('last_frame')
        last_source = next((ident for ident, rows in self.rows_by_source.items() if any(r['id'] == last for r in rows)), None)
        requested = getattr(self.args, 'sequence', None) if first else None
        ident = resolve_sequence(sources, requested) if requested else last_source
        index = self.sequence.findData(ident)
        if requested and index < 0:
            self.error('指定子包在当前时间范围内没有扫描')
            return
        index = max(0, index)
        self.sequence.blockSignals(True)
        self.sequence.setCurrentIndex(index)
        self.sequence.blockSignals(False)
        self.sequence_changed(index)

    def sequence_label(self, ident):
        counts = self.progress[ident]
        marked = sum(counts[state] for state in ANNOTATION_STATES)
        return '%s · 已标 %d / 标注帧 %d' % (Path(self.sources[ident]['path']).name, counts['complete'], marked)

    def sync_status(self, frame):
        row = self.row_by_id[frame['id']]
        old, new = row['status'], frame['review']['status']
        row['display_status'] = review_display_status(frame['review'])
        if old != new:
            counts = self.progress[frame['source_id']]
            counts[old] -= 1
            counts[new] += 1
            row['status'] = new
            index = self.sequence.findData(frame['source_id'])
            if index >= 0:
                self.sequence.setItemText(index, self.sequence_label(frame['source_id']))
        self.timeline.set_review(frame['id'], frame['review'])

    def save_interval(self):
        if self.source_id is not None:
            self.sample_intervals[self.source_id] = self.interval.value()
            self.save_metadata('sample_intervals', self.sample_intervals)

    def change_sequence(self, direction):
        target = self.sequence.currentIndex() + direction
        if 0 <= target < self.sequence.count():
            self.sequence.setCurrentIndex(target)

    def sequence_changed(self, index):
        ident = self.sequence.itemData(index)
        if ident is None or ident == self.source_id:
            return
        if not self.after_saved(lambda: self.activate_sequence(ident)):
            self.sequence.blockSignals(True)
            self.sequence.setCurrentIndex(self.sequence.findData(self.source_id))
            self.sequence.blockSignals(False)
            return
        self.source_id = ident
        self.rows = self.rows_by_source[ident]
        self.interval.blockSignals(True)
        value = self.sample_intervals.get(ident, 1.)
        if ident not in self.visited_sources and self.args.sample_interval is not None:
            value = self.args.sample_interval
        self.interval.setValue(value)
        self.interval.blockSignals(False)
        self.visited_sources.add(ident)
        self.save_interval()
        self.index = -1
        self.clear_frame()
        self.timeline.set_frames(self.all_rows_by_source[ident], [row['id'] for row in self.rows])
        positions = self.metadata.get('last_frames') or {}
        last = positions.get(ident, self.metadata.get('last_frame'))
        self.load(next((i for i, row in enumerate(self.rows) if row['id'] == last), 0))

    def activate_sequence(self, ident):
        index = self.sequence.findData(ident)
        if index >= 0:
            self.sequence.blockSignals(True)
            self.sequence.setCurrentIndex(index)
            self.sequence.blockSignals(False)
            self.sequence_changed(index)

    def load(self, index):
        if not self.rows or self.closing:
            return
        index = max(0, min(index, len(self.rows) - 1))
        row = self.rows[index]
        if self.loading:
            self.requested_frame = row['id']
            return
        if not self.after_saved(lambda: self.load(index)):
            return
        self.loading = True
        self.note.setEnabled(False)
        self.adding, self.dragging = False, False
        self.move_button.setEnabled(False)
        if self.prefetch_task is not None:
            self.prefetch_task.cancelled = True
        self.requested_frame = None
        self.message('读取扫描：' + row['id'])

        def work(progress):
            return self.backend.read_frame(row['id'])

        task = self.job(work, self.loaded)
        task.error.connect(lambda _: self.load_failed())

    def clear_frame(self):
        self.frame, self.review, self.points = None, None, None
        self.image, self.image_ns, self.selection = None, None, None
        self.xy_binding = None
        for button in list(self.state_buttons.values()) + [self.empty_button]:
            button.setStyleSheet('')
        self.caption.setStyleSheet('')
        self.cloud_caption.setText('三维点云 · 正在读取')
        self.grid_caption.setText('')
        self.timeline.set_current(None)
        self.note.setEnabled(False)
        self.adding, self.dragging = False, False
        self.move_button.setEnabled(False)
        self.cloud_artist.set_offsets(np.empty((0, 2)))
        self.box_artist.set_segments([])
        self.centers_artist.set_data([], [])
        self.selected_artist.set_data([], [])
        self.selection_artist.set_data([], [])
        for text in self.text_artists:
            text.set_visible(False)
        self.point_key = self.aux_key = self.rgb_key = None
        self.invalidate_background()
        self.canvas.draw_idle()
        self.cloud_view.scene(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), [0, 0, 0], 20, None)
        self.listing.clear()
        self.rgb.show_image(None)
        self.caption.setText('正在读取所选子包，编辑暂不可用……')
        self.update_save_notice()

    def load_failed(self):
        self.loading = False
        self.clear_frame()
        self.caption.setText('扫描读取失败，未显示旧帧；请重新选择。')
        self.message('读取失败；本次扫描不能确认完整')
        self.load_requested()

    def load_requested(self):
        requested, self.requested_frame = self.requested_frame, None
        if requested is not None:
            ident = requested
            index = next((i for i, row in enumerate(self.rows) if row['id'] == ident), None)
            if index is not None:
                self.load(index)

    def loaded(self, payload):
        frame, points, image, image_ns, image_notice = payload
        self.loading = False
        if self.requested_frame is not None:
            if self.requested_frame != frame['id']:
                self.load_requested()
                return
        self.requested_frame = None
        if frame['source_id'] != self.source_id:
            return  # A previous sequence's read must never replace the selected sequence.
        index = next(i for i, row in enumerate(self.rows) if row['id'] == frame['id'])
        self.index, self.frame = index, frame
        self.points, self.image, self.image_ns = points, image, image_ns
        self.review = copy.deepcopy(frame['review'])
        self.selection = None
        self.undo_stack, self.redo_stack = [], []
        self.metadata['last_frame'] = frame['id']
        self.metadata.setdefault('last_frames', None)
        positions = self.metadata['last_frames'] or {}
        positions[frame['source_id']] = frame['id']
        self.metadata['last_frames'] = positions
        position = dict(id=frame['id'], source_id=frame['source_id'])
        self.queue_write(lambda: self.backend.save_position(position), key='position')
        self.update_save_notice()
        self.sync_status(frame)
        self.timeline.set_current(frame['id'])
        source = self.sources[frame['source_id']]
        if getattr(self, 'group_source', None) != frame['source_id']:
            self.root_group.setText(source['root_group'])
            self.location_group.setText(source['location_group'])
            self.split.setCurrentText(source['split'])
            self.group_toggle.setChecked(False)
            self.group_source = frame['source_id']
        self.note.setText(self.review.get('note', ''))
        self.note.setEnabled(True)
        self.note.setReadOnly(not self.editable())
        self.seek.setValue((frame['record_ns'] - self.beginnings[frame['source_id']]) / 1e9)
        self.render()
        notice = '已读取预计算候选；编辑自动保存。建议框不是真值。' if frame['candidates'] is not None else '此帧无预计算候选，仅可手工标注；可先用 predict 补齐检测。'
        if self.remote and self.backend.read_only:
            notice = '只读浏览当前扫描与已有结果；建议框不是真值。'
        self.message((notice + ' ' + image_notice).strip())
        if self.remote:
            self.poll_remote()
        self.prefetch()

    def prefetch(self):
        if self.closing or self.prefetch_task is not None or self.index < 0:
            return
        current = self.rows[self.index]
        targets = []
        if self.index + 1 < len(self.rows) and self.rows[self.index + 1]['source_id'] == current['source_id']:
            targets.append(self.rows[self.index + 1]['id'])
        planned = annotation_step(self.rows, self.index, self.interval.value(), 1)
        if planned is not None:
            ident = self.rows[planned]['id']
            if ident not in targets:
                targets.append(ident)
        if not targets:
            return

        def work(progress):
            for ident in targets:
                progress('')  # Cancellation boundary; prefetch errors never alter frame/labels.
                self.backend.read_frame(ident)

        task = Job(work, self)
        self.prefetch_task = task
        self.jobs.append(task)
        task.finished.connect(lambda: self.prefetch_finished(task))
        task.start()

    def prefetch_finished(self, task):
        self.prefetch_task = None
        self.finish_job(task)

    def save_group(self):
        if not self.editable():
            self.message('当前不可编辑；请检查只读模式或同步冲突。')
            return
        if self.frame:
            ident = self.frame['source_id']
            values = self.root_group.text(), self.location_group.text(), self.split.currentText()

            def saved(source):
                self.sources[ident] = source
                if self.source_id == ident:
                    self.group_toggle.setChecked(False)
                self.message('当前 bag 的来源分组与划分已保存')

            self.queue_write(lambda: self.backend.save_group(ident, *values), saved)

    def navigate(self, direction, sampled):
        if not self.rows or self.frame is None or self.loading:
            return
        if not self.after_saved(lambda: self.navigate(direction, sampled)):
            return
        if sampled:
            target = annotation_step(self.rows, self.index, self.interval.value(), direction)
        else:
            index = self.index + direction
            target = index if 0 <= index < len(self.rows) else None
        if target is not None:
            self.load(target)
            return
        self.message('已到当前子包队列边界；可用上方选择器或上一包／下一包切换序列')

    def seek_frame(self, ident):
        index = next((i for i, row in enumerate(self.rows) if row['id'] == ident), None)
        if index is None:
            self.message('该帧不在当前 --start/--duration 范围内；未改变标注')
        else:
            self.load(index)

    def seek_time(self):
        if self.frame:
            ident = self.frame['source_id']
            target = self.beginnings[ident] + round(self.seek.value() * 1e9)
            indices = [i for i, r in enumerate(self.rows) if r['source_id'] == ident]
            self.load(min(indices, key=lambda i: abs(self.rows[i]['record_ns'] - target)))

    def candidates(self):
        return self.frame['candidates'] or [] if self.frame else []

    def candidate_state(self, ident):
        state = self.review['decisions'].get(str(ident), 'reference')
        return 'pending' if state == 'uncertain' else state  # Old records remain explicitly reviewable.

    def visible_candidates(self):
        for i, box in enumerate(self.candidates()):
            state = self.candidate_state(i)
            if box[7] >= self.score.value() and (state != 'rejected' or self.rejected.isChecked()):
                yield i, box, state

    def setup_bev(self):
        axes, b = self.axes, self.bounds
        axes.set_facecolor(BACKGROUND)
        axes.tick_params(colors='#b8c9d7')
        axes.set_xlabel('X / m (forward)', color='#b8c9d7')
        axes.set_ylabel('Y / m (left)', color='#b8c9d7')
        axes.set_aspect('equal', adjustable='box')
        axes.grid(color='#91a5b6', alpha=.22, linewidth=.6)
        for spine in axes.spines.values():
            spine.set_color('#7b90a0')
        axes.add_patch(Rectangle((b[0], b[1]), b[3] - b[0], b[4] - b[1], fill=False, edgecolor='#e2a958'))
        axes.annotate('', xy=(2, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='white'))
        cmap = LinearSegmentedColormap.from_list('cloud_height', HEIGHT_COLORS)
        self.cloud_artist = axes.scatter([], [], c=[], s=2.4, cmap=cmap, vmin=b[2], vmax=b[5], linewidths=0)
        # Anchor below the actual equal-aspect axes; fixed figure coordinates
        # overlap the X label when the window becomes shorter.
        color_axes = make_axes_locatable(axes).append_axes('bottom', size='2%', pad=.55)
        colorbar = self.figure.colorbar(self.cloud_artist, cax=color_axes, orientation='horizontal')
        colorbar.set_label('Z / m · sensor frame', color='#cbd8e2', fontsize=9, labelpad=1)
        colorbar.ax.tick_params(colors='#cbd8e2', labelsize=8, length=2, pad=1)
        colorbar.outline.set_edgecolor('#7b90a0')
        self.box_artist = LineCollection([], animated=True)
        axes.add_collection(self.box_artist)
        self.centers_artist, = axes.plot([], [], '+', color=COLORS['accepted'], markersize=11, markeredgewidth=2, animated=True)
        self.selected_artist, = axes.plot([], [], 'o', color='white', markersize=13, fillstyle='none', animated=True)
        self.selection_artist, = axes.plot([], [], 'o', color='white', markersize=8, fillstyle='none', animated=True)
        axes.set_xlim(b[0] - .5, b[3] + .5)
        axes.set_ylim(b[1] - .5, b[4] + .5)

    def invalidate_background(self):
        self.bev_background = None

    def resize_points(self, scale):
        if self.bounds is None:
            return
        # Matplotlib scatter takes marker area; Qt/OpenGL use point diameter.
        self.cloud_artist.set_sizes([2.4 * scale ** 2])
        self.cloud_view.set_point_scale(scale)
        self.invalidate_background()
        self.canvas.draw_idle()
        self.save_metadata('point_scale', scale)

    def background_drawn(self, event):
        if self.bounds is None or event.canvas.is_saving():
            return  # Saving already includes animated artists; retain the screen background.
        self.bev_background = self.canvas.copy_from_bbox(self.axes.bbox)
        # Qt will present this Agg buffer after draw_event returns. Calling
        # blit/repaint here re-enters paintEvent on older Matplotlib backends.
        self.draw_overlays(present=False)

    def draw_overlays(self, present=True):
        if self.bounds is None:
            return
        if self.bev_background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self.bev_background)
        for artist in [self.box_artist, self.centers_artist, self.selected_artist, self.selection_artist] + self.text_artists:
            if artist.get_visible():
                self.axes.draw_artist(artist)
        if present:
            self.canvas.blit(self.axes.bbox)

    def render(self):
        if self.frame is None or self.points is None:
            return
        axes = self.axes
        b = self.bounds
        point_key = (self.frame['id'], self.zlow.value(), self.zhigh.value())
        if point_key != self.point_key:
            points = self.points[np.isfinite(self.points).all(axis=1)]
            shown = points[(points[:, 2] >= point_key[1]) & (points[:, 2] <= point_key[2])]
            shown = shown[::max(1, math.ceil(len(shown) / 35000))]
            self.cloud_artist.set_offsets(shown[:, :2])
            self.cloud_artist.set_array(shown[:, 2])
            self.point_key = point_key
            self.invalidate_background()
        self.listing.clear()
        visible = list(self.visible_candidates())
        if self.selection and self.selection[0] == 'candidate' and int(self.selection[1]) not in {i for i, _, _ in visible}:
            self.selection = None
        segments, colors, widths, styles = [], [], [], []
        while len(self.text_artists) < len(visible):
            self.text_artists.append(axes.annotate('', xy=(0, 0), xytext=(5, 5), textcoords='offset points',
                fontsize=8, animated=True, clip_on=True,
                bbox=dict(facecolor=BACKGROUND, edgecolor='none', alpha=.85, pad=.8)))
        for text in self.text_artists:
            text.set_visible(False)
        for position, (ident, box, state) in enumerate(visible):
            color = COLORS['reference'] if state == 'accepted' else COLORS[state]
            corners = box_corners(box)
            selected = self.selection == ('candidate', str(ident))
            segments.append(corners[[0, 1, 2, 3, 0], :2])
            colors.append(color)
            widths.append(2.5 if selected else 1.4)
            styles.append('dashed' if state in ('accepted', 'reference') else 'solid')
            text = self.text_artists[position]
            text.xy = box[:2]
            text.set_text('%d %.2f' % (ident, box[7]))
            text.set_color(color)
            text.set_visible(True)
            item = QtWidgets.QListWidgetItem('建议 %d · %.3f · %s' % (ident, box[7], NAMES[state]))
            item.setData(QtCore.Qt.UserRole, ('candidate', str(ident)))
            item.setForeground(QtGui.QColor(color))
            self.listing.addItem(item)
        self.box_artist.set_segments(segments)
        self.box_artist.set_color(colors)
        self.box_artist.set_linewidth(widths)
        self.box_artist.set_linestyle(styles)
        centers = np.asarray([c['xy'] for c in self.review['centers']], dtype=float).reshape(-1, 2)
        self.centers_artist.set_data(centers[:, 0], centers[:, 1])
        selected_xy = self.selected_xy()
        # Refresh after drag/undo, but leave unsubmitted numeric edits intact when
        # only a display filter or view mode changed.
        binding = (self.frame['id'], self.selection, tuple(selected_xy)) if selected_xy is not None else None
        if binding != self.xy_binding:
            self.xy_binding = binding
            if selected_xy is not None:
                self.xvalue.setValue(selected_xy[0])
                self.yvalue.setValue(selected_xy[1])
        self.selected_artist.set_data([selected_xy[0]], [selected_xy[1]]) if selected_xy is not None else self.selected_artist.set_data([], [])
        movable = bool(self.selection and self.selection[0] == 'center' and selected_xy is not None and not self.loading and self.editable())
        for widget in (self.xvalue, self.yvalue, self.move_button):
            widget.setEnabled(movable)
        for center in self.review['centers']:
            xy = center['xy']
            item = QtWidgets.QListWidgetItem('中心 %.3f, %.3f%s' % (*xy, ' · 补漏' if center['candidate'] is None else ''))
            item.setData(QtCore.Qt.UserRole, ('center', center['id']))
            item.setForeground(QtGui.QColor(COLORS['accepted']))
            self.listing.addItem(item)
        for index in range(self.listing.count()):
            item = self.listing.item(index)
            if tuple(item.data(QtCore.Qt.UserRole)) == self.selection:
                self.listing.setCurrentRow(index)
                self.listing.scrollToItem(item)
                break
        if self.view_key != self.frame['source_id']:
            axes.set_xlim(b[0] - .5, b[3] + .5)
            axes.set_ylim(b[1] - .5, b[4] + .5)
        self.view_key = self.frame['source_id']
        self.selection_artist.set_data([], [])
        self.draw_overlays()
        source = self.sources[self.frame['source_id']]
        progress = self.progress[self.frame['source_id']]
        pending = sum(v in ('pending', 'uncertain') for v in self.review['decisions'].values())
        seconds = (self.frame['record_ns'] - self.beginnings[self.frame['source_id']]) / 1e9
        state_name = FRAME_STATUS_NAMES[self.review['status']]
        display_status = review_display_status(self.review)
        if display_status == 'complete_empty':
            state_name += '（无人）'
        color = ANNOTATION_COLORS.get(display_status, '#8b98a5')
        self.caption.setStyleSheet('border-left: 4px solid %s; padding-left: 6px;' % color)
        self.caption.setText('%s · %.2f s · 帧 %d / %d · %s · 人工中心 %d · 待复核 %d' %
            (Path(source['path']).name, seconds, self.frame['ordinal'], progress['total'],
             state_name, len(self.review['centers']), pending))
        self.caption.setToolTip('点云 stamp=%d ns · 坐标系 %s\n本包标注帧：待标 %d / 跳过 %d / 已标 %d' %
            (self.frame['stamp_ns'], self.frame['sensor_frame'], progress['draft'], progress['skipped'], progress['complete']))
        for status, button in list(self.state_buttons.items()) + [('complete_empty', self.empty_button)]:
            shade = QtGui.QColor(ANNOTATION_COLORS[status])
            button.setStyleSheet('border: 2px solid %s; background: %s; border-radius: 3px; padding: 4px;' %
                                (shade.darker(150).name(), shade.lighter(170).name())
                                if status == display_status else '')
        self.render_aux()

    def selected_xy(self):
        if not self.selection:
            return None
        kind, ident = self.selection
        if kind == 'candidate':
            return self.candidates()[int(ident)][:2]
        return next((c['xy'] for c in self.review['centers'] if c['id'] == ident), None)

    def render_aux(self):
        if self.frame is None or self.points is None:
            return
        rgb_key = self.frame['id'], self.image_ns
        if rgb_key != self.rgb_key:
            self.rgb_key = rgb_key
            self.rgb.show_image(self.image)
            if self.image is not None:
                self.rgb.setToolTip('RGB stamp=%d ns；落后点云 %.1f ms' % (self.image_ns, (self.frame['stamp_ns'] - self.image_ns) / 1e6))
        xy = self.selected_xy() if self.local_view.isChecked() else None
        key = (self.point_key, tuple(xy) if xy is not None else None, self.selection,
               tuple((i, state) for i, _, state in self.visible_candidates()),
               tuple((center['id'], tuple(center['xy'])) for center in self.review['centers']))
        if key == self.aux_key:
            return
        self.aux_key = key
        points = self.points[np.isfinite(self.points).all(axis=1)]
        points = points[(points[:, 2] >= self.zlow.value()) & (points[:, 2] <= self.zhigh.value())]
        if xy is not None:
            points = points[np.linalg.norm(points[:, :2] - xy, axis=1) <= 2.5]
        else:
            b = self.bounds
            points = points[(points[:, 0] >= b[0]) & (points[:, 0] <= b[3]) & (points[:, 1] >= b[1]) & (points[:, 1] <= b[4])]
        shown = points[::max(1, math.ceil(len(points) / 20000))]
        b = self.bounds
        point_colors = height_colors(shown[:, 2], b[2], b[5])
        lines, line_colors = [], []
        # A fixed sensor-coordinate reference, not a ground-height estimate.
        z = b[2]
        xmin, ymin, xmax, ymax = (xy[0] - 2.5, xy[1] - 2.5, xy[0] + 2.5, xy[1] + 2.5) if xy is not None else (b[0], b[1], b[3], b[4])
        for x in np.arange(math.ceil(xmin), xmax + .001, 1.):
            lines.extend([[x, ymin, z], [x, ymax, z]])
            line_colors.extend([[.22, .29, .35]] * 2)
        for y in np.arange(math.ceil(ymin), ymax + .001, 1.):
            lines.extend([[xmin, y, z], [xmax, y, z]])
            line_colors.extend([[.22, .29, .35]] * 2)
        reference_vertices = len(lines)
        for ident, box, state in self.visible_candidates():
            if xy is not None and np.linalg.norm(np.asarray(box[:2]) - xy) > 3:
                continue
            corners = box_corners(box)
            color = '#ffffff' if self.selection == ('candidate', str(ident)) else COLORS['reference' if state == 'accepted' else state]
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                line = corners[[a, b]]
                lines.extend(line)
                line_colors.extend([QtGui.QColor(color).getRgbF()[:3]] * 2)
        for center in self.review['centers']:
            cx, cy = center['xy']
            if xy is not None and np.linalg.norm(np.asarray([cx, cy]) - xy) > 3:
                continue
            selected = self.selection == ('center', center['id'])
            for radius, color in ([(.36, '#ffffff'), (.24, COLORS['accepted'])] if selected else [(.24, COLORS['accepted'])]):
                lines.extend([[cx - radius, cy, z + .03], [cx + radius, cy, z + .03],
                              [cx, cy - radius, z + .03], [cx, cy + radius, z + .03]])
                line_colors.extend([QtGui.QColor(color).getRgbF()[:3]] * 4)
        # Colored sensor axes are a display reference, not an estimated pose.
        for endpoint, color in [([1, 0, 0], [1., .3, .3]), ([0, 1, 0], [.3, 1., .3]), ([0, 0, 1], [.3, .6, 1.])]:
            lines.extend([[0, 0, 0], endpoint])
            line_colors.extend([color, color])
        focus = [xy[0], xy[1], .5] if xy is not None else [(self.bounds[0] + self.bounds[3]) / 2, 0, .5]
        target = ''
        if self.selection:
            target = ' · 候选 ' + self.selection[1] if self.selection[0] == 'candidate' else ' · 人工中心'
        self.cloud_caption.setText('三维点云 · ' + ('局部半径 2.5 m' if xy is not None else '全局范围') + target)
        self.grid_caption.setText('网格 1 m · 参考平面 Z=%.1f m · 绿十字仅表示人工 XY' % z)
        camera_key = (self.frame['source_id'], tuple(xy) if xy is not None else None)
        self.cloud_view.scene(shown[:, :3], point_colors, np.asarray(lines).reshape(-1, 3),
                              np.asarray(line_colors).reshape(-1, 3), focus, 7 if xy is not None else 32,
                              camera_key, reference_vertices)

    def camera(self, elevation, azimuth):
        self.cloud_view.camera(elevation, azimuth)

    def reset_view(self):
        self.view_key = None
        self.selection = None
        self.cloud_view.camera(35, -60)
        self.cloud_view.canvas.key = None
        self.aux_key = None
        self.render()

    def list_clicked(self, item):
        self.select(tuple(item.data(QtCore.Qt.UserRole)))

    def select_list_item(self, direction):
        if self.frame is None or self.loading or not self.listing.count():
            return
        current = self.listing.currentRow()
        if current < 0:
            index = 0 if direction > 0 else self.listing.count() - 1
        else:
            index = max(0, min(self.listing.count() - 1, current + direction))
        self.select(tuple(self.listing.item(index).data(QtCore.Qt.UserRole)))

    def cancel_selection(self):
        self.adding, self.dragging, self.drag_position = False, False, None
        self.selection = None
        self.render()

    def select(self, selection):
        self.selection = selection
        self.xy_binding = None
        self.render()

    def press(self, event):
        if self.frame is None or self.loading or event.inaxes != self.axes or event.button != 1 or self.toolbar.mode:
            return
        self.canvas.setFocus()
        if self.adding:
            self.change('add', xy=[event.xdata, event.ydata])
            self.adding = False
            return
        choices = [(('center', c['id']), c['xy']) for c in self.review['centers']]
        choices += [(('candidate', str(i)), box[:2]) for i, box, _ in self.visible_candidates()]
        choices = sorted(((np.linalg.norm(self.axes.transData.transform(xy) - [event.x, event.y]), selection)
                          for selection, xy in choices), key=lambda item: item[0])
        self.select(choices[0][1] if choices and choices[0][0] < 15 else None)
        self.dragging = bool(self.selection and self.selection[0] == 'center')
        self.drag_position = None

    def motion(self, event):
        if self.dragging and event.inaxes == self.axes:
            self.drag_position = [event.xdata, event.ydata]
            self.selection_artist.set_data([event.xdata], [event.ydata])
            self.draw_overlays()

    def release(self, event):
        if self.dragging and self.drag_position is not None and self.selection:
            self.change('move', center=self.selection[1], xy=self.drag_position)
        self.dragging, self.drag_position = False, None
        self.selection_artist.set_data([], [])
        self.draw_overlays()

    def scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        scale = 1.2 ** (-event.step)
        for get, set_, anchor in ((self.axes.get_xlim, self.axes.set_xlim, event.xdata),
                                  (self.axes.get_ylim, self.axes.set_ylim, event.ydata)):
            limits = get()
            set_(*[anchor + (v - anchor) * scale for v in limits])
        self.canvas.draw_idle()

    def apply(self, review, history=True, stage_note=False):
        if not self.editable() and not (stage_note and self.remote and not self.backend.explicit_read_only):
            self.message('当前不可编辑；请检查只读模式或同步冲突。')
            return False
        if self.loading or self.resetting or self.closing or self.frame is None or self.points is None:
            return
        if review == self.review:
            return True
        expected = copy.deepcopy(self.review)
        if history:
            self.undo_stack.append(expected)
            self.redo_stack.clear()
        self.review = copy.deepcopy(review)
        if review.get('note', '') != expected.get('note', ''):
            self.note.setText(review.get('note', ''))
        self.sync_status(dict(self.frame, review=self.review))
        if not self.save_error:
            self.queue_review(self.review, expected)
        self.render()
        self.update_save_notice()
        self.message(('修改保留在窗口，尚未保存 · ' if self.save_error else '正在自动保存 · ') + FRAME_STATUS_NAMES[review['status']])
        return True

    def change(self, operation, **values):
        if self.frame is None:
            return
        try:
            return self.apply(edit_review(self.review, operation, self.bounds, self.candidates(), **values))
        except (ValueError, IndexError) as error:
            self.error(str(error))

    def accept(self):
        if self.selection and self.selection[0] == 'candidate':
            ident = self.selection[1]
            self.change('decision', candidate=ident, decision='accepted')
            center = next((c for c in self.review['centers'] if c['candidate'] == ident), None)
            if center is not None:
                self.select(('center', center['id']))

    def delete(self):
        if self.selection:
            kind, ident = self.selection
            if kind == 'candidate':
                self.change('decision', candidate=ident, decision='rejected')
            else:
                self.change('delete', center=ident)
                self.selection = None

    def reset_candidate(self):
        if not self.selection or self.frame is None:
            return
        kind, ident = self.selection
        if kind == 'center':
            center = next((c for c in self.review['centers'] if c['id'] == ident), None)
            ident = center['candidate'] if center is not None else None
        if ident is None:
            self.message('人工补点请用R删除；C用于撤销候选的核验决定')
            return
        self.change('decision', candidate=ident, decision='pending')
        self.select(('candidate', ident))

    def add_mode(self):
        if not self.editable():
            return
        self.adding = True
        mode = str(self.toolbar.mode).lower()
        if 'pan' in mode:
            self.toolbar.pan()
        elif 'zoom' in mode:
            self.toolbar.zoom()
        self.canvas.setFocus()
        self.message('补漏模式：在 BEV 点击人体水平中心；范围外点不会保存')

    def move_numeric(self):
        if self.selection and self.selection[0] == 'center':
            self.change('move', center=self.selection[1], xy=[self.xvalue.value(), self.yvalue.value()])
        else:
            self.message('先选择人工中心；建议需确认后再修改 XY')

    def accept_remaining(self):
        if not self.frame or self.loading:
            return
        ids = pending_candidate_ids(self.review, self.candidates(), self.score.value())
        if not ids:
            self.message('当前阈值下没有待确认候选；低分隐藏项不会自动处理')
            return
        review = self.review
        for ident in ids:
            review = edit_review(review, 'decision', self.bounds, self.candidates(), candidate=ident, decision='accepted')
        self.apply(review)

    def save_note(self):
        if self.frame and self.note.text() != self.review.get('note', ''):
            try:
                review = edit_review(self.review, 'note', self.bounds, note=self.note.text())
                return self.apply(review, stage_note=True)
            except ValueError as error:
                self.error(str(error))
                return False
        return True

    def mark_status(self, status):
        if not self.editable():
            self.message('当前不可编辑；请检查只读模式或同步冲突。')
            return
        if self.frame is None or self.points is None or self.loading:
            return
        if not self.save_note():
            return
        try:
            if status == 'complete':
                review = complete_review(self.review, self.bounds, self.candidates(), self.score.value())
            elif status == 'empty':
                review = empty_review(self.review, self.bounds, self.candidates())
            elif status in ('draft', 'skipped'):
                operation = 'pending' if status == 'draft' else 'skip'
                review = edit_review(self.review, operation, self.bounds)
            else:
                raise ValueError('未知标注帧状态')
        except ValueError as error:
            self.message(str(error))
            return
        if status == 'empty':
            self.selection = None
            self.adding, self.dragging, self.drag_position = False, False, None
        self.apply(review)

    def clear_sequence_marks(self):
        if not self.editable():
            return
        if self.source_id is None or self.loading or self.closing:
            return
        if not self.after_saved(self.clear_sequence_marks):
            return
        ident = self.source_id
        counts = self.progress[ident]
        marked = sum(counts[state] for state in ANNOTATION_STATES)
        name = Path(self.sources[ident]['path']).name
        reply = QtWidgets.QMessageBox.question(self, '清空当前序列所有标记',
            '清空子包 %s 的全部标记和人工标注？\n'
            '共 %d 帧，其中 %d 个标注帧；包含当前时间范围之外的帧。\n'
            '将清除状态、人工中心、候选确认／拒绝和备注，恢复为非标注帧及原检测建议。\n'
            '保留检测缓存、原始数据和其他子包。此操作不能通过撤销恢复。' % (name, counts['total'], marked),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        if self.source_id != ident or self.loading:
            self.message('当前序列已改变，没有执行清空')
            return
        current = self.frame['id'] if self.frame is not None else None
        self.resetting = True
        self.centralWidget().setEnabled(False)
        self.focus_changed(None, None)
        self.queue_write(lambda: self.backend.reset_source_reviews(ident, current),
                         lambda result: self.sequence_cleared(ident, name, result), kind='reset')

    def sequence_cleared(self, ident, name, result):
        self.resetting = False
        if self.remote:
            self.remote_refresh_seen = self.backend.status()['refresh_serial']
            self.remote_refresh_requested = self.remote_refresh_seen
        self.centralWidget().setEnabled(True)
        self.focus_changed(None, QtWidgets.QApplication.focusWidget())
        for row in self.all_rows_by_source[ident]:
            row['status'] = 'unreviewed'
        self.progress[ident] = dict(total=result['frames'], unreviewed=result['frames'], draft=0, skipped=0, complete=0)
        self.sequence.setItemText(self.sequence.findData(ident), self.sequence_label(ident))
        self.timeline.set_frames(self.all_rows_by_source[ident], [row['id'] for row in self.rows])
        self.undo_stack, self.redo_stack = [], []
        self.selection, self.adding, self.dragging = None, False, False
        self.aux_key = None
        if self.frame is not None:
            self.frame = result['frame']
            self.review = copy.deepcopy(self.frame['review'])
            self.note.setText(self.review.get('note', ''))
            self.timeline.set_current(self.frame['id'])
            self.render()
        self.message('已清空 %s 的 %d 帧人工结果；检测缓存完整保留' % (name, result['cleared']))

    def undo(self):
        if self.undo_stack and not self.loading and self.editable():
            self.redo_stack.append(copy.deepcopy(self.review))
            self.apply(self.undo_stack.pop(), history=False)

    def redo(self):
        if self.redo_stack and not self.loading and self.editable():
            self.undo_stack.append(copy.deepcopy(self.review))
            self.apply(self.redo_stack.pop(), history=False)

    def closeEvent(self, event):
        if self.backend_closed:
            event.accept()
            return
        event.ignore()
        if self.synchronizing:
            self.message('正在完成本次同步操作，稍后可关闭。')
            return
        if not self.closing:
            if not self.loading and not self.resetting:
                self.save_note()
            if self.save_error:
                self.error('当前帧有未保存修改。请重试保存，或明确放弃未保存修改并重读后再关闭。')
                return
            self.close_requested = True
            if self.write_task is not None or self.writes:
                self.message('正在保存，完成后退出……')
                return  # Writes must finish and report failures; never cancel them.
            if self.remote and not self.remote_exit_confirmed and self.backend.status()['pending']:
                reply = QtWidgets.QMessageBox.question(self, '尚有待同步编辑',
                    '编辑已在本机暂存，但尚未全部保存到服务器。仍要退出？\n'
                    '下次使用相同地址、工程和缓存目录打开，将先查询原操作回执后恢复同步。',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if reply != QtWidgets.QMessageBox.Yes:
                    self.close_requested = False
                    return
                self.remote_exit_confirmed = True
            self.closing = True
            self.centralWidget().setEnabled(False)
            self.focus_changed(None, None)
        if self.jobs:
            for task in self.jobs:
                task.cancelled = True  # Only reads/prefetch remain at this point.
            QtCore.QTimer.singleShot(100, self.close)
            return
        task = Job(lambda progress: self.backend.close(), self)
        self.jobs.append(task)
        task.error.connect(self.error)

        def closed():
            self.finish_job(task)
            self.backend_closed = True
            self.close()

        task.finished.connect(closed)
        task.start()


def run(args, backend=None):
    app = QtWidgets.QApplication.instance()
    if app is None:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        # CloudView uses desktop OpenGL 2.1's fixed-function API. Windows Qt's
        # ANGLE/GLES backend cannot supply those calls; honor an explicit override.
        if os.name == 'nt' and getattr(args, 'cloud_renderer', 'auto') != 'software' and not os.environ.get('QT_OPENGL'):
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseDesktopOpenGL, True)
        app = QtWidgets.QApplication([])
    window = Window(args, backend=backend)
    window.show()
    return app.exec_()
