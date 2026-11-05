"""Independent Qt T1/T2 editor. All I/O is serialized outside the UI thread."""
from collections import OrderedDict, deque
from bisect import bisect_left, bisect_right
import copy
from datetime import datetime, timezone
import getpass
import html
import json
from pathlib import Path
import sys
import time
import uuid

from PyQt5 import QtCore, QtGui, QtWidgets

from .backend import LocalBackend, load_view
from .cloud_view import CloudView
from .rules import (CONTINUITY, COVERAGE, MATCH, RELATIONS, STATE_NAMES, VISIBILITY,
                    person_at, t1_empty, t2_empty, validate_t1, validate_t2)
from .store import Conflict, new_event
from .views import ImageThumb, ObservationCard, PersonView, Timeline, bev_context, context_extent, observation_regions,prepare_view


NAVIGATION_KEYS = {QtCore.Qt.Key_A, QtCore.Qt.Key_D, QtCore.Qt.Key_W, QtCore.Qt.Key_S}
RANGE_KEYS = {QtCore.Qt.Key_I, QtCore.Qt.Key_O}
GROUP_COLORS = ('#8854b8', '#258b52', '#bd647e', '#977514', '#366fc6',
                '#137f8c', '#b24c24', '#6b7832', '#626598', '#9a4f84')
VIEW_CACHE_BYTES = 128*1024*1024
PREPARED_CACHE_BYTES = 384*1024*1024


class Job(QtCore.QThread):
    result = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function())
        except Exception as error:
            self.failed.emit(error)


def button(text, function, layout):
    widget = QtWidgets.QPushButton(text)
    widget.clicked.connect(function)
    layout.addWidget(widget)
    return widget


def combo(choices):
    widget = QtWidgets.QComboBox()
    for key, name in choices.items():
        widget.addItem(name, key)
    return widget


class Window(QtWidgets.QMainWindow):
    def __init__(self, factory):
        super().__init__()
        self.factory, self.backend = factory, None
        self.jobs, self.active_job = deque(), None
        self.fetch_job=None
        self.snapshot_data = dict(sources=[], events=[])
        self.rows, self.frame_index, self.view_data = [], 0, None
        self.frame_positions={}
        self.event, self.revisions, self.edit_serial = None, {}, {}
        self.selected, self.undo_stack, self.redo_stack = [], [], []
        self.generation, self.rendering, self.saving, self.dirty = 0, False, 0, False
        self.writable, self.paused, self.closing, self.final_close = False, False, False, False
        self.event_editor_id, self.event_editor_source, self.updating_range = None, None, False
        self.event_valid_range = False
        self.deleting = None
        self.view_cache, self.view_cache_bytes = OrderedDict(),0
        self.prepared_cache,self.prepared_bytes = OrderedDict(),0
        self.recent_views=OrderedDict()
        self.resume_cursor=None
        self.active_kind, self.prefetch_error = None,None
        self.play_anchor, self.play_buffering = None,False
        self.play_advancing = False
        self.setWindowTitle('Social-Nav · T1/T2决策标注')
        self.resize(1480, 940)
        self.settings = QtCore.QSettings(QtCore.QSettings.defaultFormat(), QtCore.QSettings.UserScope,
                                         'SocialNav', 'vlm_annotation')
        self.make_ui()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.heartbeat = QtCore.QTimer(self)
        self.heartbeat.timeout.connect(self.renew)
        self.heartbeat.start(25000)
        self.playback = QtCore.QTimer(self)
        self.playback.setInterval(16)
        self.playback.timeout.connect(self.play_tick)
        self.task('连接工程', self.connect_backend, self.connected)

    def make_ui(self):
        base = QtWidgets.QWidget()
        self.setCentralWidget(base)
        layout = QtWidgets.QVBoxLayout(base)
        bar = QtWidgets.QHBoxLayout()
        self.sources, self.events = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.sources.setMinimumWidth(170)
        self.events.setMinimumWidth(190)
        self.events.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.events.setMinimumContentsLength(16)
        self.sources.currentIndexChanged.connect(self.source_changed)
        self.events.currentIndexChanged.connect(self.event_changed)
        bar.addWidget(QtWidgets.QLabel('子bag'))
        bar.addWidget(self.sources)
        bar.addWidget(QtWidgets.QLabel('事件'))
        bar.addWidget(self.events)
        self.add_event_button = QtWidgets.QToolButton()
        self.add_event_button.setText('新建事件')
        self.add_event_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.add_event_button.clicked.connect(self.add_event)
        event_menu = QtWidgets.QMenu(self.add_event_button)
        event_menu.addAction('整包创建／选择', self.create_full_event)
        event_menu.addAction('从当前帧到包尾', self.select_remaining_event)
        self.add_event_button.setMenu(event_menu)
        bar.addWidget(self.add_event_button)
        self.event_settings_button = button('编辑事件', self.event_settings, bar)
        bar.addStretch()
        self.reviewer = QtWidgets.QLineEdit(self.settings.value('reviewer', getpass.getuser()))
        self.reviewer.setMaximumWidth(110)
        bar.addWidget(QtWidgets.QLabel('审核者'))
        bar.addWidget(self.reviewer)
        button('同步／重连', self.reconnect, bar)
        button('处理同步冲突…', self.handle_conflict, bar)
        button('导出本机草稿', self.export_draft, bar)
        layout.addLayout(bar)
        modes = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem('上下文浏览（可看完整事件）', False)
        self.mode.addItem('输入审核（仅当前因果窗口）', True)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        modes.addWidget(self.mode)
        self.layout_choice = QtWidgets.QComboBox()
        self.layout_choice.addItems(['三帧放大','三维＋三帧同屏','三维大图'])
        self.layout_choice.setToolTip('三帧放大：RGB／BEV上下排；双击查看原始模型组合图')
        self.layout_choice.currentIndexChanged.connect(self.apply_view_layout)
        modes.addWidget(self.layout_choice)
        self.limit_event = QtWidgets.QCheckBox('仅浏览当前事件')
        self.limit_event.toggled.connect(self.scope_changed)
        modes.addWidget(self.limit_event)
        self.position = QtWidgets.QLabel('正在连接…')
        self.position.setWordWrap(True)
        self.position.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Preferred)
        modes.addWidget(self.position, 1)
        modes.addWidget(QtWidgets.QLabel('点云大小'))
        self.point_size = QtWidgets.QSpinBox()
        self.point_size.setRange(1, 5)
        self.point_size.setValue(int(self.settings.value('point_size', 3)))
        self.point_size.valueChanged.connect(self.point_changed)
        modes.addWidget(self.point_size)
        button('复位视图', self.fit_views, modes)
        self.top_view_button = button('点云俯视', lambda: self.cloud.top_view(), modes)
        layout.addLayout(modes)
        split = self.visual_split = QtWidgets.QSplitter()
        self.visual_stack = QtWidgets.QStackedWidget()
        self.context_views = QtWidgets.QTabWidget()
        self.cloud = CloudView()
        self.cloud.set_point_size(self.point_size.value())
        self.cloud.picked.connect(self.pick)
        self.rgb, self.bev = PersonView(), PersonView(bev=True)
        self.bev.point_size = self.point_size.value()
        self.rgb.picked.connect(self.pick)
        self.bev.picked.connect(self.pick)
        self.rgb.box_drawn.connect(self.add_auxiliary)
        self.context_views.addTab(self.cloud, '三维点云')
        self.context_views.addTab(self.rgb, '当前RGB／补人物')
        self.context_views.addTab(self.bev, '当前BEV')
        self.visual_stack.addWidget(self.context_views)
        self.audit_view = ImageThumb()
        self.audit_view.picked.connect(self.pick)
        self.audit_view.setMinimumSize(300,240)
        self.audit_view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.visual_stack.addWidget(self.audit_view)
        split.addWidget(self.visual_stack)
        history_column = self.history_panel = QtWidgets.QWidget()
        history_column.setMinimumWidth(280)
        hl = QtWidgets.QVBoxLayout(history_column)
        hl.setContentsMargins(0,0,0,0)
        title = self.history_title = QtWidgets.QLabel('连续三帧 · 当前RGB／BEV点击选人，Ctrl多选 · 双击查看中性原图')
        title.setWordWrap(True)
        hl.addWidget(title)
        self.history, self.history_captions = [], []
        cards = self.history_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
        hl.addLayout(cards,1)
        for index, offset in enumerate(('t − 1 s', 't − 0.5 s', '当前 t')):
            card = QtWidgets.QWidget()
            cl = QtWidgets.QVBoxLayout(card);cl.setContentsMargins(0,0,0,0)
            caption = QtWidgets.QLabel(offset)
            view = ObservationCard()
            view.context_bev.setInteractive(index == 2)
            view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
            cl.addWidget(caption);cl.addWidget(view,1)
            self.history.append(view);self.history_captions.append(caption)
            for thumb in (view.whole,view.rgb,view.bev):
                thumb.original_loader = lambda i=index:self.inspect_observation(i)
            cards.addWidget(card,1)
        self.history[2].picked.connect(self.pick)
        self.context_views.currentChanged.connect(lambda _:self.render_scene())
        split.addWidget(history_column)
        editor = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(editor);el.setContentsMargins(0,0,0,0)
        editor_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(300)
        self.tabs.currentChanged.connect(self.task_changed)
        self.make_t1()
        self.make_t2()
        editor_split.addWidget(self.tabs)
        people_panel = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(people_panel);ml.setContentsMargins(0,0,0,0)
        ml.addWidget(QtWidgets.QLabel('当前人物 · 与图像／点云联动选择'))
        self.group_summary = QtWidgets.QLabel('当前帧群组：暂无')
        self.group_summary.setWordWrap(True)
        self.group_summary.setMaximumHeight(76)
        self.group_summary.setStyleSheet('background:#f5f7fa;padding:3px;border:1px solid #d2dde8')
        ml.addWidget(self.group_summary)
        self.people = QtWidgets.QListWidget()
        self.people.setMinimumHeight(90)
        self.people.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.people.itemSelectionChanged.connect(self.selection_changed)
        ml.addWidget(self.people, 1)
        identities = QtWidgets.QHBoxLayout()
        self.person_button = button('新建人物', self.add_person, identities)
        self.link_button = button('关联／拆分片段', self.link_person, identities)
        self.aux_button = button('补RGB人物', self.start_auxiliary, identities)
        ml.addLayout(identities)
        editor_split.addWidget(people_panel)
        editor_split.setSizes([470,170])
        el.addWidget(editor_split)
        split.addWidget(editor)
        split.setSizes([700,400,330])
        layout.addWidget(split, 1)
        self.timeline = Timeline()
        self.timeline.jumped.connect(self.jump_time)
        self.timeline.range_changed.connect(self.select_event_range)
        self.timeline.range_selected.connect(self.finish_event_range)
        self.timeline.event_selected.connect(self.select_event)
        self.timeline.event_edited.connect(self.edit_timeline_event)
        self.make_event_editor(layout)
        layout.addWidget(self.timeline)
        navigation = QtWidgets.QHBoxLayout()
        button('上一原始帧 [A]', lambda: self.navigate(-1), navigation)
        button('下一原始帧 [D]', lambda: self.navigate(1), navigation)
        button('上一采样 [W]', lambda: self.navigate_sample(-1), navigation)
        button('下一采样 [S]', lambda: self.navigate_sample(1), navigation)
        button('播放／暂停 [空格]', self.toggle_play, navigation)
        self.sample_button = button('加入样本', self.add_sample, navigation)
        self.batch_button = button('区间取样', self.batch_samples, navigation)
        navigation.addWidget(QtWidgets.QLabel('间隔秒'))
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(.1, 60)
        self.interval.setValue(2)
        self.interval.valueChanged.connect(lambda _:self.schedule_prefetch())
        navigation.addWidget(self.interval)
        layout.addLayout(navigation)
        states = QtWidgets.QHBoxLayout()
        self.state_buttons = [button(text, lambda checked=False, s=state: self.mark(s), states)
                              for text, state in (('待标 [1]', 'draft'), ('跳过 [2]', 'skipped'), ('已标 [3]', 'complete'))]
        self.clear_button = button('清空当前任务标注…', self.clear_task, states)
        button('撤销', self.undo, states)
        button('重做', self.redo, states)
        button('帮助 [F1]', self.help, states)
        states.addStretch()
        layout.addLayout(states)
        self.message = QtWidgets.QLabel('正在连接工程')
        self.message.setWordWrap(True)
        self.message.setMinimumHeight(36)
        layout.addWidget(self.message)
        self.task_widgets = [self.person_button, self.link_button, self.aux_button, self.sample_button,
                             self.batch_button, self.clear_button, *self.state_buttons]
        self.context_views.currentChanged.connect(self.apply_view_layout)
        self.apply_view_layout()

    def make_event_editor(self, layout):
        self.event_editor = QtWidgets.QFrame()
        self.event_editor.setStyleSheet('QFrame#eventEditor {background:#fff7e6;border:1px solid #cfad68}')
        self.event_editor.setObjectName('eventEditor')
        panel = QtWidgets.QVBoxLayout(self.event_editor)
        panel.setContentsMargins(6,4,6,4)
        fields = QtWidgets.QHBoxLayout()
        fields.addWidget(QtWidgets.QLabel('名称'))
        self.event_name = QtWidgets.QLineEdit()
        self.event_name.setToolTip('留空时按子包与时间范围自动命名')
        fields.addWidget(self.event_name,1)
        self.event_start, self.event_end = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for title, widget in (('起始秒',self.event_start),('结束秒',self.event_end)):
            fields.addWidget(QtWidgets.QLabel(title))
            widget.setDecimals(3)
            widget.setSingleStep(.1)
            widget.valueChanged.connect(self.range_fields_changed)
            fields.addWidget(widget)
        fields.addWidget(QtWidgets.QLabel('用途'))
        self.event_split = combo(dict(unassigned='暂不划分',development='开发',validation='验证',holdout='留出'))
        fields.addWidget(self.event_split)
        self.event_more = QtWidgets.QToolButton()
        self.event_more.setText('来源…')
        self.event_more.setCheckable(True)
        fields.addWidget(self.event_more)
        self.event_delete = button('删除事件…',self.delete_event,fields)
        self.event_delete.setStyleSheet('color:#a32d26')
        self.event_delete.setToolTip('删除当前事件及其T1/T2标注，需要确认；原始数据与其他事件保留')
        panel.addLayout(fields)
        actions = QtWidgets.QHBoxLayout()
        self.event_range_hint = QtWidgets.QLabel()
        actions.addWidget(self.event_range_hint,1)
        button('取起点 [I]',lambda:self.set_event_bound(0),actions)
        button('取终点 [O]',lambda:self.set_event_bound(1),actions)
        button('选中整包',self.select_full_range,actions)
        self.event_apply = button('创建事件 [Enter]',self.commit_event_editor,actions)
        self.event_cancel = button('取消 [Esc]',lambda:self.cancel_event_editor(),actions)
        panel.addLayout(actions)
        self.event_sources = QtWidgets.QWidget()
        sources = QtWidgets.QHBoxLayout(self.event_sources)
        sources.setContentsMargins(0,0,0,0)
        self.event_root, self.event_place = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for title, widget in (('采集根来源',self.event_root),('重复地点组',self.event_place)):
            sources.addWidget(QtWidgets.QLabel(title))
            sources.addWidget(widget,1)
        panel.addWidget(self.event_sources)
        self.event_sources.hide()
        self.event_more.toggled.connect(self.event_sources.setVisible)
        layout.addWidget(self.event_editor)
        self.event_editor.hide()

    def apply_view_layout(self):
        if not hasattr(self,'history_panel'):
            return
        choice = self.layout_choice.currentIndex()
        context_visible = choice!=0 and not self.mode.currentData()
        self.point_size.setEnabled(not self.mode.currentData())
        self.top_view_button.setEnabled(context_visible and self.context_views.currentIndex()==0)
        self.visual_stack.setVisible(choice!=0)
        self.history_panel.setVisible(choice!=2)
        self.history_layout.setDirection(QtWidgets.QBoxLayout.LeftToRight if choice==0 else QtWidgets.QBoxLayout.TopToBottom)
        for view in self.history:view.set_compact(choice!=0)
        self.visual_split.setSizes([0,1100,330] if choice==0 else [700,400,330] if choice==1 else [1100,0,330])
        if (self.view_data and choice!=0 and self.context_views.currentIndex()!=1 and
                not self.mode.currentData() and self.view_data.get('cloud3d') is None):
            self.goto(self.frame_index)
        elif self.view_data:
            self.render_scene()

    def make_t1(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(QtWidgets.QLabel('Ctrl＋点击／列表多选人物后建组'))
        self.relation = combo(RELATIONS)
        layout.addWidget(self.relation)
        self.activity = QtWidgets.QLineEdit()
        self.activity.setPlaceholderText('其他共同活动的简短说明')
        layout.addWidget(self.activity)
        self.roles_known = QtWidgets.QCheckBox('拍照角色可判（摄影者→被摄者）')
        layout.addWidget(self.roles_known)
        self.photographer, self.subject = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        layout.addWidget(self.photographer)
        layout.addWidget(self.subject)
        self.group_button = button('应用群组 [Q]', self.apply_group, layout)
        self.groups_title = QtWidgets.QLabel('当前帧群组 · 点击查看成员')
        layout.addWidget(self.groups_title)
        self.groups = QtWidgets.QListWidget()
        self.groups.setMinimumHeight(75)
        self.groups.itemClicked.connect(self.choose_group)
        layout.addWidget(self.groups)
        self.single_button = button('选中人物：确认未成组', lambda: self.classify('singles'), layout)
        self.unknown_button = button('选中人物：归属未知', lambda: self.classify('unknown'), layout)
        self.remove_button = button('清除选中人物分类 [R]', self.remove_classification, layout)
        self.copy_button = button('复制上一已标样本为草稿', self.copy_previous, layout)
        self.propagate_button = button('应用到指定区间为草稿…', self.propagate, layout)
        layout.addWidget(QtWidgets.QLabel('备注（离开编辑框后自动保存）'))
        self.t1_note = QtWidgets.QLineEdit()
        self.t1_note.editingFinished.connect(lambda: self.note_changed(False))
        layout.addWidget(self.t1_note)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, 'T1 群组')

    def make_t2(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.targets = QtWidgets.QComboBox()
        self.targets.currentIndexChanged.connect(self.target_changed)
        layout.addWidget(self.targets)
        self.target_button = button('建立／修改目标描述…', self.edit_target, layout)
        self.description = QtWidgets.QLabel('先创建人物关联，再建立目标任务；也支持无目标实例。')
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.t2_controls = {}
        form = QtWidgets.QFormLayout()
        for field, title, choices in (('visibility', '可见性', VISIBILITY), ('match', '匹配证据', MATCH),
                                      ('coverage', '感知覆盖', COVERAGE), ('continuity', '连续性', CONTINUITY)):
            widget = combo(choices)
            widget.currentIndexChanged.connect(self.evidence_changed)
            self.t2_controls[field] = widget
            form.addRow(title, widget)
        layout.addLayout(form)
        self.true_button = button('选中人物设为真实对应', self.set_true_key, layout)
        self.unset_true_button = button('清除当前真实对应', lambda: self.modify_label(lambda label: label.update(true_key=None)), layout)
        self.candidates_button = button('选中人物设为合理候选 [Q]', self.set_candidates, layout)
        self.clear_candidates_button = button('清空合理候选', lambda: self.modify_label(lambda label: label.update(plausible=[])), layout)
        self.target_evidence = QtWidgets.QLabel('')
        self.target_evidence.setWordWrap(True)
        layout.addWidget(self.target_evidence)
        self.flags = QtWidgets.QLineEdit()
        self.flags.setPlaceholderText('事件标记，逗号分隔：交叉、遮挡、重现…')
        self.flags.editingFinished.connect(self.flags_changed)
        layout.addWidget(self.flags)
        self.t2_note = QtWidgets.QLineEdit()
        self.t2_note.setPlaceholderText('证据／歧义说明')
        self.t2_note.editingFinished.connect(lambda: self.note_changed(True))
        layout.addWidget(self.t2_note)
        layout.addWidget(QtWidgets.QLabel('这里只标固定事实。v/o/a/l/n需在真实回放历史接入后审核。'))
        layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, 'T2 身份')

    def task(self, name, function, success=None, kind='normal'):
        if kind == 'view':
            self.jobs = deque(j for j in self.jobs if j[3] not in ('view', 'prefetch'))
        self.jobs.append((name, function, success, kind))
        self.start_next()

    def start_next(self):
        if self.active_job is not None or not self.jobs:
            return
        name, function, callback, kind = self.jobs.popleft()
        if self.paused and kind == 'save':
            self.saving -= 1
            self.start_next()
            return
        job = Job(function, self)
        job.view_generation=self.generation if kind=='view' else None
        self.active_job = job
        self.active_kind = kind
        job.result.connect(lambda result: self.job_result(job, callback, result))
        job.failed.connect(lambda error: self.job_failed(name,error,kind,job.view_generation))
        job.finished.connect(lambda: self.job_finished(job))
        job.start()

    def job_result(self, job, callback, result):
        if callback:
            try:
                callback(result)
            except Exception as error:
                self.error(error)

    def job_failed(self, name, error, kind, generation=None):
        if kind=='view' and generation is not None and generation!=self.generation:
            return
        if kind=='view' and self.view_data is None:
            self.position.setText('读取失败，请查看底部提示')
            self.group_summary.setText('当前帧读取失败')
        if kind == 'save':
            self.saving -= 1
            self.paused, self.dirty = True, True
        if kind == 'delete':
            self.deleting = None
            self.paused, self.dirty = True, True
        if isinstance(error, Conflict):
            self.paused, self.writable = True, False
        if kind != 'prefetch':
            self.error(str(name) + '：' + str(error))
        else:
            self.prefetch_error = error
            if self.playback.isActive():
                self.playback.stop()
                self.error('播放预读失败：'+str(error))

    def job_finished(self, job):
        self.active_job = None
        self.active_kind = None
        job.deleteLater()
        if self.final_close:
            QtCore.QTimer.singleShot(0, self.close)
            return
        self.start_next()
        if self.closing and self.active_job is None and not self.jobs and self.fetch_job is None:
            self.closing = False
            self.close()
        elif not self.closing:
            self.schedule_prefetch()

    def connect_backend(self):
        if self.backend is None:
            self.backend = self.factory()
        return self.backend.snapshot()

    def connected(self, snapshot):
        self.paused, self.dirty = False, False
        same=self.snapshot_data.get('identity') is not None and self.cache_identity(snapshot)==self.cache_identity()
        self.resume_cursor=(self.sources.currentData(),self.rows[self.frame_index]['id']) if same and self.rows else None
        self.snapshot_data = snapshot
        if not same:
            self.view_cache.clear();self.view_cache_bytes = 0
            self.prepared_cache.clear();self.prepared_bytes = 0
            self.recent_views.clear()
        self.prefetch_error = None
        self.writable = snapshot['writable']
        self.revisions = {r['document']['id']: r['revision'] for r in snapshot['events']}
        self.sources.blockSignals(True)
        old = self.sources.currentData()
        self.sources.clear()
        for source in snapshot['sources']:
            self.sources.addItem(source['name'], source['id'])
        index = self.sources.findData(old)
        self.sources.setCurrentIndex(max(0, index))
        self.sources.blockSignals(False)
        self.source_changed()
        self.message.setText(snapshot.get('notice') or ('已连接，可编辑' if self.writable else '只读浏览'))
        self.enable_editing()

    def cache_identity(self, snapshot=None):
        state=self.snapshot_data if snapshot is None else snapshot
        return tuple(state.get(key) for key in ('identity','asset_identity','input_implementation','display_implementation'))

    def source_changed(self):
        source = self.sources.currentData()
        if not source or self.backend is None:
            return
        self.sources.setToolTip(self.sources.currentText())
        self.view_data = None
        self.rows,self.frame_index = [],0
        self.cancel_event_editor(restore_scope=False)
        self.timeline.show_state([],None,None,0)
        self.cloud.fit()
        self.playback.stop()
        self.history[2].set_interaction()
        self.audit_view.set_interaction()
        self.generation += 1
        self.task('读取序列索引', lambda: (source, self.backend.frames(source)), self.source_loaded)

    def source_loaded(self, result):
        source, rows = result
        if source != self.sources.currentData():
            return
        self.rows = rows
        self.frame_positions={row['id']:i for i,row in enumerate(rows)}
        if self.resume_cursor and self.resume_cursor[0]==source:
            self.frame_index=self.frame_positions.get(self.resume_cursor[1],0)
            self.resume_cursor=None
        old = self.event['id'] if self.event and self.event['source_id'] == source else None
        self.events.blockSignals(True)
        self.events.clear()
        self.events.addItem('整包浏览（未选事件）',None)
        for item in self.snapshot_data['events']:
            event = item['document']
            if event['source_id'] == source:
                self.events.addItem(event['name'], event['id'])
        index = self.events.findData(old) if old is not None else -1
        self.events.setCurrentIndex(index if index>=0 else 1 if self.events.count()>1 else 0)
        self.events.blockSignals(False)
        self.event_changed()

    def event_changed(self):
        self.cancel_event_editor(restore_scope=False)
        ident = self.events.currentData()
        self.event = next((copy.deepcopy(e['document']) for e in self.snapshot_data['events'] if e['document']['id'] == ident), None)
        self.events.setToolTip(f'{self.event["name"]} · {self.event["start"]:.3f}–{self.event["end"]:.3f}s' if self.event else '未选择事件，可浏览整包')
        self.undo_stack, self.redo_stack = [], []
        self.refresh_targets()
        if self.rows:
            current = self.rows[min(self.frame_index,len(self.rows)-1)]
            if self.navigation_event() and not self.in_event(current['elapsed']):
                index = next((i for i,row in enumerate(self.rows) if row['elapsed']>=self.event['start']-.001),0)
                self.goto(index)
            elif self.view_data is not None and self.view_data['frame']['source_id']==self.sources.currentData():
                self.selected = []
                self.render_scene()
            else:
                self.goto(min(self.frame_index,len(self.rows)-1))
        self.enable_editing()

    def source(self):
        return next(s for s in self.snapshot_data['sources'] if s['id'] == self.sources.currentData())

    def in_event(self, elapsed):
        return bool(self.event and self.event['start']-.001<=elapsed<=self.event['end']+.001)

    def navigation_event(self):
        return self.event if self.limit_event.isChecked() and not self.event_editor.isVisible() else None

    def begin_event_editor(self, edit=False, bounds=None):
        if not self.rows or (edit and not self.event):
            return
        self.playback.stop()
        self.event_editor_source = self.sources.currentData()
        self.event_editor_id = self.event['id'] if edit else None
        same_source = [r['document'] for r in self.snapshot_data['events'] if r['document']['source_id']==self.event_editor_source]
        seed = self.event or next(iter(same_source),{})
        current = max(0,self.rows[self.frame_index]['elapsed'])
        a,b = bounds if bounds is not None else (seed['start'],seed['end']) if edit else (current,current)
        self.updating_range = True
        for widget in (self.event_start,self.event_end):
            widget.setRange(max(0,self.rows[0]['elapsed']),max(0,self.rows[-1]['elapsed']))
        self.event_start.setValue(a);self.event_end.setValue(b)
        self.event_name.setText(seed['name'] if edit else '')
        self.event_root.setText(seed.get('root_group') or self.source().get('root_group') or self.event_editor_source)
        self.event_place.setText(seed.get('location_group',''))
        self.event_split.setCurrentIndex(max(0,self.event_split.findData(seed.get('split','unassigned'))))
        self.event_more.setChecked(False)
        self.event_apply.setText('保存修改 [Enter]' if edit else '创建事件 [Enter]')
        self.event_delete.setVisible(edit)
        self.updating_range = False
        self.event_editor.show()
        if self.mode.currentData():
            self.mode.setCurrentIndex(0)
        self.range_fields_changed()
        self.render_labels()

    def add_event(self):
        self.begin_event_editor()

    def event_settings(self):
        self.begin_event_editor(edit=True)

    def delete_event(self):
        if not self.event_writable() or self.event_editor_id != self.event['id']:
            return
        event = self.event
        ident = event['id']
        count = sum(len(target.get('samples',{})) for target in event.get('targets',{}).values())
        message = (f'删除事件“{event["name"]}”（{event["start"]:.3f}–{event["end"]:.3f}s）？\n\n'
                   f'同时删除 {len(event.get("t1",{}))} 个T1样本、{len(event.get("targets",{}))} 个T2任务'
                   f'及其 {count} 个样本，以及本事件的人物关联和辅助标注。\n'
                   '原始bag、感知结果和其他事件保留。此操作不能用Ctrl+Z撤销。')
        if QtWidgets.QMessageBox.question(self,'删除事件',message,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        if not self.event_writable() or self.event['id'] != ident or self.event_editor_id != ident:
            return
        self.playback.stop()
        self.deleting = ident
        self.enable_editing()
        self.message.setText('正在删除事件…')
        def deleted(result):
            self.deleting = None
            self.snapshot_data['events'] = [row for row in self.snapshot_data['events'] if row['document']['id']!=ident]
            self.revisions.pop(ident,None)
            self.edit_serial.pop(ident,None)
            index = self.events.findData(ident)
            selected = self.events.currentData()==ident
            if index>=0:
                self.events.blockSignals(True)
                self.events.removeItem(index)
                if selected:
                    self.events.setCurrentIndex(0)
                self.events.blockSignals(False)
            if selected:
                self.event_changed()
            else:
                self.render_labels()
                self.enable_editing()
            self.message.setText('删除请求已在本机暂存／待同步' if result.get('staged') else '事件已删除')
        self.task('删除事件',lambda:self.backend.delete_event(ident,self.revisions.get(ident,0)),deleted,kind='delete')

    def range_fields_changed(self):
        if self.updating_range or not self.event_editor.isVisible():
            return
        a,b = self.event_start.value(),self.event_end.value()
        count = max(0,bisect_right(self.timeline.times,b+.001)-bisect_left(self.timeline.times,a-.001)) if a<=b else 0
        self.event_valid_range = a<=b and count>0
        self.event_name.setPlaceholderText(f'{self.source()["name"]} {a:.3f}–{b:.3f}s')
        prefix = '修改范围' if self.event_editor_id else '临时选区'
        hint = f'{prefix} {a:.3f}–{b:.3f}s · {count}帧' if self.event_valid_range else '起点不能晚于终点，范围内须有可用帧'
        self.event_range_hint.setText(hint+(' · 只读，不能保存' if not self.writable else ''))
        self.timeline.set_selection(tuple(sorted((a,b))))
        self.enable_editing()

    def select_event_range(self, start, end):
        if not self.rows:
            return
        if not self.event_editor.isVisible():
            self.begin_event_editor(bounds=(start,end))
        else:
            self.updating_range = True
            self.event_start.setValue(start);self.event_end.setValue(end)
            self.updating_range = False
            self.range_fields_changed()

    def finish_event_range(self, start, end):
        self.select_event_range(start,end)
        self.jump_time(end)

    def set_event_bound(self, index):
        if not self.rows:
            return
        if not self.event_editor.isVisible():
            self.begin_event_editor()
        current = max(0,round(self.rows[self.frame_index]['elapsed'],3))
        a,b = self.event_start.value(),self.event_end.value()
        self.select_event_range(current,max(current,b)) if index==0 else self.select_event_range(min(a,current),current)

    def select_full_range(self):
        if self.rows:
            self.select_event_range(max(0,self.rows[0]['elapsed']),self.rows[-1]['elapsed'])

    def select_remaining_event(self):
        if self.rows:
            self.begin_event_editor(bounds=(max(0,self.rows[self.frame_index]['elapsed']),self.rows[-1]['elapsed']))

    def create_full_event(self):
        if not self.rows or not self.writable or self.paused or self.deleting:
            return
        a,b = max(0,self.rows[0]['elapsed']),self.rows[-1]['elapsed']
        existing = next((r['document'] for r in self.snapshot_data['events'] if r['document']['source_id']==self.sources.currentData()
                         and r['document']['start']<=a+.001 and r['document']['end']>=b-.001),None)
        if existing:
            self.cancel_event_editor(restore_scope=False)
            self.select_event(existing['id'])
            self.message.setText('已选择已有整包事件')
        else:
            self.begin_event_editor(bounds=(a,b))
            self.commit_event_editor()

    def commit_event_editor(self):
        if not self.event_editor.isVisible() or not self.writable or self.paused or self.deleting:
            return
        self.event_start.interpretText();self.event_end.interpretText()
        if not self.event_valid_range or self.event_editor_source!=self.sources.currentData():
            self.error('请选择当前子包内包含可用帧的有效范围')
            return
        values = dict(start=self.event_start.value(),end=self.event_end.value(),
                      name=self.event_name.text().strip() or self.event_name.placeholderText(),
                      root_group=self.event_root.text().strip(),location_group=self.event_place.text().strip(),
                      split=self.event_split.currentData())
        if self.event_editor_id is not None:
            if not self.event or self.event['id']!=self.event_editor_id:
                self.error('当前事件已改变，请重新选择范围')
                return
            used = set(self.event['t1']) | set(self.event.get('auxiliary',{}))
            used.update(t['reference_frame'] for t in self.event['targets'].values())
            used.update(ident for t in self.event['targets'].values() for ident in t['samples'])
            if any(r['id'] in used and not values['start']-.001<=r['elapsed']<=values['end']+.001 for r in self.rows):
                self.error('新范围会排除已有标注或目标参考帧；请先调整这些内容，或新建事件')
                return
            self.cancel_event_editor(restore_scope=False)
            self.mutate(lambda e:e.update(values),event_metadata=True)
            self.scope_changed()
        else:
            event = new_event(self.source(),values['start'],values['end'],values['name'])
            event.update(values)
            self.cancel_event_editor(restore_scope=False)
            self.event = event
            self.revisions[event['id']] = self.edit_serial[event['id']] = 0
            self.update_snapshot(event)
            self.events.addItem(event['name'],event['id'])
            self.events.setCurrentIndex(self.events.findData(event['id']))
            self.dirty = True
            self.save()

    def cancel_event_editor(self, restore_scope=True):
        self.event_editor.hide()
        self.event_editor_id,self.event_editor_source = None,None
        self.event_valid_range = False
        self.timeline.set_selection(None)
        self.enable_editing()
        if restore_scope:
            self.scope_changed()

    def select_event(self, ident):
        index = self.events.findData(ident)
        if index>=0:
            self.events.setCurrentIndex(index)

    def edit_timeline_event(self, ident):
        self.select_event(ident)
        self.event_settings()

    def scope_changed(self):
        event = self.navigation_event()
        if event and self.rows and not self.in_event(self.rows[self.frame_index]['elapsed']):
            self.jump_time(self.rows[self.frame_index]['elapsed'])
        else:
            self.render_labels()

    def update_snapshot(self, event):
        found = next((r for r in self.snapshot_data['events'] if r['document']['id'] == event['id']), None)
        row = dict(document=copy.deepcopy(event), revision=self.revisions.get(event['id'], 0))
        if found is None:
            self.snapshot_data['events'].append(row)
        else:
            found.update(row)
        index = self.events.findData(event['id'])
        if index>=0:
            self.events.setItemText(index,event['name'])
            if self.events.currentData()==event['id']:
                self.events.setToolTip(f'{event["name"]} · {event["start"]:.3f}–{event["end"]:.3f}s')

    def refresh_targets(self):
        current = self.targets.currentData()
        self.targets.blockSignals(True)
        self.targets.clear()
        if self.event:
            for tid, target in self.event.get('targets', {}).items():
                self.targets.addItem(target['description'][:35], tid)
        self.targets.setCurrentIndex(max(0, self.targets.findData(current)))
        self.targets.blockSignals(False)
        self.target_changed()

    def target(self):
        return self.event.get('targets', {}).get(self.targets.currentData()) if self.event else None

    def target_changed(self):
        target = self.target()
        self.description.setText(target['description'] if target else '尚未建立目标任务')
        self.render_labels()

    def task_changed(self):
        if not hasattr(self, 'task_widgets'):
            return
        self.render_labels()
        self.enable_editing()

    def mode_changed(self):
        self.playback.stop()
        self.view_data = None
        self.visual_stack.setCurrentIndex(1 if self.mode.currentData() else 0)
        self.layout_choice.model().item(2).setEnabled(not self.mode.currentData())
        if self.mode.currentData() and self.layout_choice.currentIndex()==2:
            self.layout_choice.blockSignals(True)
            self.layout_choice.setCurrentIndex(0)
            self.layout_choice.blockSignals(False)
        self.apply_view_layout()
        if self.rows:
            self.goto(self.frame_index)

    def goto(self, index):
        if not self.rows:
            return
        if self.playback.isActive() and not self.play_advancing:
            self.playback.stop()
        self.frame_index = max(0, min(len(self.rows)-1, index))
        self.prefetch_error = None
        self.generation += 1
        generation, frame_id, review = self.generation, self.rows[self.frame_index]['id'], self.mode.currentData()
        cached = self.view_cache.get((frame_id,bool(review)))
        if cached is not None and (frame_id,bool(review),self.point_size.value()) in self.prepared_cache:
            self.jobs = deque(job for job in self.jobs if job[3] not in ('view','prefetch'))
            self.view_cache.move_to_end((frame_id,bool(review)))
            self.selected = []
            self.view_loaded(generation,cached[0])
            return
        self.view_data = None
        self.history[2].set_interaction()
        for card in self.history:
            card.context_bev.setEnabled(False)
            if review:
                card.show_context(None)
                card.set_interaction()
        self.audit_view.set_interaction()
        self.group_summary.setText('当前帧群组：准备画面…' if cached is not None else '当前帧群组：读取中…')
        self.cloud.setEnabled(False)
        if self.mode.currentData():
            self.audit_view.show_data(None)
        self.selected = []
        self.enable_editing()
        self.position.setText(('准备缓存画面 ' if cached is not None else '读取观察 ')+f'{self.rows[self.frame_index]["elapsed"]:.3f}s…')
        size=self.point_size.value()
        identity=self.cache_identity()
        def read():
            view=cached[0] if cached is not None else load_view(self.backend,frame_id,review)
            return view,prepare_view(view,size)
        def loaded(result):
            if identity!=self.cache_identity():return
            view,prepared=result
            self.cache_prepared(view,size,prepared)
            self.view_loaded(generation,view)
        self.task('读取观察',read,loaded,kind='view')

    def view_loaded(self, generation, result):
        if generation != self.generation:
            return
        self.view_data = result
        visited=(result['frame']['id'],result['review'])
        self.recent_views[visited]=None
        self.recent_views.move_to_end(visited)
        while len(self.recent_views)>8:self.recent_views.popitem(last=False)
        self.cache_view(result)
        prepared=self.prepared_cache.get((result['frame']['id'],result['review'],self.point_size.value()))
        if prepared:
            self.prepared_cache.move_to_end((result['frame']['id'],result['review'],self.point_size.value()))
        frame = result['frame']
        self.position.setText(f'{frame["elapsed"]:.3f}s · 原始帧{frame["ordinal"]} · '
                              + ('三帧输入完整' if result['window']['ready'] else '历史／RGB不足'))
        for i, (widget, data) in enumerate(zip(self.history, result['histories'])):
            widget.show_data(data,QtGui.QPixmap.fromImage(prepared[0]['images'][i]) if prepared else None)
            offset = result['window']['fields']['dt'][i]
            self.history_captions[i].setText(('当前 t' if i==2 else f'历史 {i+1}') +
                (f' · {offset:+.3f}s' if offset is not None else ' · 缺少因果观察') +
                (' · 轨迹不可用' if result['window'].get('trajectories',{}).get(result['window']['frames'][i],{}).get('status')!='available' else ''))
        if self.mode.currentData():
            self.audit_view.show_data(result['histories'][2],QtGui.QPixmap.fromImage(prepared[0]['images'][2]) if prepared else None)
        self.cloud.setEnabled(not self.mode.currentData())
        self.render_scene()
        self.enable_editing()
        self.schedule_prefetch()

    def cache_view(self, view):
        key = (view['frame']['id'],view['review'])
        assets = [view['image'],*view['histories'],*(row['cloud'] for row in view['context_frames'] if row)]
        weight = sum(len(raw) for raw in assets if raw is not None)+128*1024
        previous = self.view_cache.pop(key,None)
        self.view_cache_bytes += weight-(previous[1] if previous else 0)
        self.view_cache[key] = (view,weight)
        while len(self.view_cache)>128 or self.view_cache_bytes>VIEW_CACHE_BYTES:
            _,size = self.view_cache.pop(self.cache_victim(self.view_cache))
            self.view_cache_bytes -= size

    def cache_prepared(self, view, size, prepared):
        key=(view['frame']['id'],view['review'],size)
        images=prepared['images']+[row[1] for row in prepared['scans'] if row]
        weight=sum(image.sizeInBytes() for image in images)
        previous=self.prepared_cache.pop(key,None)
        self.prepared_bytes+=weight-(previous[1] if previous else 0)
        self.prepared_cache[key]=prepared,weight
        while len(self.prepared_cache)>64 or self.prepared_bytes>PREPARED_CACHE_BYTES:
            _,old=self.prepared_cache.pop(self.cache_victim(self.prepared_cache))
            self.prepared_bytes-=old

    def cache_victim(self, cache):
        for key in cache:
            if len(key)>2 and key[2]!=self.point_size.value():return key
        if not self.playback.isActive():
            budget=PREPARED_CACHE_BYTES if cache is self.prepared_cache else VIEW_CACHE_BYTES
            capacity=budget//max((row[1] for row in cache.values()),default=1)
            protected=set(list(self.recent_views)[-min(8,max(1,capacity//3)):])
            current=(self.rows[self.frame_index]['id'],bool(self.mode.currentData())) if self.rows else None
            candidates=[key for key in cache if key[:2]!=current]
            for key in candidates:
                if key[:2] not in protected:return key
            return next(iter(candidates or cache))
        # Recently displayed past frames must not evict the very next playback frame.
        for key in cache:
            if (self.frame_positions.get(key[0],-1)<self.frame_index or key[1] or
                    len(key)>2 and key[2]!=self.point_size.value()):
                return key
        return max(cache,key=lambda key:self.frame_positions.get(key[0],len(self.rows)))

    def buffer_horizon(self):
        size=max((row[1] for row in self.prepared_cache.values()),default=8*1024*1024)
        return min(16,max(2,PREPARED_CACHE_BYTES//max(size,1)-2))

    def prefetch_rows(self):
        event=self.navigation_event()
        if self.playback.isActive():
            indices=range(self.frame_index+1,self.frame_index+1+self.buffer_horizon())
            limit=self.buffer_horizon()
        else:
            size=max((row[1] for row in self.prepared_cache.values()),default=8*1024*1024)
            capacity=min(64,PREPARED_CACHE_BYTES//max(size,1))
            reserve=min(8,max(1,capacity//3))
            limit=min(16,max(0,capacity-reserve-1))
            indices=[self.frame_index+1,self.frame_index-1,self.sample_index(1),self.sample_index(-1)]
            indices.extend(self.frame_index+step for step in range(2,9))
            indices.extend(self.frame_index-step for step in range(2,limit+2))
            indices.extend(self.frame_index+step for step in range(9,limit+2))
        rows=[];seen=set()
        for index in indices:
            if len(rows)>=limit:break
            if index is None or index in seen or not 0<=index<len(self.rows):continue
            row=self.rows[index]
            if event and not event['start']-.001<=row['elapsed']<=event['end']+.001:continue
            rows.append(row);seen.add(index)
        return rows

    def schedule_prefetch(self):
        if (not self.backend or not self.view_data or self.mode.currentData() or self.closing or self.final_close
                or self.prefetch_error or self.jobs or self.active_kind in ('save','delete','normal')):
            return
        rows=self.prefetch_rows()
        ahead=len(rows)
        size=self.point_size.value()
        threshold=min(8,max(1,ahead//2))
        missing=[row['id'] for row in rows if (row['id'],False) not in self.view_cache]
        ready=sum((row['id'],False,size) in self.prepared_cache for row in rows[:threshold+1])
        batch=8 if self.playback.isActive() else 4
        if self.fetch_job is None and missing and (not self.playback.isActive() or len(missing)>=4 or ready<=threshold):
            ids=missing[:batch]
            identity=self.cache_identity()
            job=Job(lambda:self.backend.views(ids,False),self)
            self.fetch_job=job
            job.result.connect(lambda views:[self.cache_view(view) for view in views] if identity==self.cache_identity() else None)
            job.failed.connect(lambda error:self.job_failed('预读后续观察',error,'prefetch'))
            job.finished.connect(lambda:self.fetch_finished(job))
            job.start()
        # Fetch the following batch while this interpreter decodes the previous one.
        # Both stages are bounded; edits still use the serialized foreground queue.
        if self.active_kind!='prefetch':
            views=[self.view_cache[(row['id'],False)][0] for row in rows if (row['id'],False) in self.view_cache
                   and (row['id'],False,size) not in self.prepared_cache][:4]
            if views:
                identity=self.cache_identity()
                def prepared(results):
                    if identity!=self.cache_identity():return
                    for view,image in results:self.cache_prepared(view,size,image)
                self.task('预解码后续观察',lambda:[(view,prepare_view(view,size)) for view in views],prepared,kind='prefetch')

    def fetch_finished(self, job):
        self.fetch_job=None
        job.deleteLater()
        if self.closing and self.active_job is None and not self.jobs:
            self.closing=False
            self.close()
        elif not self.closing:
            self.schedule_prefetch()

    def inspect_observation(self, index):
        if not self.view_data:
            return
        self.playback.stop()
        ident,generation = self.view_data['frame']['id'],self.generation
        self.message.setText('正在读取无损模型图…')
        def show(views):
            if generation==self.generation:
                self.message.setText('无损模型图已读取')
                self.history[index].rgb.show_original(views[0]['histories'][index])
        self.task('读取无损模型图',lambda:self.backend.views([ident],True),show)

    def scene_people(self):
        if not self.view_data:
            return []
        people = copy.deepcopy(self.view_data['frame']['people'])
        if self.event:
            for person in self.event.get('auxiliary', {}).get(self.view_data['frame']['id'], []):
                people.append(dict(key=person['key'], rgb_box=person['rect'], box=None, eligible=False, kind='manual'))
        return people

    def label_for_key(self, key):
        mapping = self.view_data['window']['mapping'] if self.view_data else {}
        people=self.view_data['frame']['people'] if self.view_data else []
        if self.event and self.view_data:
            people=people+self.event.get('auxiliary',{}).get(self.view_data['frame']['id'],[])
        prefix = 'P'+str(mapping[key]) if key in mapping else '参考'+str(next((i+1 for i,p in enumerate(people) if p['key']==key), '?'))
        if self.event and self.view_data and not self.mode.currentData():
            person = person_at(self.event, key, self.view_data['frame']['stamp_ns'])
            if person:
                prefix += ' / '+self.event['persons'][person]['name']
        return prefix

    def render_scene(self):
        if not self.view_data:
            return
        people = self.scene_people()
        if self.mode.currentData():
            people = [p for p in people if p['key'] in self.view_data['window']['mapping']]
        names = {p['key']: self.label_for_key(p['key']) for p in people}
        mapping = self.view_data['window']['mapping']
        context = bev_context(self.view_data.get('context_frames', []), mapping)
        extent = context_extent(context)
        prepared=self.prepared_cache.get((self.view_data['frame']['id'],self.view_data['review'],self.point_size.value()))
        for index, card in enumerate(self.history):
            row = context[index] if context else None
            card.context_bev.point_size = self.point_size.value()
            card.context_bev.setEnabled(not self.mode.currentData())
            if row is not None and self.history_panel.isVisible():
                scan=prepared[0]['scans'][index] if prepared and prepared[0]['scans'] else None
                card.context_bev.prepared_cloud=(row['cloud'],*scan) if scan else None
                displayed = row['frame']['people']
                labels = names if index==2 else {p['key']: 'P'+str(mapping[p['key']]) if p['key'] in mapping
                                                else '参考'+str(p['bev_reference']) for p in displayed}
                if index!=2:
                    card.context_bev.group_colors = {}
                card.show_context(row, displayed, labels, extent)
            else:
                card.show_context(None)
        self.rgb.image_size = self.view_data['frame']['image_size']
        if self.rgb.isVisible():
            self.rgb.show_frame(self.view_data['image'], None, people, names)
        if self.bev.isVisible():
            self.bev.show_frame(None, self.view_data['cloud'], context[-1]['frame']['people'] if context else people, names,
                            context[-1]['trajectory'] if context else self.view_data['window'].get('trajectories',{}).get(self.view_data['frame']['id']),
                            extent, self.view_data['frame'].get('camera_fov'))
        if self.cloud.isVisible():
            self.cloud.show_frame(self.view_data['cloud3d'], people, names)
        self.people.blockSignals(True)
        self.people.clear()
        for p in people:
            status = '' if p.get('eligible') else ' · 参考／无模型PID'
            item = QtWidgets.QListWidgetItem(names[p['key']]+status)
            item.setData(QtCore.Qt.UserRole, p['key'])
            item.setToolTip(p['key'])
            self.people.addItem(item)
            item.setSelected(p['key'] in self.selected)
        self.people.blockSignals(False)
        self.highlight()
        self.render_labels()

    def pick(self, key, multiple):
        if not self.view_data or key not in {p['key'] for p in self.scene_people()}:
            return
        if multiple:
            self.selected = [k for k in self.selected if k != key] if key in self.selected else self.selected+[key]
        else:
            self.selected = [key]
        self.people.blockSignals(True)
        for i in range(self.people.count()):
            item = self.people.item(i)
            item.setSelected(item.data(QtCore.Qt.UserRole) in self.selected)
        self.people.blockSignals(False)
        self.highlight()

    def selection_changed(self):
        self.selected = [i.data(QtCore.Qt.UserRole) for i in self.people.selectedItems()]
        self.highlight()

    def highlight(self):
        self.rgb.highlight(self.selected)
        self.bev.highlight(self.selected)
        self.cloud.highlight(self.selected)
        self.history[2].highlight(self.selected)
        self.audit_view.selected = set(self.selected)
        self.audit_view.update()
        old = [self.photographer.currentData(), self.subject.currentData()]
        for index, widget in enumerate((self.photographer, self.subject)):
            widget.clear()
            for key in self.selected:
                widget.addItem(self.label_for_key(key), key)
            widget.setCurrentIndex(max(0, widget.findData(old[index])))
        if len(self.selected) >= 2 and self.photographer.currentData() == self.subject.currentData():
            self.subject.setCurrentIndex(1)

    def current_labels(self, event=None):
        event = event or self.event
        if self.tabs.currentIndex() == 0:
            return event['t1'] if event else None
        target = event.get('targets', {}).get(self.targets.currentData()) if event else None
        return target['samples'] if target else None

    def label(self):
        labels = self.current_labels()
        return labels.get(self.view_data['frame']['id']) if labels is not None and self.view_data else None

    def render_labels(self):
        if not hasattr(self, 'timeline'):
            return
        self.rendering = True
        label = self.label()
        self.groups.clear()
        t1 = self.event.get('t1', {}).get(self.view_data['frame']['id']) if self.event and self.view_data else None
        if t1:
            for number, group in enumerate(t1['groups'], 1):
                roles = ' · 角色已标' if group.get('roles') else ''
                item = QtWidgets.QListWidgetItem(f'G{number} · '+RELATIONS[group['relation']]+'：'+', '.join(self.label_for_key(k) for k in group['members'])+roles)
                item.setForeground(QtGui.QColor(GROUP_COLORS[(number-1)%len(GROUP_COLORS)]))
                item.setData(QtCore.Qt.UserRole, group)
                self.groups.addItem(item)
            for number, group in enumerate(t1.get('reference_groups', []), 1):
                item = QtWidgets.QListWidgetItem(f'参考R{number} · 覆盖失败：'+RELATIONS[group['relation']]+' · '+
                                                ', '.join(self.label_for_key(k) for k in group['members']))
                item.setData(QtCore.Qt.UserRole, group)
                self.groups.addItem(item)
            self.t1_note.setText(t1.get('note', ''))
        else:
            self.t1_note.clear()
        if self.tabs.currentIndex() == 1:
            evidence = label or t2_empty('')
            for field, widget in self.t2_controls.items():
                widget.setCurrentIndex(max(0, widget.findData(evidence[field])))
            self.t2_note.setText(evidence.get('note', ''))
            self.flags.setText(','.join(evidence.get('flags', [])))
            key = evidence.get('true_key')
            self.target_evidence.setText('真实对应：'+(self.label_for_key(key) if key else '未知／未绑定')+'\n合理候选：'+
                                         ', '.join(self.label_for_key(k) for k in evidence['plausible']))
        if self.view_data:
            state = STATE_NAMES[label['status']] if label else '非标注样本'
            verified=self.view_data['window'].get('input_verified',True)
            stale = verified and label and label.get('binding') and label['binding'] != self.view_data['window']['fingerprint']
            detection = self.view_data['frame'].get('perception_context', {}).get('detection_input')
            temporal = f' · 检测输入{detection["used_sweeps"]}/{detection["max_sweeps"]}帧' if detection else ''
            self.position.setText(f'{self.view_data["frame"]["elapsed"]:.3f}s · {state}' + temporal + (' · 输入变化，需重审' if stale else ''))
            if not verified:
                self.position.setText(self.position.text()+' · 浏览预览，审核时核对输入')
            if self.event_editor.isVisible():
                self.position.setText(self.position.text()+' · 临时选区，仅浏览')
            elif self.event and not self.in_event(self.view_data['frame']['elapsed']):
                self.position.setText(self.position.text()+' · 事件外，仅浏览')
            classifications, colors, summaries = {}, {}, []
            if t1:
                for number, group in enumerate(t1['groups'], 1):
                    color = GROUP_COLORS[(number-1)%len(GROUP_COLORS)]
                    name = f'G{number} · '+RELATIONS[group['relation']]
                    summaries.append((color, name+'：'+'、'.join(self.label_for_key(k) for k in group['members'])))
                    for key in group['members']:
                        classifications[key], colors[key] = name, color
                for number, group in enumerate(t1.get('reference_groups', []), 1):
                    name = f'参考R{number} · '+RELATIONS[group['relation']]
                    summaries.append(('#8c6d4f', name+'：'+'、'.join(self.label_for_key(k) for k in group['members'])))
                    for key in group['members']:
                        classifications[key], colors[key] = name, '#8c6d4f'
                classifications.update({k: '确认未成组' for k in t1['singles']})
                classifications.update({k: '归属未知' for k in t1['unknown']})
            projections = {p['key']: bool(p.get('rgb_box')) for p in self.scene_people()}
            for i in range(self.people.count()):
                item = self.people.item(i)
                key = item.data(QtCore.Qt.UserRole)
                normal = key in self.view_data['window']['mapping']
                status = classifications.get(key, '未分类' if normal else '参考／无模型PID')
                if normal and not projections.get(key):
                    status += ' · '+('无RGB匹配' if self.view_data['frame'].get('rgb_box_source')=='yolo_association' else '无RGB投影')
                item.setText(self.label_for_key(key)+' · '+status)
                item.setForeground(QtGui.QColor(colors.get(key, '#273d50')))
            self.group_summary.setText('<b>当前帧群组</b><br>'+'<br>'.join(
                f'<span style="color:{color}">● {html.escape(text)}</span>' for color,text in summaries) if summaries else '当前帧群组：暂无')
            self.group_summary.setToolTip('\n'.join(text for _,text in summaries) or '在T1中多选人物并应用群组；分组仅属于当前标注帧')
            self.groups_title.setText(f'当前帧群组 · {len(t1["groups"]) if t1 else 0}组 · 点击选择成员')
            regions = observation_regions(self.view_data['frame'], self.view_data['window'])
            self.history[2].set_interaction(regions, self.selected, colors, not self.mode.currentData())
            self.history[2].context_bev.group_colors = colors
            self.history[2].context_bev.highlight(self.selected)
            self.bev.group_colors = colors
            self.bev.highlight(self.selected)
            self.audit_view.set_interaction(regions, self.selected)
            self.history_title.setText('输入审核 · 中性模型图；当前RGB／BEV可点选，Ctrl多选；双击原图' if self.mode.currentData() else
                                       '上下文三帧 · BEV含当前目标及历史参考（非模型图）；当前可点选，Ctrl多选')
        events = [r['document'] for r in self.snapshot_data['events'] if r['document']['source_id']==self.sources.currentData()]
        self.timeline.show_state(self.rows, self.event, self.target(), self.rows[self.frame_index]['elapsed'] if self.rows else 0,events)
        self.rendering = False

    def event_writable(self):
        return bool(self.writable and not self.paused and not self.deleting and self.event and self.event['source_id']==self.sources.currentData())

    def editable(self):
        return bool(self.event_writable() and self.view_data and self.view_data['frame']['source_id']==self.event['source_id']
                    and self.in_event(self.view_data['frame']['elapsed']) and not self.event_editor.isVisible())

    def enable_editing(self):
        if not hasattr(self, 'task_widgets'):
            return
        enabled = self.editable()
        for widget in self.task_widgets:
            widget.setEnabled(enabled)
        self.add_event_button.setEnabled(bool(self.writable and self.rows and not self.paused and not self.deleting))
        self.event_settings_button.setEnabled(self.event_writable() and bool(self.rows))
        self.event_apply.setEnabled(bool(self.writable and not self.paused and not self.deleting and self.event_valid_range and self.event_editor.isVisible()))
        self.event_delete.setEnabled(bool(self.event_writable() and self.event_editor_id==self.event['id']))
        self.limit_event.setEnabled(bool(self.event) and not self.event_editor.isVisible())
        for widget in (self.group_button, self.single_button, self.unknown_button, self.remove_button,
                       self.copy_button, self.propagate_button, self.t1_note):
            widget.setEnabled(enabled)
        self.target_button.setEnabled(enabled)
        target_enabled = enabled and self.target() is not None
        for widget in (*self.t2_controls.values(), self.true_button, self.unset_true_button, self.candidates_button, self.clear_candidates_button,
                       self.t2_note, self.flags):
            widget.setEnabled(target_enabled)

    def mutate(self, function, record_undo=True, event_metadata=False):
        if not (self.event_writable() if event_metadata else self.editable()):
            return
        before = copy.deepcopy(self.event)
        event = copy.deepcopy(before)
        try:
            function(event)
        except (ValueError, KeyError) as error:
            self.error(error)
            return
        if event == before:
            return
        if record_undo:
            self.undo_stack.append(before)
            self.undo_stack = self.undo_stack[-40:]
            self.redo_stack.clear()
        self.event = event
        self.dirty = True
        self.update_snapshot(event)
        self.save()
        self.render_labels()

    def save(self):
        if not self.event:
            return
        document = copy.deepcopy(self.event)
        ident = document['id']
        serial = self.edit_serial.get(ident, 0)+1
        self.edit_serial[ident] = serial
        self.saving += 1
        self.message.setText('正在保存／本机暂存…')
        def write():
            return self.backend.commit(document, self.revisions.get(ident, 0))
        def saved(result):
            self.saving -= 1
            self.revisions[ident] = result['revision']
            if self.edit_serial[ident] == serial:
                if self.event and self.event['id'] == ident:
                    self.event = result['document']
                    self.render_scene()
                self.update_snapshot(result['document'])
            self.dirty = self.saving > 0
            self.message.setText('已在本机暂存／待同步' if result.get('staged') else '服务器已保存' if hasattr(self.backend, 'pending') else '本机已保存')
        self.task('保存标签', write, saved, kind='save')

    def modify_label(self, change, status=None):
        if not self.editable() or self.current_labels() is None:
            return
        ident, binding = self.view_data['frame']['id'], self.view_data['window']['fingerprint']
        if self.tabs.currentIndex() == 1:
            reference = next(r for r in self.rows if r['id'] == self.target()['reference_frame'])
            if self.view_data['frame']['stamp_ns'] < reference['stamp_ns']:
                self.error('当前观察早于目标描述参考时刻，不能建立回溯样本')
                return
        def apply(event):
            labels = self.current_labels(event)
            label = labels.setdefault(ident, t1_empty(binding) if self.tabs.currentIndex() == 0 else t2_empty(binding))
            change(label)
            label['status'] = status or 'draft'
            label['binding'] = binding
        self.mutate(apply)

    def mark(self, status):
        if self.event_editor.isVisible():
            self.error('请先创建／保存事件选区，或按Esc取消选区后标注')
            return
        if self.event and self.view_data and not self.in_event(self.view_data['frame']['elapsed']):
            self.error('当前帧在事件外；请新建事件或编辑范围后标注')
            return
        if not self.editable() or self.current_labels() is None:
            self.error('请先创建事件；T2还需要选择目标任务')
            return
        binding = self.view_data['window']['fingerprint']
        value = copy.deepcopy(self.label() or (t1_empty(binding) if self.tabs.currentIndex() == 0 else t2_empty(binding)))
        value.update(status=status, binding=binding)
        if status == 'complete':
            if not self.mode.currentData():
                self.error('请先切换到“输入审核”，核对模型实际观察窗口后再标为已标')
                return
            if not self.view_data['window']['ready']:
                self.error('当前历史／RGB不完整，请另选时刻或跳过')
                return
            if not self.view_data['window'].get('fingerprint') or self.view_data['window'].get('input_verified') is False:
                self.error('请等待完整输入审核图加载后再标为已标')
                return
            if not self.reviewer.text().strip():
                self.error('请填写审核者')
                return
            value.update(reviewer=self.reviewer.text().strip(), reviewed_at=datetime.now(timezone.utc).isoformat())
            try:
                if self.tabs.currentIndex() == 0:
                    validate_t1(value, self.view_data['window']['mapping'], [p['key'] for p in self.scene_people()])
                else:
                    validate_t2(value, self.view_data['window']['mapping'], [p['key'] for p in self.scene_people()], self.target(),
                                lambda key: person_at(self.event, key, self.view_data['frame']['stamp_ns']))
            except ValueError as error:
                self.error(error)
                return
        self.modify_label(lambda label: label.update(value), status=status)

    def add_sample(self):
        if self.label() is None:
            self.modify_label(lambda label: None)

    def batch_samples(self):
        if not self.editable() or self.current_labels() is None:
            return
        event, step = copy.deepcopy(self.event), self.interval.value()
        is_t1, target_id = self.tabs.currentIndex() == 0, self.targets.currentData()
        start = event['start']
        if self.tabs.currentIndex() == 1:
            ref = next(r for r in self.rows if r['id'] == self.target()['reference_frame'])
            start = max(start, ref['elapsed'])
        selected, cursor = [], start
        for row in self.rows:
            if row['elapsed'] >= cursor-.001 and row['elapsed'] <= event['end']+.001:
                selected.append(row['id'])
                cursor = row['elapsed']+step
        if not selected:
            return
        generation = self.edit_serial.get(event['id'], 0)
        def build():
            return [(ident, None) for ident in selected]  # Pending samples need no image decoding or binding yet.
        def apply(values):
            if not self.event or self.event['id'] != event['id'] or self.edit_serial.get(event['id'], 0) != generation:
                self.error('取样期间事件已变化，请重新执行')
                return
            def change(e):
                labels = e['t1'] if is_t1 else e['targets'][target_id]['samples']
                for ident, binding in values:
                    labels.setdefault(ident, t1_empty(binding) if is_t1 else t2_empty(binding))
            self.mutate(change)
        self.task('按区间建立待标样本', build, apply)

    @staticmethod
    def remove_keys(label, keys):
        for field in ('groups', 'reference_groups'):
            groups = []
            for group in label.get(field, []):
                group['members'] = [k for k in group['members'] if k not in keys]
                if set(group.get('roles', [])) & set(keys):
                    group['roles'] = []
                if len(group['members']) >= 2:
                    groups.append(group)
            label[field] = groups
        for field in ('singles', 'unknown'):
            label[field] = [k for k in label[field] if k not in keys]

    def eligible_selection(self):
        if not self.view_data:
            return None
        if not self.selected or any(k not in self.view_data['window']['mapping'] for k in self.selected):
            self.error('请只选择具有模型PID的人物；参考人物用于身份与覆盖说明')
            return None
        return list(self.selected)

    def apply_group(self):
        keys = list(self.selected)
        if not keys or len(keys) < 2:
            self.error('建立群组至少需要两名有效人物')
            return
        roles = [self.photographer.currentData(), self.subject.currentData()] if self.roles_known.isChecked() else []
        relation = self.relation.currentData()
        if roles and (relation != 'p' or None in roles or roles[0] == roles[1]):
            self.error('角色仅用于拍照，且摄影者、主要被摄者必须不同')
            return
        group = dict(members=keys, relation=relation, roles=roles, activity=self.activity.text().strip())
        reference = any(key not in self.view_data['window']['mapping'] for key in keys)
        def change(label):
            self.remove_keys(label, keys)
            label.setdefault('reference_groups' if reference else 'groups', []).append(group)
        self.modify_label(change)
        if reference:
            self.message.setText('已记录含漏检／无PID人物的参考群组；该样本保留为覆盖失败，不生成伪造的合法响应。')

    def classify(self, field):
        keys = self.eligible_selection()
        if keys:
            def change(label):
                self.remove_keys(label, keys)
                label[field].extend(keys)
            self.modify_label(change)

    def remove_classification(self):
        if self.tabs.currentIndex() == 0 and self.selected:
            keys = list(self.selected)
            self.modify_label(lambda label: self.remove_keys(label, keys))

    def choose_group(self, item):
        group = item.data(QtCore.Qt.UserRole)
        self.selected = list(group['members'])
        self.relation.setCurrentIndex(self.relation.findData(group['relation']))
        self.activity.setText(group.get('activity', ''))
        self.roles_known.setChecked(bool(group.get('roles')))
        self.render_scene()
        roles = group.get('roles', [])
        if roles:
            self.photographer.setCurrentIndex(self.photographer.findData(roles[0]))
            self.subject.setCurrentIndex(self.subject.findData(roles[1]))

    def note_changed(self, target):
        if self.rendering or self.tabs.currentIndex() != int(target):
            return
        text = self.t2_note.text() if target else self.t1_note.text()
        if text != (self.label() or {}).get('note', ''):
            self.modify_label(lambda label: label.update(note=text))

    def evidence_changed(self):
        if not self.rendering and self.tabs.currentIndex() == 1:
            values = {field: widget.currentData() for field, widget in self.t2_controls.items()}
            self.modify_label(lambda label: label.update(values))

    def flags_changed(self):
        if self.rendering:
            return
        values = [s.strip() for s in self.flags.text().replace('，', ',').split(',') if s.strip()]
        if values != (self.label() or {}).get('flags', []):
            self.modify_label(lambda label: label.update(flags=values))

    def set_true_key(self):
        if len(self.selected) != 1:
            self.error('请选择一个真实人物；无法确定时保留未知')
            return
        key = self.selected[0]
        self.modify_label(lambda label: label.update(true_key=key))

    def set_candidates(self):
        keys = self.eligible_selection()
        if keys:
            self.modify_label(lambda label: label.update(plausible=keys))

    def add_person(self):
        if not self.editable():
            return
        name, ok = QtWidgets.QInputDialog.getText(self, '新增物理人物', '人物代号（隐藏真值）', text='人物'+str(len(self.event['persons'])+1))
        if ok and name.strip():
            ident = uuid.uuid4().hex
            self.mutate(lambda e: e['persons'].update({ident: {'name': name.strip()}}))

    def link_person(self):
        if not self.editable() or len(self.selected) != 1 or not self.event['persons']:
            self.error('先新建人物，再选一条轨迹／一个辅助人物进行关联')
            return
        key = self.selected[0]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('核对人物关联区间；重叠旧关联会在边界拆分')
        form = QtWidgets.QFormLayout(dialog)
        person = combo({k: v['name'] for k, v in self.event['persons'].items()})
        start, end = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for widget in (start, end):
            widget.setDecimals(3)
            widget.setRange(self.event['start'], self.event['end'])
            widget.setValue(self.view_data['frame']['elapsed'])
        form.addRow('物理人物', person)
        form.addRow('起始秒', start)
        form.addRow('结束秒', end)
        form.addRow(QtWidgets.QLabel('默认只关联当前帧；扩展区间前请检查身份。'))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != dialog.Accepted or start.value() > end.value():
            return
        origin = self.rows[0]['stamp_ns']-round(self.rows[0]['elapsed']*1e9)
        # Spin boxes display milliseconds; include the current scan's exact stamp.
        a, b = origin+round(start.value()*1e9)-500000, origin+round(end.value()*1e9)+500000
        pid = person.currentData()
        def change(e):
            links = []
            for old in e['links']:
                if old['key'] != key or old['end_ns'] < a or old['start_ns'] > b:
                    links.append(old)
                else:
                    if old['start_ns'] < a:
                        links.append(dict(old, end_ns=a-1))
                    if old['end_ns'] > b:
                        links.append(dict(old, start_ns=b+1))
            links.append(dict(key=key, person_id=pid, start_ns=a, end_ns=b))
            e['links'] = links
            self.invalidate(e)
        self.mutate(change)

    @staticmethod
    def invalidate(event):
        labels = list(event['t1'].values())
        labels += [s for t in event['targets'].values() for s in t['samples'].values()]
        for label in labels:
            if label['status'] == 'complete':
                label['status'] = 'draft'

    def start_auxiliary(self):
        if self.editable() and not self.mode.currentData():
            self.context_views.setCurrentIndex(1)
            if self.layout_choice.currentIndex()==0:
                self.layout_choice.setCurrentIndex(2)
            self.rgb.draw_box = True
            self.message.setText('在RGB上拖出辅助人物框；只作隐藏身份／漏检说明，不进入模型输入。')

    def add_auxiliary(self, rect):
        if not self.editable():
            return
        ident = self.view_data['frame']['id']
        person = dict(key='aux:'+uuid.uuid4().hex, rect=rect)
        self.mutate(lambda e: e['auxiliary'].setdefault(ident, []).append(person))
        self.render_scene()

    def edit_target(self):
        if not self.editable():
            return
        target = self.target()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('目标任务：初始可见描述，禁止使用未来细节')
        form = QtWidgets.QFormLayout(dialog)
        description = QtWidgets.QPlainTextEdit(target['description'] if target else '')
        description.setMaximumHeight(100)
        provenance = QtWidgets.QLineEdit(target.get('description_source', '') if target else '')
        provenance.setPlaceholderText('隐藏元数据：当前参考帧／独立参考图来源，不发送给模型')
        person = combo({k: v['name'] for k, v in self.event['persons'].items()})
        absent = QtWidgets.QCheckBox('此事件中目标始终不在场（描述需有独立来源）')
        if target:
            absent.setChecked(target.get('absent', False))
            person.setCurrentIndex(max(0, person.findData(target.get('person_id'))))
        reference = QtWidgets.QComboBox()
        for row in self.rows:
            if self.event['start'] <= row['elapsed'] <= self.event['end']:
                reference.addItem(f'{row["elapsed"]:.3f}s', row['id'])
        reference.setCurrentIndex(max(0, reference.findData(target['reference_frame'] if target else self.view_data['frame']['id'])))
        form.addRow('描述', description)
        form.addRow('描述来源（隐藏）', provenance)
        form.addRow('隐藏真实人物', person)
        form.addRow('描述参考时刻', reference)
        form.addRow(absent)
        new_instance = QtWidgets.QCheckBox('另建一个目标任务（保留当前任务）')
        form.addRow(new_instance)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != dialog.Accepted:
            return
        if not description.toPlainText().strip() or not reference.currentData() or (not absent.isChecked() and not person.currentData()):
            self.error('请填写描述并选择人物；无目标实例可不选择人物')
            return
        if absent.isChecked() and not provenance.text().strip():
            self.error('无目标实例需要单独记录描述来源；不要把“目标不在场”等答案写进模型描述')
            return
        existing = target and not new_instance.isChecked()
        tid = target['id'] if existing else uuid.uuid4().hex
        value = dict(id=tid, description=description.toPlainText().strip(), person_id=None if absent.isChecked() else person.currentData(),
                     description_source=provenance.text().strip() or 'current_reference_frame',
                     reference_frame=reference.currentData(), absent=absent.isChecked(), samples=copy.deepcopy(target['samples']) if existing else {})
        if existing:
            ref_row = next(r for r in self.rows if r['id'] == reference.currentData())
            if any(r['id'] in value['samples'] and r['stamp_ns'] < ref_row['stamp_ns'] for r in self.rows):
                self.error('已有样本早于新描述参考时刻；请保留当前参考时刻或另建目标任务')
                return
        for label in value['samples'].values():
            if label['status'] == 'complete':
                label['status'] = 'draft'
        self.mutate(lambda e: e['targets'].update({tid: value}))
        self.refresh_targets()
        self.targets.setCurrentIndex(self.targets.findData(tid))
        self.enable_editing()

    def translated_label(self, source, source_row, destination, window):
        mapping = {}
        for key in set(source['singles']+source['unknown']+[k for g in source['groups'] for k in g['members']]):
            if key in window['mapping']:
                mapping[key] = key
                continue
            identity = person_at(self.event, key, source_row['stamp_ns'])
            matches = [k for k in window['mapping'] if identity and person_at(self.event, k, destination['stamp_ns']) == identity]
            if len(matches) == 1:
                mapping[key] = matches[0]
        result = t1_empty(window['fingerprint'])
        result['note'] = source.get('note', '')
        for field in ('singles', 'unknown'):
            result[field] = [mapping[k] for k in source[field] if k in mapping]
        for group in source['groups']:
            members = [mapping[k] for k in group['members'] if k in mapping]
            if len(members) >= 2:
                roles = [mapping[k] for k in group.get('roles', []) if k in mapping]
                result['groups'].append(dict(group, members=members, roles=roles if len(roles) == 2 else []))
        return result

    def copy_previous(self):
        if not self.editable() or self.tabs.currentIndex() != 0:
            return
        before = [r for r in self.rows if r['elapsed'] < self.view_data['frame']['elapsed'] and
                  self.event['t1'].get(r['id'], {}).get('status') == 'complete']
        if not before:
            self.error('前方没有已标T1样本')
            return
        row = before[-1]
        source = self.event['t1'][row['id']]
        value = self.translated_label(source, row, self.view_data['frame'], self.view_data['window'])
        self.modify_label(lambda label: label.update(value))

    def propagate(self):
        if not self.editable() or self.tabs.currentIndex() != 0 or self.label() is None:
            return
        end, ok = QtWidgets.QInputDialog.getDouble(self, '区间传播为待标', '应用到结束秒（从当前时刻开始）',
                                                 self.event['end'], self.view_data['frame']['elapsed'], self.event['end'], 3)
        if not ok:
            return
        if QtWidgets.QMessageBox.question(self, '覆盖区间样本', '按当前步长建立／覆盖区间T1草稿？已有人工内容会被替换，全部需要复核。') != QtWidgets.QMessageBox.Yes:
            return
        event_id, serial = self.event['id'], self.edit_serial.get(self.event['id'], 0)
        source, first = copy.deepcopy(self.label()), self.view_data['frame']
        rows, cursor = [], first['elapsed']
        for row in self.rows:
            if cursor-.001 <= row['elapsed'] <= end+.001:
                rows.append(row)
                cursor = row['elapsed']+self.interval.value()
        def build():
            return [(row, self.backend.window(row['id'])) for row in rows]
        def apply(values):
            if not self.event or self.event['id'] != event_id or self.edit_serial.get(event_id, 0) != serial:
                self.error('传播期间事件已改变，请重试')
                return
            def change(e):
                for row, window in values:
                    e['t1'][row['id']] = self.translated_label(source, first, row, window)
            self.mutate(change)
        self.task('区间传播', build, apply)

    def clear_task(self):
        if not self.editable() or self.current_labels() is None:
            return
        if QtWidgets.QMessageBox.question(self, '清空当前任务', '清空当前事件T1，或当前T2目标的全部样本？人物、来源和其他任务保留。') == QtWidgets.QMessageBox.Yes:
            self.mutate(lambda e: self.current_labels(e).clear())

    def undo(self):
        if self.event_writable() and not self.event_editor.isVisible() and self.undo_stack:
            self.redo_stack.append(copy.deepcopy(self.event))
            value = self.undo_stack.pop()
            self.mutate(lambda e: (e.clear(), e.update(value)), record_undo=False,event_metadata=True)
            self.refresh_targets()

    def redo(self):
        if self.event_writable() and not self.event_editor.isVisible() and self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.event))
            value = self.redo_stack.pop()
            self.mutate(lambda e: (e.clear(), e.update(value)), record_undo=False,event_metadata=True)
            self.refresh_targets()

    def navigate(self, direction):
        if not self.rows:
            return
        index = self.frame_index+direction
        if not 0 <= index < len(self.rows):
            self.playback.stop()
            return
        event = self.navigation_event()
        if event and not event['start']-.001 <= self.rows[index]['elapsed'] <= event['end']+.001:
            self.playback.stop()
            return
        if self.mode.currentData() and self.playback.isActive():
            return
        self.goto(index)

    def sample_index(self, direction):
        if not self.rows:
            return None
        labels = self.current_labels() or {}
        current = self.rows[self.frame_index]['elapsed']
        target = current+direction*self.interval.value()
        event = self.navigation_event()
        allowed = [(i,r) for i,r in enumerate(self.rows) if not event or
                   event['start']-.001 <= r['elapsed'] <= event['end']+.001]
        candidates = [i for i,r in allowed if r['id'] in labels and 0 < (r['elapsed']-current)*direction <= self.interval.value()+.001]
        if candidates:
            return min(candidates) if direction > 0 else max(candidates)
        else:
            future = [(i,r) for i,r in allowed if (i-self.frame_index)*direction > 0]
            if future:
                return min(future,key=lambda item:abs(item[1]['elapsed']-target))[0]
        return None

    def navigate_sample(self, direction):
        index=self.sample_index(direction)
        if index is not None:
            self.goto(index)
        elif self.rows:
            self.message.setText('已到当前事件／子包的起点' if direction<0 else '已到当前事件／子包的终点')

    def jump_time(self, elapsed):
        event = self.navigation_event()
        if event:
            elapsed = min(event['end'], max(event['start'], elapsed))
        if self.rows:
            self.goto(min(range(len(self.rows)), key=lambda i: abs(self.rows[i]['elapsed']-elapsed)))

    def toggle_play(self):
        if self.mode.currentData():
            self.message.setText('输入审核固定当前观察；可双击三张RGB＋BEV观察图逐一查看原图')
            return
        if self.playback.isActive():
            self.playback.stop()
        else:
            self.play_anchor,self.play_buffering = None,True
            self.prefetch_error = None
            self.playback.start()
            self.schedule_prefetch()

    def play_tick(self):
        if not self.view_data or not self.rows:
            return
        following = self.rows[self.frame_index+1:self.frame_index+1+min(8,self.buffer_horizon())]
        event = self.navigation_event()
        following = [row for row in following if not event or row['elapsed']<=event['end']+.001]
        if not following:
            self.playback.stop()
            return
        ready = 0
        for row in following:
            if (row['id'],False) not in self.view_cache or (row['id'],False,self.point_size.value()) not in self.prepared_cache:
                break
            ready += 1
        if self.play_buffering:
            if ready<len(following):
                self.message.setText(f'播放缓冲：连续可播 {ready}/{len(following)} 帧…')
                self.schedule_prefetch()
                return
            self.play_buffering = False
            self.play_anchor = (time.monotonic(),self.rows[self.frame_index]['elapsed'])
            self.message.setText('播放中 · 按原始时间顺序，缓冲不足时等待')
        if not ready:
            self.play_buffering,self.play_anchor = True,None
            self.schedule_prefetch()
            return
        wall,elapsed = self.play_anchor
        if time.monotonic()-wall+1e-3>=following[0]['elapsed']-elapsed:
            self.play_advancing = True
            try:
                self.navigate(1)
            finally:
                self.play_advancing = False

    def point_changed(self):
        if hasattr(self, 'bev'):
            self.bev.point_size = self.point_size.value()
            self.cloud.set_point_size(self.point_size.value())
            self.render_scene()

    def fit_views(self):
        self.cloud.fit()
        self.rgb.fit()
        self.bev.fit()
        for card in self.history:
            card.context_bev.fit()

    def renew(self):
        if self.backend is not None and self.writable and not self.closing:
            self.task('续期写入会话', self.backend.renew)

    def reconnect(self):
        if self.backend is None:
            self.task('重新连接', self.connect_backend, self.connected)
            return
        if self.paused and self.dirty:
            if QtWidgets.QMessageBox.question(self, '保留草稿后重连', '先用“导出本机草稿”保留当前工作。继续将读取服务器结果；远程持久队列仍保留，不会自动覆盖冲突。') != QtWidgets.QMessageBox.Yes:
                return
        self.task('同步／重连', self.backend.snapshot, self.connected)

    def handle_conflict(self):
        if self.backend is None or not hasattr(self.backend, 'archive_conflicts'):
            self.error('本机模式请先导出工作稿，再重新读取工程')
            return
        if self.active_job or self.jobs or self.fetch_job:
            self.error('请等待当前读写结束后处理冲突')
            return
        if QtWidgets.QMessageBox.question(self, '存档草稿并读取服务器',
                '把当前工作稿及全部待同步操作保存在本机冲突存档，再读取服务器结果？不会自动合并或覆盖服务器；存档可用导出本机草稿取出。') != QtWidgets.QMessageBox.Yes:
            return
        current = copy.deepcopy(self.event)
        def recover():
            self.backend.archive_conflicts(current)
            return self.backend.snapshot()
        self.paused, self.dirty = False, False
        self.task('存档并读取服务器', recover, self.connected)

    def export_draft(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, '导出本机工作稿', '', 'JSON (*.json)')
        if filename:
            payload = dict(event=self.event, pending=self.backend.drafts() if hasattr(self.backend, 'drafts') else [],
                           warning='人工工作稿；不是服务器已保存金标准')
            Path(filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            self.message.setText('工作稿已导出')

    def error(self, error):
        self.message.setText('提示：'+str(error))
        self.message.setStyleSheet('color:#9a3f16')
        self.enable_editing()

    def help(self):
        QtWidgets.QMessageBox.information(self, '标注帮助',
            'A/D：原始帧；W/S：按间隔采样浏览，步长内优先停靠已有样本；空格：播放\n'
            '时间条拖选事件；I/O：取当前帧为起点／终点；Enter创建／保存，Esc取消选区。\n'
            '上方事件条单击选择，双击在页内编辑；默认浏览整包，可勾选仅浏览当前事件。\n'
            '1/2/3：待标／跳过／已标，不跳帧\nQ：应用群组／合理候选；R：清除T1选中分类\n'
            'Ctrl+点击：多选；Ctrl+Z/Y：撤销／重做；滚轮：缩放\n'
            '三维点云：左拖旋转、右拖平移、双击复位，点击人物框可选择。\n'
            '三帧放大模式上下排列RGB／BEV，双击查看原始模型图。\n'
            'WASD在列表、普通下拉框和数值框中用于导航；文字输入按Esc退出编辑。\n'
            '参考人物没有模型PID，可标身份与漏检，不会自动补进P1输入。\n'
            '相机视野外的已确认观测轨迹仍有正常PID；无投影时仅在BEV／三维和列表显示。\n'
            '上下文与输入审核分开；绿色已标不等于服务器已同步。')

    def keyPressEvent(self, event):
        if event.modifiers() & (QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier):
            super().keyPressEvent(event)
            return
        if self.editing_key(event):
            super().keyPressEvent(event)
            return
        if event.modifiers() & QtCore.Qt.ControlModifier:
            if event.key() in (QtCore.Qt.Key_Z, QtCore.Qt.Key_Y):
                self.undo() if event.key() == QtCore.Qt.Key_Z else self.redo()
                return
            super().keyPressEvent(event)
            return
        actions = {QtCore.Qt.Key_A: lambda: self.navigate(-1), QtCore.Qt.Key_D: lambda: self.navigate(1),
                   QtCore.Qt.Key_W: lambda: self.navigate_sample(-1), QtCore.Qt.Key_S: lambda: self.navigate_sample(1),
                   QtCore.Qt.Key_I: lambda: self.set_event_bound(0), QtCore.Qt.Key_O: lambda: self.set_event_bound(1),
                   QtCore.Qt.Key_1: lambda: self.mark('draft'), QtCore.Qt.Key_2: lambda: self.mark('skipped'),
                   QtCore.Qt.Key_3: lambda: self.mark('complete'), QtCore.Qt.Key_Space: self.toggle_play,
                   QtCore.Qt.Key_Q: self.apply_group if self.tabs.currentIndex() == 0 else self.set_candidates,
                   QtCore.Qt.Key_R: self.remove_classification, QtCore.Qt.Key_F1: self.help}
        if event.key() == QtCore.Qt.Key_Escape:
            if self.event_editor.isVisible():
                self.cancel_event_editor()
                event.accept()
                return
            self.people.setFocus(QtCore.Qt.ShortcutFocusReason)
            self.rgb.draw_box = False
            self.rgb.draw_start, self.rgb.rubber = None, None
            self.selected = []
            self.render_scene()
            event.accept()
            return
        if self.event_editor.isVisible() and event.key() in (QtCore.Qt.Key_Return,QtCore.Qt.Key_Enter):
            focus = QtWidgets.QApplication.focusWidget()
            if isinstance(focus,QtWidgets.QAbstractButton) and self.event_editor.isAncestorOf(focus):
                focus.click()
            else:
                self.commit_event_editor()
            event.accept()
            return
        action = actions.get(event.key())
        if action:
            action()
            event.accept()
        else:
            super().keyPressEvent(event)

    def editing_key(self, event):
        focus = QtWidgets.QApplication.focusWidget()
        popup = QtWidgets.QApplication.activePopupWidget()
        navigation = event.key() in (NAVIGATION_KEYS | RANGE_KEYS)
        if popup is None and event.key() == QtCore.Qt.Key_Escape:
            return False
        if (popup is None and self.event_editor.isVisible() and event.key() in (QtCore.Qt.Key_Return,QtCore.Qt.Key_Enter)
                and focus is not None and (focus is self.event_editor or self.event_editor.isAncestorOf(focus))):
            return False
        # Numeric editors contain a QLineEdit: inspect its owner before treating
        # it as a free-text field. WASD navigates; digits/arrows still edit values.
        control = popup if popup is not None else focus
        while control is not None:
            if isinstance(control, QtWidgets.QAbstractSpinBox):
                return not navigation
            if isinstance(control, QtWidgets.QComboBox):
                return control.isEditable() or not navigation
            control = control.parentWidget()
        if popup is not None:
            return True
        return isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)) and not focus.isReadOnly()

    def eventFilter(self, obj, event):
        if (event.type() not in (QtCore.QEvent.ShortcutOverride, QtCore.QEvent.KeyPress)
                or not isinstance(obj, QtWidgets.QWidget)
                or QtWidgets.QApplication.activeModalWidget() is not None
                or event.modifiers() & (QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier)):
            return False
        popup = QtWidgets.QApplication.activePopupWidget()
        if obj.window() is not self:
            if obj.window() is not popup:
                return False
            # Qt's isAncestorOf stops at popup windows despite their ownership.
            owner = popup.parentWidget()
            while owner is not self:
                if owner is None:
                    return False
                owner = owner.parentWidget()
        if self.editing_key(event):
            return False
        keys = NAVIGATION_KEYS | RANGE_KEYS | {QtCore.Qt.Key_1,
                QtCore.Qt.Key_2, QtCore.Qt.Key_3, QtCore.Qt.Key_Q, QtCore.Qt.Key_R, QtCore.Qt.Key_Space,
                QtCore.Qt.Key_Escape, QtCore.Qt.Key_F1}
        if self.event_editor.isVisible():
            keys |= {QtCore.Qt.Key_Return,QtCore.Qt.Key_Enter}
        control = event.modifiers() & QtCore.Qt.ControlModifier
        if (control and event.key() in (QtCore.Qt.Key_Z, QtCore.Qt.Key_Y)) or (not control and event.key() in keys):
            # Claim the key before Qt activates a control's own shortcut. The
            # action runs only on KeyPress, so each press navigates exactly once.
            if event.type() == QtCore.QEvent.ShortcutOverride:
                event.accept()
                return True
            if popup is not None and event.key() in (NAVIGATION_KEYS | RANGE_KEYS):
                parent = popup.parentWidget()
                while parent is not None:
                    if isinstance(parent, QtWidgets.QComboBox):
                        parent.hidePopup()
                        break
                    parent = parent.parentWidget()
            self.keyPressEvent(event)
            return True
        return False

    def closeEvent(self, event):
        if self.final_close:
            event.accept()
            return
        self.playback.stop()
        self.heartbeat.stop()
        if self.active_job or self.jobs or self.fetch_job:
            self.closing = True
            self.message.setText('等待当前读写完成后退出…')
            event.ignore()
            return
        if self.dirty and self.paused:
            if QtWidgets.QMessageBox.question(self, '尚有未保存工作', '当前工作尚未写入服务器。远程队列会保留；本机模式请先导出工作稿。仍要退出？') != QtWidgets.QMessageBox.Yes:
                event.ignore()
                self.heartbeat.start()
                return
        if hasattr(self.backend, 'pending') and self.backend.pending():
            if QtWidgets.QMessageBox.question(self, '尚有待同步草稿', '保留持久草稿退出？下次请使用相同地址、工程和缓存目录。') != QtWidgets.QMessageBox.Yes:
                event.ignore()
                self.heartbeat.start()
                return
        self.settings.setValue('reviewer', self.reviewer.text())
        self.settings.setValue('point_size', self.point_size.value())
        if self.backend:
            event.ignore()
            self.task('关闭会话', self.backend.close, lambda _: self.finish_close())
        else:
            event.accept()

    def finish_close(self):
        self.final_close = True
        QtWidgets.QApplication.instance().removeEventFilter(self)


def run(args):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    if args.server:
        from .remote import RemoteBackend
        factory = lambda: RemoteBackend(args.server, args.project_id, args.read_only, args.cache_dir)
    else:
        factory = lambda: LocalBackend(args.project,args.read_only)
    window = Window(factory)
    window.show()
    app.exec_()
