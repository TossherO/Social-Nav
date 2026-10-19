"""Independent Qt T1/T2 editor. All I/O is serialized outside the UI thread."""
from collections import deque
import copy
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys
import uuid

from PyQt5 import QtCore, QtGui, QtWidgets

from .backend import LocalBackend, load_view
from .rules import (CONTINUITY, COVERAGE, MATCH, RELATIONS, STATE_NAMES, VISIBILITY,
                    person_at, t1_empty, t2_empty, validate_t1, validate_t2)
from .store import Conflict, new_event
from .views import ImageThumb, PersonView, Timeline


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
        self.snapshot_data = dict(sources=[], events=[])
        self.rows, self.frame_index, self.view_data = [], 0, None
        self.event, self.revisions, self.edit_serial = None, {}, {}
        self.selected, self.undo_stack, self.redo_stack = [], [], []
        self.generation, self.rendering, self.saving, self.dirty = 0, False, 0, False
        self.writable, self.paused, self.closing, self.final_close = False, False, False, False
        self.setWindowTitle('Social-Nav · T1/T2决策标注')
        self.resize(1480, 940)
        self.settings = QtCore.QSettings('SocialNav', 'vlm_annotation')
        self.make_ui()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.heartbeat = QtCore.QTimer(self)
        self.heartbeat.timeout.connect(self.renew)
        self.heartbeat.start(25000)
        self.playback = QtCore.QTimer(self)
        self.playback.setInterval(100)
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
        self.sources.currentIndexChanged.connect(self.source_changed)
        self.events.currentIndexChanged.connect(self.event_changed)
        bar.addWidget(QtWidgets.QLabel('子bag'))
        bar.addWidget(self.sources)
        bar.addWidget(QtWidgets.QLabel('事件'))
        bar.addWidget(self.events)
        self.add_event_button = button('新建事件', self.add_event, bar)
        self.event_settings_button = button('事件／划分', self.event_settings, bar)
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
        self.position = QtWidgets.QLabel('正在连接…')
        modes.addWidget(self.position, 1)
        modes.addWidget(QtWidgets.QLabel('BEV点径'))
        self.point_size = QtWidgets.QSpinBox()
        self.point_size.setRange(1, 5)
        self.point_size.setValue(int(self.settings.value('point_size', 3)))
        self.point_size.valueChanged.connect(self.point_changed)
        modes.addWidget(self.point_size)
        button('复位视图', lambda: (self.rgb.fit(), self.bev.fit()), modes)
        layout.addLayout(modes)
        split = QtWidgets.QSplitter()
        self.rgb, self.bev = PersonView(), PersonView(bev=True)
        self.bev.point_size = self.point_size.value()
        self.rgb.picked.connect(self.pick)
        self.bev.picked.connect(self.pick)
        self.rgb.box_drawn.connect(self.add_auxiliary)
        split.addWidget(self.rgb)
        middle = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(middle)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.bev, 2)
        self.people = QtWidgets.QListWidget()
        self.people.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.people.itemSelectionChanged.connect(self.selection_changed)
        ml.addWidget(self.people, 1)
        identities = QtWidgets.QHBoxLayout()
        self.person_button = button('新建人物', self.add_person, identities)
        self.link_button = button('关联／拆分片段', self.link_person, identities)
        self.aux_button = button('补RGB人物', self.start_auxiliary, identities)
        ml.addLayout(identities)
        split.addWidget(middle)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(300)
        self.tabs.currentChanged.connect(self.task_changed)
        self.make_t1()
        self.make_t2()
        split.addWidget(self.tabs)
        split.setSizes([720, 430, 330])
        layout.addWidget(split, 1)
        history = QtWidgets.QHBoxLayout()
        self.history = [ImageThumb() for _ in range(3)]
        for view in self.history:
            view.setMaximumHeight(135)
            history.addWidget(view)
        layout.addLayout(history)
        self.timeline = Timeline()
        self.timeline.jumped.connect(self.jump_time)
        layout.addWidget(self.timeline)
        navigation = QtWidgets.QHBoxLayout()
        button('上一原始帧 [A]', lambda: self.navigate(-1), navigation)
        button('下一原始帧 [D]', lambda: self.navigate(1), navigation)
        button('上一样本 [W]', lambda: self.navigate_sample(-1), navigation)
        button('下一样本 [S]', lambda: self.navigate_sample(1), navigation)
        button('播放／暂停 [空格]', self.toggle_play, navigation)
        self.sample_button = button('加入样本', self.add_sample, navigation)
        self.batch_button = button('区间取样', self.batch_samples, navigation)
        navigation.addWidget(QtWidgets.QLabel('间隔秒'))
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(.1, 60)
        self.interval.setValue(2)
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
                             self.batch_button, self.clear_button, self.event_settings_button, *self.state_buttons]

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
        self.groups = QtWidgets.QListWidget()
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
        self.tabs.addTab(page, 'T1 群组')

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
        self.tabs.addTab(page, 'T2 身份')

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
        self.active_job = job
        job.result.connect(lambda result: self.job_result(job, callback, result))
        job.failed.connect(lambda error: self.job_failed(name, error, kind))
        job.finished.connect(lambda: self.job_finished(job))
        job.start()

    def job_result(self, job, callback, result):
        if callback:
            try:
                callback(result)
            except Exception as error:
                self.error(error)

    def job_failed(self, name, error, kind):
        if kind == 'save':
            self.saving -= 1
            self.paused, self.dirty = True, True
        if isinstance(error, Conflict):
            self.paused, self.writable = True, False
        if kind != 'prefetch':
            self.error(str(name) + '：' + str(error))

    def job_finished(self, job):
        self.active_job = None
        job.deleteLater()
        if self.final_close:
            QtCore.QTimer.singleShot(0, self.close)
            return
        self.start_next()
        if self.closing and self.active_job is None and not self.jobs:
            self.closing = False
            self.close()

    def connect_backend(self):
        if self.backend is None:
            self.backend = self.factory()
        return self.backend.snapshot()

    def connected(self, snapshot):
        self.paused, self.dirty = False, False
        self.snapshot_data = snapshot
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

    def source_changed(self):
        source = self.sources.currentData()
        if not source or self.backend is None:
            return
        self.playback.stop()
        self.view_data = None
        self.generation += 1
        self.task('读取序列索引', lambda: (source, self.backend.frames(source)), self.source_loaded)

    def source_loaded(self, result):
        source, rows = result
        if source != self.sources.currentData():
            return
        self.rows = rows
        old = self.event['id'] if self.event and self.event['source_id'] == source else None
        self.events.blockSignals(True)
        self.events.clear()
        for item in self.snapshot_data['events']:
            event = item['document']
            if event['source_id'] == source:
                self.events.addItem(event['name'], event['id'])
        index = self.events.findData(old)
        self.events.setCurrentIndex(max(0, index))
        self.events.blockSignals(False)
        self.event_changed()

    def event_changed(self):
        ident = self.events.currentData()
        self.event = next((copy.deepcopy(e['document']) for e in self.snapshot_data['events'] if e['document']['id'] == ident), None)
        self.undo_stack, self.redo_stack = [], []
        self.refresh_targets()
        if self.rows:
            index = next((i for i, row in enumerate(self.rows) if not self.event or row['elapsed'] >= self.event['start']), 0)
            self.goto(index)
        self.enable_editing()

    def source(self):
        return next(s for s in self.snapshot_data['sources'] if s['id'] == self.sources.currentData())

    def event_dialog(self, old=None):
        if not self.rows:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('事件与来源划分')
        form = QtWidgets.QFormLayout(dialog)
        name = QtWidgets.QLineEdit(old['name'] if old else self.source()['name']+'-事件')
        start, end = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for widget in (start, end):
            widget.setDecimals(3)
            widget.setRange(self.rows[0]['elapsed'], self.rows[-1]['elapsed'])
        start.setValue(old['start'] if old else self.rows[self.frame_index]['elapsed'])
        end.setValue(old['end'] if old else self.rows[-1]['elapsed'])
        root = QtWidgets.QLineEdit(old['root_group'] if old else self.source().get('root_group', ''))
        place = QtWidgets.QLineEdit(old['location_group'] if old else '')
        split = combo(dict(unassigned='暂不划分', development='开发', validation='验证', holdout='留出'))
        split.setCurrentIndex(max(0, split.findData(old['split'] if old else 'unassigned')))
        for title, widget in (('名称', name), ('起始秒', start), ('结束秒', end), ('采集根来源', root), ('重复地点组', place), ('用途', split)):
            form.addRow(title, widget)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != dialog.Accepted:
            return None
        if not name.text().strip() or start.value() > end.value():
            self.error('名称不能为空，起始时间不能晚于结束时间')
            return None
        event = copy.deepcopy(old) if old else new_event(self.source(), start.value(), end.value(), name.text().strip())
        event.update(name=name.text().strip(), start=start.value(), end=end.value(), root_group=root.text().strip(),
                     location_group=place.text().strip(), split=split.currentData())
        return event

    def add_event(self):
        if not self.writable or self.paused:
            return
        event = self.event_dialog()
        if event:
            self.event = event
            self.revisions[event['id']] = 0
            self.edit_serial[event['id']] = 0
            self.update_snapshot(event)
            self.events.addItem(event['name'], event['id'])
            self.events.setCurrentIndex(self.events.findData(event['id']))
            self.save()

    def event_settings(self):
        if not self.editable():
            return
        event = self.event_dialog(self.event)
        if event:
            used = set(event['t1'])
            used.update(t['reference_frame'] for t in event['targets'].values())
            used.update(ident for t in event['targets'].values() for ident in t['samples'])
            if any(r['id'] in used and not event['start']-.001 <= r['elapsed'] <= event['end']+.001 for r in self.rows):
                self.error('新事件范围会排除已有样本或描述参考帧；请另建事件，或先明确处理这些标注')
                return
            self.mutate(lambda e: e.update(event))

    def update_snapshot(self, event):
        found = next((r for r in self.snapshot_data['events'] if r['document']['id'] == event['id']), None)
        row = dict(document=copy.deepcopy(event), revision=self.revisions.get(event['id'], 0))
        if found is None:
            self.snapshot_data['events'].append(row)
        else:
            found.update(row)

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
        if self.rows:
            self.goto(self.frame_index)

    def goto(self, index):
        if not self.rows:
            return
        self.frame_index = max(0, min(len(self.rows)-1, index))
        self.generation += 1
        generation, frame_id, review = self.generation, self.rows[self.frame_index]['id'], self.mode.currentData()
        self.view_data = None
        self.selected = []
        self.enable_editing()
        self.position.setText(f'读取 {self.rows[self.frame_index]["elapsed"]:.3f}s…')
        self.task('读取观察', lambda: load_view(self.backend, frame_id, review),
                  lambda result: self.view_loaded(generation, result), kind='view')

    def view_loaded(self, generation, result):
        if generation != self.generation:
            return
        self.view_data = result
        frame = result['frame']
        self.position.setText(f'{frame["elapsed"]:.3f}s · 原始帧{frame["ordinal"]} · '
                              + ('三帧输入完整' if result['window']['ready'] else '历史／RGB不足'))
        for widget, data in zip(self.history, result['histories']):
            widget.show_data(data)
        self.render_scene()
        self.render_labels()
        self.enable_editing()
        # Only light immutable assets, one queued prefetch at a time; foreground wins.
        ids = [r['id'] for r in self.rows[self.frame_index+1:self.frame_index+5]]
        def prefetch():
            for ident in ids:
                if self.generation != generation:
                    return
                self.backend.asset(ident, 'preview')
                if self.generation != generation:
                    return
                self.backend.asset(ident, 'cloud')
        if ids and not self.playback.isActive():
            self.task('预读', prefetch, kind='prefetch')

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
        prefix = 'P'+str(mapping[key]) if key in mapping else '参考'+str(next((i+1 for i,p in enumerate(self.scene_people()) if p['key']==key), '?'))
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
        self.rgb.image_size = self.view_data['frame']['image_size']
        self.rgb.show_frame(self.view_data['image'], None, people, names)
        self.bev.show_frame(None, self.view_data['cloud'], people, names)
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

    def pick(self, key, multiple):
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
            for group in t1['groups']:
                roles = ' · 角色已标' if group.get('roles') else ''
                item = QtWidgets.QListWidgetItem(RELATIONS[group['relation']]+'：'+', '.join(self.label_for_key(k) for k in group['members'])+roles)
                item.setData(QtCore.Qt.UserRole, group)
                self.groups.addItem(item)
            for group in t1.get('reference_groups', []):
                item = QtWidgets.QListWidgetItem('覆盖失败参考：'+RELATIONS[group['relation']]+' · '+
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
            stale = label and label.get('binding') and label['binding'] != self.view_data['window']['fingerprint']
            self.position.setText(f'{self.view_data["frame"]["elapsed"]:.3f}s · {state}' + (' · 输入变化，需重审' if stale else ''))
            classifications = {}
            if t1:
                for number, group in enumerate(t1['groups'], 1):
                    for key in group['members']:
                        classifications[key] = RELATIONS[group['relation']]+'组'+str(number)
                classifications.update({k: '确认未成组' for k in t1['singles']})
                classifications.update({k: '归属未知' for k in t1['unknown']})
            for i in range(self.people.count()):
                item = self.people.item(i)
                key = item.data(QtCore.Qt.UserRole)
                item.setText(self.label_for_key(key)+' · '+classifications.get(key, '未分类' if key in self.view_data['window']['mapping'] else '参考／无模型PID'))
        self.timeline.show_state(self.rows, self.event, self.target(), self.rows[self.frame_index]['elapsed'] if self.rows else 0)
        self.rendering = False

    def editable(self):
        return bool(self.writable and not self.paused and self.event and self.view_data)

    def enable_editing(self):
        if not hasattr(self, 'task_widgets'):
            return
        enabled = self.editable()
        for widget in self.task_widgets:
            widget.setEnabled(enabled)
        self.add_event_button.setEnabled(bool(self.writable and self.rows and not self.paused))
        for widget in (self.group_button, self.single_button, self.unknown_button, self.remove_button,
                       self.copy_button, self.propagate_button, self.t1_note):
            widget.setEnabled(enabled)
        self.target_button.setEnabled(enabled)
        target_enabled = enabled and self.target() is not None
        for widget in (*self.t2_controls.values(), self.true_button, self.unset_true_button, self.candidates_button, self.clear_candidates_button,
                       self.t2_note, self.flags):
            widget.setEnabled(target_enabled)

    def mutate(self, function, record_undo=True):
        if not self.editable():
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
                    self.render_labels()
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
        if self.editable() and self.undo_stack:
            self.redo_stack.append(copy.deepcopy(self.event))
            value = self.undo_stack.pop()
            self.mutate(lambda e: (e.clear(), e.update(value)), record_undo=False)
            self.refresh_targets()

    def redo(self):
        if self.editable() and self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.event))
            value = self.redo_stack.pop()
            self.mutate(lambda e: (e.clear(), e.update(value)), record_undo=False)
            self.refresh_targets()

    def navigate(self, direction):
        if not self.rows:
            return
        index = self.frame_index+direction
        if not 0 <= index < len(self.rows):
            self.playback.stop()
            return
        if self.event and not self.event['start']-.001 <= self.rows[index]['elapsed'] <= self.event['end']+.001:
            self.playback.stop()
            return
        if self.mode.currentData() and self.playback.isActive():
            return
        self.goto(index)

    def navigate_sample(self, direction):
        labels = self.current_labels()
        if labels is None or not self.rows:
            return
        candidates = [i for i, r in enumerate(self.rows) if r['id'] in labels and (i-self.frame_index)*direction > 0]
        if candidates:
            self.goto(min(candidates) if direction > 0 else max(candidates))
        else:
            self.message.setText('该方向没有已有样本；可用区间取样或加入当前样本')

    def jump_time(self, elapsed):
        if self.event:
            elapsed = min(self.event['end'], max(self.event['start'], elapsed))
        if self.rows:
            self.goto(min(range(len(self.rows)), key=lambda i: abs(self.rows[i]['elapsed']-elapsed)))

    def toggle_play(self):
        if self.mode.currentData():
            self.message.setText('输入审核固定当前观察；可双击下方三张因果观察图逐一查看')
            return
        if self.playback.isActive():
            self.playback.stop()
        else:
            self.playback.start()

    def play_tick(self):
        if self.view_data and self.active_job is None:
            self.navigate(1)

    def point_changed(self):
        if hasattr(self, 'bev'):
            self.bev.point_size = self.point_size.value()
            self.render_scene()

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
        if self.active_job or self.jobs:
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
            'A/D：原始帧；W/S：已有样本；空格：上下文播放\n'
            '1/2/3：待标／跳过／已标，不跳帧\nQ：应用群组／合理候选；R：清除T1选中分类\n'
            'Ctrl+点击：多选；Ctrl+Z/Y：撤销／重做；滚轮：缩放；拖动：平移\n'
            '双击下方历史图：查看可缩放大图。输入框内保留文字编辑。\n'
            '参考人物没有模型PID，可标身份与漏检，不会自动补进P1输入。\n'
            '上下文与输入审核分开；绿色已标不等于服务器已同步。')

    def keyPressEvent(self, event):
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit,
                              QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
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
                   QtCore.Qt.Key_1: lambda: self.mark('draft'), QtCore.Qt.Key_2: lambda: self.mark('skipped'),
                   QtCore.Qt.Key_3: lambda: self.mark('complete'), QtCore.Qt.Key_Space: self.toggle_play,
                   QtCore.Qt.Key_Q: self.apply_group if self.tabs.currentIndex() == 0 else self.set_candidates,
                   QtCore.Qt.Key_R: self.remove_classification, QtCore.Qt.Key_F1: self.help}
        if event.key() == QtCore.Qt.Key_Escape:
            self.rgb.draw_box = False
            self.rgb.draw_start, self.rgb.rubber = None, None
            self.selected = []
            self.render_scene()
            return
        action = actions.get(event.key())
        if action:
            action()
            event.accept()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if (event.type() != QtCore.QEvent.KeyPress or not isinstance(obj, QtWidgets.QWidget)
                or obj.window() is not self or QtWidgets.QApplication.activeModalWidget() is not None):
            return False
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit,
                              QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            return False
        keys = {QtCore.Qt.Key_A, QtCore.Qt.Key_D, QtCore.Qt.Key_W, QtCore.Qt.Key_S, QtCore.Qt.Key_1,
                QtCore.Qt.Key_2, QtCore.Qt.Key_3, QtCore.Qt.Key_Q, QtCore.Qt.Key_R, QtCore.Qt.Key_Space,
                QtCore.Qt.Key_Escape, QtCore.Qt.Key_F1}
        control = event.modifiers() & QtCore.Qt.ControlModifier
        if (control and event.key() in (QtCore.Qt.Key_Z, QtCore.Qt.Key_Y)) or (not control and event.key() in keys):
            self.keyPressEvent(event)
            return True
        return False

    def closeEvent(self, event):
        if self.final_close:
            event.accept()
            return
        self.playback.stop()
        self.heartbeat.stop()
        if self.active_job or self.jobs:
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
        factory = lambda: LocalBackend(args.project, args.read_only)
    window = Window(factory)
    window.show()
    app.exec_()
