"""Local GUI for operator-driven Agent sessions; it never commands the chassis."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from PyQt5 import QtCore, QtGui, QtWidgets
from . import ROOT
from .ui_data import ContextTimeline,connect_readonly,display_value,event_detail,event_summary,goal_requirements,poll_session,select_detail

STATUS={'ready':'准备决策','waiting_navigation':'等待人工驾驶或到达确认','waiting_observation':'等待新观察',
        'waiting_user':'等待用户继续','needs_clarification':'等待用户答复','paused':'已暂停',
        'stopping':'正在结束任务','cancelled':'已取消','succeeded':'已完成','failed':'未完成',
        'idle':'空闲','running':'进行中',
        'blocked':'受阻','refused':'未接收','pending':'待执行','completed':'已完成'}
OPERATION_EVENTS={'manual_goal_evidence_wait_started':'等待到达证据',
    'manual_goal_evidence_wait_finished':'到达证据等待结果',
    'manual_goal_evidence_refresh_failed':'到达证据读取失败',
    'manual_goal_evidence_unproven':'到达证据不足',
    'manual_reference_status':'参考路径预览','manual_reference_region_conflict':'绕行路径冲突','manual_route_progress':'路线进度',
    'operator_command':'操作结果','navigation_context_lost':'定位或地图变化',
    'task_connection_lost':'连接中断','model_failed':'模型错误',
    'model_protocol_rejected':'模型格式错误','conversation_trimmed':'上下文裁剪','route_repeated_refusal':'路线重复被拒绝',
    'agent_run_finished':'会话结束'}


def message_label(message):
    roles={'system':('系统提示','#7755aa'),'user':('用户 / 状态','#2673b8'),
           'assistant':('模型输出','#b34838'),'tool':('工具输出','#118166')}
    label,color=roles.get(message['role'],(message['role'],'#566276'))
    if message['role']=='system':
        if '任务登记阶段' in message['summary']:label+=' · 登记阶段'
        elif '导航 Agent' in message['summary']:label+=' · 导航阶段'
    return label,color
class SessionReader(QtCore.QThread):
    update=QtCore.pyqtSignal(object,object)
    error=QtCore.pyqtSignal(str)

    def __init__(self,path):
        super().__init__();self.path=Path(path)

    def run(self):
        cursor=0;db=None;last_error=None
        try:
            while not self.isInterruptionRequested():
                if db is None and not self.path.is_file():
                    self.msleep(300);continue
                try:
                    if db is None:db=connect_readonly(self.path)
                    state,events=poll_session(db,cursor)
                    if events:cursor=events[-1]['seq']
                    self.update.emit(state,events)
                    last_error=None
                except (sqlite3.Error,ValueError,OSError) as error:
                    message=type(error).__name__+': '+str(error)[:160]
                    if message!=last_error:self.error.emit(message);last_error=message
                    if db is not None:db.close();db=None
                self.msleep(300)
        finally:
            if db is not None:db.close()


class DetailReader(QtCore.QThread):
    ready=QtCore.pyqtSignal(str,object,object)
    error=QtCore.pyqtSignal(str)

    def __init__(self,path,seq):
        super().__init__();self.path=path;self.seq=seq

    def run(self):
        try:
            db=connect_readonly(self.path)
            try:kind,value,images=event_detail(db,self.seq)
            finally:db.close()
            self.ready.emit(kind,value,images)
        except (sqlite3.Error,ValueError,OSError,TypeError) as error:
            self.error.emit(type(error).__name__+': '+str(error)[:160])


class ClickableImage(QtWidgets.QLabel):
    doubleClicked=QtCore.pyqtSignal()

    def mouseDoubleClickEvent(self,event):
        if event.button()==QtCore.Qt.LeftButton:self.doubleClicked.emit()
        else:super().mouseDoubleClickEvent(event)


class ImageView(QtWidgets.QGraphicsView):
    def __init__(self,pixmap,parent=None):
        super().__init__(parent)
        self.image_scene=QtWidgets.QGraphicsScene(self);self.setScene(self.image_scene)
        self.image_item=self.image_scene.addPixmap(pixmap)
        self.setSceneRect(self.image_item.boundingRect())
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

    def zoom(self,factor):
        scale=self.transform().m11()*factor
        if .01 <= scale <= 32:self.scale(factor,factor)

    def fit_image(self):
        if not self.image_item.pixmap().isNull():
            self.fitInView(self.image_item,QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self,event):
        if event.angleDelta().y():
            self.zoom(1.25 if event.angleDelta().y()>0 else .8);event.accept()
        else:super().wheelEvent(event)


class ImageDialog(QtWidgets.QDialog):
    def __init__(self,pixmap,title,parent=None):
        super().__init__(parent);self.setWindowTitle(title);self.resize(1100,800)
        layout=QtWidgets.QVBoxLayout(self);controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        self.view=ImageView(pixmap,self)
        for text,action in [('放大',lambda:self.view.zoom(1.25)),('缩小',lambda:self.view.zoom(.8)),
                            ('适应窗口',self.view.fit_image),('原始大小',self.view.resetTransform)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(action);controls.addWidget(button)
        controls.addWidget(QtWidgets.QLabel('滚轮缩放，按住左键拖动图片。'),1)
        layout.addWidget(self.view,1)
        self._first_show=True

    def showEvent(self,event):
        super().showEvent(event)
        if self._first_show:
            self._first_show=False;QtCore.QTimer.singleShot(0,self.view.fit_image)


class DetailDialog(QtWidgets.QDialog):
    def __init__(self,kind,value,images,parent=None,image_index=0,message_only=False):
        super().__init__(parent);self.setWindowTitle('事件详情 · '+kind);self.resize(1000,700)
        layout=QtWidgets.QVBoxLayout(self)
        if kind=='api_request':
            description=('这是该次模型请求中的所选消息及其图片。' if message_only else '这是该次模型请求的全部消息及图片。')
            notice=QtWidgets.QLabel(description+'内部标识已隐藏；图像来自请求当时，不代表当前画面。')
            notice.setWordWrap(True);notice.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Fixed)
            layout.addWidget(notice,0)
        split=QtWidgets.QSplitter();layout.addWidget(split,1)
        text=QtWidgets.QPlainTextEdit();text.setReadOnly(True)
        text.setPlainText(json.dumps(display_value(value),ensure_ascii=False,indent=2));split.addWidget(text)
        if images:
            panel=QtWidgets.QWidget();right=QtWidgets.QVBoxLayout(panel)
            choices=QtWidgets.QListWidget();picture=ClickableImage('选择输入图像');picture.setAlignment(QtCore.Qt.AlignCenter)
            picture.setMinimumSize(320,240);picture.setWordWrap(True)
            right.addWidget(choices);right.addWidget(picture,1)
            zoom_button=QtWidgets.QPushButton('放大查看选中图片');zoom_button.setEnabled(False);right.addWidget(zoom_button)
            for name,_ in images:choices.addItem(name)
            def show(row):
                if row<0:zoom_button.setEnabled(False);return
                pix=QtGui.QPixmap()
                if images[row][1] is not None:pix.loadFromData(images[row][1])
                zoom_button.setEnabled(not pix.isNull())
                if pix.isNull():picture.setText('无法解码该图像');return
                picture.setPixmap(pix.scaled(560,480,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            def open_image():
                row=choices.currentRow()
                if row<0 or images[row][1] is None:return
                pix=QtGui.QPixmap();pix.loadFromData(images[row][1])
                if not pix.isNull():ImageDialog(pix,images[row][0],self).exec_()
            zoom_button.clicked.connect(open_image);picture.doubleClicked.connect(open_image)
            choices.itemDoubleClicked.connect(lambda _item:open_image())
            choices.currentRowChanged.connect(show);choices.setCurrentRow(min(max(image_index,0),len(images)-1));split.addWidget(panel)


class ManualWindow(QtWidgets.QMainWindow):
    def __init__(self, socket='', catalog='', duration=1200):
        super().__init__();self.setWindowTitle('Social-Nav · Agent 人工调试');self.resize(1380,900)
        self.process=QtCore.QProcess(self);self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.buffer=b'';self.reader=None;self.detail_reader=None;self.session_path=None
        self.state=None;self.status=None;self.status_received=0.;self.pending={};self.target=None
        self.annotation_names={}
        self.timeline=ContextTimeline();self.timeline_items=[];self.active_tools=[];self.detail_selection=None
        self.latest_request_seq=None
        self._build(socket,catalog,duration)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh_status);self.timer.start(500)

    def _build(self,socket,catalog,duration):
        central=QtWidgets.QWidget();self.setCentralWidget(central);outer=QtWidgets.QVBoxLayout(central)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6fa; color: #243047; font-size: 13px; }
            QLineEdit, QSpinBox, QTreeWidget, QPlainTextEdit, QListWidget {
                background: white; border: 1px solid #d8dfeb; border-radius: 5px; padding: 4px;
            }
            QPushButton { background: #e8eef9; border: 1px solid #cbd8ed; border-radius: 5px; padding: 6px 10px; }
            QPushButton:hover { background: #d7e5fb; }
            QPushButton:disabled { color: #9ca7b8; background: #f0f2f6; }
            QHeaderView::section { background: #e9eef6; border: 0; padding: 6px; font-weight: bold; }
        """)
        self.banner=QtWidgets.QLabel('人工驾驶 · 观察会话不连接底盘 · 任务取消不会让手柄停车')
        self.banner.setStyleSheet('font-weight:bold;color:#8a4b00;padding:6px;background:#fff3d4')
        outer.addWidget(self.banner)
        self.settings_toggle=QtWidgets.QToolButton();self.settings_toggle.setCheckable(True);self.settings_toggle.setChecked(True)
        self.settings_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.settings_toggle.setArrowType(QtCore.Qt.DownArrow);self.settings_toggle.setText('收起会话设置')
        outer.addWidget(self.settings_toggle,0,QtCore.Qt.AlignLeft)
        self.settings_panel=QtWidgets.QWidget();form=QtWidgets.QGridLayout(self.settings_panel)
        form.setContentsMargins(0,0,0,0);outer.addWidget(self.settings_panel)
        self.settings_toggle.toggled.connect(self.toggle_settings)
        self.socket=QtWidgets.QLineEdit(socket);self.catalog=QtWidgets.QLineEdit(catalog)
        self.instruction=QtWidgets.QLineEdit();self.instruction.setPlaceholderText('输入长程任务，例如：先到停靠点1，再绕圆形区域一圈')
        self.duration=QtWidgets.QSpinBox();self.duration.setRange(1,86400);self.duration.setValue(duration);self.duration.setSuffix(' 秒')
        self.start_button=QtWidgets.QPushButton('开始新任务');self.start_button.clicked.connect(self.start_task)
        self.history_button=QtWidgets.QPushButton('打开历史会话');self.history_button.clicked.connect(self.open_history)
        self.browse_buttons=[]
        for row,(label,edit) in enumerate((('观察 socket',self.socket),('语义标注',self.catalog))):
            form.addWidget(QtWidgets.QLabel(label),row,0);form.addWidget(edit,row,1)
            browse=QtWidgets.QPushButton('选择会话目录' if row==0 else '选择文件')
            if row==0:browse.clicked.connect(self.choose_observation)
            else:browse.clicked.connect(lambda _=False,e=edit:self.choose_file(e))
            form.addWidget(browse,row,2);self.browse_buttons.append(browse)
        form.addWidget(QtWidgets.QLabel('任务'),2,0);form.addWidget(self.instruction,2,1)
        form.addWidget(self.duration,2,2);form.addWidget(self.start_button,0,3);form.addWidget(self.history_button,1,3)
        outer.addWidget(QtWidgets.QLabel('任务、地图、实时状态和 RGB 可能发送至已配置的托管模型服务。'))
        self.status_label=QtWidgets.QPlainTextEdit('未连接');self.status_label.setReadOnly(True)
        self.status_label.setFixedHeight(self.status_label.fontMetrics().lineSpacing()*2+14)
        self.status_label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded);outer.addWidget(self.status_label)
        split=QtWidgets.QSplitter();outer.addWidget(split,1)
        left=QtWidgets.QWidget();left.setMinimumWidth(380);left_layout=QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel('任务与进度'))
        self.current_task=QtWidgets.QLabel('当前任务：无');self.current_task.setWordWrap(True);left_layout.addWidget(self.current_task)
        self.goals=QtWidgets.QTreeWidget();self.goals.setHeaderLabels(['子任务与要求','状态'])
        self.goals.header().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch)
        self.goals.header().setSectionResizeMode(1,QtWidgets.QHeaderView.ResizeToContents)
        self.goals.setWordWrap(True);self.goals.setTextElideMode(QtCore.Qt.ElideNone);left_layout.addWidget(self.goals,2)
        self.goals.itemSelectionChanged.connect(self.show_goal_requirements)
        self.goal_details=QtWidgets.QPlainTextEdit('选择子任务查看完整要求');self.goal_details.setReadOnly(True)
        self.goal_details.setFixedHeight(self.goal_details.fontMetrics().lineSpacing()*5+14)
        left_layout.addWidget(self.goal_details)
        self.target_text=QtWidgets.QLabel('尚无目标');self.target_text.setWordWrap(True);left_layout.addWidget(self.target_text)
        self.target_image=ClickableImage('目标示意图');self.target_image.setAlignment(QtCore.Qt.AlignCenter)
        self.target_image.setMinimumSize(300,220);left_layout.addWidget(self.target_image,2)
        self.target_pixmap=None;self.target_image.doubleClicked.connect(self.open_target_image)
        self.target_zoom=QtWidgets.QPushButton('放大查看目标示意图');self.target_zoom.setEnabled(False)
        self.target_zoom.clicked.connect(self.open_target_image);left_layout.addWidget(self.target_zoom)
        split.addWidget(left)
        right=QtWidgets.QWidget();right_layout=QtWidgets.QVBoxLayout(right)
        header=QtWidgets.QHBoxLayout();right_layout.addLayout(header)
        header.addWidget(QtWidgets.QLabel('Agent 对话与上下文'),1)
        tools_button=QtWidgets.QPushButton('查看工具');tools_button.clicked.connect(self.show_tools);header.addWidget(tools_button)
        self.right_sections=QtWidgets.QSplitter(QtCore.Qt.Vertical);right_layout.addWidget(self.right_sections,1)
        self.right_sections.setChildrenCollapsible(False)
        self.context_tabs=QtWidgets.QTabWidget();self.right_sections.addWidget(self.context_tabs)
        current=QtWidgets.QWidget();current_layout=QtWidgets.QVBoxLayout(current)
        current_header=QtWidgets.QHBoxLayout();current_layout.addLayout(current_header)
        self.request_summary=QtWidgets.QLabel('尚无模型请求');current_header.addWidget(self.request_summary,1)
        self.request_button=QtWidgets.QPushButton('查看整轮请求');self.request_button.setEnabled(False)
        self.request_button.clicked.connect(self.open_request_detail);current_header.addWidget(self.request_button)
        self.current_tree=QtWidgets.QTreeWidget();self.current_tree.setHeaderLabels(['顺序','类型','内容'])
        self.current_tree.setColumnWidth(0,60);self.current_tree.setColumnWidth(1,150)
        self.current_tree.itemDoubleClicked.connect(self.open_detail);current_layout.addWidget(self.current_tree,1)
        current_layout.addWidget(QtWidgets.QLabel('按最近一次实际请求顺序显示；后续模型输出可能尚未进入请求。'))
        self.context_tabs.addTab(current,'当前请求')
        history=QtWidgets.QWidget();history_layout=QtWidgets.QVBoxLayout(history)
        self.context_tree=QtWidgets.QTreeWidget();self.context_tree.setHeaderLabels(['时间','类型 / 状态','内容'])
        self.context_tree.setColumnWidth(0,100);self.context_tree.setColumnWidth(1,110)
        self.context_tree.itemDoubleClicked.connect(self.open_detail)
        history_layout.addWidget(self.context_tree,1)
        history_layout.addWidget(QtWidgets.QLabel('按首次出现的时间排列；状态标明最近请求中、历史或等待下一次请求。'))
        self.context_tabs.addTab(history,'历史记录')
        events_panel=QtWidgets.QWidget();events_layout=QtWidgets.QVBoxLayout(events_panel)
        events_layout.setContentsMargins(0,0,0,0)
        events_layout.addWidget(QtWidgets.QLabel('运行事件（不是模型对话；双击查看详情）'))
        self.operation_events=QtWidgets.QTreeWidget();self.operation_events.setHeaderLabels(['时间','类型','内容'])
        self.operation_events.setColumnWidth(0,85);self.operation_events.setColumnWidth(1,170)
        self.operation_events.itemDoubleClicked.connect(self.open_detail);events_layout.addWidget(self.operation_events,1)
        self.right_sections.addWidget(events_panel);self.right_sections.setStretchFactor(0,3);self.right_sections.setStretchFactor(1,1)
        self.right_sections.setSizes([450,180])
        split.addWidget(right);split.setStretchFactor(1,2)
        self.notice=QtWidgets.QLabel('历史会话只读；Agent 不会因打开记录而恢复。');self.notice.setWordWrap(True);outer.addWidget(self.notice)
        controls=QtWidgets.QHBoxLayout();outer.addLayout(controls)
        self.buttons={}
        for name,label in [('arrived','确认到达'),('pause','暂停任务'),('resume','恢复'),('continue','继续'),('cancel','取消任务')]:
            button=QtWidgets.QPushButton(label);button.clicked.connect(lambda _=False,n=name:self.send_command(n))
            controls.addWidget(button);self.buttons[name]=button
        self.message=QtWidgets.QLineEdit();self.message.setPlaceholderText('需要澄清时输入答复，或输入用于替换任务的新指令')
        controls.addWidget(self.message,1)
        for name,label in [('answer','发送答复'),('replace','替换任务')]:
            button=QtWidgets.QPushButton(label);button.clicked.connect(lambda _=False,n=name:self.send_command(n))
            controls.addWidget(button);self.buttons[name]=button
        self.refresh_status()

    def toggle_settings(self,expanded):
        self.settings_panel.setVisible(expanded)
        self.settings_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.settings_toggle.setText('收起会话设置' if expanded else '展开会话设置')

    def choose_file(self,edit):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择文件',edit.text() or str(ROOT))
        if path:edit.setText(path)

    def choose_observation(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'选择观察会话目录',str(ROOT/'outputs/navigation/live_observation'))
        if not path:return
        directory=Path(path)
        candidate=directory/'run/commands.sock'
        if not candidate.exists():candidate=directory/'commands.sock'
        self.socket.setText(str(candidate))
        if not candidate.exists():self.notice.setText('该目录中没有观察会话 socket；请确认选择的是正在运行的会话目录。')

    def set_reader(self,path):
        if self.reader is not None:
            self.reader.requestInterruption()
            if not self.reader.wait(2000):
                self.notice.setText('上一会话仍在读取，请稍后重试。');return False
        self.session_path=Path(path);self.state=None;self.status=None;self.goals.clear();self.annotation_names={}
        self.goal_details.setPlainText('选择子任务查看完整要求')
        self.context_tree.clear();self.current_tree.clear();self.operation_events.clear();self.timeline=ContextTimeline();self.timeline_items=[];self.active_tools=[]
        self.latest_request_seq=None;self.request_summary.setText('尚无模型请求');self.request_button.setEnabled(False)
        self.current_task.setText('当前任务：无')
        self.target=None;self.target_text.setText('尚无目标');self.target_image.clear()
        self.target_pixmap=None;self.target_zoom.setEnabled(False)
        self.reader=SessionReader(path);self.reader.update.connect(self.update_session);self.reader.error.connect(self.reader_error)
        self.reader.start()
        return True

    def open_history(self):
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.notice.setText('当前任务仍在运行，请先结束再打开历史会话。');return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择 session.sqlite',str(ROOT/'outputs/decision/agent'),'SQLite (*.sqlite)')
        if path:
            if self.set_reader(path):
                self.settings_toggle.setChecked(False)
                self.notice.setText('只读历史会话；不会自动恢复或调用模型。')

    def start_task(self):
        if self.process.state()!=QtCore.QProcess.NotRunning:return
        socket=self.socket.text().strip();catalog=self.catalog.text().strip();instruction=self.instruction.text().strip()
        if not socket or not Path(socket).exists() or not Path(catalog).is_file() or not instruction:
            self.notice.setText('请提供存在的观察 socket、语义标注和非空任务。');return
        socket=str(Path(socket).resolve());catalog=str(Path(catalog).resolve())
        name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output=ROOT/'outputs/decision/agent'/('manual_ui_'+name)
        if not self.set_reader(output/'session.sqlite'):return
        self.buffer=b'';self.pending={};self.target=None
        self.target_text.setText('等待 Agent 选择目标');self.target_image.clear()
        args=['-B','-m','decision.agent','manual','--runtime-socket',socket,'--catalog',catalog,
              '--instruction',instruction,'--interactive','--gui-commands','--duration',str(self.duration.value()),
              '--output',str(output)]
        self.process.start(sys.executable,args)
        self.settings_toggle.setChecked(False)
        self.notice.setText('正在启动人工 Agent；等待定位和会话初始化。')
        self.refresh_status()

    def read_stdout(self):
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:value=json.loads(line.decode('utf-8'))
            except (ValueError,UnicodeDecodeError):continue
            if not isinstance(value,dict):continue
            event=value.get('event')
            if event=='manual_status':
                self.status=value;self.status_received=time.monotonic()
            elif event=='operator_command':
                self.pending.pop(value.get('command_id'),None)
                self.notice.setText('操作已接收' if value.get('accepted') else '操作未执行：'+str(value.get('reason')))
                self.status=None
            elif event=='manual_target':self.show_target(value)
            elif event=='manual_reference_region_conflict':self.notice.setText(value.get('instruction','参考路径穿过绕行区域，请手动停稳。'))
            elif event=='manual_observation_stage':self.notice.setText(value.get('instruction','进入观察阶段'))
            elif event=='manual_rviz_unavailable':self.notice.setText('RViz Agent 标记不可用：'+str(value.get('reason')))
            elif 'event_rejected' in value:self.notice.setText('操作未执行：'+str(value['event_rejected']))
        self.refresh_status()

    def read_stderr(self):
        text=bytes(self.process.readAllStandardError()).decode('utf-8','replace').strip()
        if text:self.notice.setText('Agent 进程提示：'+text[-1000:])

    def process_finished(self,code,_status):
        self.status=None;self.pending.clear();self.notice.setText(f'Agent 进程已结束（退出码 {code}）；会话记录可继续查看。')
        self.refresh_status()

    def process_error(self,error):
        self.notice.setText('Agent 进程不可用：'+self.process.errorString());self.refresh_status()

    def show_goal_requirements(self):
        selected=self.goals.selectedItems()
        if not selected or self.state is None:
            self.goal_details.setPlainText('选择子任务查看完整要求');return
        item=selected[0]
        while item.parent() is not None:item=item.parent()
        ident=item.data(0,QtCore.Qt.UserRole)
        goal=next((g for g in self.state.get('task_goals',[]) if (g.get('requirement') or {}).get('goal_id')==ident),None)
        self.goal_details.setPlainText('选择子任务查看完整要求' if goal is None else goal_requirements(goal,self.annotation_names))

    def update_goals(self):
        selected=self.goals.selectedItems();previous=None
        if selected:
            item=selected[0]
            while item.parent() is not None:item=item.parent()
            previous=item.data(0,QtCore.Qt.UserRole)
        self.goals.clear();chosen=None
        state=self.state or {};navigation=state.get('navigation') or {}
        for index,goal in enumerate(state.get('task_goals',[]),1):
            req=goal.get('requirement') or {};ident=req.get('goal_id')
            status=goal.get('status')
            if status=='pending' and navigation.get('goal_id')==ident and navigation.get('status')=='running':status='running'
            title=f"{index}. {req.get('description') or ident}"
            item=QtWidgets.QTreeWidgetItem([title,STATUS.get(status,str(status or ''))])
            item.setData(0,QtCore.Qt.UserRole,ident);item.setToolTip(0,goal_requirements(goal,self.annotation_names))
            self.goals.addTopLevelItem(item)
            target=req.get('target_id')
            if target:
                label=self.annotation_names.get(target,target)
                item.addChild(QtWidgets.QTreeWidgetItem(['目标：'+str(label),'']))
            if req.get('kind')=='circuit':
                item.addChild(QtWidgets.QTreeWidgetItem(['用户要求方向：'+{'clockwise':'顺时针','counterclockwise':'逆时针'}.get(req.get('direction'),'未指定'),'']))
            item.setExpanded(True)
            if ident==previous:chosen=item
        if chosen is None and self.goals.topLevelItemCount():chosen=self.goals.topLevelItem(0)
        if chosen is not None:self.goals.setCurrentItem(chosen)
        else:self.show_goal_requirements()

    def update_session(self,state,events):
        previous_goals=None if self.state is None else self.state.get('task_goals')
        previous_navigation=None if self.state is None else self.state.get('navigation')
        self.state=state
        if state is not None:
            self.current_task.setText('当前任务：'+str(state.get('instruction') or ''))
        if state is not None and (state.get('task_goals')!=previous_goals or state.get('navigation')!=previous_navigation):
            self.update_goals()
        for event in events:
            stamp=datetime.fromtimestamp(event['wall_s']).strftime('%H:%M:%S')
            if event['kind']=='reference_map_prepared':
                annotations=event['value'].get('annotations') or {}
                self.annotation_names={item[key]:item.get('name') or item[key]
                    for group,key in (('points','place_id'),('regions','region_id'))
                    for item in annotations.get(group,[]) if key in item}
                self.update_goals();continue
            if event['kind'] in OPERATION_EVENTS:
                item=QtWidgets.QTreeWidgetItem([stamp,OPERATION_EVENTS[event['kind']],display_value(event_summary(event))])
                item.setData(0,QtCore.Qt.UserRole,event['seq'])
                self.operation_events.addTopLevelItem(item)
                if self.process.state()==QtCore.QProcess.Running:self.operation_events.scrollToBottom()
                continue
            before=len(self.timeline.entries)
            if event['kind']=='api_request' and 'context' in event:
                self.active_tools=event['context']['tools']
                self.show_current_request(event,stamp)
                count=len(event['context']['messages'])
                images=sum(len(message['images']) for message in event['context']['messages'])
                item=QtWidgets.QTreeWidgetItem([stamp,'模型请求',f'{count} 条消息，{images} 张图片 · 双击查看整轮'])
                item.setData(0,QtCore.Qt.UserRole,event['seq'])
                self.context_tree.addTopLevelItem(item)
                self.timeline.request(event['context'],event['seq'])
            elif event['kind']=='api_response':
                self.timeline.response(event['value'].get('message'),event['seq'])
            elif event['kind']=='conversation_exchange':
                self.timeline.exchange(event['value'].get('messages',[]),event['seq'])
            else:continue
            self.update_timeline(stamp,before)
        self.refresh_status()

    def reader_error(self,message):self.notice.setText('会话读取暂不可用：'+message)

    def show_current_request(self,event,stamp):
        context=event['context'];messages=context['messages'];self.latest_request_seq=event['seq']
        self.current_tree.clear()
        self.request_summary.setText(f"最近请求 {stamp} · {len(messages)} 条消息 · {sum(len(m['images']) for m in messages)} 张图片")
        self.request_button.setEnabled(True)
        for index,message in enumerate(messages):
            label,color=message_label(message)
            item=QtWidgets.QTreeWidgetItem([str(index+1),label,display_value(message['summary'])])
            item.setData(0,QtCore.Qt.UserRole,event['seq']);item.setData(0,QtCore.Qt.UserRole+1,index)
            self.current_tree.addTopLevelItem(item)
            for column in (0,1,2):item.setForeground(column,QtGui.QBrush(QtGui.QColor(color)))
            for number,image_index in enumerate(message['images']):
                names=message.get('image_names',[])
                name=names[number] if number<len(names) else '输入图像'
                child=QtWidgets.QTreeWidgetItem(['','图片',name+' · 双击预览'])
                child.setData(0,QtCore.Qt.UserRole,event['seq']);child.setData(0,QtCore.Qt.UserRole+1,index)
                child.setData(0,QtCore.Qt.UserRole+2,image_index);item.addChild(child)
            if message['images']:item.setExpanded(True)

    def update_timeline(self,stamp,before):
        for message in self.timeline.entries[before:]:
            label,color=message_label(message)
            item=QtWidgets.QTreeWidgetItem([stamp,label,display_value(message['summary'])])
            item.setData(0,QtCore.Qt.UserRole,message['seq'])
            item.setData(0,QtCore.Qt.UserRole+1,message['index'])
            self.context_tree.addTopLevelItem(item)
            self.timeline_items.append((item,label,color))
        for (item,label,color),message in zip(self.timeline_items,self.timeline.entries):
            signature=(message['seq'],message['index'],tuple(message['images']),tuple(message.get('image_names',[])))
            if item.data(0,QtCore.Qt.UserRole+4)!=signature:
                item.takeChildren()
                item.setData(0,QtCore.Qt.UserRole+4,signature)
                for number,image_index in enumerate(message['images']):
                    name=message.get('image_names',[])
                    title=name[number] if number<len(name) else '输入图像'
                    image=QtWidgets.QTreeWidgetItem(['','图片',title+' · 双击预览'])
                    image.setData(0,QtCore.Qt.UserRole,message['seq'])
                    image.setData(0,QtCore.Qt.UserRole+1,message['index'])
                    image.setData(0,QtCore.Qt.UserRole+2,image_index)
                    item.addChild(image)
                if message['images']:item.setExpanded(True)
            brush=QtGui.QBrush(QtGui.QColor(color if message['active'] and not message['pending'] else '#9aa3b0'))
            status='等待下一次请求' if message['pending'] and message['active'] else ('最近请求中' if message['active'] else '历史')
            item.setText(1,label+' · '+status)
            item.setText(2,display_value(message['summary'])+('（模型输入采用精简表示）' if message.get('representation_changed') else ''))
            item.setData(0,QtCore.Qt.UserRole,message['seq'])
            item.setData(0,QtCore.Qt.UserRole+1,message['index'])
            for column in (0,1,2):item.setForeground(column,brush)
            for index in range(item.childCount()):
                for column in (0,1,2):item.child(index).setForeground(column,brush)
        if len(self.timeline.entries)>before and self.process.state()==QtCore.QProcess.Running:
            self.context_tree.scrollToBottom()

    def show_tools(self):
        from .tools.framework import native_tools
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('Agent 工具与参数');dialog.resize(950,650)
        outer=QtWidgets.QVBoxLayout(dialog)
        outer.addWidget(QtWidgets.QLabel('● 已包含于最近一次模型请求；○ 其他已实现工具。模型每轮可用工具可能不同。'))
        layout=QtWidgets.QHBoxLayout();outer.addLayout(layout)
        items=QtWidgets.QListWidget();detail=QtWidgets.QPlainTextEdit()
        detail.setReadOnly(True);layout.addWidget(items,1);layout.addWidget(detail,3)
        tools=native_tools()
        for tool in tools:
            name=tool['function']['name']
            items.addItem(('● ' if name in self.active_tools else '○ ')+name)
        items.currentRowChanged.connect(lambda row:detail.setPlainText(
            json.dumps(tools[row]['function'],ensure_ascii=False,indent=2) if row>=0 else ''))
        if tools:items.setCurrentRow(0)
        dialog.exec_()

    def open_detail(self,item,_column):
        if self.session_path is None or self.detail_reader is not None:return
        seq=item.data(0,QtCore.Qt.UserRole)
        self.detail_selection=(item.data(0,QtCore.Qt.UserRole+1),item.data(0,QtCore.Qt.UserRole+2))
        self.detail_reader=DetailReader(self.session_path,seq)
        self.detail_reader.ready.connect(self.show_detail)
        self.detail_reader.error.connect(lambda message:self.notice.setText('详情读取失败：'+message))
        self.detail_reader.finished.connect(self.detail_finished);self.detail_reader.start()

    def open_request_detail(self):
        if self.session_path is None or self.detail_reader is not None or self.latest_request_seq is None:return
        self.detail_selection=(None,None)
        self.detail_reader=DetailReader(self.session_path,self.latest_request_seq)
        self.detail_reader.ready.connect(self.show_detail)
        self.detail_reader.error.connect(lambda message:self.notice.setText('详情读取失败：'+message))
        self.detail_reader.finished.connect(self.detail_finished);self.detail_reader.start()

    def show_detail(self,kind,value,images):
        index,image_index=self.detail_selection or (None,None)
        value,images,selected=select_detail(kind,value,images,index,image_index)
        dialog=DetailDialog(kind,value,images,self,image_index=selected,message_only=index is not None);dialog.exec_()

    def detail_finished(self):
        self.detail_reader=None

    def show_target(self,value):
        self.target=value
        self.target_pixmap=None;self.target_image.clear();self.target_zoom.setEnabled(False)
        heading=('必须满足，容差 '+str(value.get('yaw_tolerance_degrees'))+'°' if value.get('yaw_required')
                 else '仅供参考，任务未要求对齐')
        route=value.get('route_waypoints')
        route_text=(f'共 {len(route)} 个点，顺序见 RViz 和示意图' if route else '单目标')
        missing=value.get('rviz_missing_height_indices') or []
        approx=value.get('rviz_approximate_height_indices') or []
        height_note=('\nRViz 未显示的点：'+str([i+1 for i in missing]) if missing else '')
        if approx:height_note+='\nRViz 高度近似的点：'+str([i+1 for i in approx])
        self.target_text.setText('目标：'+str(value.get('goal_id'))+'\n地图位姿：'+str(value.get('target_map_xy_yaw'))+
            '\n位置容差：'+str(value.get('position_tolerance_m'))+' m；朝向：'+heading+
            '\n建议路线：'+route_text+
            height_note+'\n路线仅供人工参考；请结合 RViz 和现场判断。')
        path=value.get('target_image')
        if path and self.session_path is not None:
            try:
                candidate=Path(path).resolve();candidate.relative_to(self.session_path.parent.resolve())
                if candidate.suffix.lower()=='.png':
                    pix=QtGui.QPixmap(str(candidate))
                    if not pix.isNull():
                        self.target_pixmap=pix;self.target_zoom.setEnabled(True)
                        self.target_image.setPixmap(pix.scaled(450,350,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            except (OSError,ValueError):pass

    def open_target_image(self):
        if self.target_pixmap is not None and not self.target_pixmap.isNull():
            ImageDialog(self.target_pixmap,'目标示意图',self).exec_()

    def set_status_text(self,text):
        if self.status_label.toPlainText()!=text:
            scroll=self.status_label.verticalScrollBar();position=scroll.value()
            self.status_label.setPlainText(text);scroll.setValue(position)

    def refresh_status(self):
        running=self.process.state()==QtCore.QProcess.Running
        fresh=running and self.status is not None and time.monotonic()-self.status_received<3.
        if fresh and self.pending and any(time.monotonic()-value['sent']>5. for value in self.pending.values()):
            self.pending.clear();self.notice.setText('操作确认超时，结果未确定；请核对当前状态。程序不会自动重试。')
        if fresh:
            s=self.status;pose=s.get('origin_pose')
            loc='有效' if s.get('localization_status')=='valid' and not s.get('input_stale') and pose is not None else '不可用或过期'
            route=''
            if s.get('route_index') is not None:
                route=f" ｜ 下一途经点 {s['route_index']+1}/{s.get('route_count')}（定位估计）"
            task=STATUS.get(s.get('status'),s.get('status'));nav=STATUS.get(s.get('navigation_status'),s.get('navigation_status'))
            stopped='是' if s.get('stopped') is True else '否' if s.get('stopped') is False else '未知'
            if s.get('reason')=='model_text_without_task_registration':
                task='模型只返回文字，任务尚未登记；请查看输出并答复或替换任务'
            preview=s.get('reference_preview') or {}
            index=s.get('route_index') if s.get('route_index') is not None else 0
            expected=str(s.get('call_id'))+':'+str(index)
            preview=preview if preview.get('owner')==s.get('call_id') and preview.get('request_id')==expected else {}
            preview_error=s.get('reference_preview_error')
            ref=('观测失效' if s.get('input_stale') or s.get('localization_status')!='valid' else
                 '不可用：'+str(preview_error) if preview_error else
                 '已生成' if preview.get('status')=='reference_ready' else
                 '未生成：'+str(preview.get('reason') or preview.get('status') or '等待目标'))
            self.set_status_text(f"人工驾驶 ｜ 定位：{loc} ｜ 位姿：{pose} ｜ Agent：{task} ｜ "
                f"静止估计：{stopped} ｜ 导航：{nav} ｜ 模型：{'请求中' if s.get('model_busy') else '空闲'}{route}"
                f" ｜ 参考路径：{ref} ｜ 定位源时间：{s.get('source_stamp_s')}")
        else:
            self.set_status_text('人工驾驶 ｜ '+('等待新状态或连接已失效' if running else '未连接；历史记录为只读'))
        busy=bool(self.pending);status=self.status.get('status') if fresh else None
        nav=self.status.get('navigation_status') if fresh else None
        for name,button in self.buttons.items():
            enabled=fresh and not busy
            if name=='arrived':enabled=enabled and nav=='running' and self.status.get('call_id') is not None
            elif name=='resume':enabled=enabled and status=='paused'
            elif name=='continue':enabled=enabled and status=='waiting_user'
            elif name=='answer':enabled=enabled and status=='needs_clarification'
            elif name=='pause':enabled=enabled and status not in ('paused','stopping','succeeded','failed','cancelled')
            elif name in ('cancel','replace'):enabled=enabled and status not in ('stopping','succeeded','failed','cancelled')
            button.setEnabled(bool(enabled))
        self.start_button.setEnabled(self.process.state()==QtCore.QProcess.NotRunning)
        self.history_button.setEnabled(self.process.state()==QtCore.QProcess.NotRunning)
        editable=self.process.state()==QtCore.QProcess.NotRunning
        for field in (self.socket,self.catalog,self.instruction,self.duration,*self.browse_buttons):field.setEnabled(editable)

    def send_command(self,name):
        if self.status is None or self.pending or time.monotonic()-self.status_received>=3.:return
        text=self.message.text().strip() if name in ('answer','replace') else None
        if name in ('answer','replace') and not text:self.notice.setText('请先输入答复或新任务。');return
        identity=uuid.uuid4().hex;s=self.status
        value=dict(command_id=identity,name=name,task_id=s['task_id'],revision=s['revision'],
                   goal_id=s['goal_id'],call_id=s['call_id'],text=text)
        if self.process.write((json.dumps(value,ensure_ascii=False)+'\n').encode('utf-8'))<0:
            self.notice.setText('命令发送失败，未自动重试。');return
        self.pending[identity]=dict(name=name,sent=time.monotonic());self.notice.setText('等待 Agent 确认操作……');self.refresh_status()

    def closeEvent(self,event):
        if self.process.state()!=QtCore.QProcess.NotRunning:
            if self.status is not None and not self.pending:self.send_command('cancel')
            if not self.process.waitForFinished(2000):
                self.process.terminate()
                if not self.process.waitForFinished(1000):self.process.kill();self.process.waitForFinished(1000)
        if self.reader is not None:
            self.reader.requestInterruption()
            if not self.reader.wait(2000):
                self.notice.setText('会话读取尚未结束，请稍后关闭。');event.ignore();return
        if self.detail_reader is not None and not self.detail_reader.wait(2000):
            self.notice.setText('事件详情仍在读取，请稍后关闭。');event.ignore();return
        event.accept()


def main():
    parser=argparse.ArgumentParser(description='人工驾驶 Agent 本机可视化')
    parser.add_argument('--runtime-socket',default='')
    parser.add_argument('--catalog',default='')
    parser.add_argument('--duration',type=int,default=1200)
    parser.add_argument('--session',default='',help='只读打开已有 session.sqlite')
    args=parser.parse_args()
    if not 1<=args.duration<=86400:parser.error('duration 必须在 1 至 86400 秒之间')
    if args.session and not Path(args.session).is_file():parser.error('session 文件不存在')
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window=ManualWindow(args.runtime_socket,args.catalog,args.duration)
    if args.session:
        window.set_reader(args.session)
        window.settings_toggle.setChecked(False)
        window.notice.setText('只读历史会话；图片是当时发送的观察，不代表当前实时画面。')
    window.show()
    return app.exec_()


if __name__=='__main__':
    raise SystemExit(main())
