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
from .ui_data import connect_readonly,event_detail,event_summary,poll_session

STATUS={'ready':'准备决策','waiting_navigation':'等待人工驾驶或到达确认','waiting_observation':'等待新观察',
        'waiting_user':'等待用户继续','needs_clarification':'等待用户答复','paused':'已暂停',
        'stopping':'正在结束任务','cancelled':'已取消','succeeded':'已完成','failed':'未完成',
        'idle':'空闲','running':'进行中',
        'blocked':'受阻','refused':'未接收','pending':'待执行','completed':'已完成'}
EVENT_LABELS={'conversation_exchange':'对话','conversation_trimmed':'上下文裁剪','tool_intent':'工具调用',
              'model_action_applied':'工具结果','api_request':'模型请求','api_response':'模型返回',
              'api_failure':'模型请求失败','model_failed':'模型错误','model_protocol_rejected':'模型格式错误',
              'model_result_discarded':'过期模型结果','manual_arrival_confirmation':'人工到达确认',
              'manual_goal_evidence_wait_started':'等待到达证据','manual_goal_evidence_wait_finished':'到达证据等待结果',
              'manual_goal_evidence_refresh_failed':'到达证据读取失败','manual_goal_evidence_unproven':'到达证据不足',
              'operator_command':'窗口操作结果',
              'manual_reference_status':'参考路径预览',
              'manual_route_progress':'路线进度','manual_route_trace':'路线检查',
              'navigation_completed':'导航阶段完成','navigation_ack':'目标已接收',
              'navigation_cancel_confirmed':'目标已取消','navigation_context_lost':'定位或地图变化',
              'task_connection_lost':'连接中断','user_event':'用户操作','task_created':'创建任务',
              'assistant_message':'模型文字回复','agent_run_finished':'会话结束'}


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


class DetailDialog(QtWidgets.QDialog):
    def __init__(self,kind,value,images,parent=None):
        super().__init__(parent);self.setWindowTitle('事件详情 · '+kind);self.resize(1000,700)
        layout=QtWidgets.QVBoxLayout(self)
        if kind=='api_request':
            layout.addWidget(QtWidgets.QLabel('这是该次模型请求实际包含的消息；图像来自该次请求，不代表当前画面。'))
        split=QtWidgets.QSplitter();layout.addWidget(split)
        text=QtWidgets.QPlainTextEdit();text.setReadOnly(True)
        text.setPlainText(json.dumps(value,ensure_ascii=False,indent=2));split.addWidget(text)
        if images:
            panel=QtWidgets.QWidget();right=QtWidgets.QVBoxLayout(panel)
            choices=QtWidgets.QListWidget();picture=QtWidgets.QLabel('选择输入图像');picture.setAlignment(QtCore.Qt.AlignCenter)
            picture.setMinimumSize(320,240);picture.setWordWrap(True)
            right.addWidget(choices);right.addWidget(picture,1)
            for name,_ in images:choices.addItem(name)
            def show(row):
                if row<0:return
                pix=QtGui.QPixmap();pix.loadFromData(images[row][1])
                if pix.isNull():picture.setText('无法解码该图像');return
                picture.setPixmap(pix.scaled(560,480,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            choices.currentRowChanged.connect(show);choices.setCurrentRow(0);split.addWidget(panel)


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
        self._build(socket,catalog,duration)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh_status);self.timer.start(500)

    def _build(self,socket,catalog,duration):
        central=QtWidgets.QWidget();self.setCentralWidget(central);outer=QtWidgets.QVBoxLayout(central)
        self.banner=QtWidgets.QLabel('人工驾驶 · 观察会话不连接底盘 · 任务取消不会让手柄停车')
        self.banner.setStyleSheet('font-weight:bold;color:#8a4b00;padding:6px;background:#fff3d4')
        outer.addWidget(self.banner)
        form=QtWidgets.QGridLayout();outer.addLayout(form)
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
        outer.addWidget(QtWidgets.QLabel('任务、地图、当前局部图及 RGB 可能发送至已配置的托管模型服务。'))
        self.status_label=QtWidgets.QLabel('未连接');self.status_label.setWordWrap(True);outer.addWidget(self.status_label)
        split=QtWidgets.QSplitter();outer.addWidget(split,1)
        left=QtWidgets.QWidget();left.setMinimumWidth(380);left_layout=QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel('任务与进度'))
        self.current_task=QtWidgets.QLabel('当前任务：无');self.current_task.setWordWrap(True);left_layout.addWidget(self.current_task)
        self.goals=QtWidgets.QTreeWidget();self.goals.setHeaderLabels(['阶段','状态'])
        self.goals.header().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch)
        self.goals.header().setSectionResizeMode(1,QtWidgets.QHeaderView.ResizeToContents)
        self.goals.setWordWrap(True);self.goals.setTextElideMode(QtCore.Qt.ElideNone);left_layout.addWidget(self.goals,2)
        self.target_text=QtWidgets.QLabel('尚无目标');self.target_text.setWordWrap(True);left_layout.addWidget(self.target_text)
        self.target_image=QtWidgets.QLabel('目标示意图');self.target_image.setAlignment(QtCore.Qt.AlignCenter)
        self.target_image.setMinimumSize(300,220);left_layout.addWidget(self.target_image,2)
        split.addWidget(left)
        right=QtWidgets.QWidget();right_layout=QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel('完整会话事件（双击查看详情；模型请求可查看该轮输入）'))
        self.events=QtWidgets.QTreeWidget();self.events.setHeaderLabels(['时间','类型','摘要'])
        self.events.setColumnWidth(0,85);self.events.setColumnWidth(1,190)
        self.events.itemDoubleClicked.connect(self.open_detail);right_layout.addWidget(self.events,1)
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
        self.session_path=Path(path);self.state=None;self.status=None;self.events.clear();self.goals.clear()
        self.current_task.setText('当前任务：无')
        self.target=None;self.target_text.setText('尚无目标');self.target_image.clear()
        self.reader=SessionReader(path);self.reader.update.connect(self.update_session);self.reader.error.connect(self.reader_error)
        self.reader.start()
        return True

    def open_history(self):
        if self.process.state()!=QtCore.QProcess.NotRunning:
            self.notice.setText('当前任务仍在运行，请先结束再打开历史会话。');return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择 session.sqlite',str(ROOT/'outputs/decision/agent'),'SQLite (*.sqlite)')
        if path:
            if self.set_reader(path):self.notice.setText('只读历史会话；不会自动恢复或调用模型。')

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

    def update_session(self,state,events):
        self.state=state
        if state is not None:
            self.current_task.setText('当前任务：'+str(state.get('instruction') or ''))
            self.goals.clear()
            for goal in state.get('task_goals',[]):
                req=goal.get('requirement') or {}
                item=QtWidgets.QTreeWidgetItem([str(req.get('description') or req.get('goal_id')),
                    STATUS.get(goal.get('status'),str(goal.get('status','')))])
                self.goals.addTopLevelItem(item)
        for event in events:
            stamp=datetime.fromtimestamp(event['wall_s']).strftime('%H:%M:%S')
            item=QtWidgets.QTreeWidgetItem([stamp,EVENT_LABELS.get(event['kind'],event['kind']),event_summary(event)])
            item.setData(0,QtCore.Qt.UserRole,event['seq']);self.events.addTopLevelItem(item)
        if events:self.events.scrollToBottom()
        self.refresh_status()

    def reader_error(self,message):self.notice.setText('会话读取暂不可用：'+message)

    def open_detail(self,item,_column):
        if self.session_path is None or self.detail_reader is not None:return
        seq=item.data(0,QtCore.Qt.UserRole)
        self.detail_reader=DetailReader(self.session_path,seq)
        self.detail_reader.ready.connect(self.show_detail)
        self.detail_reader.error.connect(lambda message:self.notice.setText('详情读取失败：'+message))
        self.detail_reader.finished.connect(self.detail_finished);self.detail_reader.start()

    def show_detail(self,kind,value,images):
        dialog=DetailDialog(kind,value,images,self);dialog.exec_()

    def detail_finished(self):
        self.detail_reader=None

    def show_target(self,value):
        self.target=value
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
                    if not pix.isNull():self.target_image.setPixmap(pix.scaled(450,350,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            except (OSError,ValueError):pass

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
            self.status_label.setText(f"人工驾驶 ｜ 定位：{loc} ｜ 位姿：{pose} ｜ Agent：{task} ｜ "
                f"静止估计：{stopped} ｜ 导航：{nav} ｜ 模型：{'请求中' if s.get('model_busy') else '空闲'}{route}"
                f" ｜ 参考路径：{ref} ｜ 定位源时间：{s.get('source_stamp_s')}")
        else:
            self.status_label.setText('人工驾驶 ｜ '+('等待新状态或连接已失效' if running else '未连接；历史记录为只读'))
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
    args=parser.parse_args()
    if not 1<=args.duration<=86400:parser.error('duration 必须在 1 至 86400 秒之间')
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window=ManualWindow(args.runtime_socket,args.catalog,args.duration);window.show()
    return app.exec_()


if __name__=='__main__':
    raise SystemExit(main())
