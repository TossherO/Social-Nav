"""Record raw live sensors during one Agent task, without blocking its event loop."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .. import ROOT


class TaskBagRecorder:
    def __init__(self,output,session,emit,*,output_root='/home/tl/rosbag',popen=subprocess.Popen):
        self.output=Path(output).resolve();self.session=session;self.emit=emit
        self.output_root=str(Path(output_root).expanduser().absolute())
        self.popen=popen;self.process=None;self.revision=None;self.status_file=None
        self.log_file=None;self.started_at=None;self.ready=False;self.stopping=False
        self.failed=False;self.last_event=None;self.error=None

    def _event(self,status,**extra):
        value=dict(event='agent_bag_recording',status=status,revision=self.revision,**extra)
        key=json.dumps(value,ensure_ascii=False,sort_keys=True)
        if key!=self.last_event:
            self.last_event=key;self.session.store.event('agent_bag_recording',value);self.emit(value)

    def _status(self):
        if self.status_file is None or not self.status_file.is_file():return {}
        try:
            value=json.loads(self.status_file.read_text(encoding='utf-8'))
            return value if isinstance(value,dict) else {}
        except (OSError,ValueError):return {}

    def _write_failure(self,error):
        if self.status_file is None:return
        current=self._status()
        value=dict(status='failed',error=error,**{k:current[k] for k in ('bag','capture_report') if k in current})
        temporary=self.status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False)+'\n',encoding='utf-8')
        temporary.replace(self.status_file)

    def start(self):
        self.revision=self.session.state['revision'];self.ready=self.stopping=self.failed=False
        self.error=None;self.started_at=time.monotonic();self.last_event=None
        self.status_file=self.output/('bag_recording_%s.json'%self.revision)
        self.log_file=self.output/('bag_recording_%s.log'%self.revision)
        if self.status_file.exists() or self.log_file.exists():
            raise ValueError('Task bag status or log already exists')
        master=getattr(self.session.navigation,'ros_master_uri',None)
        if not isinstance(master,str) or not master.startswith('http://'):
            raise ValueError('观察会话未提供 ROS master 地址，请重启观察程序')
        command=['bash',str(ROOT/'perception/capture/record_navigation.sh'),
                 '--output-root',self.output_root,'--status-file',str(self.status_file),
                 '--parent-pid',str(os.getpid()),'--note','Agent session: '+str(self.output)]
        with self.log_file.open('x',encoding='utf-8') as log:
            self.process=self.popen(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env=dict(os.environ,SOCIAL_NAV_RECORD_ROS_MASTER_URI=master))
        self._event('preparing',log=str(self.log_file))

    def stop(self):
        if self.process is not None and not self.stopping:
            self.stopping=True
            if self.process.poll() is None:
                try:self.process.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
            self._event('finalizing',log=str(self.log_file),bag=self._status().get('bag'))

    def update(self):
        if self.process is None:return
        state=self._status();status=state.get('status')
        if self.session.state['revision']!=self.revision or self.session.state['status'] in ('succeeded','partial','failed','cancelled'):
            self.stop()
        code=self.process.poll()
        if status=='recording' and code is None and not self.stopping:
            self.ready=True
            self._event('recording',bag=state.get('bag'),capture_report=state.get('capture_report'))
        elif status in ('failed','needs_review') and not self.stopping:
            self.failed=True;self.error=state.get('error') or 'recording_needs_review'
            self._event('failed',error=self.error,bag=state.get('bag'),log=str(self.log_file))
        elif not self.ready and not self.stopping and time.monotonic()-self.started_at>30:
            self.failed=True;self.error='recording_preparation_timeout'
            self._event('failed',error=self.error,log=str(self.log_file));self.stop()
        if code is not None:
            if not self.stopping:
                self.failed=True;self.error=state.get('error') or 'recorder_exited_unexpectedly'
                if status!='failed':self._write_failure(self.error)
                self._event('failed',error=self.error,bag=state.get('bag'),log=str(self.log_file))
            else:
                self._event(status if status in ('saved','needs_review','failed') else 'finished',
                            bag=state.get('bag'),capture_report=state.get('capture_report'),exit_code=code)
            if self.session.state['revision']!=self.revision:
                self.process=None;self.revision=None;self.ready=False;self.stopping=False

    def before_request(self,_actor):
        self.update()
        if not self.session.task_goals.goals:return True
        if self.process is None and self.revision is not None and self.revision!=self.session.state['revision']:
            self.revision=None;self.failed=False;self.error=None
        if self.process is None and not self.failed:
            try:self.start()
            except (OSError,ValueError) as error:
                self.failed=True;self.error=str(error)[:300]
                try:self._write_failure(self.error)
                except OSError:pass
                self._event('failed',error=self.error,log=str(self.log_file))
        if self.failed and not self.ready:
            if self.session.state['status']=='ready':
                self.session.stop('needs_clarification','agent_bag_recording_unavailable')
            return False
        return self.ready and not self.stopping

    def close(self):
        # The capture process remains independent while it closes, indexes and
        # hashes the bag. It also watches this parent if the Agent is killed.
        self.stop()
