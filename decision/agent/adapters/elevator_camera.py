"""Read independent RGB and local LiDAR observations from the ROS bridge."""
import base64
import copy
import json
import math
import subprocess
import threading
import time


class ElevatorCamera:
    def __init__(self):
        self.process=None
        self.thread=None
        self.lock=threading.Lock()
        self.latest=None
        self.scan=None
        self.error=None

    def start(self, base_from_lidar):
        if self.process is not None:raise ValueError('Camera bridge already started')
        self.process=subprocess.Popen(
            ['bash','-c','source perception/scripts/ros1_env.sh && exec /usr/bin/python3 -B -m decision.agent.workflows.elevator_camera "$@"',
             'elevator-sensors','--base-from-lidar',json.dumps(base_from_lidar.tolist(),allow_nan=False)],
            stdout=subprocess.PIPE,text=True,bufsize=1)
        self.thread=threading.Thread(target=self._read,daemon=True,name='elevator-camera')
        self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith('SCAN '):
                    if len(line)>2*1024*1024:raise ValueError('Local scan image exceeds size limit')
                    record=json.loads(line[5:])
                    data=base64.b64decode(record.pop('image_base64'),validate=True)
                    points=record.pop('base_xy_dm')
                    if (record['source']!='local_lidar_returns' or record['display_frame']!='base_link' or
                            record['clock_domain']!='ros_time' or not record['frame_id'] or
                            not math.isfinite(record['stamp_s']) or record['stamp_s']<=0 or
                            not data.startswith(b'\x89PNG\r\n\x1a\n')):
                        raise ValueError('Invalid local scan')
                    with self.lock:
                        if self.scan and record['stamp_s']<self.scan[0]['stamp_s']:
                            raise ValueError('LiDAR source time regressed')
                        self.scan=(record,data,time.monotonic(),points)
                    continue
                if not line.startswith('FRAME '):continue
                if len(line)>2*1024*1024:raise ValueError('Camera frame exceeds size limit')
                record=json.loads(line[6:])
                if (record.get('source')!='ros_camera_independent_of_localization' or record.get('clock_domain')!='ros_time' or
                    not record.get('frame_id') or type(record.get('stamp_s')) not in (int,float) or
                    not math.isfinite(record['stamp_s']) or record['stamp_s']<=0 or
                    type(record.get('sequence')) is not int or record['sequence']<0 or
                    record.get('mime_type')!='image/jpeg'):
                    raise ValueError('Invalid camera frame metadata')
                image=base64.b64decode(record.pop('image_base64'),validate=True)
                if not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9'):
                    raise ValueError('Invalid JPEG frame')
                with self.lock:
                    if self.latest is not None and record['sequence']<=self.latest['record']['sequence']:
                        raise ValueError('Camera sequence regressed')
                    if self.latest is not None and record['stamp_s']<self.latest['record']['stamp_s']:
                        raise ValueError('Camera source time regressed')
                    self.latest=dict(record=record,image=image,received_mono=time.monotonic())
        except (ValueError,KeyError,TypeError) as error:
            with self.lock:self.error=type(error).__name__

    def snapshot(self, *, max_receive_age_s=2.):
        poll=getattr(self.process,'poll',None)
        if callable(poll) and poll() is not None:
            return dict(status='unavailable',reason='elevator_sensor_bridge_exited')
        with self.lock:value=copy.deepcopy(self.latest);error=self.error
        if error is not None:return dict(status='unavailable',reason='invalid_camera_stream')
        if value is None:return dict(status='unavailable',reason='camera_frame_missing')
        age=time.monotonic()-value['received_mono']
        if age<0 or age>max_receive_age_s:return dict(status='unavailable',reason='camera_frame_stale')
        result=dict(status='available',record=value['record'],image=value['image'],receive_age_s=age)
        with self.lock:scan=self.scan
        if (scan and 0<=time.monotonic()-scan[2]<=max_receive_age_s and
                abs(scan[0]['stamp_s']-value['record']['stamp_s'])<=max_receive_age_s):
            result.update(scan_record=scan[0],scan_image=scan[1],base_xy_dm=scan[3])
        return result

    def close(self):
        if self.process is None:return
        self.process.terminate()
        try:self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.thread is not None:self.thread.join(timeout=2.)
        with self.lock:self.latest=self.scan=None
