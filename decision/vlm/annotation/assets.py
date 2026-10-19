"""Bounded, persistent system-ROS reader; only requested frames are materialized."""
from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import threading
import time
import zipfile


ROOT = Path(__file__).resolve().parents[3]


class LazyAssets:
    def __init__(self, project, python):
        self.project, self.python = str(project), python
        self.process, self.cache = None, OrderedDict()
        self.lock = threading.RLock()

    def _start(self):
        if os.name != 'posix':
            raise ValueError('原bag按需读取在Linux执行；Windows请使用--server连接')
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT), '/opt/ros/noetic/lib/python3/dist-packages'])
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, ['/opt/ros/noetic/lib', env.get('LD_LIBRARY_PATH', '')]))
        env['PYTHONNOUSERSITE'] = '1'
        self.process = subprocess.Popen([self.python, '-B', '-m', 'decision.vlm.annotation.ros_reader',
                                         '--worker', self.project], cwd=str(ROOT), env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read(self, size, deadline):
        result = bytearray()
        fd = self.process.stdout.fileno()
        while len(result) < size:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError('按需读取bag超时；未使用旧帧替代')
            block = os.read(fd, min(1024*1024, size-len(result)))
            if not block:
                raise ValueError('ROS读取进程已退出，请检查原bag与系统ROS解释器')
            result.extend(block)
        return bytes(result)

    def read(self, ident):
        with self.lock:
            if ident in self.cache:
                self.cache.move_to_end(ident)
                return self.cache[ident]
            if self.process is None or self.process.poll() is not None:
                self.close()
                self._start()
            try:
                self.process.stdin.write((json.dumps({'frame_id': ident})+'\n').encode())
                self.process.stdin.flush()
                deadline = time.monotonic()+30
                size, = struct.unpack('!I', self._read(4, deadline))
                if not 0 < size <= 32*1024*1024:
                    raise ValueError('按需帧长度无效')
                payload = self._read(size, deadline)
                if payload[:1] == b'E':
                    raise ValueError(json.loads(payload[1:])['detail'])
                with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
                    if sum(i.file_size for i in bundle.infolist()) > 32*1024*1024:
                        raise ValueError('按需帧解包大小超限')
                    result = dict(metadata=json.loads(bundle.read('frame.json')),
                                  cloud=bundle.read('cloud.npy'), rgb=bundle.read('rgb.png') if 'rgb.png' in bundle.namelist() else None,
                                  preview=bundle.read('preview.jpg') if 'preview.jpg' in bundle.namelist() else None)
                self.cache[ident] = result
                if len(self.cache) > 48:
                    self.cache.popitem(last=False)
                return result
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(3)
                self.process.stdin.close()
                self.process.stdout.close()
                self.process = None
