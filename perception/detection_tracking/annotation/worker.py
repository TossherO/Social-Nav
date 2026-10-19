"""Persistent single-GPU detector, JSON lines on stdout, logs on stderr. No ROS/Qt."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def detector_environment(python, gpu):
    from perception.detection_tracking.inference.config import ROOT
    environment = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        environment.pop(key, None)
    environment.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       CUDA_VISIBLE_DEVICES=str(gpu))
    environment['PATH'] = str(Path(python).parent) + os.pathsep + environment.get('PATH', '')
    return environment


class DetectorClient:
    """Blocking offline client. One model and one temporary array, no ROS in the child."""
    def __init__(self, python, gpu, settings, effective):
        from perception.detection_tracking.inference.config import ROOT
        self.temporary = tempfile.TemporaryDirectory(prefix='social_nav_detector_')
        self.process = subprocess.Popen([str(python), '-u', '-m', 'perception.detection_tracking.annotation.worker'],
            cwd=str(ROOT), env=detector_environment(python, gpu), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True)
        try:
            response = self.request(dict(kind='init', config=settings['detector'],
                effective_hash=hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()))
            self.metadata = response['metadata']
        except BaseException:
            self.close()
            raise

    def request(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('检测子进程退出，未自动重试')
        response = json.loads(line)
        if response['kind'] == 'error':
            raise RuntimeError(response['message'])
        return response

    def predict(self, frame, points):
        path = Path(self.temporary.name) / 'input.npy'
        np.save(str(path), points, allow_pickle=False)
        response = self.request(dict(kind='predict', id=frame['id'], source_id=frame['source_id'],
            stamp_s=frame['stamp_ns'] / 1e9, sensor_frame=frame['sensor_frame'],
            fingerprint=frame['fingerprint'], points=str(path)))
        if response['id'] != frame['id'] or response['fingerprint'] != frame['fingerprint']:
            raise ValueError('检测响应与原始扫描不匹配')
        return response['boxes']

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"kind":"quit"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.temporary.cleanup()


def main():
    from perception.detection_tracking.inference.detector import SimpleDet3dDetector
    from perception.detection_tracking.inference.pipeline import Frame
    from perception.detection_tracking.inference.config import project_path
    from .data import read_yaml

    detector = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['kind'] == 'init':
                    if detector is not None:
                        raise ValueError('模型已加载，不能重复初始化')
                    current = read_yaml(project_path(request['config']['config']))
                    actual = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
                    if actual != request['effective_hash']:
                        raise ValueError('模型配置已变化；请明确选择配置并新建工程')
                    detector = SimpleDet3dDetector(request['config'])
                    result = dict(kind='ready', metadata=detector.metadata)
                elif request['kind'] == 'predict':
                    if detector is None:
                        raise ValueError('模型尚未就绪')
                    points = np.load(request['points'], allow_pickle=False)
                    frame = Frame(request['id'], request['source_id'], request['stamp_s'], points,
                                  sensor_frame=request['sensor_frame'])
                    if frame.input_fingerprint() != request['fingerprint']:
                        raise ValueError('检测输入与请求指纹不一致')
                    boxes = detector.predict(frame)
                    result = dict(kind='prediction', id=request['id'], fingerprint=request['fingerprint'],
                                  boxes=boxes.tolist(), metadata=detector.metadata)
                elif request['kind'] == 'quit':
                    return 0
                else:
                    raise ValueError('未知子进程请求')
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps(dict(kind='error', id=request.get('id'),
                                  message=type(error).__name__ + ': ' + str(error)), ensure_ascii=False), flush=True)
            return 1  # No hidden retry or switch of model/environment.
    return 0


if __name__ == '__main__':
    sys.exit(main())
