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
    def __init__(self, python, gpu, settings, effective, tracking=None):
        from perception.detection_tracking.inference.config import ROOT
        self.temporary = tempfile.TemporaryDirectory(prefix='social_nav_detector_')
        self.process = subprocess.Popen([str(python), '-u', '-m', 'perception.detection_tracking.annotation.worker'],
            cwd=str(ROOT), env=detector_environment(python, gpu), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True)
        try:
            response = self.request(dict(kind='init', config=settings['detector'],
                effective_hash=hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest(),
                tracking=tracking))
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

    def _frame_request(self, kind, frame, points, pose=None):
        path = Path(self.temporary.name) / 'input.npy'
        np.save(str(path), points, allow_pickle=False)
        if pose is not None and (pose['source_id'] != frame['source_id'] or pose['query_stamp_ns'] != frame['stamp_ns']):
            raise ValueError('检测位姿与原始扫描绑定不一致')
        response = self.request(dict(kind=kind, id=frame['id'], source_id=frame['source_id'],
            stamp_s=frame['stamp_ns'] / 1e9, sensor_frame=frame['sensor_frame'],
            fingerprint=frame['fingerprint'], points=str(path), pose=pose,
            clock_domain=frame.get('clock_domain','dataset'), world_frame=frame.get('world_frame','dataset_world'),
            timestamp_source=frame.get('timestamp_source','sensor')))
        if response['id'] != frame['id'] or response['fingerprint'] != frame['fingerprint']:
            raise ValueError('检测响应与原始扫描不匹配')
        self.last_input_context = response['input_context']
        return response

    def predict(self, frame, points, pose=None):
        return self._frame_request('predict', frame, points, pose)['boxes']

    def observe(self, frame, points, pose):
        """Warm history when resuming an already predicted prefix; no network forward."""
        return self._frame_request('observe', frame, points, pose)['input_context']

    def track(self, frame, points, pose):
        return self._frame_request('track', frame, points, pose)['observation']

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
    from perception.detection_tracking.inference.pipeline import Frame, PerceptionPipeline
    from perception.detection_tracking.inference.config import project_path
    from .data import read_yaml

    detector, pipeline = None, None
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
                    if request.get('tracking') is not None:
                        pipeline = PerceptionPipeline(detector, request['tracking'])
                    result = dict(kind='ready', metadata=detector.metadata)
                elif request['kind'] in ('predict', 'observe', 'track'):
                    if detector is None:
                        raise ValueError('模型尚未就绪')
                    points = np.load(request['points'], allow_pickle=False)
                    frame = Frame(request['id'], request['source_id'], request['stamp_s'], points,
                        sensor_frame=request['sensor_frame'], clock_domain=request.get('clock_domain','dataset'),
                        world_frame=request.get('world_frame','dataset_world'),
                        timestamp_source=request.get('timestamp_source','sensor'))
                    if frame.input_fingerprint() != request['fingerprint']:
                        raise ValueError('检测输入与请求指纹不一致')
                    pose = request.get('pose')
                    if pose is not None:
                        frame.world_frame = pose.get('frame_id') or 'dataset_world'
                        frame.pose_identity = (pose['source_id'], pose.get('session_id'), pose.get('generation'))
                        if pose['valid']:
                            frame.world_from_lidar = np.asarray(pose['local_from_lidar'])
                            frame.pose_source, frame.pose_health = 'localization', 'valid'
                            frame.pose_stamp_s = pose['stamp_ns'] / 1e9
                    observation = None
                    if request['kind'] == 'track':
                        if pipeline is None:
                            raise ValueError('Tracking was not configured for this worker')
                        observation = pipeline.step(frame)
                        boxes = observation['detections']
                    elif request['kind'] == 'predict':
                        boxes = detector.predict(frame).tolist()
                    else:
                        detector.prepare_frame(frame)
                        boxes = None
                    result = dict(kind='prediction', id=request['id'], fingerprint=request['fingerprint'],
                                  boxes=boxes, metadata=detector.metadata,
                                  input_context=detector.last_input_context)
                    if observation is not None:
                        result['observation'] = observation
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
