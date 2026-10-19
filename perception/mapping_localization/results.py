"""Read-only result data and playback timing, independent of ROS/Qt/model runtimes."""
from bisect import bisect_right
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np


def file_fingerprint(path):
    """Bounded size/edge fingerprint, portable across copies; not a whole-file hash."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode() + b'\0')
    with path.open('rb') as stream:
        digest.update(stream.read(65536))
        if size > 65536:
            stream.seek(max(65536, size - 65536))
            digest.update(stream.read())
    return {'kind': 'size_and_edge_sha256', 'size': size, 'sha256': digest.hexdigest()}


def map_fingerprint(path):
    path = Path(path)
    if not (path / 'COMPLETE').is_file():
        raise ValueError('Map has no COMPLETE marker: ' + str(path))
    names = ('manifest.yaml', 'keyframes/poses.txt', 'GlobalMap.pcd') if (path / 'GlobalMap.pcd').is_file() else ('manifest.json', 'map.pt')
    return {name: file_fingerprint(path / name) for name in names}


def rpy_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


def checked_pose(position, quaternion):
    if position is None or quaternion is None:
        return None
    p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        return None
    if abs(np.linalg.norm(q)-1) > 1e-3:
        return None
    return p, q


@dataclass
class Result:
    path: Path
    rows: list
    metadata: dict = field(default_factory=dict)
    map_dir: object = None
    notices: list = field(default_factory=list)
    kind: str = 'recorded_states'

    def __post_init__(self):
        stamps = [float(row['stamp_s']) for row in self.rows]
        if any(not math.isfinite(t) for t in stamps) or any(b < a for a, b in zip(stamps, stamps[1:])):
            raise ValueError('Acquisition time is nonfinite or reversed; split sessions rather than reorder records')
        self.times = np.asarray(stamps) - (stamps[0] if stamps else 0)
        for row in self.rows:
            if row.get('health') is not None and (type(row['health']) is not int or row['health'] not in (0, 1, 2)):
                raise ValueError('Unknown recorded health value')
            if row.get('health') == 0 and self.pose(row, 'odom') is None:
                # Old keyframe/CSV sources explicitly mark their available frame.
                if self.pose(row, 'map') is None:
                    raise ValueError('A VALID record has no finite unit-quaternion pose')
            if row.get('health') == 0 and row.get('has_map_pose') and self.pose(row, 'map') is None:
                raise ValueError('A VALID record advertises an invalid map pose')
        if self.map_dir is not None:
            expected = self.metadata.get('playback', {}).get('map_fingerprint')
            if expected is not None and map_fingerprint(self.map_dir) != expected:
                raise ValueError('Map no longer matches the result binding')

    @property
    def duration(self):
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def default_frame(self):
        return 'map' if any(row.get('has_map_pose') for row in self.rows) or self.map_dir is not None else 'odom'

    def index_at(self, elapsed):
        return max(0, bisect_right(self.times, elapsed)-1) if self.rows else -1

    @staticmethod
    def pose(row, frame):
        if frame == 'map':
            return checked_pose(row.get('map_position'), row.get('map_quaternion_xyzw')) if row.get('has_map_pose') else None
        return checked_pose(row.get('position'), row.get('quaternion_xyzw'))

    def frame_name(self, frame):
        names = {row.get('map_frame' if frame == 'map' else 'frame_id', frame) for row in self.rows
                 if self.pose(row, frame) is not None and row.get('health') in (0, None)}
        if len(names) > 1 or any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Inconsistent coordinate frames; do not overlay unrelated poses')
        return next(iter(names), frame)

    def trajectory(self, frame, limit=50000):
        """Sample display geometry, retaining breaks at failures/session/map changes."""
        if limit < 2:
            raise ValueError('Trajectory display budget must be at least two')
        stride = max(1, math.ceil(len(self.rows)/max(2, limit//2)))
        segments, identity = [], None
        last_selected = None
        for i, row in enumerate(self.rows):
            pose = self.pose(row, frame)
            current_identity = (row.get('source'), row.get('session_id'), row.get('generation'),
                                row.get('map_revision') if frame == 'map' else None)
            if pose is None or row.get('health') not in (0, None):
                last_selected = None
                continue
            if current_identity != identity:
                last_selected = None
            identity = current_identity
            if last_selected is None or i % stride == 0 or i == len(self.rows)-1:
                if last_selected is not None:
                    segments.append((i, last_selected, pose[0], row.get('health')))
                last_selected = pose[0]
        return segments


def _keyframes(directory):
    import yaml
    manifest = yaml.safe_load((directory / 'manifest.yaml').read_text())
    frame_name = manifest.get('frame')
    if not isinstance(frame_name, str) or not frame_name:
        raise ValueError('Saved keyframes have no recorded map frame')
    rows = []
    for line in (directory / 'keyframes/poses.txt').read_text().splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise ValueError('Expected finite nine-column keyframe poses')
        rows.append(dict(stamp_s=values[0], health=None, status='KEYFRAME_HEALTH_NOT_RECORDED', reason='',
                         source='saved_keyframes', session_id=None, generation=None, map_revision=None,
                         has_map_pose=True, map_frame=frame_name, map_position=values[1:4],
                         map_quaternion_xyzw=rpy_quaternion(*values[4:7])))
    return rows


def load_result(path, *, phase=None, backend=None, map_override=None):
    path = Path(path).resolve()
    directory = path.parent if path.is_file() else path
    notices, metadata, map_dir, kind = [], {}, None, 'recorded_states'
    rows = []
    comparison = (directory / 'comparison.json').is_file() and (directory / 'trajectories.csv').is_file()
    if (phase or backend) and not comparison:
        raise ValueError('--phase/--backend select an existing comparison series, not an algorithm')
    if path.is_file() and path.suffix == '.jsonl':
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        map_dir = Path(map_override).resolve() if map_override else None
        notices.append('独立轨迹文件未绑定地图来源。')
    elif (directory / 'comparison.json').is_file() and (directory / 'trajectories.csv').is_file():
        if not phase or not backend:
            raise ValueError('Comparison results require --phase and --backend to select one trajectory')
        if map_override:
            raise ValueError('Comparison maps are from separate runs; view a map directory separately')
        with (directory / 'trajectories.csv').open(newline='') as stream:
            for raw in csv.DictReader(stream):
                if raw['phase'] != phase or raw['backend'] != backend:
                    continue
                row = dict(stamp_s=float(raw['stamp_s']), health=int(raw['health']),
                           status=raw['status'], reason=raw['reason'], source=backend,
                           session_id=None, generation=None, map_revision=int(raw['map_revision']),
                           frame_id='odom', child_frame_id='', has_map_pose=bool(raw['map_x']), map_frame='map')
                for prefix, pkey, qkey in (('odom', 'position', 'quaternion_xyzw'),
                                          ('map', 'map_position', 'map_quaternion_xyzw')):
                    if raw[prefix+'_x']:
                        row[pkey] = [float(raw[prefix+'_'+axis]) for axis in ('x', 'y', 'z')]
                        row[qkey] = [float(raw[prefix+'_q'+axis]) for axis in ('x', 'y', 'z', 'w')]
                rows.append(row)
        if not rows:
            raise ValueError('No trajectory matches the selected phase/backend')
        notices.append('对比历史轨迹；地图来源独立，未自动叠加。原会话与代次未记录，轨迹不是外部真值。')
        kind = 'comparison_trajectory'
    elif (directory / 'run.json').is_file():
        metadata = json.loads((directory / 'run.json').read_text())
        if map_override and metadata.get('mode') == 'odometry':
            raise ValueError('Local odometry has no map alignment; do not overlay a prior map')
        recording = (directory / metadata.get('playback', {}).get('records', 'localization.jsonl')).resolve()
        if directory not in recording.parents:
            raise ValueError('Playback records must stay inside their run directory')
        if recording.is_file():
            with recording.open() as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        if map_override:
            map_dir = Path(map_override).resolve()
        elif metadata.get('mode') == 'mapping' and (directory / 'map/COMPLETE').is_file():
            map_dir = directory / 'map'
        elif metadata.get('map'):
            original = Path(metadata['map']).expanduser()
            map_dir = original if original.is_absolute() else directory / original
            if not (map_dir / 'COMPLETE').is_file():
                notices.append('原地图路径不可达；使用--map重定位同一地图，当前只显示轨迹。')
                map_dir = None
        if metadata.get('success') is False:
            notices.append('该次运行未成功；仅查看实际保存的记录。')
        person_records = directory/'people.jsonl'
        has_people = person_records.is_file() and person_records.stat().st_size > 0
        if not rows and not has_people and map_dir is not None and (map_dir / 'keyframes/poses.txt').is_file():
            if metadata.get('mode') == 'mapping':
                rows = _keyframes(map_dir)
                kind = 'optimized_keyframes'
            else:
                kind = 'map_only'
                notices.append('本次没有保存定位轨迹，仅查看输入地图；建图关键帧不充当定位结果。')
        if not rows and map_dir is None and not has_people:
            raise ValueError('This run kept only a summary; a full trajectory cannot be recovered')
    elif (directory / 'COMPLETE').is_file():
        map_dir = directory
        if map_override:
            raise ValueError('--map relocation applies to a run directory')
        if (directory / 'keyframes/poses.txt').is_file():
            rows, kind = _keyframes(directory), 'optimized_keyframes'
        else:
            kind = 'map_only'
    else:
        raise ValueError('Select a run directory, complete map, localization.jsonl or comparison directory')
    if kind == 'optimized_keyframes':
        notices.append('最终优化关键帧的稀疏轨迹；逐帧健康及建图几何历史未记录。')
    elif rows:
        notices.append('历史观测位姿；最终地图不代表每个历史时刻的地图。')
    if map_dir is not None:
        if not (map_dir / 'COMPLETE').is_file():
            raise ValueError('Select a complete map')
        if metadata and not metadata.get('playback', {}).get('map_fingerprint'):
            notices.append('旧结果未记录地图指纹；地图关联仅来自原路径。')
        if rows and not any(row.get('has_map_pose') for row in rows):
            raise ValueError('Trajectory has no map-frame poses; cannot overlay this map')
        if (map_dir / 'map.pt').is_file():
            notices.append('PIN神经点几何显示；不运行模型或解码SDF表面。')
    return Result(path, rows, metadata, map_dir, notices, kind)


def read_pcd(path, limit=250000):
    """Project ASCII/binary PCD; mmap binary input and cap display before copying."""
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            if stream.tell() > 65536:
                raise ValueError('PCD header exceeds budget')
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete PCD header')
            words = line.decode('ascii').strip().split()
            if not words or words[0].startswith('#'):
                continue
            header[words[0]] = words[1:]
            if words[0] == 'DATA':
                break
        offset = stream.tell()
        names = header['FIELDS']
        sizes = [int(x) for x in header['SIZE']]
        counts = [int(x) for x in header.get('COUNT', ['1']*len(names))]
        if not len(names) == len(sizes) == len(counts) == len(header['TYPE']) or any(n < 1 for n in counts):
            raise ValueError('Inconsistent PCD fields')
        n = int(header['POINTS'][0])
        if n <= 0 or limit < 1 or len(set(names)) != len(names):
            raise ValueError('Invalid/empty PCD or display budget')
        for axis in ('x', 'y', 'z'):
            if counts[names.index(axis)] != 1:
                raise ValueError('PCD coordinates must be scalar')
        step = max(1, math.ceil(n/limit))
        if header['DATA'] == ['binary']:
            formats = []
            for kind, size, count in zip(header['TYPE'], sizes, counts):
                dtype = np.dtype('<' + {'F': 'f', 'I': 'i', 'U': 'u'}[kind] + str(size))
                formats.append(dtype if count == 1 else (dtype, (count,)))
            data = np.memmap(path, dtype=np.dtype(list(zip(names, formats))), mode='r', offset=offset, shape=(n,))[::step]
            points = np.column_stack([data[axis] for axis in ('x', 'y', 'z')])
        elif header['DATA'] == ['ascii']:
            data = np.loadtxt(stream, ndmin=2)
            if data.shape != (n, sum(counts)):
                raise ValueError('PCD payload disagrees with header')
            columns = np.cumsum([0]+counts[:-1])
            points = data[::step, [columns[names.index(axis)] for axis in ('x', 'y', 'z')]]
        else:
            raise ValueError('Viewer supports project ASCII/binary PCD, not binary_compressed')
    return np.asarray(points[np.isfinite(points).all(axis=1)], dtype=np.float32)


def map_points(directory, limit, backend_python=None):
    directory = Path(directory)
    if (directory / 'GlobalMap.pcd').is_file():
        return read_pcd(directory / 'GlobalMap.pcd', limit)
    if not backend_python:
        raise ValueError('PIN map geometry requires explicit --backend-python; extraction runs on CPU only')
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')
    root = Path(__file__).resolve().parents[2]
    code = ('import sys,io,numpy as np; from perception.mapping_localization.backends.pin_slam.map_io import load_map; '
            'p,_=load_map(sys.argv[1]); a=p["neural_points"].numpy(); '
            'a=a[::max(1,(len(a)+int(sys.argv[2])-1)//int(sys.argv[2]))]; '
            'b=io.BytesIO(); np.save(b,a,allow_pickle=False); sys.stdout.buffer.write(b.getvalue())')
    completed = subprocess.run([str(backend_python), '-c', code, str(directory), str(limit)],
                               cwd=root, env=env, capture_output=True, timeout=60)
    if completed.returncode:
        raise ValueError('PIN geometry read failed: ' + completed.stderr.decode(errors='replace')[-2000:])
    return np.load(io.BytesIO(completed.stdout), allow_pickle=False)


class Timeline:
    def __init__(self, duration, rate=1.0, now=None):
        self.duration, self.position, self.playing = float(duration), 0.0, False
        self.rate = float(rate)
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError('Playback rate must be finite and positive')
        self.anchor = time.monotonic() if now is None else now

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if self.playing:
            self.position = min(self.duration, self.position+max(0, now-self.anchor)*self.rate)
            if self.position >= self.duration:
                self.playing = False
        self.anchor = now
        return self.position

    def seek(self, value, now=None):
        if not math.isfinite(value):
            raise ValueError('Seek time must be finite')
        self.position = min(self.duration, max(0, float(value)))
        self.anchor = time.monotonic() if now is None else now

    def toggle(self, now=None):
        self.tick(now)
        if not self.playing and self.position >= self.duration:
            self.position = 0.0
        self.playing = not self.playing and self.duration > 0

    def set_rate(self, rate, now=None):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Playback rate must be finite and positive')
        self.tick(now)
        self.rate = float(rate)
