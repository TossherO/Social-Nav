"""ROS-free annotation state/storage; bag imports are lazy. System Python compatible."""
import copy
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid

import numpy as np
import yaml

from perception.detection_tracking.inference.config import ROOT, load_config, project_path
from perception.sensors.pointcloud import input_fingerprint, pointcloud_fields


ANNOTATION_STATES = frozenset(('draft', 'skipped', 'complete'))
FRAME_STATUS_NAMES = dict(unreviewed='非标注帧', draft='待标', skipped='跳过', complete='已标')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def decode_cloud(cloud):
    fields = pointcloud_fields(cloud, ('x', 'y', 'z', 'intensity'))
    points = np.stack([fields[k] for k in ('x', 'y', 'z', 'intensity')], axis=1).astype(np.float32)
    if not np.isfinite(points).all(axis=1).any():
        raise ValueError('扫描不含有效 XYZI 点，不能标为无人')
    return points


def fingerprint(path):
    """Bounded source identity, not a whole-file integrity claim."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode() + b'\0')
    with path.open('rb') as stream:
        digest.update(stream.read(65536))
        if size > 65536:
            stream.seek(max(65536, size - 65536))
            digest.update(stream.read())
    return dict(kind='size_and_edge_sha256', size=size, sha256=digest.hexdigest())


def read_yaml(path):
    path = Path(path)
    value = yaml.safe_load(path.read_text())
    base = value.pop('_BASE_CONFIG_', None)
    result = read_yaml(path.parent / base) if base else {}

    def merge(target, source):
        for key, item in source.items():
            if isinstance(item, dict) and isinstance(target.get(key), dict):
                merge(target[key], item)
            else:
                target[key] = item
    merge(result, value)
    return result


def detector_settings(config=None, checkpoint=None):
    settings = load_config(config)
    if checkpoint:
        settings['detector']['checkpoint'] = str(Path(checkpoint).expanduser().resolve())
    effective = read_yaml(project_path(settings['detector']['config']))
    sweeps = effective['DATA_CONFIG']['MAX_SWEEPS']
    if type(sweeps) is not int or sweeps not in (1, 2, 3, 4) or effective['CLASS_NAMES'] != ['Pedestrian']:
        raise ValueError('标注工具要求1～4帧 Pedestrian 检测配置')
    return settings, effective


def safe_directory(path):
    path = Path(path).expanduser().absolute()
    resolved = path.resolve()
    protected = [ROOT / 'models']
    aliases = ROOT / 'data/datasets'
    if aliases.is_dir():
        protected.extend(p.resolve() for p in aliases.iterdir() if p.exists())
    for parent in protected:
        parent = parent.resolve()
        if resolved == parent or parent in resolved.parents:
            raise ValueError('人工工程／导出不能写入共享数据别名或模型目录')
    return path


def blank_review():
    return dict(status='unreviewed', decisions={}, centers=[], note='')


def pending_candidate_ids(review, candidates, min_score=0.):
    """Batch actions include only displayed pending proposals, leaving hidden labels intact."""
    return [str(i) for i, box in enumerate(candidates)
            if box[7] >= min_score and review['decisions'].get(str(i)) in ('pending', 'uncertain')]


def annotation_step(rows, current, interval_s, direction):
    """Find a marked frame within the step, otherwise browse a real scan."""
    if direction not in (-1, 1) or not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError('采样步长须为正有限数，方向须为前或后')
    if not 0 <= current < len(rows):
        return None
    indices = range(current + direction, len(rows) if direction > 0 else -1, direction)
    indices = list(indices)
    if not indices:
        return None
    origin = rows[current]
    step_ns = max(1, round(interval_s * 1e9))
    nearby = [i for i in indices if rows[i]['status'] in ANNOTATION_STATES and
              0 <= direction * (rows[i]['record_ns'] - origin['record_ns']) <= step_ns]
    if nearby:
        index = min(nearby, key=lambda i: (abs(rows[i]['record_ns'] - origin['record_ns']), abs(i - current)))
        return index
    target_ns = origin['record_ns'] + direction * step_ns
    index = min(indices, key=lambda i: (abs(rows[i]['record_ns'] - target_ns), abs(i - current)))
    return index


def resolve_sequence(sources, name):
    """Choose an existing bag without confusing its ID with evaluation groups."""
    matches = [s['id'] for s in sources if name in
               (s['id'], s['path'], Path(s['path']).name, Path(s['path']).stem)]
    if len(matches) != 1:
        raise ValueError('子包序列不存在或名称不唯一，请使用完整路径或来源ID：' + name)
    return matches[0]


def in_scope(xy, bounds):
    xy = np.asarray(xy, dtype=float)
    return (xy.shape == (2,) and np.isfinite(xy).all()
            and bounds[0] <= xy[0] <= bounds[3] and bounds[1] <= xy[1] <= bounds[4])


def has_human_processing(review):
    return bool(review['centers']) or any(d in ('accepted', 'rejected', 'uncertain') for d in review['decisions'].values())


def edit_review(review, operation, bounds, candidates=(), **values):
    """Pure edit; one candidate produces at most one human center. No proximity merging."""
    result = copy.deepcopy(review)
    if operation == 'decision':
        ident, decision = values['candidate'], values['decision']
        if decision not in ('pending', 'accepted', 'rejected', 'uncertain'):
            raise ValueError('未知候选状态')
        box = candidates[int(ident)]
        existing = [c for c in result['centers'] if c.get('candidate') == ident]
        result['decisions'][ident] = decision
        if decision != 'accepted' or not existing:
            result['centers'] = [c for c in result['centers'] if c.get('candidate') != ident]
            if decision == 'accepted':
                result['centers'].append(dict(id=uuid.uuid4().hex, xy=list(box[:2]), candidate=ident))
    elif operation == 'add':
        result['centers'].append(dict(id=uuid.uuid4().hex, xy=list(values['xy']), candidate=None))
    elif operation == 'move':
        found = next((c for c in result['centers'] if c['id'] == values['center']), None)
        if found is None:
            raise ValueError('中心不存在')
        found['xy'] = list(values['xy'])
    elif operation == 'delete':
        found = next((c for c in result['centers'] if c['id'] == values['center']), None)
        if found is None:
            raise ValueError('中心不存在')
        if found.get('candidate') is not None:
            result['decisions'][found['candidate']] = 'rejected'
        result['centers'].remove(found)
    elif operation == 'note':
        result['note'] = str(values['note'])
    elif operation == 'skip':
        result['status'] = 'skipped'
    elif operation == 'select':
        pass  # Browsing is never annotation work.
    elif operation == 'pending':
        result['status'] = 'draft' if has_human_processing(result) else 'unreviewed'
    else:
        raise ValueError('未知编辑操作')
    if operation not in ('select', 'pending', 'skip', 'note'):
        if all(result[key] == review[key] for key in ('decisions', 'centers')):
            return copy.deepcopy(review)
        result['status'] = 'draft' if has_human_processing(result) else 'unreviewed'
    validate_review(result, bounds)
    return result


def complete_review(review, bounds, candidates, min_score):
    """Prepare an explicitly confirmed final frame, never deciding hidden proposals."""
    visible = pending_candidate_ids(review, candidates, min_score)
    hidden = len(pending_candidate_ids(review, candidates)) - len(visible)
    if hidden:
        raise ValueError('仍有 %d 个低于显示阈值的待复核候选，请降低阈值后确认／拒绝；不会自动处理隐藏项' % hidden)
    result = copy.deepcopy(review)
    for ident in visible:
        result = edit_review(result, 'decision', bounds, candidates, candidate=ident, decision='accepted')
    result['status'] = 'complete'
    validate_review(result, bounds)
    return result


def empty_review(review, bounds, candidates):
    """Explicitly confirm no people in scope, including hidden model proposals."""
    result = copy.deepcopy(review)
    result['centers'] = []
    result['decisions'] = {str(i): 'rejected' for i, box in enumerate(candidates)
                           if in_scope(box[:2], bounds)}
    result['status'] = 'complete'
    validate_review(result, bounds)
    return result


def validate_review(review, bounds):
    if review['status'] not in ('unreviewed', 'draft', 'complete', 'skipped'):
        raise ValueError('未知帧状态')
    ids, candidate_ids = set(), set()
    for center in review['centers']:
        if center['id'] in ids or not in_scope(center['xy'], bounds):
            raise ValueError('中心重复、非有限或位于复核范围外')
        ids.add(center['id'])
        cid = center.get('candidate')
        if cid is not None:
            if cid in candidate_ids or review['decisions'].get(cid) != 'accepted':
                raise ValueError('候选与最终中心不一致')
            candidate_ids.add(cid)
    for cid, decision in review['decisions'].items():
        if decision not in ('pending', 'accepted', 'rejected', 'uncertain'):
            raise ValueError('未知候选状态')
        if decision == 'accepted' and cid not in candidate_ids:
            raise ValueError('已接受候选缺少人工中心')
    if review['status'] == 'complete' and any(
            x in ('pending', 'uncertain') for x in review['decisions'].values()):
        raise ValueError('仍有待复核候选，不能标记完整')


def initial_review(candidates, bounds):
    review = blank_review()
    review['decisions'] = {str(i): 'pending' for i, box in enumerate(candidates) if in_scope(box[:2], bounds)}
    return review


class Store:
    def __init__(self, project, create=False, settings=None, effective=None, read_only=False):
        self.path = safe_directory(project)
        database = self.path / 'annotations.sqlite'
        if create:
            if self.path.exists():
                raise FileExistsError('新工程目录已存在；继续标注请不要使用 --bag 新建同名目录')
            self.path.mkdir(parents=True)
        elif not database.is_file():
            raise FileNotFoundError('标注工程不存在：' + str(database))
        self.connection = (sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=20)
                           if read_only else sqlite3.connect(str(database), timeout=20))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA foreign_keys=ON')
        if create:
            self.connection.executescript('''
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE sources (id TEXT PRIMARY KEY, path TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    topic TEXT NOT NULL, rgb_topic TEXT NOT NULL, root_group TEXT NOT NULL,
                    location_group TEXT NOT NULL, split TEXT NOT NULL, indexed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE frames (id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                    ordinal INTEGER NOT NULL, stamp_ns INTEGER NOT NULL, record_ns INTEGER NOT NULL,
                    record_rank INTEGER NOT NULL, sensor_frame TEXT NOT NULL,
                    fingerprint TEXT, candidates TEXT, model TEXT, review TEXT NOT NULL,
                    UNIQUE(source_id,ordinal));
                CREATE INDEX frames_time ON frames(stamp_ns);
            ''')
            self.set_meta('settings', settings)
            self.set_meta('effective', effective)
        self.bounds = self.meta('effective')['DATA_CONFIG']['POINT_CLOUD_RANGE']

    def close(self):
        self.connection.close()

    def coordinated(self):
        return self.connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotation_writer'").fetchone() is not None

    def assert_local_writer(self):
        if self.coordinated() and self.connection.execute(
                'SELECT 1 FROM annotation_writer WHERE expires>?', (time.time(),)).fetchone():
            raise ValueError('工程已由远程写入会话占用；请先在远程客户端完成同步并关闭，再本机编辑')

    def source_state(self, ident):
        row = self.connection.execute('SELECT epoch,revision FROM annotation_sources WHERE source_id=?', (ident,)).fetchone() if self.coordinated() else None
        return dict(row) if row is not None else dict(epoch=0, revision=0)

    def bump_source(self, ident, reset=False):
        if self.coordinated():
            self.connection.execute('INSERT OR IGNORE INTO annotation_sources VALUES (?,0,0)', (ident,))
            self.connection.execute('UPDATE annotation_sources SET revision=revision+1,epoch=epoch+? WHERE source_id=?', (int(reset), ident))

    def bump_frame(self, ident):
        if self.coordinated():
            self.connection.execute('INSERT OR IGNORE INTO annotation_frames VALUES (?,0)', (ident,))
            self.connection.execute('UPDATE annotation_frames SET revision=revision+1 WHERE frame_id=?', (ident,))

    def meta(self, key):
        row = self.connection.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key, value):
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, dumps(value)))

    def save_position(self, frame):
        positions = self.meta('last_frames') or {}
        positions[frame['source_id']] = frame['id']
        with self.connection:
            self.connection.executemany('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                [('last_frame', dumps(frame['id'])), ('last_frames', dumps(positions))])

    def sources(self):
        return [dict(dict(r), **self.source_state(r['id'])) for r in self.connection.execute('SELECT * FROM sources ORDER BY rowid')]

    def add_bag(self, path, topic, rgb_topic, root_group='', location_group='', split='unassigned'):
        path = Path(path).expanduser().resolve()
        identity = dumps(fingerprint(path))
        for source in self.sources():
            if source['fingerprint'] == identity and source['topic'] == topic:
                with self.connection:
                    self.connection.execute('BEGIN IMMEDIATE')
                    self.assert_local_writer()
                    self.connection.execute('UPDATE sources SET path=? WHERE id=?', (str(path), source['id']))
                return source['id']
        ident = uuid.uuid4().hex[:12]
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            self.assert_local_writer()
            self.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,0)',
                (ident, str(path), identity, topic, rgb_topic, root_group or path.stem, location_group, split))
        return ident

    def source(self, ident):
        row = self.connection.execute('SELECT * FROM sources WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return dict(dict(row), **self.source_state(ident))

    def set_group(self, ident, root_group, location_group, split):
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            self.assert_local_writer()
            self._set_group(ident, root_group, location_group, split)

    def _set_group(self, ident, root_group, location_group, split):
        if split not in ('unassigned', 'train', 'val', 'holdout'):
            raise ValueError('未知划分')
        self.connection.execute('UPDATE sources SET root_group=?,location_group=?,split=? WHERE id=?',
                                (root_group.strip(), location_group.strip(), split, ident))
        self.bump_source(ident)

    def rows(self):
        if self.coordinated():
            return [dict(row) for row in self.connection.execute(
                'SELECT f.id,f.source_id,f.ordinal,f.stamp_ns,f.record_ns,f.review,'
                'COALESCE(r.revision,0) AS revision,COALESCE(s.epoch,0) AS epoch FROM frames f '
                'LEFT JOIN annotation_frames r ON r.frame_id=f.id '
                'LEFT JOIN annotation_sources s ON s.source_id=f.source_id ORDER BY f.source_id,f.ordinal')]
        return [dict(r) for r in self.connection.execute('SELECT id,source_id,ordinal,stamp_ns,record_ns,review '
                                                       'FROM frames ORDER BY source_id,ordinal')]

    def frame(self, ident):
        if self.coordinated():
            row = self.connection.execute('SELECT f.*,COALESCE(r.revision,0) AS revision,COALESCE(s.epoch,0) AS epoch '
                'FROM frames f LEFT JOIN annotation_frames r ON r.frame_id=f.id '
                'LEFT JOIN annotation_sources s ON s.source_id=f.source_id WHERE f.id=?', (ident,)).fetchone()
        else:
            row = self.connection.execute('SELECT *,0 AS revision,0 AS epoch FROM frames WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        result = dict(row)
        for key in ('candidates', 'model', 'review'):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        return result

    def bind_points(self, frame, points):
        value = input_fingerprint(points, frame['source_id'], frame['id'],
                                  frame['stamp_ns'] / 1e9, frame['sensor_frame'])
        if frame['fingerprint'] is not None and frame['fingerprint'] != value:
            raise ValueError('扫描内容已变化，不能继续使用既有标注')
        if frame['fingerprint'] is None:
            with self.connection:
                self.connection.execute('BEGIN IMMEDIATE')
                self.assert_local_writer()
                self.connection.execute('UPDATE frames SET fingerprint=? WHERE id=?', (value, frame['id']))
        return value

    def save_review(self, ident, review, expected=None):
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            self.assert_local_writer()
            self._save_review(ident, review, expected)

    def _save_review(self, ident, review, expected=None):
        validate_review(review, self.bounds)
        if review['status'] == 'complete' and not self.frame(ident)['fingerprint']:
            raise ValueError('必须先读取有效扫描，才能确认完整／无人')
        if expected is None:
            self.connection.execute('UPDATE frames SET review=? WHERE id=?', (dumps(review), ident))
        else:
            result = self.connection.execute('UPDATE frames SET review=? WHERE id=? AND review=?',
                                             (dumps(review), ident, dumps(expected)))
            if result.rowcount != 1:
                raise ValueError('标注已被其他窗口修改，请重新读取该帧；没有覆盖其结果')
        self.bump_frame(ident)
        source = self.connection.execute('SELECT source_id FROM frames WHERE id=?', (ident,)).fetchone()
        if source is None:
            raise KeyError(ident)
        self.bump_source(source['source_id'])

    def reset_source_reviews(self, ident):
        """Explicitly discard ONE bag's human work, retaining all immutable input/proposals."""
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            self.assert_local_writer()
            return self._reset_source_reviews(ident)

    def _reset_source_reviews(self, ident):
        if not self.connection.execute('SELECT 1 FROM sources WHERE id=?', (ident,)).fetchone():
            raise KeyError('子包不存在：' + ident)
        updates, count = [], 0
        for row in self.connection.execute('SELECT id,candidates,review FROM frames WHERE source_id=?', (ident,)):
            count += 1
            boxes = json.loads(row['candidates']) if row['candidates'] is not None else []
            review = initial_review(boxes, self.bounds)
            if review != json.loads(row['review']):
                updates.append((dumps(review), row['id']))
                self.bump_frame(row['id'])
        self.connection.executemany('UPDATE frames SET review=? WHERE id=?', updates)
        self.bump_source(ident, reset=True)
        return dict(frames=count, cleared=len(updates))

    def save_candidates(self, ident, boxes, metadata, expected_fingerprint):
        frame = self.frame(ident)
        boxes = np.asarray(boxes, dtype=float)
        if boxes.size == 0:
            boxes = np.empty((0, 8), dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 8:
            raise ValueError('候选必须为 N×8 的三维框与分数')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0) or np.any((boxes[:, 7] < 0) | (boxes[:, 7] > 1)):
            raise ValueError('非法检测框／分数')
        if not expected_fingerprint or expected_fingerprint != frame['fingerprint']:
            raise ValueError('候选与当前扫描指纹不一致')
        if frame['candidates'] is not None:
            if frame['model'] == metadata and frame['candidates'] == boxes.tolist():
                return
            raise ValueError('本帧已有候选缓存；不同模型请新建工程，不覆盖人工复核依据')
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            self.assert_local_writer()
            fresh = self.frame(ident)
            if fresh['candidates'] is not None:
                if fresh['model'] == metadata and fresh['candidates'] == boxes.tolist():
                    return
                raise ValueError('本帧已有不同候选缓存，不能覆盖')
            review = fresh['review']
            # Read human work INSIDE the write transaction; late proposals cannot
            # replace a concurrent manual edit made after the first frame read.
            if review == blank_review():
                review = initial_review(boxes, self.bounds)
            self.connection.execute('UPDATE frames SET candidates=?,model=?,review=? WHERE id=?',
                                    (dumps(boxes.tolist()), dumps(metadata), dumps(review), ident))
            self.bump_frame(ident)
            self.bump_source(frame['source_id'])


class BagReader:
    def __init__(self, source):
        import rosbag
        self.source = source
        if dumps(fingerprint(source['path'])) != source['fingerprint']:
            raise ValueError('bag 来源已变化，请使用新工程')
        self.bag = rosbag.Bag(source['path'], 'r')

    def close(self):
        self.bag.close()

    def read(self, frame, rgb=True, max_rgb_age_s=.1):
        import genpy
        begin = genpy.Time(frame['record_ns'] // 1000000000, frame['record_ns'] % 1000000000)
        end_ns = frame['record_ns'] + 1
        end = genpy.Time(end_ns // 1000000000, end_ns % 1000000000)
        rank, cloud = 0, None
        for _, message, recorded in self.bag.read_messages(topics=[self.source['topic']], start_time=begin, end_time=end):
            if recorded.to_nsec() != frame['record_ns']:
                continue
            if rank == frame['record_rank']:
                cloud = message
                break
            rank += 1
        if cloud is None or cloud.header.stamp.to_nsec() != frame['stamp_ns'] or cloud.header.frame_id != frame['sensor_frame']:
            raise ValueError('无法找到原始同一扫描，拒绝使用邻帧替代')
        points = decode_cloud(cloud)
        image, image_ns = None, None
        self.image_notice = ''
        if rgb and self.source['rgb_topic']:
            start = genpy.Time.from_sec(max(0, frame['record_ns'] / 1e9 - 1))
            stop = genpy.Time.from_sec(frame['record_ns'] / 1e9 + 1)
            best = None
            for _, message, _ in self.bag.read_messages(topics=[self.source['rgb_topic']], start_time=start, end_time=stop):
                stamp = message.header.stamp.to_nsec()
                if 0 <= frame['stamp_ns'] - stamp <= max_rgb_age_s * 1e9 and (image_ns is None or stamp > image_ns):
                    best, image_ns = message, stamp
            if best is not None:
                from perception.sensors.image import rgb_array
                try:
                    image = rgb_array(best)
                except (ValueError, AttributeError) as error:
                    self.image_notice = 'RGB 未显示：' + str(error)
        return points, image, image_ns


class FrameLoader:
    """Serialized I/O with persistent bag handles and bounded raw-asset LRU, never label caching."""
    def __init__(self, project, max_frames=16, max_bytes=64 * 1024 * 1024, read_only=False):
        self.project, self.max_frames, self.max_bytes = project, max_frames, max_bytes
        self.read_only = read_only
        self.readers, self.cache = OrderedDict(), OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def read(self, ident):
        with self.lock:
            store = Store(self.project, read_only=self.read_only)
            try:
                frame = store.frame(ident)
                source = store.source(frame['source_id'])
                stat = Path(source['path']).stat()
                signature = (source['path'], stat.st_size, stat.st_mtime_ns)
                item = self.readers.get(source['id'])
                if item is not None and item[0] != signature:
                    item[1].close()
                    del self.readers[source['id']]
                    self.cache.clear()
                    self.bytes = 0
                    item = None
                if item is None:
                    item = signature, BagReader(source)
                    self.readers[source['id']] = item
                    while len(self.readers) > 2:
                        _, old = self.readers.popitem(last=False)
                        old[1].close()
                self.readers.move_to_end(source['id'])
                cached = self.cache.get(ident)
                if cached is None:
                    points, image, image_ns = item[1].read(frame)
                    if self.read_only:
                        bound = input_fingerprint(points, frame['source_id'], frame['id'], frame['stamp_ns'] / 1e9, frame['sensor_frame'])
                        if frame['fingerprint'] != bound:
                            raise ValueError('远程读取需要已绑定的原始扫描；请先完成本机预计算')
                    else:
                        bound = store.bind_points(frame, points)
                    cached = points, image, image_ns, item[1].image_notice, bound
                    size = points.nbytes + (image.nbytes if image is not None else 0)
                    self.cache[ident] = cached
                    self.bytes += size
                    while len(self.cache) > self.max_frames or self.bytes > self.max_bytes:
                        _, old = self.cache.popitem(last=False)
                        self.bytes -= old[0].nbytes + (old[1].nbytes if old[1] is not None else 0)
                else:
                    if frame['fingerprint'] != cached[4]:
                        raise ValueError('缓存资产与当前标签绑定不一致')
                    self.cache.move_to_end(ident)
                # Fetch fresh review/candidates after I/O; human edits never come from the LRU.
                return store.frame(ident), cached[0], cached[1], cached[2], cached[3]
            finally:
                store.close()

    def close(self):
        with self.lock:
            for _, reader in self.readers.values():
                reader.close()
            self.readers.clear()
            self.cache.clear()
            self.bytes = 0


def index_source(project, ident, progress=None):
    store = Store(project)
    reader = None
    try:
        source = store.source(ident)
        if source['indexed']:
            return
        reader = BagReader(source)
        rows, last_record, rank = [], None, 0
        for ordinal, (_, cloud, recorded) in enumerate(reader.bag.read_messages(topics=[source['topic']])):
            if cloud._type != 'sensor_msgs/PointCloud2' or cloud.header.stamp.to_nsec() <= 0 or not cloud.header.frame_id:
                raise ValueError('只支持具有有效采集时间、坐标系的 PointCloud2')
            ns = recorded.to_nsec()
            rank = rank + 1 if ns == last_record else 0
            last_record = ns
            rows.append((f'{ident}/{ordinal:08d}', ident, ordinal, cloud.header.stamp.to_nsec(), ns, rank,
                         cloud.header.frame_id, dumps(blank_review())))
            if progress and ordinal % 200 == 0:
                progress('正在索引 %s：%d 帧' % (Path(source['path']).name, ordinal + 1))
        if not rows:
            raise ValueError('bag 中没有指定点云话题')
        if dumps(fingerprint(source['path'])) != source['fingerprint']:
            raise ValueError('索引期间 bag 发生变化')
        with store.connection:
            store.connection.execute('BEGIN IMMEDIATE')
            store.assert_local_writer()
            store.connection.executemany('INSERT INTO frames '
                '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,review) VALUES (?,?,?,?,?,?,?,?)', rows)
            store.connection.execute('UPDATE sources SET indexed=1 WHERE id=?', (ident,))
    finally:
        if reader:
            reader.close()
        store.close()


def import_detections(project, path, progress=None):
    """Bind legacy floating-time caches by their ORIGINAL frame fingerprint, not timestamp alone."""
    path = Path(path)
    settings_path = path.parent / 'settings.json'
    if not settings_path.is_file():
        settings_path = path.parent / 'manifest.json'
    if not settings_path.is_file():
        raise ValueError('候选导入需要同目录 settings.json 或 manifest.json 记录模型来源')
    metadata = json.loads(settings_path.read_text())
    detector = metadata.get('detector')
    if not isinstance(detector, dict) or not detector.get('checkpoint_sha256'):
        raise ValueError('缓存缺少 detector.checkpoint_sha256，不能自动复用')
    store, readers, count = Store(project), {}, 0
    try:
        if detector.get('max_sweeps') != 1 or list(detector.get('point_cloud_range', [])) != list(store.bounds):
            raise ValueError('缓存必须记录匹配复核范围的单帧检测模型')
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if 'detections' not in row or not row.get('input_fingerprint') or row.get('health') == 'invalid':
                    continue
                stamp_ns = round(float(row['stamp_s']) * 1e9)
                matches = list(store.connection.execute('SELECT id FROM frames WHERE stamp_ns BETWEEN ? AND ? AND sensor_frame=?',
                    (stamp_ns - 1000, stamp_ns + 1000, row['detection_frame_id'])))
                if not matches:
                    continue
                if len(matches) != 1:
                    raise ValueError('缓存时间对应多个扫描，不能自动消歧')
                frame = store.frame(matches[0]['id'])
                if frame['source_id'] not in readers:
                    readers[frame['source_id']] = BagReader(store.source(frame['source_id']))
                points, _, _ = readers[frame['source_id']].read(frame, rgb=False)
                expected = input_fingerprint(points, row['sequence_id'], row['observation_id'],
                                             row['stamp_s'], row['detection_frame_id'])
                if expected != row['input_fingerprint']:
                    raise ValueError('历史候选与实际扫描不匹配，未导入该帧')
                bound = store.bind_points(frame, points)
                store.save_candidates(frame['id'], row['detections'], detector, bound)
                count += 1
                if progress and count % 50 == 0:
                    progress('已绑定 %d 帧候选' % count)
        if not count:
            raise ValueError('没有与工程匹配的候选；不会按最近帧借用')
        return count
    finally:
        for reader in readers.values():
            reader.close()
        store.close()
