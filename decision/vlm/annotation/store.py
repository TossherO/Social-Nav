"""Separate immutable observations and transactional event labels."""
from contextlib import contextmanager
import copy
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import time
import threading
import uuid

from .rules import digest, dumps, person_at, require, validate_links, validate_t1, validate_t2
from decision.vlm.scene_inputs import build_window, model_eligible, MODEL_BACKGROUND


def perception_view():
    """Current deployment ROI is the fixed display policy, not historical truth."""
    import hashlib
    import yaml
    root = Path(__file__).resolve().parents[3]
    policy = json.loads((root/'perception/detection_tracking/configs/inference.json').read_text())
    path = Path(policy['detector']['config'])
    if not path.is_absolute():path = root/path
    content = path.read_bytes()
    bounds = yaml.safe_load(content)['DATA_CONFIG']['POINT_CLOUD_RANGE']
    require(len(bounds) == 6, '检测配置缺少完整点云范围')
    from decision.vlm.scene_inputs import checked_extent
    extent = checked_extent([bounds[0],bounds[3],bounds[1],bounds[4]])
    return list(extent), dict(source='current_deployment_detector_display_roi',
        config=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        config_sha256=hashlib.sha256(content).hexdigest(),extent=list(extent),
        history_outside='clip_at_view_boundary',historical_detector_provenance_unchanged=True)


class Conflict(ValueError):
    pass


@contextmanager
def database(path, readonly=False):
    uri = Path(path).resolve().as_uri() + ('?mode=ro' if readonly else '?mode=rw')
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    try:
        yield connection
    finally:
        connection.close()


def initialize_observations(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(directory / 'observations.sqlite')
    db.executescript('''
      CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE sources(id TEXT PRIMARY KEY,name TEXT NOT NULL,metadata TEXT NOT NULL);
      CREATE TABLE frames(id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(id),
        ordinal INTEGER NOT NULL,stamp_ns INTEGER NOT NULL,elapsed REAL NOT NULL,
        metadata TEXT NOT NULL,rgb BLOB,preview BLOB,cloud BLOB,
        UNIQUE(source_id,ordinal),UNIQUE(source_id,stamp_ns));
      CREATE INDEX frame_times ON frames(source_id,stamp_ns);
    ''')
    db.execute('INSERT INTO meta VALUES(?,?)', ('project_id', dumps(uuid.uuid4().hex)))
    db.execute('INSERT INTO meta VALUES(?,?)', ('complete', 'false'))
    db.commit()
    return db


def initialize_labels(directory):
    path = Path(directory) / 'annotations.sqlite'
    if path.exists():
        return
    db = sqlite3.connect(path)
    db.executescript('''
      PRAGMA journal_mode=WAL;
      CREATE TABLE events(id TEXT PRIMARY KEY,revision INTEGER NOT NULL,document TEXT NOT NULL);
      CREATE TABLE receipts(id TEXT PRIMARY KEY,request_hash TEXT NOT NULL,response TEXT NOT NULL);
      CREATE TABLE writer(singleton INTEGER PRIMARY KEY CHECK(singleton=1),owner TEXT,expires REAL);
      INSERT INTO writer VALUES(1,NULL,0);
    ''')
    db.commit()
    db.close()


class Store:
    def __init__(self, directory, *, use_project_defaults=True):
        self.path = Path(directory).resolve()
        with database(self.path / 'observations.sqlite', True) as db:
            meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
        require(meta.get('complete') is True, '观察工程尚未完成准备')
        self.identity = meta['project_id']
        self.asset_identity = meta['asset_identity']
        self.view_extent, self.view_binding = perception_view()
        self.rgb_cache = None
        selected = meta.get('default_rgb_associations') if use_project_defaults else None
        if selected is not None:
            require(selected.get('status') == 'ready', '工程的RGB关联准备尚未完成；请先恢复prepare-rgb，不能自动使用投影框')
            rgb_associations = self.path / selected['path']
            from perception.detection_tracking.inference.rgb_cache import RGBCache
            self.rgb_cache = RGBCache(rgb_associations,self.identity,self.asset_identity)
            require(self.rgb_cache.identity == selected['identity'], '工程登记的RGB关联来源已改变；请重新准备并审核')
        self.ros_python = meta.get('ros_python', '/usr/bin/python3')
        self._assets, self._assets_lock = None, threading.RLock()
        self._materialized={}
        # Observations are immutable; cache only bounded display/input computations.
        self.frame = lru_cache(maxsize=192)(self.frame)
        self.window = lru_cache(maxsize=128)(self.window)
        self._context_frame = lru_cache(maxsize=192)(self._context_frame)
        self._trajectory_frame = lru_cache(maxsize=192)(self._trajectory_frame)
        self.display_frame=lru_cache(maxsize=192)(self.display_frame)
        self.display_window=lru_cache(maxsize=128)(self.display_window)

    def _read_lazy(self, ident, display=False):
        with self._assets_lock:
            if self._assets is None:
                from .assets import LazyAssets
                self._assets = LazyAssets(self.path, self.ros_python)
            if self.rgb_cache is not None:
                return self._assets.read(ident,display=display,association=True)
            return self._assets.read(ident,display=True) if display else self._assets.read(ident)

    def close(self):
        if self._assets is not None:
            self._assets.close()
        for cached in (self.frame,self.window,self._context_frame,self._trajectory_frame,self.display_frame,self.display_window):
            cached.cache_clear()
        if hasattr(self,'_display_images'):
            self._display_images.clear()

    def sources(self):
        with database(self.path / 'observations.sqlite', True) as db:
            return [dict(id=r['id'], name=r['name'], **json.loads(r['metadata'])) for r in db.execute('SELECT * FROM sources')]

    def frames(self, source_id):
        with database(self.path / 'observations.sqlite', True) as db:
            return [dict(id=r['id'], source_id=r['source_id'], ordinal=r['ordinal'], stamp_ns=r['stamp_ns'],
                         elapsed=r['elapsed'], **{k: v for k, v in json.loads(r['metadata']).items() if k != 'result'}) for r in
                    db.execute('SELECT id,source_id,ordinal,stamp_ns,elapsed,metadata FROM frames WHERE source_id=? ORDER BY stamp_ns', (source_id,))]

    def frame(self, ident):
        return self._frame(ident,False)

    def display_frame(self, ident):
        return self._frame(ident,True)

    def _frame(self, ident, display):
        with database(self.path / 'observations.sqlite', True) as db:
            row = db.execute('SELECT id,source_id,ordinal,stamp_ns,elapsed,metadata FROM frames WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise KeyError('观察不存在')
            metadata = json.loads(row['metadata'])
            if metadata.get('lazy'):
                metadata = self._read_lazy(ident,display=display)['metadata']
            result = dict(id=row['id'], source_id=row['source_id'], ordinal=row['ordinal'], stamp_ns=row['stamp_ns'], elapsed=row['elapsed'])
            result.update(metadata)
            people = [dict(person, eligible=model_eligible(person)) for person in result['people']]
            if people != result['people']:
                result['people'] = people
                # Derive a new input without rewriting frozen observations or labels.
                result['source_fingerprint'] = metadata['fingerprint']
                result['fingerprint'] = digest(dict(source=metadata['fingerprint'], people=people))
            self._materialized[ident]=dict(source_id=result['source_id'],stamp_ns=result['stamp_ns'])
            if len(self._materialized)>512:
                self._materialized.pop(next(iter(self._materialized)))
            if display:
                result['fingerprint']=None
                result['input_verified']=False
            if self.rgb_cache is not None:
                result = self.rgb_cache.apply(result,display=display)
            return result

    def asset(self, ident, kind, display=False):
        require(kind in ('rgb', 'preview', 'cloud', 'cloud3d', 'model_scan'), '未知资产类型')
        with database(self.path / 'observations.sqlite', True) as db:
            column = 'cloud' if kind in ('cloud3d', 'model_scan') else kind
            row = db.execute('SELECT ' + column + ',metadata FROM frames WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise KeyError('观察不存在')
            if json.loads(row[1]).get('lazy'):
                return self._read_lazy(ident,display=display)[kind]
            if kind == 'model_scan':
                require(bool(json.loads(row[1]).get('model_scan_binding')),
                        '旧物化工程没有完整扫描绑定，不能用浏览点云代替模型输入；请从原bag重新准备工程')
            return row[0]

    def window(self, ident):
        return self._window(ident,False)

    def display_window(self, ident):
        return self._window(ident,True)

    def _window(self, ident, display):
        get_frame=self.display_frame if display else self.frame
        current = get_frame(ident)
        with database(self.path / 'observations.sqlite', True) as db:
            # Resolve only the historical candidates required by the three-frame protocol.
            # The 1us margin handles float serialization; build_window still rejects actual future frames.
            from decision.vlm.scene_inputs import OFFSETS_NS
            candidates = {ident}
            offsets = () if 'continuity' in current and current['continuity'] is None else OFFSETS_NS[:-1]
            for offset in offsets:
                desired = current['stamp_ns']+offset
                for row in db.execute('SELECT id,stamp_ns,metadata FROM frames WHERE source_id=? AND stamp_ns BETWEEN ? AND ? ORDER BY stamp_ns DESC LIMIT 2',
                                      (current['source_id'], desired-150_001_000, desired+1000)):
                    if 'continuity' in current and json.loads(row['metadata']).get('continuity') != current['continuity']:
                        break  # A reset/failed observation cannot be hidden by an older success.
                    candidates.add(row['id'])
                    if desired-row['stamp_ns'] >= 1000:
                        break
            rows = [get_frame(fid) for fid in candidates]
        window=build_window(rows,current,self.asset_identity,extent=self.view_extent)
        trajectories = {}
        for fid in window['frames']:
            if fid is not None:
                trace = self._trajectory_frame(fid)
                people = {key:trace['people'].get(key,[]) for key in window['mapping']} if trace['status']=='available' else {}
                trajectories[fid] = dict(trace,people=people)
        window['display_view'] = self.view_binding
        window['trajectories']=trajectories
        scans = {fid:get_frame(fid).get('model_scan_binding') for fid in window['frames'] if fid is not None} if not display else {}
        if not display:
            require(all(value and value.get('kind') == MODEL_BACKGROUND for value in scans.values()),
                    '模型输入缺少完整单帧扫描绑定，不能用浏览点云代替')
        window['background'] = dict(kind=MODEL_BACKGROUND, scans=scans)
        window['fingerprint']=None if display else digest(dict(observations=window['fingerprint'],fields=window['fields'],
                                                             trajectories=trajectories,display_view=self.view_binding,
                                                             background=window['background']))
        if display:window['input_verified']=False
        return window

    def model_image(self, window, index):
        """Shared review/export renderer, using a bound complete scan, never browse clouds."""
        import hashlib
        import io
        import numpy as np
        from decision.vlm.scene_inputs import pointcloud_background, render_observation
        ident = window['frames'][index]
        frame = self.frame(ident) if ident else None
        background = None
        if frame is not None:
            binding = window['background']['scans'][ident]
            require(binding == frame.get('model_scan_binding'), '模型扫描来源已变化，请重新读取输入窗口')
            raw = self.asset(ident, 'model_scan')
            require(raw is not None and hashlib.sha256(raw).hexdigest() == binding['sha256'], '模型扫描与绑定不符')
            background = pointcloud_background(np.load(io.BytesIO(raw), allow_pickle=False), window['fields']['e'])
        return render_observation(frame, self.asset(ident, 'rgb') if ident else None, window['mapping'],
            window['fields']['dt'][index], window['fields']['e'], window.get('trajectories', {}).get(ident),
            bev_background=background)

    def context_frames(self, ident):
        """Display-only histories also cover reference tracks; model IDs stay frozen."""
        window = self.window(ident)
        return [self._context_frame(fid) if fid else None for fid in window['frames']]

    def _context_frame(self, ident):
        return self.context_frame(self.display_frame(ident))

    def context_frame(self, frame):
        trace = self._trajectory_frame(frame['id'])
        present = {person['key'] for person in frame['people'] if person.get('kind')=='track' and person.get('box') is not None}
        people = {key:trace['people'].get(key,[]) for key in sorted(present)} if trace['status']=='available' else {}
        return dict(frame=frame,trajectory=dict(trace,people=people))

    def _trajectory_frame(self, ident):
        from decision.vlm.trajectory_inputs import HISTORY_S, build_trajectories
        frame = self._materialized.get(ident) or self.frame(ident)
        with database(self.path / 'observations.sqlite', True) as db:
            entries = [dict(id=row['id'], stamp_ns=row['stamp_ns'], result=json.loads(row['metadata']).get('result'))
                       for row in db.execute('SELECT id,stamp_ns,metadata FROM frames WHERE source_id=? AND stamp_ns BETWEEN ? AND ? ORDER BY stamp_ns',
                            (frame['source_id'], frame['stamp_ns']-int(HISTORY_S*1e9)-1000, frame['stamp_ns']+1000))]
        keys = sorted({track['track_id']+'/g'+str(track['lifecycle_generation']) for entry in entries
                       for track in (entry.get('result') or {}).get('tracks',[])})
        return build_trajectories(entries,[ident],dict.fromkeys(keys))[ident]

    def events(self):
        with database(self.path / 'annotations.sqlite', True) as db:
            rows = [dict(revision=r['revision'], document=json.loads(r['document'])) for r in db.execute('SELECT * FROM events ORDER BY rowid')]
            return [row for row in rows if not row['document'].get('deleted')]

    def snapshot(self):
        from decision.vlm.scene_inputs import renderer_identity
        implementation = dict(renderer_identity(),display_view=self.view_binding)
        if self.rgb_cache is not None:
            implementation = dict(implementation,rgb_association=self.rgb_cache.identity,
                                  rgb_association_consumer=self.rgb_cache.implementation)
        return dict(identity=self.identity, asset_identity=self.asset_identity, input_implementation=digest(implementation),
                    model_input_background=MODEL_BACKGROUND,
                    rgb_box_source='yolo_association' if self.rgb_cache else 'projection',
                    rgb_association_identity=self.rgb_cache.identity if self.rgb_cache else None,
                    display_implementation=digest((Path(__file__).with_name('backend.py').read_bytes()+
                                                  Path(__file__).with_name('ros_reader.py').read_bytes()).hex()),
                    sources=self.sources(), events=self.events())

    def writer(self, owner, acquire=False, release=False):
        require(isinstance(owner, str) and 1 <= len(owner) <= 128, '写入会话标识错误')
        with database(self.path / 'annotations.sqlite') as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM writer WHERE singleton=1').fetchone()
            now = time.time()
            if release:
                db.execute('UPDATE writer SET owner=NULL,expires=0 WHERE owner=?', (owner,))
                db.commit()
                return dict(writable=False)
            if row['owner'] != owner and (row['expires'] > now or not acquire):
                db.rollback()
                raise Conflict('已有写入会话；可只读打开，或在原会话关闭／过期后重新连接')
            if row['owner'] == owner and row['expires'] <= now and not acquire:
                db.rollback()
                raise Conflict('写入会话已过期，请重新连接')
            db.execute('UPDATE writer SET owner=?,expires=? WHERE singleton=1', (owner, now+90))
            db.commit()
            return dict(writable=True, expires=now+90)

    def receipt(self, operation):
        with database(self.path / 'annotations.sqlite', True) as db:
            row = db.execute('SELECT response FROM receipts WHERE id=?', (operation,)).fetchone()
            return json.loads(row[0]) if row else None

    def validate_event(self, event, previous=None):
        require(isinstance(event, dict) and isinstance(event.get('id'), str) and event['id'], '事件ID错误')
        require(isinstance(event.get('name'), str) and bool(event['name'].strip()), '事件名称不能为空')
        validate_links(event)
        require(event.get('split') in ('development', 'validation', 'holdout', 'unassigned'), '数据划分错误')
        frames = self.frames(event['source_id'])
        require(bool(frames), '来源不存在／没有观察')
        require(event['end'] <= frames[-1]['elapsed'] + .001, '事件超过序列范围')
        frame_map = {r['id']: r for r in frames}
        all_keys = {p['key'] for row in frames for p in row['people']}
        for ident, auxiliary in event.get('auxiliary', {}).items():
            if ident in frame_map and frame_map[ident].get('lazy'):
                frame_map[ident] = self.frame(ident)
            require(ident in frame_map and frame_map[ident]['image_size'] is not None, '辅助人物需要有效RGB观察')
            width, height = frame_map[ident]['image_size']
            for person in auxiliary:
                rect = person.get('rect')
                require(isinstance(person.get('key'), str) and person['key'].startswith('aux:'), '辅助人物身份错误')
                require(isinstance(rect, list) and len(rect) == 4 and
                        all(isinstance(v, (int, float)) for v in rect) and
                        0 <= rect[0] < rect[2] <= width and 0 <= rect[1] < rect[3] <= height, '辅助人物框超出原图')
                all_keys.add(person['key'])
        for link in event.get('links', []):
            require(link['key'] in all_keys, '关联轨迹不存在')
        targets = event.get('targets', {})
        for target_id, target in targets.items():
            require(target['id'] == target_id and bool(target.get('description', '').strip()), '目标描述不能为空')
            ref = frame_map.get(target.get('reference_frame'))
            require(ref is not None, '目标描述参考帧不存在')
            require(target.get('absent') is True or target.get('person_id') in event.get('persons', {}), '目标人物不存在')
            if target.get('absent'):
                require(bool(target.get('description_source', '').strip()), '无目标实例缺少独立描述来源')
        collections = [(None, event.get('t1', {}))] + [(target, target.get('samples', {})) for target in targets.values()]
        for target, labels in collections:
            for ident, label in labels.items():
                frame = frame_map.get(ident)
                require(frame is not None and event['start']-.001 <= frame['elapsed'] <= event['end']+.001, '样本不在事件范围')
                if target:
                    require(frame['stamp_ns'] >= frame_map[target['reference_frame']]['stamp_ns'], '样本早于描述参考时刻')
                old_labels = (previous.get('t1', {}) if target is None else
                              previous.get('targets', {}).get(target['id'], {}).get('samples', {})) if previous else {}
                if old_labels.get(ident) == label:
                    continue  # Unchanged labels were already checked; export rechecks all reviewed inputs.
                if label['status'] == 'complete':
                    window = self.window(ident)
                    frame = self.frame(ident)
                    keys = window['mapping']
                else:
                    # Draft edits and interval sampling do not decode every referenced bag frame.
                    keys = {p['key'] for p in frame['people']}
                    if target is None:
                        keys = {k for g in label.get('groups', []) for k in g['members']} | set(label.get('singles', [])) | set(label.get('unknown', []))
                        require(keys <= {p['key'] for p in frame['people']}, '草稿引用了不存在的人物')
                if target is None:
                    validate_t1(label, keys, [p['key'] for p in frame['people']] +
                                [p['key'] for p in event.get('auxiliary', {}).get(ident, [])])
                else:
                    validate_t2(label, keys, [p['key'] for p in frame['people']] +
                                [p['key'] for p in event.get('auxiliary', {}).get(ident, [])], target,
                                lambda key: person_at(event, key, frame['stamp_ns']))
                if label['status'] == 'complete':
                    require(label.get('binding') == window['fingerprint'], '输入已改变，请重新审核当前样本')
                    require(window['ready'], '历史观察或RGB不足，不能作为完整输入已标样本')
                    require(bool(label.get('reviewer', '').strip()), '缺少审核者')
                    require(isinstance(label.get('reviewed_at'), str) and bool(label['reviewed_at'].strip()), '缺少审核时间')

    def commit(self, request):
        require(request.get('project_id') == self.identity, '工程身份不匹配')
        deleting = request.get('action') == 'delete'
        require(request.get('action') in (None, 'delete'), '未知写入操作')
        event = dict(id=request.get('event_id')) if deleting else copy.deepcopy(request['event'])
        require(isinstance(event, dict) and isinstance(event.get('id'), str) and bool(event['id']), '事件ID错误')
        require(deleting or not event.get('deleted'), '删除事件请使用删除操作')
        require(isinstance(request.get('operation_id'), str) and len(request['operation_id']) <= 128, '操作ID错误')
        require(type(request.get('expected_revision')) is int, '期望修订错误')
        fingerprint = digest(request)
        with database(self.path / 'annotations.sqlite') as db:
            db.execute('BEGIN IMMEDIATE')
            old_receipt = db.execute('SELECT * FROM receipts WHERE id=?', (request['operation_id'],)).fetchone()
            if old_receipt:
                if old_receipt['request_hash'] != fingerprint:
                    raise Conflict('同一操作ID不能携带不同内容')
                return json.loads(old_receipt['response'])
            lease = db.execute('SELECT * FROM writer WHERE singleton=1').fetchone()
            if lease['owner'] != request.get('writer_id') or lease['expires'] <= time.time():
                raise Conflict('写入会话不存在或已过期；草稿保留，请重新连接')
            old = db.execute('SELECT * FROM events WHERE id=?', (event['id'],)).fetchone()
            revision = old['revision'] if old else 0
            if revision != request['expected_revision']:
                raise Conflict('事件已被修改；保留本机草稿后重新读取，不能自动覆盖')
            if old and json.loads(old['document']).get('deleted'):
                raise Conflict('事件已删除，不能用旧草稿重新创建')
            if deleting:
                if old is None:
                    raise Conflict('事件不存在或已删除，请重新读取工程')
                revision += 1
                # Retain only identity/revision to reject stale writes, never active labels.
                db.execute('UPDATE events SET revision=?,document=? WHERE id=?',
                           (revision, dumps(dict(id=event['id'], deleted=True)), event['id']))
                response = dict(operation_id=request['operation_id'], revision=revision, event_id=event['id'], deleted=True)
                db.execute('INSERT INTO receipts VALUES(?,?,?)', (request['operation_id'], fingerprint, dumps(response)))
                db.commit()
                return response
            before = None
            if old:
                before = json.loads(old['document'])
                changed_identity = any(before.get(k) != event.get(k) for k in ('persons', 'links', 'auxiliary'))
                if changed_identity:
                    for label in event.get('t1', {}).values():
                        if label['status'] == 'complete':
                            label['status'] = 'draft'
                    for target in event.get('targets', {}).values():
                        for label in target.get('samples', {}).values():
                            if label['status'] == 'complete':
                                label['status'] = 'draft'
                for tid, target in event.get('targets', {}).items():
                    previous = before.get('targets', {}).get(tid, {})
                    if any(previous.get(k) != target.get(k) for k in ('description', 'description_source', 'reference_frame', 'person_id', 'absent')):
                        for label in target.get('samples', {}).values():
                            if label['status'] == 'complete':
                                label['status'] = 'draft'
            self.validate_event(event, previous=before)
            # Source/location split changes must be consistent with other events.
            for row in db.execute('SELECT document FROM events WHERE id<>?', (event['id'],)):
                other = json.loads(row[0])
                if other.get('deleted'):
                    continue
                same = event['source_id'] == other['source_id'] or any(
                    event.get(k) and event.get(k) == other.get(k) for k in ('root_group', 'location_group'))
                if same and 'unassigned' not in (event['split'], other['split']):
                    require(event['split'] == other['split'], '相同采集根来源／地点不能跨数据划分')
            revision += 1
            db.execute('INSERT OR REPLACE INTO events VALUES(?,?,?)', (event['id'], revision, dumps(event)))
            response = dict(operation_id=request['operation_id'], revision=revision, document=event)
            db.execute('INSERT INTO receipts VALUES(?,?,?)', (request['operation_id'], fingerprint, dumps(response)))
            db.commit()
            return response


def new_event(source, start, end, name):
    return dict(id=uuid.uuid4().hex, source_id=source['id'], name=name, start=start, end=end,
                root_group=source.get('root_group') or source['id'], location_group='', split='unassigned',
                persons={}, links=[], auxiliary={}, t1={}, targets={})
