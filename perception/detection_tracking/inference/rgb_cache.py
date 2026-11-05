"""Bound image detections/associations, shared by preparation and read-only consumers."""
import hashlib
import json
from pathlib import Path
import sqlite3
from collections import Counter


def dumps(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def frame_binding(frame):
    return {key:frame.get(key) for key in ('id','source_id','stamp_ns','rgb_stamp_ns','image_size',
                                          'perception_fingerprint','image_association_input')}


class RGBCache:
    def __init__(self, directory, project_id, asset_identity):
        self.path = Path(directory).resolve()
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if (not self.manifest.get('complete') or self.manifest['source']['project_id'] != project_id or
                self.manifest['source']['asset_identity'] != asset_identity):
            raise ValueError('RGB关联缓存未完成或与当前工程来源不符')
        self.database = self.path/'associations.sqlite'
        if file_digest(self.database) != self.manifest['cache_sha256']:
            raise ValueError('RGB关联缓存内容与冻结来源不符')
        self.stat = (self.database.stat().st_size,self.database.stat().st_mtime_ns)
        self.identity = self.manifest['identity']
        self.implementation = file_digest(__file__)

    def lookup(self, frame):
        stat = self.database.stat()
        if (stat.st_size,stat.st_mtime_ns) != self.stat:
            raise ValueError('RGB关联缓存已改变，请重启并核对新来源')
        with sqlite3.connect(self.database.as_uri()+'?mode=ro',uri=True) as db:
            row = db.execute('SELECT binding,result FROM observations WHERE frame_id=?',(frame['id'],)).fetchone()
        if row is None:
            raise ValueError('当前观察的RGB关联尚未准备：'+frame['id'])
        binding=json.loads(row[0])
        if binding.get('kind')=='recorded_ros':
            from .rgb_recording import validate_binding
            validate_binding(binding,frame)
        elif binding != frame_binding(frame):
            raise ValueError('RGB关联与原图、标定或三维观察不匹配')
        return json.loads(row[1])

    def apply(self, frame, display=False):
        association = self.lookup(frame)
        people = []
        for original in frame['people']:
            record = association['tracks'].get(original['key'])
            if record is None:
                raise ValueError('RGB关联缺少源人物身份，不能补造对应')
            person = dict(original,rgb_box=record.get('rgb_box') if record['status']=='matched' else None,
                          rgb_association=record)
            people.append(person)
        result = dict(frame,people=people,rgb_box_source='yolo_association',
                      rgb_association_identity=self.identity,rgb_association_status=association['status'],
                      rgb_association_implementation=self.implementation,
                      association_source_fingerprint=frame.get('fingerprint'))
        result['fingerprint'] = None if display else digest(result)
        if display:result['input_verified'] = False
        return result


class CacheWriter:
    def __init__(self,directory,source,detector,association):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True,exist_ok=True)
        self.manifest_path = self.path/'manifest.json'
        self.spec = dict(source=source,detector=detector,association=association)
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if any(old[key] != value for key,value in self.spec.items()):
                raise ValueError('输出缓存已有不同来源或参数，请使用新目录')
        elif any(self.path.iterdir()):
            raise ValueError('输出目录非空，不能作为新RGB缓存')
        self.db = sqlite3.connect(self.path/'associations.sqlite')
        self.pending = 0
        self.db.executescript('''CREATE TABLE IF NOT EXISTS images(key TEXT PRIMARY KEY,binding TEXT,detections TEXT,timing_s REAL);
          CREATE TABLE IF NOT EXISTS observations(frame_id TEXT PRIMARY KEY,binding TEXT,result TEXT);''')
        self._write_manifest(dict(self.spec,complete=False))

    def _write_manifest(self,value):
        temporary = self.path/'.manifest.tmp'
        temporary.write_text(dumps(value)+'\n',encoding='utf-8')
        temporary.replace(self.manifest_path)

    def image(self,binding):
        key = digest(dict(input=binding,detector=self.spec['detector']))
        row = self.db.execute('SELECT detections,timing_s FROM images WHERE key=?',(key,)).fetchone()
        return key,(json.loads(row[0]),row[1]) if row else None

    def put_image(self,key,binding,detections,timing_s):
        self.db.execute('INSERT INTO images VALUES(?,?,?,?)',(key,dumps(binding),dumps(detections),timing_s))

    def observation(self,frame):
        row = self.db.execute('SELECT binding,result FROM observations WHERE frame_id=?',(frame['id'],)).fetchone()
        if row is None:return None
        if json.loads(row[0]) != frame_binding(frame):
            raise ValueError('已缓存观察的源数据改变，请使用新目录')
        return json.loads(row[1])

    def put_observation(self,frame,result):
        self.put_bound_observation(frame['id'],frame_binding(frame),result)

    def put_bound_observation(self,ident,binding,result):
        self.db.execute('INSERT INTO observations VALUES(?,?,?)',(ident,dumps(binding),dumps(result)))
        # Derived data only: checkpoint small batches instead of two disk flushes
        # per frame. An interruption may recompute the final uncommitted batch.
        self.pending += 1
        if self.pending >= 25:
            self.db.commit()
            self.pending = 0

    def finish(self,summary):
        self.db.commit()
        count = self.db.execute('SELECT count(*) FROM observations').fetchone()[0]
        totals = Counter(observations=count)
        for (raw,) in self.db.execute('SELECT result FROM observations'):
            result = json.loads(raw)
            totals['matches'] += result.get('matches',0)
            totals['unmatched_detections'] += result.get('unmatched_detection_count',0)
            for record in result['tracks'].values():totals['track_'+record['status']] += 1
        value = dict(self.spec,complete=True,observations=count,summary=summary,
                     totals=dict(totals),
                     cache_sha256=file_digest(self.path/'associations.sqlite'))
        value['identity'] = digest(dict(self.spec,cache_sha256=value['cache_sha256']))
        self._write_manifest(value)
        return value

    def close(self):
        self.db.close()
