"""Explicit saved RGB observations, never current execution evidence."""
import hashlib
import json
from perception.environment.context import fingerprint
from .store import encode


class Keyframes:
    capacity = 24
    visible_limit = 2

    def __init__(self, store, task_id):
        self.store, self.task_id = store, task_id

    def save(self, observation, revision, reason):
        observation.verify()
        record = dict(task_id=self.task_id,task_revision=revision,observation=observation.prompt(),
                      observation_binding=observation.record['input_fingerprint'],
                      source_stamp_s=observation.record['stamp_s'],reason=reason,
                      image_sha256=hashlib.sha256(observation.image).hexdigest(),
                      historical_only=True,proves_arrival=False,proves_current_clearance=False)
        identity = fingerprint(dict(task_id=self.task_id,revision=revision,
                                    observation_binding=record['observation_binding']))
        if self.store.db.execute('SELECT 1 FROM keyframes WHERE keyframe_id=?',(identity,)).fetchone():
            return self.read(identity,revision)[0]
        if len(self.index(revision)) >= self.capacity:
            raise ValueError('Saved keyframe capacity reached; existing records retained')
        record['keyframe_id'] = identity
        record['record_fingerprint'] = fingerprint(record)
        with self.store.db:
            self.store.db.execute('INSERT INTO keyframes VALUES (?,?,?,?,?)',
                (identity,self.task_id,revision,encode(record),observation.image))
        return record

    def read(self, identity, revision):
        row = self.store.db.execute('SELECT metadata,image FROM keyframes WHERE keyframe_id=? AND task_id=? AND revision=?',
                                   (identity,self.task_id,revision)).fetchone()
        if row is None:
            raise ValueError('No saved keyframe in this task revision')
        record,image = json.loads(row[0]),bytes(row[1])
        if (fingerprint({k:v for k,v in record.items() if k!='record_fingerprint'}) != record['record_fingerprint'] or
                hashlib.sha256(image).hexdigest() != record['image_sha256']):
            raise ValueError('Saved keyframe changed')
        return record,image

    def index(self, revision):
        rows = self.store.db.execute('SELECT metadata FROM keyframes WHERE task_id=? AND revision=? ORDER BY rowid',
                                    (self.task_id,revision)).fetchall()
        return [dict(keyframe_id=(r:=json.loads(row[0]))['keyframe_id'],source_stamp_s=r['source_stamp_s'],
                     reason=r['reason'],historical_only=True,map_pose=r['observation'].get('map_base_pose'),
                     map_frame=(r['observation'].get('map_binding') or {}).get('map_frame'),
                     clock_domain=r['observation']['clock_domain']) for row in rows]

    def attachments(self, messages, revision):
        identities = []
        for message in messages:
            if message.get('role') != 'user' or not isinstance(message.get('content'),str):
                continue
            try:
                value = json.loads(message['content'])
            except ValueError:
                continue
            if isinstance(value,dict) and value.get('source')=='historical_keyframe':
                identity = value['keyframe']['keyframe_id']
                if identity in identities:
                    identities.remove(identity)
                identities.append(identity)
        return {identity:self.read(identity,revision) for identity in identities[-self.visible_limit:]}
