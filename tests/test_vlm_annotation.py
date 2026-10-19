"""Behavioral checks for the independent social-decision annotation tool."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
from PIL import Image

from decision.vlm.annotation.backend import LocalBackend
from decision.vlm.annotation.export import export_project
from decision.vlm.annotation.rules import digest, dumps, t1_empty, t2_empty, validate_t1
from decision.vlm.annotation.store import Conflict, Store, initialize_labels, initialize_observations, new_event


def fixture(path):
    db = initialize_observations(path)
    db.execute('INSERT INTO sources VALUES(?,?,?)', ('source', '室内事件', dumps(dict(root_group='recording', frame_id='rslidar'))))
    image = Image.new('RGB', (640, 480), '#809cad')
    raw = io.BytesIO()
    image.save(raw, format='PNG')
    cloud = io.BytesIO()
    np.save(cloud, np.array([[2., .5, 0., 8.], [3., -.3, 1., 10.]], dtype='<f4'), allow_pickle=False)
    for i in range(31):
        people = [dict(key='track:'+key, eligible=True, box=[2.+j, float(j), 0., .7, .7, 1.7, 0., .9],
                       rgb_box=[50.+200*j, 50., 150.+200*j, 350.], kind='track', observed=True, confirmed=True)
                  for j, key in enumerate(('A', 'B'))]
        metadata = dict(people=people, image_size=[640, 480], has_rgb=True, fingerprint=digest(['frame', i]),
                        timestamp_source='sensor', clock_domain='bag', sensor_frame='rslidar', health='valid')
        db.execute('INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?)',
                   ('source/'+str(i), 'source', i, 1700000000000000000+i*100000000, i/10,
                    dumps(metadata), raw.getvalue(), raw.getvalue(), cloud.getvalue()))
    db.execute('INSERT INTO meta VALUES(?,?)', ('asset_identity', dumps('asset-fixture')))
    db.execute('UPDATE meta SET value=? WHERE key=?', ('true', 'complete'))
    db.commit()
    db.close()
    initialize_labels(path)


class LazyIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.path = root/'project'
        results = root/'people.jsonl'
        bag = root/'source.bag'
        bag.write_bytes(b'synthetic; must not be decoded by registration')
        calibration = root/'camera.yaml'
        calibration.write_text(dumps(dict(lidar_to_camera_optical=dict(matrix=np.eye(4).tolist(), source_frame='rslidar', target_frame='camera'))))
        rows = []
        for i in range(31):
            rows.append(dict(observation_id='original/'+str(i), sequence_id='sequence', stamp_s=1700000000+i*.1001,
                             clock_domain='bag', timestamp_source='sensor', detection_frame_id='rslidar', health='valid',
                             pose_health='valid', world_from_lidar=np.eye(4).tolist(), detections=[],
                             tracks=[dict(track_id='track:A', lifecycle_generation=1, state='observed', confirmed=True,
                                          box=[2,0,0,.7,.7,1.7,0,.9])], input_fingerprint='original-'+str(i), rgb_stamp_s=None))
        results.write_text(''.join(dumps(row)+'\n' for row in rows))
        from decision.vlm.annotation.__main__ import arguments
        from decision.vlm.annotation.prepare import prepare
        args = arguments(['prepare','--results',str(results),'--bag',str(bag),'--calibration',str(calibration),'--output',str(self.path)])
        def headers(command, **kwargs):
            recipe = json.loads(Path(command[-1]).read_text())
            probes = {job['id']:dict(origin_ns=1700000000000000000,record_offset_ns=0,sensor_frame='rslidar',
                                     bag_fingerprint={'synthetic':True}) for job in recipe['jobs']}
            return SimpleNamespace(returncode=0, stdout=dumps(probes), stderr='')
        with patch('decision.vlm.annotation.prepare.subprocess.run', side_effect=headers), redirect_stdout(io.StringIO()):
            prepare(args)
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_registration_has_only_index_and_never_encodes_all_assets(self):
        import sqlite3
        rows = self.store.frames(self.store.sources()[0]['id'])
        self.assertEqual(len(rows),31)
        self.assertTrue(all(r['lazy'] for r in rows))
        self.assertIsNone(self.store._assets)
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM frames WHERE rgb IS NOT NULL OR cloud IS NOT NULL').fetchone()[0],0)

    def test_resolved_metadata_overrides_index_and_only_window_candidates_are_read(self):
        rows = self.store.frames(self.store.sources()[0]['id'])
        calls = set()
        frames = {r['id']:r for r in rows}
        class Reader:
            def read(self, ident):
                calls.add(ident)
                row = frames[ident]
                return dict(metadata=dict(stamp_ns=row['stamp_ns']+111,elapsed=row['elapsed']+.000000111,
                                          has_rgb=True,image_size=[640,480],fingerprint='materialized-'+ident,
                                          people=[dict(key='track:A/g1',eligible=True,box=[2,0,0,.7,.7,1.7,0,.9],rgb_box=[10,10,80,350])]),
                            rgb=b'raw',preview=b'preview',cloud=b'cloud')
            def close(self):pass
        self.store._assets = Reader()
        frame = self.store.frame(rows[20]['id'])
        self.assertEqual(frame['stamp_ns'],rows[20]['stamp_ns']+111)
        window = self.store.window(frame['id'])
        self.assertTrue(window['ready'])
        self.assertEqual(window['frames'],[rows[i]['id'] for i in (10,15,20)])
        self.assertEqual(calls,{rows[i]['id'] for i in (10,15,20)})

    def test_bulk_pending_samples_save_without_raw_frame_reads(self):
        source = self.store.sources()[0]
        rows = self.store.frames(source['id'])
        event = new_event(source,rows[0]['elapsed'],rows[-1]['elapsed'],'待标计划')
        event['t1'] = {r['id']:t1_empty(None) for r in rows[::5]}
        self.store.writer('writer',acquire=True)
        with patch.object(self.store,'_read_lazy',side_effect=AssertionError('pending samples must not decode bags')):
            saved = self.store.commit(dict(project_id=self.store.identity,writer_id='writer',operation_id='pending',expected_revision=0,event=event))
        self.assertEqual(len(saved['document']['t1']),7)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'project'
        fixture(self.path)
        self.store = Store(self.path)
        self.owner = 'writer'
        self.store.writer(self.owner, acquire=True)
        self.event = new_event(self.store.sources()[0], 0, 3, '同行与交叉')
        self.event['split'] = 'development'

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, event=None, revision=0):
        return dict(project_id=self.store.identity, writer_id=self.owner, operation_id=uuid.uuid4().hex,
                    expected_revision=revision, event=copy.deepcopy(event or self.event))

    def complete_t1(self, ident='source/20'):
        label = t1_empty(self.store.window(ident)['fingerprint'])
        label.update(status='complete', groups=[dict(members=['track:A', 'track:B'], relation='w', roles=[])],
                     reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['t1'][ident] = label
        return label

    def test_causal_window_and_missing_history(self):
        window = self.store.window('source/20')
        self.assertEqual(window['frames'], ['source/10', 'source/15', 'source/20'])
        self.assertEqual(window['fields']['dt'], [-1., -.5, 0.])
        self.assertEqual(window['mapping'], {'track:A': 0, 'track:B': 1})
        self.assertFalse(self.store.window('source/0')['ready'])

    def test_unknown_is_not_automatic_and_queue_is_out_of_scope(self):
        label = t1_empty('input')
        label['status'] = 'complete'
        with self.assertRaisesRegex(ValueError, '未分类'):
            validate_t1(label, ['A', 'B'])
        label.update(groups=[dict(members=['A', 'B'], relation='q')])
        with self.assertRaisesRegex(ValueError, '不支持'):
            validate_t1(label, ['A', 'B'])

    def test_receipt_is_idempotent_but_not_reusable_for_different_content(self):
        request = self.request()
        first = self.store.commit(request)
        self.assertEqual(self.store.commit(request), first)
        request['event']['name'] = 'different'
        with self.assertRaises(Conflict):
            self.store.commit(request)
        self.assertEqual(self.store.events()[0]['revision'], 1)

    def test_single_writer_and_cas_reject_stale_edits(self):
        with self.assertRaises(Conflict):
            self.store.writer('another', acquire=True)
        result = self.store.commit(self.request())
        with self.assertRaises(Conflict):
            self.store.commit(self.request(revision=0))
        self.assertEqual(result['revision'], 1)

    def test_input_binding_and_incomplete_history_cannot_be_gold(self):
        label = self.complete_t1()
        label['binding'] = 'old-input'
        with self.assertRaisesRegex(ValueError, '输入已改变'):
            self.store.commit(self.request())
        self.event['t1'].clear()
        self.complete_t1('source/0')
        with self.assertRaisesRegex(ValueError, '不足'):
            self.store.commit(self.request())

    def test_identity_edits_invalidate_reviewed_answers(self):
        self.complete_t1()
        result = self.store.commit(self.request())
        updated = result['document']
        updated['persons']['person-A'] = dict(name='A')
        updated['links'].append(dict(key='track:A', person_id='person-A', start_ns=1700000000000000000,
                                     end_ns=1700000003000000000))
        result = self.store.commit(self.request(updated, 1))
        self.assertEqual(result['document']['t1']['source/20']['status'], 'draft')

    def test_same_source_cannot_escape_split_by_renaming_root(self):
        self.store.commit(self.request())
        another = new_event(self.store.sources()[0], 0, 1, '相同bag另一事件')
        another.update(split='holdout', root_group='renamed')
        with self.assertRaisesRegex(ValueError, '跨数据划分'):
            self.store.commit(self.request(another))

    def test_t2_facts_export_does_not_invent_history_or_repair_missing_pid(self):
        self.event['persons']['person-A'] = dict(name='目标A')
        self.event['links'].append(dict(key='track:A', person_id='person-A', start_ns=1700000000000000000,
                                       end_ns=1700000003000000000))
        label = t2_empty(self.store.window('source/20')['fingerprint'])
        label.update(status='complete', visibility='clear', true_key='track:A', plausible=['track:A'], match='unique',
                     coverage='represented', reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['targets']['target'] = dict(id='target', description='穿灰衣的人', person_id='person-A',
                                             absent=False, reference_frame='source/10', samples={'source/20': label})
        self.store.commit(self.request())
        out = Path(self.temporary.name) / 'export'
        summary = export_project(self.path, out)
        result = json.loads((out / 't2.jsonl').read_text())
        self.assertEqual(summary['counts'], dict(t1=0, t2=1))
        self.assertNotIn('s', result)
        self.assertEqual(result['label_kind'], 'fixed_facts_not_history_state_gold')
        self.assertEqual(result['facts']['true_key'], 'track:A')

    def test_auxiliary_missing_person_stays_out_of_model_input(self):
        self.event['persons']['hidden'] = dict(name='漏检人物')
        self.event['auxiliary']['source/20'] = [dict(key='aux:missing', rect=[100, 100, 160, 350])]
        self.event['links'].append(dict(key='aux:missing', person_id='hidden', start_ns=1700000002000000000,
                                       end_ns=1700000002000000000))
        label = t2_empty(self.store.window('source/20')['fingerprint'])
        label.update(status='complete', visibility='clear', true_key='aux:missing', plausible=[], match='insufficient',
                     coverage='missed', reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['targets']['target'] = dict(id='target', description='背包的人', person_id='hidden', absent=False,
                                             reference_frame='source/10', samples={'source/20': label})
        self.store.commit(self.request())
        self.assertNotIn('aux:missing', self.store.window('source/20')['mapping'])

    def test_t1_unrepresentable_group_is_retained_without_fake_response(self):
        label = t1_empty(self.store.window('source/20')['fingerprint'])
        self.event['auxiliary']['source/20'] = [dict(key='aux:missing', rect=[100, 100, 160, 350])]
        label.update(status='complete', singles=['track:B'],
                     reference_groups=[dict(members=['track:A', 'aux:missing'], relation='w', roles=[])],
                     reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['t1']['source/20'] = label
        self.store.commit(self.request())
        out = Path(self.temporary.name)/'export'
        export_project(self.path, out)
        row = json.loads((out/'t1.jsonl').read_text())
        self.assertFalse(row['semantic_eligible'])
        self.assertIsNone(row['response'])
        self.assertEqual(row['unrepresented_groups'][0]['members'], ['track:A', 'aux:missing'])


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'project'
        fixture(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_service_and_remote_durable_save(self):
        from fastapi.testclient import TestClient
        from decision.vlm.annotation.service import create_app
        from decision.vlm.annotation.remote import RemoteBackend
        with TestClient(create_app(self.path, writable=True)) as client:
            remote = RemoteBackend('http://testserver', 'wheelchair-vlm', cache_dir=Path(self.temporary.name)/'cache', transport=client._transport)
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0], 0, 3, '远程事件')
                saved = remote.commit(event, 0)
                self.assertEqual(saved['revision'], 1)
                self.assertEqual(remote.pending(), 0)
                self.assertEqual(remote.asset('source/20', 'rgb')[:8], b'\x89PNG\r\n\x1a\n')
                self.assertEqual(remote.window('source/20')['frames'][-1], 'source/20')
                self.assertEqual(len(remote.review_images('source/20')), 3)
                self.assertEqual(client.get('/projects/wheelchair-vlm/frame', params={'frame_id': '../escape'}).status_code, 404)
            finally:
                remote.close()

    def test_revision_conflict_preserves_queue_and_explicit_archive(self):
        from fastapi.testclient import TestClient
        from decision.vlm.annotation.service import create_app
        from decision.vlm.annotation.remote import RemoteBackend
        with TestClient(create_app(self.path, writable=True)) as client:
            remote = RemoteBackend('http://testserver', 'wheelchair-vlm', cache_dir=Path(self.temporary.name)/'cache', transport=client._transport)
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0], 0, 3, '远程事件')
                result = remote.commit(event, 0)
                stale = copy.deepcopy(event)
                stale['name'] = '过期编辑'
                with self.assertRaises(Conflict):
                    remote.commit(stale, 0)
                self.assertEqual(remote.pending(), 1)
                self.assertEqual(Store(self.path).events()[0]['document']['name'], '远程事件')
                remote.archive_conflicts(stale)
                self.assertEqual(remote.pending(), 0)
                self.assertEqual(remote.drafts()['archives'][0]['current_event']['name'], '过期编辑')
            finally:
                remote.close()

    def test_unknown_commit_result_is_recovered_by_original_receipt(self):
        import httpx
        from decision.vlm.annotation.remote import RemoteBackend
        from decision.vlm.annotation.service import create_app
        from fastapi.testclient import TestClient
        with TestClient(create_app(self.path, writable=True)) as client:
            class LoseAck(httpx.BaseTransport):
                lost = False
                def handle_request(self, request):
                    response = client._transport.handle_request(request)
                    if request.url.path.endswith('/commit') and not self.lost:
                        self.lost = True
                        response.read()
                        raise httpx.ReadError('simulated lost acknowledgement')
                    return response
            cache = Path(self.temporary.name)/'cache'
            remote = RemoteBackend('http://testserver', 'wheelchair-vlm', cache_dir=cache, transport=LoseAck())
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0], 0, 3, '断网事件')
                response = remote.commit(event, 0)
                self.assertTrue(response['staged'])
                self.assertEqual(remote.pending(), 1)
                self.assertEqual(Store(self.path).events()[0]['revision'], 1)
                remote.snapshot()
                self.assertEqual(remote.pending(), 0)
                self.assertEqual(Store(self.path).events()[0]['revision'], 1)
            finally:
                remote.close()


if __name__ == '__main__':
    unittest.main()
