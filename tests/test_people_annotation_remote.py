"""Synthetic API/cache fault tests; no sockets, ROS, models or formal labels."""
from contextlib import closing
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid
import zipfile

import numpy as np
try:
    import httpx
    from fastapi.testclient import TestClient
    from perception.detection_tracking.annotation.cache import Cache
    from perception.detection_tracking.annotation.coordination import Conflict, WriterBusy, commit, receipt, writer
    from perception.detection_tracking.annotation.protocol import digest, pack_assets, unpack_assets
    from perception.detection_tracking.annotation.remote import RemoteBackend, RemoteError
    from perception.detection_tracking.annotation.service import create_app
    HTTP_READY = True
except ImportError:
    HTTP_READY = False

from perception.detection_tracking.annotation.data import Store, blank_review, dumps, edit_review, input_fingerprint
from perception.detection_tracking.annotation.package_client import package_client, FILES


BOUNDS = [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5]
POINTS = np.array([[0., 0., .2, 17.], [3., 1., .5, 42.], [np.nan, 0., 0., 0.]], dtype=np.float32)
RGB = np.full((24, 32, 3), 120, dtype=np.uint8)
BOXES = [[3., 1., .5, .6, .6, 1.7, 0., .9]]


def project_at(path):
    with closing(Store(path, create=True, settings={'private': 'do not transport'},
                       effective={'DATA_CONFIG': {'POINT_CLOUD_RANGE': BOUNDS}})) as store:
        with store.connection:
            for source in ('a', 'b'):
                store.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)',
                    (source, '/server-only/' + source + '.bag', '{}', '/points', '/rgb', 'capture', source, 'train', 1))
                for ordinal in range(3):
                    ident, stamp = source + '/' + str(ordinal), 1000000000 + ordinal * 100000000
                    bound = input_fingerprint(POINTS, source, ident, stamp / 1e9, 'rslidar')
                    store.connection.execute('INSERT INTO frames '
                        '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,fingerprint,review) VALUES (?,?,?,?,?,?,?,?,?)',
                        (ident, source, ordinal, stamp, stamp + 1000000000, 0, 'rslidar', bound, dumps(blank_review())))
        for source in ('a', 'b'):
            for ordinal in range(3):
                ident = source + '/' + str(ordinal)
                store.save_candidates(ident, BOXES, {'checkpoint_sha256': 'test', 'checkpoint': '/server-only/weight'},
                                      store.frame(ident)['fingerprint'])


class Assets:
    def __init__(self, path, python):
        self.path, self.calls = path, 0

    def read(self, ident, encoding):
        self.calls += 1
        with closing(Store(self.path, read_only=True)) as store:
            frame = store.frame(ident)
            return pack_assets(frame, POINTS, RGB, frame['stamp_ns'] - 10000000, encoding=encoding)

    def close(self):
        pass


class Wire(httpx.BaseTransport if HTTP_READY else object):
    def __init__(self, client):
        self.client, self.offline, self.lose_reply, self.calls = client, False, False, []

    def handle_request(self, request):
        self.calls.append((request.method, request.url.path))
        if self.offline:
            raise httpx.ConnectError('synthetic disconnect', request=request)
        response = self.client.request(request.method, str(request.url), content=request.read(), headers=dict(request.headers))
        if self.lose_reply and request.method == 'POST' and request.url.path.endswith('/mutations') and response.status_code == 200:
            self.lose_reply = False
            raise httpx.ReadError('synthetic committed response loss', request=request)
        headers = {key: value for key, value in response.headers.items() if key not in ('content-encoding', 'content-length')}
        return httpx.Response(response.status_code, content=response.content, headers=headers, request=request)

    def close(self):
        pass


class ManualRemote(RemoteBackend if HTTP_READY else object):
    def sync_loop(self):
        self.stop.wait()  # Tests control delivery boundaries; close wakes this thread.


@unittest.skipUnless(HTTP_READY, 'requires annotation HTTP dependencies')
class RemoteAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        project_at(self.project)
        self.app = create_app(dict(server_id='test-server', projects={'test': {'path': str(self.project), 'read_only': False}}), Assets)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.wire = Wire(self.client)

    def backend(self, name='client', read_only=False):
        backend = ManualRemote('http://testserver', 'test', self.root / name,
                               read_only=read_only, encoding='png', transport=self.wire)
        self.addCleanup(backend.close)
        backend.prepare(lambda _: None)
        return backend

    def request_body(self, store, session, kind='review'):
        frame, source = store.frame('a/0'), store.source('a')
        body = dict(mutation_id=uuid.uuid4().hex, client_id='direct', session=session, source_id='a',
                    kind=kind, epoch=source['epoch'], revision=source['revision'])
        if kind == 'review':
            body.update(frame_id=frame['id'], fingerprint=frame['fingerprint'], revision=frame['revision'],
                        expected_hash=digest(frame['review']), review=edit_review(frame['review'], 'add', BOUNDS, xy=[0., 0.]))
        elif kind == 'reset':
            body['current'] = frame['id']
        return body

    def test_codec_preserves_nan_bits_xyzi_and_lossless_rgb_rejects_wrong_or_oversized_arrays(self):
        with closing(Store(self.project)) as store:
            frame = store.frame('a/0')
        payload = pack_assets(frame, POINTS, RGB, frame['stamp_ns'], encoding='png')
        state, points, image, _, _ = unpack_assets(payload, 'a/0')
        self.assertEqual(points.tobytes(), POINTS.tobytes())
        np.testing.assert_array_equal(image, RGB)
        self.assertNotIn('checkpoint', state['model'])
        absent = pack_assets(frame, POINTS, None, None, '无匹配RGB', encoding='png')
        self.assertIsNone(unpack_assets(absent, 'a/0')[2])
        with self.assertRaises(ValueError):
            unpack_assets(payload, 'b/0')
        forged = io.BytesIO()
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False, shape=(2 ** 40, 4)))
        with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(forged, 'w') as target:
            for name in source.namelist():
                target.writestr(name, header.getvalue() if name == 'points.npy' else source.read(name))
        with self.assertRaises(ValueError):
            unpack_assets(forged.getvalue(), 'a/0')

    def test_direct_connection_allowlist_readonly_and_input_limits(self):
        self.assertEqual(self.client.get('/projects').status_code, 200)
        self.assertNotIn('authorization', self.backend(read_only=True).http.headers)
        index = self.client.get('/projects/test/index')
        self.assertEqual(index.status_code, 200)
        self.assertEqual(index.headers['cache-control'], 'no-store')
        self.assertNotIn('/server-only/', index.text)
        self.assertNotIn('private', index.text)
        self.assertEqual(self.client.get('/projects/unknown/index').status_code, 404)
        self.assertEqual(self.client.post('/projects/test/mutations', content=b' ' * 131073).status_code, 400)
        plain = self.root / 'read-only-project'
        project_at(plain)
        app = create_app(dict(server_id='readonly', projects={'read': {'path': str(plain)}}), Assets)
        with TestClient(app) as client:
            snapshot = client.get('/projects/read/index').json()
            identity = dict(server_id='readonly', project_id='read', project_identity=snapshot['identity'])
            self.assertEqual(client.post('/projects/read/writer',
                json=dict(action='acquire', client_id='test', **identity)).status_code, 423)
            self.assertEqual(client.get('/projects/read/sources/a/frames/0/bundle').status_code, 200)
        with closing(Store(plain, read_only=True)) as store:
            self.assertFalse(store.coordinated(), 'readonly startup migrated the database')

    def test_transaction_receipt_reset_idempotency_and_old_epoch(self):
        with closing(Store(self.project)) as store:
            session = writer(store, 'acquire', 'direct')['session']
            body = self.request_body(store, session)
            before, original = store.frame('a/0'), Store._save_review

            def fail_after_update(instance, *args, **kwargs):
                original(instance, *args, **kwargs)
                raise OSError('synthetic crash before receipt')

            with patch.object(Store, '_save_review', fail_after_update), self.assertRaises(OSError):
                commit(store, body)
            self.assertEqual(store.frame('a/0'), before)
            self.assertIsNone(receipt(store, body['mutation_id']))
            result = commit(store, body)
            self.assertEqual(commit(store, body), result)
            modified = dict(body, review=blank_review())
            with self.assertRaises(Conflict):
                commit(store, modified)
            reset = self.request_body(store, session, 'reset')
            cleared = commit(store, reset)
            new = self.request_body(store, session)
            new_result = commit(store, new)
            self.assertEqual(commit(store, reset), cleared)
            self.assertEqual(store.frame('a/0')['review'], new_result['frame']['review'])
            stale = dict(body, mutation_id=uuid.uuid4().hex)
            with self.assertRaises(Conflict):
                commit(store, stale)
            with store.connection:
                store.connection.execute('UPDATE annotation_writer SET expires=0')
            renewed = writer(store, 'acquire', 'direct', session)['session']
            self.assertNotEqual(renewed, session)
            with self.assertRaises(WriterBusy):
                commit(store, self.request_body(store, session))

    def test_single_writer_local_guard_and_wrong_project_binding(self):
        backend = self.backend()
        second = self.backend('second')
        self.assertTrue(second.read_only)
        with closing(Store(self.project)) as store:
            frame = store.frame('a/0')
            with self.assertRaisesRegex(ValueError, '远程写入会话'):
                store.save_review(frame['id'], edit_review(frame['review'], 'add', BOUNDS, xy=[1., 1.]))
            with self.assertRaises(ValueError):
                store.set_group('a', 'capture', 'place', 'train')
        bound = dict(backend.identity, project_identity='wrong')
        with self.assertRaises(RemoteError) as error:
            backend.request('POST', '/projects/test/writer', dict(action='acquire', client_id='wrong', **bound))
        self.assertEqual(error.exception.status, 409)

    def test_offline_drafts_restart_recovery_and_warm_assets(self):
        backend = self.backend()
        frame, _, image, _, _ = backend.read_frame('a/0')
        backend.read_frame('a/0')
        self.assertEqual(self.app.state.projects['test'].assets.calls, 1)
        np.testing.assert_array_equal(image, RGB)
        self.wire.offline = True
        backend.publish(online=False)
        review = edit_review(frame['review'], 'add', BOUNDS, xy=[.1, .1])
        backend.save_review(frame['id'], review, frame['review'])
        self.assertEqual(backend.read_frame('a/0')[0]['review'], review)
        with self.assertRaises(KeyError):
            backend.read_frame('a/1')
        backend.close()
        self.wire.offline = False
        resumed = self.backend()
        self.assertEqual(resumed.cache.counts()['pending'], 1)
        resumed.sync_once()
        self.assertEqual(resumed.cache.counts()['pending'], 0)
        with closing(Store(self.project)) as store:
            self.assertEqual(store.frame('a/0')['review'], review)

    def test_encoding_switch_refetches_assets_instead_of_labeling_jpeg_lossless(self):
        backend = self.backend()
        backend.read_frame('a/0')
        backend.encoding = 'jpeg'
        backend.read_frame('a/0')
        self.assertIsNone(backend.cache.bundle('a/0', 'png'))
        self.assertEqual(self.app.state.projects['test'].assets.calls, 2)
        backend.encoding = 'png'
        _, points, image, _, _ = backend.read_frame('a/0')
        self.assertEqual(self.app.state.projects['test'].assets.calls, 3)
        self.assertEqual(points.tobytes(), POINTS.tobytes())
        np.testing.assert_array_equal(image, RGB)

    def test_corrupt_asset_recovery_keeps_durable_edits(self):
        backend = self.backend()
        frame = backend.read_frame('a/0')[0]
        review = edit_review(frame['review'], 'add', BOUNDS, xy=[0., 0.])
        backend.save_review('a/0', review, frame['review'])
        with backend.cache.open() as database:
            database.execute('UPDATE frames SET bundle=? WHERE id=?', (b'broken ZIP', 'a/0'))
        with self.assertRaisesRegex(ValueError, '草稿保留'):
            backend.read_frame('a/0')
        self.assertEqual(backend.cache.counts()['pending'], 1)
        self.assertEqual(backend.cache.state('a/0')['review'], review)
        self.assertIsNone(backend.cache.bundle('a/0'))
        self.assertEqual(backend.read_frame('a/0')[0]['review'], review)
        self.assertEqual(self.app.state.projects['test'].assets.calls, 2)

    def test_late_frame_read_cannot_revive_invalidated_old_review(self):
        backend = self.backend()
        stale = backend.read_frame('a/0')[0]
        with closing(Store(self.project)) as store:
            with store.connection:
                store.connection.execute('DELETE FROM annotation_writer')
            updated = edit_review(stale['review'], 'note', BOUNDS, note='newer server result')
            store.save_review('a/0', updated)
        snapshot = backend.request('GET', backend.base + '/index')
        backend.cache.update_index(snapshot)
        self.assertFalse(backend.cache.save_frame(stale))
        with self.assertRaises(KeyError):
            backend.cache.state('a/0')
        self.assertEqual(backend.read_frame('a/0')[0]['review'], updated)

    def test_response_loss_queries_same_receipt_and_late_ack_keeps_newer_draft(self):
        backend = self.backend()
        frame = backend.read_frame('a/0')[0]
        first = edit_review(frame['review'], 'add', BOUNDS, xy=[.1, .1])
        last = edit_review(first, 'add', BOUNDS, xy=[.3, .3])
        backend.save_review('a/0', first, frame['review'])
        backend.save_review('a/0', last, first)
        ident = backend.cache.first()['mutation_id']
        self.wire.lose_reply = True
        with self.assertRaises(httpx.ReadError):
            backend.sync_once()
        self.assertEqual(backend.cache.state('a/0')['review'], last)
        backend.sync_once(budget=1)
        self.assertEqual(backend.cache.state('a/0')['review'], last)
        self.assertEqual(backend.cache.counts()['pending'], 1)
        backend.sync_once()
        posts = [call for call in self.wire.calls if call == ('POST', '/projects/test/mutations')]
        self.assertEqual(len(posts), 2, 'response loss caused a duplicate POST')
        self.assertIn(('GET', '/projects/test/mutations/' + ident), self.wire.calls)
        with closing(Store(self.project)) as store:
            self.assertEqual(store.frame('a/0')['review'], last)

    def test_conflict_archives_local_work_without_overwriting_server(self):
        backend = self.backend()
        frame = backend.read_frame('a/0')[0]
        local = edit_review(frame['review'], 'add', BOUNDS, xy=[0., 0.])
        backend.save_review('a/0', local, frame['review'])
        with closing(Store(self.project)) as store:
            store.connection.execute('DELETE FROM annotation_writer')
            store.connection.commit()
            remote = edit_review(frame['review'], 'note', BOUNDS, note='other writer')
            store.save_review('a/0', remote)
        backend.lease_checked = 0.
        with self.assertRaises(RemoteError) as error:
            backend.sync_once()
        self.assertEqual(error.exception.status, 409)
        self.assertEqual(backend.cache.counts()['conflicts'], 1)
        backend.resolve_conflicts()
        self.assertEqual(backend.cache.counts()['pending'], 0)
        self.assertEqual(backend.cache.drafts()['conflicts'][0]['body']['review'], local)
        self.assertEqual(backend.read_frame('a/0')[0]['review'], remote)

    def test_group_order_reset_unknown_result_and_client_readonly(self):
        backend = self.backend()
        frame = backend.read_frame('a/0')[0]
        local = edit_review(frame['review'], 'add', BOUNDS, xy=[0., 0.])
        backend.save_review('a/0', local, frame['review'])
        backend.save_group('a', 'new capture', 'new place', 'val')
        backend.sync_once()
        self.assertEqual(backend.cache.counts()['pending'], 0)
        self.wire.lose_reply = True
        with self.assertRaises(httpx.ReadError):
            backend.reset_source_reviews('a', 'a/0')
        self.assertTrue(backend.cache.counts()['reset_pending'])
        with self.assertRaisesRegex(ValueError, '清空结果'):
            backend.save_review('a/0', local, local)
        backend.sync_once()
        snapshot = backend.synchronize()
        self.assertFalse(backend.cache.counts()['reset_pending'])
        self.assertEqual(backend.cache.state('a/0')['review']['status'], 'unreviewed')
        self.assertEqual(snapshot['sources'][0]['root_group'], 'new capture')
        backend.close()
        readonly = self.backend(read_only=True)
        before = len([call for call in self.wire.calls if call[0] == 'POST'])
        readonly.sync_once()
        self.assertEqual(len([call for call in self.wire.calls if call[0] == 'POST']), before)
        with self.assertRaises(ValueError):
            readonly.save_review('a/0', local, local)

    def test_cache_identity_lru_and_portable_package_has_no_server_dependencies(self):
        backend = self.backend()
        frame = backend.read_frame('a/0')[0]
        backend.save_review('a/0', edit_review(frame['review'], 'add', BOUNDS, xy=[0., 0.]), frame['review'])
        backend.cache.max_bytes = 1
        backend.read_frame('a/1')
        backend.read_frame('a/2')
        self.assertIsNone(backend.cache.bundle('a/0'))
        self.assertEqual(backend.cache.counts()['pending'], 1, 'asset LRU deleted a durable edit')
        self.assertEqual(len(backend.cache.state('a/0')['review']['centers']), 1)
        with self.assertRaises(ValueError):
            backend.cache.bind(dict(backend.identity, server_id='another'), backend.cache.meta('index'))
        archive = self.root / 'client.zip'
        package_client(archive)
        with zipfile.ZipFile(archive) as package:
            self.assertEqual(set(package.namelist()), set(FILES) | {'使用说明.md'})
            self.assertIn('conda create', package.read('使用说明.md').decode('utf-8'))
            self.assertNotIn('SOCIAL_NAV_ANNOTATION_TOKEN', package.read('使用说明.md').decode('utf-8'))
            destination = self.root / 'portable'
            package.extractall(destination)  # Fixed allowlist artifact generated by this test.
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; from perception.detection_tracking.annotation.remote import RemoteBackend; '
            'assert not set(("torch","rosbag","rospy","fastapi")) & set(sys.modules)'],
            cwd=str(destination), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

        result = subprocess.run([sys.executable, '-B', '-m',
            'perception.detection_tracking.annotation', 'open', '--help'],
            cwd=str(destination), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--project-id', result.stdout)


if __name__ == '__main__':
    unittest.main()
