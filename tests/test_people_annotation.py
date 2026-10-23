"""Annotation tests focus on mislabeled background, lost edits and input binding."""
import hashlib
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception.detection_tracking.annotation.data import (FrameLoader, Store, annotation_step,
    blank_review, complete_review, dumps, edit_review, empty_review, input_fingerprint, pending_candidate_ids,
    resolve_sequence, validate_review)
from perception.detection_tracking.annotation.export import center_info, check_splits
from perception.detection_tracking.annotation.backend import LocalBackend


BOUNDS = [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5]
BOXES = [[3., 1., .6, .6, .6, 1.7, 0., .85], [6., -2., .5, .6, .6, 1.7, 0., .4]]


class PeopleAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name) / 'review', create=True,
                           settings={}, effective={'DATA_CONFIG': {'POINT_CLOUD_RANGE': BOUNDS}})
        self.addCleanup(self.store.close)
        with self.store.connection:
            self.store.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)',
                ('source', 'unused.bag', '{}', '/points', '', 'capture', 'place', 'train', 1))
            self.store.connection.execute('INSERT INTO frames '
                '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,fingerprint,review) '
                'VALUES (?,?,?,?,?,?,?,?,?)', ('source/0', 'source', 0, 1000000001, 2000000000, 0, 'rslidar', 'bound', dumps(blank_review())))

    def reviewed(self):
        self.store.save_candidates('source/0', BOXES, {'checkpoint_sha256': 'test'}, 'bound')
        review = self.store.frame('source/0')['review']
        review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='0', decision='rejected')
        review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='1', decision='accepted')
        review = edit_review(review, 'add', BOUNDS, xy=[4., 3.])
        review['status'] = 'complete'
        self.store.save_review('source/0', review)
        return review

    def test_local_backend_uses_thread_local_connections_and_preserves_cas(self):
        backend = LocalBackend(self.store.path)
        self.addCleanup(backend.close)
        original = self.store.frame('source/0')['review']
        review = edit_review(original, 'add', BOUNDS, xy=[0., 0.])
        with ThreadPoolExecutor(max_workers=1) as worker:
            snapshot = worker.submit(backend.prepare, lambda _: None).result()
            self.assertEqual(snapshot['bounds'], BOUNDS)
            self.assertEqual(snapshot['rows'][0]['id'], 'source/0')
            frame = worker.submit(backend.save_review, 'source/0', review, original).result()
            self.assertEqual(frame['review'], review)
            with self.assertRaisesRegex(ValueError, '其他窗口'):
                worker.submit(backend.save_review, 'source/0', original, original).result()
            worker.submit(backend.set_meta, 'point_scale', 2.).result()
            worker.submit(backend.save_position, frame).result()
        self.assertEqual(self.store.meta('point_scale'), 2.)
        self.assertEqual(self.store.meta('last_frames'), {'source': 'source/0'})
        self.assertEqual(self.store.frame('source/0')['review'], review)

    def test_backend_reset_preserves_candidates_and_filters_view_metadata(self):
        self.reviewed()
        self.store.set_meta('not_a_view_preference', 'server-only')
        backend = LocalBackend(self.store.path)
        self.addCleanup(backend.close)
        snapshot = backend.snapshot()
        self.assertNotIn('not_a_view_preference', snapshot['metadata'])
        with self.assertRaises(ValueError):
            backend.set_meta('settings', {})
        result = backend.reset_source_reviews('source', 'source/0')
        self.assertEqual(result['cleared'], 1)
        self.assertEqual(result['frame']['review']['status'], 'unreviewed')
        self.assertEqual(result['frame']['candidates'], BOXES)
        self.assertEqual(result['frame']['fingerprint'], 'bound')

    def test_backend_acknowledges_own_write_when_another_writer_intervenes(self):
        backend = LocalBackend(self.store.path)
        self.addCleanup(backend.close)
        original = self.store.frame('source/0')['review']
        ours = edit_review(original, 'add', BOUNDS, xy=[0., 0.])
        theirs = edit_review(ours, 'note', BOUNDS, note='intervening writer')
        read_frame = Store.frame

        def interleaved(store, ident):
            self.store.save_review(ident, theirs, expected=ours)
            return read_frame(store, ident)

        with patch.object(Store, 'frame', interleaved):
            acknowledged = backend.save_review('source/0', ours, original)
        self.assertEqual(acknowledged['review'], ours)
        self.assertEqual(self.store.frame('source/0')['review'], theirs)
        with self.assertRaisesRegex(ValueError, '其他窗口'):
            backend.save_review('source/0', original, acknowledged['review'])

    def test_rejection_and_missing_person_export_only_final_centers(self):
        self.reviewed()
        info = center_info(self.store.frame('source/0'), BOUNDS)
        np.testing.assert_allclose(info['gt_boxes'][:, :2], [[6., -2.], [4., 3.]])
        np.testing.assert_array_equal(info['gt_box_reg_mask'], [[1, 1, 0, 0, 0, 0, 0, 0]] * 2)
        self.assertFalse(info['roi_supervision_enabled'])
        self.assertTrue(info['is_target_domain'])
        self.assertNotIn('lidar2global', info)

    def test_complete_empty_is_distinct_from_unreviewed_empty(self):
        frame = self.store.frame('source/0')
        with self.assertRaises(ValueError):
            center_info(frame, BOUNDS)
        frame['review']['status'] = 'complete'
        info = center_info(frame, BOUNDS)
        self.assertEqual(info['gt_boxes'].shape, (0, 7))
        self.assertEqual(info['gt_box_reg_mask'].shape, (0, 8))
        frame['fingerprint'] = None
        with self.assertRaises(ValueError):
            center_info(frame, BOUNDS)

    def test_pending_and_uncertain_cannot_be_background(self):
        review = blank_review()
        for state in ('pending', 'uncertain'):
            review['decisions'] = {'0': state}
            review['status'] = 'complete'
            with self.assertRaises(ValueError):
                validate_review(review, BOUNDS)

    def test_explicit_empty_rejects_hidden_candidates_and_exports_bound_background(self):
        original = self.reviewed()
        original['note'] = 'checked full point-cloud range'
        original['decisions']['0'] = 'uncertain'
        empty = empty_review(original, BOUNDS, BOXES)
        self.assertEqual(empty['status'], 'complete')
        self.assertEqual(empty['centers'], [])
        self.assertEqual(empty['decisions'], {'0': 'rejected', '1': 'rejected'})
        self.assertEqual(empty['note'], original['note'])
        self.assertEqual(len(original['centers']), 2)
        self.store.save_review('source/0', empty)
        frame = self.store.frame('source/0')
        self.assertEqual(frame['candidates'], BOXES)
        info = center_info(frame, BOUNDS)
        self.assertTrue(info['center_label_is_exhaustive'])
        self.assertEqual(info['gt_boxes'].shape, (0, 7))
        self.assertEqual(empty_review(blank_review(), BOUNDS, [])['status'], 'complete')
        with self.store.connection:
            self.store.connection.execute('UPDATE frames SET fingerprint=NULL WHERE id=?', ('source/0',))
        with self.assertRaisesRegex(ValueError, '有效扫描'):
            self.store.save_review('source/0', empty)

    def test_direct_confirmation_enrolls_untouched_frame_without_mutating_draft(self):
        review = blank_review()
        review['decisions'] = {'0': 'pending', '1': 'pending'}
        done = complete_review(review, BOUNDS, BOXES, .1)
        self.assertEqual(done['status'], 'complete')
        self.assertEqual(len(done['centers']), 2)
        self.assertEqual(review['status'], 'unreviewed')
        self.assertEqual(review['centers'], [])
        with self.assertRaisesRegex(ValueError, '低于显示阈值'):
            complete_review(review, BOUNDS, BOXES, .7)
        self.assertEqual(review['decisions'], {'0': 'pending', '1': 'pending'})
        empty = complete_review(blank_review(), BOUNDS, [], .1)
        self.assertEqual(empty['status'], 'complete')
        self.assertEqual(empty['centers'], [])

    def test_enrollment_skip_edit_and_noop_preserve_membership_rules(self):
        ordinary = blank_review()
        pending = edit_review(ordinary, 'select', BOUNDS)
        self.assertEqual(pending['status'], 'unreviewed')
        self.assertEqual(edit_review(ordinary, 'pending', BOUNDS)['status'], 'unreviewed')
        noted = edit_review(ordinary, 'note', BOUNDS, note='only a note')
        self.assertEqual(noted['status'], 'unreviewed')
        self.assertEqual(ordinary['status'], 'unreviewed')
        skipped = edit_review(ordinary, 'skip', BOUNDS)
        self.assertEqual(skipped['status'], 'skipped')
        self.assertEqual(edit_review(skipped, 'select', BOUNDS)['status'], 'skipped')
        self.assertEqual(edit_review(skipped, 'note', BOUNDS, note='review again')['status'], 'skipped')
        done = complete_review(dict(status='unreviewed', decisions={'0':'pending'}, centers=[], note=''), BOUNDS, BOXES, .1)
        ident = done['centers'][0]['id']
        self.assertEqual(edit_review(done, 'move', BOUNDS, center=ident, xy=[3., 1.])['status'], 'complete')
        self.assertEqual(edit_review(done, 'move', BOUNDS, center=ident, xy=[3.1, 1.])['status'], 'draft')
        done = complete_review(dict(status='unreviewed', decisions={'0':'pending','1':'pending'}, centers=[], note=''), BOUNDS, BOXES, .1)
        self.assertEqual(edit_review(done, 'decision', BOUNDS, BOXES, candidate='0', decision='accepted'), done)

    def test_explicit_pending_reopens_reviewed_or_skipped_without_changing_labels(self):
        done = self.reviewed()
        for status in ('complete', 'skipped'):
            review = dict(done, status=status)
            pending = edit_review(review, 'pending', BOUNDS)
            self.assertEqual(pending['status'], 'draft')
            for field in ('centers', 'decisions', 'note'):
                self.assertEqual(pending[field], review[field])
            self.assertEqual(review['status'], status)

    def test_clear_sequence_only_resets_its_human_work_and_preserves_detection_cache(self):
        self.reviewed()
        other = edit_review(blank_review(), 'add', BOUNDS, xy=[4., 2.])
        other['note'] = 'keep other bag'
        with self.store.connection:
            self.store.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)',
                ('other', 'other.bag', '{}', '/points', '', 'capture', 'place', 'train', 1))
            for ident, source, ordinal, fingerprint, review in [
                ('source/1', 'source', 1, None, dict(blank_review(), status='draft', note='outside visible time range')),
                ('other/0', 'other', 0, 'other-bound', other),
            ]:
                self.store.connection.execute('INSERT INTO frames '
                    '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,fingerprint,review) '
                    'VALUES (?,?,?,?,?,?,?,?,?)', (ident, source, ordinal, 3000000000, 3000000000, 0, 'rslidar', fingerprint, dumps(review)))
        self.store.save_position(self.store.frame('source/0'))
        self.store.set_meta('sample_intervals', {'source': .5, 'other': 1.})
        positions = self.store.meta('last_frames')
        immutable = list(self.store.connection.execute('SELECT id,candidates,model,fingerprint FROM frames ORDER BY id'))
        result = self.store.reset_source_reviews('source')
        self.assertEqual(result, {'frames': 2, 'cleared': 2})
        first = self.store.frame('source/0')['review']
        self.assertEqual(first, dict(blank_review(), decisions={'0':'pending','1':'pending'}))
        self.assertEqual(self.store.frame('source/1')['review'], blank_review())
        self.assertEqual(self.store.frame('other/0')['review'], other)
        self.assertEqual(immutable, list(self.store.connection.execute('SELECT id,candidates,model,fingerprint FROM frames ORDER BY id')))
        self.assertEqual(self.store.meta('last_frames'), positions)
        self.assertEqual(self.store.meta('sample_intervals'), {'source': .5, 'other': 1.})
        with self.assertRaises(KeyError):
            self.store.reset_source_reviews('unknown')

    def test_annotation_step_reuses_known_frame_after_manual_backtracking(self):
        rows = [dict(record_ns=i*100000000, status='unreviewed') for i in range(121)]
        for state in ('draft', 'complete', 'skipped'):
            rows[100]['status'] = state
            self.assertEqual(annotation_step(rows, 95, 1., 1), 100)
            self.assertEqual(annotation_step(rows, 100, 1., 1), 110)
            rows[105]['status'] = state
            self.assertEqual(annotation_step(rows, 100, 1., 1), 105)
            rows[105]['status'] = 'unreviewed'

    def test_annotation_step_does_not_enroll_ordinary_browsing(self):
        rows = [dict(record_ns=i*100000000, status='unreviewed') for i in range(121)]
        self.assertEqual(annotation_step(rows, 95, 1., 1), 105)
        self.assertTrue(all(row['status']=='unreviewed' for row in rows))
        rows[100]['status'] = 'complete'
        self.assertEqual(annotation_step(rows, 100, .5, 1), 105)
        self.assertEqual(annotation_step(rows, 100, 1., -1), 90)
        rows[95]['status'] = 'skipped'
        self.assertEqual(annotation_step(rows, 100, 1., -1), 95)

    def test_annotation_step_handles_boundaries_sparse_scans_and_equal_times(self):
        rows = [dict(record_ns=t, status='unreviewed') for t in (0, 0, 100000000, 300000000)]
        rows[0]['status'] = 'draft'
        self.assertEqual(annotation_step(rows, 0, .01, 1), 1)
        self.assertEqual(annotation_step(rows, 0, 1., 1), 3)
        self.assertIsNone(annotation_step(rows, 3, 1., 1))
        self.assertIsNone(annotation_step(rows, 0, 1., -1))

    def test_batch_candidates_respect_score_threshold_without_deciding_hidden_labels(self):
        review = blank_review()
        review['decisions'] = {'0': 'pending', '1': 'uncertain'}
        self.assertEqual(pending_candidate_ids(review, BOXES, .7), ['0'])
        self.assertEqual(pending_candidate_ids(review, BOXES, .4), ['0', '1'])
        for ident in pending_candidate_ids(review, BOXES, .7):
            review = edit_review(review, 'decision', BOUNDS, BOXES, candidate=ident, decision='accepted')
        self.assertEqual([c['candidate'] for c in review['centers']], ['0'])
        self.assertEqual(review['decisions']['1'], 'uncertain')
        self.assertEqual(pending_candidate_ids(review, BOXES, .7), [])
        self.assertEqual(pending_candidate_ids(review, BOXES, .4), ['1'])
        review['status'] = 'complete'
        with self.assertRaises(ValueError):
            validate_review(review, BOUNDS)

    def test_removing_last_manual_operation_returns_to_ordinary(self):
        review = dict(blank_review(), decisions={'0': 'pending', '1': 'pending'})
        added = edit_review(review, 'add', BOUNDS, xy=[.1, .2])
        self.assertEqual(added['status'], 'draft')
        deleted = edit_review(added, 'delete', BOUNDS, center=added['centers'][0]['id'])
        self.assertEqual(deleted, review)
        for decision in ('accepted', 'rejected'):
            handled = edit_review(review, 'decision', BOUNDS, BOXES, candidate='0', decision=decision)
            self.assertEqual(handled['status'], 'draft')
            cleared = edit_review(handled, 'decision', BOUNDS, BOXES, candidate='0', decision='pending')
            self.assertEqual(cleared, review)
        empty_complete = dict(blank_review(), status='complete')
        self.assertEqual(edit_review(empty_complete, 'pending', BOUNDS)['status'], 'unreviewed')

    def test_repeated_accept_preserves_moved_center_without_duplication(self):
        review = blank_review()
        review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='1', decision='accepted')
        ident = review['centers'][0]['id']
        review = edit_review(review, 'move', BOUNDS, center=ident, xy=[6.2, -2.1])
        review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='1', decision='accepted')
        self.assertEqual(len(review['centers']), 1)
        self.assertEqual(review['centers'][0]['xy'], [6.2, -2.1])
        for xy in ([20, 0], [float('nan'), 2]):
            with self.assertRaises(ValueError):
                edit_review(review, 'add', BOUNDS, xy=xy)

    def test_near_origin_centers_can_be_added_moved_confirmed_and_exported(self):
        near_boxes = [[.25, -.4, .6, .6, .6, 1.7, 0., .85]]
        self.store.save_candidates('source/0', near_boxes, {'checkpoint_sha256': 'test'}, 'bound')
        original = self.store.frame('source/0')['review']
        self.assertEqual(original['decisions'], {'0': 'pending'})
        review = edit_review(original, 'add', BOUNDS, xy=[0., 0.])
        ident = review['centers'][0]['id']
        review = edit_review(review, 'move', BOUNDS, center=ident, xy=[-.2, .3])
        review = complete_review(review, BOUNDS, near_boxes, .1)
        self.store.save_review('source/0', review, expected=original)
        frame = self.store.frame('source/0')
        self.assertEqual(frame['review'], review)
        info = center_info(frame, BOUNDS)
        np.testing.assert_allclose(info['gt_boxes'][:, :2], [[-.2, .3], [.25, -.4]])
        np.testing.assert_array_equal(info['gt_box_reg_mask'], [[1, 1, 0, 0, 0, 0, 0, 0]] * 2)

    def test_late_candidate_does_not_overwrite_completed_manual_review(self):
        review = edit_review(blank_review(), 'add', BOUNDS, xy=[4., 3.])
        review['status'] = 'complete'
        self.store.save_review('source/0', review)
        self.store.save_candidates('source/0', BOXES, {'checkpoint_sha256': 'test'}, 'bound')
        self.assertEqual(self.store.frame('source/0')['review'], review)
        self.assertEqual(len(center_info(self.store.frame('source/0'), BOUNDS)['gt_boxes']), 1)

    def test_wrong_input_and_model_cannot_replace_review_basis(self):
        with self.assertRaises(ValueError):
            self.store.save_candidates('source/0', [[1, 2], [3, 4], [5, 6], [7, 8]], {}, 'bound')
        with self.assertRaises(ValueError):
            self.store.save_candidates('source/0', BOXES, {}, 'different')
        original = self.reviewed()
        with self.assertRaises(ValueError):
            self.store.save_candidates('source/0', BOXES, {'checkpoint_sha256': 'other'}, 'bound')
        self.assertEqual(self.store.frame('source/0')['review'], original)

    def test_concurrent_edit_is_not_silently_overwritten(self):
        original = blank_review()
        one = edit_review(original, 'add', BOUNDS, xy=[4., 2.])
        self.store.save_review('source/0', one, expected=original)
        two = edit_review(original, 'add', BOUNDS, xy=[5., 2.])
        with self.assertRaisesRegex(ValueError, '其他窗口'):
            self.store.save_review('source/0', two, expected=original)

    def test_source_and_location_splits_do_not_leak(self):
        train = dict(root_group='capture_a', location_group='room', split='train')
        val = dict(root_group='capture_b', location_group='room', split='val')
        with self.assertRaises(ValueError):
            check_splits([train, val])
        val['location_group'] = 'other_room'
        check_splits([train, val])
        val['root_group'] = 'capture_a'
        with self.assertRaises(ValueError):
            check_splits([train, val])

    def test_sub_bags_with_same_capture_keep_separate_positions_and_labels(self):
        with self.store.connection:
            self.store.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)',
                ('other', 'other.bag', '{}', '/points', '', 'capture', 'place', 'train', 1))
            self.store.connection.execute('INSERT INTO frames '
                '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,fingerprint,review) '
                'VALUES (?,?,?,?,?,?,?,?,?)', ('other/0', 'other', 0, 3000000000, 3000000000, 0, 'rslidar', 'bound', dumps(blank_review())))
        reviewed = self.reviewed()
        self.store.save_position(self.store.frame('source/0'))
        self.store.save_position(self.store.frame('other/0'))
        self.assertEqual(self.store.meta('last_frames'), {'source': 'source/0', 'other': 'other/0'})
        self.assertEqual(self.store.meta('last_frame'), 'other/0')
        self.assertEqual(self.store.frame('source/0')['review'], reviewed)
        self.assertEqual(self.store.frame('other/0')['review'], blank_review())

    def test_sequence_selection_disambiguates_duplicate_bag_names(self):
        sources = [dict(id='a', path='/one/clip.bag'), dict(id='b', path='/two/clip.bag')]
        self.assertEqual(resolve_sequence(sources, 'a'), 'a')
        self.assertEqual(resolve_sequence(sources, '/two/clip.bag'), 'b')
        with self.assertRaisesRegex(ValueError, '不唯一'):
            resolve_sequence(sources, 'clip.bag')

    def test_open_rejects_gpu_options_and_predict_keeps_them(self):
        import contextlib
        import io
        from perception.detection_tracking.annotation.__main__ import arguments
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                arguments(['open', '--project', self.temporary.name, '--gpu', '0'])
        args = arguments(['open', '--project', self.temporary.name, '--sequence', 'clip.bag'])
        self.assertEqual(args.sequence, 'clip.bag')
        args = arguments(['predict', '--project', self.temporary.name,
                          '--detector-python', '/usr/bin/python3', '--gpu', '2'])
        self.assertEqual(args.gpu, '2')

    def test_fingerprint_retains_existing_cache_contract(self):
        points = np.array([[3., 1., .4, 55.]], dtype='<f4')
        value = hashlib.sha256()
        value.update('seq\nobs\n1.25\nrslidar\n(1, 4)\n'.encode())
        value.update(points.tobytes())
        self.assertEqual(input_fingerprint(points, 'seq', 'obs', 1.25, 'rslidar'), value.hexdigest())

    def test_frame_lru_reuses_bag_assets_but_reads_fresh_human_labels(self):
        path = Path(self.temporary.name) / 'source.bag'
        path.write_bytes(b'synthetic source, fake decoder below')
        with self.store.connection:
            self.store.connection.execute('UPDATE sources SET path=?', (str(path),))
            self.store.connection.execute('UPDATE frames SET fingerprint=NULL')
        points = np.array([[3., 1., .5, 128]], dtype=np.float32)
        with patch('perception.detection_tracking.annotation.data.BagReader') as reader:
            reader.return_value.read.return_value = points, None, None
            reader.return_value.image_notice = ''
            loader = FrameLoader(self.store.path)
            try:
                first = loader.read('source/0')
                review = edit_review(first[0]['review'], 'add', BOUNDS, xy=[4., 3.])
                self.store.save_review('source/0', review)
                second = loader.read('source/0')
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(reader.return_value.read.call_count, 1)
                self.assertIs(first[1], second[1])
                self.assertEqual(second[0]['review'], review)
                self.assertLessEqual(loader.bytes, loader.max_bytes)
            finally:
                loader.close()

    def test_offline_processes_each_scan_and_resume_preserves_human_edits(self):
        from perception.detection_tracking.annotation.offline import predict_project
        with self.store.connection:
            self.store.connection.execute('UPDATE frames SET fingerprint=NULL')
            self.store.connection.execute('INSERT INTO frames '
                '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,review) '
                'VALUES (?,?,?,?,?,?,?,?)', ('source/1', 'source', 1, 1100000001, 2100000000, 0, 'rslidar', dumps(blank_review())))
        manual = edit_review(blank_review(), 'add', BOUNDS, xy=[4., 3.])
        self.store.save_review('source/0', manual)
        def stamp(ns):
            return SimpleNamespace(to_nsec=lambda: ns)
        messages = [('/points', SimpleNamespace(header=SimpleNamespace(stamp=stamp(s), frame_id='rslidar')), stamp(r))
                    for s, r in [(1000000001, 2000000000), (1100000001, 2100000000)]]
        with patch('perception.detection_tracking.annotation.offline.index_source'), \
             patch('perception.detection_tracking.annotation.offline.BagReader') as reader, \
             patch('perception.detection_tracking.annotation.offline.decode_cloud', return_value=np.array([[3., 1., .5, 128]], dtype=np.float32)), \
             patch('perception.detection_tracking.annotation.offline.DetectorClient') as client:
            reader.return_value.bag.read_messages.side_effect = lambda **_: iter(messages)
            client.return_value.metadata = dict(point_cloud_range=BOUNDS, checkpoint_sha256='test')
            client.return_value.predict.return_value = BOXES
            result = predict_project(self.store.path, 'unused', '0', lambda _: None)
            self.assertTrue(result['complete'])
            self.assertEqual(result['processed'], 2)
            self.assertAlmostEqual(result['sources'][0]['observation_hz'], 10.)
            self.assertEqual(self.store.frame('source/0')['review'], manual)
            self.assertEqual(result['sources'][0]['missing_predictions'], 0)
            client.return_value.predict.reset_mock()
            result = predict_project(self.store.path, 'unused', '0', lambda _: None)
            self.assertEqual(result['reused'], 2)
            client.return_value.predict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
