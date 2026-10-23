"""Dataset export: supervision boundaries, real raw-history counts and portability."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from perception.detection_tracking.annotation.data import Store, blank_review, dumps, edit_review, input_fingerprint
from perception.detection_tracking.annotation.export import export_project, plan_export
from perception.detection_tracking.training.simpledet3d.simple_det3d.datasets.rosbag_centers import load_rosbag_centers
from perception.detection_tracking.annotation.poses import RecordedPoses


BOUNDS = [-4.8, -9.6, -1.5, 14.4, 9.6, 4.5]
BOXES = [[3., 1., .6, .6, .6, 1.7, 0., .85], [6., -2., .5, .6, .6, 1.7, 0., .4]]


def points_for(frame):
    return np.array([[frame['ordinal'] + 1., 0., .5, 128.]], dtype='<f4')


class FakeBagReader:
    calls = []
    missing_rgb = False

    def __init__(self, source):
        self.source = source

    def read(self, frame, rgb=True, max_rgb_age_s=.1):
        self.calls.append((frame['id'], rgb))
        image = np.full((2, 3, 3), frame['ordinal'], dtype=np.uint8) if rgb and not self.missing_rgb else None
        return points_for(frame), image, frame['stamp_ns'] - 1000000 if image is not None else None

    def close(self):
        pass


class PeopleDatasetExportTests(unittest.TestCase):
    def pose_run(self, source, mode='localization', invalid=(), generations=None):
        directory = self.root / 'poses' / mode / source
        directory.mkdir(parents=True, exist_ok=True)
        run = dict(bag=source + '.bag', mode=mode, success=True, backend='roll_fast_lio',
                   assume_stationary=False, geometry_accuracy_verified=False,
                   playback={'bag_fingerprint': {}})
        (directory/'run.json').write_text(json.dumps(run))
        rows = []
        for i in range(6 if source == 'a' else 1):
            # One quarter-turn orientation makes world X motion become local Y.
            row = dict(stamp_ns=1000000000+i*100000000, position=[float(i),0.,0.],
                       quaternion_xyzw=[0.,0.,2**-.5,2**-.5], frame_id='odom', child_frame_id='rslidar',
                       session_id='session-'+source, generation=(generations or {}).get(i,0),
                       health=2 if i in invalid else 0, status='INITIALIZING' if i in invalid else 'TRACKING',
                       reason='imu_initialization' if i in invalid else '', source='fast_lio',
                       has_map_pose=mode=='localization', map_frame='map', map_revision=i,
                       map_position=[float(i*100),0.,0.], map_quaternion_xyzw=[0.,0.,0.,1.])
            rows.append(row)
        (directory/'localization.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return directory

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        store = Store(self.project, create=True, settings={}, effective={'DATA_CONFIG': {'POINT_CLOUD_RANGE': BOUNDS}})
        try:
            for source, count, split in [('a', 6, 'train'), ('b', 1, 'val')]:
                with store.connection:
                    store.connection.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)',
                        (source, source + '.bag', '{}', '/points', '/rgb', 'capture-' + source, 'place-' + source, split, 1))
                    for ordinal in range(count):
                        ident = '%s/%08d' % (source, ordinal)
                        stamp = 1000000000 + ordinal * 100000000
                        record = 2000000000 + ordinal * 100000000
                        frame = dict(id=ident, source_id=source, ordinal=ordinal, stamp_ns=stamp, sensor_frame='rslidar')
                        binding = input_fingerprint(points_for(frame), source, ident, stamp / 1e9, 'rslidar')
                        store.connection.execute('INSERT INTO frames '
                            '(id,source_id,ordinal,stamp_ns,record_ns,record_rank,sensor_frame,fingerprint,review) VALUES (?,?,?,?,?,?,?,?,?)',
                            (ident, source, ordinal, stamp, record, 0, 'rslidar', binding, dumps(blank_review())))
                for ordinal in range(count):
                    ident = '%s/%08d' % (source, ordinal)
                    store.save_candidates(ident, BOXES, {'checkpoint_sha256': 'fixture'}, store.frame(ident)['fingerprint'])
                if source == 'a':
                    review = store.frame('a/00000001')['review']
                    review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='0', decision='accepted')
                    review = edit_review(review, 'add', BOUNDS, xy=[.2, .3])
                    store.save_review('a/00000001', review)
                    review = store.frame('a/00000002')['review']
                    review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='0', decision='rejected')
                    review = edit_review(review, 'decision', BOUNDS, BOXES, candidate='1', decision='accepted')
                    store.save_review('a/00000002', dict(review, status='complete'))
                    store.save_review('a/00000003', edit_review(store.frame('a/00000003')['review'], 'skip', BOUNDS))
                    # Existing legacy auto-enrolled draft: preserve source state, but no fake negative label.
                    store.save_review('a/00000005', dict(store.frame('a/00000005')['review'], status='draft'))
                else:
                    review = store.frame('b/00000000')['review']
                    for candidate in ('0', '1'):
                        review = edit_review(review, 'decision', BOUNDS, BOXES, candidate=candidate, decision='rejected')
                    store.save_review('b/00000000', dict(review, status='complete'))
        finally:
            store.close()
        FakeBagReader.calls = []
        FakeBagReader.missing_rgb = False
        self.reader_patch = patch('perception.detection_tracking.annotation.export.BagReader', FakeBagReader)
        self.reader_patch.start()
        self.addCleanup(self.reader_patch.stop)

    def test_raw_history_uses_unmarked_scans_deduplicates_and_stays_within_bag(self):
        plan = plan_export(self.project, history_frames=2)
        self.assertEqual(plan['counts']['samples'], 4)
        self.assertEqual(plan['counts']['pointclouds'], 7)
        self.assertEqual(plan['counts']['history_only_pointclouds'], 3)
        self.assertEqual(plan['counts']['samples_with_short_history'], 2)
        sample = next(s for s in plan['samples'] if s['id'] == 'a/00000005')
        self.assertEqual([h['pointcloud_id'] for h in sample['history']], ['a/00000004', 'a/00000003'])
        self.assertTrue(all(h['time_lag_s'] > 0 for s in plan['samples'] for h in s['history']))
        self.assertEqual(plan['samples'][-1]['history'], [])
        stride = plan_export(self.project, history_frames=2, history_stride=2, sequences=['a'])
        sample = next(s for s in stride['samples'] if s['ordinal'] == 5)
        self.assertEqual([h['pointcloud_id'] for h in sample['history']], ['a/00000003', 'a/00000001'])
        self.assertFalse(FakeBagReader.calls)

    def test_export_has_only_clouds_current_images_and_verified_xy_labels(self):
        out = self.root / 'dataset'
        counts = export_project(self.project, out, history_frames=2)
        self.assertEqual(set(p.name for p in out.iterdir()), {'points', 'images', 'annotations.json'})
        document = json.loads((out / 'annotations.json').read_text())
        self.assertEqual(counts['images'], 4)
        self.assertEqual(len(list(out.glob('points/**/*.bin'))), 7)
        self.assertEqual(len(list(out.glob('images/**/*.png'))), 4)
        self.assertEqual(len(FakeBagReader.calls), 7)
        self.assertEqual(sum(rgb for _, rgb in FakeBagReader.calls), 4)
        first = document['samples'][0]
        self.assertEqual([c['xy'] for c in first['annotation']['centers']], [[3., 1.], [.2, .3]])
        self.assertFalse(first['annotation']['center_label_is_exhaustive'])
        self.assertEqual(first['annotation']['ignored_candidate_count'], 1)
        self.assertNotIn('gt_boxes', first['annotation'])
        for sample in document['samples']:
            pixels = np.asarray(Image.open(out / sample['image']['path']))
            self.assertTrue(np.all(pixels == sample['ordinal']))
        clouds = {p['id']: p for p in document['pointclouds']}
        for cloud in clouds.values():
            raw = np.fromfile(out / cloud['path'], dtype='<f4').reshape(-1,4)
            self.assertEqual(input_fingerprint(raw, cloud['source_id'], cloud['id'], cloud['stamp_ns']/1e9, cloud['sensor_frame']), cloud['input_fingerprint'])
        infos = load_rosbag_centers(out / 'annotations.json', 'train', point_cloud_range=BOUNDS)
        self.assertEqual(len(infos), 3)
        self.assertFalse(infos[0]['center_label_is_exhaustive'])
        self.assertTrue(infos[1]['center_label_is_exhaustive'])
        self.assertFalse(infos[2]['center_label_is_exhaustive'])
        self.assertEqual(infos[2]['gt_boxes'].shape, (0,7))
        np.testing.assert_array_equal(infos[0]['gt_box_reg_mask'], [[1,1,0,0,0,0,0,0]]*2)
        with self.assertRaisesRegex(ValueError, '补偿'):
            load_rosbag_centers(out / 'annotations.json', 'train', max_sweeps=3)

    def test_complete_only_and_source_time_filter_do_not_clip_history(self):
        only = plan_export(self.project, complete_only=True)
        self.assertEqual(only['counts']['samples'], 2)
        self.assertEqual(only['counts']['statuses'], {'complete': 2})
        scoped = plan_export(self.project, sequences=['a'], start=.5, duration=.1, history_frames=2)
        self.assertEqual(scoped['counts']['samples'], 1)
        self.assertEqual(scoped['counts']['pointclouds'], 3)

    def test_image_pairing_window_is_explicit_and_finite(self):
        out = self.root / 'image-window'
        export_project(self.project, out, max_image_age_s=.15)
        document=json.loads((out/'annotations.json').read_text())
        self.assertEqual(document['image_pairing'], {'mode':'latest_past','max_age_s':.15})
        for age in (0., -1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, '时间窗'):
                export_project(self.project, self.root/'invalid-window', max_image_age_s=age)

    def test_unsplit_export_is_allowed_but_training_is_blocked(self):
        store = Store(self.project)
        try:
            store.set_group('a', 'capture-a', '', 'unassigned')
        finally:
            store.close()
        out = self.root / 'unsplit'
        export_project(self.project, out)
        with self.assertRaisesRegex(ValueError, '划分'):
            load_rosbag_centers(out / 'annotations.json', 'train')

    def test_missing_rgb_is_explicit_atomic_and_does_not_edit_source(self):
        before = (self.project / 'annotations.sqlite').read_bytes()
        FakeBagReader.missing_rgb = True
        out = self.root / 'missing'
        with self.assertRaisesRegex(ValueError, 'RGB'):
            export_project(self.project, out)
        self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob('.missing-*')))
        export_project(self.project, out, allow_missing_images=True)
        document=json.loads((out/'annotations.json').read_text())
        self.assertEqual(document['counts']['missing_images'], 4)
        self.assertTrue(all(s['image'] is None for s in document['samples']))
        self.assertEqual((self.project / 'annotations.sqlite').read_bytes(), before)
        with self.assertRaises(FileExistsError):
            export_project(self.project, out)

    def test_wrong_history_binding_and_noncausal_time_fail(self):
        store = Store(self.project)
        try:
            with store.connection:
                store.connection.execute("UPDATE frames SET fingerprint='wrong' WHERE id='a/00000000'")
        finally:
            store.close()
        with self.assertRaisesRegex(ValueError, '绑定'):
            export_project(self.project, self.root/'wrong', history_frames=1)
        store = Store(self.project)
        try:
            with store.connection:
                store.connection.execute("UPDATE frames SET stamp_ns=9000000000 WHERE id='a/00000000'")
        finally:
            store.close()
        with self.assertRaisesRegex(ValueError, '因果'):
            plan_export(self.project, history_frames=1)

    def test_recorded_localization_is_preferred_and_map_corrections_do_not_move_history(self):
        self.pose_run('a'); self.pose_run('a','odometry'); self.pose_run('b','odometry')
        out=self.root/'motion'
        export_project(self.project,out,history_frames=2,pose_results=self.root/'poses')
        d=json.loads((out/'annotations.json').read_text())
        self.assertTrue(d['history']['transforms_available'])
        self.assertFalse(d['history']['motion_compensated'])  # Files remain original XYZI.
        self.assertEqual([s['pose_source']['mode'] for s in d['sources']],['localization','odometry'])
        clouds={p['id']:p for p in d['pointclouds']}
        sample=next(s for s in d['samples'] if s['id']=='a/00000002')
        self.assertEqual(clouds[sample['pointcloud_id']]['pose']['map_from_lidar'][0][3],200.)
        matrix=np.asarray(sample['history'][1]['current_from_history'])
        np.testing.assert_allclose(matrix[:3,:3],np.eye(3),atol=1e-12)
        np.testing.assert_allclose(matrix[:3,3],[0.,2.,0.],atol=1e-12)
        infos=load_rosbag_centers(out/'annotations.json','train',max_sweeps=3,point_cloud_range=BOUNDS)
        self.assertTrue(infos[1]['has_pose'])
        self.assertEqual(len(infos[1]['sweeps']),2)
        np.testing.assert_allclose(infos[1]['sweeps'][1]['transform_matrix'],matrix)
        bad=json.loads((out/'annotations.json').read_text())
        bad['samples'][1]['history'][0]['current_from_history'][0][3]+=1
        (out/'bad.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'不一致'):
            load_rosbag_centers(out/'bad.json','train',max_sweeps=3)

    def test_missing_initial_pose_keeps_current_and_records_unusable_history(self):
        self.pose_run('a',invalid=(0,1)); self.pose_run('b','odometry',invalid=(0,))
        out=self.root/'startup'
        export_project(self.project,out,history_frames=2,pose_results=self.root/'poses')
        d=json.loads((out/'annotations.json').read_text())
        self.assertEqual(d['counts']['samples_without_current_pose'],2)
        self.assertEqual(d['samples'][0]['motion']['status'],'unavailable')
        self.assertIsNone(d['samples'][0]['history'][0]['current_from_history'])
        self.assertIn('current:pose_invalid',d['samples'][0]['history'][0]['pose_reason'])
        infos=load_rosbag_centers(out/'annotations.json','train',max_sweeps=3)
        self.assertFalse(infos[0]['has_pose'])
        self.assertTrue(infos[0]['motion_compensation_prepared'])
        self.assertEqual(infos[0]['sweeps'],[])
        self.assertEqual(infos[1]['sweeps'],[])
        self.assertEqual(len(infos[2]['sweeps']),2)

    def test_pose_lifecycle_boundary_is_not_used_as_motion(self):
        self.pose_run('a',generations={2:1}); self.pose_run('b','odometry')
        plan=plan_export(self.project,history_frames=2,pose_results=self.root/'poses')
        sample=next(s for s in plan['samples'] if s['id']=='a/00000002')
        self.assertTrue(sample['motion']['current_pose_valid'])
        self.assertEqual(sample['motion']['usable_history_frames'],0)
        self.assertTrue(all(h['current_from_history'] is None and h['pose_reason']=='pose_lifecycle_mismatch'
                            for h in sample['history']))

    def test_pose_matching_never_uses_future_or_hides_a_new_invalid_state(self):
        self.pose_run('a',invalid=(1,)); self.pose_run('b','odometry')
        store=Store(self.project,read_only=True)
        try:
            source=dict(store.connection.execute("SELECT * FROM sources WHERE id='a'").fetchone())
        finally:
            store.close()
        provider=RecordedPoses(self.root/'poses',source,.02)
        frame=dict(stamp_ns=999999999,sensor_frame='rslidar')
        self.assertEqual(provider.resolve(frame)['reason'],'before_first_pose')
        frame['stamp_ns']=1099000000  # A valid future estimate exists but is ineligible.
        self.assertEqual(provider.resolve(frame)['reason'],'pose_too_old')
        frame['stamp_ns']=1100000000
        self.assertTrue(provider.resolve(frame)['reason'].startswith('pose_invalid:'))
        frame.update(stamp_ns=1200000000,sensor_frame='other_lidar')
        self.assertEqual(provider.resolve(frame)['reason'],'sensor_frame_mismatch')

    def test_wrong_pose_input_binding_is_rejected_before_export(self):
        directory=self.pose_run('a'); self.pose_run('b','odometry')
        run=json.loads((directory/'run.json').read_text())
        run['playback']['bag_fingerprint']={'sha256':'other'}
        (directory/'run.json').write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError,'bag指纹'):
            plan_export(self.project,history_frames=2,pose_results=self.root/'poses')


if __name__ == '__main__':
    unittest.main()
