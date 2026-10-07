"""CPU regressions for causal perception/ROS transport; no models, API or robot motion."""
import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from perception.detection_tracking.inference.config import DEFAULT, load_config, merge
from perception.detection_tracking.inference.geometry import (
    corners_xyz, similarity, transform_boxes, validate_boxes, validate_transform,
)
from perception.detection_tracking.inference.detector import prepare_single_frame
from perception.detection_tracking.inference.dataset import WheelchairSource, ManifestSource, CachedDetector
from perception.detection_tracking.inference.pipeline import Frame, PerceptionPipeline, to_snapshot
from perception.detection_tracking.inference.tracker import SimpleTrack
from perception.detection_tracking.inference.ros_inputs import decode_pointcloud, PoseBuffer, LocalizedPose, LatestSlot

CONFIG = json.loads(DEFAULT.read_text())


def box(x=3, y=0, score=.9):
    return np.array([x, y, 0, .6, .5, 1.7, 0, score], dtype=float)


class Detector:
    def __init__(self, boxes=None):
        self.boxes = np.array([box()]) if boxes is None else boxes

    def predict(self, _):
        return self.boxes.copy()


def frame(stamp=1, pose=None, source='dataset_pose', sequence='fixture'):
    return Frame(str(stamp), sequence, stamp, np.array([[3, 0, 0, 255]], dtype=np.float32),
                 world_from_lidar=np.eye(4) if pose is None else pose,
                 pose_health='valid', pose_source=source)


class GeometryTests(unittest.TestCase):
    def test_geometry_identical_and_disjoint_boxes(self):
        for metric in ('iou','giou','diou'):
            self.assertAlmostEqual(similarity(box(), box(), metric), 1)
            self.assertLessEqual(similarity(box(), box(20), metric), 0)
        self.assertAlmostEqual(similarity(box(), box(20), 'iou'), 0)

    def test_rotation_and_translation_are_correct(self):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_euler('z', np.pi/2).as_matrix()
        pose[:3,3] = [1,2,3]
        result = transform_boxes([box()], pose)[0]
        np.testing.assert_allclose(result[:3], [1,5,3], atol=1e-8)
        np.testing.assert_allclose(result[3:6], box()[3:6], atol=1e-8)
        self.assertAlmostEqual(result[6], np.pi/2)

    def test_tilted_envelope_contains_all_transformed_corners(self):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_euler('xyz', [.2,.3,.5]).as_matrix()
        result = transform_boxes([box()], pose)[0]
        corners = corners_xyz(box()) @ pose[:3,:3].T
        local = (corners-result[:3]) @ Rotation.from_euler('z', result[6]).as_matrix()
        self.assertTrue(np.all(np.abs(local) <= result[3:6]/2+1e-8))

    def test_invalid_transform_and_boxes_fail_closed(self):
        bad = np.eye(4); bad[0,0] = 2
        with self.assertRaises(ValueError): validate_transform(bad)
        for value in (np.ones((2,7)), [np.r_[box()[:3], [-1,.5,1], [0,.9]]], [box(score=2)]):
            with self.assertRaises(ValueError): validate_boxes(value)

    def test_single_frame_features_match_training_convention(self):
        points = np.array([[.5,.5,0,255],[3,0,1,255],[4,0,1,127.5],[np.nan,0,0,1]], dtype=np.float32)
        result = prepare_single_frame(points)
        np.testing.assert_allclose(result, [[3,0,1,1,0],[4,0,1,.5,0]])
        self.assertTrue(np.isnan(points[-1,0]))
        with self.assertRaises(ValueError): prepare_single_frame(np.empty((0,4)))
        with self.assertRaises(ValueError): prepare_single_frame(np.zeros((1,5)))


class TrackerTests(unittest.TestCase):
    def tracker(self): return SimpleTrack(CONFIG['tracking'])

    def test_identity_confirmation_velocity_and_empty_prediction(self):
        tracker = self.tracker()
        a = tracker.step([box()], 1)[0]
        b = tracker.step([box(3.1)], 1.1)[0]
        c = tracker.step([box(3.2)], 1.2)[0]
        self.assertEqual(a['track_id'], c['track_id'])
        self.assertFalse(a['confirmed']); self.assertTrue(b['confirmed'])
        self.assertIsNone(b['velocity_xyz_mps']); self.assertIsNotNone(c['velocity_xyz_mps'])
        self.assertGreater(c['velocity_xyz_mps'][0], 0)
        prediction = tracker.step([], 1.3)[0]
        self.assertEqual(prediction['state'], 'predicted')
        self.assertEqual(prediction['last_observed_stamp_s'], 1.2)
        self.assertEqual(len(prediction['covariance']), 100)
        self.assertGreater(np.trace(np.array(prediction['covariance']).reshape(10,10)),
                           np.trace(np.array(c['covariance']).reshape(10,10)))

    def test_low_score_support_is_not_a_fake_observation(self):
        tracker = self.tracker()
        self.assertEqual(tracker.step([box(score=.2)], 1), [])
        tracker.step([box()], 1.1)
        supported = tracker.step([box(score=.2)], 1.2)[0]
        self.assertEqual(supported['support'], 'low_score_support')
        self.assertEqual(supported['state'], 'predicted')
        self.assertEqual(supported['last_observed_stamp_s'], 1.1)
        self.assertEqual(supported['last_supported_stamp_s'], 1.2)

    def test_expired_tracks_cannot_be_resurrected_across_a_gap(self):
        tracker = self.tracker()
        old = tracker.step([box()], 1)[0]
        new = tracker.step([box()], 1.8)[0]
        self.assertNotEqual(old['track_id'], new['track_id'])

    def test_duplicate_and_reverse_time_are_rejected(self):
        tracker = self.tracker(); tracker.step([box()], 1)
        for stamp in (1,.9,float('nan')):
            with self.assertRaises(ValueError): tracker.step([], stamp)

    def test_class_config_not_mutated_and_cpu_metrics_do_not_depend_on_cuda(self):
        config = copy.deepcopy(CONFIG['tracking'])
        for metric in ('giou','iou','diou'):
            config['association'] = metric
            tracker = SimpleTrack(config)
            self.assertEqual(tracker.step([box()], 1)[0]['track_id'], tracker.step([box()], 1.1)[0]['track_id'])
            self.assertEqual(config['association'], metric)

    def test_distance_gate_prevents_far_person_assignment(self):
        tracker = self.tracker(); old = tracker.step([box()], 1)[0]
        result = tracker.step([box(9)], 1.1)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[1]['track_id'], old['track_id'])


class PipelineTests(unittest.TestCase):
    def pipeline(self, detector=None): return PerceptionPipeline(detector or Detector(), CONFIG['tracking'], 'test')

    def test_raw_odometry_never_becomes_reliable_localization(self):
        pipeline = self.pipeline()
        result = pipeline.step(frame(source='raw_odometry'))
        self.assertEqual(result['health'], 'degraded')
        self.assertEqual(result['tracks'], [])
        self.assertEqual(len(result['detections']), 1)
        self.assertIsNone(result['world_from_lidar'])

    def test_pose_loss_and_recovery_invalidate_identity_lifecycle(self):
        pipeline = self.pipeline()
        before = pipeline.step(frame())
        missing = frame(1.1); missing.world_from_lidar = None
        self.assertEqual(pipeline.step(missing)['tracks'], [])
        after = pipeline.step(frame(1.2))
        self.assertNotEqual(before['tracks'][0]['track_id'], after['tracks'][0]['track_id'])
        self.assertGreater(after['generation'], before['generation'])

    def test_ego_translation_does_not_create_person_velocity(self):
        detector = Detector(); pipeline = self.pipeline(detector)
        for index in range(4):
            pose = np.eye(4); pose[0,3] = index*.1
            detector.boxes = np.array([box(3-index*.1)])
            result = pipeline.step(frame(1+index*.1, pose))
        np.testing.assert_allclose(result['tracks'][0]['velocity_xyz_mps'], [0,0,0], atol=1e-9)

    def test_sequence_reset_changes_lifecycle_and_duplicate_time_fails(self):
        pipeline = self.pipeline()
        a = pipeline.step(frame())
        b = pipeline.step(frame(.1, sequence='other'))
        self.assertGreater(b['generation'], a['generation'])
        with self.assertRaises(ValueError): pipeline.step(frame(.1, sequence='other'))

    def test_shared_snapshot_preserves_validity_and_covariance_order(self):
        result = self.pipeline().step(frame())
        snapshot = to_snapshot(result)
        self.assertEqual(snapshot.health, 'valid')
        self.assertEqual(snapshot.header.frame_id, 'dataset_world')
        self.assertFalse(snapshot.tracks[0].confirmed)
        self.assertEqual(snapshot.tracks[0].covariance_order[-3:], ('vx','vy','vz'))
        self.assertEqual(snapshot.header.timestamp_source, 'sensor')

    def test_localization_time_is_not_rewritten_as_cloud_time(self):
        observation = frame(); observation.pose_stamp_s = .98
        snapshot = to_snapshot(self.pipeline().step(observation))
        self.assertEqual(snapshot.header.stamp_s, 1)
        self.assertEqual(snapshot.ego_pose.header.stamp_s, .98)
        observation.pose_stamp_s = 1.01
        with self.assertRaises(ValueError): self.pipeline().step(observation)

    def test_failure_is_invalid_not_a_successful_empty_scene(self):
        pipeline = self.pipeline()
        result = pipeline.failure(frame(), 'bad input')
        self.assertEqual(result['health'], 'invalid')
        self.assertEqual(to_snapshot(result).health, 'invalid')

    def test_localization_jump_invalidates_tracks_instead_of_creating_velocity(self):
        pipeline = self.pipeline()
        old = pipeline.step(frame())
        pose = np.eye(4); pose[0,3] = 10
        jump = pipeline.step(frame(1.1, pose))
        self.assertEqual(jump['reason'], 'localization_discontinuity')
        self.assertEqual(jump['tracks'], [])
        recovered = pipeline.step(frame(1.2, pose))
        self.assertNotEqual(old['tracks'][0]['track_id'], recovered['tracks'][0]['track_id'])

    def test_cached_predictions_are_bound_to_points_and_time(self):
        source = frame()
        result = self.pipeline().step(source)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'observations.jsonl'
            path.write_text(json.dumps(result)+'\n')
            replay = CachedDetector(path)
            np.testing.assert_allclose(replay.predict(source), result['detections'])
            source.points[0,0] += 1
            with self.assertRaises(ValueError): replay.predict(source)


class DatasetTests(unittest.TestCase):
    def test_dataset_pose_is_used_but_missing_pose_never_defaults_to_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            scene = Path(folder)/'indoor1/test'
            (scene/'points').mkdir(parents=True); (scene/'poses').mkdir()
            for index in (1,3):
                np.array([[3,0,0,255]], dtype='<f4').tofile(scene/'points'/f'{index:04d}.bin')
            (scene/'poses/0001.txt').write_text('pose0001: 1 2 3 0 0 0 1\n')
            rows = list(WheelchairSource(folder, 'indoor1/test'))
            np.testing.assert_allclose(rows[0].world_from_lidar[:3,3], [1,2,3])
            self.assertIsNone(rows[1].world_from_lidar)
            self.assertAlmostEqual(rows[1].stamp_s-rows[0].stamp_s, .2)
            self.assertEqual(rows[0].timestamp_source, 'nominal_frame_index')
            with self.assertRaises(ValueError): WheelchairSource(folder, '../outside')

    def test_manifest_requires_explicit_pose_trust_and_supports_real_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.array([[3,0,0,255]], dtype='<f4').tofile(root/'cloud.bin')
            row = {'observation_id':'a','sequence_id':'s','stamp_s':42.123,'pointcloud':'cloud.bin',
                   'sensor_frame':'sensor','world_frame':'world','timestamp_source':'sensor',
                   'world_from_lidar':np.eye(4).tolist()}
            (root/'frames.jsonl').write_text(json.dumps(row)+'\n')
            observation = next(iter(ManifestSource(root/'frames.jsonl')))
            self.assertEqual(observation.stamp_s, 42.123)
            self.assertEqual(observation.pose_health, 'invalid')

    def test_local_config_merge_is_explicit_and_unknown_fields_fail(self):
        merged = merge(CONFIG, {'detector': {'checkpoint':'/tmp/selected.pth'}})
        self.assertIsNone(CONFIG['detector']['checkpoint'])
        self.assertEqual(merged['detector']['checkpoint'], '/tmp/selected.pth')
        with self.assertRaises(ValueError): merge(CONFIG, {'unknown':True})


class RosTransportTests(unittest.TestCase):
    def cloud(self, bigendian=False):
        fields = [NS(name=n, offset=o, datatype=7, count=1) for n,o in [('intensity',0),('z',4),('x',8),('y',12)]]
        endian = '>' if bigendian else '<'
        # Two padded rows; do not treat row padding as a point.
        data = b''.join(struct.pack(endian+'4f', 255,3,1,2)+b'PAD!' for _ in range(2))
        return NS(fields=fields, width=1, height=2, point_step=16, row_step=20,
                  is_bigendian=bigendian, data=data)

    def test_pointcloud_field_order_padding_and_endianness(self):
        for endian in (False,True):
            np.testing.assert_allclose(decode_pointcloud(self.cloud(endian)), [[1,2,3,255],[1,2,3,255]])

    def test_pointcloud_invalid_layout_and_missing_intensity_fail(self):
        cloud = self.cloud(); cloud.row_step = 1
        with self.assertRaises(ValueError): decode_pointcloud(cloud)
        cloud = self.cloud(); cloud.fields.pop(0)
        with self.assertRaises(ValueError): decode_pointcloud(cloud)

    def test_pose_buffer_uses_only_past_localization_and_explicit_extrinsics(self):
        poses = PoseBuffer()
        poses.add(LocalizedPose(1, 'odom','base',np.eye(4)))
        self.assertIsNone(poses.resolve(.9,.1,'odom','lidar','base')[0])
        self.assertIsNone(poses.resolve(1.01,.1,'odom','lidar','base')[0])
        extrinsic = np.eye(4); extrinsic[0,3] = 1
        matrix, age = poses.resolve(1.01,.1,'odom','lidar','base',extrinsic)
        self.assertAlmostEqual(age,.01); self.assertEqual(matrix[0,3],1)
        self.assertIsNone(poses.resolve(2,.1,'odom','lidar','base',extrinsic)[0])
        self.assertIsNone(poses.resolve(1,.1,'map','lidar','base',extrinsic)[0])

    def test_latest_slot_drops_only_pending_observations(self):
        slot = LatestSlot(); slot.put(1); slot.put(2)
        self.assertEqual(slot.dropped,1); self.assertEqual(slot.take(),2)
        slot.close(); slot.put(3); self.assertIsNone(slot.take(0))

    def test_ros_package_xml_and_launch_are_well_formed(self):
        root = DEFAULT.parents[1]/'ros'
        package = ET.parse(root/'package.xml').getroot()
        self.assertEqual(package.findtext('name'), 'social_nav_people')
        ET.parse(root/'launch/people.launch')
        self.assertTrue((root/'scripts/people_node').stat().st_mode & 0o111)

    def test_ros_publication_preserves_missing_velocity_and_observation_time(self):
        from perception.detection_tracking.ros.node import PeopleNode
        node = PeopleNode.__new__(PeopleNode)
        class Publisher:
            def __init__(self): self.messages = []
            def publish(self, message): self.messages.append(message)
        class People:
            def __init__(self):
                self.header = NS(stamp=None, frame_id='')
                self.detections, self.tracks = [], []
        class TrackState:
            OBSERVED, PREDICTED = 0, 1
        node.People, node.TrackState, node.Detection = People, TrackState, NS
        node.String = lambda **kwargs: NS(**kwargs)
        node.rospy = NS(Time=NS(from_sec=lambda value:value))
        node.publisher, node.json_publisher, node.rgb_publisher = Publisher(), Publisher(), Publisher()
        node.pending = NS(dropped=2)
        result = PerceptionPipeline(Detector(), CONFIG['tracking'], 'test').step(frame())
        node.publish(result, .01, None)
        message = node.publisher.messages[0]
        self.assertEqual(message.header.stamp, 1)
        self.assertEqual(message.header.frame_id, 'dataset_world')
        self.assertFalse(message.tracks[0].has_velocity)
        self.assertEqual(message.dropped_busy, 2)
        self.assertEqual(message.tracks[0].last_observed_stamp, 1)
        self.assertEqual(node.last_result_health, 'valid')


if __name__ == '__main__':
    unittest.main()
