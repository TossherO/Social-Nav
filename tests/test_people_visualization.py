"""Display geometry, causal history and seek behavior; no ROS/GPU runtime."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from perception.detection_tracking.inference.geometry import corners_xyz
from perception.detection_tracking.inference.config import ROOT
from perception.visualization.people import PeopleScene, RecordedPeople, place_specs
from perception.visualization.scans import ObservationOrder, ScanCache, configure_rviz, scan_geometry


def observation(stamp=1.0):
    box = [3, 0, 0, .6, .5, 1.7, 0, .9]
    track = dict(track_id='session:scene:1:7', lifecycle_generation=1, box=box.copy(),
                 state='observed', confirmed=True, velocity_xyz_mps=None)
    return dict(stamp_s=stamp, session_id='session', sequence_id='scene', generation=1,
                frame_id='odom', detection_frame_id='rslidar', health='valid',
                world_from_lidar=np.eye(4).tolist(), detections=[box], tracks=[track],
                pose_source='stationary_assumption')


class PeopleVisualizationTests(unittest.TestCase):
    def test_late_results_do_not_replace_the_scene_or_retired_identity(self):
        scene = PeopleScene()
        scene.update(observation(2))
        self.assertIsNone(scene.update(observation(1.9)))
        self.assertEqual(scene.last_stamp, 2)
        order = ObservationOrder()
        self.assertTrue(order.accept(observation(2)))
        self.assertFalse(order.accept(observation(2)))
        row = observation(2.1)
        row['generation'] = 2
        self.assertTrue(order.accept(row))
        self.assertFalse(order.accept(observation(2.2)))

    def test_display_cloud_is_same_observation_and_pose_as_detection_and_clears_on_failure(self):
        stamp = SimpleNamespace(to_sec=lambda: 1.)
        cloud = SimpleNamespace(header=SimpleNamespace(stamp=stamp, frame_id='rslidar'),
            height=1, width=1, point_step=12, row_step=12, is_bigendian=False,
            fields=[SimpleNamespace(name=name, offset=i*4, datatype=7, count=1)
                    for i, name in enumerate(('x', 'y', 'z'))],
            data=np.array([[3, 0, 0]], dtype='<f4').tobytes())
        cache = ScanCache()
        cache.add(cloud)
        self.assertIs(cache.at(1., 'rslidar'), cloud)
        self.assertIsNone(cache.at(1.1, 'rslidar'))
        self.assertIsNone(cache.at(1., 'unrelated_lidar'))
        row = observation()
        row['world_from_lidar'][0][3] = 5
        frame, points = scan_geometry(cloud, row)
        self.assertEqual(frame, 'odom')
        detection = next(s for s in PeopleScene().update(row) if s['namespace'] == 'detections')
        np.testing.assert_allclose(points[0], detection['points'].mean(axis=0), atol=1e-7)
        row.update(health='degraded', world_from_lidar=None)
        self.assertEqual(len(scan_geometry(cloud, row)[1]), 0)
        np.testing.assert_allclose(scan_geometry(cloud, row, sensor_only=True)[1], [[3, 0, 0]])
        self.assertEqual(len(scan_geometry(None, row)[1]), 0)
        row['stamp_s'] = 1.1
        with self.assertRaisesRegex(ValueError, 'not bound'):
            scan_geometry(cloud, row)

    def test_rviz_uses_configured_output_topics(self):
        configuration = {'Visualization Manager': {'Displays': [
            {'Marker Topic': '/perception/people/markers'}, {'Image Topic': '/perception/people/rgb'}],
            'Global Options': {'Fixed Frame': 'odom'}}}
        configure_rviz(configuration, {'output_topic': '/test/people'}, 'local_world')
        manager = configuration['Visualization Manager']
        self.assertEqual(manager['Displays'][0]['Marker Topic'], '/test/people/markers')
        self.assertEqual(manager['Displays'][1]['Image Topic'], '/test/people/rgb')
        self.assertEqual(manager['Global Options']['Fixed Frame'], 'local_world')

    def test_launch_root_is_independent_of_catkin_share_location_and_display_is_optional(self):
        directory = ROOT/'perception/detection_tracking/ros/launch'
        for name in ('people.launch', 'visualization.launch'):
            launch = ET.parse(directory/name).getroot()
            value = launch.find("arg[@name='project_root']").attrib['default']
            self.assertEqual(Path(value.replace('$(dirname)', str(directory))).resolve(), ROOT)
        launch = ET.parse(directory/'people.launch').getroot()
        self.assertEqual(launch.find("arg[@name='rviz']").attrib['default'], 'false')
        self.assertEqual(launch.find('include').attrib['if'], '$(arg visualize)')

    def test_raw_detection_uses_full_measured_transform_and_unknown_velocity_has_no_arrow(self):
        row = observation()
        transform = np.eye(4)
        transform[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        transform[:3, 3] = [1, 2, 3]
        row['world_from_lidar'] = transform.tolist()
        specs = PeopleScene().update(row)
        detection = next(s for s in specs if s['namespace'] == 'detections')
        np.testing.assert_allclose(detection['points'].mean(axis=0), [1, 5, 3])
        expected = corners_xyz(row['detections'][0]) @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(detection['points'].min(axis=0), expected.min(axis=0))
        self.assertEqual(detection['frame_id'], 'odom')
        self.assertTrue(all(s['stamp_s'] == 1 for s in specs))
        self.assertFalse(any(s['namespace'] == 'velocity' for s in specs))

    def test_prediction_degradation_and_identity_reset_do_not_extend_observed_trails(self):
        scene = PeopleScene()
        scene.update(observation(1))
        row = observation(1.1)
        row['tracks'][0]['velocity_xyz_mps'] = [1, 0, 0]
        specs = scene.update(row)
        self.assertTrue(any(s['namespace'] == 'history' for s in specs))
        self.assertTrue(any(s['namespace'] == 'velocity' for s in specs))
        row = observation(1.2)
        row['tracks'][0]['state'] = 'predicted'
        specs = scene.update(row)
        self.assertFalse(any(s['namespace'] == 'history' for s in specs))
        self.assertTrue(any('PRED' in s['text'] for s in specs))
        row = observation(1.3)
        row.update(health='degraded', world_from_lidar=None, frame_id='rslidar')
        specs = scene.update(row)
        self.assertFalse(any(s['namespace'] in ('tracks', 'history', 'velocity') for s in specs))
        self.assertTrue(all(s['frame_id'] == 'rslidar' for s in specs))
        self.assertEqual(scene.history, {})
        scene.update(observation(1.4))
        row = observation(1.5)
        row['generation'] = 2
        self.assertFalse(any(s['namespace'] == 'history' for s in scene.update(row)))

    def test_history_is_bounded_and_seek_never_uses_future_or_another_scan(self):
        config = dict(history_s=.3, max_history_points=3, stale_s=.5)
        rows = [observation(i/10) for i in range(1, 11)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'people.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            playback = RecordedPeople(path, config)
            self.assertEqual(playback.at(.05), [])
            self.assertEqual(playback.at(.25), [])
            latest = playback.at(1)
            self.assertLessEqual(len(next(s for s in latest if s['namespace'] == 'history')['points']), 3)
            earlier = playback.at(.4)
            self.assertTrue(all(s['stamp_s'] == .4 for s in earlier))
            rows[-1]['stamp_s'] = .1
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'reversed'):
                RecordedPeople(path, config)

    def test_map_view_uses_recorded_transform_without_double_translation_or_sensor_fallback(self):
        specs = PeopleScene().update(observation())
        transform = np.eye(4)
        transform[:3, 3] = [10, 20, 30]
        placed = place_specs(specs, 'odom', 'map', transform)
        box = next(s for s in placed if s['namespace'] == 'detections')
        np.testing.assert_allclose(box['points'].mean(axis=0), [13, 20, 30])
        np.testing.assert_allclose(box['position'], [0, 0, 0])
        self.assertEqual(box['frame_id'], 'map')
        original = next(s for s in specs if s['namespace'] == 'detections')
        np.testing.assert_allclose(original['points'].mean(axis=0), [3, 0, 0], atol=1e-12)
        self.assertEqual(place_specs(specs, 'another_frame', 'map', transform), [])

    def test_bad_box_and_nonrigid_pose_are_rejected(self):
        row = observation()
        row['world_from_lidar'][0][0] = 2
        with self.assertRaises(ValueError):
            PeopleScene().update(row)
        row = observation()
        row['detections'][0][3] = -1
        with self.assertRaises(ValueError):
            PeopleScene().update(row)


if __name__ == '__main__':
    unittest.main()
