"""Display geometry, causal history and seek behavior; no ROS/GPU runtime."""
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
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

    def test_runtime_slots_follow_tracks_and_reuse_retired_ids(self):
        scene = PeopleScene(stable_marker_ids=True)
        row = observation()
        row['tracks'].append(dict(row['tracks'][0], track_id='session:scene:1:17'))

        def slots(specs):
            return {spec['text'].split()[1]: spec['id'] for spec in specs if spec['namespace'] == 'labels'}

        first = slots(scene.update(row))
        self.assertEqual(first, {'7': 0, '17': 1})
        row['tracks'].reverse()
        row['stamp_s'] = 2.
        self.assertEqual(slots(scene.update(row)), first)
        row['tracks'].pop()
        for index in range(20):
            row['stamp_s'] += 1.
            new = dict(row['tracks'][0], track_id='session:scene:1:%d' % (100 + index))
            row['tracks'][1:] = [new]
            self.assertEqual(slots(scene.update(row)), {'17': 1, str(100 + index): 0})
        row['generation'] += 1
        row['stamp_s'] = 1.
        self.assertEqual(slots(scene.update(row)), {'17': 0, '119': 1})

    def test_runtime_marker_recovery_retains_source_time_and_points(self):
        from io import BytesIO
        try:
            from visualization_msgs.msg import Marker
        except ImportError:
            self.skipTest('source perception/scripts/ros1_env.sh to check ROS display messages')
        from perception.visualization.people import MarkerUpdates, marker_array
        scene = PeopleScene(stable_marker_ids=True)
        updates = MarkerUpdates()
        row = observation()
        row['tracks'][0]['velocity_xyz_mps'] = [1., 0., 0.]
        specs = scene.update(row)
        first = updates.message(1, 0., *updates.update(specs))
        first.serialize(BytesIO())
        self.assertEqual(first.markers[0].action, Marker.DELETEALL)
        self.assertIsNone(updates.message(1, .1, *updates.update(specs)))
        original = next(m for m in first.markers if m.ns == 'tracks')
        row['stamp_s'] = 1.1
        next_specs = scene.update(row)
        delta = updates.message(1, .2, *updates.update(next_specs))
        shifted = next(m for m in delta.markers if m.ns == 'tracks')
        self.assertIs(shifted.points, original.points)
        self.assertEqual(original.header.stamp.to_sec(), 1.)
        self.assertAlmostEqual(shifted.header.stamp.to_sec(), 1.1)
        self.assertFalse(any(m.action == Marker.DELETEALL for m in delta.markers))
        row['stamp_s'] = 1.2
        row['tracks'] = []
        removed = updates.message(1, .3, *updates.update(scene.update(row)))
        missing = {(m.ns, m.id) for m in removed.markers if m.action == Marker.DELETE}
        self.assertEqual(missing, {(name, 0) for name in ('tracks', 'labels', 'history', 'velocity')})
        self.assertIsNone(scene.update(observation(1.1)))
        # Recovery remains complete after a transport drops the removal array.
        for epoch, now in ((1, 1.), (2, 1.1)):
            recovered = updates.message(epoch, now)
            self.assertEqual({(m.ns, m.id) for m in recovered.markers if m.action == Marker.DELETE}, missing)
            self.assertEqual({m.ns for m in recovered.markers if m.action == Marker.ADD}, {'detections', 'status'})
            self.assertTrue(all(abs(m.header.stamp.to_sec()-1.2)<1e-9
                                for m in recovered.markers if m.action == Marker.ADD))
        legacy = marker_array(specs, lifetime_s=.5)
        self.assertEqual([m.id for m in legacy.markers[1:]], list(range(len(specs))))
        self.assertTrue(all(m.lifetime.to_sec() == .5 for m in legacy.markers[1:]))
        row['generation'] += 1
        row['stamp_s'] = .5
        reset = updates.message(2, 1.2, *updates.update(scene.update(row), reset=True))
        self.assertEqual(reset.markers[0].action, Marker.DELETEALL)
        self.assertTrue(all(m.header.stamp.to_sec() == .5 for m in reset.markers[1:]))

    def test_runtime_people_publish_the_bound_transform_before_the_marker_array(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('source perception/scripts/ros1_env.sh to check ROS display messages')
        from perception.visualization.people import MarkerUpdates
        from perception.visualization.runtime import DisplayBuffer, RuntimeDisplay
        display = RuntimeDisplay.__new__(RuntimeDisplay)
        display.rospy = rospy
        display.buffer = DisplayBuffer({})
        display.people = PeopleScene(stable_marker_ids=True)
        display._marker_updates = {'people': MarkerUpdates()}
        events = []
        display.publishers = {'people': SimpleNamespace(
            publish=lambda message: events.append(('marker', message)), get_num_connections=lambda: 1)}
        display._publish_map_transform = lambda row: events.append(('transform', row))
        row = dict(stamp_s=1., health=0)
        display._publish('people', (observation(), row))
        display._publish('people', (observation(), row))  # duplicate or delayed render
        self.assertEqual([name for name, _ in events], ['transform', 'marker'])
        self.assertIs(events[0][1], row)
        self.assertTrue(all(m.header.stamp.to_sec() == row['stamp_s'] for m in events[1][1].markers[1:]))

    def test_streaming_markers_attach_periodic_recovery_to_a_measurement(self):
        try:
            from visualization_msgs.msg import Marker
        except ImportError:
            self.skipTest('ROS marker messages unavailable')
        from perception.visualization.people import MarkerUpdates
        scene = PeopleScene(stable_marker_ids=True)
        updates = MarkerUpdates()
        def send(stamp, now):
            return updates.message(1, now, *updates.update(scene.update(observation(stamp))))
        send(1., 0.)
        send(1.9, .9)
        self.assertIsNone(updates.message(1, 1.01))
        current = send(2.1, 1.1)
        self.assertEqual({(m.ns,m.id) for m in current.markers if m.action==Marker.ADD},set(updates.active))
        self.assertIsNotNone(updates.message(2, 1.11))

    def test_runtime_marker_queue_recovers_removals_during_idle_and_shutdown(self):
        try:
            from visualization_msgs.msg import Marker
        except ImportError:
            self.skipTest('source perception/scripts/ros1_env.sh to check ROS display messages')
        from perception.visualization.people import MarkerUpdates
        from perception.visualization.runtime import DisplayBuffer, RuntimeDisplay
        display = RuntimeDisplay.__new__(RuntimeDisplay)
        display.last_view = None
        display.buffer = DisplayBuffer({})
        display._marker_updates = {'people': MarkerUpdates()}
        queue = deque(maxlen=1)
        display.publishers = {'people': SimpleNamespace(publish=queue.append, get_num_connections=lambda: 1)}
        scene = PeopleScene(stable_marker_ids=True)
        specs = scene.update(observation())
        current_time = [0.]
        dt = 'perception.visualization.runtime.time.monotonic'
        with patch(dt, side_effect=lambda: current_time[0]):
            display._publish_marker_update('people', specs)
            seen = {(m.ns, m.id) for m in queue.pop().markers if m.action == Marker.ADD}
            display._publish_marker_update('people', [])
            queue.clear()  # A queued removal is dropped before delivery.
            current_time[0] = 1.1
            display._refresh_marker_snapshots()
            self.assertEqual({(m.ns, m.id, m.action) for m in queue.pop().markers},
                             {(ns, mid, Marker.DELETE) for ns, mid in seen})
            current_time[0] = 1.2
            display._publish_marker_update('people', specs)
            display._publish_marker_update('people', [])
            queue.clear()
            display.buffer.close()
            display.publish_static = False
            display.last_navigation = None
            display._show_basemap = lambda **_: None
            display.error = None
            display._work()
            self.assertIsNone(display.error)
            self.assertEqual({(m.ns, m.id, m.action) for m in queue.pop().markers},
                             {(ns, mid, Marker.DELETE) for ns, mid in seen})

    def test_runtime_reconnect_cannot_restore_people_from_before_hidden_observations(self):
        try:
            from visualization_msgs.msg import Marker
        except ImportError:
            self.skipTest('source perception/scripts/ros1_env.sh to check ROS display messages')
        from perception.visualization.people import MarkerUpdates
        from perception.visualization.runtime import DisplayBuffer, RuntimeDisplay

        display = RuntimeDisplay.__new__(RuntimeDisplay)
        display.people = PeopleScene(stable_marker_ids=True)
        display.buffer = DisplayBuffer({'people_hz': 10.})
        display.last_view = None
        display._marker_updates = {'people': MarkerUpdates()}
        display.navigation_lock = threading.Lock()
        display.navigation_result = True
        display.submit_ms = deque()
        connected = [1]
        published = []
        display.publishers = {'people': SimpleNamespace(
            publish=published.append, get_num_connections=lambda: connected[0])}
        display._publish_map_transform = lambda row: None
        display._connection_epoch = lambda name: connected[0]
        row = dict(stamp_s=1., health=0, session_id='session', generation=1, frame_id='odom')
        display.submit(None, row, observation())
        display._publish(*display.buffer.take(timeout=0))
        self.assertTrue(any(m.ns == 'tracks' for m in published[0].markers))
        connected[0] = 0
        with patch('perception.visualization.runtime.copy.deepcopy', side_effect=AssertionError('hidden copy')):
            display.submit(None, dict(row, stamp_s=2., generation=2, health=1), dict(tracks=[]))
            display.submit(None, dict(row, stamp_s=3., generation=2, health=1), dict(tracks=[]))
        connected[0] = 2
        display._refresh_marker_snapshots()
        self.assertEqual(len(published), 1)
        display._publish(*display.buffer.take(timeout=0))
        self.assertEqual([m.action for m in published[-1].markers], [Marker.DELETEALL])
        self.assertEqual(display.buffer.take(timeout=0), ('idle', None))
        self.assertEqual(display.people.history, {})
        display.submit(None, dict(row, stamp_s=3.1, generation=2), dict(observation(3.1), generation=2))
        display._publish(*display.buffer.take(timeout=0))
        marks = [m for m in published[-1].markers if m.action == Marker.ADD]
        self.assertIn('tracks', {m.ns for m in marks})
        self.assertNotIn('history', {m.ns for m in marks})
        self.assertTrue(all(abs(m.header.stamp.to_sec()-3.1)<1e-9 for m in marks))

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
