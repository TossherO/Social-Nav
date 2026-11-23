import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from navigation.planning.footprint import VehicleGeometry
from perception.environment.layers import GridSnapshot
from perception.environment.scene import assemble_scene
from scripts.field_video.__main__ import RGB_TOPIC, recorded_frames, stationary_intervals, video_intervals
from scripts.field_video.canvas import _SemanticMap
from scripts.field_video.grid_archive import SceneGrids
from scripts.field_video.local_view import _Map, _Perspective, _to_world


def poses(end):
    return [dict(t=t, x=0., y=0., yaw=0., floor=1, segment=1) for t in range(end + 1)]


class FieldVideoEditingTests(unittest.TestCase):
    def test_compact_preserves_decision_cancel_and_observation_time(self):
        data = dict(poses=poses(40), events=[dict(t=12., kind='decision'), dict(t=24., kind='cancel')],
                    evidence=[dict(t=8.)])
        args = SimpleNamespace(mode='compact', fps=12, wait_min=8., wait_context=2.,
                               cut=[(3., 38.)], keep=[(33., 35.)])
        self.assertEqual(video_intervals(data, 0., 40., args),
                         [(0., 2.), (5.5, 14.5), (21.5, 26.5), (33., 35.), (38., 40.)])

    def test_localization_gaps_and_map_changes_break_stationary_evidence(self):
        samples = poses(12)
        self.assertEqual(stationary_intervals(samples, 8., 2.), [(2., 10.)])
        self.assertEqual(stationary_intervals(samples[:6] + samples[7:], 8., 2.), [])
        for pose in samples[6:]:
            pose['segment'] = 2
        self.assertEqual(stationary_intervals(samples, 8., 2.), [])

    def test_rotation_prevents_wait_clip(self):
        samples = poses(12)
        for pose in samples:
            pose['yaw'] = math.radians(pose['t'] * 3)
        self.assertEqual(stationary_intervals(samples, 8., 2.), [])

    def test_full_preserves_footage_and_bookend_events(self):
        data = dict(poses=[], events=[dict(t=-6.), dict(t=11.)], evidence=[])
        args = SimpleNamespace(mode='full', fps=12)
        self.assertEqual(video_intervals(data, 0., 10., args), [(-6., -3.5), (0., 10.), (11., 13.5)])

    def test_tail_holds_final_measurement_with_separate_event_time(self):
        args = SimpleNamespace(mode='full', fps=2)
        rows = [(100., 'first'), (101., 'previous'), (101.2, 'last')]
        data = dict(poses=[], events=[dict(t=97.), dict(t=101.3)], evidence=[])
        with patch('scripts.field_video.__main__.measurement_rows', return_value=iter(rows)), \
                patch('scripts.field_video.__main__.last_measurement_row', return_value=rows[-1]), \
                patch('scripts.field_video.__main__.decode_rgb', side_effect=lambda rgb: rgb), \
                patch('scripts.field_video.canvas.Canvas') as canvas:
            draw = canvas.return_value.frame
            list(recorded_frames(None, False, data, args, []))

        self.assertEqual(draw.call_args_list[0].args, (97., None))
        in_recording = [call for call in draw.call_args_list if 100. <= call.args[0] <= 101.2]
        self.assertEqual(in_recording[-1].args, (101., 'previous'))
        self.assertIsNone(in_recording[-1].kwargs['frozen_at'])
        tail = [call for call in draw.call_args_list if call.args[0] > 101.2]
        self.assertTrue(tail)
        self.assertTrue(all(call.args[1] == 'last' and call.kwargs['frozen_at'] == 101.2 for call in tail))

    def test_scan_is_drawn_with_localization_only_after_its_measurement(self):
        rgb = [(100., 'first'), (100.5, 'second'), (101., 'third'), (101.5, 'last')]
        scan = [(100.3, 'scan')]
        data = dict(poses=[], events=[], evidence=[])
        args = SimpleNamespace(mode='full', fps=2)
        with patch('scripts.field_video.__main__.measurement_rows',
                   side_effect=lambda _, topic: iter(rgb if topic == RGB_TOPIC else scan)), \
                patch('scripts.field_video.__main__.last_measurement_row', return_value=rgb[-1]), \
                patch('scripts.field_video.__main__.decode_rgb', side_effect=lambda image: image), \
                patch('scripts.field_video.__main__.decode_scan', return_value=([(1., 2., 3.)], 'rslidar')), \
                patch('scripts.field_video.canvas.Canvas') as canvas:
            canvas.return_value.has_localization.return_value = True
            list(recorded_frames(None, True, data, args, []))
        frames = [call.kwargs for call in canvas.return_value.frame.call_args_list]
        self.assertEqual([frame['scan_stamp'] for frame in frames], [None, 100.3, None])
        self.assertEqual(frames[1]['scan_xyz'], [(1., 2., 3.)])


class FieldVideoGeometryTests(unittest.TestCase):
    def test_semantic_crop_shows_north_up_occupancy_and_unfilled_regions(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'map.pgm'
            Image.fromarray(np.array([[0, 254], [205, 0]], dtype=np.uint8)).save(path)
            occupancy = _Map(dict(image=path, origin=(-1., -1., 0.), resolution=1.))
        annotations = {'regions': [{'polygon': [(-2, -2), (2, -2), (2, 2), (-2, 2)], 'name': '区域'}]}
        crop = _SemanticMap('floor1', [], annotations, [], occupancy)

        for x, y, rgb in ((-.5, .5, (58, 63, 71)), (.5, .5, (246, 246, 239)),
                          (-.5, -.5, (169, 177, 186)), (.5, -.5, (58, 63, 71)),
                          (1.5, -.5, (169, 177, 186))):
            pixel = tuple(map(round, crop.project(x, y)))
            self.assertEqual(crop.background.getpixel(pixel), rgb)

    def test_high_view_retains_vehicle_and_forward_ground(self):
        perspective = _Perspective((1200, 1000))
        angle = math.degrees(math.asin(-perspective.axes[2, 2]))
        self.assertGreaterEqual(angle, 34.)
        self.assertLessEqual(angle, 35.)
        self.assertGreater(perspective.position[2], 6.)
        vehicle, road = perspective.ground(np.array(((0., 0.), (15., 0.))))
        self.assertTrue(0 < road[1] < vehicle[1] < 1000)


class FieldVideoGridArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.bag = Path(directory.name) / 'capture' / 'navigation.bag'
        self.planning = Path(directory.name) / 'agent' / 'session' / 'run' / 'planning'
        self.planning.mkdir(parents=True)
        self.binding = dict(valid=True, map_id='first_floor', map_frame='map', map_revision=3,
                            association_session='prior', association_generation=2)
        self.geometry = VehicleGeometry([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]],
                                        np.eye(4), source='test')
        self.config = dict(max_geometry_age_s=.5, max_association_age_s=.5)
        self.pose = dict(x=10., y=20., yaw=math.pi / 2, local_identity=['slam', 1, 'odom'],
                         map_binding=self.binding)
        self.points = np.array([[[.5, .5]]])

    def save(self, stamp, cycle_stamp, *, measured=None, localized=True):
        map_from_local = [[0., -1., 0., 10.], [1., 0., 0., 20.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        association = dict(self.binding, status='PRIOR_ASSOCIATED', stamp_s=stamp,
                           measurement_stamp_s=stamp if measured is None else measured,
                           local_session_id='slam', local_generation=1, local_frame='odom',
                           map_from_local=map_from_local)
        metadata = dict(stamp_s=stamp, frame_id='odom', session_id='slam', generation=1,
                        snapshot_id=f'grid:{stamp}', resolution=1., origin_xy=[0., 0.], shape=[2, 3],
                        geometry_status='OBSERVED' if localized else 'LOCALIZATION_UNAVAILABLE',
                        world_from_lidar=np.eye(4).tolist(), association=association)
        grid = GridSnapshot(metadata, dict(occupancy=np.array([[100, 0, 0], [-1, 100, 0]], dtype=np.int8)))
        scene = assemble_scene(grid, clock_domain='runtime_live', settings=self.config)
        binding = scene.record['input_fingerprint']
        scene.save(self.planning / 'scenes' / binding)
        self.cycle(cycle_stamp, binding, binding)
        return binding

    def cycle(self, stamp, binding, saved=None):
        with (self.planning / 'cycles.jsonl').open('a', encoding='utf-8') as cycles:
            print(json.dumps(dict(stamp_s=stamp, scene_binding=binding, saved_scene=saved)), file=cycles)

    def sample(self, archive, t, *, pose=None, not_before=9.):
        pose = self.pose if pose is None else pose
        return archive.sample(t, pose, 'first_floor', _to_world(self.points, pose), not_before=not_before)

    def test_archives_cover_normal_recording_gap_and_follow_vehicle_in_map_coordinates(self):
        binding = self.save(10., 10.2)
        self.cycle(10.6, binding)  # Ordinary cycles with no capture do not end the snapshot.
        self.save(10.6, 10.82)
        archive = SceneGrids(self.bag, self.geometry, self.config)

        self.assertIsNone(self.sample(archive, 10.19))
        for frame in range(9):
            t = 10.2 + frame / 12
            self.assertEqual(self.sample(archive, t).item(), 100)
        moved = dict(self.pose, y=21.)
        self.assertEqual(self.sample(archive, 10.75, pose=moved).item(), 0)
        self.assertIsNone(self.sample(archive, 11.61))

    def test_archive_stops_at_binding_changes_and_new_invalid_evidence(self):
        self.save(10., 10.2)
        self.save(10.5, 10.7, localized=False)
        self.save(10.6, 10.9, measured=10.3)  # Valid at acquisition, stale at the recording cycle.
        self.save(10.9, 11.0)
        self.cycle(11.1, None)
        archive = SceneGrids(self.bag, self.geometry, self.config)

        self.assertIsNotNone(self.sample(archive, 10.65))
        for field in ('map_id', 'map_frame', 'map_revision', 'association_session', 'association_generation'):
            with self.subTest(field=field):
                changed = dict(self.binding, **{field: 'changed'})
                self.assertIsNone(self.sample(archive, 10.65, pose=dict(self.pose, map_binding=changed)))
        self.assertIsNone(self.sample(archive, 10.65, pose=dict(self.pose, local_identity=['slam', 2, 'odom'])))
        self.assertIsNone(self.sample(archive, 10.65, not_before=10.1))
        self.assertIsNone(self.sample(archive, 10.75))
        self.assertIsNone(self.sample(archive, 10.95))
        self.assertIsNotNone(self.sample(archive, 11.05))
        self.assertIsNone(self.sample(archive, 11.15))
