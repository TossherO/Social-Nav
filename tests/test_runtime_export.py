"""Archives retain host evidence from the sealed device observation."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from perception.environment.free_memory import ConfirmedFreeMemory, remembered_free
from perception.environment.layers import GridSnapshot, load_config
from perception.environment.navigation import safety_capsule
from perception.environment.scene import SceneInputs, assemble_scene
from perception.environment.support import NavigationSupport
from runtime.export import SceneExporter, materialize_scene
from tests.test_environment_navigation import observation, prior_asset
from tests.test_environment_scene import person, rgb
from tests.test_environment_support import points


class SceneExportTests(unittest.TestCase):
    def fixture(self):
        config = load_config()
        asset = prior_asset()
        raw = observation()
        fused = NavigationSupport(config).apply(raw, asset, observation=points(raw))
        capsule = safety_capsule(fused, extra_fields=('last_observed', 'near_field_free', 'coverage', 'ground_support'))
        host = ConfirmedFreeMemory().update(capsule, asset)
        hits = np.zeros((60, 60), bool)
        hits[8, 12] = True
        host = GridSnapshot(host.metadata, dict(host.layers, current_obstacle_hits=hits))
        for value in (*host.layers.values(), *fused.layers.values()):
            value.flags.writeable = False
        people = dict(person(), localization_binding=dict(session_id='test', generation=0, frame_id='odom'))
        scene = assemble_scene(host, clock_domain='synthetic', settings=config['scene'],
                               stamp_ns=10000000000, people=people)
        missing = {name: array for name, array in fused.layers.items() if name not in host.layers}
        device = SimpleNamespace(metadata=dict(fused.metadata), owner='source',
            _layers=dict.fromkeys(fused.layers, SimpleNamespace(device='cuda:0')),
            materialize=Mock(return_value=GridSnapshot(fused.metadata, missing)))
        return scene, device, asset, config['scene'], list(missing)

    def assert_source_evidence(self, archived, scene, asset):
        self.assertEqual(archived.record['stamp_ns'], 10000000000)
        for field in ('as_of', 'clock_domain', 'observation_id', 'people', 'prior'):
            self.assertEqual(archived.record[field], scene.record[field])
        np.testing.assert_array_equal(archived.geometry.layers['current_obstacle_hits'],
                                      scene.geometry.layers['current_obstacle_hits'])
        np.testing.assert_array_equal(remembered_free(archived, asset=asset), remembered_free(scene, asset=asset))
        self.assertIn('ground_z', archived.geometry.layers)
        self.assertTrue(all(not value.flags.writeable for value in archived.geometry.layers.values()))

    def test_final_review_keeps_bound_layers_and_refuses_a_different_frame(self):
        scene, device, asset, settings, missing = self.fixture()
        scene = scene.with_rgb(rgb(), settings)
        final = materialize_scene(device, scene, consumer='final_review').verify()
        device.materialize.assert_called_once_with(consumer='final_review', fields=missing, metrics=None)
        self.assertIs(final.rgb, scene.rgb)
        self.assert_source_evidence(final, scene, asset)
        self.assertIs(materialize_scene(device, final, consumer='final_review'), final)
        device.metadata['stamp_s'] = 10.1
        with self.assertRaisesRegex(ValueError, 'Scene export/device observation mismatch'):
            materialize_scene(device, scene, consumer='final_review')
        self.assertEqual(device.materialize.call_count, 1)

    def test_explicit_export_persists_host_evidence_and_the_selected_rgb(self):
        scene, device, asset, settings, missing = self.fixture()
        image = dict(rgb(), stamp_ns=9950000000)
        with tempfile.TemporaryDirectory() as directory, \
                patch('torch.cuda.Stream'), patch('torch.cuda.stream', return_value=nullcontext()):
            exporter = SceneExporter(directory, settings)
            self.assertTrue(exporter.submit(device, scene, rgb=image, name='frame'))
            self.assertEqual(exporter.close(), dict(written=1, dropped=0, complete=True))
            restored = SceneInputs.load(Path(directory)/'frame')
        device.materialize.assert_called_once_with(consumer='explicit_scene_export', fields=missing, metrics=None)
        self.assert_source_evidence(restored, scene, asset)
        self.assertEqual(restored.record['rgb']['stamp_ns'], 9950000000)
        self.assertEqual(restored.record['health'], 'valid')
        np.testing.assert_array_equal(restored.rgb, image['pixels'])
