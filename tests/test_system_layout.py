"""System registration and cross-module behavior; synthetic assets, no GPU or network."""
import contextlib
from dataclasses import fields
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision.vlm import benchmark_vlm as api
from decision.vlm import package_vlm_test as pack
from decision.vlm.jrdb_t1_inputs import rel_schema
from decision.vlm.project_config import ROOT, VLM_ROOT
from interfaces import contracts
from perception.mapping_localization.__main__ import finalize_artifacts
from scripts import system_status


class SystemLayoutTests(unittest.TestCase):
    def test_three_registered_domains_exist_but_are_not_online(self):
        report = system_status.inspect_layout()
        self.assertTrue(report['layout_valid'])
        self.assertFalse(report['online_system_ready'])
        self.assertEqual((report['processes_started'], report['network_requests']), (0, 0))
        self.assertEqual({c['component'].split('.')[0] for c in report['components']},
                         {'perception', 'decision', 'navigation'})
        self.assertNotIn('perception.detection_tracking', report['pending_components'])
        self.assertIn('perception.mapping_localization', {c['component'] for c in report['components']})
        self.assertNotIn('perception.mapping_localization', report['pending_components'])
        self.assertIn('decision.agent', report['pending_components'])
        self.assertIn('navigation.safety', report['pending_components'])

    def test_require_online_fails_without_starting_any_process(self):
        with patch('subprocess.Popen') as process, contextlib.redirect_stdout(io.StringIO()) as output:
            result = system_status.main(['--require-online'])
        process.assert_not_called()
        self.assertEqual(result, 1)
        self.assertFalse(json.loads(output.getvalue())['online_system_ready'])

    def layout_fixture(self, root):
        root = Path(root)
        config = json.loads(system_status.CONFIG_PATH.read_text())
        (root / 'configs').mkdir()
        for entries in config['modules'].values():
            for entry in entries.values():
                path = root / entry['path']
                path.mkdir(parents=True)
                (path / '__init__.py').write_text('')
        (root / 'interfaces').mkdir()
        (root / 'interfaces/contracts.py').write_text('')
        path = root / 'configs/system.json'
        path.write_text(json.dumps(config))
        return config, path

    def test_missing_component_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            self.layout_fixture(folder)
            (Path(folder) / 'navigation/control/__init__.py').unlink()
            report = system_status.inspect_layout(folder)
        self.assertFalse(report['layout_valid'])
        self.assertIn('navigation.control', report['missing_layout_paths'])

    def test_registry_rejects_paths_outside_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            config, path = self.layout_fixture(folder)
            config['modules']['decision']['agent']['path'] = '../outside'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'path/status'):
                system_status.inspect_layout(folder)

    def test_scaffold_packages_import_without_gpu_or_runtime(self):
        modules = ['perception', 'perception.detection_tracking.training', 'perception.detection_tracking.inference',
                   'perception.detection_tracking.ros', 'perception.mapping_localization',
                   'perception.sensors', 'decision', 'decision.agent', 'navigation',
                   'navigation.planning', 'navigation.control', 'navigation.safety', 'interfaces.contracts']
        code = ('import importlib, sys\n'
                f'for name in {modules!r}: importlib.import_module(name)\n'
                "assert not {'torch', 'sglang', 'vllm', 'spconv'} & set(sys.modules)\n")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')

    def test_canonical_vlm_paths_and_private_config_location(self):
        self.assertEqual(VLM_ROOT, ROOT / 'decision/vlm')
        self.assertEqual(pack.ROOT, ROOT)
        self.assertEqual(api.arguments(['--dry-run']).bundle, ROOT / 'data/datasets/jrdb_t1_dev')
        self.assertEqual(api.arguments(['--dry-run']).config, ROOT / 'configs/vlm_server.local.json')
        self.assertTrue((VLM_ROOT / 'configs/prompts/jrdb_t1_rel.txt').is_file())

    def test_replay_retention_preserves_maps_and_failure_diagnostics(self):
        for success, keep_details in ((True, False), (True, True), (False, False)):
            with self.subTest(success=success, keep_details=keep_details), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                (output / 'map').mkdir()
                map_file = output / 'map/GlobalMap.pcd'
                map_file.write_bytes(b'SYNTHETIC_MAP_NOT_LOADED')
                (output / 'localization.jsonl').write_text('{"health": "valid"}\n')
                (output / 'people.jsonl').write_text('{"health": "valid", "tracks": []}\n')
                (output / 'algorithm.log').write_text('Algorithm diagnostics\n')
                (output / 'master.log').write_text('Master startup\n')
                (output / 'notes.txt').write_text('Unrelated result\n')
                settings = {'success': success, 'counts': {'localization': 1, 'valid': 1}}
                finalize_artifacts(output, settings, keep_details=keep_details)
                record = json.loads((output / 'run.json').read_text())
                self.assertEqual(record['counts'], {'localization': 1, 'valid': 1})
                self.assertEqual(map_file.read_bytes(), b'SYNTHETIC_MAP_NOT_LOADED')
                self.assertIn('map/', record['retained_artifacts'])
                self.assertIn('run.json', record['retained_artifacts'])
                self.assertEqual((output / 'notes.txt').read_text(), 'Unrelated result\n')
                self.assertTrue((output / 'localization.jsonl').exists())
                self.assertTrue((output / 'people.jsonl').exists())
                self.assertEqual((output / 'algorithm.log').exists(), keep_details or not success)
                self.assertEqual((output / 'master.log').exists(), not success)

    def test_draft_contracts_do_not_define_motor_actions_or_change_wire_format(self):
        self.assertFalse(hasattr(contracts, 'CONTRACT_VERSION'))
        self.assertIn('lifecycle_generation', {f.name for f in fields(contracts.TrackIdentity)})
        self.assertIn('clock_domain', {f.name for f in fields(contracts.Header)})
        self.assertIn('track_mapping_revision', {f.name for f in fields(contracts.VLMResultEnvelope)})
        self.assertFalse({'velocity', 'cmd_vel', 'steering'} &
                         {f.name for f in fields(contracts.AgentIntent)})
        self.assertEqual(set(rel_schema([0, 1])['properties']), {'g', 'u'})

    def test_new_portable_package_runs_outside_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            case_dir = source / 'cases/synthetic'
            case_dir.mkdir(parents=True)
            (source / 'system_prompt.txt').write_text('Return typed REL JSON')
            images = []
            for index in range(3):
                name = f'cases/synthetic/image_{index}.png'
                (source / name).write_bytes(b'SYNTHETIC_NOT_FOR_INFERENCE')
                images.append(name)
            model = {'wire_protocol': 'object', 'images': images,
                     'fields': {'v': [0, 1], 'dt': [-1, -.5, 0], 'ts': 'synthetic',
                                'e': [-2, 12, -6, 6], 'o': [], 'd': []},
                     'user_prompt': 'Synthetic observations', 'response_schema': rel_schema([0, 1])}
            input_file = 'cases/synthetic/model_input.json'
            (source / input_file).write_text(json.dumps(model))
            manifest = {'profile': 'synthetic', 'prompt_profile': 'object',
                        'wire_protocol': 'object', 'credentials_included': False,
                        'native_social_labels_included': False,
                        'cases': [{'case_id': 'synthetic', 'scene': 'fixture', 'input_file': input_file}],
                        'image_bytes_total': sum((source / p).stat().st_size for p in images),
                        'files': {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source.rglob('*') if p.is_file()}}
            (source / 'manifest.json').write_text(json.dumps(manifest))
            out = root / 'portable'
            report = pack.package(out, source)
            self.assertEqual((report['cases'], report['inference_requests_sent']), (1, 0))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='')
            result = subprocess.run([sys.executable, str(out / 'benchmark.py'), '--dry-run'],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((out / 'results').exists())
            self.assertEqual(json.loads((out / 'server_config.json').read_text())['api_key'], '')
            exported = out / 'benchmark_defaults.json'
            exported.write_text('{}')
            result = subprocess.run([sys.executable, str(out / 'benchmark.py'), '--dry-run'],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            self.assertFalse((out / 'results').exists())


if __name__ == '__main__':
    unittest.main()
