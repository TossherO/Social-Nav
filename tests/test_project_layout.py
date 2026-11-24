"""Project registration, public contracts and documentation; no runtime or devices."""
import contextlib
from dataclasses import fields
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision.vlm.project_config import ROOT, load_config
from decision.vlm.t1.schema import groups_schema
from interfaces import contracts
from scripts import system_status


class ProjectLayoutTests(unittest.TestCase):
    def test_current_document_links_resolve(self):
        for doc in [ROOT / 'AGENTS.md', ROOT / 'README.md', *(ROOT / 'docs').rglob('*.md')]:
            if 'archive' in doc.parts:
                continue
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', doc.read_text()):
                if target.startswith(('https:', 'http:', 'mailto:', '#')):
                    continue
                path = (doc.parent / target.split('#', 1)[0]).resolve()
                if path == ROOT / 'ENVIRONMENT.local.md' or any(path.is_relative_to(ROOT / p) for p in ('outputs', 'data')):
                    continue
                with self.subTest(document=doc.name, target=target):
                    self.assertTrue(path.exists())

    def test_local_vlm_tasks_and_recording_defaults(self):
        config = load_config()
        self.assertEqual(set(config['tasks']), {'groups', 'follow', 'seating'})
        for task in config['tasks'].values():
            if 'prompt' in task:
                self.assertTrue((ROOT / task['prompt']).is_file())
        self.assertTrue(config['model']['frozen'])
        self.assertFalse(config['model']['finetuning'])
        self.assertEqual((config['inference']['period_s'], config['inference']['rate_hz']), (2, .5))

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
        self.assertIn('decision.vlm', report['pending_components'])
        self.assertIn('decision.osnet', report['pending_components'])
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
                   'perception.sensors', 'decision', 'decision.osnet', 'decision.agent', 'navigation',
                   'navigation.planning', 'navigation.control', 'navigation.safety', 'interfaces.contracts']
        code = ('import importlib, sys\n'
                f'for name in {modules!r}: importlib.import_module(name)\n'
                "assert not {'torch', 'sglang', 'vllm', 'spconv'} & set(sys.modules)\n")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')

    def test_draft_contracts_do_not_define_motor_actions_or_change_wire_format(self):
        self.assertFalse(hasattr(contracts, 'CONTRACT_VERSION'))
        self.assertIn('lifecycle_generation', {f.name for f in fields(contracts.TrackIdentity)})
        self.assertIn('clock_domain', {f.name for f in fields(contracts.Header)})
        self.assertIn('track_mapping_revision', {f.name for f in fields(contracts.VLMResultEnvelope)})
        self.assertFalse({'velocity', 'cmd_vel', 'steering'} &
                         {f.name for f in fields(contracts.AgentIntent)})
        self.assertEqual(set(groups_schema([0, 1])['properties']), {'walking','talking','photographing','other'})


if __name__ == '__main__':
    unittest.main()
