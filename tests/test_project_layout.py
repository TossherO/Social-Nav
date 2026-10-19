"""Current documentation/configuration/CLI contracts; zero network or GPU calls."""
import re
import unittest

from decision.vlm import jrdb_t1_inputs as inputs
from decision.vlm import benchmark_vlm as api
from decision.vlm.project_config import ROOT, VLM_ROOT, load_config


class ProjectLayoutTests(unittest.TestCase):
    def test_current_document_links_resolve(self):
        for doc in [ROOT / 'AGENTS.md', ROOT / 'README.md',
                    VLM_ROOT / 'configs/vlm_test_guide.md',
                    *(ROOT / 'docs').rglob('*.md')]:
            if doc.parent == ROOT / 'docs/archive':
                continue  # Historical paths are intentionally preserved in the archive.
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', doc.read_text()):
                if target.startswith(('https:', 'http:', 'mailto:', '#')):
                    continue
                resolved = (doc.parent / target.split('#', 1)[0]).resolve()
                if resolved == ROOT / 'ENVIRONMENT.local.md':
                    continue  # Ignored machine-specific guide may not exist on another machine.
                if resolved.is_relative_to(ROOT / 'outputs') or resolved.is_relative_to(ROOT / 'data'):
                    continue
                with self.subTest(document=doc.name, target=target):
                    self.assertTrue(resolved.exists())

    def test_current_defaults_frozen_fp8_and_two_second_period(self):
        config = load_config()
        self.assertNotIn('config_version', config)
        self.assertEqual(config['inference']['prompt_profile'], 'object')
        self.assertEqual(inputs.PROTOCOL, 'jrdb_t1_object')
        self.assertTrue((VLM_ROOT / 'configs/schemas/social_nav_prompt.json').is_file())
        self.assertEqual(config['model']['served_name'], 'qwen3.8-27b-fp8')
        self.assertTrue(config['model']['frozen'])
        self.assertFalse(config['model']['finetuning'])
        self.assertEqual((config['inference']['period_s'], config['inference']['rate_hz']), (2, .5))
        self.assertEqual(config['inference']['prompt_profile'], inputs.PROFILES[0])
        self.assertEqual(api.arguments(['--dry-run']).rate_hz, .5)
        self.assertEqual(api.arguments(['--dry-run']).constraint, 'json-schema')

    def test_active_client_is_independent_of_gpu_libraries(self):
        for filename in ('benchmark_vlm.py', 'vlm_api_common.py', 'package_vlm_test.py'):
            code = (VLM_ROOT / filename).read_text()
            for banned in ('import torch', 'import vllm', 'import sglang', 'codex_vlm_client',
                           'benchmark_qianwen_jrdb_api'):
                self.assertNotIn(banned, code)
        self.assertTrue((VLM_ROOT / 'configs/vlm_server.example.json').exists())


if __name__ == '__main__':
    unittest.main()
