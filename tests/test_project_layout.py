"""Current documentation/configuration/CLI contracts; zero network or GPU calls."""
import contextlib
import io
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from decision.vlm import jrdb_t1_inputs as inputs
from decision.vlm import benchmark_vlm as api
from decision.vlm.project_config import ROOT, VLM_ROOT, load_config, server_command, main


class ProjectLayoutTests(unittest.TestCase):
    def test_four_current_documents_plus_explicit_legacy_plan(self):
        self.assertEqual({p.name for p in (ROOT / 'docs').glob('*.md')}, {
            '当前整体方案.md', '旧版_基于VLM的长程社交导航.md',
            '纯公开数据可行性验证方案.md', 'JRDB数据与测试实现.md', '项目框架.md'})

    def test_current_document_links_resolve(self):
        module_docs = [ROOT / name / 'README.md'
                       for name in ('perception', 'decision', 'navigation', 'interfaces')]
        for doc in [ROOT / 'AGENTS.md', ROOT / 'README.md', *module_docs,
                    *(ROOT / 'docs').glob('*.md')]:
            if doc.name == '旧版_基于VLM的长程社交导航.md':
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

    def test_module_readmes_and_portable_guide_use_markdown(self):
        for name in ('perception', 'decision', 'navigation', 'interfaces'):
            with self.subTest(module=name):
                self.assertTrue((ROOT / name / 'README.md').is_file())
                self.assertFalse((ROOT / name / 'README.txt').exists())
        self.assertTrue((VLM_ROOT / 'configs/vlm_test_guide.md').is_file())
        self.assertFalse((VLM_ROOT / 'configs/vlm_test_README.txt').exists())

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

    def test_system_configuration_has_no_project_release_number(self):
        import json
        config = json.loads((ROOT / 'configs/system.json').read_text(encoding='utf-8'))
        self.assertNotIn('system_config_version', config)
        self.assertNotIn('version', config['interfaces'])

    def test_manual_server_command_no_hopper_only_kernel_or_automatic_mtp(self):
        cmd = server_command()
        self.assertEqual(cmd[:3], ['python', '-m', 'sglang.launch_server'])
        self.assertEqual(cmd[cmd.index('--host') + 1], '127.0.0.1')
        self.assertEqual(cmd[cmd.index('--grammar-backend') + 1], 'xgrammar')
        for flag in ('--attention-backend', '--fp8-gemm-backend', '--speculative-algorithm',
                     '--disable-radix-cache', '--api-key'):
            self.assertNotIn(flag, cmd)
        self.assertEqual(cmd[cmd.index('--max-running-requests') + 1], '1')
        custom = server_command(model_path='/remote/weights', port=4567, python='/isolated/python')
        self.assertEqual(custom[0], '/isolated/python')
        self.assertEqual(custom[custom.index('--model-path') + 1], '/remote/weights')
        self.assertEqual(custom[custom.index('--port') + 1], '4567')

    def test_server_cli_only_prints(self):
        with patch('subprocess.Popen') as process, contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--port', '4567'])
        process.assert_not_called()
        self.assertIn('sglang.launch_server', output.getvalue())

    def test_active_client_is_independent_of_gpu_libraries(self):
        for filename in ('benchmark_vlm.py', 'vlm_api_common.py', 'package_vlm_test.py'):
            code = (VLM_ROOT / filename).read_text()
            for banned in ('import torch', 'import vllm', 'import sglang', 'codex_vlm_client',
                           'benchmark_qianwen_jrdb_api'):
                self.assertNotIn(banned, code)
        self.assertTrue((VLM_ROOT / 'configs/vlm_server.example.json').exists())


if __name__ == '__main__':
    unittest.main()
