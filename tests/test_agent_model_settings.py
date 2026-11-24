import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision.agent.core.model_settings import DEFAULT_MODEL, GPT_MODEL, credentials_for
from decision.agent.ui_data import event_summary


class AgentModelSettingsTests(unittest.TestCase):
    def test_keys_and_endpoints_are_isolated(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            folder = Path(directory)
            config = folder / 'agent.json'
            config.write_text(json.dumps({'base_url': 'https://example.invalid/v1', 'api_key': 'qwen-fixture',
                                          'openai_api_key_file': 'openai.txt'}))
            self.assertEqual(credentials_for(DEFAULT_MODEL, config),
                             ('https://example.invalid/v1/chat/completions', 'qwen-fixture'))
            token = 'sk-' + 'x' * 24
            (folder / 'openai.txt').write_text(f'OpenAI\n{token}\n')
            self.assertEqual(credentials_for(GPT_MODEL, config),
                             ('https://api.openai.com/v1/responses', token))
            (folder / 'openai.txt').write_text(f'{token}\n{token}\n')
            with self.assertRaises(ValueError) as failure:
                credentials_for(GPT_MODEL, config)
            self.assertNotIn(token, str(failure.exception))
            os.environ['OPENAI_API_KEY'] = 'env-fixture'
            self.assertEqual(credentials_for(GPT_MODEL, folder / 'absent.json')[1], 'env-fixture')
            self.assertEqual(credentials_for(DEFAULT_MODEL, config)[1], 'qwen-fixture')

    def test_request_summary_displays_actual_selection_and_old_events(self):
        self.assertEqual(event_summary(dict(kind='api_request', value=dict(
            model=GPT_MODEL, thinking_mode='high', decision_phase='monitor'))),
            '模型请求 · gpt-6.1-terra · 监控复核 · high')
        self.assertEqual(event_summary(dict(kind='api_request', value=dict(model=DEFAULT_MODEL))),
                         '模型请求 · qwen3.8-flash')

    def test_configured_gpt_endpoint_uses_its_own_inline_key(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}, clear=True):
            config = Path(directory) / 'agent.json'
            values = dict(base_url='https://qwen.invalid/v1', api_key='qwen-key',
                          openai_base_url='https://gpt.invalid/v1/', openai_api_key='gpt-key')
            config.write_text(json.dumps(values))
            self.assertEqual(credentials_for(GPT_MODEL, config), ('https://gpt.invalid/v1/responses', 'gpt-key'))
            self.assertEqual(credentials_for(DEFAULT_MODEL, config), ('https://qwen.invalid/v1/chat/completions', 'qwen-key'))
            values['openai_base_url'] = 'http://gpt.invalid/v1'
            config.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                credentials_for(GPT_MODEL, config)
            values['openai_base_url'] = 'https://gpt.invalid/v1/responses'
            del values['openai_api_key']
            config.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                credentials_for(GPT_MODEL, config)
