"""Local runner boundaries with synthetic outputs; no model, CUDA or network."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from decision.vlm import local_inference as local
from decision.vlm import vlm_api_common as common
from test_vlm_api import fixture


class LocalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'inputs'
        fixture(self.bundle)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.out = self.root / 'result'

    def args(self, *flags):
        return local.arguments(['--bundle', str(self.bundle), '--model-path', str(self.model),
                                '--config', str(self.root / 'missing.json'), '--out', str(self.out), *flags])

    def engine(self, content='{"g":[],"u":[]}', **changes):
        engine = Mock(metadata={'synthetic': True})
        engine.generate.return_value = dict(raw_content=content, generate_s=.1,
                finish_reason='stop', grammar_terminated=True, usage={'prompt_tokens': 17}, **changes)
        return engine

    def run_local(self, engine, *flags):
        with contextlib.redirect_stdout(io.StringIO()):
            return local.run(self.args(*flags), engine_factory=Mock(return_value=engine))

    def rows(self):
        return [json.loads(line) for line in (self.out / 'requests.jsonl').read_text().splitlines()]

    def test_dry_run_ignores_private_config_and_never_constructs_engine(self):
        args = self.args('--dry-run')
        args.config.write_text('PRIVATE_CONFIG_MUST_NOT_BE_READ')
        engine = Mock(side_effect=AssertionError('No model loading'))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(local.run(args, engine), 0)
        engine.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_same_logical_request_and_image_order_without_sending_http(self):
        engine = self.engine()
        self.assertEqual(self.run_local(engine, '--limit', '1'), 0)
        _, system, cases = common.load_bundle(self.bundle)
        expected = common.make_body(system, cases[0])
        body = engine.generate.call_args.args[0]
        self.assertEqual(body, expected)
        self.assertEqual(self.rows()[0]['reference_request_sha256'], local.fingerprint(expected))
        content = local.local_messages(body)[1]['content']
        for i in range(3):
            self.assertEqual(content[2*i]['text'], f'IMAGE_{i}')
            self.assertEqual(content[2*i+1]['image'], body['messages'][1]['content'][2*i+1]['image_url']['url'])
        self.assertNotIn('request_sha256', self.rows()[0])  # No HTTP request was actually sent.

    def test_invalid_duplicate_membership_preserved_and_stops_without_retry(self):
        content = '{"g":[{"m":[0,1],"r":"w","k":0}],"u":[0]}'
        engine = self.engine(content)
        self.assertEqual(self.run_local(engine), 1)
        engine.generate.assert_called_once()
        row = self.rows()[0]
        self.assertEqual(row['raw_content'], content)
        self.assertFalse(row['valid'])
        self.assertEqual(row['failure_stage'], 'membership')

    def test_incomplete_grammar_is_not_accepted_even_when_json_is_valid(self):
        engine = self.engine()
        engine.generate.return_value['grammar_terminated'] = False
        self.assertEqual(self.run_local(engine), 1)
        self.assertEqual(self.rows()[0]['failure_stage'], 'completion')
        engine.generate.assert_called_once()

    def test_output_cannot_overwrite_existing_or_frozen_assets(self):
        for target in (self.out, self.model / 'nested', self.bundle / 'nested'):
            if target == self.out:
                target.mkdir()
                (target / 'preserve.txt').write_text('original')
            args = self.args()
            args.out = target
            engine = Mock()
            with self.subTest(target=target), self.assertRaises((ValueError, FileExistsError)):
                local.run(args, engine)
            engine.assert_not_called()
        self.assertEqual((self.out / 'preserve.txt').read_text(), 'original')

    def test_reference_binding_checked_after_generation(self):
        engine = self.engine()
        self.assertEqual(self.run_local(engine, '--limit', '1'), 0)
        row = self.rows()[0]
        settings = common.read_json(self.out / 'settings.json')
        reference = self.root / 'reference'
        reference.mkdir()
        common.save_json(reference / 'summary.json', {'settings': settings})
        old = dict(row, request_sha256=row['reference_request_sha256'])
        (reference / 'requests.jsonl').write_text(json.dumps(old) + '\n')
        result = local.compare_reference([row], reference, settings)
        self.assertEqual(result['identical_rel'], 1)
        old['request_sha256'] = '0' * 64
        (reference / 'requests.jsonl').write_text(json.dumps(old) + '\n')
        with self.assertRaisesRegex(ValueError, 'binding'):
            local.compare_reference([row], reference, settings)


if __name__ == '__main__':
    unittest.main()
