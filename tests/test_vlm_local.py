"""Current request-level local fallback boundaries; no model or GPU loading."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from decision.vlm import local_inference as local
from tests.test_vlm_api import request_fixture


class LocalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.inputs=self.root/'inputs'
        self.body=request_fixture(self.inputs)
        self.request=self.inputs/'request.json';self.request.write_text(json.dumps(self.body))
        self.model=self.root/'model';self.model.mkdir();(self.model/'config.json').write_text('{}')
        self.out=self.root/'result'

    def args(self,*extra):
        return local.arguments(['--request',str(self.request),'--task','appearance','--out',str(self.out),
            '--model-path',str(self.model),'--config',str(self.root/'private.json'),*extra])

    def engine(self,raw='{"id":0,"match":"unique"}',complete=True):
        return Mock(metadata={'synthetic':True},generate=Mock(return_value=dict(raw_content=raw,
                    finish_reason='stop',grammar_terminated=complete)))

    def test_dry_run_does_not_read_private_config_or_construct_engine(self):
        args=self.args('--dry-run');args.config.write_text('DO_NOT_READ')
        factory=Mock(side_effect=AssertionError('GPU must remain unloaded'))
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(local.run(args,factory),0)
        factory.assert_not_called();self.assertFalse(self.out.exists())

    def test_explicit_fallback_uses_exact_current_request_and_preserves_raw(self):
        engine=self.engine();factory=Mock(return_value=engine)
        self.assertEqual(local.run(self.args('--fallback-reason','remote_gpu_busy'),factory),0)
        self.assertEqual(engine.generate.call_args.args[0],self.body)
        result=json.loads((self.out/'response.json').read_text())
        self.assertEqual(result['raw_content'],'{"id":0,"match":"unique"}')
        self.assertNotIn('request_sha256',result) # No HTTP request was sent.
        translated=local.local_messages(self.body)
        self.assertEqual(translated[1]['content'][1]['image'],self.body['messages'][1]['content'][1]['image_url']['url'])

    def test_invalid_identity_or_incomplete_grammar_is_saved_without_retry(self):
        for raw,complete in [('{"id":9,"match":"unique"}',True),('{"id":0,"match":"unique"}',False)]:
            with self.subTest(raw=raw,complete=complete):
                engine=self.engine(raw,complete);args=self.args('--fallback-reason','remote_unavailable')
                args.out=self.root/('bad_'+str(complete))
                self.assertEqual(local.run(args,Mock(return_value=engine)),1)
                engine.generate.assert_called_once()
                result=json.loads((args.out/'response.json').read_text())
                self.assertEqual(result['raw_content'],raw);self.assertFalse(result['valid'])

    def test_reason_output_and_source_boundaries_are_required(self):
        for target in [self.out,self.inputs/'nested',self.model/'nested']:
            if target==self.out:target.mkdir()
            args=self.args('--fallback-reason','remote_unavailable');args.out=target
            factory=Mock()
            with self.subTest(target=target),self.assertRaises((ValueError,FileExistsError)):local.run(args,factory)
            factory.assert_not_called()
        with self.assertRaises(ValueError):local.run(self.args(),Mock())


if __name__=='__main__':unittest.main()
