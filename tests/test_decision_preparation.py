"""Joint offline orchestration and default input selection; no models or ROS required."""
from pathlib import Path
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decision.vlm.annotation.store import Store
from decision.vlm.annotation.rgb_preparation import prepare_rgb
from perception.detection_tracking.inference.rgb_cache import CacheWriter
from perception.mapping_localization.__main__ import arguments, prepare_decision_project
from tests.test_vlm_annotation import fixture


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.project=self.root/'annotation'
        fixture(self.project)

    def generate_cache(self,command,**kwargs):
        output=Path(command[command.index('--output')+1])
        store=Store(self.project,use_project_defaults=False)
        try:
            writer=CacheWriter(output,dict(project_id=store.identity,asset_identity=store.asset_identity),
                               dict(model='fixture'),dict(method='fixture'))
            for row in store.frames('source')[:kwargs.get('limit',31)]:
                frame=store.frame(row['id'])
                tracks={p['key']:dict(status='matched',rgb_box=[40.,50.,100.,300.]) for p in frame['people']}
                writer.put_observation(frame,dict(status='complete',tracks=tracks))
            writer.finish({});writer.close()
        finally:store.close()
        return subprocess.CompletedProcess(command,0)

    def test_preparation_registers_default_and_local_worker_uses_it(self):
        from decision.vlm.annotation.backend import LocalBackend
        original=Store(self.project)
        before=original.events();original.close()
        with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=self.generate_cache), \
                contextlib.redirect_stdout(io.StringIO()):
            result=prepare_rgb(self.project,self.root/'derived_rgb')
        store=Store(self.project)
        self.assertEqual(store.rgb_cache.identity,result['identity'])
        self.assertEqual(store.events(),before)
        expected=store.window('source/20')
        backend=LocalBackend(self.project,readonly=True)
        try:
            self.assertEqual(backend.snapshot()['rgb_box_source'],'yolo_association')
            self.assertEqual(backend.views(['source/20'],True)[0]['window'],expected)
        finally:backend.close();store.close()

    def test_failure_blocks_projection_fallback_and_resume_uses_registered_path(self):
        output=self.root/'external_rgb'
        with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=subprocess.CalledProcessError(1,[])), \
                contextlib.redirect_stdout(io.StringIO()),self.assertRaises(subprocess.CalledProcessError):
            prepare_rgb(self.project,output)
        with self.assertRaisesRegex(ValueError,'尚未完成'):Store(self.project)
        producer=Store(self.project,use_project_defaults=False)
        self.assertEqual(len(producer.frames('source')),31);producer.close()
        with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=self.generate_cache) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            prepare_rgb(self.project)
        self.assertEqual(Path(run.call_args.args[0][run.call_args.args[0].index('--output')+1]),output)
        ready=Store(self.project);self.assertIsNotNone(ready.rgb_cache);ready.close()

    def test_partial_cache_cannot_be_selected_as_complete_project_input(self):
        def partial(command,**kwargs):
            self.generate_cache(command,limit=30)
            return subprocess.CompletedProcess(command,0)
        with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=partial), \
                contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'全部观察'):
            prepare_rgb(self.project)
        with self.assertRaisesRegex(ValueError,'尚未完成'):Store(self.project)

    def test_existing_cache_registers_without_inference_and_foreign_cache_keeps_selection(self):
        output=self.root/'existing_rgb'
        self.generate_cache(['--output',str(output)])
        with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=AssertionError('must not infer')), \
                contextlib.redirect_stdout(io.StringIO()):
            ready=prepare_rgb(self.project,output)
            again=prepare_rgb(self.project)
        self.assertEqual(ready,again)
        foreign=self.root/'foreign'
        writer=CacheWriter(foreign,dict(project_id='other',asset_identity='other'),dict(model='foreign'),{})
        writer.finish({});writer.close()
        with self.assertRaisesRegex(ValueError,'来源不符'):prepare_rgb(self.project,foreign)
        store=Store(self.project)
        try:self.assertEqual(store.rgb_cache.identity,ready['identity'])
        finally:store.close()

    def test_worker_rejects_source_changed_after_window_creation(self):
        from decision.vlm.annotation.backend import LocalBackend
        backend=LocalBackend(self.project,readonly=True)
        try:
            self.generate_cache(['--output',str(self.root/'new_rgb')])
            with contextlib.redirect_stdout(io.StringIO()):prepare_rgb(self.project,self.root/'new_rgb')
            with self.assertRaisesRegex(ValueError,'来源已改变'):backend.views(['source/20'])
        finally:backend.close()


class OrchestrationTests(unittest.TestCase):
    def test_consumers_no_longer_accept_rgb_source_override(self):
        from decision.vlm.annotation.__main__ import arguments as annotation_arguments
        for command in (['open','--project','fixture'],['open','--server','http://127.0.0.1:8766'],
                        ['serve','--project','fixture'],['export','--project','fixture','--output','export']):
            with self.subTest(command=command[0]):
                args=annotation_arguments(command)
                self.assertFalse(hasattr(args,'rgb_associations'))
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    annotation_arguments(command+['--rgb-associations','old-cache'])

    def test_cli_requires_bag_detector_and_new_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bag=root/'recording.bag';bag.write_bytes(b'CLI_FIXTURE_ONLY')
            weights=root/'model.pt';weights.write_bytes(b'NOT_LOADED')
            base=['--mode','odometry','--bag',str(bag),'--out',str(root/'run'),
                  '--decision-project',str(root/'annotation')]
            with patch('perception.detection_tracking.inference.config.load_config',return_value={
                    'image_association':dict(backend='pytorch',weights=str(weights))}):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(base)
                args=arguments(base+['--people-python',sys.executable])
                self.assertEqual(args.decision_project,str(root/'annotation'))
                self.assertTrue(args.image_association)
                self.assertFalse(arguments(base+['--people-python',sys.executable,'--no-image-association']).image_association)
                core=['--mode','odometry','--bag',str(bag),'--out',str(root/'run'),'--people-python',sys.executable]
                self.assertFalse(arguments(core).image_association)
                self.assertTrue(arguments(core+['--image-association']).image_association)
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    arguments(['--mode','odometry','--bag',str(bag),'--out',str(root/'run'),'--image-association'])
                self.assertFalse((root/'run').exists())
                bad=['--mode','odometry','--source','live','--out',str(root/'run'),
                     '--decision-project',str(root/'annotation'),'--people-python',sys.executable]
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(bad)
                (root/'annotation').mkdir()
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(base+['--people-python',sys.executable])

    def test_handoff_preserves_frozen_run_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=root/'run.json'
            frozen=json.dumps(dict(success=True,people_transport={'rgb_topic':'/front/rgb'})).encode()
            run.write_bytes(frozen)
            (root/'rgb_association').mkdir()
            (root/'rgb_association/manifest.json').write_text(json.dumps(dict(complete=True,observations=10,identity='derived')))
            args=SimpleNamespace(decision_project=str(root/'annotation'),people_python=sys.executable,
                bag=str(root/'recording.bag'),decision_calibration=str(root/'camera.yaml'),decision_root_group='original')
            process=SimpleNamespace(wait=lambda:0,poll=lambda:0)
            with patch('perception.mapping_localization.__main__.subprocess.Popen',return_value=process) as spawn, \
                    contextlib.redirect_stdout(io.StringIO()):
                ready=prepare_decision_project(args,root)
            self.assertEqual(ready['status'],'ready')
            command=spawn.call_args.args[0]
            self.assertIn('--with-rgb-associations',command)
            self.assertEqual(command[command.index('--rgb-topic')+1],'/front/rgb')
            self.assertEqual(run.read_bytes(),frozen)
            failed=SimpleNamespace(wait=lambda:1,poll=lambda:1)
            with patch('perception.mapping_localization.__main__.subprocess.Popen',return_value=failed), \
                    contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'已有感知结果保留'):
                prepare_decision_project(args,root)
            self.assertEqual(run.read_bytes(),frozen)


if __name__=='__main__':unittest.main()
