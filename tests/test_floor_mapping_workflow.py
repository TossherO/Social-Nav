"""Default mapping workflow preserves raw data and publishes only complete stages."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from perception.mapping_localization import build_map
from perception.mapping_localization.results import file_fingerprint, map_fingerprint
from perception.environment.__main__ import bundle_main


class FloorMappingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bag = self.root/'source.bag'; self.bag.write_bytes(b'synthetic bag')
        self.library = self.root/'roll.so'; self.library.write_bytes(b'synthetic library')
        self.output = self.root/'mapping'

    def args(self, *extra):
        return build_map.parser().parse_args(['--bag', str(self.bag), '--out', str(self.output),
                                              '--roll-library', str(self.library), *extra])

    def test_default_enables_robust_loop_without_thinning_or_adaptive_noise(self):
        captured = {}
        def raw(args, stop):
            captured.update(vars(args)); Path(args.out).mkdir()
            return {'counts': {'valid': 1, 'scans': 1}}
        with patch.object(build_map, '_build_raw', side_effect=raw), \
             patch.object(build_map, 'finish_floor_pipeline', return_value={'success': True}):
            result = build_map.build(self.args())
        self.assertTrue(result['success'])
        self.assertEqual(captured['pipeline'], 'raw')
        self.assertTrue(captured['robust_loop'])
        self.assertEqual(captured['loop_min_gap'], 0.)
        self.assertFalse(captured['loop_adaptive_noise'])

    def test_failure_preserves_raw_and_marks_whole_pipeline_incomplete(self):
        def raw(args, stop):
            path = Path(args.out); path.mkdir(); (path/'saved').write_text('raw evidence')
            return {'counts': {'valid': 1, 'scans': 1}}
        def fail_after_partial_publication(source, output, summary, stop):
            (output/'map').symlink_to('raw/map', target_is_directory=True)
            raise ValueError('floor unavailable')
        with patch.object(build_map, '_build_raw', side_effect=raw), \
             patch.object(build_map, 'finish_floor_pipeline', side_effect=fail_after_partial_publication):
            with self.assertRaisesRegex(ValueError, 'floor unavailable'):
                build_map.build(self.args())
        self.assertTrue((self.output/'raw/saved').is_file())
        self.assertFalse(json.loads((self.output/'run.json').read_text())['success'])
        self.assertFalse((self.output/'map').is_symlink())

    def test_floor_pipeline_rejects_missing_scans_and_disabled_loop_before_writing(self):
        for options in (('--no-record-scans',), ('--loop-closure', 'off')):
            with self.assertRaises(ValueError): build_map.build(self.args(*options))
            self.assertFalse(self.output.exists())

    def test_publish_exposes_the_optimized_map_and_bound_raw_scans(self):
        self.output.mkdir(); raw = self.output/'raw'; raw.mkdir()
        (raw/'environment').mkdir(); (raw/'environment/scans.sqlite').write_bytes(b'scans')
        (raw/'frontend.jsonl').write_text('')
        (raw/'algorithm.yaml').write_text('roll: {}\n')
        def level(source, static, destination):
            Path(destination).mkdir(); return {'rotation_angle_deg': 2.}
        def refine(source, destination):
            destination = Path(destination); (destination/'map/keyframes').mkdir(parents=True)
            (destination/'map/COMPLETE').write_text('complete')
            (destination/'map/manifest.yaml').write_text('frame: map\n')
            (destination/'map/keyframes/poses.txt').write_text('0 0 0 0 0 0 0 0 0\n')
            (destination/'map/GlobalMap.pcd').write_bytes(b'synthetic')
            (destination/'localization.jsonl').write_text('')
            (destination/'run.json').write_text(json.dumps(dict(success=True, source='bag', mode='mapping',
                bag=str(self.bag), playback=dict(map_fingerprint=map_fingerprint(destination/'map'),
                                                 bag_fingerprint=file_fingerprint(self.bag)))))
            return {'solver': {'correction_range_m': [0., 0.]}}
        with patch('perception.mapping_localization.level_map.create', side_effect=level), \
             patch('perception.mapping_localization.ground_refine.create', side_effect=refine):
            summary = build_map.finish_floor_pipeline(raw, self.output, {'success': True, 'counts': {}})
        run = json.loads((self.output/'run.json').read_text())
        self.assertEqual(run['playback']['map_fingerprint'], map_fingerprint(self.output/'map'))
        self.assertEqual((self.output/'environment/scans.sqlite').read_bytes(), b'scans')
        self.assertEqual(summary['pipeline'], 'single_floor_robust_ground')
        self.assertFalse(run['pipeline']['verified_for_navigation'])

    def test_static_export_defaults_to_tracked_scans_for_mapping_runs(self):
        with patch('perception.environment.export.export_environment',
                   return_value={'snapshot_id': 'synthetic'}) as export, contextlib.redirect_stdout(io.StringIO()):
            bundle_main(['export', '--run', str(self.output), '--out', str(self.root/'static')])
        self.assertEqual(export.call_args.kwargs['ground_method'], 'tracked_scans')
        with patch('perception.environment.export.export_environment',
                   return_value={'snapshot_id': 'synthetic'}) as export, contextlib.redirect_stdout(io.StringIO()):
            bundle_main(['export', '--run', str(self.output), '--out', str(self.root/'static'),
                         '--ground-method', 'map_endpoints'])
        self.assertEqual(export.call_args.kwargs['ground_method'], 'map_endpoints')


if __name__ == '__main__': unittest.main()
