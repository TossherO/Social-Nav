import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from perception.mapping_localization import update_map_workflow as workflow


class MapUpdateWorkflowTests(unittest.TestCase):
    def test_stages_share_bag_and_use_selected_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag, old_map, coverage, out = (root/'input.bag', root/'old map', root/'coverage.yaml', root/'output')
            calls = []

            def execute(command, **kwargs):
                calls.append(command)
                if 'pick_start.sh' in command[1]:
                    (out/'initial_pose.json').write_text('{}')

            args = workflow.arguments(['--bag', str(bag), '--gpu', '0'])
            with patch.object(workflow, 'prepare', return_value=(bag, old_map, coverage, out)), \
                    patch.object(workflow.subprocess, 'run', side_effect=execute):
                result = workflow.run(args)
            self.assertEqual(result, out/'candidate')
            self.assertEqual(len(calls), 4)
            for command in calls[:3]:
                self.assertEqual(command[command.index('--bag')+1], str(bag))
            self.assertEqual(calls[2][calls[2].index('--coverage')+1], str(coverage))
            self.assertEqual(calls[2][calls[2].index('--pose-file')+1], str(out/'initial_pose.json'))
            self.assertEqual(calls[3][calls[3].index('--localization-run')+1], str(out/'local_reference'))
            self.assertIn('--gpu', calls[1])
            self.assertIn('--gpu', calls[2])

    def test_picker_closed_without_save_stops_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = workflow.arguments(['--bag', str(root/'input.bag')])
            with patch.object(workflow, 'prepare', return_value=(root/'input.bag', root/'map',
                                                                  root/'coverage', root/'out')), \
                    patch.object(workflow.subprocess, 'run') as execute:
                with self.assertRaisesRegex(RuntimeError, '人工选择'):
                    workflow.run(args)
            self.assertEqual(execute.call_count, 1)

    def test_existing_or_symlink_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag = root/'data.bag'
            bag.write_bytes(b'bag')
            old = root/'map'
            old.mkdir()
            (old/'COMPLETE').touch()
            (old/'GlobalMap.pcd').touch()
            out = root/'out'
            os.symlink(root/'missing', out)
            args = workflow.arguments(['--bag', str(bag), '--map', str(old), '--coverage',
                                       str(root/'coverage'), '--out', str(out)])
            with patch.object(workflow, 'asset_digest', return_value='digest'), \
                    patch.object(workflow.PriorCoverage, 'load'):
                with self.assertRaises(FileExistsError):
                    workflow.prepare(args)

    def test_custom_map_without_coverage_uses_automatic_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag = root/'data.bag'; bag.write_bytes(b'bag')
            old = root/'map'; old.mkdir()
            (old/'COMPLETE').touch(); (old/'GlobalMap.pcd').touch()
            args = workflow.arguments(['--bag', str(bag), '--map', str(old), '--out', str(root/'out')])
            with patch.object(workflow, 'full_map_coverage') as auto:
                _, selected, coverage, _ = workflow.prepare(args)
            self.assertEqual(selected, old)
            self.assertIsNone(coverage)
            auto.assert_called_once_with(old)


if __name__ == '__main__':
    unittest.main()
