"""Dataset aliases and path adaptation; no real dataset mutation or GPU imports."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from decision.vlm import project_config, package_vlm_test
from scripts import dataset_links


class DatasetLinkTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        entries = root / 'data/datasets'
        entries.mkdir(parents=True)
        target = root / 'raw_target'
        target.mkdir()
        for name in dataset_links.NAMES:
            (entries / name).symlink_to(target, target_is_directory=True)
        detector = root / dataset_links.DETECTOR_DATA
        detector.parent.mkdir(parents=True)
        detector.symlink_to(entries, target_is_directory=True)
        return entries, target, detector

    def test_link_checker_is_read_only_and_ready_for_valid_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            entries, target, _ = self.fixture(folder)
            before = {p.name: p.readlink() for p in entries.iterdir()}
            report = dataset_links.inspect_links(folder)
            self.assertEqual(before, {p.name: p.readlink() for p in entries.iterdir()})
            self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(report['status'], 'ready')
        self.assertTrue(report['detector_data_alias_valid'])
        self.assertFalse(report['pickle_loaded'] or report['training_executed'])
        self.assertFalse(report['training_readiness_verified'])

    def test_missing_broken_and_non_symlink_aliases_are_reported(self):
        for mode in ('missing', 'broken', 'directory', 'loop'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                entries, _, _ = self.fixture(folder)
                alias = entries / 'jrdb'
                alias.unlink()
                if mode == 'broken':
                    alias.symlink_to(Path(folder) / 'missing_target')
                elif mode == 'directory':
                    alias.mkdir()
                elif mode == 'loop':
                    alias.symlink_to(alias.name)
                self.assertEqual(dataset_links.inspect_links(folder)['status'], 'missing_or_invalid_links')

    def test_detector_data_cannot_point_to_a_different_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            _, target, detector = self.fixture(folder)
            detector.unlink()
            detector.symlink_to(target, target_is_directory=True)
            self.assertFalse(dataset_links.inspect_links(folder)['detector_data_alias_valid'])

    def test_vlm_paths_use_the_repository_and_preserve_symlink_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(project_config, 'ROOT', root):
                self.assertEqual(project_config.dataset_path('jrdb'), root / 'data/datasets/jrdb')
                self.assertEqual(project_config.dataset_path('socialnav_sub_pilot'), root / 'data/datasets/socialnav_sub')
                self.assertEqual(project_config.project_path('/tmp/explicit_dataset'), Path('/tmp/explicit_dataset'))
        self.assertEqual(package_vlm_test.SOURCE, project_config.project_path(project_config.load_config()['development']['bundle']))

    def test_detector_dataset_path_function_is_repo_relative_without_gpu_import(self):
        # Extract only the pure path function; do not import Torch/CUDA or load infos.
        source = project_config.ROOT / 'perception/detection_tracking/training/simpledet3d/simple_det3d/datasets/coda_dataset.py'
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_dataset_path')
        namespace = {'Path': Path, '__file__': str(source)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        resolver = namespace['_dataset_path']
        copied = source.parents[2]
        self.assertEqual(resolver('data/coda_raw'), copied / 'data/coda_raw')
        self.assertEqual(resolver('data/wheelchair_weak'), copied / 'data/wheelchair_weak')
        self.assertEqual(resolver('/tmp/explicit_raw'), Path('/tmp/explicit_raw'))

    def test_detector_configs_and_stage_checks_use_aliases(self):
        copied = project_config.ROOT / 'perception/detection_tracking/training/simpledet3d'
        for path in (copied / 'configs').glob('*.yaml'):
            config = yaml.safe_load(path.read_text())
            for key in ('DATA_CONFIG', 'SOURCE_DATA_CONFIG'):
                raw = config.get(key, {}).get('RAW_DATA_PATH')
                if raw is not None:
                    with self.subTest(config=path.name, section=key):
                        self.assertIn(raw, ('data/coda_raw', 'data/wheelchair_raw'))
        script = (copied / 'train_wheelchair_3stage.sh').read_text()
        self.assertIn('require_dir "${PROJECT_DIR}/data/coda_raw"', script)
        self.assertIn('require_dir "${PROJECT_DIR}/data/wheelchair_raw"', script)

    def test_preparation_tool_defaults_read_aliases_and_write_local_outputs(self):
        copied = project_config.ROOT / 'perception/detection_tracking/training/simpledet3d'
        names = ('generate_coda_info', 'generate_wheelchair_info', 'downsample_coda_lidar',
                 'prepare_wheelchair_weak_data', 'visualize_wheelchair_weak_labels')
        for name in names:
            path = copied / 'tools' / (name + '.py')
            code = path.read_text()
            with self.subTest(tool=name):
                ast.parse(code)
                self.assertNotIn('/home/heyushen/data/', code)
                self.assertNotIn('/share/public_datasets/', code)
                if name != 'visualize_wheelchair_weak_labels':
                    self.assertIn('output/data_preparation/', code)


if __name__ == '__main__':
    unittest.main()
