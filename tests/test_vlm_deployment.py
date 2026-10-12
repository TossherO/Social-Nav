"""Remote deployment plans and read-only result checks; no installs/GPU/network."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import zipfile

from decision.vlm.deployment import deploy, compare_results


class DeploymentTests(unittest.TestCase):
    def test_official_runtime_pin_does_not_claim_base_torch_compatibility(self):
        profile = deploy.profile()
        self.assertEqual(profile['packages']['torch'], '2.13.0+cu130')
        self.assertEqual(profile['packages']['sglang'], '0.5.19')
        self.assertEqual(profile['packages']['cuda-tile'], '1.6.0rc5')
        self.assertIn('2.12', profile['base_image'])
        self.assertEqual(profile['installation_mode'], 'current_environment')
        self.assertFalse(profile['evidence']['remote_cuda13_installation_tested'])

    def test_install_plan_does_not_install_create_env_or_import_torch(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'not-created'
            with patch.object(deploy.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
                code = deploy.main(['install', '--out', str(target)])
            self.assertEqual(code, 0)
            run.assert_not_called()
            self.assertFalse(target.exists())
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['installation_mode'], 'current_environment')
            self.assertEqual(plan['python_executable'], deploy.sys.executable)
            self.assertFalse(plan['creates_virtual_environment'])
            self.assertTrue(plan['environment_will_be_modified_on_execute'])
            self.assertFalse(plan['execute'])
            self.assertIn('torch==2.13.0+cu130', plan['commands'][0])
            self.assertIn('--only-binary=:all:', plan['commands'][0])
            self.assertIn('--retries 0', plan['commands'][0])
            self.assertIn('--extra-index-url https://pypi.nvidia.com', plan['commands'][0])
            self.assertIn('cuda-tile==1.6.0rc5', plan['commands'][0])

    def test_install_plan_targets_invoking_python_without_creating_any_environment(self):
        commands = deploy.install_commands(python='/existing/python3.12')
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[:4], ['/existing/python3.12', '-m', 'pip', '--isolated'])
            for forbidden in ('venv', '--user', '--no-deps', '--force-reinstall', '--break-system-packages'):
                self.assertNotIn(forbidden, command)
        self.assertIn('--upgrade', commands[0])
        self.assertIn('--no-user', commands[0])
        self.assertEqual(commands[1][-1], 'check')

    def test_wheel_only_install_includes_both_official_extra_indexes(self):
        command = deploy.install_commands()[0]
        indexes = [command[i + 1] for i, token in enumerate(command) if token == '--extra-index-url']
        self.assertEqual(indexes, ['https://download.pytorch.org/whl/cu130', 'https://pypi.nvidia.com'])
        self.assertIn('--only-binary=:all:', command)
        self.assertIn('cuda-tile==1.6.0rc5', command)

    def test_execute_requires_environment_change_acknowledgement_and_records_directory(self):
        for flags in (['--execute'], ['--execute', '--allow-env-change']):
            with self.subTest(flags=flags), patch.object(deploy.subprocess, 'run') as run, \
                    patch.object(deploy, 'output_directory') as output, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', *flags]), 1)
            run.assert_not_called()
            output.assert_not_called()

    def test_existing_installation_records_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'records'
            target.mkdir()
            (target / 'user.txt').write_text('preserve')
            with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                    patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                    patch.object(deploy.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', '--execute', '--allow-env-change', '--out', str(target)]), 1)
            run.assert_not_called()
            self.assertEqual((target / 'user.txt').read_text(), 'preserve')

    def test_execute_rejects_wrong_python_before_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'records'
            args = argparse.Namespace(out=target, execute=True, allow_env_change=True, min_free_gib=20)
            with patch.object(deploy.sys, 'version_info', (3, 11)), patch.object(deploy.subprocess, 'run') as run:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Python 3.12'):
                    deploy.install(args)
            self.assertFalse(target.exists())
            run.assert_not_called()

    def test_current_environment_install_records_mocked_changes_without_gpu_import(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'records'
            before = [{'name': 'torch', 'version': '2.12.0'}]
            after = [{'name': 'torch', 'version': '2.13.0+cu130'}]
            with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                    patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                    patch.object(deploy, 'installed_packages', side_effect=[before, after]), \
                    patch.object(deploy, 'package_versions', return_value=deploy.profile()['packages']), \
                    patch.object(deploy.importlib, 'import_module') as imports, \
                    patch.object(deploy.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', '--execute', '--allow-env-change', '--out', str(target)]), 0)
            self.assertEqual(run.call_count, 2)
            imports.assert_not_called()
            report = deploy.read_json(target / 'installation.json')
            self.assertTrue(report['installation_completed'])
            self.assertTrue(report['target_environment_modified'])
            self.assertFalse(report['creates_virtual_environment'])
            self.assertFalse(report['gpu_or_model_inference_verified'])
            self.assertEqual(deploy.read_json(target / 'installed_packages_before.json'), before)
            self.assertEqual(deploy.read_json(target / 'installed_packages_after.json'), after)

    def test_failed_pip_check_keeps_before_record_and_does_not_auto_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'records'
            with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                    patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                    patch.object(deploy, 'installed_packages', return_value=[]), \
                    patch.object(deploy.subprocess, 'run', side_effect=[None, deploy.subprocess.CalledProcessError(1, ['pip', 'check'])]) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', '--execute', '--allow-env-change', '--out', str(target)]), 1)
            self.assertEqual(run.call_count, 2)
            self.assertTrue((target / 'installed_packages_before.json').is_file())
            self.assertFalse((target / 'installation.json').exists())
            failed = deploy.read_json(target / 'installation_failed.json')
            self.assertTrue(failed['environment_may_be_partially_modified'])
            self.assertFalse(failed['automatic_rollback'])

    def test_insufficient_space_or_python_path_override_stops_before_install(self):
        for free, python_path in ((0, ''), (100 * 1024**3, '/untrusted/extra-packages')):
            with self.subTest(free=free, python_path=python_path), \
                    patch.dict(deploy.os.environ, {'PYTHONPATH': python_path, 'PYTHONHOME': ''}), \
                    patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), \
                    patch.object(deploy.subprocess, 'run') as run, patch.object(deploy, 'output_directory') as output, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', '--execute', '--allow-env-change', '--out', '/not-created']), 1)
            run.assert_not_called()
            output.assert_not_called()

    def test_serve_without_execute_only_prints_portable_baseline(self):
        with patch.object(deploy.subprocess, 'Popen') as process, patch.object(deploy, 'inspect_environment') as inspect:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = deploy.main(['serve', '--model-path', '/remote/Qwen3.8-27B-FP8'])
            self.assertEqual(code, 0)
            command = output.getvalue()
            self.assertIn('--host 127.0.0.1', command)
            self.assertIn('--grammar-backend xgrammar', command)
            self.assertIn('--max-running-requests 1', command)
            for forbidden in ('--attention-backend', '--fp8-gemm-backend', '--speculative-algorithm', '--api-key'):
                self.assertNotIn(forbidden, command)
            process.assert_not_called()
            inspect.assert_not_called()

    def test_serve_execute_requires_explicit_log_directory(self):
        with patch.object(deploy.subprocess, 'Popen') as process, contextlib.redirect_stdout(io.StringIO()):
            code = deploy.main(['serve', '--model-path', '/remote/model', '--execute'])
        self.assertEqual(code, 1)
        process.assert_not_called()

    def test_invalid_serve_parameters_rejected_without_gpu(self):
        for flags in (['--port', '0'], ['--context-length', '100'], ['--mem-fraction-static', '0.99']):
            with self.subTest(flags=flags), patch.object(deploy.subprocess, 'Popen') as process:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = deploy.main(['serve', '--model-path', '/remote/model', *flags])
                self.assertEqual(code, 1)
                process.assert_not_called()

    def model_fixture(self, root):
        root = Path(root)
        root.mkdir()
        config = {'vision_config': {}, 'quantization_config': {'quant_method': 'fp8'},
                  'architectures': ['SyntheticForConditionalGeneration'], 'model_type': 'synthetic'}
        (root / 'config.json').write_text(json.dumps(config))
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'synthetic': 'part.safetensors'}}))
        (root / 'part.safetensors').write_bytes(b'SYNTHETIC_NOT_LOADED')
        for name in ('tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json'):
            (root / name).write_text('{}')
        return root

    def test_model_inspection_only_checks_assets_not_inference_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.model_fixture(Path(folder) / 'model')
            result = deploy.inspect_model(root)
            self.assertEqual(result['shards'], 1)
            self.assertEqual(result['weight_bytes'], len(b'SYNTHETIC_NOT_LOADED'))
            self.assertFalse(result['all_weight_bytes_hashed'])
            self.assertFalse(result['checkpoint_identity_verified'])

    def test_model_index_path_escape_and_missing_shard_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.model_fixture(Path(folder) / 'model')
            for name in ('../outside.safetensors', 'missing.safetensors'):
                (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': name}}))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    deploy.inspect_model(root)

    def test_non_fp8_or_missing_visual_processor_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.model_fixture(Path(folder) / 'model')
            (root / 'preprocessor_config.json').unlink()
            with self.assertRaisesRegex(ValueError, 'processor'):
                deploy.inspect_model(root)
            (root / 'preprocessor_config.json').write_text('{}')
            (root / 'config.json').write_text(json.dumps({'vision_config': {}}))
            with self.assertRaisesRegex(ValueError, 'FP8'):
                deploy.inspect_model(root)

    def test_version_pin_allows_native_cuda_suffix_but_not_wrong_torch_cuda(self):
        self.assertTrue(deploy.version_matches('0.4.6.post1+cu130', '0.4.6.post1'))
        self.assertFalse(deploy.version_matches('2.13.0+cu129', '2.13.0+cu130'))
        self.assertFalse(deploy.version_matches(None, '2.13.0+cu130'))

    def test_metadata_inspection_never_initializes_gpu(self):
        packages = dict(deploy.profile()['packages'])
        with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                patch.object(deploy, 'package_versions', return_value=packages), \
                patch.object(deploy.importlib, 'import_module') as imports:
            report = deploy.inspect_environment(gpu=False)
        imports.assert_not_called()
        self.assertFalse(report['gpu_checked'])
        self.assertFalse(report['launch_preflight_passed'])
        self.assertFalse(report['model_loaded'])

    def test_mock_gpu_preflight_is_not_claimed_as_multimodal_inference(self):
        packages = dict(deploy.profile()['packages'])
        tensor = MagicMock()
        torch = SimpleNamespace(version=SimpleNamespace(cuda='13.0'), ones=MagicMock(return_value=tensor),
            cuda=SimpleNamespace(is_available=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(name='Synthetic FP8 GPU', major=8, minor=9),
                mem_get_info=lambda index: (48 * 1024**3, 48 * 1024**3), synchronize=lambda: None))
        for driver, expected in (('580.65.06', True), ('579.00', False)):
            with self.subTest(driver=driver), tempfile.TemporaryDirectory() as folder:
                model = self.model_fixture(Path(folder) / 'model')
                with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                        patch.object(deploy, 'package_versions', return_value=packages), \
                        patch.object(deploy.importlib, 'import_module', side_effect=lambda n: torch if n == 'torch' else object()), \
                        patch.object(deploy.subprocess, 'run', return_value=SimpleNamespace(
                            stdout=f'Synthetic FP8 GPU, {driver}, 49152, 49152\n')):
                    report = deploy.inspect_environment(gpu=True, model_path=model)
                self.assertEqual(report['launch_preflight_passed'], expected)
                self.assertFalse(report['real_multimodal_schema_inference_verified'])
                self.assertFalse(report['model_loaded'])

    def test_reports_cannot_be_written_into_model_or_frozen_case_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model'
            model.mkdir()
            with self.assertRaises(ValueError):
                deploy.output_directory(model / 'logs', model_path=model)
        with self.assertRaises(ValueError):
            deploy.output_directory(deploy.HERE / 'cases/new')


class WheelhouseTests(unittest.TestCase):
    """Synthetic wheels only; never install or resolve real third-party packages."""

    def wheel(self, folder, name='example', version='1.0', dependency=None):
        dist = name.replace('-', '_')
        path = Path(folder) / f'{dist}-{version}-py3-none-any.whl'
        metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'
        if dependency:
            metadata += f'Requires-Dist: {dependency}\n'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(f'{dist}-{version}.dist-info/METADATA', metadata)
            archive.writestr(f'{dist}-{version}.dist-info/WHEEL',
                             'Wheel-Version: 1.0\nTag: py3-none-any\n')
        return path

    def fixture(self, folder):
        folder = Path(folder)
        for name, version in deploy.profile()['packages'].items():
            self.wheel(folder, name, version)
        self.wheel(folder, 'synthetic-transitive', '1.0')
        with patch.object(deploy.subprocess, 'run') as run:
            report = deploy.seal_wheelhouse(folder)
        command = run.call_args.args[0]
        self.assertIn('--dry-run', command)
        self.assertIn('--ignore-installed', command)
        self.assertIn('--no-index', command)
        self.assertIn('--require-hashes', command)
        return report

    def test_download_is_print_only_by_default_and_never_installs(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'not-created'
            with patch.object(deploy.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()) as text:
                self.assertEqual(deploy.main(['download', '--out', str(out), '--archive']), 0)
            self.assertFalse(out.exists())
            run.assert_not_called()
            report = json.loads(text.getvalue())
            self.assertFalse(report['environment_will_be_modified'])
            self.assertIn('download --dest', report['command'])
            self.assertIn('--extra-index-url https://pypi.nvidia.com', report['command'])
            self.assertNotIn('--no-deps', report['command'])

    def test_inventory_full_lock_and_closure_verified_without_importing_wheels(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.fixture(folder)
            actual = deploy.verify_wheelhouse(folder)
            self.assertEqual(report, actual)
            self.assertTrue(actual['offline_dependency_resolution_passed'])
            self.assertFalse(actual['environment_modified'])
            self.assertFalse(actual['gpu_or_model_inference_verified'])
            self.assertEqual(actual['wheel_count'], len(deploy.profile()['packages']) + 1)
            self.assertIn('synthetic-transitive==1.0 --hash=sha256:',
                          (Path(folder) / 'requirements.lock').read_text())

    def test_offline_install_command_uses_full_hash_lock_and_no_remote_indexes(self):
        command = deploy.install_commands(wheelhouse='/assets/wheels', python='/remote/python')[0]
        self.assertEqual(command[:4], ['/remote/python', '-m', 'pip', '--isolated'])
        for option in ('--no-index', '--require-hashes', '--only-binary=:all:', '--disable-pip-version-check', '--no-user'):
            self.assertIn(option, command)
        for forbidden in ('--index-url', '--extra-index-url', '--no-deps', '--force-reinstall'):
            self.assertNotIn(forbidden, command)
        self.assertEqual(command[-2:], ['-r', '/assets/wheels/requirements.lock'])

    def test_vendored_dist_info_is_not_mistaken_for_the_wheel_own_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.wheel(folder)
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('example/_vendor/embedded-9.0.dist-info/METADATA',
                                 'Name: embedded\nVersion: 9.0\n')
            rows = deploy.wheel_inventory(folder)
            self.assertEqual([(r['name'], r['version']) for r in rows], [('example', '1.0')])

    def test_missing_modified_or_extra_wheel_and_changed_lock_are_rejected(self):
        for mutation in ('missing', 'changed', 'extra', 'lock'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder)
                wheel = next(Path(folder).glob('*.whl'))
                if mutation == 'missing':
                    wheel.unlink()
                elif mutation == 'changed':
                    with zipfile.ZipFile(wheel, 'a') as archive:
                        archive.writestr('synthetic_extra.txt', 'CHANGED')
                elif mutation == 'extra':
                    self.wheel(folder, 'unlisted', '2.0')
                else:
                    (Path(folder) / 'requirements.lock').write_text('example==1.0\n')
                with self.assertRaises(ValueError):
                    deploy.verify_wheelhouse(folder)

    def test_wrong_profile_and_unvalidated_closure_rejected(self):
        for mutation in ('profile', 'closure'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder)
                path = Path(folder) / 'wheelhouse_manifest.json'
                manifest = deploy.read_json(path)
                if mutation == 'profile':
                    manifest['installation_profile_sha256'] = '0' * 64
                else:
                    manifest['offline_dependency_resolution_passed'] = False
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    deploy.verify_wheelhouse(folder)

    def test_unmatched_roots_duplicate_packages_and_remote_dependency_rejected(self):
        for mutation in ('missing-roots', 'duplicate', 'remote'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                self.wheel(folder)
                if mutation == 'duplicate':
                    self.wheel(folder, 'example', '2.0')
                elif mutation == 'remote':
                    self.wheel(folder, 'other', dependency='external @ https://example.com/package.whl')
                with patch.object(deploy.subprocess, 'run') as run, self.assertRaises(ValueError):
                    deploy.seal_wheelhouse(folder)
                run.assert_not_called()
                self.assertFalse((Path(folder) / 'wheelhouse_manifest.json').exists())

    def test_corrupt_archive_or_linked_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.whl'
            path.write_bytes(b'INVALID_ZIP')
            with self.assertRaises(zipfile.BadZipFile):
                deploy.wheel_inventory(folder)
            path.unlink()
            self.wheel(folder)
            (Path(folder) / 'linked.whl').symlink_to(next(Path(folder).glob('*.whl')))
            with self.assertRaises(ValueError):
                deploy.wheel_inventory(folder)

    def test_failed_offline_resolution_does_not_claim_complete_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, version in deploy.profile()['packages'].items():
                self.wheel(folder, name, version)
            with patch.object(deploy.subprocess, 'run', side_effect=deploy.subprocess.CalledProcessError(1, ['pip'])), \
                    self.assertRaises(deploy.subprocess.CalledProcessError):
                deploy.seal_wheelhouse(folder)
            self.assertFalse((Path(folder) / 'wheelhouse_manifest.json').exists())

    def test_invalid_wheelhouse_stops_before_modifying_environment_or_creating_records(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'records'
            with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                    patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                    patch.object(deploy.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', '--wheelhouse', str(Path(folder) / 'missing'),
                                             '--execute', '--allow-env-change', '--out', str(target)]), 1)
            self.assertFalse(target.exists())
            run.assert_not_called()

    def test_wheel_archive_only_exports_locked_resources_not_private_files(self):
        with tempfile.TemporaryDirectory() as folder:
            wheels = Path(folder) / 'wheels'
            wheels.mkdir()
            report = self.fixture(wheels)
            (wheels / 'server_config.json').write_text('PRIVATE_DO_NOT_EXPORT')
            result = deploy.archive_wheelhouse(wheels)
            with zipfile.ZipFile(result['archive']) as archive:
                self.assertEqual(len(archive.namelist()), report['wheel_count'] + 2)
                self.assertNotIn('wheels/server_config.json', archive.namelist())
            with self.assertRaises(ValueError):
                deploy.archive_wheelhouse(wheels)

    def test_mock_offline_install_records_complete_lock_without_gpu_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            wheels = Path(folder) / 'wheels'
            wheels.mkdir()
            manifest = self.fixture(wheels)
            versions = {r['name']: r['version'] for r in manifest['wheels']}
            target = Path(folder) / 'records'
            with patch.dict(deploy.os.environ, {'PYTHONPATH': '', 'PYTHONHOME': ''}), \
                    patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                    patch.object(deploy, 'installed_packages', return_value=[]), \
                    patch.object(deploy, 'package_versions', side_effect=lambda names: {n: versions[deploy.canonical_name(n)] for n in names}), \
                    patch.object(deploy.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(deploy.main(['install', '--wheelhouse', str(wheels), '--execute',
                                             '--allow-env-change', '--out', str(target)]), 0)
            self.assertEqual(run.call_count, 2)
            self.assertIn('--no-index', run.call_args_list[0].args[0])
            report = deploy.read_json(target / 'installation.json')
            self.assertEqual(report['dependency_source'], 'offline_wheelhouse')
            self.assertEqual(report['locked_packages'], versions)


class ResultComparisonTests(unittest.TestCase):
    def result(self, delay=1.0, sha='a' * 64):
        row = {'case_id': 'synthetic', 'valid': True, 'request_sha256': sha,
               'raw_content': '{"g":[],"u":[]}',
               'timing': {'end_to_end_s': delay, 'first_output_text_s': delay / 2}}
        summary = {'submitted': 1, 'valid': 1, 'latency_valid_only_s': {'end_to_end_s': {'mean': delay}}}
        return summary, [row]

    def test_paired_differences_are_not_claimed_as_network_or_gpu_time(self):
        result = compare_results.compare(self.result(1.1), self.result(1.6))
        self.assertAlmostEqual(result['candidate_minus_reference_mean_complete_s'], .5)
        self.assertEqual(result['input_binding_methods'], ['request_sha256'])
        self.assertFalse(result['pure_gpu_or_network_time_measured'])

    def test_mismatched_or_absent_binding_is_rejected(self):
        for sha in ('different', None):
            with self.subTest(sha=sha), self.assertRaisesRegex(ValueError, 'fingerprints'):
                compare_results.compare(self.result(), self.result(sha=sha))

    def test_request_parameters_take_priority_over_observation_binding(self):
        left, right = self.result(), self.result(sha='different')
        left[1][0]['input_fingerprint'] = right[1][0]['input_fingerprint'] = 'same-observations'
        with self.assertRaises(ValueError):
            compare_results.compare(left, right)

    def test_duplicate_or_zero_valid_pairs_are_rejected(self):
        summary, rows = self.result()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compare_results.compare((summary, rows * 2), self.result())
        rows[0]['valid'] = False
        with self.assertRaisesRegex(ValueError, 'No paired'):
            compare_results.compare((summary, rows), self.result())

    def test_zip_only_reads_two_whitelisted_logs_without_extracting_private_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.zip'
            summary, rows = self.result()
            with zipfile.ZipFile(path, 'x') as archive:
                archive.writestr('results/serial/summary.json', json.dumps(summary))
                archive.writestr('results/serial/requests.jsonl', json.dumps(rows[0]) + '\n')
                archive.writestr('server_config.json', 'SYNTHETIC_PRIVATE_DO_NOT_READ')
            self.assertEqual(compare_results.read_results(path, 'serial'), (summary, rows))
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
