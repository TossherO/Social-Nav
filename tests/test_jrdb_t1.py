"""Current single-protocol T1 rendering, causal inputs, provenance and review gates."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError
import numpy as np

from decision.vlm import jrdb_t1_common as t1, jrdb_t1_inputs as inputs, jrdb_t1 as workflow
from decision.vlm import vlm_api_common as api
from test_jrdb_t1_render import fixture


class T1Tests(unittest.TestCase):
    def test_dynamic_schema_pid_and_extra_fields(self):
        validator = Draft202012Validator(inputs.rel_schema([2, 5, 9]))
        validator.validate({'g': [{'m': [2, 9], 'r': 'w', 'k': 0}], 'u': [5]})
        for value in ({'g': [], 'u': [4]}, {'g': [], 'u': [], 'why': 'extra'},
                      {'g': [{'m': [2], 'r': 't', 'k': 0}], 'u': []}):
            with self.assertRaises(ValidationError):
                validator.validate(value)
        for ids in ([], [7]):
            Draft202012Validator(inputs.rel_schema(ids)).validate({'g': [], 'u': ids})
        with self.assertRaises(ValueError):
            inputs.rel_schema([2, 2])

    def test_no_semantic_leakage_into_text_or_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'dataset'
            sequence, frames, path, _ = fixture(root)
            before = t1.build_window(root, sequence, frames, lidar=False)
            pixels = np.asarray(inputs.render_focus(before, 2, gallery=False, context_bev=True))
            model = inputs.model_input(before)
            data = t1.jrdb.load_json(path)
            for rows in data['labels'].values():
                rows.reverse()
                for row in rows:
                    row['social_group'] = {'cluster_ID': 993}
                    row['H-interaction'] = {'PRIVATE_LABEL': True}
            t1.write_json(path, data)
            after = t1.build_window(root, sequence, frames, lidar=False)
            np.testing.assert_array_equal(pixels, np.asarray(inputs.render_focus(after, 2, gallery=False, context_bev=True)))
            self.assertEqual(model, inputs.model_input(after))
            self.assertEqual(set(model[0]), {'v', 'dt', 'ts', 'e', 'o', 'd'})
            self.assertNotIn('PRIVATE_LABEL', model[2])
            self.assertNotIn('cluster_ID', model[2])
            self.assertEqual(inputs.render_focus(after, 2, gallery=False).size, (1280, 544))

    def test_bundle_integrity_no_duplicate_contact_sheet(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'dataset', Path(temp) / 'input'
            sequence, frames, _, _ = fixture(root)
            window = t1.build_window(root, sequence, frames, lidar=False)
            inputs.save_bundle(window, 'object', out)
            self.assertEqual(inputs.load_window(out)[0].geometry_signature, window.geometry_signature)
            self.assertFalse((out / 'input_contact_sheet.png').exists())
            model, _ = inputs.load_input(out)
            model['user_prompt'] += ' changed'
            t1.write_json(out / 'model_input.json', model)
            with self.assertRaisesRegex(ValueError, 'changed since rendering'):
                inputs.load_input(out)

    def test_missing_geometry_is_null_not_stationary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'dataset'
            sequence, frames, _, three = fixture(root)
            data = t1.jrdb.load_json(three)
            data['labels'][frames[-1] + '.pcd'] = []
            t1.write_json(three, data)
            obs = inputs.model_input(t1.build_window(root, sequence, frames, lidar=False))[0]['o']
            self.assertTrue(all(row[1][-1] is None for row in obs))

    def test_review_renderer_hashes_must_match_saved_implementation(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'dataset', Path(temp) / 'input'
            sequence, frames, _, _ = fixture(root)
            inputs.save_bundle(t1.build_window(root, sequence, frames, lidar=False), 'object', out)
            inputs.load_window(out)
            with patch.object(inputs, 'implementation_hashes', return_value={'different.py': '0' * 64}), \
                    self.assertRaisesRegex(ValueError, 'Rendering implementation changed'):
                inputs.load_window(out)
            manifest = t1.jrdb.load_json(out / 'manifest.json')
            del manifest['implementation_sha256']
            t1.write_json(out / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Rendering implementation changed'):
                inputs.load_window(out)

    def test_logical_jrdb_reference_survives_repository_relocation_and_checks_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'raw_dataset', Path(temp) / 'input'
            sequence, frames, _, _ = fixture(root)
            before = Path(temp) / 'repo_before/data/datasets/jrdb'
            after = Path(temp) / 'repo_after/data/datasets/jrdb'
            for alias in (before, after):
                alias.parent.mkdir(parents=True)
                alias.symlink_to(root, target_is_directory=True)
            with patch.object(inputs, 'dataset_path', return_value=before):
                window = t1.build_window(before, sequence, frames, lidar=False)
                self.assertEqual(window.root, before)
                manifest = inputs.save_bundle(window, 'object', out)
                self.assertEqual(manifest['dataset_root_ref'], 'jrdb')
            before.unlink()  # Only a synthetic fixture link, never the target.
            with patch.object(inputs, 'dataset_path', return_value=after):
                recovered, _, _ = inputs.load_window(out)
                self.assertEqual(recovered.root, after)
                self.assertEqual(recovered.geometry_signature, window.geometry_signature)
                changed = Path(temp) / 'changed_raw'
                fixture(changed)
                (changed / 'images/image_0' / sequence / (frames[0] + '.jpg')).write_bytes(b'TAMPERED_FIXTURE')
                after.unlink()
                after.symlink_to(changed, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'Source asset changed'):
                    inputs.load_window(out)

    def test_unknown_dataset_root_reference_is_not_a_path_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'dataset', Path(temp) / 'input'
            sequence, frames, _, _ = fixture(root)
            inputs.save_bundle(t1.build_window(root, sequence, frames, lidar=False), 'object', out)
            manifest = t1.jrdb.load_json(out / 'manifest.json')
            manifest['dataset_root_ref'] = '../outside'
            t1.write_json(out / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Unsupported JRDB dataset root reference'):
                inputs.load_window(out)

    def test_input_fingerprint_binds_exact_prompt_and_image_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'dataset', Path(temp) / 'bundle'
            sequence, _, _, _ = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.prepare(argparse.Namespace(root=root, out=out, scenes=[sequence], frames=[15], contact_sheet=False))
            before = api.load_bundle(out)[2][0]['fingerprint']
            system = out / 'system_prompt.txt'
            system.write_text(system.read_text() + ' changed', encoding='utf-8')
            manifest = t1.jrdb.load_json(out / 'manifest.json')
            manifest['files']['system_prompt.txt'] = t1.file_digest(system)
            t1.write_json(out / 'manifest.json', manifest)
            self.assertNotEqual(before, api.load_bundle(out)[2][0]['fingerprint'])

    def test_compare_accepts_matching_prediction_and_rejects_cross_input_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root, bundle = Path(temp) / 'dataset', Path(temp) / 'bundle'
            sequence, _, _, _ = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.prepare(argparse.Namespace(root=root, out=bundle, scenes=[sequence], frames=[15], contact_sheet=False))
            case = api.load_bundle(bundle)[2][0]
            row = {'case_id': case['entry']['case_id'], 'valid': True,
                   'input_fingerprint': case['fingerprint'], 'canonical_rel': {'g': [], 'u': case['model']['fields']['v']}}
            log = Path(temp) / 'requests.jsonl'
            log.write_text(json.dumps(row) + '\n', encoding='utf-8')
            args = argparse.Namespace(bundle=bundle, requests=log, out=Path(temp) / 'compare')
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.compare(args)
            self.assertTrue((args.out / f"{row['case_id']}.png").exists())
            row['input_fingerprint'] = 'wrong'
            log.write_text(json.dumps(row) + '\n', encoding='utf-8')
            args.out = Path(temp) / 'wrong'
            with self.assertRaisesRegex(ValueError, 'exact bundle inputs'):
                workflow.compare(args)
            self.assertFalse(args.out.exists())

    def test_compare_preflights_paths_duplicates_and_all_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root, bundle = Path(temp) / 'dataset', Path(temp) / 'bundle'
            sequence, _, _, _ = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.prepare(argparse.Namespace(root=root, out=bundle, scenes=[sequence], frames=[15], contact_sheet=False))
            case = api.load_bundle(bundle)[2][0]
            row = {'case_id': case['entry']['case_id'], 'valid': True,
                   'input_fingerprint': case['fingerprint'], 'canonical_rel': {'g': [], 'u': []}}
            log = Path(temp) / 'requests.jsonl'
            log.write_text(json.dumps(row) + '\n')
            for out in (root / 'forbidden', bundle / 'forbidden'):
                with self.subTest(out=out), self.assertRaises(ValueError):
                    workflow.compare(argparse.Namespace(bundle=bundle, requests=log, out=out))
                self.assertFalse(out.exists())
            out = Path(temp) / 'comparison'
            args = argparse.Namespace(bundle=bundle, requests=log, out=out)
            log.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate prediction'):
                workflow.compare(args)
            self.assertFalse(out.exists())
            wrong = {**row, 'case_id': 'unknown_case'}
            log.write_text(json.dumps(row) + '\n' + json.dumps(wrong) + '\n')
            with self.assertRaises(KeyError):
                workflow.compare(args)
            self.assertFalse(out.exists())
            log.write_text(json.dumps({'valid': False}) + '\n')
            with self.assertRaisesRegex(ValueError, 'No valid predictions'):
                workflow.compare(args)
            self.assertFalse(out.exists())

    def test_output_guard_resolves_symlinks_into_protected_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root, bundle = Path(temp) / 'dataset', Path(temp) / 'bundle'
            root.mkdir()
            bundle.mkdir()
            alias = Path(temp) / 'alias'
            alias.symlink_to(bundle, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'separate from model inputs'):
                t1.check_output(root, alias / 'new', input_dirs=(bundle,))
            self.assertFalse((bundle / 'new').exists())

    def test_old_profiles_rejected_and_unsupported_motion_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'dataset'
            sequence, frames, _, _ = fixture(root)
            window = t1.build_window(root, sequence, frames, lidar=False)
            with self.assertRaises(ValueError):
                inputs.model_input(window, 'spatial')
            window.scope = 'all-bev'
            with self.assertRaises(ValueError):
                inputs.model_input(window)

    def test_preparation_exports_only_whitelist_and_review_metadata_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'dataset', Path(temp) / 'bundle'
            sequence, _, _, _ = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.prepare(argparse.Namespace(root=root, out=out, scenes=[sequence], frames=[15], contact_sheet=False))
            manifest, system, cases = api.load_bundle(out)
            self.assertEqual((len(cases), len(manifest['files'])), (1, 5))
            self.assertEqual(len(cases[0]['paths']), 3)
            folder = out / 'cases' / f'{sequence}_000015'
            _, model, local = inputs.load_window(folder)
            self.assertEqual(local['input_fingerprint'], manifest['cases'][0]['input_fingerprint'])
            body = api.make_body(system, cases[0])
            for forbidden in ('pedestrian:', 'cluster_ID', 'native', 'input_fingerprint', 'reviewer'):
                self.assertNotIn(forbidden, json.dumps(body))
            self.assertTrue((folder / 'annotation_template.json').exists())
            self.assertTrue((out / 'references' / f'{sequence}_000015.native.json').exists())
            with self.assertRaises(ValueError):
                workflow.prepare(argparse.Namespace(root=root, out=out, scenes=[sequence], frames=[15], contact_sheet=False))

    def test_reviewed_draft_outputs_and_fingerprint_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / 'dataset', Path(temp) / 'input'
            sequence, frames, _, _ = fixture(root)
            inputs.save_bundle(t1.build_window(root, sequence, frames, lidar=False), 'object', out)
            annotation = out / 'annotation_template.json'
            args = argparse.Namespace(input_dir=out, labels=annotation, out=Path(temp) / 'gold', allow_draft=False)
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                workflow.reviewed(args)
            args.allow_draft = True
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.reviewed(args)
            self.assertTrue((args.out / 'draft.png').exists())
            self.assertFalse((args.out / 'assistant_response.json').exists())
            label = t1.jrdb.load_json(annotation)
            label.update(review_status='reviewed', reviewer='synthetic_fixture', reviewed_at='2026-10-05', label_provenance='unit_test_only')
            t1.write_json(annotation, label)
            args.out, args.allow_draft = Path(temp) / 'reviewed', False
            with contextlib.redirect_stdout(io.StringIO()):
                workflow.reviewed(args)
            self.assertTrue((args.out / 'assistant_response.json').exists())
            label['input_fingerprint'] = 'wrong'
            t1.write_json(annotation, label)
            args.out = Path(temp) / 'wrong'
            with self.assertRaisesRegex(ValueError, 'exact model inputs'):
                workflow.reviewed(args)


if __name__ == '__main__':
    unittest.main()
