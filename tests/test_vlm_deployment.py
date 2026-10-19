"""Read-only result comparison; zero installation, GPU or network calls."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from decision.vlm.deployment import compare_results


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
