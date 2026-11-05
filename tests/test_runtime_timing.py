"""Optional timing must preserve source times and report bounded-record loss."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from runtime.metrics import Metrics


class TimingTraceTests(unittest.TestCase):
    def test_disabled_trace_does_not_retain_events(self):
        metrics=Metrics()
        metrics.trace('observation',measurement_stamp_s=123.)
        self.assertEqual(len(metrics.trace_rows),0)
        self.assertIsNone(metrics.write_trace())

    def test_concurrent_trace_preserves_measurements_and_reports_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'timing.jsonl';metrics=Metrics(trace_path=path,trace_limit=7)
            def produce(source):
                for i in range(20):metrics.trace('observation',source=source,measurement_stamp_s=100.+i)
            workers=[threading.Thread(target=produce,args=(source,)) for source in ('a','b')]
            for worker in workers:worker.start()
            for worker in workers:worker.join()
            self.assertFalse(path.exists())
            metrics.write_trace()
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            report=json.loads(path.with_suffix('.summary.json').read_text())
            self.assertEqual((report['written'],report['dropped'],report['complete']),(7,33,False))
            self.assertTrue(all(100.<=row['measurement_stamp_s']<120. for row in rows))
            self.assertEqual([r['monotonic_s'] for r in rows],sorted(r['monotonic_s'] for r in rows))
            with self.assertRaises(FileExistsError):metrics.write_trace()


if __name__=='__main__':unittest.main()
