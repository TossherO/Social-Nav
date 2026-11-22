"""Optional timing must preserve source times and report bounded-record loss."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from runtime.metrics import Metrics
from runtime.timing_report import map_and_goal_timings


class TimingTraceTests(unittest.TestCase):
    def test_first_use_matches_goal_revision_and_counts_a_grid_once(self):
        def cycle(t, revision, ready, epoch=0):
            return dict(event='control_cycle', scene_binding='grid', now=50+t, measurement_stamp_s=50.,
                        reference_epoch=epoch, goal_key=['task','request',revision],
                        reference_goal_key=None if ready is None else ['task','request',ready],
                        goal_checked_mono=t, reference_checked_mono=t+.01)
        report=map_and_goal_timings([
            cycle(.1,1,None),cycle(.2,2,1),cycle(.3,2,1),
            cycle(.4,1,None,epoch=1),cycle(.5,1,1,epoch=1)])
        self.assertEqual(report['observed_goal_revisions'],3)
        self.assertEqual(report['goal_to_first_available_reference_ms']['n'],2)
        self.assertAlmostEqual(report['goal_to_first_available_reference_ms']['p50'],110.)
        self.assertEqual(report['map_source_to_first_control_ms']['n'],1)
        self.assertAlmostEqual(report['map_source_to_first_control_ms']['p50'],100.)

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
