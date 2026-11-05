"""Map backlog, failed localization and worker exceptions have bounded effects."""
import threading
import unittest

from runtime.environment import LatestMapWorker


class LatestMapTests(unittest.TestCase):
    def setup_worker(self,process):
        self.published=[];self.resets=[]
        worker=LatestMapWorker(process,self.published.append,lambda:self.resets.append(True))
        return worker

    def test_one_active_one_latest_pending_keeps_original_observation(self):
        entered=threading.Event();release=threading.Event()
        def process(job):
            if job['stamp']==1:entered.set();self.assertTrue(release.wait(3))
            return job
        worker=self.setup_worker(process)
        try:
            worker.submit(dict(stamp=1),identity='a');self.assertTrue(entered.wait(3))
            worker.submit(dict(stamp=2),identity='a');worker.submit(dict(stamp=3),identity='a')
            release.set();worker.flush(3)
            self.assertEqual([r['stamp'] for r in self.published],[1,3])
            self.assertEqual(worker.counts['replaced'],1);self.assertEqual(len(self.resets),1)
            self.assertEqual(worker.latest(),dict(stamp=3))
        finally:release.set();worker.close()

    def test_invalid_pose_and_new_lifecycle_discard_inflight_result(self):
        for lifecycle_change in (False,True):
            entered=threading.Event();release=threading.Event()
            def process(job):
                if job==1:entered.set();self.assertTrue(release.wait(3))
                return job
            worker=self.setup_worker(process)
            try:
                worker.submit(1,identity='a');self.assertTrue(entered.wait(3))
                if not lifecycle_change:worker.invalidate()
                worker.submit(2,identity='b' if lifecycle_change else 'a')
                release.set();worker.flush(3)
                self.assertEqual(self.published,[None,2]);self.assertEqual(len(self.resets),2)
                self.assertEqual(worker.counts['discarded'],1)
            finally:release.set();worker.close()

    def test_failure_is_reported_to_owner_and_close(self):
        def fail(job):raise ValueError('bad map input')
        worker=self.setup_worker(fail);worker.submit(1,identity='a')
        with self.assertRaisesRegex(RuntimeError,'bad map input'):worker.flush(3)
        with self.assertRaisesRegex(RuntimeError,'bad map input'):worker.close()
