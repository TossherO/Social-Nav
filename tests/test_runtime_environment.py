"""Map backlog, failed localization and worker exceptions have bounded effects."""
import threading
import unittest

from runtime.environment import EnvironmentFault, LatestMapWorker


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

    def test_budget_fault_unpublishes_the_old_map_and_worker_recovers(self):
        fault=EnvironmentFault(2,'odom',('a',0),1,2)
        worker=self.setup_worker(lambda job: {1:'old',2:fault,3:None,4:'new'}[job])
        try:
            for job, latest in ((1,'old'),(2,None),(3,None),(4,'new')):
                worker.submit(job,identity='a');worker.flush(3)
                self.assertEqual(worker.latest(),latest)
            self.assertEqual(self.published,['old',fault,None,'new'])
            self.assertEqual(worker.counts['faults'],1)
            self.assertEqual(worker.counts['published'],2)
        finally:worker.close()

    def test_reset_blocks_late_agent_observations_from_the_old_map(self):
        from runtime.command import AgentState
        state=AgentState()
        packet=dict(stamp_s=1.,clock_domain='runtime_live')
        control=dict(session_id='a',generation=0,frame_id='odom',health=0)
        old=dict(local_identity=['a',0,'odom'],stamp_ns=1_000_000_000)
        fault=EnvironmentFault(2_000_000_000,'odom',('a',0),1,2)
        state.publish_control(packet,control,manual_preview=dict(reference_path=[[0,0],[1,0]]))
        state.publish_observation(old,b'old')
        state.invalidate_environment(fault,1)
        state.publish_control(packet,control)
        value=state.snapshot(with_image=True)
        self.assertEqual(value['control']['environment_fault'],fault.reason)
        self.assertIsNone(value['control']['manual_preview'])
        self.assertIsNone(value['observation'])
        state.publish_control(dict(packet,stamp_s=3.),control,environment_epoch=1)
        state.publish_observation(old,b'late')
        self.assertIsNone(state.snapshot()['observation'])
        state.publish_observation(dict(old,stamp_ns=3_000_000_000),b'fresh')
        self.assertEqual(state.snapshot()['observation']['stamp_ns'],3_000_000_000)
