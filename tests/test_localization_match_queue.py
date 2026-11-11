"""Online matching drops queued old scans without changing their timestamps."""
import queue
import unittest
from perception.mapping_localization.runtime import LocalizationSession


class MatchQueueTests(unittest.TestCase):
    @staticmethod
    def job(stamp):
        return ('scan',None,None,None,dict(stamp_s=stamp))

    def test_live_localization_replaces_oldest_unprocessed_scan(self):
        session=object.__new__(LocalizationSession)
        session.latest_matching=True;session.jobs=queue.Queue(2);session.metrics=None
        for stamp in (1.,2.,3.):session.submit_match(self.job(stamp))
        self.assertEqual([session.jobs.get_nowait()[4]['stamp_s'] for _ in range(2)],[2.,3.])

    def test_mapping_and_replay_keep_existing_overflow_behavior(self):
        session=object.__new__(LocalizationSession)
        session.latest_matching=False;session.jobs=queue.Queue(1);session.metrics=None
        session.submit_match(self.job(1.))
        with self.assertRaisesRegex(RuntimeError,'backlog exceeded'):
            session.submit_match(self.job(2.))
        self.assertEqual(session.jobs.get_nowait()[4]['stamp_s'],1.)
