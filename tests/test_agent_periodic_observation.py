"""Retention policies are independent of the monitored task."""
import unittest

from decision.agent.core.periodic_observation import PeriodicObservation


class PeriodicObservationTests(unittest.TestCase):
    def observer(self,policy):
        return PeriodicObservation(None,None,interval_s=1.,history_limit=2,history_policy=policy)

    def test_none_bounded_and_agent_retention(self):
        for policy in ('none','bounded','agent'):
            observer=self.observer(policy)
            for seq,retain in enumerate((True,False,True)):
                observer.remember(dict(sequence=seq),bytes([seq]),dict(retain_observation=retain))
            self.assertEqual([x['record']['sequence'] for x in observer.history],
                             {'none':[],'bounded':[1,2],'agent':[0,2]}[policy])
            observer.clear();self.assertEqual(observer.history,[])
        with self.assertRaises(ValueError):self.observer('agent').remember({},b'',{})
