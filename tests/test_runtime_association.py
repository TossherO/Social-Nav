"""Fresh associations remain causal, confirmed and tied to the original pose."""
import copy
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from perception.mapping_localization.context import LocalizationContext
from perception.mapping_localization.runtime import LocalizationSession
from runtime.association import AssociationInbox
from runtime.metrics import Metrics
from tests.test_environment import pose,match


class AssociationRefreshTests(unittest.TestCase):
    def test_confirmation_uses_consecutive_matches_after_failure(self):
        context=LocalizationContext('roll_fast_lio',allow_local_only=True,confirm_matches=3)
        valid=[]
        for stamp,health in ((1.,0),(1.1,2),(1.2,0),(1.3,0),(1.4,0)):
            context.update(pose(stamp));context.add_match(match(stamp,health=health))
            valid.append(context.refresh()[1]['valid'])
        self.assertEqual(valid,[False,False,False,False,True])

    def test_match_worker_publishes_confirmed_result_without_another_frontend_frame(self):
        class Frontend:
            def __init__(self,*args):pass
            def step(self,imu,points,begin,end):
                transform=np.eye(4);transform[0,3]=end
                return dict(world_from_lidar=transform,stamp_s=end,valid=True,reason=0,initial_lidar_pose=np.eye(4)),np.zeros((4,5))
            def close(self):pass
        class Matcher:
            def __init__(self,*args):pass
            def step(self,cloud,pose,reference,stamp):
                transform=pose.copy();transform[0,3]+=20
                return dict(available=True,valid=stamp<1.2,map_from_lidar=transform,map_revision=0)
            def close(self):pass
        updates=[];ready=threading.Event()
        def publish(value):updates.append(value);ready.set()
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.yaml';config.write_text('{}\n')
            with patch('perception.mapping_localization.runtime.NativeFrontend',Frontend),patch('perception.mapping_localization.runtime.NativeRoll',Matcher):
                session=LocalizationSession(config,mode='mapping',allow_local_only=True,association_ready=publish)
                try:
                    self.assertEqual(session.context.max_map_age, .5)
                    for stamp in (1.,1.1,1.2):
                        ready.clear()
                        session.step(dict(imu=None,points=None,begin=stamp-.1,end=stamp,sensor_frame='lidar'))
                        self.assertTrue(ready.wait(3))
                        self.assertEqual(updates[-1]['stamp_s'],stamp)
                        self.assertEqual(updates[-1]['measurement_stamp_s'],stamp)
                    self.assertEqual([u['valid'] for u in updates],[False,True,False])
                    self.assertEqual(updates[1]['map_from_local'][0][3],20.)
                finally:session.close()

    def test_same_pose_refresh_requires_confirmation_and_keeps_source_time(self):
        context=LocalizationContext('roll_fast_lio',allow_local_only=True)
        original=context.update(pose(1))[1];context.add_match(match(1))
        self.assertFalse(context.refresh()[1]['valid'])
        context.update(pose(1.1));context.add_match(match(1.1))
        state,new=context.refresh()
        self.assertTrue(new['valid']);self.assertEqual(new['measurement_stamp_s'],1.1)
        self.assertEqual(state['stamp_s'],1.1);self.assertEqual(context.last_stamp,1.1)
        self.assertFalse(original['valid']);self.assertEqual(len(context.history),2)
        self.assertIsNone(context.update(pose(1.1)))
        context.add_match(match(1.2))
        self.assertEqual(context.refresh()[1]['measurement_stamp_s'],1.1)

    def test_failed_match_invalid_pose_and_retired_session_cannot_restore_old_association(self):
        context=LocalizationContext('roll_fast_lio',allow_local_only=True,confirm_matches=1)
        context.update(pose(1));context.add_match(match(1));self.assertTrue(context.refresh()[1]['valid'])
        context.add_match(match(1,health=2));self.assertFalse(context.refresh()[1]['valid'])
        context.update(pose(1.1,health=2));context.add_match(match(1.1))
        self.assertFalse(context.refresh()[1]['valid'])
        context.update(pose(1.2,generation=1));context.add_match(match(1.1))
        self.assertFalse(context.refresh()[1]['valid'])

    def test_refresh_checks_coverage_for_the_original_pose(self):
        class Coverage:
            fingerprint='test'
            def contains(self,position,frame):return position[0]<22
        context=LocalizationContext('roll_fast_lio',allow_local_only=True,confirm_matches=1,coverage=Coverage())
        context.update(pose(1));context.add_match(match(1));self.assertTrue(context.refresh()[1]['valid'])
        context.update(pose(3));context.add_match(match(3))
        self.assertFalse(context.refresh()[1]['valid'])
        self.assertEqual(context.refresh()[1]['coverage_status'],'OUTSIDE_COVERAGE')


class AssociationInboxTests(unittest.TestCase):
    def setUp(self):
        self.reader,self.writer=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
        self.inbox=AssociationInbox(self.reader,Metrics())
        self.original=dict(stamp_s=2.,measurement_stamp_s=1.9,valid=True,local_session_id='s',
            local_generation=0,local_frame='odom',map_id='m',asset_fingerprint='asset')

    def tearDown(self):self.inbox.close();self.writer.close()
    def send(self,**fields):self.writer.send(json.dumps(dict(self.original,**fields)).encode())

    def test_newer_result_and_failure_apply_without_mutating_the_original(self):
        frozen=copy.deepcopy(self.original)
        self.send(measurement_stamp_s=2.)
        fresh=self.inbox.select(self.original);self.assertEqual(fresh['measurement_stamp_s'],2.)
        self.send(measurement_stamp_s=2.,valid=False)
        self.assertFalse(self.inbox.select(self.original)['valid'])
        self.assertEqual(self.original,frozen);self.assertTrue(fresh['valid'])

    def test_future_wrong_frame_generation_and_asset_are_not_substituted(self):
        for fields in (dict(measurement_stamp_s=2.1),dict(stamp_s=2.1,measurement_stamp_s=2.),
                       dict(local_generation=1),dict(local_frame='other'),dict(asset_fingerprint='other')):
            self.inbox.cache.clear();self.send(**fields)
            self.assertIs(self.inbox.select(self.original),self.original)

    def test_older_late_result_cannot_replace_a_newer_failure_and_cache_is_bounded(self):
        self.send(measurement_stamp_s=2.,valid=False);self.send(measurement_stamp_s=1.9)
        self.assertFalse(self.inbox.select(self.original)['valid'])
        for i in range(80):
            self.send(stamp_s=3.+i,measurement_stamp_s=3.+i)
            self.inbox.select(self.original)
        self.assertEqual(len(self.inbox.cache),64)


if __name__=='__main__':unittest.main()
