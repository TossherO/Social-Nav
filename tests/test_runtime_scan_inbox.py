"""Dropped scan leases and localization barriers survive latest-only reception."""
import queue
import threading
import unittest

from runtime.scan_inbox import LatestScanInbox


class Wire:
    def __init__(self):self.messages=queue.Queue();self.acks=[];self.socket=self
    def recv(self):
        result=self.messages.get(timeout=3)
        if result is None:raise EOFError()
        return result
    def send(self,value):self.acks.append(value)
    def settimeout(self,value):pass
    def shutdown(self,value):self.messages.put(None)


def scan(index,*,health=0,generation=0):
    return dict(kind='scan',stamp_s=float(index),buffer=dict(slot=index,epoch=1),
        state=dict(health=health,session_id='test',generation=generation,frame_id='odom'))


class ScanInboxTests(unittest.TestCase):
    def test_replacement_releases_only_pending_and_preserves_every_pose(self):
        wire=Wire();observed=[];complete=threading.Event()
        def observe(packet):
            observed.append(packet['stamp_s'])
            if len(observed)==3:complete.set()
        inbox=LatestScanInbox(wire,observe,lambda:None,10.)
        try:
            for i in (1,2,3):wire.messages.put(scan(i))
            self.assertTrue(complete.wait(3));packet=inbox.recv()
            self.assertEqual(packet['stamp_s'],3.);self.assertEqual(observed,[1.,2.,3.])
            self.assertEqual([r['slot'] for r in wire.acks],[1,2])
            inbox.release(packet);wire.messages.put(dict(kind='end'))
            self.assertEqual(inbox.recv()['kind'],'end')
            self.assertEqual([r['slot'] for r in wire.acks],[1,2,3]);self.assertEqual(inbox.replaced,2)
        finally:inbox.close()

    def test_skipped_invalid_pose_or_lifecycle_change_blocks_old_map_submission(self):
        for change in ('failure','generation'):
            wire=Wire();invalid=[];seen=threading.Event()
            inbox=LatestScanInbox(wire,lambda p:seen.set(),lambda:invalid.append(True),10.)
            try:
                wire.messages.put(scan(1));old=inbox.recv();inbox.release(old);seen.clear()
                wire.messages.put(scan(2,health=1 if change=='failure' else 0,generation=1 if change=='generation' else 0))
                self.assertTrue(seen.wait(3));new=inbox.recv();inbox.release(new)
                submitted=[];inbox.submit_if_current(old,lambda:submitted.append('old'))
                inbox.submit_if_current(new,lambda:submitted.append('new'))
                self.assertEqual(submitted,['new']);self.assertEqual(invalid,[True])
            finally:inbox.close()

    def test_receiver_failure_reaches_consumer(self):
        wire=Wire();inbox=LatestScanInbox(wire,lambda p:None,lambda:None,10.)
        wire.messages.put(None)
        try:
            with self.assertRaisesRegex(RuntimeError,'EOFError'):inbox.recv()
        finally:inbox.close()
