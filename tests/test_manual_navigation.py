"""Manual target lifecycle and legacy protocol boundaries without any hardware."""
import math
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace

from navigation.planning.manual_goal import ManualGoals
from navigation.control.chassis_protocol import encode_command,decode_feedback,FeedbackFrames


class ManualGoalTests(unittest.TestCase):
    def make_bridge(self,**kwargs):
        self.sent=[]
        def send(kind,value):
            self.sent.append((kind,value))
            return dict(accepted=True,executable=False,execution_feedback=False)
        return ManualGoals(send,task_id='manual-test',**kwargs)

    def pose(self,bridge,**changes):
        values=dict(frame=bridge.frame,position=[2.,3.,0.],quaternion=[0.,0.,math.sin(.3),math.cos(.3)],stamp_s=12.)
        values.update(changes)
        return bridge.goal(**values)

    def test_mapped_goal_retains_heading_source_time_and_map_binding(self):
        bridge=self.make_bridge(frame='map',map_snapshot_id='verified-map')
        result=self.pose(bridge);request=self.sent[-1][1]
        self.assertEqual(request['goal'][:2],[2.,3.]);self.assertAlmostEqual(request['goal'][2],.6)
        self.assertEqual(request['map_snapshot_id'],'verified-map')
        self.assertEqual(request['target_reference'],'base_link');self.assertEqual(request['goal_behavior'],'stop')
        self.assertEqual(request['source_intent_stamp_s'],12.)
        self.assertFalse(result['executable']);self.assertFalse(result['task_completed'])

    def test_replacement_and_cancel_use_the_last_acknowledged_revision(self):
        bridge=self.make_bridge();self.pose(bridge);self.pose(bridge,position=[4.,3.,0.])
        self.assertEqual([r['goal_revision'] for _,r in self.sent],[1,2])
        bridge.cancel();self.assertEqual(self.sent[-1],('cancel',dict(task_id='manual-test',goal_revision=2)))
        with self.assertRaises(ValueError):bridge.cancel()

    def test_cancel_covers_latest_attempt_after_a_negative_acknowledgement(self):
        bridge=self.make_bridge();self.pose(bridge)
        original=bridge.send
        bridge.send=lambda *args:dict(accepted=False,error='cancelled_goal_revision')
        self.pose(bridge);bridge.send=original;bridge.cancel()
        self.assertEqual(self.sent[-1][1]['goal_revision'],2)

    def test_lost_acknowledgement_can_still_cancel_the_attempted_revision(self):
        bridge=self.make_bridge();self.pose(bridge);original=bridge.send
        def lost(*args):raise TimeoutError('Runtime may already have received the goal')
        bridge.send=lost
        with self.assertRaises(TimeoutError):self.pose(bridge)
        bridge.send=original;bridge.cancel()
        self.assertEqual(self.sent[-1][1]['goal_revision'],2)

    def test_invalid_frames_and_orientations_never_reach_the_runtime(self):
        bridge=self.make_bridge()
        for change in (dict(frame='map'),dict(position=[float('nan'),0,0]),dict(stamp_s=-1),
                       dict(quaternion=[0,0,0,0]),dict(quaternion=[1,0,0,0]),dict(quaternion=[0,0,0,2])):
            with self.subTest(change=change),self.assertRaises(ValueError):self.pose(bridge,**change)
        self.assertEqual(self.sent,[])
        with self.assertRaises(ValueError):self.make_bridge(frame='map')

    def test_real_command_socket_accepts_goal_then_cancel_without_execution(self):
        from runtime.command import CommandInbox,send_request
        with tempfile.TemporaryDirectory() as directory:
            events=[];path=Path(directory)/'commands.sock'
            inbox=CommandInbox(path,SimpleNamespace(command=lambda *v:events.append(v)))
            try:
                bridge=ManualGoals(lambda kind,value:send_request(path,kind,value,timeout=1),task_id='socket-test')
                self.assertTrue(self.pose(bridge)['reply']['accepted'])
                self.assertTrue(bridge.cancel()['reply']['accepted'])
                self.assertEqual([k for k,_ in inbox.drain()],['goal','cancel'])
                self.assertEqual([k for k,_ in events],['goal','cancel'])
            finally:inbox.close()


class LegacyProtocolTests(unittest.TestCase):
    def feedback(self):
        return b'\xbf\x01'+struct.pack('>2h2i2HBHB2h2hBB',-200,150,1200,-3400,6280,80,0,1,3,120,-30,9,-8,0,77)

    def test_command_bytes_match_driver_arithmetic_including_negative_fields(self):
        frame=encode_command(-.2,.15,mode=3,led=2)
        self.assertEqual(len(frame),12)
        self.assertEqual(frame.hex(),'af01ff3800960305dc00025d')
        # The inspected code sums signed field values, not individual bytes or LED.
        self.assertNotEqual(frame[-1],sum(frame[:-1])&255)

    def test_overflow_nonfinite_and_invalid_bytes_fail_without_clipping(self):
        for velocity in (float('nan'),float('inf'),32.768,-32.769,True):
            with self.subTest(velocity=velocity),self.assertRaises(ValueError):encode_command(velocity,0,mode=1)
        with self.assertRaises(ValueError):encode_command(0,0,mode=256)
        with self.assertRaises(ValueError):encode_command(0,0,mode=1,led=-1)

    def test_feedback_remains_unverified_and_has_no_fabricated_measurement_time(self):
        row=decode_feedback(self.feedback())
        self.assertEqual(row['velocity_mps_rps'],[-.2,.15])
        self.assertEqual(row['encoder_pose_m_m_rad'],[1.2,-3.4,6.28])
        self.assertEqual(row['mode_raw'],3);self.assertEqual(row['joystick_axes_raw'],[9,-8])
        self.assertIsNone(row['device_measurement_stamp_s'])
        self.assertFalse(row['checksum_verified']);self.assertFalse(row['execution_feedback'])
        self.assertEqual(row['status'],'unverified_protocol')
        corrupt=bytearray(self.feedback());corrupt[-1]^=255
        self.assertFalse(decode_feedback(corrupt)['checksum_verified'])

    def test_fragmented_stream_noise_and_partial_tail(self):
        parser=FeedbackFrames();frame=self.feedback()
        self.assertEqual(parser.feed(b'noise\xbf'),[])
        self.assertEqual(parser.feed(frame[1:8]),[])
        self.assertEqual(len(parser.feed(frame[8:]+frame+b'\xbf')),2)
        self.assertEqual(bytes(parser.pending),b'\xbf');self.assertEqual(parser.discarded,5)
        self.assertEqual(parser.feed(b'x'*10000),[]);self.assertLessEqual(len(parser.pending),1)

    def test_short_spreadsheet_layout_is_not_silently_accepted(self):
        for frame in (self.feedback()[:22],b'\xaf\x01'+self.feedback()[2:]):
            with self.assertRaises(ValueError):decode_feedback(frame)


if __name__=='__main__':unittest.main()
