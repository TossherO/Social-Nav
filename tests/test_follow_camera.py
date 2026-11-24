import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from runtime.follow_camera import CameraPairClient, CameraPairServer, FollowCameraHistory
from runtime.decision import DecisionBuffer
from tests.decision_fixtures import observation


def msg(stamp, **values):
    header=SimpleNamespace(stamp=SimpleNamespace(to_nsec=lambda:stamp),frame_id='camera')
    return SimpleNamespace(header=header,width=2,height=1,**values)


class FollowCameraTests(unittest.TestCase):
    def test_sensor_receipt_budget_and_causal_image_and_intrinsics(self):
        history=FollowCameraHistory('/scan',live=False)
        t=1_000_000_000
        history.observe('/scan',msg(t),t)
        old=msg(t-60_000_000)
        chosen=msg(t-25_000_000)
        info=msg(t-30_000_000)
        history.observe('/camera/color/camera_info',info,t)
        history.observe('/camera/color/image_raw',old,t)
        history.observe('/camera/color/image_raw',chosen,t+30_000_000)
        history.observe('/camera/color/image_raw',msg(t+1),t+30_000_000)
        history.observe('/camera/color/camera_info',msg(t-1),t+30_000_000)
        history.observe('/camera/color/image_raw',msg(t),t+30_000_001)
        self.assertEqual(history.pair(t),(chosen,info))
        history.images.clear()
        self.assertIsNone(history.pair(t))

    def test_camera_reply_is_owned_and_matches_requested_scan(self):
        history=FollowCameraHistory('/scan',live=False)
        t=1_000_000_000
        image=msg(t-20,encoding='rgb8',step=6,data=bytes(range(6)))
        info=msg(t-20,K=[1.]*9,D=[],distortion_model='none')
        history.observe('/scan',msg(t),t)
        history.observe('/camera/color/camera_info',info,t)
        history.observe('/camera/color/image_raw',image,t+1)
        reader,writer=socket.socketpair()
        server=CameraPairServer(writer,history,1024)
        client=CameraPairClient(reader)
        try:
            pixels,meta=client.lookup(t)
            self.assertIsNone(client.lookup(t+1))
            self.assertEqual(meta['rgb_stamp_ns'],t-20)
            np.testing.assert_equal(pixels.ravel(),np.arange(6))
        finally:
            client.close()
            server.close()

    def test_start_stamped_scan_keeps_arrival_deadline_in_live_mode(self):
        history=FollowCameraHistory('/scan',live=True)
        start, end, received = 2_000_000_000, 2_100_000_000, 9_000_000_000
        history.observe('/scan',msg(start),received)
        history.set_scan_end(start,end)
        image=msg(end-50_000_000)
        info=msg(end-50_000_000)
        def deliver(timeout):
            self.assertAlmostEqual(timeout,.01)
            history.observe('/camera/color/image_raw',image,received+25_000_000)
            history.observe('/camera/color/camera_info',info,received+30_000_000)
        with patch('runtime.follow_camera.time.monotonic_ns',return_value=received+20_000_000), \
                patch.object(history.changed,'wait',side_effect=deliver) as wait:
            self.assertEqual(history.pair(end),(image,info))
            wait.assert_called_once()
            self.assertIsNone(history.pair(start))
        history.images.clear()
        with patch('runtime.follow_camera.time.monotonic_ns',return_value=received+30_000_000), \
                patch.object(history.changed,'wait') as wait:
            self.assertIsNone(history.pair(end))
            wait.assert_not_called()

    def test_preview_repairs_only_latest_frame_and_preserves_fallback(self):
        calls=[]
        def lookup(ns):
            calls.append(ns)
            return None
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768,camera_lookup=lookup)
        try:
            for i in range(12):
                stamp=10+i/10
                row,arr,_,scene=observation(stamp)
                rgb_ns=round(stamp*1e9)-20_000_000
                info=dict(height=48,width=64,stamp_ns=rgb_ns,frame_id='camera')
                if i==11:info['height']=12
                buffer.push(row,arr,None,scene=scene,image_metadata=dict(
                    rgb_stamp_ns=rgb_ns,rgb_frame='camera',camera_info=info))
            lease=buffer.acquire()
            with buffer.consume(lease,latest_only=True) as read:
                entry=lease['entries'][-1]
                self.assertTrue((read(entry['buffer'])['rgb']==120).all())
            self.assertEqual(calls,[entry['stamp_ns']])
            with buffer.consume(buffer.acquire()):pass
            self.assertEqual(calls,[entry['stamp_ns']]*2)
        finally:buffer.close()

    def test_repaired_window_is_resealed_without_overwriting_scan_history(self):
        used=[]
        def lookup(ns):
            used.append(ns)
            return np.ones((48,64,3),np.uint8),dict(rgb_stamp_ns=ns-20_000_000)
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768,camera_lookup=lookup)
        try:
            for i in range(12):
                stamp=10+i/10
                row,arr,_,scene=observation(stamp)
                del arr['rgb']
                buffer.push(row,arr,None,scene=scene)
            lease=buffer.acquire()
            original=lease['entries']
            with buffer.consume(lease) as read:
                for entry in lease['entries']:
                    self.assertEqual(entry['image_metadata']['rgb_stamp_ns'],entry['stamp_ns']-20_000_000)
                    self.assertTrue(read(entry['buffer'])['rgb'].all())
            self.assertEqual(used,[e['stamp_ns'] for e in original])
            self.assertTrue(all('rgb' not in dict(e['buffer'].arrays) for e in buffer.frames))
        finally:buffer.close()
