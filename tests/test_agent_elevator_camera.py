"""Independent camera bridge checks with synthetic images and no ROS master."""
import io
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from decision.agent.adapters.elevator_camera import ElevatorCamera
from decision.agent.workflows.elevator_camera import image_record


class ElevatorCameraTests(unittest.TestCase):
    def record(self, stamp=10., sequence=0):
        return image_record(np.zeros((30,40,3),dtype=np.uint8),stamp_s=stamp,
                            frame_id='camera_color_optical_frame',sequence=sequence)

    def read(self, records):
        camera=ElevatorCamera()
        camera.process=SimpleNamespace(stdout=io.StringIO(''.join('FRAME '+json.dumps(x)+'\n' for x in records)))
        camera._read()
        return camera

    def test_independent_frame_preserves_camera_source_timestamp(self):
        camera=self.read([self.record(10.,0),self.record(11.,1)])
        result=camera.snapshot()
        self.assertEqual(result['status'],'available')
        self.assertEqual(result['record']['stamp_s'],11.)
        self.assertEqual(result['record']['clock_domain'],'ros_time')
        self.assertTrue(result['image'].startswith(b'\xff\xd8'))
        with patch('decision.agent.adapters.elevator_camera.time.monotonic',return_value=camera.latest['received_mono']+3.):
            self.assertEqual(camera.snapshot()['reason'],'camera_frame_stale')

    def test_invalid_or_regressing_frames_are_not_used_as_current(self):
        for records in ([self.record(10.,0),self.record(9.,1)],
                        [self.record(10.,0),self.record(11.,0)]):
            self.assertEqual(self.read(records).snapshot()['reason'],'invalid_camera_stream')
        bad=self.record();bad['image_base64']='not-valid'
        self.assertEqual(self.read([bad]).snapshot()['status'],'unavailable')
