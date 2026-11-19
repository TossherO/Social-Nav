"""Independent camera bridge checks with synthetic images and no ROS master."""
import io
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from decision.agent.adapters.elevator_camera import ElevatorCamera
from decision.agent.workflows.elevator_camera import image_record,scan_record


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

    def test_lidar_and_rgb_keep_distinct_source_times_and_frames(self):
        fields=[SimpleNamespace(name=n,offset=i*4,count=1,datatype=7) for i,n in enumerate('xyz')]
        cloud=SimpleNamespace(fields=fields,width=1,height=1,point_step=12,row_step=12,is_bigendian=False,
            data=np.array([1.,2.,.8],dtype='<f4').tobytes(),
            header=SimpleNamespace(frame_id='lidar',stamp=SimpleNamespace(to_sec=lambda:10.5)))
        with self.assertRaises(ValueError):scan_record(cloud,np.eye(4),'wrong',1)
        scan=scan_record(cloud,np.eye(4),'lidar',1)
        camera=ElevatorCamera()
        camera.process=SimpleNamespace(stdout=io.StringIO('SCAN '+json.dumps(scan)+'\nFRAME '+json.dumps(self.record(11))+'\n'))
        camera._read()
        frame=camera.snapshot()
        self.assertEqual((frame['record']['stamp_s'],frame['scan_record']['stamp_s']),(11.,10.5))
        self.assertEqual(frame['scan_record']['display_frame'],'base_link')
        self.assertEqual(frame['base_xy_dm'],[[10,20]])
        self.assertNotIn('base_xy_dm',frame['scan_record'])
        self.assertTrue(frame['scan_image'].startswith(b'\x89PNG'))

    def test_failed_pose_overlay_rotates_the_original_scan_into_map_coordinates(self):
        from PIL import Image
        from decision.agent.tools.spatial import compact_map_layout
        from decision.agent.workflows.floor_handoff import scan_alignment_preview
        bounds=[0,0,10,10]
        layout=compact_map_layout(bounds,max_edge=1024)
        source=io.BytesIO()
        Image.new('RGB',layout['size'],'white').save(source,format='PNG')
        data=scan_alignment_preview(source.getvalue(),bounds,[5.,4.,np.pi/2],[[10,20]])
        pixel=(round(layout['left']+3*layout['scale']),round(layout['top']+5*layout['scale']))
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.getpixel(pixel),(192,26,153))

    def test_invalid_or_regressing_frames_are_not_used_as_current(self):
        for records in ([self.record(10.,0),self.record(9.,1)],
                        [self.record(10.,0),self.record(11.,0)]):
            self.assertEqual(self.read(records).snapshot()['reason'],'invalid_camera_stream')
        bad=self.record();bad['image_base64']='not-valid'
        self.assertEqual(self.read([bad]).snapshot()['status'],'unavailable')
