"""Unconfirmed live scans are display-only and confirmation is freshness-bound."""
import math
import types
import unittest

import numpy as np

from runtime.initial_preview import PreviewState, preview_points


def cloud(stamp=10., frame='rslidar'):
    points=np.array([[1.,0.,0.],[0.,1.,0.]],dtype='<f4')
    return types.SimpleNamespace(header=types.SimpleNamespace(frame_id=frame,
        stamp=types.SimpleNamespace(to_sec=lambda:stamp)),fields=[types.SimpleNamespace(
            name=axis,datatype=7,count=1,offset=i*4) for i,axis in enumerate('xyz')],
        is_bigendian=False,width=2,height=1,point_step=12,row_step=24,data=points.tobytes())


class InitialPreviewTests(unittest.TestCase):
    def test_candidate_applies_lidar_pose_only_to_display_points(self):
        actual=preview_points(cloud(),[0.,0.,math.pi/2,2.,3.,1.],'rslidar')
        np.testing.assert_allclose(actual,[[2.,4.,1.],[1.,3.,1.]],atol=1e-6)
        with self.assertRaisesRegex(ValueError,'frame'):
            preview_points(cloud(frame='other'),[0.,0.,0.,0.,0.,0.],'rslidar')

    def test_confirmation_requires_recent_render_of_latest_candidate(self):
        p=PreviewState(max_age_s=1.)
        p.scan(cloud(),5.);p.select([0.,0.,0.,1.,2.,3.],10.)
        self.assertFalse(p.ready(5.1))
        p.rendered=(p.revision,p.source_stamp,p.received_mono)
        self.assertTrue(p.ready(5.1))
        self.assertFalse(p.ready(6.1))
        self.assertFalse(p.scan(cloud(),6.2))  # A repeated source stamp does not refresh it.
        self.assertFalse(p.ready(6.2))
        p.scan(cloud(stamp=11.),6.3)
        self.assertFalse(p.ready(6.3))  # The last rendered source has expired.
        p.rendered=(p.revision,p.source_stamp,p.received_mono)
        self.assertTrue(p.ready(6.4))
        p.select([0.,0.,0.,4.,5.,6.],12.)
        self.assertFalse(p.ready(6.4))  # Reselection cannot confirm the old preview.
        with self.assertRaisesRegex(ValueError,'measurement time'):
            p.scan(cloud(stamp=12.),6.5,now_source=14.)


if __name__ == '__main__': unittest.main()
