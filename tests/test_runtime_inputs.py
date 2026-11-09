"""Time failures retain source evidence; no ROS master or native inference."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from perception.sensors.observations import bag_observations,selected_messages


class ImuTimeTests(unittest.TestCase):
    def messages(self,stamps):
        for index,ns in enumerate(stamps):
            stamp=SimpleNamespace(to_sec=lambda ns=ns:ns/1.e9,to_nsec=lambda ns=ns:ns)
            vector=SimpleNamespace(x=0.,y=0.,z=0.)
            msg=SimpleNamespace(header=SimpleNamespace(stamp=stamp,seq=index),linear_acceleration=vector,
                angular_velocity=vector,_connection_header=dict(callerid='/synthetic_camera'))
            yield '/imu',msg,None

    def observe(self,stamps):
        config=dict(common=dict(lid_topic='/points',imu_topic='/imu',imu_time_offset=0.))
        return list(bag_observations(None,config,messages=self.messages(stamps)))

    def test_duplicate_keeps_nanoseconds_and_publisher_in_failure(self):
        with self.assertRaisesRegex(ValueError,r'previous_stamp_ns=1000000000, current_stamp_ns=1000000000.*caller=/synthetic_camera'):
            self.observe([1000000000,1000000000])

    def test_reversed_stamp_is_rejected_without_reordering(self):
        with self.assertRaisesRegex(ValueError,r'current_stamp_ns=999000000.*effective_delta_s=-0.001000000'):
            self.observe([1000000000,999000000])

    def test_monotonic_measurements_are_accepted_without_new_scans(self):
        self.assertEqual(self.observe([1000000000,1000500000,1001000000]),[])

    def test_interval_keeps_leading_imu_and_each_input_mode_camera_history(self):
        def header(t):
            return SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:t,to_nsec=lambda:round(t*1e9)),
                seq=0,frame_id='sensor')
        vector=SimpleNamespace(x=0.,y=0.,z=0.)
        def imu(t):return SimpleNamespace(header=header(t),linear_acceleration=vector,angular_velocity=vector)
        def lidar(t):return SimpleNamespace(header=header(t))
        camera=SimpleNamespace(header=header(1.1),K=[1.]*9,D=[],width=1,height=1,distortion_model='none')
        image=SimpleNamespace(header=header(1.12))
        rows=[('/imu',imu(.9),None),('/points',lidar(1.),None),('/info',camera,None),
              ('/rgb',image,None),('/imu',imu(1.15),None),('/points',lidar(1.2),None),
              ('/imu',imu(1.25),None),('/points',lidar(1.45),None)]
        config=dict(common=dict(lid_topic='/points',imu_topic='/imu'),preprocess={})
        topics=dict(rgb='/rgb',camera_info='/info')
        fields={k:np.array([0.],np.float32) for k in ('x','y','z','intensity','timestamp')}
        def decode(source,**kwargs):
            with patch('perception.sensors.observations.pointcloud_fields',return_value=fields), \
                    patch('perception.sensors.observations.rs16_preprocess',side_effect=lambda f,c,t:(np.zeros((1,5)),t-.1,t)), \
                    patch('perception.sensors.observations.rgb_array',return_value=np.zeros((1,1,3),np.uint8)):
                return list(bag_observations(None,config,messages=source,image_topics=topics,**kwargs))
        direct=decode(rows,start=.2,duration=.2)
        selected=list(selected_messages(None,config,start=.2,duration=.2,messages=rows,image_topics=topics))
        chained=decode(selected)
        self.assertEqual([row[0] for row in selected],['/imu','/imu','/points','/imu'])
        self.assertEqual(len(direct),1);self.assertEqual(len(chained),1)
        self.assertEqual(direct[0]['rgb_stamp'],1.12);self.assertIsNone(chained[0]['rgb_stamp'])
        np.testing.assert_array_equal(direct[0]['imu'],chained[0]['imu'])
        self.assertEqual(direct[0]['stamp_ns'],chained[0]['stamp_ns'])


if __name__=='__main__':unittest.main()
