"""Time failures retain source evidence; no ROS master or native inference."""
from types import SimpleNamespace
import unittest
from runtime.inputs import bag_observations


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


if __name__=='__main__':unittest.main()
