"""Bounded synthetic inertial input and explicit source isolation; no ROS/GPU."""
import struct
from types import SimpleNamespace as NS
import unittest

import numpy as np

from perception.mapping_localization.stationary_imu import StationaryImuClock, localization_source
from perception.mapping_localization.__main__ import replay_topics


def cloud(end_ns, duration=.1):
    end=end_ns/1e9
    values=[end-duration, float('nan'), end]
    return NS(header=NS(stamp=NS(to_nsec=lambda:end_ns)),
              fields=[NS(name='timestamp',offset=0,datatype=8,count=1)],
              width=3,height=1,point_step=8,row_step=24,is_bigendian=False,
              data=struct.pack('<3d',*values))


class StationaryInputTests(unittest.TestCase):
    def test_samples_cover_only_observed_interval_and_continue_monotonically(self):
        generator=StationaryImuClock()
        end=1700000000*10**9
        first=generator.sample_times(cloud(end))
        second=generator.sample_times(cloud(end+100000000))
        self.assertEqual(first[-1],end)
        self.assertLess(abs(first[0]-(end-100000000)),512)
        self.assertTrue(all(a<b for a,b in zip(first+second,(first+second)[1:])))
        self.assertLessEqual(max(np.diff(first+second)),5000512)
        self.assertEqual(second[-1],end+100000000)
        self.assertLessEqual(len(first),22)
        self.assertLessEqual(len(second),22)

    def test_bad_time_is_rejected_without_filling_unobserved_gap(self):
        end=1700000000*10**9
        for next_end in (end,end-100000000,end+1000000000):
            generator=StationaryImuClock()
            generator.sample_times(cloud(end))
            with self.assertRaises(ValueError):generator.sample_times(cloud(next_end))
            self.assertEqual(generator.last_end_ns,end)
        with self.assertRaises(ValueError):StationaryImuClock().sample_times(cloud(end,duration=2))
        message=cloud(end)
        message.data=struct.pack('<3d',0,.05,.1)
        with self.assertRaises(ValueError):StationaryImuClock().sample_times(message)

    def test_real_and_synthetic_topics_and_sources_do_not_mix(self):
        args=NS(backend='fast_lio',assume_stationary=False)
        parameters={'common':{'lid_topic':'/raw','imu_topic':'/real_imu'}}
        self.assertEqual(replay_topics(args,parameters),['/raw','/tf_static','/real_imu'])
        args.assume_stationary=True
        parameters['stationary_imu']={'raw_topic':'/raw'}
        topics=replay_topics(args,parameters,{'rgb_topic':'/rgb'})
        self.assertEqual(topics,['/raw','/tf_static','/rgb'])
        self.assertNotIn('/real_imu',topics)
        self.assertEqual(localization_source('fast_lio'),'fast_lio')
        self.assertEqual(localization_source('fast_lio',True),'fast_lio_stationary_assumption')
        self.assertNotEqual(localization_source('roll_fast_lio',True),'roll_fast_lio')


if __name__=='__main__':unittest.main()
