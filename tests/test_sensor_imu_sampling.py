"""Targeted IIO ownership, restoration and actual-stream startup checks."""
import copy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from perception.sensors import imu_sampling as sampling


def policy():
    return dict(enabled=True, sdk_serial='12345', usb_serial='67890', firmware='synthetic',
                topic='/camera/imu', normal_value='2', temporary_value='500', expected_hz=200,
                source='synthetic tests only')


def samples(rate=200):
    return [(int(i*1e9/rate), 10000000000+int(i*1e9/rate), '/one_camera') for i in range(2*rate+1)]


class SamplingTests(unittest.TestCase):
    def test_local_policy_does_not_reload_lidar_only_disabled_or_other_cameras(self):
        p=policy();s={'imu_sampling_reload':p}
        self.assertIsNone(sampling.select_policy(s,'12345',lidar_only=True))
        self.assertIsNone(sampling.select_policy(s,'99999'))
        self.assertIsNone(sampling.select_policy({'imu_sampling_reload':dict(p,enabled=False)},'12345'))
        self.assertEqual(sampling.select_policy(s,'12345'),p)

    def test_incomplete_or_unbound_policy_is_rejected_before_device_access(self):
        for key,value in [('usb_serial','../other'),('firmware',''),('source',''),
                          ('temporary_value','nan'),('temporary_value','2'),('expected_hz',True),('enabled','true')]:
            p=dict(policy(),**{key:value})
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                sampling.select_policy({'imu_sampling_reload':p},'12345')

    def make_device(self,root,index,serial,*,active=True):
        usb=root/('usb'+str(index));usb.mkdir()
        for key,value in dict(idVendor='8086',idProduct='0b3a',serial=serial).items():
            (usb/key).write_text(value)
        device=usb/'gyro';(device/'buffer').mkdir(parents=True)
        (device/'name').write_text('gyro_3d')
        (device/'buffer/enable').write_text('1' if active else '0')
        (device/'in_anglvel_sampling_frequency').write_text('2.000000\n')
        (root/'iio'/('iio:device'+str(index))).symlink_to(device,target_is_directory=True)
        return device/'in_anglvel_sampling_frequency'

    def test_usb_identity_selects_front_gyro_without_touching_the_rear(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'iio').mkdir()
            rear=self.make_device(root,0,'99999');front=self.make_device(root,1,'67890')
            chosen=sampling.find_gyro('67890',root/'iio')
            self.assertEqual(chosen.resolve(),front)
            sampling.reload_attribute(chosen,policy(),lambda _:None)
            self.assertEqual(rear.read_text(),'2.000000\n')

    def test_missing_ambiguous_or_inactive_gyro_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'iio').mkdir()
            with self.assertRaises(RuntimeError):sampling.find_gyro('67890',root/'iio')
            self.make_device(root,0,'67890',active=False)
            with self.assertRaises(RuntimeError):sampling.find_gyro('67890',root/'iio')
            self.make_device(root,1,'67890')
            with self.assertRaises(RuntimeError):sampling.find_gyro('67890',root/'iio')

    def test_reload_always_switches_then_restores_even_from_an_already_normal_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'sampling';target.write_text('2.000000\n');seen=[]
            sampling.reload_attribute(target,policy(),lambda _:seen.append(target.read_text().strip()))
            self.assertEqual(seen,['500','2.000000'])
            self.assertEqual(target.read_text(),'2.000000\n')

    def test_cancellation_during_temporary_phase_restores_original_setting(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'sampling';target.write_text('2.000000\n')
            def cancel(_):raise sampling.StartupCancelled('cancelled')
            with self.assertRaises(sampling.StartupCancelled):sampling.reload_attribute(target,policy(),cancel)
            self.assertEqual(target.read_text(),'2.000000\n')

    def test_unverified_current_setting_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'sampling';target.write_text('123\n')
            with self.assertRaises(RuntimeError):sampling.reload_attribute(target,policy(),lambda _:None)
            self.assertEqual(target.read_text(),'123\n')

    def test_actual_two_khz_stream_is_rejected_despite_declared_two_hundred_hz(self):
        with self.assertRaises(RuntimeError):sampling.verify_samples(samples(2000),200)
        self.assertAlmostEqual(sampling.verify_samples(samples(),200)['rate_hz'],200)

    def test_frequency_alone_does_not_accept_a_backwards_timestamp_or_mixed_publishers(self):
        rows=samples();bad=copy.deepcopy(rows);bad[200]=(bad[200][0],bad[199][1]-1,bad[200][2])
        with self.assertRaises(RuntimeError):sampling.verify_samples(bad,200)
        bad=copy.deepcopy(rows);bad[200]=(*bad[200][:2],'/another_camera')
        with self.assertRaises(RuntimeError):sampling.verify_samples(bad,200)

    def test_duplicate_timestamp_and_short_observation_are_rejected(self):
        bad=samples();bad[200]=(bad[200][0],bad[199][1],bad[200][2])
        with self.assertRaises(RuntimeError):sampling.verify_samples(bad,200)
        with self.assertRaises(RuntimeError):sampling.verify_samples(samples()[:20],200)

    def test_second_startup_cannot_take_the_same_camera_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            with sampling.device_lock(Path(folder),'67890'):
                with self.assertRaises(RuntimeError):
                    with sampling.device_lock(Path(folder),'67890'):pass

    def test_startup_error_stops_only_owned_driver_before_releasing_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);driver=Mock(pid=123456);driver.poll.return_value=None
            def stop_owned(p):
                self.assertIs(p,driver)
                with self.assertRaises(RuntimeError):
                    with sampling.device_lock(root,'67890'):pass
            old=signal.getsignal(signal.SIGINT)
            with patch.object(sampling,'check_no_publisher'),patch.object(sampling.subprocess,'Popen',return_value=driver),\
                    patch.object(sampling,'prepare',side_effect=RuntimeError('bad stream')),\
                    patch.object(sampling,'stop_driver',side_effect=stop_owned) as cleanup:
                self.assertEqual(sampling.supervise(['synthetic'],policy(),root),1)
            cleanup.assert_called_once_with(driver)
            self.assertEqual(signal.getsignal(signal.SIGINT),old)


if __name__=='__main__':unittest.main()
