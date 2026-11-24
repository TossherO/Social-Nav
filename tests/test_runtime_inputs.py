"""Time failures retain source evidence; no ROS master or native inference."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from perception.mapping_localization.native import rs16_preprocess
from perception.sensors.observations import bag_observations,selected_messages
from perception.sensors.pointcloud import pointcloud_fields


class Rs16ScanTests(unittest.TestCase):
    def test_padded_cloud_preserves_raw_points_and_original_sample_indices(self):
        width, height, point_step = 6, 2, 30
        row_step = width * point_step + 11
        names = ('x', 'y', 'z', 'intensity', 'timestamp')
        offsets = (0, 8, 12, 16, 22)
        dtype = np.dtype(dict(names=names, formats=('>f8', '>f4', '>f4', '>u2', '>f8'),
                              offsets=offsets, itemsize=point_step))
        data = bytearray(row_step * height)
        cloud = np.ndarray((height, width), dtype=dtype, buffer=data, strides=(row_step, point_step))
        cloud['x'] = [[2., 2., 2., 2., 1., 2.], [2., 2., 2., 2., 3., 2.]]
        cloud['z'][0, 0] = np.nan
        cloud['z'][1, 0] = 2.
        cloud['y'][1, 4] = -0.
        cloud['intensity'] = np.arange(100, 112).reshape(height, width)
        offset = np.arange(12).reshape(height, width) * .01
        offset[1, 2] = np.nan
        config = dict(common=dict(lid_topic='/points', imu_topic='/imu'),
                      preprocess=dict(lidar_type=4, scan_line=16, blind=1.),
                      feature_extract_enable=False, point_filter_num=2)
        stamp = SimpleNamespace(to_sec=lambda: 200., to_nsec=lambda: 200_000_000_000)
        scan = SimpleNamespace(
            width=width, height=height, point_step=point_step, row_step=row_step, is_bigendian=True,
            fields=[SimpleNamespace(name=name, offset=field_offset, count=1, datatype=datatype)
                    for name, field_offset, datatype in zip(names, offsets, (8, 7, 7, 4, 8))],
            header=SimpleNamespace(stamp=stamp, frame_id='rslidar'))
        imu_stamp = SimpleNamespace(to_sec=lambda: 201., to_nsec=lambda: 201_000_000_000)
        vector = SimpleNamespace(x=0., y=0., z=0.)
        imu = SimpleNamespace(header=SimpleNamespace(stamp=imu_stamp),
                              linear_acceleration=vector, angular_velocity=vector)
        rows = [('/points', scan, None), ('/imu', imu, None)]
        raw = np.column_stack([cloud[k].reshape(-1) for k in names[:4]]).astype(np.float32)
        for base, stamp_at_end in ((0., True), (1_795_000_000., False)):
            with self.subTest(base=base, stamp_at_end=stamp_at_end):
                cloud['timestamp'] = base + offset
                scan.data = bytes(data)
                config['preprocess']['stamp_at_scan_end'] = stamp_at_end
                with np.errstate(invalid='ignore'):
                    batch, = bag_observations(None, config, messages=rows)
                self.assertEqual(batch['raw'].tobytes(), raw.tobytes())
                times = ((base + offset.reshape(-1)[[2, 10]] - base) * 1000).astype(np.float32)
                expected = np.column_stack((raw[[2, 10]], times))
                self.assertEqual(batch['points'].tobytes(), expected.tobytes())
                duration = float(times[-1]) / 1000
                end = 200. if stamp_at_end else 200. + duration
                self.assertEqual((batch['begin'], batch['end']), (end - duration, end))

        # An unsampled final point still determines whether timestamps are valid.
        cloud['timestamp'][-1, -1] = base
        scan.data = bytes(data)
        with self.assertRaisesRegex(ValueError, 'requires measured point timestamps'):
            rs16_preprocess(pointcloud_fields(scan, names), config, 200.)


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

    def test_camera_selection_preserves_sensor_nanoseconds(self):
        scan_ns=1_760_000_000_100_000_001
        def message(topic,ns):
            msg=next(self.messages([ns]))[1]
            msg.header.frame_id='sensor'
            return topic,msg,None
        rows=[message('/rgb',scan_ns+1),message('/rgb',scan_ns),message('/points',scan_ns),
              message('/imu',scan_ns+1_000_000)]
        config=dict(common=dict(lid_topic='/points',imu_topic='/imu'),preprocess={})
        fields={k:np.zeros(1) for k in ('x','y','z','intensity','timestamp')}
        with patch('perception.sensors.observations.pointcloud_fields',return_value=fields), \
                patch('perception.sensors.observations.rs16_preprocess',side_effect=lambda f,c,t:(np.zeros((1,5)),t-.1,t)), \
                patch('perception.sensors.observations.rgb_array',return_value=np.zeros((1,1,3),np.uint8)):
            batch,=bag_observations(None,config,messages=rows,image_topics=dict(rgb='/rgb'))
        self.assertEqual(batch['stamp_ns'],scan_ns)
        self.assertEqual(batch['rgb_stamp_ns'],scan_ns)

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
