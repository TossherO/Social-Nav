"""VLM image work is optional, causal and does not block the scan producer."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.decision import DecisionBuffer,DecisionRecorder
from decision.vlm.image_inputs import ImagePreparer
from perception.detection_tracking.inference.config import load_config
from tests.test_decision_runtime import observation
from tests.test_rgb_association import TRANSFORM


def raw_observation(stamp):
    row,arrays,association,scene=observation(stamp)
    ns=association['image_input']['rgb_stamp_ns']
    image=dict(rgb_stamp_ns=ns,rgb_frame='camera',camera_info=dict(
        K=[32.,0.,32.,0.,32.,24.,0.,0.,1.],D=[0.]*5,width=64,height=48,
        distortion_model='plumb_bob',frame_id='camera',stamp_ns=ns))
    return row,arrays,image,scene


def cpu_preparer():
    result=ImagePreparer.__new__(ImagePreparer)
    result.policy=load_config(local_path='/tmp/nonexistent-social-nav-config')['image_association']
    result.transform=TRANSFORM;result.stream=None;result.calls=0
    result.calibration=dict(source_frame=observation(10)[0]['detection_frame_id'],target_frame='camera',
        matrix=TRANSFORM.tolist(),sha256='synthetic-calibration')
    result.detector=SimpleNamespace(metadata=dict(backend='synthetic_cpu_test'),
        detect=lambda pixels:([dict(box=[20.,5.,35.,40.],score=.9,class_id=0)],.001))
    result.metadata=dict(detector=result.detector.metadata,calibration=result.calibration)
    return result


class VLMImageBranchTests(unittest.TestCase):
    def test_raw_rgb_window_keeps_source_time_without_requiring_early_yolo(self):
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            for i in range(12):
                row,arrays,image,scene=raw_observation(10+i/10)
                buffer.push(row,arrays,None,image_metadata=image,scene=scene)
            lease=buffer.acquire();self.assertIsNotNone(lease)
            self.assertTrue(all(e['association'] is None for e in lease['entries']))
            with buffer.consume(lease) as read:
                arrays['rgb'][:]=0
                self.assertEqual(int(read(lease['entries'][-1]['buffer'])['rgb'][0,0,0]),120)
        finally:buffer.close()

    def test_future_raw_rgb_does_not_become_a_window(self):
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            for i in range(12):
                row,arrays,image,scene=raw_observation(10+i/10)
                if i==11:image['rgb_stamp_ns']=round((row['stamp_s']+.01)*1e9)
                buffer.push(row,arrays,None,image_metadata=image,scene=scene)
            self.assertIsNone(buffer.acquire())
        finally:buffer.close()

    def test_slow_image_work_does_not_block_new_scan_submission_and_freezes_provenance(self):
        entered=threading.Event();release=threading.Event();finished=threading.Event();preparer=cpu_preparer()
        original=preparer.prepare
        def delayed(*args):
            entered.set()
            if not release.wait(3):raise RuntimeError('Test image worker was not released')
            return original(*args)
        preparer.prepare=delayed
        with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.stream',side_effect=lambda _:nullcontext()):
            recorder=DecisionRecorder(Path(tmp)/'inputs',image_preparer_factory=lambda:preparer)
            try:
                for i in range(12):
                    row,arrays,image,scene=raw_observation(10+i/10)
                    recorder.submit(row,arrays,image_metadata=image,scene=scene)
                self.assertTrue(entered.wait(2))
                row,arrays,image,scene=raw_observation(11.2);before=copy.deepcopy(row)
                def producer():
                    try:recorder.submit(row,arrays,image_metadata=image,scene=scene)
                    finally:finished.set()
                thread=threading.Thread(target=producer);thread.start()
                self.assertTrue(finished.wait(1),'Producer waited for YOLO')
                self.assertFalse(release.is_set());self.assertEqual(row,before)
                release.set();thread.join(1)
            finally:
                release.set();report=recorder.close()
            self.assertTrue(report['complete'],report)
            self.assertEqual(report['written'],1);self.assertEqual(report['image_detector_calls'],3)
            record=json.loads(next((Path(tmp)/'inputs').glob('*/input.json')).read_text())
            self.assertEqual(record['image_preparation']['calibration']['sha256'],'synthetic-calibration')
            self.assertTrue(all(s['association_record']['image_input']['raw_rgb_sha256'] for s in record['sources']))
            self.assertEqual(record['pid_to_track'],{'0':['tracker_session',7,'a']})

    def test_failed_optional_image_backend_does_not_fail_later_scan_submission(self):
        def fail():raise ValueError('synthetic unavailable image model')
        with tempfile.TemporaryDirectory() as tmp:
            recorder=DecisionRecorder(Path(tmp)/'inputs',image_preparer_factory=fail)
            for i in range(12):
                row,arrays,image,scene=raw_observation(10+i/10)
                recorder.submit(row,arrays,image_metadata=image,scene=scene)
            recorder.worker.join(timeout=2)
            row,arrays,image,scene=raw_observation(11.2)
            self.assertFalse(recorder.submit(row,arrays,image_metadata=image,scene=scene))
            report=recorder.close()
            self.assertFalse(report['complete']);self.assertIn('unavailable image model',report['error'])
            self.assertFalse(recorder.buffer.pool.leases)

    def test_saved_associations_do_not_initialize_a_new_image_model(self):
        def fail():raise AssertionError('Saved image input should not rerun YOLO')
        with tempfile.TemporaryDirectory() as tmp:
            recorder=DecisionRecorder(Path(tmp)/'inputs',image_preparer_factory=fail)
            for i in range(12):
                row,arrays,association,scene=observation(10+i/10)
                recorder.submit(row,arrays,association,scene=scene)
            report=recorder.close()
            self.assertTrue(report['complete']);self.assertEqual(report['written'],1)
            self.assertEqual(report['image_detector_calls'],0)

    def test_live_edge_only_subscribes_camera_when_explicitly_requested(self):
        from runtime.ros_input import messages
        for enabled in (False,True):
            topics=[]
            ros=SimpleNamespace(init_node=lambda *a,**k:None,is_shutdown=lambda:True,
                Subscriber=lambda topic,*a,**k:topics.append(topic) or SimpleNamespace(unregister=lambda:None))
            sensor=SimpleNamespace(PointCloud2=object,Imu=object,Image=object,CameraInfo=object)
            with patch.dict('sys.modules',{'rospy':ros,'sensor_msgs.msg':sensor}):
                self.assertEqual(list(messages(dict(common=dict(lid_topic='/lidar',imu_topic='/imu')),include_rgb=enabled)),[])
            self.assertEqual(topics,['/lidar','/imu']+(['/camera/color/image_raw','/camera/color/camera_info'] if enabled else []))

    def test_annotation_preparation_keeps_explicit_calibration_without_yolo_records(self):
        from decision.vlm.annotation.__main__ import arguments
        from decision.vlm.annotation.prepare import prepare
        from decision.vlm.annotation.store import Store
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();bag=root/'source.bag';bag.write_bytes(b'NOT_READ_BY_CPU_TEST')
            row=observation(10)[0]
            (run/'people.jsonl').write_text(json.dumps(row)+'\n')
            calibration=dict(source_frame=row['detection_frame_id'],target_frame='camera',
                matrix=TRANSFORM.tolist(),sha256='frozen-explicit-camera')
            metadata=dict(bag=str(bag),success=True,camera_calibration=calibration)
            (run/'run.json').write_text(json.dumps(metadata));original=(run/'run.json').read_bytes()
            args=arguments(['prepare','--results',str(run),'--output',str(root/'project')])
            def probe(command,**kwargs):
                self.assertIn('--probe',command)
                jobs=json.loads(Path(command[-1]).read_text())['jobs']
                return SimpleNamespace(returncode=0,stdout=json.dumps({j['id']:dict(origin_ns=10_000_000_000,
                    record_offset_ns=0,sensor_frame=row['detection_frame_id'],bag_fingerprint={'test':'fixture'}) for j in jobs}))
            with patch('decision.vlm.annotation.prepare.subprocess.run',side_effect=probe), \
                    patch('decision.vlm.annotation.rgb_preparation.prepare_rgb',side_effect=AssertionError('No implicit YOLO')),redirect_stdout(io.StringIO()):
                prepare(args)
            store=Store(root/'project')
            try:
                source=store.sources()[0]
                self.assertEqual(source['calibration_sha256'],'frozen-explicit-camera')
                self.assertEqual(source['reader']['transform'],TRANSFORM.tolist())
                self.assertIsNone(store.rgb_cache)
            finally:store.close()
            self.assertEqual((run/'run.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
