"""Recorded association reuse, exact input binding, and controller drain boundaries."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from decision.vlm.annotation.store import Store
from decision.vlm.annotation.rgb_preparation import prepare_rgb
from perception.detection_tracking.inference.rgb_cache import digest,file_digest
from perception.detection_tracking.inference.rgb_recording import import_recordings,load_records
from perception.mapping_localization.__main__ import wait_for_image_drain,replay_topics,arguments
from tests.test_vlm_annotation import fixture
from tests.test_rgb_association import CAMERA


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.project=self.root/'annotation';fixture(self.project)
        original=Store(self.project);image=Image.open(io.BytesIO(original.asset('source/0','rgb'))).convert('RGB');original.close()
        pixel_hash=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        self.records=[];raw=[]
        with sqlite3.connect(self.project/'observations.sqlite') as db:
            for ident,stamp,metadata in db.execute('SELECT id,stamp_ns,metadata FROM frames').fetchall():
                m=json.loads(metadata);index=int(ident.split('/')[-1])
                fp='original-scan-'+str(index)
                model=dict(rgb_stamp_ns=stamp,image_size=[640,480],raw_rgb_sha256=pixel_hash,camera=CAMERA,
                    camera_info_stamp_ns=stamp,camera_frame='camera',calibration_sha256='frozen')
                for p in m['people']:p['key']=p['key'].split(':')[-1]+'/g0'
                m.update(perception_fingerprint=fp,perception_observation_id=str(index),rgb_stamp_ns=stamp,
                    image_association_input={k:v for k,v in model.items() if k not in ('rgb_stamp_ns','image_size')})
                db.execute('UPDATE frames SET metadata=? WHERE id=?',(json.dumps(m),ident))
                row=dict(session_id='session',generation=0,sequence_id='sequence',clock_domain='bag',
                    observation_id=str(index),stamp_s=stamp/1e9,input_fingerprint=fp,health='valid',pose_health='valid',
                    world_from_lidar=np.eye(4).tolist(),tracks=[dict(track_id=p['key'].split('/')[0],lifecycle_generation=0,
                        box=p['box'],state='observed',confirmed=True) for p in m['people']])
                raw.append(row)
                m['result']=row
                db.execute('UPDATE frames SET metadata=? WHERE id=?',(json.dumps(m),ident))
                if index!=10:
                    result=dict(status='complete',matches=2,tracks={p['key']:dict(status='matched',rgb_box=p['rgb_box']) for p in m['people']})
                    self.records.append(dict(**{k:row[k] for k in ('session_id','generation','sequence_id','clock_domain','observation_id','stamp_s')},
                        perception_input_fingerprint=fp,image_input=model,detector_identity=digest({'sha':'fixture'}),result=result))
            self.raw_path=self.root/'people.jsonl';self.raw_path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
            self.rgb_path=self.root/'rgb.jsonl';self.rgb_path.write_text(''.join(json.dumps(r)+'\n' for r in self.records))
            source=json.loads(db.execute('SELECT metadata FROM sources').fetchone()[0])
            source.update(records=str(self.raw_path),records_sha256=file_digest(self.raw_path),calibration_sha256='frozen',
                rgb_recording=dict(path=str(self.rgb_path),sha256=file_digest(self.rgb_path),metadata={'detector':{'sha':'fixture'}}))
            db.execute('UPDATE sources SET metadata=?',(json.dumps(source),))

    def test_preparation_imports_without_inference_and_marks_missing_work(self):
        with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=AssertionError('must not run YOLO')):
            result=prepare_rgb(self.project)
        store=Store(self.project)
        try:
            self.assertEqual(store.rgb_cache.identity,result['identity'])
            self.assertIsNotNone(store.frame('source/20')['people'][0]['rgb_box'])
            missing=store.frame('source/10')
            self.assertEqual(missing['rgb_association_status'],'not_processed')
            self.assertTrue(all(p['rgb_box'] is None for p in missing['people']))
            self.assertEqual(store.rgb_cache.manifest['summary']['inferred_images'],0)
            self.assertEqual(store.rgb_cache.manifest['summary']['not_processed'],1)
            changed=store.frame('source/20').copy()
            changed['image_association_input']=dict(changed['image_association_input'],raw_rgb_sha256='different')
            with self.assertRaisesRegex(ValueError,'原图'):store.rgb_cache.lookup(changed)
        finally:store.close()

    def test_changed_record_file_is_rejected(self):
        self.rgb_path.write_text(self.rgb_path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'已改变'):load_records(self.rgb_path,'original-hash')

    def test_foreign_model_or_observation_is_not_silently_imported(self):
        self.records[0]['detector_identity']='another-model'
        self.rgb_path.write_text(''.join(json.dumps(r)+'\n' for r in self.records))
        with sqlite3.connect(self.project/'observations.sqlite') as db:
            s=json.loads(db.execute('SELECT metadata FROM sources').fetchone()[0]);s['rgb_recording']['sha256']=file_digest(self.rgb_path)
            db.execute('UPDATE sources SET metadata=?',(json.dumps(s),))
        store=Store(self.project,use_project_defaults=False)
        try:
            with self.assertRaisesRegex(ValueError,'模型'):import_recordings(store,self.root/'bad-cache')
        finally:store.close()


class DrainTests(unittest.TestCase):
    def test_overlay_draws_only_matched_source_targets_and_keeps_raw_pixels(self):
        from perception.visualization.people import render_rgb_associations
        from perception.visualization.scans import configure_rviz
        pixels=np.zeros((100,150,3),dtype=np.uint8)
        people=[dict(key='run:7/g1',observed=True),dict(key='run:8/g1',observed=True)]
        result=dict(tracks={'run:7/g1':dict(status='matched',rgb_box=[10,30,40,80]),
                            'run:8/g1':dict(status='ambiguous',rgb_box=None)})
        rendered=render_rgb_associations(pixels,people,result)
        self.assertFalse(pixels.any());self.assertTrue(rendered[30,10].any())
        self.assertFalse(rendered[35:,60:].any())
        config={'Visualization Manager':{'Displays':[{'Image Topic':'/perception/people/rgb'}]}}
        configure_rviz(config,{'output_topic':'/people','people_image_topic':'/matched'})
        self.assertEqual(config['Visualization Manager']['Displays'][0]['Image Topic'],'/matched')

    def test_late_observation_cannot_satisfy_drain_and_clock_need_not_advance(self):
        last=[time.monotonic()-2,['latest']]
        condition=threading.Condition()
        with self.assertRaisesRegex(RuntimeError,'did not drain'):
            wait_for_image_drain(condition,lambda:dict(busy=False,last_processed=['old'],processed=1),last,lambda:1,lambda:True,timeout=.01)
        state=dict(busy=False,last_processed=['latest'],processed=2)
        self.assertEqual(wait_for_image_drain(condition,lambda:state,last,lambda:2,lambda:True,timeout=.1),state)

    def test_pending_archive_is_not_treated_as_saved(self):
        with self.assertRaises(RuntimeError):
            wait_for_image_drain(threading.Condition(),lambda:dict(busy=False,last_processed=['latest'],processed=2),
                [time.monotonic()-2,['latest']],lambda:1,lambda:True,timeout=.01)


if __name__=='__main__':unittest.main()
