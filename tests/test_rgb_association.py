"""Image/3D association boundaries and annotation binding; no detector or GPU required."""
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from perception.detection_tracking.inference.config import load_config
from perception.detection_tracking.inference.rgb_association import ImageAssociator, causal_geometry
from perception.detection_tracking.inference.rgb_cache import CacheWriter, RGBCache


CAMERA = dict(K=[320.,0.,320.,0.,320.,240.,0.,0.,1.],D=[0.]*5,
              width=640,height=480,distortion_model='plumb_bob')
TRANSFORM = np.array([[0.,-1.,0.,0.],[0.,0.,-1.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.]])


def person(key='A/g0',y=1.,observed=True,confirmed=True):
    return dict(key=key,kind='track',box=[5.,y,0.,.7,.7,1.7,0.,.9],observed=observed,confirmed=confirmed)


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(local_path='/tmp/nonexistent-social-nav-config')['image_association']
        self.matcher = ImageAssociator(self.config)
        self.detections = [dict(box=[210.,120.,300.,360.],score=.9,class_id=0),
                           dict(box=[340.,120.,430.,360.],score=.8,class_id=0)]

    def step(self,people,detections=None,delta=.01,**kwargs):
        return self.matcher.step(people,self.detections if detections is None else detections,
                                 TRANSFORM,CAMERA,delta,kwargs.pop('stamp',1.),kwargs.pop('scope','test'),**kwargs)

    def test_one_to_one_and_unmatched_image_people_never_create_tracks(self):
        rows=[person(),person('B/g0',y=-1.),person('tentative',confirmed=False)]
        before=copy.deepcopy(rows)
        detections=self.detections+[dict(box=[500.,10.,600.,100.],score=.99,class_id=0)]
        output=self.step(rows,detections)
        self.assertEqual(output['matches'],2)
        self.assertEqual(output['unmatched_detection_count'],1)
        self.assertEqual(set(output['tracks']),{'A/g0','B/g0','tentative'})
        self.assertEqual(output['tracks']['A/g0']['rgb_box'],detections[0]['box'])
        self.assertEqual(output['tracks']['B/g0']['rgb_box'],detections[1]['box'])
        self.assertEqual(output['tracks']['tentative']['status'],'unconfirmed')
        self.assertEqual(rows,before)

    def test_ambiguity_rejects_two_tracks_competing_for_the_same_body(self):
        output=self.step([person(),person('other/g0')],self.detections[:1])
        self.assertEqual(output['matches'],0)
        self.assertTrue(all(r['status']=='ambiguous' and r['rgb_box'] is None for r in output['tracks'].values()))

    def test_observed_track_precedes_prediction_and_history_resets(self):
        output=self.step([person('predicted',observed=False),person()],self.detections[:1])
        self.assertEqual(output['tracks']['A/g0']['status'],'matched')
        self.assertIsNone(output['tracks']['predicted']['rgb_box'])
        self.assertTrue(self.matcher.previous)
        self.step([],[],stamp=2.,scope='new lifecycle')
        self.assertFalse(self.matcher.previous)

    def test_global_assignment_must_not_force_a_second_choice(self):
        config=dict(self.config,overlap_weight=0.,history_weight=0.,center_gate=2.)
        self.matcher=ImageAssociator(config)
        # A strongly prefers the left body; B also prefers it. Hungarian could
        # place B on the distant right body to produce two otherwise finite matches.
        a,b=person(),person('B/g0',y=.72)
        detections=[dict(box=[210.,120.,300.,360.],score=.9,class_id=0),
                    dict(box=[280.,120.,370.,360.],score=.9,class_id=0)]
        output=self.step([a,b],detections)
        self.assertLess(output['matches'],2)
        self.assertTrue(all(r.get('margin') is None or r['margin']>=config['ambiguity_margin']
                            for r in output['tracks'].values() if r['status']=='matched'))

    def test_skew_and_missing_pose_do_not_fall_back_to_projection(self):
        for output in (self.step([person()],delta=.08),self.step([person()],health='degraded')):
            self.assertEqual(output['matches'],0)
            self.assertIsNone(output['tracks']['A/g0']['rgb_box'])
        compensated=dict(person(),compensation='causal_track_prediction')
        self.assertEqual(self.step([compensated],delta=.08)['matches'],1)

    def test_temporal_geometry_uses_only_earlier_tracks_and_preserves_source(self):
        current=dict(stamp_s=1.,health='valid',pose_health='valid',pose_source='localization',
                     pose_stamp_s=1.,world_from_lidar=np.eye(4).tolist(),sequence_id='bag',session_id='run',
                     generation=0,clock_domain='bag',frame_id='odom')
        past=dict(current,stamp_s=.97,pose_stamp_s=.97,tracks=[dict(track_id='A',lifecycle_generation=0,
                  confirmed=True,box=person()['box'],velocity_xyz_mps=[1.,0.,0.])])
        future=dict(past,stamp_s=1.02,pose_stamp_s=1.02)
        original=[person()]
        result=causal_geometry(original,current,[past,future],.98,.05)
        self.assertAlmostEqual(result[0]['box'][0],5.01)
        self.assertEqual(result[0]['geometry_stamp_s'],.97)
        self.assertNotIn('compensation',original[0])
        self.assertNotIn('compensation',causal_geometry(original,current,[future],.98,.05)[0])


class AnnotationCacheTests(unittest.TestCase):
    def test_rgb_preparation_and_full_reader_use_the_same_exact_header_time(self):
        from types import SimpleNamespace as S
        import sys
        import zipfile
        from decision.vlm.annotation.ros_reader import FrameReader
        ns=1700000000123456789
        def header(stamp,frame):return S(stamp=S(to_nsec=lambda:stamp),frame_id=frame)
        image=S(header=header(ns-1000000,'camera'),width=2,height=2,step=6,encoding='rgb8',
                data=bytes([20,30,40]*4))
        info=S(**dict(CAMERA,width=2,height=2,header=header(ns-1000000,'camera')))
        cloud=S(header=header(ns,'lidar'))
        raw=dict(observation_id='scan',sequence_id='bag',stamp_s=ns/1e9,detection_frame_id='lidar',
                 input_fingerprint='finger',timestamp_source='sensor',clock_domain='bag',
                 health='degraded',pose_health='invalid',detections=[])
        frame=dict(id='one',source_id='s',stamp_ns=ns+128,metadata=json.dumps(dict(result=raw)))
        config=dict(pointcloud_topic='cloud',rgb_topic='rgb',camera_info_topic='info',lidar_frame='lidar',
                    camera_frame='camera',transform=TRANSFORM.tolist(),max_rgb_age_s=.1,max_points=6000)
        source=dict(reader=config,origin_ns=ns-20000000000,record_offset_ns=0,calibration_sha256='calibration')
        reader=FrameReader.__new__(FrameReader)
        reader.db=S(execute=lambda *args:S(fetchone=lambda:frame))
        reader.source=lambda *_:(source,None,info)
        reader.cached_messages=lambda *args:[('cloud',cloud,None),('rgb',image,None),('info',info,None)]
        fields={name:np.array([0.],dtype=np.float32) for name in ('x','y','z','intensity')}
        with patch.dict(sys.modules,{'rospy':S()}), \
                patch('perception.sensors.pointcloud.pointcloud_fields',return_value=fields), \
                patch('perception.sensors.pointcloud.input_fingerprint',return_value='finger'):
            output=[]
            for rgb_only in (True,False):
                with zipfile.ZipFile(io.BytesIO(reader.read('one',association=True,rgb_only=rgb_only))) as bundle:
                    output.append(json.loads(bundle.read('frame.json')))
        self.assertEqual([row['stamp_ns'] for row in output],[ns,ns])
        self.assertEqual(output[0]['image_association_input'],output[1]['image_association_input'])

    def test_local_worker_and_export_consume_the_same_bound_boxes(self):
        from decision.vlm.annotation.backend import LocalBackend,build_view
        from decision.vlm.annotation.store import Store,new_event
        from decision.vlm.annotation.export import export_project
        from decision.vlm.annotation.rgb_preparation import prepare_rgb
        from decision.vlm.annotation.rules import t1_empty
        from tests.test_vlm_annotation import fixture
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            project=root/'project'
            fixture(project)
            with sqlite3.connect(project/'observations.sqlite') as db:
                original_frames=db.execute('SELECT * FROM frames ORDER BY id').fetchall()
                original_meta=dict(db.execute('SELECT key,value FROM meta'))
            source=Store(project)
            frames=[source.frame('source/'+str(i)) for i in range(31)]
            source.writer('old',acquire=True)
            event=new_event(source.sources()[0],0.,3.,'原投影模式标注')
            event['split']='development'
            label=t1_empty(source.window('source/20')['fingerprint'])
            label.update(status='complete',singles=['track:A','track:B'],reviewer='fixture',reviewed_at='original review')
            event['t1']['source/20']=label
            source.commit(dict(project_id=source.identity,writer_id='old',operation_id='original',expected_revision=0,event=event))
            source.writer('old',release=True)
            writer=CacheWriter(root/'associations',dict(project_id=source.identity,asset_identity=source.asset_identity),
                               dict(model='fixture'),dict(method='fixture'))
            for frame in frames:
                tracks={p['key']:dict(status='matched',rgb_box=[20.,40.,100.,300.]) if p['key']=='track:A'
                        else dict(status='unmatched',rgb_box=None) for p in frame['people']}
                writer.put_observation(frame,dict(status='complete',tracks=tracks))
            writer.finish(dict(observations=31));writer.close()
            with patch('decision.vlm.annotation.rgb_preparation.subprocess.run',side_effect=AssertionError('must reuse complete cache')):
                prepare_rgb(project,root/'associations')
            bound=Store(project)
            frame=bound.frame('source/20')
            self.assertEqual(frame['people'][0]['rgb_box'],[20.,40.,100.,300.])
            self.assertIsNone(frame['people'][1]['rgb_box'])
            self.assertEqual(len(frame['people']),2)
            self.assertEqual(bound.window('source/20')['mapping'],source.window('source/20')['mapping'])
            self.assertNotEqual(bound.window('source/20')['fingerprint'],source.window('source/20')['fingerprint'])
            self.assertNotEqual(bound.snapshot()['input_implementation'],source.snapshot()['input_implementation'])
            bound.writer('new',acquire=True)
            updated=bound.events()[0]['document']
            updated['name']='新模式保留旧答案，尚未重审'
            saved=bound.commit(dict(project_id=bound.identity,writer_id='new',operation_id='keep-old',expected_revision=1,event=updated))
            self.assertEqual(saved['document']['t1']['source/20'],label)
            with self.assertRaisesRegex(ValueError,'输入已改变'):
                export_project(project,root/'stale-export')
            updated=saved['document']
            updated['t1']['source/20']['binding']=bound.window('source/20')['fingerprint']
            updated['t1']['source/20']['reviewed_at']='explicit new review'
            bound.commit(dict(project_id=bound.identity,writer_id='new',operation_id='reviewed',expected_revision=2,event=updated))
            exported=export_project(project,root/'export',inputs=True)
            self.assertEqual(exported['rgb_association_identity'],bound.rgb_cache.identity)
            self.assertEqual(exported['counts']['t1'],1)
            bound.writer('new',release=True)
            direct=build_view(bound,'source/20',review=True)
            backend=LocalBackend(project,readonly=True)
            try:
                remote=backend.views(['source/20'],review=True)[0]
                self.assertEqual(remote['window'],direct['window'])
                self.assertEqual(remote['histories'],direct['histories'])
            finally:backend.close()
            with sqlite3.connect(project/'observations.sqlite') as db:
                self.assertEqual(db.execute('SELECT * FROM frames ORDER BY id').fetchall(),original_frames)
                current_meta=dict(db.execute("SELECT key,value FROM meta WHERE key!='default_rgb_associations'"))
                self.assertEqual(current_meta,original_meta)
            source.close();bound.close()

    def test_cache_requires_matching_source_and_prepared_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            source=dict(project_id='project',asset_identity='assets')
            writer=CacheWriter(path,source,dict(model='test'),dict(method='test'))
            frame=dict(id='one',source_id='s',stamp_ns=1,rgb_stamp_ns=1,image_size=[640,480],
                       perception_fingerprint='raw',image_association_input={'raw_rgb_sha256':'pixels'},
                       people=[person()],fingerprint='original')
            result=dict(status='complete',tracks={'A/g0':dict(status='unmatched',rgb_box=None)})
            writer.put_observation(frame,result);writer.finish({});writer.close()
            with self.assertRaisesRegex(ValueError,'来源不符'):RGBCache(path,'another','assets')
            cache=RGBCache(path,'project','assets')
            self.assertIsNone(cache.apply(frame)['people'][0]['rgb_box'])
            with self.assertRaisesRegex(ValueError,'尚未准备'):cache.apply(dict(frame,id='missing'))
            changed=dict(frame,image_association_input={'raw_rgb_sha256':'different pixels'})
            with self.assertRaisesRegex(ValueError,'不匹配'):cache.apply(changed)


class LiveInputTests(unittest.TestCase):
    def test_bounded_slot_rejects_late_observations_and_retired_generations(self):
        from collections import deque
        import threading
        from types import SimpleNamespace
        from perception.detection_tracking.ros.rgb_node import RGBAssociationNode
        from perception.detection_tracking.inference.ros_inputs import LatestSlot
        node=RGBAssociationNode.__new__(RGBAssociationNode)
        node.condition,node.pending=threading.Condition(),LatestSlot()
        node.latest,node.retired_scopes=None,deque(maxlen=16)
        node.rospy=SimpleNamespace(logwarn_throttle=lambda *args:None)
        def send(stamp,generation,ident):
            node.people(SimpleNamespace(data=json.dumps(dict(stamp_s=stamp,generation=generation,
                observation_id=ident,session_id='run',sequence_id='bag',clock_domain='sim'))))
        send(1.,0,'one');send(2.,0,'two')
        self.assertEqual(node.pending.dropped,1)
        send(1.5,0,'late')
        self.assertEqual(node.pending.take()['observation_id'],'two')
        send(3.,2,'new')
        send(4.,1,'retired-but-never-seen')
        self.assertEqual(node.pending.take()['observation_id'],'new')


if __name__=='__main__':unittest.main()
