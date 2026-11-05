"""Behavioral checks for the independent social-decision annotation tool."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
from PIL import Image

from decision.vlm.annotation.backend import LocalBackend
from decision.vlm.annotation.export import export_project
from decision.vlm.annotation.rules import digest, dumps, perception_continuity, t1_empty, t2_empty, validate_t1
from decision.vlm.annotation.store import Conflict, Store, initialize_labels, initialize_observations, new_event


def fixture(path):
    db = initialize_observations(path)
    db.execute('INSERT INTO sources VALUES(?,?,?)', ('source', '室内事件', dumps(dict(root_group='recording', frame_id='rslidar'))))
    image = Image.new('RGB', (640, 480), '#809cad')
    raw = io.BytesIO()
    image.save(raw, format='PNG')
    cloud = io.BytesIO()
    np.save(cloud, np.array([[2., .5, 0., 8.], [3., -.3, 1., 10.]], dtype='<f4'), allow_pickle=False)
    for i in range(31):
        people = [dict(key='track:'+key, eligible=True, box=[2.+j, float(j), 0., .7, .7, 1.7, 0., .9],
                       rgb_box=[50.+200*j, 50., 150.+200*j, 350.], kind='track', observed=True, confirmed=True)
                  for j, key in enumerate(('A', 'B'))]
        metadata = dict(people=people, image_size=[640, 480], has_rgb=True, fingerprint=digest(['frame', i]),
                        timestamp_source='sensor', clock_domain='bag', sensor_frame='rslidar', health='valid',
                        model_scan_binding=dict(kind='full_height_single_scan',stamp_ns=1700000000000000000+i*100000000,
                            sensor_frame='rslidar',perception_input_fingerprint='synthetic-'+str(i),
                            sha256=hashlib.sha256(cloud.getvalue()).hexdigest()))
        db.execute('INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?)',
                   ('source/'+str(i), 'source', i, 1700000000000000000+i*100000000, i/10,
                    dumps(metadata), raw.getvalue(), raw.getvalue(), cloud.getvalue()))
    db.execute('INSERT INTO meta VALUES(?,?)', ('asset_identity', dumps('asset-fixture')))
    db.execute('UPDATE meta SET value=? WHERE key=?', ('true', 'complete'))
    db.commit()
    db.close()
    initialize_labels(path)


class CameraViewTests(unittest.TestCase):
    def test_current_confirmed_tracks_are_valid_outside_the_camera_view(self):
        from decision.vlm.annotation.ros_reader import people_for
        info = SimpleNamespace(K=[320.,0.,320.,0.,320.,240.,0.,0.,1.],D=[0.]*5,
                               width=640,height=480,distortion_model='plumb_bob')
        transform = np.eye(4)
        transform[:3,:3] = [[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]]
        tracks = [dict(track_id=key,lifecycle_generation=0,confirmed=confirmed,state=state,
                       box=[x,0.,0.,.7,.7,1.7,0.,.9]) for key,x,confirmed,state in
                  [('front',5.,True,'observed'),('rear',-5.,True,'observed'),
                   ('predicted',-6.,True,'predicted'),('tentative',-7.,False,'observed')]]
        row = dict(health='valid',pose_health='valid',world_from_lidar=np.eye(4).tolist(),tracks=tracks)
        people = {p['key']:p for p in people_for(row,transform,info,[640,480])}
        self.assertTrue(people['front/g0']['rgb_box'])
        self.assertIsNone(people['rear/g0']['rgb_box'])
        self.assertEqual({k for k,p in people.items() if p['eligible']},{'front/g0','rear/g0'})
        self.assertEqual({p['key'] for p in people_for(row,transform,None,None) if p['eligible']},
                         {'front/g0','rear/g0'})

    def test_fov_uses_optical_extrinsics_and_distortion_in_lidar_xy(self):
        from decision.vlm.annotation.ros_reader import camera_fov
        info = SimpleNamespace(K=[320., 0., 320., 0., 320., 240., 0., 0., 1.], D=[0.]*5,
                               width=640, height=480, distortion_model='plumb_bob')
        transform = np.eye(4)
        transform[:3, :3] = [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]
        origin = np.array([.4, .2, 1.])
        transform[:3, 3] = -transform[:3, :3] @ origin
        fov = camera_fov(transform, info, [640, 480])
        np.testing.assert_allclose(fov['origin_xy_m'], origin[:2])
        np.testing.assert_allclose(fov['forward_xy'], [1., 0.])
        np.testing.assert_allclose(fov['boundary_directions_xy'][0], np.array([1., 1.])/np.sqrt(2))
        info.D = [.1, -.02, 0., 0., 0.]
        corrected = camera_fov(transform, info, [640, 480])
        self.assertLess(corrected['boundary_directions_xy'][0][1], fov['boundary_directions_xy'][0][1])
        # Turn the camera left by 90 degrees; the indication must turn with it.
        transform[:3, :3] = transform[:3, :3] @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        transform[:3, 3] = -transform[:3, :3] @ origin
        turned = camera_fov(transform, info, [640, 480])
        np.testing.assert_allclose(turned['forward_xy'], [0., 1.], atol=1e-7)
        self.assertIsNone(camera_fov(transform, None, [640, 480]))
        self.assertIsNone(camera_fov(transform, info, [320, 240]))
        info.distortion_model = 'unsupported'
        self.assertIsNone(camera_fov(transform, info, [640, 480]))


class LazyIndexTests(unittest.TestCase):
    def test_overlapping_bag_reads_reuse_bounded_chunks_without_duplicate_boundaries(self):
        from collections import OrderedDict
        from decision.vlm.annotation.ros_reader import FrameReader,MESSAGE_CACHE_BYTES
        class Stamp:
            def __init__(self,ns):self.ns=ns
            def to_nsec(self):return self.ns
        records=[('rgb',SimpleNamespace(data=b'pixels'),Stamp(ns)) for ns in (100000000,200000000,250000000,500000000,600000000)]
        class Bag:
            calls=0
            def read_messages(self,topics,start_time,end_time):
                self.calls+=1
                return [row for row in records if start_time.ns<=row[2].ns<=end_time.ns]
        reader=FrameReader.__new__(FrameReader)
        reader.messages,reader.message_bytes=OrderedDict(),0
        bag=Bag()
        rospy=SimpleNamespace(Time=lambda seconds,nanos:Stamp(seconds*1000000000+nanos))
        with patch.dict('sys.modules',rospy=rospy):
            first=list(reader.cached_messages('source',bag,['rgb'],.1,.5))
            calls=bag.calls
            second=list(reader.cached_messages('source',bag,['rgb'],.2,.6))
        self.assertEqual([row[2].ns for row in first],[100000000,200000000,250000000,500000000])
        self.assertEqual([row[2].ns for row in second],[200000000,250000000,500000000,600000000])
        self.assertEqual(bag.calls,calls)
        self.assertLessEqual(reader.message_bytes,MESSAGE_CACHE_BYTES)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.path = root/'project'
        results = root/'people.jsonl'
        bag = root/'source.bag'
        bag.write_bytes(b'synthetic; must not be decoded by registration')
        calibration = root/'camera.yaml'
        calibration.write_text(dumps(dict(mounting_scope='fixture_mount',
            lidar_to_camera_optical=dict(matrix=np.eye(4).tolist(), source_frame='rslidar', target_frame='camera'))))
        rows = []
        for i in range(31):
            rows.append(dict(observation_id='original/'+str(i), sequence_id='sequence', stamp_s=1700000000+i*.1001,
                             clock_domain='bag', timestamp_source='sensor', detection_frame_id='rslidar', health='valid',
                             pose_health='valid', world_from_lidar=np.eye(4).tolist(), detections=[],
                             frame_id='odom', session_id='detector-session', generation=1, pose_source='localization',
                             pose_stamp_s=1700000000+i*.1001,
                             localization_binding=dict(session_id='local-session',generation=2,frame_id='odom',source='fast_lio'),
                             detection_input=dict(max_sweeps=3,used_sweeps=min(3,i+1),history=[],status='ready' if i>=2 else 'partial'),
                             tracks=[dict(track_id='track:A', lifecycle_generation=1, state='observed', confirmed=True,
                                          box=[2,0,0,.7,.7,1.7,0,.9])], input_fingerprint='original-'+str(i), rgb_stamp_s=None))
        results.write_text(''.join(dumps(row)+'\n' for row in rows))
        from decision.vlm.annotation.__main__ import arguments
        from decision.vlm.annotation.prepare import prepare
        args = arguments(['prepare','--results',str(results),'--bag',str(bag),'--calibration',str(calibration),
                          '--max-rgb-age','.15','--output',str(self.path)])
        def headers(command, **kwargs):
            recipe = json.loads(Path(command[-1]).read_text())
            probes = {job['id']:dict(origin_ns=1700000000000000000,record_offset_ns=0,sensor_frame='rslidar',
                                     bag_fingerprint={'synthetic':True}) for job in recipe['jobs']}
            return SimpleNamespace(returncode=0, stdout=dumps(probes), stderr='')
        with patch('decision.vlm.annotation.prepare.subprocess.run', side_effect=headers), redirect_stdout(io.StringIO()):
            prepare(args)
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_registration_has_only_index_and_never_encodes_all_assets(self):
        import sqlite3
        rows = self.store.frames(self.store.sources()[0]['id'])
        self.assertEqual(len(rows),31)
        self.assertTrue(all(r['lazy'] for r in rows))
        self.assertIsNone(self.store._assets)
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM frames WHERE rgb IS NOT NULL OR cloud IS NOT NULL').fetchone()[0],0)

    def test_temporal_input_and_recorded_provenance_survive_import(self):
        import sqlite3
        source = self.store.sources()[0]
        self.assertEqual(source['calibration_scope'],'fixture_mount')
        self.assertEqual(source['reader']['max_rgb_age_s'],.15)
        self.assertEqual(source['perception_provenance']['detector']['status'],'not_recorded')
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            metadata = json.loads(db.execute('SELECT metadata FROM frames WHERE ordinal=2').fetchone()[0])
        result = metadata['result']
        self.assertEqual(result['detection_input']['used_sweeps'],3)
        self.assertEqual(result['localization_binding']['generation'],2)
        self.assertEqual(metadata['continuity'],perception_continuity(result))
        association_changed = copy.deepcopy(result)
        association_changed['localization_binding']['map_association_generation'] = 99
        self.assertEqual(perception_continuity(result),perception_continuity(association_changed))
        association_changed['generation'] += 1
        self.assertNotEqual(perception_continuity(result),perception_continuity(association_changed))

    def test_resolved_metadata_overrides_index_and_only_window_candidates_are_read(self):
        rows = self.store.frames(self.store.sources()[0]['id'])
        calls = set()
        frames = {r['id']:r for r in rows}
        class Reader:
            def read(self, ident):
                calls.add(ident)
                row = frames[ident]
                return dict(metadata=dict(stamp_ns=row['stamp_ns']+111,elapsed=row['elapsed']+.000000111,
                                          has_rgb=True,image_size=[640,480],fingerprint='materialized-'+ident,
                                          people=[dict(key='track:A/g1',eligible=True,box=[2,0,0,.7,.7,1.7,0,.9],rgb_box=[10,10,80,350])],
                                          model_scan_binding=dict(kind='full_height_single_scan',sha256='synthetic')),
                            rgb=b'raw',preview=b'preview',cloud=b'cloud')
            def close(self):pass
        self.store._assets = Reader()
        frame = self.store.frame(rows[20]['id'])
        self.assertEqual(frame['stamp_ns'],rows[20]['stamp_ns']+111)
        window = self.store.window(frame['id'])
        self.assertTrue(window['ready'])
        self.assertEqual(window['frames'],[rows[i]['id'] for i in (10,15,20)])
        self.assertEqual(calls,{rows[i]['id'] for i in (10,15,20)})

    def test_bulk_pending_samples_save_without_raw_frame_reads(self):
        source = self.store.sources()[0]
        rows = self.store.frames(source['id'])
        event = new_event(source,rows[0]['elapsed'],rows[-1]['elapsed'],'待标计划')
        event['t1'] = {r['id']:t1_empty(None) for r in rows[::5]}
        self.store.writer('writer',acquire=True)
        with patch.object(self.store,'_read_lazy',side_effect=AssertionError('pending samples must not decode bags')):
            saved = self.store.commit(dict(project_id=self.store.identity,writer_id='writer',operation_id='pending',expected_revision=0,event=event))
        self.assertEqual(len(saved['document']['t1']),7)


class AnnotationTests(unittest.TestCase):
    def test_model_review_uses_full_scan_binding_and_rejects_browse_substitute(self):
        import sqlite3
        from decision.vlm.annotation.backend import render_review
        from decision.vlm.scene_inputs import bev_xy
        buffer = io.BytesIO()
        np.save(buffer,np.array([[5.,3.,100.,1.],[6.,-3.,-100.,1.]],dtype='<f4'),allow_pickle=False)
        full = buffer.getvalue()
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            metadata=json.loads(db.execute('SELECT metadata FROM frames WHERE id=?',('source/20',)).fetchone()[0])
            metadata['model_scan_binding']['sha256']=hashlib.sha256(full).hexdigest()
            db.execute('UPDATE frames SET metadata=? WHERE id=?',(dumps(metadata),'source/20'))
        window=self.store.window('source/20')
        original=self.store.asset
        def asset(ident,kind,display=False):
            return full if ident=='source/20' and kind=='model_scan' else original(ident,kind,display)
        with patch.object(self.store,'asset',side_effect=asset):
            image=self.store.model_image(window,2)
            png=Image.open(io.BytesIO(render_review(self.store,'source/20')[2]))
            np.testing.assert_array_equal(image,png)
        for x,y in ((5.,3.),(6.,-3.)):
            pixel=tuple(map(int,bev_xy(x,y,window['fields']['e'])))
            self.assertNotEqual(image.getpixel(pixel),(245,247,250))
        browse=original('source/20','cloud')
        with patch.object(self.store,'asset',side_effect=lambda ident,kind: browse if kind=='model_scan' else original(ident,kind)):
            with self.assertRaisesRegex(ValueError,'扫描与绑定不符'):
                self.store.model_image(window,2)

    def test_reference_context_histories_do_not_change_model_window(self):
        import sqlite3
        from tests.test_vlm_trajectories import entry
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            for ident,ordinal,stamp,raw in db.execute('SELECT id,ordinal,stamp_ns,metadata FROM frames').fetchall():
                metadata=json.loads(raw)
                result=entry(ordinal)['result']
                result['stamp_s']=result['pose_stamp_s']=stamp/1e9
                result['tracks'][0]['state']='predicted'
                metadata['result']=result
                metadata['people'].append(dict(key='person/g0',kind='track',eligible=False,rgb_box=[300.,50.,380.,350.],
                                               observed=False,confirmed=True,box=[8.,2.,0.,.7,.7,1.7,0.,.9]))
                metadata['people'].append(dict(key='outside/g0',kind='track',eligible=False,rgb_box=None,
                                               observed=True,confirmed=True,box=[-8.,2.,0.,.7,.7,1.7,0.,.9]))
                result['tracks'].append(dict(result['tracks'][0], track_id='outside', state='observed',
                                             box=[-8.,2.,0.,.7,.7,1.7,0.,.9]))
                metadata['fingerprint']=digest({k:v for k,v in metadata.items() if k!='fingerprint'})
                db.execute('UPDATE frames SET metadata=? WHERE id=?',(dumps(metadata),ident))
        before=self.store.window('source/20')
        context=self.store.context_frames('source/20')
        self.assertNotIn('person/g0',before['mapping'])
        self.assertIn('outside/g0',before['mapping'])
        self.assertEqual(self.store.window('source/20'),before)
        self.assertIn('outside/g0', {p['key'] for p in self.store.frame('source/20')['people']})
        for row in context:
            self.assertEqual({p['key'] for p in row['frame']['people']}, {'track:A', 'track:B', 'person/g0', 'outside/g0'})
            trace=row['trajectory']
            self.assertTrue(trace['people']['outside/g0'])
            self.assertEqual(trace['status'],'available')
            points=[point for segment in trace['people']['person/g0'] for point in segment['points']]
            self.assertTrue(points)
            self.assertTrue(all(point[0]<=0 and point[1:3]==[8.,2.] for point in points))
        from decision.vlm.trajectory_inputs import build_trajectories
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            entries=[dict(id=ident,stamp_ns=stamp,result=json.loads(raw).get('result')) for ident,stamp,raw in
                     db.execute('SELECT id,stamp_ns,metadata FROM frames ORDER BY stamp_ns')]
        expected=build_trajectories(entries,before['frames'],before['mapping'])
        self.assertEqual(before['trajectories'],expected)

    def test_outside_pid_has_geometry_without_rgb_and_old_input_binding_is_rejected(self):
        import sqlite3
        from decision.vlm.annotation.backend import render_review
        from decision.vlm.scene_inputs import render_observation
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            for ident,raw in db.execute('SELECT id,metadata FROM frames').fetchall():
                metadata = json.loads(raw)
                metadata['people'].append(dict(key='rear',kind='track',observed=True,confirmed=True,eligible=False,
                                               box=[-4.3,1.,0.,.7,.7,1.7,0.,.9],rgb_box=None))
                metadata['fingerprint'] = digest({k:v for k,v in metadata.items() if k!='fingerprint'})
                db.execute('UPDATE frames SET metadata=? WHERE id=?',(dumps(metadata),ident))
        frozen = (self.path/'observations.sqlite').read_bytes()
        # Simulate the former camera-only eligibility using the same frozen source.
        with patch('decision.vlm.annotation.store.model_eligible',side_effect=lambda p:bool(p.get('eligible'))):
            old = Store(self.path)
            old_window = old.window('source/20')
            old_frame = old.frame('source/20')
        window = self.store.window('source/20')
        frame = self.store.frame('source/20')
        self.assertIn('rear',window['mapping'])
        self.assertNotEqual(window['fingerprint'],old_window['fingerprint'])
        self.assertEqual(frame['source_fingerprint'],old_frame['fingerprint'])
        rear = next(row for row in window['fields']['o'] if row[0]==window['mapping']['rear'])
        self.assertEqual(rear[1],[[-4.3,1.]]*3)
        self.assertEqual(rear[2],[None]*3)
        self.assertEqual(rear[3],[None]*3)
        self.assertEqual(window['fields']['e'],self.store.view_extent)
        rgb = self.store.asset('source/20','rgb')
        image = render_observation(frame,rgb,window['mapping'],0,window['fields']['e'])
        baseline = render_observation(old_frame,rgb,window['mapping'],0,window['fields']['e'])
        from decision.vlm.scene_inputs import MODEL_SIZE,RGB_RECT,BEV_RECT
        self.assertEqual(image.crop((0,0,RGB_RECT[2],MODEL_SIZE[1])).tobytes(),baseline.crop((0,0,RGB_RECT[2],MODEL_SIZE[1])).tobytes())
        self.assertNotEqual(image.crop(BEV_RECT).tobytes(),baseline.crop(BEV_RECT).tobytes())
        self.assertEqual(len(render_review(self.store,'source/20')),3)
        label = t1_empty(old_window['fingerprint'])
        label.update(status='complete',singles=['track:A','track:B'],unknown=['rear'],
                     reviewer='人工测试',reviewed_at='2026-10-24T00:00:00+00:00')
        self.event['t1']['source/20'] = label
        with self.assertRaisesRegex(ValueError,'输入已改变'):
            self.store.commit(self.request())
        label['binding'] = window['fingerprint']
        self.store.commit(self.request())
        self.assertEqual((self.path/'observations.sqlite').read_bytes(),frozen)

    def test_view_bundle_preserves_assets_and_separates_preview_from_review(self):
        from decision.vlm.annotation.backend import build_views,pack_views,unpack_views,render_review
        views=build_views(self.store,['source/20','source/21'])
        self.assertTrue(all(view['histories'][0].startswith(b'\xff\xd8') for view in views))
        for compress in (False,True):
            restored=unpack_views(pack_views(views,compress=compress))
            self.assertEqual(restored,views)
            self.assertIs(restored[0]['cloud'],restored[0]['context_frames'][-1]['cloud'])
        reviewed=build_views(self.store,['source/20'],True)[0]
        self.assertEqual(reviewed['histories'],render_review(self.store,'source/20'))
        self.assertIsNone(views[0]['window']['fingerprint'])
        self.assertFalse(views[0]['window']['input_verified'])
        self.assertEqual(reviewed['window']['fingerprint'],self.store.window('source/20')['fingerprint'])
        self.assertFalse(reviewed['context_frames'])

    @unittest.skipUnless(__import__('os').name=='posix','本地工作进程用于Linux')
    def test_local_worker_preserves_assets_errors_and_closes_cleanly(self):
        backend=LocalBackend(self.path,readonly=True)
        try:
            views=backend.views(['source/20'])
            process=backend.worker.process
            self.assertIsNone(process.poll())
            self.assertIsNone(views[0]['window']['fingerprint'])
            self.assertEqual(backend.window('source/20'),self.store.window('source/20'))
            self.assertEqual(backend.asset('source/20','rgb'),self.store.asset('source/20','rgb'))
            with self.assertRaises(KeyError):
                backend.frame('missing')
            self.assertEqual(backend.frame('source/20')['id'],'source/20')
        finally:
            backend.close()
        self.assertEqual(process.poll(),0)
        self.assertIsNone(backend.worker.process)

    def test_window_isolated_by_local_lifecycle_and_invalid_pose(self):
        import sqlite3
        with sqlite3.connect(self.path/'observations.sqlite') as db:
            for ident, ordinal, raw in db.execute('SELECT id,ordinal,metadata FROM frames').fetchall():
                metadata = json.loads(raw)
                metadata['continuity'] = 'before-reset' if ordinal < 18 else 'after-reset'
                if ordinal == 25:
                    metadata['continuity'] = None
                db.execute('UPDATE frames SET metadata=? WHERE id=?',(dumps(metadata),ident))
        early = self.store.window('source/20')
        self.assertEqual(early['frames'],[None,None,'source/20'])
        self.assertFalse(early['ready'])
        invalid = self.store.window('source/25')
        self.assertEqual(invalid['frames'],[None,None,'source/25'])
        after = self.store.window('source/30')
        self.assertEqual(after['frames'],['source/20',None,'source/30'])
        self.assertFalse(after['ready'])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'project'
        fixture(self.path)
        self.store = Store(self.path)
        self.owner = 'writer'
        self.store.writer(self.owner, acquire=True)
        self.event = new_event(self.store.sources()[0], 0, 3, '同行与交叉')
        self.event['split'] = 'development'

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, event=None, revision=0):
        return dict(project_id=self.store.identity, writer_id=self.owner, operation_id=uuid.uuid4().hex,
                    expected_revision=revision, event=copy.deepcopy(event or self.event))

    def complete_t1(self, ident='source/20'):
        label = t1_empty(self.store.window(ident)['fingerprint'])
        label.update(status='complete', groups=[dict(members=['track:A', 'track:B'], relation='w', roles=[])],
                     reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['t1'][ident] = label
        return label

    def test_causal_window_and_missing_history(self):
        window = self.store.window('source/20')
        self.assertEqual(window['frames'], ['source/10', 'source/15', 'source/20'])
        self.assertEqual(window['fields']['dt'], [-1., -.5, 0.])
        self.assertEqual(window['mapping'], {'track:A': 0, 'track:B': 1})
        self.assertFalse(self.store.window('source/0')['ready'])

    def test_unknown_is_not_automatic_and_queue_is_out_of_scope(self):
        label = t1_empty('input')
        label['status'] = 'complete'
        with self.assertRaisesRegex(ValueError, '未分类'):
            validate_t1(label, ['A', 'B'])
        label.update(groups=[dict(members=['A', 'B'], relation='q')])
        with self.assertRaisesRegex(ValueError, '不支持'):
            validate_t1(label, ['A', 'B'])

    def test_receipt_is_idempotent_but_not_reusable_for_different_content(self):
        request = self.request()
        first = self.store.commit(request)
        self.assertEqual(self.store.commit(request), first)
        request['event']['name'] = 'different'
        with self.assertRaises(Conflict):
            self.store.commit(request)
        self.assertEqual(self.store.events()[0]['revision'], 1)

    def test_delete_removes_only_selected_event_and_rejects_stale_recreation(self):
        self.complete_t1()
        self.event['auxiliary']['source/20'] = [dict(key='aux:missing', rect=[100,100,160,350])]
        self.event['targets']['target'] = dict(id='target', description='穿红衣的人', absent=True,
            description_source='独立任务指令', reference_frame='source/10', samples={'source/20':t2_empty(None)})
        self.store.commit(self.request())
        other = new_event(self.store.sources()[0],0,3,'保留事件')
        other['split'] = 'development'
        saved_other = self.store.commit(self.request(other))
        observations = (self.path/'observations.sqlite').read_bytes()
        request = dict(project_id=self.store.identity,writer_id=self.owner,operation_id=uuid.uuid4().hex,
                       expected_revision=1,action='delete',event_id=self.event['id'])
        with patch.object(self.store,'_read_lazy',side_effect=AssertionError('deletion must not read bags')):
            result = self.store.commit(request)
        self.assertTrue(result['deleted'])
        self.assertEqual(result['revision'],2)
        self.assertEqual(self.store.commit(request),result)
        self.assertEqual(self.store.receipt(request['operation_id']),result)
        self.assertEqual(self.store.events(),[dict(document=other,revision=saved_other['revision'])])
        self.assertEqual((self.path/'observations.sqlite').read_bytes(),observations)
        for revision in (0,1,2):
            with self.assertRaises(Conflict):
                self.store.commit(self.request(revision=revision))
        with self.assertRaisesRegex(ValueError,'没有可导出'):
            export_project(self.path,Path(self.temporary.name)/'export')
        # Deleted events do not constrain the data split of remaining events.
        other['split'] = 'validation'
        self.store.commit(self.request(other,revision=1))

    def test_delete_requires_current_revision_writer_and_matching_receipt(self):
        self.store.commit(self.request())
        request = dict(project_id=self.store.identity,writer_id=self.owner,operation_id=uuid.uuid4().hex,
                       expected_revision=0,action='delete',event_id=self.event['id'])
        with self.assertRaises(Conflict):
            self.store.commit(request)
        request.update(expected_revision=1,writer_id='another-writer')
        with self.assertRaises(Conflict):
            self.store.commit(request)
        request['writer_id'] = self.owner
        self.store.commit(request)
        request['event_id'] = 'different-event'
        with self.assertRaisesRegex(Conflict,'同一操作ID'):
            self.store.commit(request)

    def test_single_writer_and_cas_reject_stale_edits(self):
        with self.assertRaises(Conflict):
            self.store.writer('another', acquire=True)
        result = self.store.commit(self.request())
        with self.assertRaises(Conflict):
            self.store.commit(self.request(revision=0))
        self.assertEqual(result['revision'], 1)

    def test_input_binding_and_incomplete_history_cannot_be_gold(self):
        label = self.complete_t1()
        label['binding'] = 'old-input'
        with self.assertRaisesRegex(ValueError, '输入已改变'):
            self.store.commit(self.request())
        self.event['t1'].clear()
        self.complete_t1('source/0')
        with self.assertRaisesRegex(ValueError, '不足'):
            self.store.commit(self.request())

    def test_identity_edits_invalidate_reviewed_answers(self):
        self.complete_t1()
        result = self.store.commit(self.request())
        updated = result['document']
        updated['persons']['person-A'] = dict(name='A')
        updated['links'].append(dict(key='track:A', person_id='person-A', start_ns=1700000000000000000,
                                     end_ns=1700000003000000000))
        result = self.store.commit(self.request(updated, 1))
        self.assertEqual(result['document']['t1']['source/20']['status'], 'draft')

    def test_same_source_cannot_escape_split_by_renaming_root(self):
        self.store.commit(self.request())
        another = new_event(self.store.sources()[0], 0, 1, '相同bag另一事件')
        another.update(split='holdout', root_group='renamed')
        with self.assertRaisesRegex(ValueError, '跨数据划分'):
            self.store.commit(self.request(another))

    def test_t2_facts_export_does_not_invent_history_or_repair_missing_pid(self):
        self.event['persons']['person-A'] = dict(name='目标A')
        self.event['links'].append(dict(key='track:A', person_id='person-A', start_ns=1700000000000000000,
                                       end_ns=1700000003000000000))
        label = t2_empty(self.store.window('source/20')['fingerprint'])
        label.update(status='complete', visibility='clear', true_key='track:A', plausible=['track:A'], match='unique',
                     coverage='represented', reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['targets']['target'] = dict(id='target', description='穿灰衣的人', person_id='person-A',
                                             absent=False, reference_frame='source/10', samples={'source/20': label})
        self.store.commit(self.request())
        out = Path(self.temporary.name) / 'export'
        summary = export_project(self.path, out)
        result = json.loads((out / 't2.jsonl').read_text())
        self.assertEqual(summary['counts'], dict(t1=0, t2=1))
        self.assertNotIn('s', result)
        self.assertEqual(result['label_kind'], 'fixed_facts_not_history_state_gold')
        self.assertEqual(result['facts']['true_key'], 'track:A')

    def test_auxiliary_missing_person_stays_out_of_model_input(self):
        self.event['persons']['hidden'] = dict(name='漏检人物')
        self.event['auxiliary']['source/20'] = [dict(key='aux:missing', rect=[100, 100, 160, 350])]
        self.event['links'].append(dict(key='aux:missing', person_id='hidden', start_ns=1700000002000000000,
                                       end_ns=1700000002000000000))
        label = t2_empty(self.store.window('source/20')['fingerprint'])
        label.update(status='complete', visibility='clear', true_key='aux:missing', plausible=[], match='insufficient',
                     coverage='missed', reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['targets']['target'] = dict(id='target', description='背包的人', person_id='hidden', absent=False,
                                             reference_frame='source/10', samples={'source/20': label})
        self.store.commit(self.request())
        self.assertNotIn('aux:missing', self.store.window('source/20')['mapping'])

    def test_t1_unrepresentable_group_is_retained_without_fake_response(self):
        label = t1_empty(self.store.window('source/20')['fingerprint'])
        self.event['auxiliary']['source/20'] = [dict(key='aux:missing', rect=[100, 100, 160, 350])]
        label.update(status='complete', singles=['track:B'],
                     reference_groups=[dict(members=['track:A', 'aux:missing'], relation='w', roles=[])],
                     reviewer='人工测试', reviewed_at='2026-10-19T00:00:00+00:00')
        self.event['t1']['source/20'] = label
        self.store.commit(self.request())
        out = Path(self.temporary.name)/'export'
        export_project(self.path, out)
        row = json.loads((out/'t1.jsonl').read_text())
        self.assertFalse(row['semantic_eligible'])
        self.assertIsNone(row['response'])
        self.assertEqual(row['unrepresented_groups'][0]['members'], ['track:A', 'aux:missing'])


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'project'
        fixture(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_service_and_remote_durable_save(self):
        from fastapi.testclient import TestClient
        from decision.vlm.annotation.service import create_app
        from decision.vlm.annotation.remote import RemoteBackend
        with TestClient(create_app(self.path, writable=True)) as client:
            remote = RemoteBackend('http://testserver', 'wheelchair-vlm', cache_dir=Path(self.temporary.name)/'cache', transport=client._transport)
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0], 0, 3, '远程事件')
                saved = remote.commit(event, 0)
                self.assertEqual(saved['revision'], 1)
                self.assertEqual(remote.pending(), 0)
                self.assertEqual(remote.asset('source/20', 'rgb')[:8], b'\x89PNG\r\n\x1a\n')
                self.assertEqual(remote.asset('source/20', 'cloud3d'),remote.asset('source/20', 'cloud'))
                self.assertEqual(remote.window('source/20')['frames'][-1], 'source/20')
                self.assertEqual(len(remote.review_images('source/20')), 3)
                context=remote.context_frames('source/20')
                self.assertEqual([row['frame']['id'] for row in context],['source/10','source/15','source/20'])
                self.assertEqual(remote.window('source/20')['mapping'],{'track:A':0,'track:B':1})
                with patch.object(remote.http,'request',wraps=remote.http.request) as requests:
                    views=remote.views(['source/20','source/21'])
                    self.assertEqual(len(requests.call_args_list),1)
                    self.assertEqual(requests.call_args.args[1],remote.prefix+'/views')
                with remote.db() as db:
                    access_before=dict(db.execute('SELECT key,access FROM assets'))
                with patch.object(remote.http,'request',side_effect=AssertionError('view disk cache must work offline')):
                    self.assertEqual(remote.views(['source/20','source/21']),views)
                with remote.db() as db:
                    self.assertEqual(dict(db.execute('SELECT key,access FROM assets')),access_before)
                with remote.db() as db:
                    db.execute("DELETE FROM assets WHERE key LIKE '%:view-bundle:%'")
                with patch.object(remote.http,'request',wraps=remote.http.request) as requests:
                    self.assertEqual(remote.views(['source/20','source/21']),views)
                    self.assertEqual(len(requests.call_args_list),1)
                reviewed=remote.views(['source/20'],True)[0]
                self.assertEqual(reviewed['histories'],remote.review_images('source/20'))
                self.assertIsNone(views[0]['window']['fingerprint'])
                self.assertEqual(reviewed['window']['fingerprint'],remote.window('source/20')['fingerprint'])
                self.assertEqual(client.post('/projects/wheelchair-vlm/views',json=dict(frame_ids=['source/20']*13)).status_code,422)
                import httpx
                with patch.object(remote.http,'request',return_value=httpx.Response(404,json={'detail':'Not Found'})):
                    with self.assertRaisesRegex(ValueError,'更新并重启'):
                        remote.request('GET','/context-frames')
                    with self.assertRaisesRegex(ValueError,'更新并重启'):
                        remote.request('POST','/views',json={})
                self.assertEqual(client.get('/projects/wheelchair-vlm/frame', params={'frame_id': '../escape'}).status_code, 404)
            finally:
                remote.close()

    def test_asset_eviction_keeps_pending_labels_and_recent_cache_reads_do_not_write(self):
        from decision.vlm.annotation.remote import RemoteBackend
        remote=RemoteBackend('http://testserver','wheelchair-vlm',cache_dir=Path(self.temporary.name)/'cache')
        try:
            remote.put('retained-metadata','keep')
            with remote.db() as db:
                db.execute('INSERT INTO queue(id,request,state) VALUES(?,?,?)',('draft','{"human":"keep"}','queued'))
            with patch('decision.vlm.annotation.remote.ASSET_CACHE_BYTES',4096):
                for i in range(10):remote.cached('sample',str(i),lambda i=i:bytes([i])*1024,binary=True)
            self.assertEqual(remote.pending(),1)
            self.assertEqual(remote.drafts()['pending'],[{'human':'keep'}])
            self.assertEqual(remote.get('retained-metadata'),'keep')
            with remote.db() as db:
                self.assertLessEqual(db.execute('SELECT SUM(LENGTH(payload)) FROM assets').fetchone()[0],4096)
                access_before=dict(db.execute('SELECT key,access FROM assets'))
            remote.cached('sample','9',lambda:(_ for _ in ()).throw(AssertionError('cache hit must not reload')),binary=True)
            with remote.db() as db:
                self.assertEqual(dict(db.execute('SELECT key,access FROM assets')),access_before)
        finally:remote.close()

    def test_remote_delete_recovers_lost_ack_and_offline_queue_in_order(self):
        import httpx
        from fastapi.testclient import TestClient
        from decision.vlm.annotation.service import create_app
        from decision.vlm.annotation.remote import RemoteBackend
        with TestClient(create_app(self.path,writable=True)) as client:
            class Transport(httpx.BaseTransport):
                offline = False
                lost = False
                deletes = 0
                def handle_request(self,request):
                    if self.offline:
                        raise httpx.ConnectError('offline')
                    response = client._transport.handle_request(request)
                    if request.url.path.endswith('/delete-event'):
                        self.deletes += 1
                        if not self.lost:
                            self.lost = True
                            response.read()
                            raise httpx.ReadError('lost delete acknowledgement')
                    return response
            transport = Transport()
            remote = RemoteBackend('http://testserver','wheelchair-vlm',cache_dir=Path(self.temporary.name)/'cache',transport=transport)
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0],0,3,'删除事件')
                remote.commit(event,0)
                result = remote.delete_event(event['id'],1)
                self.assertTrue(result['staged'])
                self.assertEqual(remote.pending(),1)
                self.assertEqual(Store(self.path).events(),[])
                self.assertEqual(remote.with_drafts(dict(events=[dict(document=event,revision=1)]))['events'],[])
                self.assertEqual(remote.snapshot()['events'],[])
                self.assertEqual(remote.pending(),0)
                self.assertEqual(transport.deletes,1)  # Resolve receipt, never replay an acknowledged delete.
                other = new_event(index['sources'][0],0,3,'待同步创建后删除')
                transport.offline = True
                remote.commit(other,0)
                remote.delete_event(other['id'],1)
                self.assertEqual(remote.pending(),2)
                self.assertEqual(remote.with_drafts(dict(events=[]))['events'],[])
                transport.offline = False
                self.assertEqual(remote.snapshot()['events'],[])
                self.assertEqual(remote.pending(),0)
            finally:
                transport.offline = False
                remote.close()

    def test_remote_delete_conflict_readonly_and_old_service(self):
        import httpx
        from fastapi.testclient import TestClient
        from decision.vlm.annotation.service import create_app
        from decision.vlm.annotation.remote import RemoteBackend
        with TestClient(create_app(self.path,writable=True)) as client:
            remote = RemoteBackend('http://testserver','wheelchair-vlm',cache_dir=Path(self.temporary.name)/'cache',transport=client._transport)
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0],0,3,'保留冲突事件')
                remote.commit(event,0)
                with self.assertRaises(Conflict):
                    remote.delete_event(event['id'],0)
                self.assertEqual(remote.pending(),1)
                self.assertEqual(Store(self.path).events()[0]['document'],event)
                self.assertEqual(remote.drafts()['pending'][0]['action'],'delete')
                self.assertEqual(remote.snapshot()['events'][0]['document'],event)
                self.assertFalse(remote.writable)
                with patch.object(remote.http,'request',return_value=httpx.Response(404,json={'detail':'Not Found'})):
                    with self.assertRaisesRegex(ValueError,'更新并重启'):
                        remote.request('POST','/delete-event',json={})
                remote.archive_conflicts(event)
            finally:
                remote.close()
        with TestClient(create_app(self.path,writable=False)) as client:
            response = client.post('/projects/wheelchair-vlm/delete-event',json={})
            self.assertEqual(response.status_code,409)
        local = LocalBackend(self.path,readonly=True)
        try:
            with self.assertRaises(Conflict):
                local.delete_event(event['id'],1)
        finally:
            local.close()

    def test_revision_conflict_preserves_queue_and_explicit_archive(self):
        from fastapi.testclient import TestClient
        from decision.vlm.annotation.service import create_app
        from decision.vlm.annotation.remote import RemoteBackend
        with TestClient(create_app(self.path, writable=True)) as client:
            remote = RemoteBackend('http://testserver', 'wheelchair-vlm', cache_dir=Path(self.temporary.name)/'cache', transport=client._transport)
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0], 0, 3, '远程事件')
                result = remote.commit(event, 0)
                stale = copy.deepcopy(event)
                stale['name'] = '过期编辑'
                with self.assertRaises(Conflict):
                    remote.commit(stale, 0)
                self.assertEqual(remote.pending(), 1)
                self.assertEqual(Store(self.path).events()[0]['document']['name'], '远程事件')
                remote.archive_conflicts(stale)
                self.assertEqual(remote.pending(), 0)
                self.assertEqual(remote.drafts()['archives'][0]['current_event']['name'], '过期编辑')
            finally:
                remote.close()

    def test_unknown_commit_result_is_recovered_by_original_receipt(self):
        import httpx
        from decision.vlm.annotation.remote import RemoteBackend
        from decision.vlm.annotation.service import create_app
        from fastapi.testclient import TestClient
        with TestClient(create_app(self.path, writable=True)) as client:
            class LoseAck(httpx.BaseTransport):
                lost = False
                def handle_request(self, request):
                    response = client._transport.handle_request(request)
                    if request.url.path.endswith('/commit') and not self.lost:
                        self.lost = True
                        response.read()
                        raise httpx.ReadError('simulated lost acknowledgement')
                    return response
            cache = Path(self.temporary.name)/'cache'
            remote = RemoteBackend('http://testserver', 'wheelchair-vlm', cache_dir=cache, transport=LoseAck())
            try:
                index = remote.snapshot()
                event = new_event(index['sources'][0], 0, 3, '断网事件')
                response = remote.commit(event, 0)
                self.assertTrue(response['staged'])
                self.assertEqual(remote.pending(), 1)
                self.assertEqual(Store(self.path).events()[0]['revision'], 1)
                remote.snapshot()
                self.assertEqual(remote.pending(), 0)
                self.assertEqual(Store(self.path).events()[0]['revision'], 1)
            finally:
                remote.close()


if __name__ == '__main__':
    unittest.main()
