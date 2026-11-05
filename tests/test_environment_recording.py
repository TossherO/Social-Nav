"""Archive independence and live-source exports; no ROS master or model."""
import json
from pathlib import Path
import tempfile
import threading
import sqlite3
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import yaml

from perception.environment.scans import AsyncScanJournal,ScanJournal,read_scans,sealed_scan_summary
from perception.environment.layers import load_config,GridSnapshot
from perception.environment.export import export_environment
from perception.environment.maps import static_geometry
from perception.mapping_localization.results import file_fingerprint,map_fingerprint


class RecordingTests(unittest.TestCase):
    def test_failed_marker_overrides_a_leftover_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            writer=ScanJournal(path,load_config()['scan_recording'],{})
            writer.append(np.ones((1,3),np.float32),np.eye(4),10**9,dict(rays_used=False));writer.close()
            with sqlite3.connect(path) as database:
                database.execute("INSERT INTO metadata VALUES ('failed','{}')")
            with self.assertRaises(ValueError):sealed_scan_summary(path)
            with self.assertRaises(ValueError):list(read_scans(path))

    def test_publication_precedes_archive_failure_without_invalidating_geometry(self):
        from perception.environment.ros.node import EnvironmentNode
        from perception.environment.layers import GridSnapshot
        node=EnvironmentNode.__new__(EnvironmentNode);events=[]
        snapshot=GridSnapshot(dict(stamp_s=1.,geometry_status='OBSERVED',rays_used=False),{})
        node.config=load_config();node.rate_identity,node.rate_anchor,node.rate_tick=None,None,0
        node.engine=SimpleNamespace(last_stamp=None,update=lambda *a,**k:snapshot,
            reset=lambda:events.append('reset'))
        node.counts=dict(processed=0,rays_used=0,geometry_errors=0)
        node.timings=[];node.publish_timings=[];node.observation_ages=[]
        node.record_scans=True;node.recording_error=None;node.clock='test'
        def submit(*args):events.append('archive');return False
        node.journal=SimpleNamespace(submit=submit,summary=lambda:dict(error='disk unavailable'))
        node.publish=lambda *args:events.append('publish');node.publish_status=lambda:None
        node.rospy=SimpleNamespace(Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=lambda:1.1)),
            logerr_throttle=lambda *args:None)
        pose=SimpleNamespace(health=0,VALID=0,session_id='s',generation=0,source='fast_lio',child_frame_id='rslidar',
            header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:1.),frame_id='odom'),
            pose=SimpleNamespace(position=SimpleNamespace(x=0,y=0,z=0),orientation=SimpleNamespace(x=0,y=0,z=0,w=1)))
        with patch('perception.environment.ros.node.pointcloud_fields',return_value=dict(x=np.ones(1),y=np.zeros(1),z=np.zeros(1))):
            node.process(10**9,SimpleNamespace(fields=[]),pose,.1,{})
        self.assertEqual(events,['publish','archive'])
        self.assertEqual(node.counts['geometry_errors'],0)
        self.assertEqual(node.status,'OBSERVED');self.assertEqual(node.recording_error,'disk unavailable')

    def test_archive_freezes_inputs_and_drains_before_sealing(self):
        entered,release=threading.Event(),threading.Event()
        class SlowJournal(ScanJournal):
            def append(self,*args):
                entered.set();release.wait(2);return super().append(*args)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            journal=AsyncScanJournal(path,load_config()['scan_recording'],{},writer_factory=SlowJournal)
            points=np.array([[1.,0,.5]],np.float32);metadata=dict(rays_used=False,ray_origin_model='hits_only')
            try:
                self.assertTrue(journal.submit(points,np.eye(4),10**9,metadata))
                self.assertTrue(entered.wait(1));points[:]=99;metadata['rays_used']=True
                self.assertEqual(journal.summary()['pending'],1)
            finally:release.set();journal.close()
            records=list(read_scans(path))
            np.testing.assert_array_equal(records[0][1]['points'],[[1.,0,.5]])
            self.assertFalse(records[0][0]['rays_used'])
            self.assertEqual(journal.summary()['state'],'sealed')
            self.assertEqual(sealed_scan_summary(path)['records'],1)
            self.assertFalse(journal.submit(points,np.eye(4),2*10**9,metadata))

    def test_overflow_is_nonblocking_explicit_and_never_sealed(self):
        entered,release=threading.Event(),threading.Event()
        class SlowJournal(ScanJournal):
            def append(self,*args):
                entered.set();release.wait(2);return super().append(*args)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            journal=AsyncScanJournal(path,dict(load_config()['scan_recording'],queue_size=1),{},writer_factory=SlowJournal)
            args=(np.ones((1,3),np.float32),np.eye(4),10**9,dict(rays_used=False))
            try:
                self.assertTrue(journal.submit(*args));self.assertTrue(entered.wait(1))
                self.assertFalse(journal.submit(args[0],args[1],2*10**9,args[3]))
                self.assertEqual(journal.summary()['state'],'failed')
            finally:
                release.set()
                with self.assertRaisesRegex(RuntimeError,'archive failed'):journal.close()
            with self.assertRaisesRegex(ValueError,'sealed'):sealed_scan_summary(path)

    def test_writer_failure_is_explicit_and_unsealed(self):
        class BrokenJournal(ScanJournal):
            def append(self,*args):raise OSError('simulated disk failure')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            journal=AsyncScanJournal(path,load_config()['scan_recording'],{},writer_factory=BrokenJournal)
            journal.submit(np.ones((1,3),np.float32),np.eye(4),10**9,dict(rays_used=False))
            with self.assertRaisesRegex(RuntimeError,'simulated disk failure'):journal.close()
            with self.assertRaises(ValueError):list(read_scans(path))

    def test_shutdown_timeout_cannot_later_become_a_sealed_success(self):
        entered,release=threading.Event(),threading.Event()
        class SlowJournal(ScanJournal):
            def append(self,*args):entered.set();release.wait(2);return super().append(*args)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            journal=AsyncScanJournal(path,load_config()['scan_recording'],{},writer_factory=SlowJournal)
            journal.submit(np.ones((1,3),np.float32),np.eye(4),10**9,dict(rays_used=False))
            try:
                self.assertTrue(entered.wait(1))
                with self.assertRaisesRegex(RuntimeError,'shutdown budget'):journal.close(timeout=.001)
            finally:
                release.set()
                with self.assertRaisesRegex(RuntimeError,'archive failed'):journal.close()
            self.assertEqual(journal.summary()['state'],'failed')
            with self.assertRaises(ValueError):sealed_scan_summary(path)

    def test_live_export_binds_journal_and_never_requires_or_fabricates_a_bag(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'live';mapping=run/'map'
            (mapping/'keyframes').mkdir(parents=True);(mapping/'COMPLETE').touch()
            (mapping/'manifest.yaml').write_text('frame: map\n')
            (mapping/'keyframes/poses.txt').write_text('1 0 0 0 0 0 0 0 0\n1.1 0 0 0 0 0 0 1 0\n')
            axis=np.arange(-2.5,2.5,.15);x,y=np.meshgrid(axis,axis)
            points=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1.)]
            with (mapping/'GlobalMap.pcd').open('w') as stream:
                stream.write('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS %d\nDATA ascii\n'%len(points));np.savetxt(stream,points)
            rows=[dict(stamp_s=t,stamp_ns=int(round(t*1e9)),position=[0,0,0],quaternion_xyzw=[0,0,0,1],session_id='live',generation=0,
                       frame_id='odom',health=0) for t in (1.,1.1)]
            (run/'localization.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            journal=run/'environment/scans.sqlite'
            writer=ScanJournal(journal,load_config()['scan_recording'],dict(source='synthetic_live_fixture'))
            writer.append(points,np.eye(4),1100000000,dict(session_id='live',generation=0,frame_id='odom',
                rays_used=False,ray_origin_model='hits_only'));writer.close()
            record=dict(success=True,source='live',mode='mapping',bag=None,use_sim_time=True,
                environment_scans_fingerprint=file_fingerprint(journal),
                playback=dict(bag_fingerprint=None,map_fingerprint=map_fingerprint(mapping)))
            (run/'run.json').write_text(json.dumps(record))
            manifest=export_environment(run,root/'environment',export_pgm=True)
            self.assertEqual(manifest['source']['input_kind'],'live')
            self.assertIsNone(manifest['source']['bag_fingerprint'])
            self.assertNotIn('bag',manifest['source'])
            self.assertTrue((root/'environment/map.pgm').is_file())
            snapshot=GridSnapshot.load(root/'environment/layers.npz')
            self.assertEqual(snapshot.metadata['dynamic_filter']['mode'],'geometric')
            self.assertIsNone(snapshot.metadata['dynamic_filter']['people'])
            (run/'people.jsonl').write_text(json.dumps(dict(stamp_s=1.1,observation_id='live:1.1',
                health='valid',frame_id='odom',clock_domain='ros_sim_time',tracks=[],
                localization_binding=dict(session_id='live',generation=0,frame_id='odom')))+'\n')
            export_environment(run,root/'with_people',people=run)
            self.assertIsNotNone(GridSnapshot.load(root/'with_people/layers.npz').metadata['dynamic_filter']['people'])
            # Removed modes fail explicitly instead of changing the requested export.
            with self.assertRaisesRegex(ValueError,'Unknown static dynamic-filter mode'):
                export_environment(run,root/'missing_people',filter_mode='people')
            config=load_config()
            unfiltered=export_environment(run,root/'unfiltered',config=config,filter_mode='none')
            self.assertEqual(unfiltered['configuration']['static']['dynamic_filter'],'none')
            self.assertEqual(config['static']['dynamic_filter'],'geometric')
            self.assertNotIn('dynamic_filter',GridSnapshot.load(root/'unfiltered/layers.npz').metadata)
            config['static']['dynamic_filter']='none'
            export_environment(run,root/'configured_unfiltered',config=config)
            self.assertNotIn('dynamic_filter',GridSnapshot.load(root/'configured_unfiltered/layers.npz').metadata)
            with self.assertRaisesRegex(ValueError,'own bound sealed'):export_environment(run,root/'wrong',bag='other.bag')
            with journal.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'no longer matches'):export_environment(run,root/'changed')

    def test_region_is_explicit_bounded_and_has_distinct_identity(self):
        config=load_config();axis=np.arange(-2.5,2.5,.1);x,y=np.meshgrid(axis,axis)
        points=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1.)]
        config['static']['max_cells']=30
        with self.assertRaisesRegex(ValueError,'budget'):static_geometry(points,[[0,0,0]],config,map_id='map',frame='map',stamp=1)
        a=static_geometry(points,[[0,0,0]],config,map_id='map',frame='map',stamp=1,bounds=[-.2,-.2,.2,.2])
        b=static_geometry(points,[[0,0,0]],config,map_id='map',frame='map',stamp=1,bounds=[-.2,-.2,.3,.2])
        self.assertEqual(a.metadata['origin_xy'],[-.2,-.2]);self.assertEqual(a.metadata['shape'],[4,4])
        self.assertNotEqual(a.metadata['snapshot_id'],b.metadata['snapshot_id'])
        self.assertLess(a.metadata['point_count'],a.metadata['source_point_count'])
        self.assertFalse((a.layers['occupancy']==0).any())


if __name__=='__main__':unittest.main()
