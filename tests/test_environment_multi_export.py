"""Chronological updates preserve unknowns and source/map identities."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from perception.environment.layers import GridSnapshot
from perception.environment.multi_export import _merge_session, _registered_source


class MultiSurveyTests(unittest.TestCase):
    def test_only_classified_measured_newer_cells_replace_existing(self):
        shape = (1, 5)
        layers = dict(occupancy=np.array([[100, 0, 100, 0, -1]],np.int8),
            observed=np.ones(shape,bool), last_observed=np.full(shape,10.),
            coverage=np.ones(shape,np.float32), obstacle_min_z=np.ones(shape,np.float32),
            obstacle_max_z=np.ones(shape,np.float32), endpoint_count=np.ones(shape,np.int32),
            survey_index=np.zeros(shape,np.int16))
        newer = dict(occupancy=np.array([[0, 100, -1, 0, 100]],np.int8),
            observed=np.array([[1,1,1,0,1]],bool), last_observed=np.array([[20.,20.,20.,20.,np.nan]]),
            coverage=np.ones(shape,np.float32), obstacle_min_z=np.full(shape,np.nan,np.float32),
            obstacle_max_z=np.full(shape,np.nan,np.float32), endpoint_count=np.zeros(shape,np.int32))
        counts = _merge_session(layers, GridSnapshot({},newer), 1)
        np.testing.assert_array_equal(layers['occupancy'], [[0,100,100,0,-1]])
        np.testing.assert_array_equal(layers['survey_index'], [[1,1,0,0,0]])
        self.assertEqual(counts['cleared_occupied'],1)
        self.assertEqual(counts['newly_occupied'],1)
        self.assertEqual(layers['last_observed'][0,2],10.)

    def test_reference_must_match_selected_map_and_source_bag(self):
        from tests.test_local_map_update import LocalMapUpdateTests
        from perception.environment.layers import load_config
        from perception.mapping_localization.results import map_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            old, mapping, reference = LocalMapUpdateTests().fixture(tmp)
            selected = _registered_source(mapping,reference,map_fingerprint(old),'map',load_config())
            self.assertEqual(selected['source']['recorded_scans'],6)
            with self.assertRaisesRegex(ValueError, 'selected final map'):
                _registered_source(mapping,reference,{'different':'map'},'map',load_config())
            ref=reference/'run.json';record=json.loads(ref.read_text())
            record['algorithm_parameters']['common']['imu_time_offset']=.1
            ref.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'different sensor extrinsics or IMU timing'):
                _registered_source(mapping,reference,map_fingerprint(old),'map',load_config())

    def test_old_free_cannot_clear_newer_or_undated_final_map_feature(self):
        shape=(1,3)
        layers=dict(occupancy=np.full(shape,100,np.int8),observed=np.ones(shape,bool),
            last_observed=np.full(shape,np.nan),coverage=np.zeros(shape,np.float32),
            obstacle_min_z=np.ones(shape,np.float32),obstacle_max_z=np.ones(shape,np.float32),
            endpoint_count=np.ones(shape,np.int32),survey_index=np.full(shape,-1,np.int16))
        newer=dict(occupancy=np.zeros(shape,np.int8),observed=np.ones(shape,bool),
            last_observed=np.full(shape,12.),coverage=np.ones(shape,np.float32),
            obstacle_min_z=np.full(shape,np.nan,np.float32),obstacle_max_z=np.full(shape,np.nan,np.float32),
            endpoint_count=np.zeros(shape,np.int32))
        _merge_session(layers,GridSnapshot({},newer),0,baseline_occupied=np.ones(shape,bool),
                       feature_times=np.array([[15.,5.,np.nan]]),survey_start=10.)
        np.testing.assert_array_equal(layers['occupancy'],[[100,0,100]])

    def test_overlapping_surveys_are_not_given_an_invented_time_order(self):
        from perception.environment.layers import load_config
        from perception.environment.multi_export import export_registered_multi_environment
        from tests.test_local_map_update import LocalMapUpdateTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old,_,_=LocalMapUpdateTests().fixture(root)
            def source(evidence, reference, *unused):
                return dict(timeline=SimpleNamespace(times=np.array([1.,3. if evidence=='a' else 4.])),
                            source={'bag_fingerprint':{'id':evidence}})
            with patch('perception.environment.multi_export._registered_source',side_effect=source):
                with self.assertRaisesRegex(ValueError,'timestamps overlap'):
                    export_registered_multi_environment(old,['a','b'],['ar','br'],root/'out',config=load_config())

    def test_cli_pairs_sessions_without_changing_single_export(self):
        from perception.environment.__main__ import bundle_main
        with patch('perception.environment.multi_export.export_registered_multi_environment',
                   return_value={'snapshot_id':'test'}) as export, contextlib.redirect_stdout(io.StringIO()):
            bundle_main(['export','--localization-map','map','--evidence-run','old','--reference-run','old-ref',
                         '--evidence-run','new','--reference-run','new-ref','--out','output'])
        self.assertEqual(export.call_args.args[:4],('map',['old','new'],['old-ref','new-ref'],'output'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bundle_main(['export','--localization-map','map','--evidence-run','one',
                         '--evidence-run','two','--reference-run','one-ref','--out','output'])

    def test_full_export_sorts_surveys_and_persists_latest_known_state(self):
        from perception.environment.layers import load_config
        from perception.environment.multi_export import export_registered_multi_environment
        from perception.environment.storage import StaticEnvironmentMap
        from tests.test_local_map_update import LocalMapUpdateTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old,_,_=LocalMapUpdateTests().fixture(root)
            shape=(3,3); ground=np.zeros(shape,np.float32); support=np.ones(shape,bool)
            layers=dict(occupancy=np.full(shape,-1,np.int8),observed=np.zeros(shape,bool),
                last_observed=np.full(shape,np.nan),coverage=np.zeros(shape,np.float32),
                obstacle_min_z=np.full(shape,np.nan,np.float32),obstacle_max_z=np.full(shape,np.nan,np.float32),
                endpoint_count=np.zeros(shape,np.int32),ground_z=ground,ground_support=support,
                clearance=np.full(shape,np.nan,np.float32),unknown_distance=np.zeros(shape,np.float32),
                frontier=np.zeros(shape,bool))
            meta=dict(snapshot_id='candidate',map_id='test',frame_id='map',shape=list(shape),origin_xy=[0.,0.],
                      resolution=.1,stamp_s=1.,entity_type='map',map_type='static_environment_map')
            candidate=GridSnapshot(meta,layers)
            def source(evidence, reference, *unused):
                start=10. if evidence=='old' else 20.
                fingerprint={'id':evidence}
                return dict(path=Path(evidence),timeline=SimpleNamespace(times=np.array([start,start+1]),
                    sensor_positions=np.array([[0.,0.,1.]])), bag=Path(evidence+'.bag'),
                    evidence=Path(evidence+'-evidence'),reference=Path(evidence+'-reference'),
                    source=dict(bag_fingerprint=fingerprint,scans_fingerprint=fingerprint,
                    evidence_run_fingerprint=fingerprint,evidence_records_fingerprint=fingerprint,
                    reference_run_fingerprint=fingerprint,reference_records_fingerprint=fingerprint,
                    evidence_run={'path':evidence}))
            def fusion(candidate, path, *unused, **kwargs):
                result={k:v.copy() for k,v in candidate.layers.items()}
                result['occupancy'].fill(-1);result['observed'].fill(False);result['last_observed'].fill(np.nan)
                result['occupancy'][1,1]=100 if str(path)=='old' else 0
                result['observed'][1,1]=True;result['last_observed'][1,1]=10. if str(path)=='old' else 20.
                return GridSnapshot(dict(fusion={},evidence_scans=4,unbound_scans=0,ray_scans=4),result)
            with patch('perception.environment.multi_export._registered_source',side_effect=source), \
                    patch('perception.environment.multi_export.static_geometry',return_value=candidate), \
                    patch('perception.environment.multi_export.file_fingerprint',side_effect=lambda path:{'id':'old' if 'old' in str(path) else 'new'}), \
                    patch('perception.environment.fusion.fuse_scans_tiled',side_effect=fusion):
                export_registered_multi_environment(old,['new','old'],['nr','or'],root/'out',config=load_config())
            result=StaticEnvironmentMap(root/'out')
            self.assertEqual(result.snapshot.layers['occupancy'][1,1],0)
            self.assertEqual(result.snapshot.layers['survey_index'][1,1],1)
            self.assertEqual(result.snapshot.layers['last_observed'][1,1],20.)
            self.assertEqual([s['evidence_run']['path'] for s in result.manifest['source']['surveys']],['old','new'])

    def test_workflow_selects_all_poses_then_inherits_each_source_configuration(self):
        from perception.environment import multi_workflow
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'output'
            runs=[dict(path=root/'mapping1',bag=root/'bag1',offset=0.),dict(path=root/'mapping2',bag=root/'bag2',offset=0.)]
            calls=[]
            def execute(command, **kwargs):
                calls.append(command)
                if 'pick_start.sh' in command[1]:
                    Path(command[command.index('--out')+1]).write_text('{}')
            args=multi_workflow.arguments(['--map',str(root/'map'),'--evidence-run','a','--evidence-run','b',
                                           '--out',str(out),'--gpu','0'])
            with patch.object(multi_workflow,'prepare',return_value=(root/'map',runs,out)), \
                    patch.object(multi_workflow.subprocess,'run',side_effect=execute):
                result=multi_workflow.run(args)
            self.assertEqual(result,out/'static_map')
            self.assertIn('pick_start.sh',calls[0][1]);self.assertIn('pick_start.sh',calls[1][1])
            for command, source in zip(calls[2:4],runs):
                self.assertEqual(command[command.index('--source-run')+1],str(source['path']))
                self.assertEqual(command[command.index('--bag')+1],str(source['bag']))
            self.assertEqual(calls[4].count('--evidence-run'),2)
            self.assertEqual(calls[4].count('--reference-run'),2)

    def test_automatic_initial_pose_requires_unchanged_map_lineage(self):
        from perception.environment.multi_workflow import _ancestor_map
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); parent=root/'parent'; target=root/'target'
            for path in (parent,target):
                (path/'keyframes').mkdir(parents=True)
                (path/'keyframes/poses.txt').write_text('1 0 0 0 0 0 0 0 1\n')
            (parent/'manifest.yaml').write_text('frame: map\n')
            report=json.dumps({'old_map':{'base':'absolute','path':str(parent)},
                               'old_binding':{'map_fingerprint':{'id':'parent'}}}).encode()
            (target/'update_source.json').write_bytes(report)
            (target/'manifest.yaml').write_text('frame: map\nsource: local_bag_update_candidate\nupdate_source_sha256: '+hashlib.sha256(report).hexdigest()+'\n')
            def fingerprint(path):return {'id':Path(path).name}
            with patch('perception.environment.multi_workflow.map_fingerprint',side_effect=fingerprint):
                self.assertEqual(_ancestor_map(target,{'id':'parent'}),parent)
                (target/'keyframes/poses.txt').write_text('1 2 0 0 0 0 0 0 1\n')
                with self.assertRaisesRegex(ValueError,'preserve parent keyframe'):
                    _ancestor_map(target,{'id':'parent'})

    def test_real_survey_fusion_does_not_inject_unobserved_final_endpoints(self):
        from perception.environment.fusion import fuse_scans_tiled
        from perception.environment.layers import load_config
        from perception.environment.maps import static_geometry
        from perception.environment.poses import PoseTimeline
        from perception.environment.scans import ScanJournal
        config=load_config()
        config['static']['tile_size_m']=.5
        axis=np.arange(-1.,1.1,.1); x,y=np.meshgrid(axis,axis)
        ground=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1.)]
        historical=np.array([[.75,.75,-.5],[.75,.75,0.]])
        points=np.r_[ground,historical,[[5.,5.,-1.]]].astype(np.float32)
        candidate=static_geometry(points,[[0,0,0]],config,map_id='test',frame='map',stamp=1.)
        times=(1.,1.6,2.2,2.8)
        rows=[dict(stamp_s=t,stamp_ns=int(t*1e9),position=[0,0,0],quaternion_xyzw=[0,0,0,1],
                   session_id='local',generation=0,frame_id='odom',health=0) for t in times]
        timeline=PoseTimeline(rows,config['poses'],[[1.,0,0,0,0,0,0,0,0]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scans.sqlite';writer=ScanJournal(path,config['scan_recording'],{})
            meta=dict(session_id='local',generation=0,frame_id='odom',rays_used=True,
                      ray_origin_model='pose_interpolated_origins')
            for stamp in times:
                writer.append(ground.astype(np.float32),np.eye(4),int(stamp*1e9),meta,np.zeros_like(ground,np.float32))
            writer.close()
            result=fuse_scans_tiled(candidate,path,timeline,config,{},filter_mode='geometric',
                                    candidate_points=np.empty((0,3),np.float32))
        self.assertEqual(int((candidate.layers['occupancy']==100).sum()),1)
        self.assertEqual(int((result.layers['occupancy']==100).sum()),0)

    def test_real_paired_survey_export_keeps_time_and_source_bindings(self):
        from perception.environment.layers import load_config
        from perception.environment.multi_export import export_registered_multi_environment
        from perception.environment.scans import ScanJournal,read_scans
        from perception.mapping_localization.results import file_fingerprint,map_fingerprint
        from tests.test_local_map_update import LocalMapUpdateTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); first=root/'first';second=root/'second';first.mkdir();second.mkdir()
            old,mapping1,reference1=LocalMapUpdateTests().fixture(first)
            _,mapping2,reference2=LocalMapUpdateTests().fixture(second)
            bag=second/'input.bag';bag.write_bytes(b'different bag')
            for folder in (mapping2,reference2):
                path=folder/'localization.jsonl'
                records=[json.loads(line) for line in path.read_text().splitlines()]
                for row in records:
                    row['stamp_s']+=100.;row['stamp_ns']+=100_000_000_000
                path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            original=mapping2/'environment/scans.sqlite'; data=list(read_scans(original))
            original.unlink();writer=ScanJournal(original,load_config()['scan_recording'],{})
            for record,arrays in data:
                stamp=record.pop('stamp_ns')+100_000_000_000
                writer.append(arrays['points'],arrays['local_from_lidar'],stamp,record,
                              arrays.get('origins'),arrays.get('offset_ms'))
            writer.close()
            for folder in (mapping2,reference2):
                path=folder/'run.json';record=json.loads(path.read_text())
                record['playback']['bag_fingerprint']=file_fingerprint(bag)
                if folder==reference2:record['playback']['map_fingerprint']=map_fingerprint(old)
                else:record['environment_scans_fingerprint']=file_fingerprint(original)
                path.write_text(json.dumps(record))
            result=export_registered_multi_environment(old,[mapping2,mapping1],[reference2,reference1],root/'out')
            sessions=result['source']['surveys']
            self.assertEqual([s['recorded_scans'] for s in sessions],[6,6])
            self.assertEqual([s['observation_interval_s'][0] for s in sessions],[200.,300.])
            self.assertTrue((root/'out/layers.npz').is_file())


if __name__ == '__main__':
    unittest.main()
