"""Candidate binding and phase feedback boundaries; no robot or model needed."""
import copy
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from decision.vlm.annotation.rules import digest
from decision.vlm.seating import geometry, inputs, inference
from decision.vlm.seating.__main__ import main
from decision.vlm.seating.inputs import select_poses
from decision.vlm.seating.stage import CompletionLimits,SeatingStage


def candidates():
    return dict(instruction='在桌边就座',observation_id='scan1',stamp_s=10.,frame_id='odom',
        clock_domain='ros_sim_time',session_id='s1',generation=0,revision=0,
        candidates=[dict(id=0,pose=[1.,2.,0.],motion='forward',geometry='unknown'),
                    dict(id=3,pose=[3.,2.,0.],motion='forward',geometry='blocked')])


class EdgeGeometryTests(unittest.TestCase):
    def context(self):
        return dict(kind='historical_footprint_edge_hypothesis',pose_reference='base_link_hypothesis',
            reference_sha256='a'*64,reference_status='historical_image_reference_current_platform_not_confirmed',
            footprint_xy_m=[[-.28,-.33],[.82,-.33],[.82,.33],[-.28,.33]],
            base_from_lidar=np.eye(4).tolist(),edge=dict(observed_segment_xy=[[0.,0.],[3.,0.]],
                estimated_interior_normal_xy=[0.,1.]),source_cutoff_s=10.,frame_id='odom',session_id='s1',
            generation=0,source_bindings=[dict(stamp_s=10.,observation_sha256='b'*64)],
            front_gaps_m=[.05,.15],end_guard_m=.1)

    def test_perpendicular_body_front_gap_and_no_extent_extrapolation(self):
        context=self.context();data=candidates();rows=geometry.edge_candidates(context,data)
        self.assertEqual(len(rows),8)
        for row in rows:
            polygon=geometry.footprint_world(context['footprint_xy_m'],row['pose'])
            self.assertAlmostEqual(row['pose'][2],np.pi/2)
            self.assertGreaterEqual(polygon[:,0].min(),.1-1e-8)
            self.assertLessEqual(polygon[:,0].max(),2.9+1e-8)
            self.assertAlmostEqual(-polygon[:,1].max(),.05 if row['id']<4 else .15)
            self.assertEqual(row['geometry'],'unknown')
        context['edge']['observed_segment_xy']=[[0.,0.],[.48,0.]]
        self.assertEqual(geometry.edge_candidates(context,data),[])

    def test_future_geometry_reset_and_context_changes_are_bound(self):
        data=candidates();context=self.context();data['context']=context
        original=inference.binding(data);inference.validate_candidate_set(data)
        context['front_gaps_m']=[.1]
        self.assertNotEqual(inference.binding(data),original)
        for key,value in [('source_cutoff_s',10.1),('session_id','reset'),('generation',1)]:
            wrong=copy.deepcopy(context);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):geometry.edge_candidates(wrong,data)

    def test_lidar_to_base_is_inverse_not_forward(self):
        base_from_lidar=np.eye(4);base_from_lidar[:3,3]=[.67,.33,-.2]
        pose=geometry.base_pose(np.eye(4),base_from_lidar)
        np.testing.assert_allclose(pose[:3,3],[-.67,-.33,.2])

    def test_line_fit_retains_only_observed_extent(self):
        rng=np.random.default_rng(2);x=np.linspace(0,2,200)
        cloud=np.c_[x,np.ones_like(x)+rng.normal(0,.003,len(x)),np.zeros_like(x)]
        edge=geometry.fit_edge(cloud,[1.,0.])
        self.assertGreater(edge['observed_length_m'],1.8)
        self.assertLess(edge['observed_length_m'],2.)
        self.assertGreater(edge['estimated_interior_normal_xy'][1],.99)
        self.assertFalse(edge['endpoints_are_confirmed_corners'])

    def test_missing_ground_and_zero_returns_never_prove_clearance(self):
        context=self.context();rows=geometry.edge_candidates(context,candidates())
        points=np.array([[1.,1.,0.]])
        with patch('perception.environment.geometry.ground_plane',return_value=None):
            result=geometry.overlap_diagnostics(points,[0,0,1],np.eye(4),context,rows,{})
        self.assertEqual(result['status'],'ground_unavailable')
        self.assertEqual(result['reachability'],'unknown')
        with patch('perception.environment.geometry.ground_plane',return_value=np.array([0.,0.,-.2])):
            result=geometry.overlap_diagnostics(points,[0,0,1],np.eye(4),context,rows,{})
        self.assertEqual(result['ground_status'],'unreviewed_plane_candidate')
        self.assertTrue(all(c['geometry']=='unknown' for c in result['candidates']))

    def policy(self):
        return json.loads((Path(__file__).resolve().parents[1]/'decision/vlm/configs/seating_geometry.json').read_text())

    def observation(self):
        row=candidates()
        row.pop('candidates')
        row['world_from_lidar']=np.eye(4).tolist()
        return row

    def refine(self,scans=None):
        context=self.context();context['source_bindings']=[dict(observation_id='scan1',stamp_s=10.,observation_sha256='b'*64)]
        context['edge']['observed_segment_xyz']=[[0.,0.,1.],[3.,0.,1.]]
        obs=self.observation()
        if scans is None:scans=[dict(observation=obs,world_points=np.empty((0,3)))]
        return geometry.refine_candidates(context,obs,scans,self.policy())

    def test_oblique_signed_gaps_are_explicit_and_preserve_unknown(self):
        context,rows,report=self.refine()
        self.assertEqual(len(rows),8);self.assertFalse(report['ground_plane_used'])
        self.assertFalse(report['selection_uses_demo_endpoint'])
        angles={round(np.degrees(c['pose'][2]-np.pi/2)) for c in rows}
        self.assertTrue(angles<={-10,0,10});self.assertTrue(angles- {0})
        for c in rows:
            poly=geometry.footprint_world(context['footprint_xy_m'],c['pose'])
            self.assertGreaterEqual(poly[:,0].min(),.1-1e-8)
            self.assertLessEqual(poly[:,0].max(),2.9+1e-8)
            self.assertAlmostEqual(-poly[:,1].max(),-.05 if c['id']<4 else .05)
            self.assertEqual(c['geometry'],'unknown')
        for a,b in zip(rows[:3],rows[1:4]):
            self.assertGreaterEqual(b['pose'][0]-a['pose'][0],context['sampling_policy']['min_along_separation_m']-1e-8)
        denied=self.policy();denied['footrest_under_table_allowed']=False
        with self.assertRaises(ValueError):geometry.validate_sampling_policy(denied)

    def test_support_has_no_floor_dependency_and_never_accepts_future_scans(self):
        obs=self.observation()
        points=np.array([[1.2,-.2,.72],[1.2,-.2,.721]])
        with patch('perception.environment.geometry.ground_plane',side_effect=AssertionError('No floor fit')):
            context,rows,_=self.refine([dict(observation=obs,world_points=points)])
        self.assertTrue(all(c['geometry']=='unknown' for c in rows))
        current=context['table_band_evidence']['sources'][0]
        self.assertTrue(all(c['total_voxels']<=1 for c in current['candidates']))
        data=candidates();data['candidates']=rows;data['context']=context
        inference.validate_candidate_set(data)
        changed=copy.deepcopy(data);changed['context']['table_band_evidence']['sources'][0]['stamp_s']=11.
        with self.assertRaises(ValueError):inference.validate_candidate_set(changed)
        future=copy.deepcopy(obs);future['stamp_s']=10.1
        with self.assertRaises(ValueError):self.refine([dict(observation=future,world_points=points)])

    def test_approach_distances_use_current_hypothetical_base(self):
        context=self.context();context['sampling_policy']=self.policy()
        transform=np.eye(4);transform[:3,3]=[.67,.33,-.2];context['base_from_lidar']=transform.tolist()
        rows=[dict(id=0,pose=[-.67,-.33,0.]),dict(id=1,pose=[.33,-.33,np.pi/2])]
        fields=geometry.model_context(context,np.eye(4),rows,include_approach=True)
        self.assertEqual(fields['candidate_approach'],[
            dict(id=0,straight_distance_m=0.,heading_change_deg=0.),
            dict(id=1,straight_distance_m=1.,heading_change_deg=90.)])
        self.assertNotIn('demo_endpoint',fields)

    def test_prompt_routing_is_pre_request_and_requires_both_distance_and_heading(self):
        context=self.context();gate={'distance_m':.15,'heading_deg':15.}
        goal=dict(id=0,pose=[.1,0.,.1],geometry='unknown')
        mode=geometry.selection_context(context,np.eye(4),[goal],gate)
        self.assertEqual(mode['mode'],'adjust_nearby_pose')
        for pose in [[.3,0.,0.],[.1,0.,np.pi/2]]:
            row=dict(goal,pose=pose)
            self.assertEqual(geometry.selection_context(context,np.eye(4),[row],gate)['mode'],'select_location')
        blocked=dict(goal,geometry='blocked')
        self.assertEqual(geometry.selection_context(context,np.eye(4),[blocked],gate)['mode'],'select_location')


class WindowTests(unittest.TestCase):
    def rows(self):
        return [dict(frame_id='odom',clock_domain='ros_sim_time',session_id='s1',generation=0,
                     stamp_s=t,rgb_stamp_s=t-.03) for t in (9.,9.5,10.)]

    def table(self):
        return dict(kind='approximate_table_outline',id='table',metric_verified=False,
            polygon_xy_m=[[0.,0.],[2.8,0.],[2.8,1.],[0.,1.]],edge_z_m=.05,
            available_after_s=8.,sources=[dict(stamp_s=8.,observation_sha256='a'*64)],
            frame_id='odom',clock_domain='ros_sim_time',session_id='s1',generation=0)

    def test_three_frame_window_rejects_future_rgb_wrong_intervals_and_reset(self):
        rows=self.rows();inputs.validate_window(rows)
        for index,key,value in [(0,'stamp_s',8.),(1,'session_id','reset'),(2,'rgb_stamp_s',10.01),(1,'stamp_s',10.)]:
            wrong=copy.deepcopy(rows);wrong[index][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):inputs.validate_window(wrong)

    def test_approximate_prior_cannot_be_used_before_its_source_or_in_another_frame(self):
        table=self.table();row=self.rows()[-1]
        geometry.validate_prior(table,row)
        for key,value in [('stamp_s',7.),('frame_id','map'),('generation',2)]:
            with self.subTest(key=key),self.assertRaises(ValueError):geometry.validate_prior(table,dict(row,**{key:value}))
        table['polygon_xy_m']=[[0,0],[2.8,0],[0,1],[2.8,1]]
        with self.assertRaises(ValueError):geometry.validate_prior(table,row)

    def test_generation_and_three_images_are_causal_bound_and_unexecuted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[]
            for i,row in enumerate(self.rows()):
                p=root/str(i);p.mkdir();paths.append(p)
                Image.new('RGB',(640,480)).save(p/'rgb.png');np.save(p/'scan.npy',np.empty((0,4),np.float32))
                row.update(observation_id=str(i),world_from_lidar=np.eye(4).tolist(),
                    assets={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ('rgb.png','scan.npy')})
                (p/'observation.json').write_text(json.dumps(row))
            ref=Path(__file__).resolve().parents[1]/'configs/wheelchair_geometry_reference.json'
            policy=EdgeGeometryTests().policy()
            context,goals,report=geometry.generate(self.table(),paths,ref,policy)
            self.assertFalse(report['executable']);self.assertEqual(len(goals),8)
            self.assertEqual({x['edge_id'] for x in context['candidate_evidence']},{0,1,2,3})
            self.assertTrue(all(x['geometry']=='unknown' for x in goals))
            data={k:row[k] for k in (*geometry.IDENTITY,'observation_id','stamp_s')}
            data.update(instruction='就座',revision=0,context=context,candidates=goals)
            inference.validate_candidate_set(data)
            record=inputs.freeze_window(paths,data,root/'window')
            self.assertEqual(len(record['images']),3);self.assertEqual(record['fields']['dt'],[-1.,-.5,0.])
            with Image.open(root/'window/frame_2.png') as image:
                self.assertEqual(image.size,(1280,576))
            (root/'table.json').write_text(json.dumps(self.table()))
            instruction='选择桌子的另一侧，保持方便退出的朝向。'
            main(['prepare-window','--observations',*map(str,paths),
                '--table-prior',str(root/'table.json'),'--vehicle-reference',str(ref),
                '--instruction',instruction,'--out',str(root/'custom')])
            custom=json.loads((root/'custom/input.json').read_text())
            self.assertEqual(custom['fields']['instruction'],instruction)
            self.assertEqual(custom['candidate_set']['instruction'],instruction)
            self.assertNotEqual(custom['candidate_binding'],record['candidate_binding'])
            (root/'window/frame_1.png').write_bytes(b'changed')
            with patch('httpx.Client') as client,self.assertRaises(ValueError):inference.run(root/'window','http://localhost/v1')
            client.assert_not_called()


class CandidateTests(unittest.TestCase):
    def test_sampling_never_uses_future_stale_or_pre_failure_pose(self):
        rows=[dict(stamp_s=10.,stamp_ns=100,health=0),dict(stamp_s=10.1,stamp_ns=101,health=2)]
        self.assertEqual(select_poses(rows,10,[0.])[0]['stamp_ns'],100)
        for offsets in [[.1],[1.],[0.,0.],[float('nan')]]:
            with self.subTest(offsets=offsets),self.assertRaises(ValueError):select_poses(rows,10,offsets)

    def test_current_ids_and_unknown_are_distinct_from_blocked(self):
        data=candidates()
        inference.validate_selection({'p':0,'s':'v'},data)
        for response in [{'p':3,'s':'v'},{'p':1,'s':'v'},{'p':0.,'s':'v'},{'p':True,'s':'v'},
                         {'p':0,'s':'u'},{'p':-1,'s':'v'},{'p':0,'s':'v','reason':'x'}]:
            with self.subTest(response=response),self.assertRaises(Exception):inference.validate_selection(response,data)
        for state in ['n','u']:inference.validate_selection({'p':-1,'s':state},data)

    def test_empty_candidates_cannot_return_a_goal(self):
        data=candidates();data['candidates']=[]
        inference.validate_selection({'p':-1,'s':'n'},data)
        with self.assertRaises(Exception):inference.validate_selection({'p':0,'s':'v'},data)

    def test_table_hypotheses_do_not_assert_reachability(self):
        table=dict(center_xy=[0.,0.],yaw_rad=0.,length_m=2.,width_m=1.,source_observation_id='o',source_stamp_s=1.)
        rows=geometry.table_candidates(table,stand_off_m=.6,spacing_m=.4)
        self.assertEqual(len(rows),6);self.assertTrue(all(row['geometry']=='unknown' for row in rows))
        for row in rows:
            self.assertLess(np.dot(np.array(row['pose'][:2]),[np.cos(row['pose'][2]),np.sin(row['pose'][2])]),0)

    def test_frozen_input_changes_stop_before_http(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'input.json').write_text(json.dumps(dict(input_fingerprint='changed')))
            with patch('httpx.Client') as client,self.assertRaises(ValueError):inference.run(root,'http://localhost/v1')
            client.assert_not_called()

    def test_remote_raw_failure_is_saved_once_without_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);root=parent/'input';root.mkdir()
            Image.new('RGB',(16,16)).save(parent/'display.png')
            (root/'image.png').symlink_to('../display.png')
            data=candidates();item=dict(candidate_set=data,candidate_binding=inference.binding(data),
                image_sha256=hashlib.sha256((root/'image.png').read_bytes()).hexdigest(),fields={'instruction':data['instruction']},
                response_schema=inference.selection_schema(data))
            item['input_fingerprint']=digest(item);(root/'input.json').write_text(json.dumps(item))
            posts=[]
            def handler(request):
                if request.method=='GET':
                    return httpx.Response(200,json={'data':[{'id':'qwen3.8-27b-fp8'}]} if request.url.path.endswith('/models') else {'server_args':{'speculative_algorithm':'DSPARK'}})
                posts.append(request)
                raw=dict(model='qwen3.8-27b-fp8',choices=[dict(index=0,delta=dict(content='{invalid'),finish_reason='stop')])
                return httpx.Response(200,text='data: '+json.dumps(raw)+'\n\ndata: [DONE]\n\n')
            client=httpx.Client(transport=httpx.MockTransport(handler))
            with patch('httpx.Client',return_value=client),self.assertRaises(RuntimeError):inference.run(root,'http://localhost/v1')
            self.assertEqual(len(posts),1)
            saved=json.loads((root/'remote/warmup.json').read_text())
            self.assertEqual(saved['raw_content'],'{invalid');self.assertFalse(saved['valid'])


class StageTests(unittest.TestCase):
    def setUp(self):
        self.data=candidates();self.stage=SeatingStage('task',self.data,CompletionLimits(.15,.15,.02,.03,1.,.3,.5),started_s=10.,timeout_s=30.)

    def begin(self):
        self.stage.select({'p':0,'s':'v'},self.stage.binding)
        return self.stage.geometry(task_id='task',candidate_binding=self.stage.binding,feasible=True,source='synthetic_unit_fixture',stamp_s=10.,now_s=10.)

    def feedback(self,t,**kwargs):
        row=dict(task_id='task',candidate_binding=self.stage.binding,goal_id=0,frame_id='odom',clock_domain='ros_sim_time',
            session_id='s1',generation=0,stamp_s=t,health='valid',geometry_valid=True,navigation_state='arrived',
            stopped=False,x_m=1.,y_m=2.,yaw_rad=0.,linear_m_s=0.,angular_rad_s=0.,table_relation_valid=True)
        row.update(kwargs);return row

    def test_selection_does_not_start_navigation_without_geometry(self):
        result=self.stage.select({'p':0,'s':'v'},self.stage.binding)
        self.assertEqual(result['action']['operation'],'validate_goal')
        with self.assertRaises(ValueError):self.stage.observe(self.feedback(10),now_s=10)
        rejected=self.stage.geometry(task_id='task',candidate_binding=self.stage.binding,feasible=False,source='synthetic_unit_fixture',stamp_s=10,now_s=10)
        self.assertEqual(rejected['state'],'failed');self.assertIsNone(rejected['action'])

    def test_stale_selection_cannot_replace_a_current_goal(self):
        with self.assertRaises(ValueError):self.stage.select({'p':0,'s':'v'},'old')
        self.assertEqual(self.stage.state,'selecting')
        self.begin()
        with self.assertRaises(ValueError):self.stage.select({'p':0,'s':'v'},self.stage.binding)

    def test_completion_requires_dwell_and_stop_ack(self):
        self.begin()
        for t in [10.,10.2,10.4,10.6,10.8]:
            self.assertEqual(self.stage.observe(self.feedback(t),now_s=t)['state'],'verifying')
        result=self.stage.observe(self.feedback(11.),now_s=11.)
        self.assertEqual(result['state'],'stopping');self.assertEqual(result['action']['operation'],'stop')
        result=self.stage.observe(self.feedback(11.1,stopped=True),now_s=11.1)
        self.assertEqual(result['state'],'succeeded');self.assertFalse(self.stage.control_active)

    def test_wrong_yaw_table_relation_or_motion_never_completes(self):
        self.begin()
        for i,changes in enumerate([dict(yaw_rad=1.),dict(table_relation_valid=False),dict(linear_m_s=.1),dict(navigation_state='running')]):
            t=10.+i*.2;row=self.feedback(t,**changes)
            self.assertNotIn(self.stage.observe(row,now_s=t)['state'],{'stopping','succeeded'})
            self.assertIsNone(self.stage.since)

    def test_missing_interval_breaks_stable_hold(self):
        self.begin();self.stage.observe(self.feedback(10),now_s=10)
        self.assertEqual(self.stage.observe(self.feedback(11),now_s=11)['state'],'verifying')
        self.assertEqual(self.stage.since,11)

    def test_invalid_pose_requires_stop_and_ack_even_if_pose_stays_invalid(self):
        self.begin();result=self.stage.observe(self.feedback(10,health='invalid'),now_s=10)
        self.assertEqual(result['state'],'stopping')
        result=self.stage.observe(self.feedback(10.1,health='invalid',stopped=True),now_s=10.1)
        self.assertEqual(result['state'],'failed')

    def test_wrong_identity_and_late_feedback_leave_current_state_intact(self):
        self.begin()
        with self.assertRaises(ValueError):self.stage.observe(self.feedback(10,generation=1),now_s=10)
        self.assertEqual(self.stage.state,'approaching')
        self.stage.observe(self.feedback(10),now_s=10)
        with self.assertRaises(ValueError):self.stage.observe(self.feedback(10),now_s=10)

    def test_cancellation_waits_for_control_stop(self):
        self.begin();self.assertEqual(self.stage.cancel()['state'],'stopping')
        result=self.stage.observe(self.feedback(10,stopped=True),now_s=10)
        self.assertEqual(result['state'],'cancelled')

    def test_source_and_returned_actions_cannot_mutate_the_goal(self):
        self.data['candidates'][0]['pose'][0]=99
        result=self.begin();result['action']['goal']['pose'][0]=88
        self.assertEqual(self.stage.goal['pose'][0],1.)

    def test_old_feedback_cannot_seed_completion_hold(self):
        self.begin()
        with self.assertRaises(ValueError):self.stage.observe(self.feedback(9.9),now_s=10)
        self.assertIsNone(self.stage.since)

    def test_completion_is_not_accepted_after_a_feedback_gap(self):
        self.begin()
        for t in [10.,10.2,10.4,10.6,10.8,11.]:self.stage.observe(self.feedback(t),now_s=t)
        result=self.stage.observe(self.feedback(12.,stopped=True),now_s=12.)
        self.assertEqual(result['state'],'stopping');self.assertEqual(self.stage.pending,'failed')

    def test_missing_feedback_and_overall_deadline_have_explicit_tick(self):
        self.begin();self.assertEqual(self.stage.tick(10.6)['state'],'stopping')
        self.assertEqual(self.stage.reason,'feedback_timeout')
        waiting=SeatingStage('waiting',self.data,self.stage.limits,started_s=10.,timeout_s=2.)
        self.assertEqual(waiting.tick(12.)['state'],'failed')
        self.assertEqual(waiting.reason,'task_timeout')


if __name__=='__main__':unittest.main()
