"""Direct-control causality, bounded actions, real braking and reference lifecycle."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from perception.environment.context import fingerprint
from navigation.control.dwa import DwaController
from navigation.control.runtime import DwaNavigationControl
from navigation.planning.planner import Planner
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.continuity import reference_envelope
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.reference import ReferenceRuntime
from tests.navigation_fixtures import bundle,request
from tests.navigation_fixtures import FakeWorker,scene_at


class DwaTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config();self.cfg['algorithm']='hybrid_astar';self.cfg['local_replanning']['enabled']=False
        self.motion=load_motion_config()
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.scene=bundle();self.req=request('local',goal=[4.,2.,0.],goal_behavior='stop')
        self.feedback=dict(status='valid',source='synthetic_actual_plant',pose=[2.,2.,0.],velocity=[0.,0.],stamp_s=10.,
            clock_domain='synthetic',local_identity=['test',0,'odom'])
        result=Planner(self.cfg,self.geometry).query(self.req,scene=self.scene,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        self.reference=reference_envelope(result,self.req,self.scene,None)
        self.controller=DwaController(self.cfg,self.motion,self.geometry)

    def test_small_reachable_start_and_bounded_candidate_count_without_optimizers(self):
        r=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(r['status'],'candidate',r);self.assertGreater(r['velocity'][0],0.)
        self.assertLessEqual(r['velocity'][0],self.motion['max_acceleration_mps2']*.1+1e-9)
        self.assertLessEqual(r['candidate_count'],24);self.assertAlmostEqual(r['velocity'][1],0.)
        self.assertFalse(r['executable']);self.assertEqual(self.controller.result['trajectory']['initialization'],'dwa')
        self.assertEqual(self.reference['result']['search_backend'],'bounded_curvature_hybrid_astar')
        from navigation.planning.continuity import content_valid
        self.assertTrue(content_valid(self.controller.result))
        again=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertLessEqual(again['velocity'][0],self.motion['max_acceleration_mps2']*.1+1e-9)

    def test_expired_computation_clears_the_previous_proposal(self):
        first=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(first['status'],'candidate',first)
        self.controller.settings['max_compute_s']=1e-12
        result=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(result['reason'],'certification_time_budget_exceeded')
        self.assertEqual(result['velocity'],[0.,0.]);self.assertIsNone(self.controller.result)
        self.assertIsNone(self.controller.previous);self.assertEqual(result['certification_attempts'],0)

    def test_exact_work_allowance_is_shared_across_candidates(self):
        self.controller.settings['certification_max_checks']=3
        def uncertain(*args,**kwargs):
            status='unknown' if kwargs['budget'].check(2) else 'budget_exceeded'
            return dict(status=status)
        with patch.object(self.controller.sweep,'check',side_effect=uncertain):
            result=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(result['reason'],'certification_work_budget_exceeded')
        self.assertEqual(result['certification']['exact_checks'],2)
        self.assertEqual(result['certification_attempts'],2);self.assertIsNone(self.controller.result)

    def test_late_people_check_cannot_publish_a_previously_free_candidate(self):
        model=self.controller._model(self.scene);original=model.check;clock=[0.]
        def late(*args,**kwargs):
            result=original(*args,**kwargs);clock[0]+=.1;return result
        with patch('navigation.control.dwa.time.perf_counter',side_effect=lambda:clock[0]), \
                patch.object(model,'check',side_effect=late):
            result=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(result['reason'],'certification_time_budget_exceeded')
        self.assertIsNone(self.controller.result);self.assertEqual(result['velocity'],[0.,0.])
        self.assertGreaterEqual(result['certification']['people_ms'],100.)

    def test_measurement_triggered_shadow_uses_actual_short_control_interval(self):
        from navigation.control.shadow import DwaShadowController
        rows=[dict(stamp_s=t,position=[2.,2.,0.],quaternion_xyzw=[0.,0.,0.,1.],
            health=0,session_id='test',generation=0,frame_id='odom') for t in (9.7,9.8,9.9,10.)]
        shadow=DwaShadowController(self.cfg,self.motion,self.geometry)
        packet=shadow.step(self.req,self.scene,rows,self.reference,now=10.,clock_domain='synthetic',mono=0.,dt=.05)
        self.assertEqual(packet['proposal']['status'],'candidate',packet)
        self.assertGreater(packet['proposal']['velocity'][0],0.)
        self.assertLessEqual(packet['proposal']['velocity'][0],self.motion['max_acceleration_mps2']*.05+1e-9)

    def test_trial_profile_can_start_and_obeys_speed_and_acceleration(self):
        motion=load_motion_config(profile='trial')
        controller=DwaController(self.cfg,motion,self.geometry)
        for speed in (0.,.095):
            result=controller.command(self.req,self.scene,dict(self.feedback,velocity=[speed,0.]),
                                      self.reference,now=10.,clock_domain='synthetic')
            self.assertEqual(result['status'],'candidate',result)
            self.assertGreater(result['velocity'][0],0.)
            self.assertLessEqual(result['velocity'][0],motion['max_forward_mps'])
            self.assertLessEqual(result['velocity'][0],speed+motion['max_acceleration_mps2']*.1+1e-9)
            self.assertLessEqual(abs(result['velocity'][1]),motion['max_angular_rps'])
            self.assertFalse(result['executable'])

    def test_new_thin_wall_forces_deceleration_and_never_uses_reference_as_free_space(self):
        # Keep the wall inside the current profile's predicted stopping range.
        cells=np.zeros((60,60),np.int8);cells[:,28]=100;current=bundle(cells)
        f=dict(self.feedback,velocity=[.4,0.])
        r=self.controller.command(self.req,current,f,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(r['status'],'candidate',r)
        self.assertLessEqual(r['velocity'][0],.4+1e-9)
        self.assertGreater(sum(r['rejections'].values()),0)
        self.assertEqual(current.geometry.layers['occupancy'][20,28],100)

    def test_speed_recovery_keeps_actual_initial_state_and_can_reduce_without_full_stop(self):
        f=dict(self.feedback,velocity=[.95,0.])
        r=self.controller.command(self.req,self.scene,f,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(r['status'],'candidate',r)
        self.assertGreater(r['velocity'][0],0.);self.assertLessEqual(r['velocity'][0],self.motion['max_forward_mps'])
        self.assertEqual(f['velocity'],[.95,0.])

    def test_speed_recovery_certifies_actual_initial_motion(self):
        self.controller.diagnostics=True
        feedback=dict(self.feedback,velocity=[.95,0.])
        r=self.controller.command(self.req,self.scene,feedback,self.reference,now=10.,clock_domain='synthetic')
        choices=np.asarray(r['sampling']['choices'])
        self.assertLessEqual(len(choices),24)
        self.assertLessEqual(choices[:,0].max(),self.motion['max_forward_mps'])
        self.assertEqual(r['sampling']['initial'],[.95,0.])
        cells=np.zeros((60,60),np.int8);cells[:,25]=100
        blocked=self.controller.command(self.req,bundle(cells),feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(blocked['status'],'unavailable');self.assertIsNone(self.controller.result)

    def test_reference_recovers_after_short_outage_without_refreshing_source_time(self):
        with ReferenceRuntime(self.cfg,self.motion,self.geometry,worker_factory=FakeWorker) as runtime:
            runtime.update(self.req,self.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
            runtime.worker.reply(self.reference);runtime.tick(now=10.,clock_domain='synthetic')
            runtime.tick(now=10.6,clock_domain='synthetic');self.assertIsNone(runtime.current)
            newer=scene_at(10.7);runtime.update(self.req,newer,None,now=10.7)
            state=runtime.tick(now=10.7,clock_domain='synthetic')
            self.assertEqual(state['status'],'reference_ready');self.assertEqual(runtime.current['as_of'],10.)
            self.assertEqual(runtime.current['binding'],self.reference['binding']);self.assertIsNone(runtime.worker.job)
            runtime.cancel(self.req);self.assertIsNone(runtime.reference)

    def test_direct_reference_rejects_a_free_tail_that_does_not_start_at_actual_pose(self):
        from unittest.mock import patch
        planner=Planner(self.cfg,self.geometry,require_actual_start=True)
        original=planner.query(self.req,scene=self.scene,now=10.,clock_domain='synthetic')
        tail=np.asarray(original['path']).copy();tail[0,0]+=.15
        with patch('navigation.planning.planner.reference_tails',return_value=iter([tail])):
            result=planner.query(self.req,scene=self.scene,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        np.testing.assert_allclose(result['path'][0],[2.,2.,0.])
        self.assertNotEqual(result.get('reference_continuity',{}).get('start_is_actual'),False)

    def test_unsupported_intent_and_behavior_cannot_silently_drive_forward(self):
        for changes in (dict(motion_intent='reverse_required'),dict(goal_behavior='rotate')):
            r=self.controller.command(dict(self.req,**changes),self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
            self.assertEqual(r['status'],'unavailable');self.assertEqual(r['velocity'],[0.,0.])

    def test_failed_reference_cools_down_across_scans_but_new_goal_can_run_immediately(self):
        from unittest.mock import patch
        runtime=ReferenceRuntime(self.cfg,self.motion,self.geometry,worker_factory=FakeWorker)
        failed=copy.deepcopy(self.reference['result']);failed.update(status='no_path',path=[])
        failed.pop('result_binding');failed['result_binding']=fingerprint(failed)
        with patch('navigation.planning.reference.time.monotonic',return_value=0.):
            runtime.update(self.req,self.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
        runtime.worker.reply(reference_envelope(failed,self.req,self.scene,None))
        with patch('navigation.planning.reference.time.monotonic',return_value=.1):
            runtime.tick(now=10.,clock_domain='synthetic')
        with patch('navigation.planning.reference.time.monotonic',return_value=1.5):
            runtime.update(self.req,scene_at(10.1),None,now=10.1);runtime.tick(now=10.1,clock_domain='synthetic')
            self.assertIsNone(runtime.worker.job)
            runtime.update(dict(self.req,goal_revision=1),scene_at(10.1),None,now=10.1)
            runtime.tick(now=10.1,clock_domain='synthetic')
            self.assertIsNotNone(runtime.worker.job)
        runtime.close()

    def test_expired_query_input_waits_for_new_measurement_without_geometric_backoff(self):
        failed=copy.deepcopy(self.reference['result']);failed.update(status='unavailable',reason='navigation_prior_future_or_stale',path=[])
        failed.pop('result_binding');failed['result_binding']=fingerprint(failed)
        with ReferenceRuntime(self.cfg,self.motion,self.geometry,worker_factory=FakeWorker) as runtime:
            with patch('navigation.planning.reference.time.monotonic',return_value=100.):
                runtime.update(self.req,self.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
            runtime.worker.reply(reference_envelope(failed,self.req,self.scene,None))
            with patch('navigation.planning.reference.time.monotonic',return_value=100.1):
                runtime.tick(now=10.,clock_domain='synthetic')
            self.assertIsNone(runtime.failure);self.assertIsNone(runtime.current)
            with patch('navigation.planning.reference.time.monotonic',return_value=101.1):
                runtime.update(self.req,self.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
                self.assertIsNone(runtime.worker.job,'same measurement must not be retried')
                newer=scene_at(10.2);runtime.update(self.req,newer,None,now=10.2)
                runtime.tick(now=10.2,clock_domain='synthetic')
            self.assertIsNotNone(runtime.worker.job)
            self.assertEqual(runtime.worker.job['scene'].record['as_of'],10.2)

    def test_untrusted_temporal_failure_cannot_bypass_failure_backoff(self):
        failed=copy.deepcopy(self.reference);failed['result']['reason']='navigation_prior_future_or_stale'
        with ReferenceRuntime(self.cfg,self.motion,self.geometry,worker_factory=FakeWorker) as runtime:
            with patch('navigation.planning.reference.time.monotonic',return_value=100.):
                runtime.update(self.req,self.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
            runtime.worker.reply(failed)
            with patch('navigation.planning.reference.time.monotonic',return_value=100.1):
                runtime.tick(now=10.,clock_domain='synthetic')
            self.assertGreater(runtime.next_retry,103.);self.assertIsNone(runtime.retry_after_observation)

    def test_stale_feedback_corruption_and_changed_goal_are_not_reused(self):
        for feedback,ref,req,now in [(self.feedback,self.reference,self.req,10.16),
                (self.feedback,self.reference,dict(self.req,goal_revision=1),10.)]:
            r=self.controller.command(req,self.scene,feedback,ref,now=now,clock_domain='synthetic')
            self.assertEqual(r['status'],'unavailable');self.assertEqual(r['velocity'],[0.,0.]);self.assertIsNone(self.controller.result)
        self.scene.geometry.layers['occupancy'][20,22]=100
        r=self.controller.command(self.req,self.scene,self.feedback,self.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(r['status'],'unavailable')

    def test_reference_diagnostics_do_not_bypass_malformed_reply_rejection(self):
        runtime=ReferenceRuntime(self.cfg,self.motion,self.geometry,worker_factory=FakeWorker)
        try:
            runtime.update(self.req,self.scene,None,now=10.)
            runtime.tick(now=10.,clock_domain='synthetic')
            runtime.worker.reply({})
            runtime.tick(now=10.,clock_domain='synthetic')
            self.assertIsNone(runtime.current)
            self.assertTrue(any(e['event']=='discarded' and e['reason']=='reference_envelope_binding_mismatch'
                for e in runtime.drain_events()))
        finally:runtime.close()

    def test_batched_stopping_matches_scalar_model_with_delay_and_unequal_stop_times(self):
        from navigation.safety.guard import stopping_trajectory,stopping_trajectories
        settings=self.controller.safety_settings;f=dict(self.feedback,velocity=[.45,.1])
        targets=np.array([[0,0],[.3,.05],[.48,.12],[.2,-.05]])
        batch=stopping_trajectories(f,targets,self.motion,settings,.1,pending=[.5,.1],observation_age_s=.07)
        for desired,actual in zip(targets,batch):
            expected=stopping_trajectory(f,desired,self.motion,settings,.1,pending=[.5,.1],observation_age_s=.07)
            np.testing.assert_allclose(actual,expected,atol=2e-12)

    def test_vector_interpolation_preserves_scalar_samples_and_error(self):
        from navigation.planning.motion import interpolate_se2,twists
        from navigation.planning.footprint import wrap
        random=np.random.default_rng(4)
        poses=np.cumsum(random.normal(size=(25,3))*.2,axis=0)
        poses[:,2]+=np.linspace(2.8,4.,len(poses))
        rates=twists(poses,np.ones(len(poses)-1));expected=[poses[0]];error=0.
        for a,b,rate in zip(poses[:-1],poses[1:],rates):
            turn=wrap(b[2]-a[2]);length=np.linalg.norm(rate[:2]);n=max(1,int(np.ceil(max(length/.05,abs(turn)/.05))))
            u=np.arange(1,n+1)/n;theta=turn*u;A=u*np.sinc(theta/np.pi);B=u*theta/2*np.sinc(theta/(2*np.pi))**2
            xy=np.c_[A*rate[0]-B*rate[1],B*rate[0]+A*rate[1]]
            c,s=np.cos(a[2]),np.sin(a[2]);expected.extend(np.c_[xy@np.array([[c,s],[-s,c]])+a[:2],a[2]+theta])
            error=max(error,length/n*abs(turn/n)/8)
        actual,bound=interpolate_se2(poses,.1,2000)
        np.testing.assert_allclose(actual,expected,atol=2e-14);self.assertAlmostEqual(bound,error,places=15)
        with self.assertRaises(ValueError):interpolate_se2(poses,.1,len(actual)-1)

    def test_independent_safety_preserves_feedback_time_and_models_age_before_braking(self):
        control=DwaNavigationControl(self.cfg,self.motion,self.geometry,simulation=True)
        f=dict(self.feedback,velocity=[.2,0.])
        r=control.step(self.req,self.scene,f,now=10.03,clock_domain='synthetic',dt=.1,result=self.reference)
        self.assertEqual(r['command']['state'],'allowed',r)
        self.assertEqual(r['command']['stopping_sweep']['prediction_start_s'],10.)
        self.assertEqual(r['proposal']['trajectory_as_of'],10.)
        self.assertEqual(len(r['command']['velocity_filter']['attempts']),1)
        self.assertFalse(r['goal']['task_completed'])
        for key in ('manual_override','emergency_stop'):
            stop=control.step(self.req,self.scene,f,now=10.,clock_domain='synthetic',dt=.1,result=self.reference,**{key:True})
            self.assertEqual(stop['command']['velocity'],[0.,0.])

    def test_one_reference_worker_retains_route_on_new_observation_and_latches_cancel(self):
        runtime=ReferenceRuntime(self.cfg,self.motion,self.geometry,worker_factory=FakeWorker)
        try:
            runtime.update(self.req,self.scene,None,now=10.);runtime.tick(now=10.,clock_domain='synthetic')
            self.assertEqual(runtime.worker.job['epoch'],0)
            runtime.worker.reply(self.reference);runtime.tick(now=10.,clock_domain='synthetic')
            self.assertIsNotNone(runtime.current)
            newer=scene_at(10.1)
            runtime.update(self.req,newer,None,now=10.1);runtime.tick(now=10.1,clock_domain='synthetic')
            self.assertIsNotNone(runtime.current);self.assertIsNone(runtime.worker.job)
            runtime.cancel(self.req)
            self.assertFalse(runtime.update(self.req,newer,None,now=10.1));self.assertIsNone(runtime.current)
        finally:runtime.close()


if __name__=='__main__':unittest.main()
