"""Timed crossing/uncertainty, binding and independent navigation-policy checks."""
import copy
import unittest
import numpy as np
from perception.environment.context import fingerprint
from perception.environment.scene import assemble_scene
from perception.environment.layers import load_config as environment_config
from navigation.planning.inputs import load_config
from navigation.planning.people import PeopleModel,social_groups
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.motion import load_motion_config
from navigation.safety.guard import LocalSafety
from navigation.tasks import TaskPolicy
from tests.navigation_fixtures import grid,request,asset
from tests.test_environment_scene import person


def tracked_scene(tracks,*,stamp=10.,generation=0):
    current=grid(generation=generation);current.metadata['stamp_s']=stamp
    source=person(stamp);source['localization_binding']=dict(session_id='test',generation=generation,frame_id='odom')
    source['tracks']=tracks;source['covariance_order']=['x','y','z','dx','dy','dz','yaw','vx','vy','vz']
    source['input_fingerprint']=fingerprint(tracks)
    return assemble_scene(current,clock_domain='synthetic',settings=environment_config()['scene'],people=source)


def track(x=3.,y=2.,velocity=(0.,0.,0.),id='a',*,stamp=10.):
    return dict(track_id=id,box=[x,y,0.,.4,.4,1.7,.0,.9],confirmed=True,
        velocity_xyz_mps=velocity,covariance=np.zeros(100).tolist(),last_observed_stamp_s=stamp)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config();self.cfg=self.config['people']
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')

    def test_crossing_between_knots_is_rejected_but_frozen_current_box_is_clear(self):
        moving=PeopleModel(tracked_scene([track(2.5,3.,(0.,-2.,0.))]),self.cfg)
        stopped=PeopleModel(tracked_scene([track(2.5,3.)]),self.cfg)
        path=np.array([[0.,2.,2.,0.],[1.,3.,2.,0.]])
        self.assertEqual(moving.check(path,self.geometry)['reason'],'predicted_people_collision')
        self.assertTrue(stopped.check(path,self.geometry)['valid'])

    def test_unknown_velocity_keeps_an_occupancy_proxy_without_claiming_stationarity(self):
        model=PeopleModel(tracked_scene([track(3,4,None)]),self.cfg)
        centre,radius=model.positions([10.,11.])
        np.testing.assert_allclose(centre[0],centre[1]);np.testing.assert_array_equal(radius[0],radius[1])
        self.assertIsNone(model.tracks[0]['velocity'])
        self.assertEqual(model.summary()['unknown_velocities'],1)
        self.assertEqual(model.summary()['unknown_velocity_policy'],'current_occupancy_hold_not_measured_motion')

    def test_retired_people_model_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError,'Unsupported people prediction model'):
            PeopleModel(tracked_scene([]),dict(self.cfg,prediction_model='bounded'))

    def test_people_check_reports_first_conflict_and_uncertainty_separately(self):
        moving=PeopleModel(tracked_scene([track(2.5,3.,(0.,-2.,0.))]),self.cfg)
        result=moving.check([[0.,2.,2.,0.],[1.,3.,2.,0.]],self.geometry,details=True)
        self.assertFalse(result['valid']);self.assertIsNotNone(result['first_conflict_interval_s'])
        self.assertEqual(result['prediction_start_s'],10.);self.assertIn('covariance_radius_m',result['critical_envelope'])
        self.assertIn('nominal_clearance',result['diagnostic'])

    def test_covariance_order_and_future_track_are_invalid_not_background(self):
        for bad in [track(stamp=10.1),dict(track(),covariance=[-1.]+[0.]*99)]:
            model=PeopleModel(tracked_scene([bad]),self.cfg)
            self.assertEqual(model.status,'invalid');self.assertFalse(model.check([[0,2,2,0],[1,3,2,0]],self.geometry)['valid'])
        scene=tracked_scene([track()]);scene.record['people']['data']['covariance_order']=[]
        self.assertEqual(PeopleModel(scene,self.cfg).reason,'unknown_person_covariance_order')

    def test_missing_people_scope_and_horizon_are_explicit(self):
        from tests.navigation_fixtures import bundle
        path=[[0,2,2,0],[1,2,2,0]]
        self.assertEqual(PeopleModel(bundle(),self.cfg).check(path,self.geometry)['scope'],'geometry_only')
        self.assertFalse(PeopleModel(bundle(),dict(self.cfg,required=True)).check(path,self.geometry)['valid'])
        model=PeopleModel(tracked_scene([track()]),self.cfg)
        self.assertEqual(model.check([[0,2,2,0],[7,2,2,0]],self.geometry)['reason'],'people_prediction_horizon_exceeded')

    def test_social_cost_penalizes_cutting_group_without_changing_occupancy(self):
        scene=tracked_scene([track(3,1.,id='a'),track(3,3.,id='b')]);model=PeopleModel(scene,self.cfg)
        path=[[0,2,2,0],[2,4,2,0]];before=scene.geometry.layers['occupancy'].copy()
        individual=model.social_cost(path);grouped=model.social_cost(path,[model.ids])
        self.assertGreater(grouped,individual);np.testing.assert_array_equal(before,scene.geometry.layers['occupancy'])

    def test_independent_safety_rejects_future_crossing_even_for_a_bound_proposal(self):
        scene=tracked_scene([track(2.8,3.5,(0.,-1.5,0.))]);motion=load_motion_config();req=request('local')
        guard=LocalSafety(self.geometry,motion,.05,people_settings=self.cfg,simulation=True)
        f=dict(status='valid',source='synthetic',pose=[2,2,0],velocity=[.6,0],stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'])
        p=dict(status='candidate',control_mode='dwa',velocity=[.7,0],request_binding=fingerprint(req),scene_binding=scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic')
        result=guard.gate(p,req,scene,f,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['reason'],'predicted_people_collision');self.assertEqual(result['velocity'],[0.,0.])

    def test_safety_people_cutoff_starts_at_now_and_keeps_future_diagnostics(self):
        motion=load_motion_config();req=request('local')
        feedback=dict(status='valid',source='synthetic',pose=[2.,2.,0.],velocity=[.3,0.],stamp_s=10.,
            clock_domain='synthetic',local_identity=['test',0,'odom'])
        def check(person,age):
            scene=tracked_scene([person])
            proposal=dict(status='candidate',control_mode='dwa',velocity=[.336,0.],request_binding=fingerprint(req),
                scene_binding=scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic')
            guard=LocalSafety(self.geometry,motion,.05,people_settings=self.cfg,shadow=True)
            return guard.gate(proposal,req,scene,feedback,now=10.+age,clock_domain='synthetic',dt=.1)
        for age in (0.,.1):
            far=check(track(3.,3.5,(0.,-1.,0.)),age)
            self.assertEqual(far['state'],'allowed',far)
            self.assertAlmostEqual(far['people_check']['hard_horizon_s'],age+1.)
            self.assertFalse(far['people_check']['full_forecast_clear'])
            self.assertGreater(far['people_check']['future_conflict_duration_s'],0.)
            near=check(track(2.8,3.,(0.,-1.5,0.)),age)
            self.assertEqual(near['reason'],'predicted_people_collision')
            self.assertEqual(near['velocity'],[0.,0.])

    def test_safety_still_checks_static_obstacles_beyond_people_cutoff(self):
        from tests.navigation_fixtures import bundle
        cells=np.zeros((60,60),np.int8);cells[20,33]=100;scene=bundle(cells)
        motion=load_motion_config();req=request('local')
        feedback=dict(status='valid',source='synthetic',pose=[2.,2.,0.],velocity=[.8,0.],stamp_s=10.,
            clock_domain='synthetic',local_identity=['test',0,'odom'])
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.8,0.],request_binding=fingerprint(req),
            scene_binding=scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic')
        result=LocalSafety(self.geometry,motion,.05,people_settings=self.cfg,shadow=True).gate(
            proposal,req,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['reason'],'motion_sweep_occupied')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_independent_gate_rejects_people_collision_without_changing_the_selected_velocity(self):
        geometry=self.geometry;motion=load_motion_config()
        scene=tracked_scene([track(2.8,3.5,(0.,-1.5,0.))]);req=request('local')
        feedback=dict(status='valid',source='synthetic',pose=[2.,2.,0.],velocity=[.6,0.],stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'])
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.7,0.],request_binding=fingerprint(req),
            scene_binding=scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic_checked')
        guard=LocalSafety(geometry,motion,.05,people_settings=self.cfg,shadow=True)
        result=guard.gate(proposal,req,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['state'],'stop_request');self.assertEqual(result['velocity'],[0.,0.])
        self.assertEqual(len(result['velocity_filter']['attempts']),1)
        self.assertEqual(result['velocity_filter']['attempts'][0]['reason'],'predicted_people_collision')
        self.assertEqual(result['check']['status'],'free');self.assertEqual(proposal['velocity'],[.7,0.])
        self.assertEqual(guard.gate(proposal,req,scene,feedback,now=10.,clock_domain='synthetic',dt=.1,emergency_stop=True)['reason'],'emergency_stop')
        # Same path and proposal with an occupied current footprint must stop;
        # a rejected proposal cannot bypass geometry evidence.
        from tests.navigation_fixtures import bundle
        cells=np.zeros((60,60),np.int8);cells[20,20]=100;occupied=bundle(cells)
        blocked_proposal=dict(proposal,scene_binding=occupied.record['input_fingerprint'])
        blocked=guard.gate(blocked_proposal,req,occupied,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(blocked['state'],'stop_request');self.assertEqual(blocked['reason'],'motion_sweep_occupied')

    def test_deterministic_body_does_not_grow_with_box_covariance_or_prediction_time(self):
        target=track(4.,2.,(.5,0.,0.));target['box'][3:5]=[2.,3.]
        target['covariance']=(np.eye(10)*100.).ravel().tolist()
        model=PeopleModel(tracked_scene([target]),self.cfg)
        xy,radius,parts=model.positions([10.,14.],components=True)
        np.testing.assert_allclose(xy[:,0],[[4.,2.],[6.,2.]])
        np.testing.assert_allclose(radius,.3)
        self.assertFalse(model.summary()['uncertainty_applied'])
        self.assertTrue(all(not v.any() for v in parts.values()))
        unknown=PeopleModel(tracked_scene([track(4.,2.,None)]),self.cfg)
        self.assertIsNone(unknown.tracks[0]['velocity'])
        self.assertEqual(unknown.summary()['unknown_velocity_policy'],'current_occupancy_hold_not_measured_motion')

    def test_distant_conflict_is_soft_for_planning_but_full_check_still_rejects(self):
        model=PeopleModel(tracked_scene([track(6.,2.,(-1.,0.,0.))]),self.cfg)
        path=[[0.,2.,2.,0.],[4.,2.,2.,0.]]
        near=model.check(path,self.geometry,hard_until=1.)
        self.assertTrue(near['valid']);self.assertFalse(near['full_forecast_clear'])
        self.assertGreater(near['future_conflict_duration_s'],0.)
        self.assertFalse(model.check(path,self.geometry)['valid'])

    def test_person_background_is_a_bound_collision_view_and_keeps_unknown_or_mixed_geometry(self):
        current=grid();current.metadata['stamp_s']=10.
        current.layers['occupancy'][20,27]=100
        mask=np.zeros_like(current.layers['occupancy'],bool);mask[20,27]=True
        current.layers['person_background_free']=mask
        data=person(10.);data.update(tracks=[track(2.8,2.,(0.,1.,0.))],
            localization_binding=dict(session_id='test',generation=0,frame_id='odom'),
            covariance_order=['x','y','z','dx','dy','dz','yaw','vx','vy','vz'])
        data['input_fingerprint']=fingerprint(data['tracks'])
        current.metadata['dynamic_filter']=dict(status='valid',input_fingerprint=data['input_fingerprint'])
        scene=assemble_scene(current,clock_domain='synthetic',settings=environment_config()['scene'],people=data)
        model=PeopleModel(scene,self.cfg);derived,binding,report=model.navigation_grid(scene)
        self.assertEqual(derived.layers['occupancy'][20,27],0)
        self.assertEqual(scene.geometry.layers['occupancy'][20,27],100);scene.verify()
        self.assertNotEqual(binding,scene.record['geometry']['layer_binding'])
        self.assertEqual(report['restored_cells'],1)
        self.assertAlmostEqual(report['projection_margin_m'],.1/np.sqrt(2))
        # Unknown velocity keeps original geometry, rather than supplying zero speed.
        model.known[:]=False;model._navigation_grid=None
        self.assertIs(model.navigation_grid(scene)[0],scene.geometry)
        # No provenance: the same occupied cell is an ordinary obstacle.
        ordinary=grid();ordinary.metadata['stamp_s']=10.;ordinary.layers['occupancy'][20,27]=100
        other=assemble_scene(ordinary,clock_domain='synthetic',settings=environment_config()['scene'],people=data).verify()
        self.assertEqual(PeopleModel(other,self.cfg).navigation_grid(other)[0].layers['occupancy'][20,27],100)

    def test_stopping_check_replaces_only_proven_person_cells_and_checks_actual_contact(self):
        current=grid();current.metadata['stamp_s']=10.;current.layers['occupancy'][20,27]=100
        mask=np.zeros_like(current.layers['occupancy'],bool);mask[20,27]=True
        current.layers['person_background_free']=mask
        data=person(10.);data.update(tracks=[track(2.8,2.,(0.,1.,0.))],
            localization_binding=dict(session_id='test',generation=0,frame_id='odom'),
            covariance_order=['x','y','z','dx','dy','dz','yaw','vx','vy','vz'])
        data['input_fingerprint']=fingerprint(data['tracks'])
        current.metadata['dynamic_filter']=dict(status='valid',input_fingerprint=data['input_fingerprint'])
        scene=assemble_scene(current,clock_domain='synthetic',settings=environment_config()['scene'],people=data)
        req=request('local');motion=load_motion_config()
        feedback=dict(status='valid',source='synthetic',pose=[2.,2.,0.],velocity=[.3,0.],stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'])
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.6,0.],request_binding=fingerprint(req),scene_binding=scene.record['input_fingerprint'],
            trajectory_as_of=10.,trajectory_binding='synthetic')
        guard=LocalSafety(self.geometry,motion,.05,people_settings=self.cfg,shadow=True)
        result=guard.gate(proposal,req,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['state'],'allowed',result['reason'])
        self.assertEqual(result['dynamic_geometry']['restored_cells'],1)
        cached=guard._people_cache[2];derived=guard.sweep.snapshot
        result['dynamic_geometry']['projection_margin_m']=999.
        again=guard.gate(proposal,req,scene,feedback,now=10.01,clock_domain='synthetic',dt=.1)
        self.assertIs(guard._people_cache[2],cached);self.assertIs(guard.sweep.snapshot,derived)
        self.assertLess(again['dynamic_geometry']['projection_margin_m'],.1)
        self.assertEqual(again['motion_sweep']['prediction_start_s'],10.)
        guard.people_settings=dict(self.cfg,body_radius_m=.31)
        guard.gate(proposal,req,scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertIsNot(guard._people_cache[2],cached)
        other=copy.deepcopy(scene)
        guard.gate(proposal,req,other,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertIs(guard._people_cache[0],other)
        other.geometry.layers['occupancy'][20,22]=100
        corrupt=guard.gate(proposal,req,other,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(corrupt['reason'],'scene_binding_invalid');self.assertEqual(corrupt['velocity'],[0.,0.])
        occupied_feedback=dict(feedback,pose=[2.7,2.,0.])
        result=guard.gate(proposal,req,scene,occupied_feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['reason'],'motion_sweep_occupied')

    def test_cropped_distance_field_matches_full_map_including_exterior_and_conditional_risk(self):
        from scipy.ndimage import distance_transform_edt
        from navigation.planning.costmap import build_costmap
        cells=np.full((400,500),-1,np.int8);cells[:60,:80]=0;cells[12:18,40:45]=100
        snapshot=grid(cells);conditional=np.zeros_like(cells,bool);conditional[45,30]=True
        snapshot.layers['conditional_obstacle']=conditional
        costs,distance=build_costmap(snapshot,.2,self.config,return_clearance=True)
        field=distance_transform_edt(np.pad(cells==0,1))[1:-1,1:-1]*.1
        np.testing.assert_allclose(distance,field,rtol=1e-7)
        free=(cells==0)&(field-np.sqrt(2)*.1-.2-self.config['safety_margin_m']>1e-8)
        np.testing.assert_array_equal(costs<254,free)
        near=distance_transform_edt(~conditional)*.1<=.2+self.config['safety_margin_m']+.1
        self.assertTrue((costs[near&free]>=self.config['conditional_obstacle_cost']).all())

    def test_native_search_follows_three_bends_without_spinning_or_reverse(self):
        from types import SimpleNamespace
        from navigation.planning.planner import Planner
        from navigation.planning.curvature import motion_valid
        c=np.zeros((110,180),np.int8);c[:2]=c[-2:]=100;c[:,:2]=c[:,-2:]=100
        c[:58,48:54]=100;c[52:,88:94]=100;c[:58,128:134]=100
        snapshot=grid(c);snapshot.metadata.update(frame_id='map',snapshot_id='synthetic-bends')
        a=SimpleNamespace(snapshot=snapshot,manifest=dict(snapshot_id='synthetic-bends',layers_sha256='synthetic'))
        cfg=copy.deepcopy(self.config);cfg['motion_constraints']=dict(min_turning_radius_m=1.12,allow_reverse=False)
        req=request('static',start=[2.,2.,0.],goal=[16.,8.,0.],map_snapshot_id='synthetic-bends')
        result=Planner(cfg,self.geometry).query(req,asset=a)
        self.assertEqual(result['status'],'path_found',result.get('reason'))
        self.assertTrue(motion_valid(np.array(result['path']),cfg['motion_constraints']))
        self.assertEqual(result['footprint_check']['status'],'free')

    def test_planning_only_prior_preserves_occupancy_and_corrections(self):
        from perception.environment.changes import PriorRevisions,reference_grid
        from tests.test_environment_changes import evidence
        prior=evidence(np.zeros((20,20),np.int8));prior.layers['occupancy'][10,10]=100
        current=evidence(np.zeros((20,20),np.int8),stamp=11.)
        association=dict(map_from_local=np.eye(4).tolist(),association_session='s',association_generation=0)
        a=PriorRevisions().apply(prior,current,association,scene_binding='test')
        b=PriorRevisions().apply(prior,current,association,scene_binding='test',geometry_only=True)
        self.assertEqual(a.metadata['prior_revision'],b.metadata['prior_revision'])
        np.testing.assert_array_equal(a.layers['occupancy'],b.layers['occupancy'])
        self.assertNotIn('clearance',b.layers)
        np.testing.assert_array_equal(reference_grid(a).layers['occupancy'],reference_grid(b,geometry_only=True).layers['occupancy'])


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config();self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.scene=tracked_scene([track(4,2,(.5,0,0))]);self.key=['tracker_session',7,'a']

    def test_late_response_and_pid_reuse_cannot_replace_current_goal(self):
        policy=TaskPolicy('follow','follow',self.config,self.geometry)
        old=policy.observe(self.scene,pid_to_track={0:self.key});new=policy.observe(self.scene,pid_to_track={1:self.key})
        with self.assertRaises(ValueError):policy.respond(old,dict(id=0,state='visible'),scene=self.scene,now=10.)
        result=policy.respond(new,dict(id=1,state='visible'),scene=self.scene,now=10.)
        self.assertEqual(result['target_identity'],self.key);self.assertEqual(result['request']['goal'],[2.5,2.])
        lost=tracked_scene([track(id='reused')],stamp=10.1)
        self.assertEqual(policy.update(lost,now=10.1)['operation'],'stop')
        self.assertEqual(policy.update(tracked_scene([track()],generation=1),now=10.)['reason'],'task_localization_changed')

    def test_occlusion_does_not_refresh_visual_confirmation_or_report_success(self):
        p=TaskPolicy('follow','follow',self.config,self.geometry,occlusion_hold_s=1.)
        t=p.observe(self.scene,pid_to_track={0:self.key});p.respond(t,dict(id=0,state='visible'),scene=self.scene,now=10.)
        later=tracked_scene([track(4,2,(.5,0,0),stamp=10.5)],stamp=10.5)
        t=p.observe(later,pid_to_track={3:self.key});r=p.respond(t,dict(id=3,state='occluded'),scene=later,now=10.5)
        self.assertEqual(r['operation'],'plan');self.assertEqual(p.last_visual,10.)
        self.assertEqual(p.update(later,now=11.1)['reason'],'follow_visual_confirmation_expired')
        self.assertFalse(r['task_completed']);self.assertEqual(p.cancel()['operation'],'stop')

    def test_social_groups_expire_and_keep_tracker_generation(self):
        scene=tracked_scene([track(id='a'),track(id='b')]);p=TaskPolicy('social','social',self.config,self.geometry)
        t=p.observe(scene,pid_to_track={0:self.key,1:['tracker_session',7,'b']})
        r=p.respond(t,dict(talking=[[0,1]]),scene=scene,now=10.)
        self.assertEqual(r['social']['groups'][0][0],self.key)
        late=tracked_scene([],stamp=13.);_,status=social_groups(dict(social=r['social']),late,PeopleModel(late,self.config['people']))
        self.assertEqual(status['status'],'expired')

    def test_mapped_goal_uses_current_association_and_actual_feedback(self):
        from navigation.control.tracking import goal_feedback,load_config as control_config
        from tests.navigation_fixtures import bundle
        scene=bundle();a=asset();req=request('mapped',goal=[2.,2.],goal_behavior='stop')
        feedback=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'],pose=[2,2,0],velocity=[0,0])
        result=goal_feedback(req,scene,feedback,control_config(),now=10.,clock_domain='synthetic',active=True,geometry=self.geometry,asset=a,planning=self.config)
        self.assertTrue(result['criteria_met']);self.assertFalse(result['task_completed'])

    def test_seating_goal_is_bound_local_pose_and_still_requires_geometry(self):
        p=TaskPolicy('seat','seating',self.config,self.geometry);r=self.scene.record
        candidates=dict(instruction='Sit by table',observation_id=r['observation_id'],stamp_s=r['as_of'],
            frame_id='odom',clock_domain='synthetic',session_id='test',generation=0,revision=1,
            candidates=[dict(id=0,pose=[3.,2.,0.],motion='forward',geometry='unknown')])
        t=p.observe(self.scene,candidates=candidates)
        output=p.respond(t,dict(s='v',p=0),scene=self.scene,now=10.)
        self.assertEqual(output['operation'],'plan');self.assertTrue(output['requires_geometry_verification'])
        self.assertEqual(output['request']['goal_behavior'],'stop');self.assertFalse(output['task_completed'])
        revised=copy.deepcopy(candidates);revised['revision']=2
        revised['candidates'].append(dict(id=1,pose=[3.1,2.,0.],motion='forward',geometry='unknown'))
        t=p.observe(self.scene,candidates=revised)
        changed=p.respond(t,dict(s='v',p=1),scene=self.scene,now=10.)
        self.assertEqual(changed['request']['goal'],[3.1,2.,0.])
        self.assertGreater(changed['request']['goal_revision'],output['request']['goal_revision'])
        bad=copy.deepcopy(candidates);bad['frame_id']='rslidar'
        with self.assertRaises(ValueError):p.observe(self.scene,candidates=bad)

    def test_policy_deadline_is_checked_again_by_control_and_safety(self):
        from navigation.control.runtime import DwaNavigationControl
        req=request('local',goal=[4.,2.],valid_until_s=9.9)
        control=DwaNavigationControl(self.config,load_motion_config(),self.geometry,simulation=True)
        feedback=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'],pose=[2,2,0],velocity=[0,0])
        result=control.step(req,self.scene,feedback,now=10.,clock_domain='synthetic',dt=.1)
        self.assertEqual(result['command']['reason'],'request_expired');self.assertFalse(result['goal']['task_completed'])

    def test_follow_request_runs_through_planning_tracking_and_independent_gate(self):
        from navigation.planning.planner import Planner
        from navigation.planning.reference import planning_config
        from navigation.planning.continuity import reference_envelope
        from navigation.control.runtime import DwaNavigationControl
        p=TaskPolicy('follow','follow',self.config,self.geometry)
        ticket=p.observe(self.scene,pid_to_track={0:self.key})
        req=p.respond(ticket,dict(id=0,state='visible'),scene=self.scene,now=10.)['request']
        config=planning_config(self.config)
        planner=Planner(config,self.geometry,require_actual_start=True)
        reference=planner.query(req,scene=self.scene,now=10.,clock_domain='synthetic')
        self.assertEqual(reference['status'],'path_found',reference.get('reason'))
        result=reference_envelope(reference,req,self.scene,None)
        control=DwaNavigationControl(config,load_motion_config(),self.geometry,simulation=True)
        f=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'],pose=[2,2,0],velocity=[0,0])
        outcome=control.step(req,self.scene,f,now=10.,clock_domain='synthetic',dt=.1,result=result)
        self.assertEqual(outcome['command']['state'],'allowed');self.assertGreater(outcome['command']['velocity'][0],0.)
        self.assertFalse(outcome['goal']['task_completed'])

    def test_holding_follow_distance_does_not_complete_a_continuous_task(self):
        from navigation.control.tracking import GoalMonitor
        req=request('local',goal=[2.,2.],goal_behavior='stop',completion_policy='continuous')
        monitor=GoalMonitor(geometry=self.geometry,planning=self.config)
        f=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'],pose=[2,2,0],velocity=[0,0])
        monitor.update(req,self.scene,f,now=10.,clock_domain='synthetic',active=True)
        result=monitor.update(req,self.scene,dict(f,stamp_s=10.5),now=10.5,clock_domain='synthetic',active=True)
        self.assertEqual(result['status'],'distance_held');self.assertTrue(result['completion_criteria_sustained'])
        self.assertFalse(result['task_completed'])


if __name__=='__main__':unittest.main()
