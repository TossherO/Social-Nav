"""Explicit mixed-route references; no motion or execution success simulation."""
import json
from dataclasses import replace
import unittest
import numpy as np

from navigation.planning.inputs import load_config
from navigation.planning.route import RoutePlan, RouteSession, _lookahead
from navigation.planning.footprint import VehicleGeometry
from tests.test_navigation_planning import bundle,asset


def reference():
    return dict(route_id='demo',source='explicit_synthetic_route',anchors=dict(door=dict(
        map_id='map-a',map_frame='map',map_from_anchor=np.eye(3).tolist(),source='synthetic_portal')),
        segments=[dict(segment_id='inside',mode='mapped',floor_id='f1',map_id='map-a',frame_id='map',
                       points_xy=[[2,2],[5,2]],source='synthetic'),
                  dict(segment_id='outside',mode='local',floor_id='f1',anchor_id='door',
                       points_xy=[[2,2],[5,2]],source='synthetic'),
                  dict(segment_id='lift',mode='transition',from_floor='f1',to_floor='f2',
                       confirmation_source='external_confirmed_floor',source='synthetic')])


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.route=RoutePlan(reference());self.config=load_config()
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.args=dict(task_id='task',goal_revision=1,now=10.,clock_domain='synthetic',config=self.config)

    def test_mapped_and_local_references_use_current_pose_and_never_complete_tasks(self):
        scene=bundle();anchor=self.route.bind_anchor('door',scene,now=10.,clock_domain='synthetic',config=self.config)
        result=self.route.request('inside',scene,self.geometry,asset=asset(),**self.args)
        self.assertEqual(result['request']['goal'],[4.,2.]);self.assertFalse(result['task_completed'])
        json.dumps(result,allow_nan=False)
        outside=bundle(associated=False)
        result=self.route.request('outside',outside,self.geometry,anchor=anchor,**self.args)
        self.assertEqual(result['request']['goal'],[4.,2.]);self.assertEqual(result['request']['mode'],'local')
        with self.assertRaises(ValueError):self.route.request('inside',outside,self.geometry,asset=asset(),**self.args)

    def test_route_requests_stop_only_at_an_explicit_stopping_endpoint(self):
        record=reference();record['segments'][0]['goal_behavior']='stop';route=RoutePlan(record)
        result=route.request('inside',bundle(),self.geometry,asset=asset(),**self.args)
        self.assertEqual(result['request']['goal_behavior'],'pass_through')
        result=route.request('inside',bundle(),self.geometry,asset=asset(),lookahead_m=4.,**self.args)
        self.assertEqual(result['request']['goal_behavior'],'stop')

    def test_anchor_rejects_reset_future_wrong_route_and_clock(self):
        scene=bundle();anchor=self.route.bind_anchor('door',scene,now=10.,clock_domain='synthetic',config=self.config)
        for target in (bundle(generation=1),):
            with self.assertRaises(ValueError):self.route.request('outside',target,self.geometry,anchor=anchor,**self.args)
        with self.assertRaises(ValueError):self.route.bind_anchor('door',bundle(associated=False),now=10.,clock_domain='synthetic',config=self.config)
        other=reference();other['source']='different';changed=RoutePlan(other)
        with self.assertRaises(ValueError):changed.request('outside',scene,self.geometry,anchor=anchor,**self.args)
        with self.assertRaises(ValueError):self.route.request('outside',scene,self.geometry,anchor=anchor,**dict(self.args,clock_domain='other'))
        with self.assertRaises(ValueError):self.route.request('outside',scene,self.geometry,anchor=replace(anchor,stamp_s=11.),**self.args)
        wrong=asset();wrong.snapshot.metadata['map_id']='other'
        with self.assertRaises(ValueError):self.route.request('inside',scene,self.geometry,asset=wrong,**self.args)

    def test_anchor_transform_is_goal_reference_not_reused_localization(self):
        record=reference();record['anchors']['door']['map_from_anchor']=[[0,-1,4],[1,0,0],[0,0,1]]
        record['segments'][1]['points_xy']=[[2,2],[5,2]]
        route=RoutePlan(record);scene=bundle();anchor=route.bind_anchor('door',scene,now=10.,clock_domain='synthetic',config=self.config)
        result=route.request('outside',bundle(associated=False),self.geometry,anchor=anchor,**self.args)
        np.testing.assert_allclose(result['request']['goal'],[2.,4.])

    def test_floor_transition_never_advances_from_a_timestamp(self):
        result=self.route.request('lift',bundle(),self.geometry,**self.args)
        self.assertEqual(result['status'],'waiting_for_external_confirmation');self.assertIsNone(result['request'])

    def test_task_route_requires_matching_external_floor_feedback_and_reselection(self):
        session=RouteSession(self.route,'task');session.select('lift',1)
        with self.assertRaisesRegex(ValueError,'external confirmation'):session.select('inside',2)
        feedback=dict(event='floor_transition_confirmed',task_id='task',segment_id='lift',goal_revision=1,
            from_floor='f1',to_floor='f2',source='external_confirmed_floor',source_event_id='synthetic_event',
            stamp_s=10.,clock_domain='synthetic')
        with self.assertRaises(ValueError):session.confirm_transition(dict(feedback,goal_revision=0))
        result=session.confirm_transition(feedback);self.assertFalse(result['task_completed'])
        self.assertEqual(session.segment_id,'lift')
        session.select('inside',2)
        args=dict(now=10.,clock_domain='synthetic',config=self.config,asset=asset())
        session.request(bundle(),self.geometry,**args)
        with self.assertRaisesRegex(ValueError,'reselection'):session.request(bundle(generation=1),self.geometry,**args)

    def test_matching_does_not_jump_to_end_of_loop_or_go_backwards(self):
        points=np.array([[0,0],[10,0],[10,10],[0,10],[0,0],[0,-10]])
        _,progress,_,_=_lookahead(points,np.array([0,0]),0.,2.,5.,2.)
        self.assertEqual(progress,0.)
        with self.assertRaises(ValueError):_lookahead(points,np.array([8,8]),0.,2.,5.,2.)
        _,progress,_,_=_lookahead(points,np.array([2,0]),3.,2.,5.,2.)
        self.assertEqual(progress,3.)

    def test_reference_binding_is_independent_of_callers_mutations(self):
        record=reference();route=RoutePlan(record);record['anchors']['door']['map_id']='changed'
        self.assertEqual(route.anchors['door']['map_id'],'map-a')
        record=reference();record['segments'][1]['points_xy']=[[0,0],[0,0]]
        with self.assertRaises(ValueError):RoutePlan(record)

