"""T2 choices are applied to the original task and continuous 3D evidence."""
import copy
import json
import unittest

import numpy as np

from decision.vlm.follow_online import prepare
from decision.vlm.observations import render_window
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from perception.environment.context import fingerprint
from perception.environment.layers import load_config as environment_config
from perception.environment.scene import assemble_scene
from runtime.decision import DecisionBuffer
from runtime.follow import FollowRunner
from tests.test_decision_runtime import observation
from tests.test_navigation_prediction import track


def window(start, *, predicted=False):
    buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
    try:
        for i in range(12):
            stamp=round(start+i/10,1)
            row,arrays,association,scene=observation(stamp)
            if predicted and i==11:
                row['tracks'][0].update(state='predicted',last_observed_stamp_s=round(stamp-.1,1))
            second=dict(track(4.,2.5,id='b',stamp=stamp),lifecycle_generation=7,state='observed')
            row['tracks'].append(second)
            row['input_fingerprint']=fingerprint({k:v for k,v in row.items() if k!='input_fingerprint'})
            association['perception_input_fingerprint']=row['input_fingerprint']
            association['result']['tracks']['b/g7']=dict(status='matched',rgb_box=[40,5,55,40])
            scene=assemble_scene(scene.geometry,clock_domain='synthetic',settings=environment_config()['scene'],people=row)
            buffer.push(row,arrays,association,scene=scene)
        lease=buffer.acquire()
        with buffer.consume(lease) as read:
            record,images=render_window(lease['entries'],lease['history'],read,
                extent=[-4,4,-4,4],view_binding={},include_predicted=True)
        return scene,record,images
    finally:buffer.close()


def request(window, images, handle):
    search=dict(clock_domain=window['clock_domain'],source_stamp_s=window['stamp_s'],
                candidates=[dict(id=0,kind='wait',pose=None)])
    search['binding']=fingerprint(search)
    return prepare(window,images,'red jacket',search,task=handle)


def response(prepared,letter):
    kind=prepared['choices'][letter]['kind']
    value=dict(target=letter if kind=='person' else None,observation=letter if kind=='observation' else None)
    return dict(status='complete',request_binding=prepared['binding'],raw_content=json.dumps(value),selection=letter,
                choice=copy.deepcopy(prepared['choices'][letter]))


class FollowT2ApplicationTests(unittest.TestCase):
    def setUp(self):
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.runner=FollowRunner(load_config(),geometry)
        self.handle=dict(task_id='mission',goal_id='g',plan_revision=1,goal_revision=1,call_id='call')
        self.scene,self.window,self.images=window(10.)
        self.runner.observe(self.scene)
        self.prepared=request(self.window,self.images,self.handle)

    def test_selection_starts_with_original_observation_and_rejects_late_result(self):
        outcome=self.runner.apply_t2(self.handle,self.prepared,response(self.prepared,'P0'),self.window,self.scene,now=11.1)
        self.assertEqual(outcome['operation'],'plan')
        self.assertEqual(outcome['target_identity'],['tracker_session',7,'a'])
        self.assertEqual(self.runner.policy.last_visual,11.08)
        self.runner.clear();self.runner.observe(self.scene)
        with self.assertRaisesRegex(ValueError,'expired'):
            self.runner.apply_t2(self.handle,self.prepared,response(self.prepared,'P0'),self.window,self.scene,now=15.2)
        self.assertIsNone(self.runner.policy)

    def test_delayed_occlusion_does_not_interrupt_new_observation(self):
        self.runner.apply_t2(self.handle,self.prepared,response(self.prepared,'P0'),self.window,self.scene,now=11.1)
        covered,source,images=window(10.1,predicted=True);self.runner.observe(covered)
        latest,_,_=window(10.2);self.runner.observe(latest)
        pending=request(source,images,self.handle)
        wait=next(k for k,v in pending['choices'].items() if v['kind']=='observation')
        result=self.runner.apply_t2(self.handle,pending,response(pending,wait),source,latest,now=11.3)
        self.assertEqual(result['operation'],'continue')
        self.assertTrue(result['request_fresh_observation'])
        self.assertEqual(self.runner.policy.last_visual,11.08)
        self.assertEqual(self.runner.update(latest,now=11.3)['operation'],'plan')

    def test_other_visible_person_can_replace_target_with_continuity(self):
        self.runner.apply_t2(self.handle,self.prepared,response(self.prepared,'P0'),self.window,self.scene,now=11.1)
        current,source,images=window(10.1);self.runner.observe(current)
        pending=request(source,images,self.handle)
        person=next(k for k,v in pending['choices'].items() if v.get('identity')==['tracker_session',7,'b'])
        outcome=self.runner.apply_t2(self.handle,pending,response(pending,person),source,current,now=11.2)
        self.assertEqual(outcome['target_identity'],['tracker_session',7,'b'])

    def test_late_search_choice_with_newer_observation_needs_new_visual_confirmation(self):
        self.runner.apply_t2(self.handle,self.prepared,response(self.prepared,'P0'),self.window,self.scene,now=11.1)
        old,source,images=window(10.1);self.runner.observe(old)
        latest,_,_=window(10.2);self.runner.observe(latest)
        pending=request(source,images,self.handle)
        wait=next(k for k,v in pending['choices'].items() if v['kind']=='observation')
        result=self.runner.apply_t2(self.handle,pending,response(pending,wait),source,latest,now=11.3)
        self.assertEqual(result['operation'],'continue')
        self.assertTrue(result['request_fresh_observation'])
        self.assertEqual(self.runner.policy.last_visual,11.08)

    def test_task_change_and_mutated_choice_do_not_replace_active_target(self):
        self.runner.apply_t2(self.handle,self.prepared,response(self.prepared,'P0'),self.window,self.scene,now=11.1)
        wrong=dict(self.handle,call_id='other')
        with self.assertRaises(ValueError):
            self.runner.apply_t2(wrong,self.prepared,response(self.prepared,'P1'),self.window,self.scene,now=11.1)
        changed=response(self.prepared,'P1');changed['choice']['identity']=['fake']
        with self.assertRaises(ValueError):
            self.runner.apply_t2(self.handle,self.prepared,changed,self.window,self.scene,now=11.1)
        self.assertEqual(self.runner.policy.target,('tracker_session',7,'a'))


if __name__=='__main__':unittest.main()
