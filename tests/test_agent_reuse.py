"""Historical requirements can seed a new task without replaying execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision.agent.core.reuse import load_registered_task,verify_registered_task
from decision.agent.core.store import Store


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'session.sqlite'
        self.goal=dict(goal_id='a',kind='point',target_id='door',description='到门')
        self.state=dict(task_id='old',revision=1,instruction='到门',initial_floor='floor2',clarifications=[],
            task_goals=[dict(requirement=self.goal,status='completed',selected=dict(pose=[1,2,0]),completion={'proof':'old'})])
        self.annotations=dict(points=[dict(place_id='door',name='门',pose=[1,2,0],reference='base_link')],regions=[])
        self.asset=dict(frame_id='map',localization_map_id='map-id',global_binding='static',annotations=self.annotations)
        store=Store(self.path)
        store.save(self.state,'task_created',{})
        store.event('reference_map_prepared',dict(frame_id='map',global_binding='static',annotations=self.annotations))
        store.event('model_action_applied',dict(call=dict(tool='record_task'),context=dict(task_id='old',revision=1,
            map_binding=dict(valid=True,map_id='map-id')),result=dict(status='recorded',original_instruction='到门',
            goals=[dict(requirement=self.goal)])))
        store.close()

    def test_legacy_registration_is_loaded_and_static_identity_rechecked(self):
        plan=load_registered_task(self.path)
        self.assertEqual(plan['goals'],[self.goal])
        self.assertNotIn('selected',plan['goals'][0]);self.assertNotIn('completion',plan['goals'][0])
        with patch('decision.agent.core.reuse.catalog_assets',return_value={'floor2':self.asset}):
            self.assertEqual(verify_registered_task(plan,{'floor2':object()},'floor2'),{'floor2':self.asset})
            with self.assertRaisesRegex(ValueError,'起始楼层'):verify_registered_task(plan,{'floor2':object()},'floor1')
        modified=copy.deepcopy(self.asset);modified['global_binding']='another'
        with patch('decision.agent.core.reuse.catalog_assets',return_value={'floor2':modified}):
            with self.assertRaisesRegex(ValueError,'静态地图'):verify_registered_task(plan,{'floor2':object()},'floor2')

    def test_mismatched_registration_and_missing_floor_evidence_are_rejected(self):
        db=Store(self.path)
        changed=copy.deepcopy(self.state);changed['task_goals'][0]['requirement']['description']='不同'
        db.save(changed,'user_event',{});db.close()
        with self.assertRaisesRegex(ValueError,'一致的成功登记'):load_registered_task(self.path)

    def test_missing_second_floor_provenance_is_not_assumed(self):
        db=Store(self.path)
        elevator=dict(goal_id='lift',kind='skill',target_id='',description='乘梯',stage_type='elevator',
                      floor_id='floor2',to_floor='floor1')
        state=copy.deepcopy(self.state);state['task_goals']=[dict(requirement=elevator,status='pending')]
        db.save(state,'user_event',{})
        db.event('model_action_applied',dict(call=dict(tool='record_task'),context=dict(task_id='old',revision=1,
            map_binding=dict(valid=True,map_id='map-id')),result=dict(status='recorded',original_instruction='到门',
            goals=[dict(requirement=elevator)])))
        db.close()
        with self.assertRaisesRegex(ValueError,'缺少部分楼层'):load_registered_task(self.path)

    def test_cross_floor_provenance_covers_each_registered_floor(self):
        db=Store(self.path)
        second=dict(self.asset,localization_map_id='floor1-map',global_binding='floor1-static')
        db.event('task_asset_provenance',dict(floors={'floor2':self.asset,'floor1':second}))
        elevator=dict(goal_id='lift',kind='skill',target_id='',description='乘梯',stage_type='elevator',
                      floor_id='floor2',to_floor='floor1')
        state=copy.deepcopy(self.state);state['task_goals']=[dict(requirement=elevator,status='pending')]
        db.save(state,'user_event',{})
        db.event('model_action_applied',dict(call=dict(tool='record_task'),context=dict(task_id='old',revision=1,
            map_binding=dict(valid=True,map_id='map-id')),result=dict(status='recorded',original_instruction='到门',
            goals=[dict(requirement=elevator)])))
        db.close()
        plan=load_registered_task(self.path)
        self.assertEqual(set(plan['assets']),{'floor1','floor2'})
        with patch('decision.agent.core.reuse.catalog_assets',return_value={'floor2':self.asset,'floor1':second}):
            verify_registered_task(plan,{'floor1':object(),'floor2':object()},'floor2')
            with self.assertRaisesRegex(ValueError,'全部楼层'):verify_registered_task(plan,{'floor2':object()},'floor2')

    def test_preview_digest_changes_when_a_new_registration_replaces_the_task(self):
        before=load_registered_task(self.path)
        db=Store(self.path);state=copy.deepcopy(self.state);state.update(task_id='new',revision=2,instruction='新任务')
        db.save(state,'user_event',{})
        db.event('model_action_applied',dict(call=dict(tool='record_task'),context=dict(task_id='new',revision=2,
            map_binding=dict(valid=True,map_id='map-id')),result=dict(status='recorded',original_instruction='新任务',
            goals=[dict(requirement=self.goal)])))
        db.close()
        self.assertNotEqual(load_registered_task(self.path)['digest'],before['digest'])


if __name__=='__main__':unittest.main()
