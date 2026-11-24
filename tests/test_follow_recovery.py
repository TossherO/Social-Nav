"""Test fixtures and disabled legacy follow interfaces."""
import copy
import tempfile
import unittest
from pathlib import Path

from decision.agent import ROOT
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.observations import Observation
from decision.agent.tools.places import Catalog
from decision.vlm.t2.target import candidates
from perception.environment.context import fingerprint
from perception.environment.layers import load_config
from perception.environment.scene import assemble_scene
from tests.test_agent import ScriptedFeedback
from tests.decision_fixtures import observation


def candidate_observation(stamp,track_id='a',*,associated=True,mapped=False):
    row,arrays,_,scene=observation(stamp)
    row['tracks'][0]['track_id']=track_id
    row['input_fingerprint']=fingerprint({k:v for k,v in row.items() if k!='input_fingerprint'})
    if mapped:
        from tests.navigation_fixtures import bundle
        scene.geometry.metadata['association']=copy.deepcopy(bundle().geometry.metadata['association'])
        scene.geometry.metadata['association'].update(stamp_s=stamp,measurement_stamp_s=stamp-.1)
    scene=assemble_scene(scene.geometry,clock_domain='synthetic',people=row,settings=load_config()['scene'])
    detections=[dict(box=[20.,5.,35.,40.],score=.9,class_id=0)]
    association=dict(status='complete',tracks={track_id+'/g7':dict(status='matched' if associated else 'unmatched',
        rgb_box=[20.,5.,35.,40.],detection_index=0 if associated else None)})
    value=candidates(scene,arrays['rgb'],stamp-.02,detections,association,detector=dict(backend='test'))
    return Observation.from_scene(scene,rgb=dict(pixels=arrays['rgb'],stamp_s=stamp-.02,frame_id='camera'),
                                  follow_candidates=value)


class FollowInterfaceTests(unittest.TestCase):
    def test_old_target_selection_and_recovery_cannot_start_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(Path(directory)/'session.sqlite')
            try:
                nav=ScriptedFeedback()
                catalog=copy.deepcopy(Catalog.load(ROOT/'decision/agent/configs/simulation.json').value)
                session=AgentSession('跟随红衣人物',Catalog(catalog),nav,store)
                session.call('record_task',goals=[dict(goal_id='g',kind='skill',stage_type='follow',
                    target_id='',description='跟随红衣人物')])
                self.assertIn('select_follow_target',session.registry)
                self.assertNotIn('request_follow_recovery',session.registry)
                self.assertEqual(session.call('start_local_task',goal_id='g',task_type='follow',target_description='红衣人物')['status'],'refused')
                self.assertIsNone(session.skill.record['handle'])
            finally:store.close()


if __name__=='__main__':
    unittest.main()
