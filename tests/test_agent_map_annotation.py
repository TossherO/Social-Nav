"""Semantic maps stay source-bound and shadow commands never reach a live session."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from decision.agent.map_annotation import Annotation
from decision.agent.tools.places import Catalog
from decision.agent.adapters.shadow import mapped_origin
from decision.agent.adapters.runtime import RuntimeClient
from runtime.command import CommandInbox


class MapAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.map=self.root/'map';self.map.mkdir()
        (self.map/'manifest.yaml').write_text('manifest')
        (self.map/'layers.npz').write_bytes(b'layers')
        grid=SimpleNamespace(metadata=dict(map_id='localization',snapshot_id='static',frame_id='map',
            origin_xy=[-2.,-2.],resolution=1.),layers={'occupancy':np.zeros((6,6),dtype=np.int8)})
        self.asset=SimpleNamespace(manifest=dict(snapshot_id='static',frame_id='map'),snapshot=grid)
        self.mock=patch('decision.agent.map_annotation.StaticEnvironmentMap',return_value=self.asset);self.mock.start();self.addCleanup(self.mock.stop)
        self.editor=Annotation(self.map,'floor')

    def test_export_roundtrip_and_map_identity(self):
        self.editor.update('点','entrance','入口',['门'],[0,1],'靠近入口')
        self.editor.update('区域','hall','大厅',[],[[0,0],[2,0],[2,2],[0,2]])
        self.editor.update('地标','desk','前台',[],[1,1])
        out=self.root/'catalog.json';digest=self.editor.save(out)
        catalog=Catalog.load(out)
        self.assertEqual(catalog.value['map_snapshot_id'],'static')
        self.assertEqual(catalog.value['localization_map_id'],'localization')
        self.assertEqual(catalog.annotations()['points'][0]['description'],'靠近入口')
        self.assertEqual(catalog.places['mission_start']['pose'],None)
        observation=dict(stamp_s=1,map_base_pose=[0,0,0],map_binding=dict(valid=True,map_id='localization',map_frame='map'))
        self.assertEqual(mapped_origin(catalog,observation)['origin_pose'],[0,0,0])
        observation['map_binding']['map_id']='other'
        self.assertIsNone(mapped_origin(catalog,observation)['origin_pose'])
        self.editor.save(out,digest)
        out.write_text('{}')
        with self.assertRaisesRegex(ValueError,'外部修改'):self.editor.save(out,digest)

    def test_invalid_elements_do_not_change_the_catalog(self):
        original=json.dumps(self.editor.value,sort_keys=True)
        for kind,identity,coordinates in [('点','outside',[99,99]),('区域','cross',[[0,0],[2,2],[0,2],[3,0]]),
                                            ('地标','mission_start',[0,0])]:
            with self.assertRaises(ValueError):self.editor.update(kind,identity,'名称',[],coordinates)
        self.assertEqual(json.dumps(self.editor.value,sort_keys=True),original)

    def test_loading_other_map_is_rejected(self):
        path=self.root/'catalog.json';self.editor.save(path)
        self.asset.manifest['snapshot_id']='another'
        with self.assertRaisesRegex(ValueError,'不一致'):Annotation(self.map,'floor',path)

    def test_changed_source_map_cannot_be_saved(self):
        (self.map/'layers.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'已改变'):self.editor.save(self.root/'catalog.json')


class ShadowIsolationTests(unittest.TestCase):
    def test_live_session_refuses_agent_goal_and_cancel_before_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            sent=[];session=SimpleNamespace(execute=True,note_command=lambda *a:None)
            inbox=CommandInbox(Path(directory)/'commands.sock',SimpleNamespace(command=lambda *args:sent.append(args)),session=session)
            self.addCleanup(inbox.close)
            client=RuntimeClient(Path(directory)/'commands.sock')
            goal=dict(request_id='call',task_id='task',goal_revision=1,mode='mapped',frame_id='map',
                      target_reference='base_link',goal=[1.,2.,0.],map_snapshot_id='static')
            self.assertFalse(client.exchange('goal',goal)['accepted'])
            self.assertFalse(client.exchange('cancel',dict(task_id='task',goal_revision=1))['accepted'])
            self.assertEqual(sent,[]);self.assertEqual(inbox.drain(),[])
            session.execute=False
            self.assertTrue(client.exchange('goal',goal)['accepted'])
            self.assertEqual(sent,[('goal',goal)])


if __name__=='__main__':unittest.main()
