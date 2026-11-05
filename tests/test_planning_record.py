import json
from pathlib import Path
import tempfile
import unittest

from runtime.planning_record import PlanningRecorder
from tests import test_navigation_dwa as fixtures


class PlanningRecordTests(unittest.TestCase):
    def test_saved_scene_and_reference_keep_original_bindings(self):
        f=fixtures.DwaTests();f.setUp()
        from perception.environment.scene import SceneInputs
        from navigation.planning.continuity import check_reference
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'planning';recorder=PlanningRecorder(root)
            packet=dict(sequence_id=1,stamp_s=10.,packet_binding='test-packet',scene_binding=f.scene.record['input_fingerprint'])
            job=dict(job_id=1,epoch=0,request=f.req,now=10.,clock_rate=1.,sent_mono=100.,scene=f.scene)
            reply=(job,dict(job_id=1,result=f.reference,error=None,compute_s=.01))
            self.assertTrue(recorder.submit(packet,f.scene,None,f.reference,None,job,reply,dt=.1,timing={}))
            self.assertTrue(recorder.submit(dict(packet,sequence_id=2,stamp_s=10.1),f.scene,None,f.reference,None,None,reply,dt=.1,timing={}))
            report=recorder.close();self.assertTrue(report['complete'],report)
            self.assertEqual((report['written'],report['scenes'],report['replies']),(2,1,1))
            scene=SceneInputs.load(root/'scenes'/f.scene.record['input_fingerprint'])
            ref=json.loads((root/'references'/(f.reference['binding']+'.json')).read_text())
            check_reference(ref,f.req,scene,None,5.)
            rows=[json.loads(x) for x in (root/'cycles.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['saved_scene'],scene.record['input_fingerprint'])
            self.assertIsNone(rows[1]['saved_scene'])
            # Rendering must use the exact consumer scene and reject a foreign
            # control packet rather than attaching an attractive but wrong map.
            from navigation.planning.timeline import attach_planning
            (root/'config.json').write_text(json.dumps(dict(geometry=dict(footprint=f.geometry.footprint.tolist(),
                base_from_lidar=f.geometry.base_from_lidar.tolist(),source='synthetic'))))
            (root.parent/'summary.json').write_text(json.dumps(dict(planning_record=report)))
            second=dict(packet,sequence_id=2,stamp_s=10.1)
            (root.parent/'control.jsonl').write_text(json.dumps(packet)+'\n'+json.dumps(second)+'\n')
            data=dict(frames=[dict(sequence=i,scene_binding=scene.record['input_fingerprint']) for i in (1,2)])
            attach_planning(data,root.parent)
            self.assertEqual(len(data['navigation_maps']),1)
            self.assertEqual(data['recorded_references'][ref['binding']]['path'],ref['result']['path'])
            (root.parent/'control.jsonl').write_text(json.dumps(dict(packet,packet_binding='foreign'))+'\n'+json.dumps(second)+'\n')
            with self.assertRaisesRegex(ValueError,'control cycle'):attach_planning(data,root.parent)

    def test_reference_details_preserve_unknown_rejection(self):
        from navigation.planning.reference import ReferenceRuntime
        from tests.test_navigation_runtime import FakeWorker
        from tests.test_navigation_planning import bundle
        import numpy as np
        f=fixtures.DwaTests();f.setUp();scene=bundle(np.full((60,60),-1,np.int8))
        with ReferenceRuntime(f.cfg,f.motion,f.geometry,worker_factory=FakeWorker) as reference:
            reference.diagnostics=True
            self.assertTrue(reference.update(f.req,scene,None,now=10.))
            result=reference.tick(now=10.,clock_domain='synthetic')
            self.assertEqual(result['reason'],'start_footprint_not_free')
            self.assertGreater(result['start_evidence']['unknown'],0)
            self.assertIsNone(reference.worker.job)

    def test_sampling_diagnostic_identifies_speed_recovery_without_changing_proposal(self):
        f=fixtures.DwaTests();f.setUp();feedback=dict(f.feedback,velocity=[.95,0.])
        plain=f.controller.command(f.req,f.scene,feedback,f.reference,now=10.,clock_domain='synthetic')
        f.controller.clear();f.controller.diagnostics=True
        traced=f.controller.command(f.req,f.scene,feedback,f.reference,now=10.,clock_domain='synthetic')
        self.assertEqual(plain['velocity'],traced['velocity'])
        self.assertTrue(traced['sampling']['empty_window'])
        self.assertEqual(traced['sampling']['observation_velocity'],[.95,0.])
