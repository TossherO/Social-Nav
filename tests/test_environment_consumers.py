"""Neutral model inputs and read-only agent tools; no model request or motion."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from decision.vlm.environment_inputs import prepare_environment_context, render_environment_context
from decision.agent.environment_tools import query_environment
from perception.environment.layers import GridSnapshot
from perception.environment.memory import create_index
from perception.environment.storage import EnvironmentMap, content_hash


def snapshot():
    occupancy=np.array([[-1,-1,100],[-1,0,100],[-1,0,0]],np.int8)
    return GridSnapshot(dict(snapshot_id='local:0:1',session_id='local',generation=0,frame_id='odom',
        stamp_s=10.,geometry_status='OBSERVED',world_from_lidar=np.eye(4).tolist(),
        origin_xy=[-1,-1],resolution=1.,shape=[3,3]),
        dict(occupancy=occupancy,ground_z=np.zeros((3,3)),clearance=np.zeros((3,3)),
             coverage=np.ones((3,3)),last_observed=np.full((3,3),10.),unknown_distance=np.zeros((3,3))))


class ConsumerTests(unittest.TestCase):
    def context(self, current, history=()):
        return prepare_environment_context(current,history,np.zeros((20,30,3),np.uint8),
            rgb_stamp_s=9.9,as_of=10.,observation_id='camera_observation',clock_domain='synthetic',
            rgb_reference=dict(kind='synthetic'))

    def test_context_is_source_bound_and_rejects_future_stale_or_final_map_input(self):
        current=snapshot()
        context=self.context(current)
        self.assertEqual(context['protocol'],'environment_context')
        self.assertFalse(context['navigation_reachability_verified'])
        self.assertEqual(context['geometry']['cell_counts']['-1'],4)
        wrong=GridSnapshot(dict(current.metadata,time_scope='offline_final_prior'),current.layers)
        with self.assertRaises(ValueError):self.context(wrong)
        old=GridSnapshot(dict(current.metadata,stamp_s=8.),current.layers)
        with self.assertRaises(ValueError):self.context(old)
        future=dict(id='future',stamp_s=10.1,pose_stamp_s=10.,session_id='local',generation=0,frame_id='odom')
        with self.assertRaises(ValueError):self.context(current,[future])
        with self.assertRaises(ValueError):
            self.context(current,[dict(future,stamp_s=9.9,pose_stamp_s=9.8,session_id='other')])

    def test_renderer_rejects_changed_image_or_layers_and_uses_actual_rgb(self):
        current=snapshot(); context=self.context(current)
        rgb=np.zeros((20,30,3),np.uint8)
        result=render_environment_context(current,context,rgb)
        self.assertEqual(result.dtype,np.uint8)
        self.assertEqual(result.shape[2],3)
        with self.assertRaises(ValueError):render_environment_context(current,context,np.ones_like(rgb))
        current.layers['occupancy'][0,0]=0
        with self.assertRaises(ValueError):render_environment_context(current,context,rgb)

    def test_agent_tools_preserve_unknown_and_reject_writes_and_unbound_queries(self):
        current=snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);current.save(path/'layers.npz')
            binding=dict(snapshot_id=current.metadata['snapshot_id'])
            db=create_index(path/'observations.sqlite',binding);db.close()
            manifest=dict(snapshot_id=current.metadata['snapshot_id'],frame_id='odom',
                memory_binding=binding,layers_sha256=content_hash(path/'layers.npz'),
                time_scope='recorded_local_session',source={'kind':'synthetic'})
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            package=EnvironmentMap(path)
            try:
                req=dict(operation='query_geometry',snapshot_id=manifest['snapshot_id'],frame='odom',points=[[-.5,-.5]])
                answer=query_environment(package,req)
                self.assertEqual(answer['data'][0]['state'],'unknown')
                self.assertFalse(answer['navigation_reachability_verified'])
                with self.assertRaises(ValueError):query_environment(package,dict(req,operation='move_robot'))
                with self.assertRaises(KeyError):query_environment(package,dict(req,snapshot_id='other'))
                with self.assertRaises(ValueError):query_environment(package,dict(req,frame='map'))
            finally:package.close()


if __name__=='__main__':unittest.main()
