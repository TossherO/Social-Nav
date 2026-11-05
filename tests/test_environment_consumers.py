"""Neutral model inputs and read-only agent tools; no model request or motion."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from decision.vlm.environment_inputs import prepare_environment_context, render_environment_context
from decision.agent.tools.environment import query_environment
from perception.environment.layers import GridSnapshot
from perception.environment.context import layer_binding
from perception.environment.storage import StaticEnvironmentMap, content_hash


def snapshot():
    occupancy=np.array([[-1,-1,100],[-1,0,100],[-1,0,0]],np.int8)
    return GridSnapshot(dict(snapshot_id='local:0:1',session_id='local',generation=0,frame_id='odom',
        stamp_s=10.,geometry_status='OBSERVED',world_from_lidar=np.eye(4).tolist(),
        origin_xy=[-1,-1],resolution=1.,shape=[3,3]),
        dict(occupancy=occupancy,ground_z=np.zeros((3,3)),clearance=np.zeros((3,3)),
             coverage=np.ones((3,3)),last_observed=np.full((3,3),10.),unknown_distance=np.zeros((3,3))))


class ConsumerTests(unittest.TestCase):
    def test_layer_binding_keeps_legacy_bytes_for_different_array_layouts(self):
        current = snapshot()
        expected = 'a17d05835a8fd03eba2755671f0e4456f8ed5be5ac2e099ee656572ff3ed0825'
        for layout in ('C', 'F', 'strided'):
            with self.subTest(layout=layout):
                layers = {}
                for name, values in current.layers.items():
                    if layout == 'strided':
                        buffer = np.empty((values.shape[0], values.shape[1]*2), dtype=values.dtype)
                        buffer[:, ::2] = values
                        layers[name] = buffer[:, ::2]
                    else:
                        layers[name] = np.array(values, order=layout)
                self.assertEqual(layer_binding(GridSnapshot(current.metadata, layers)), expected)

    def context(self, current, rgb_stamp_s=9.9):
        return prepare_environment_context(current,np.zeros((20,30,3),np.uint8),
            rgb_stamp_s=rgb_stamp_s,as_of=10.,observation_id='camera_observation',clock_domain='synthetic',
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
        with self.assertRaises(ValueError):self.context(current,10.1)
        with self.assertRaises(ValueError):self.context(current,8.)

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
            manifest=dict(kind='static_environment',snapshot_id=current.metadata['snapshot_id'],frame_id='odom',
                layers_sha256=content_hash(path/'layers.npz'),
                time_scope='offline_final_prior',source={'kind':'synthetic'})
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            package=StaticEnvironmentMap(path)
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
