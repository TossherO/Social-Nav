"""Shared buffer leases and read-only scene transport."""
import dataclasses
import unittest
import numpy as np
from interfaces.shared import SharedArrayPool,BufferFull
from perception.environment.shared_scene import ScenePool,read_scene
from tests.navigation_fixtures import bundle

class BufferTests(unittest.TestCase):
    def test_leases_read_only_and_epoch_reuse(self):
        pool=SharedArrayPool(1,1024);reader=pool.reader()
        ref=pool.publish({'xyz':np.arange(18,dtype=np.float32).reshape(6,3)},readers=['gpu','safety'])
        values=reader.arrays(ref)
        np.testing.assert_array_equal(values['xyz'],np.arange(18).reshape(6,3))
        with self.assertRaises(ValueError):values['xyz'][0,0]=9
        with self.assertRaises(BufferFull):pool.publish({'a':np.ones(3)},readers=['gpu'])
        pool.release(ref,'gpu')
        with self.assertRaises(BufferFull):pool.publish({'a':np.ones(3)},readers=['gpu'])
        pool.release(ref,'safety');values.clear()
        newer=pool.publish({'a':np.ones(3)},readers=['gpu'])
        with self.assertRaises(ValueError):reader.arrays(ref)
        with self.assertRaises(ValueError):pool.release(ref,'gpu')
        pool.release(newer,'gpu');reader.close();pool.close()

    def test_capacity_and_mutated_descriptor_rejected(self):
        pool=SharedArrayPool(1,256);reader=pool.reader()
        with self.assertRaises(BufferFull):pool.publish({'big':np.ones(200)},readers=['x'])
        ref=pool.publish({'x':np.ones(4)},readers=['x'])
        bad=dataclasses.replace(ref,content_sha256='wrong')
        with self.assertRaises(ValueError):reader.arrays(bad)
        with self.assertRaises(RuntimeError):pool.close()
        pool.release(ref,'x');reader.close();pool.close()


class SharedSceneTests(unittest.TestCase):
    def test_embedded_people_restore_bound_clock_and_health_without_losing_tracks(self):
        from perception.environment.scene import assemble_scene,people_observation
        from perception.environment.layers import load_config
        from tests.test_navigation_prediction import tracked_scene,track
        scene=tracked_scene([track()]);people=people_observation(scene)
        restored=assemble_scene(scene.geometry,clock_domain=scene.record['clock_domain'],people=people,settings=load_config()['scene'])
        self.assertEqual(restored.record['people'],scene.record['people'])
        with self.assertRaises(ValueError):
            scene.record['people']['status']='missing';people_observation(scene)

    def test_bound_scene_arrays_are_read_only_and_not_serialized(self):
        scene=bundle();pool=ScenePool(1,1024*1024);reader=pool.pool.reader()
        description=pool.publish(scene,'reference')
        restored=read_scene(reader,description)
        self.assertEqual(restored.record['input_fingerprint'],scene.record['input_fingerprint'])
        self.assertFalse(restored.geometry.layers['occupancy'].flags.writeable)
        self.assertNotIn('scene',description)
        with self.assertRaises(ValueError):restored.geometry.layers['occupancy'][0,0]=100
        del restored
        pool.release(description,'reference');reader.close();pool.close()

