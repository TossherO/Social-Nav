import unittest
from tests.test_navigation_planning import bundle
from runtime.scenes import ScenePool,read_scene


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
