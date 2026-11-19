"""Cross-floor evidence checks; no camera, model service or wheelchair."""
import unittest

from decision.agent.core.elevator_transition import ElevatorTransition


class ElevatorTransitionTests(unittest.TestCase):
    def setUp(self):
        self.transition=ElevatorTransition(task_id='task',goal_id='floor-change',from_floor='floor2',
                                           to_floor='floor1',destination_map_id='floor1-map')
        self.image=b'\xff\xd8synthetic\xff\xd9'

    def frame(self, sequence, stamp):
        return dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
                    frame_id='camera',stamp_s=stamp,sequence=sequence)

    def association(self, **changes):
        result=dict(valid=True,map_id='floor1-map',map_frame='map',local_session_id='new-session',
                    local_generation=0,local_frame='odom',association_session='association',
                    association_generation=1,measurement_stamp_s=12.)
        result.update(changes)
        return result

    def test_unknown_does_not_advance_and_fresh_destination_map_is_required(self):
        t=self.transition
        t.observe(camera_record=self.frame(0,10.),image=self.image,judgment='unknown',evidence='画面不足')
        self.assertEqual(t.status,'monitoring_ride')
        t.observe(camera_record=self.frame(1,11.),image=self.image,judgment='not_completed',evidence='仍在途中')
        t.observe(camera_record=self.frame(2,12.),image=self.image,judgment='completed',evidence='已完成乘梯')
        self.assertEqual(t.status,'awaiting_destination_localization')
        with self.assertRaisesRegex(ValueError,'localization'):
            t.accept_localization(association=self.association(map_id='old-map'),now_s=12.2,max_age_s=1.,clock_domain='ros_time')
        with self.assertRaisesRegex(ValueError,'localization'):
            t.accept_localization(association=self.association(measurement_stamp_s=11.),now_s=12.2,max_age_s=2.,clock_domain='ros_time')
        result=t.accept_localization(association=self.association(),now_s=12.2,max_age_s=1.,clock_domain='ros_time')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['task_id'],'task')
        self.assertFalse(result['physical_elevator_control'])

    def test_duplicate_or_changed_camera_cannot_advance(self):
        t=self.transition
        t.observe(camera_record=self.frame(1,11.),image=self.image,judgment='not_completed',evidence='轿厢')
        with self.assertRaisesRegex(ValueError,'advance'):
            t.observe(camera_record=self.frame(1,11.),image=self.image,judgment='completed',evidence='走廊')
        bad=self.frame(2,12.);bad['frame_id']='another-camera'
        with self.assertRaisesRegex(ValueError,'identity'):
            t.observe(camera_record=bad,image=self.image,judgment='completed',evidence='走廊')
        self.assertEqual(t.status,'monitoring_ride')

    def test_direct_completion_requires_no_intermediate_entry_judgment(self):
        self.transition.observe(camera_record=self.frame(0,10.),image=self.image,
                                judgment='completed',evidence='结合当前与历史观察已完成乘梯')
        self.assertEqual(self.transition.status,'awaiting_destination_localization')
