"""Log review preserves asynchronous source times and never invents success."""
import unittest

from navigation.planning.timeline import assemble,classify,compact_safety,reference_records


class PlanningTimelineTests(unittest.TestCase):
    def test_reference_root_reason_does_not_conflate_wait_with_failed_search(self):
        control=dict(request={'goal_revision':1},proposal=dict(status='unavailable',reason='waiting_for_background_reference'))
        self.assertEqual(classify(control,dict(reason='start_footprint_not_free')),('reference','start_footprint_not_free'))
        self.assertEqual(classify(control,dict(reason='waiting_for_background_reference')),('reference','waiting_for_background_reference'))
        control['proposal']=dict(status='candidate')
        self.assertEqual(classify(control,dict(reason='reference_has_no_path')),('generated','candidate'))
        control['request']=None
        self.assertEqual(classify(control,None),('idle','no_active_goal'))

    def test_result_arrival_is_not_replaced_by_old_scene_time(self):
        rows=[dict(event='status',as_of=12.,reason=None),
              dict(event='reference_ready',as_of=10.,job_id=1)]
        status,events=reference_records(rows)
        self.assertIn(12.,status);self.assertEqual(events[0]['recorded_tick_s'],12.)
        self.assertEqual(events[0]['as_of'],10.)

    def test_latest_safety_and_later_matching_candidate_are_separate(self):
        run=dict(route_replay=dict(source=dict(source_start_s=100.)))
        controls=[dict(stamp_s=101.,sequence_id=1,request=dict(goal_revision=1),
            proposal=dict(status='candidate',trajectory_binding='mine')),
            dict(stamp_s=101.2,sequence_id=2,request=None,proposal=None)]
        refs=[dict(event='status',as_of=101.,reason=None,blocked=False)]
        safety=[dict(stamp_s=100.9,state='stop_request',reason='old',trajectory_binding='old'),
                dict(stamp_s=101.1,state='allowed',trajectory_binding='mine')]
        data=assemble(run,controls,refs,safety,[],[]);frame=data['frames'][0]
        self.assertEqual(frame['safety']['trajectory_binding'],'old')
        self.assertAlmostEqual(frame['matched_safety']['t'],1.1)
        self.assertTrue(frame['generated']);self.assertEqual(data['summary']['active_ticks'],1)
        self.assertEqual(data['summary']['no_goal_ticks'],1)
        self.assertIsNone(data['frames'][1]['reference_available'])

    def test_missing_reference_record_and_infinite_clearance_stay_missing(self):
        run=dict(route_replay=dict(source=dict(source_start_s=0.)))
        controls=[dict(stamp_s=1.,sequence_id=1,request=dict(goal_revision=1),
            proposal=dict(status='unavailable',reason='feedback_invalid'))]
        data=assemble(run,controls,[],[],[],[])
        self.assertIsNone(data['frames'][0]['reference_available'])
        self.assertEqual(data['frames'][0]['stage'],'input')
        self.assertIsNone(data['frames'][0]['safety'])
        row=dict(stamp_s=1.,state='allowed',people_check=dict(minimum_clearance_m=float('inf')))
        self.assertIsNone(compact_safety(row,0.)['minimum_clearance'])

    def test_absent_proposal_is_not_a_recorded_planning_failure(self):
        run=dict(route_replay=dict(source=dict(source_start_s=0.)))
        controls=[dict(stamp_s=1.,sequence_id=1,request=dict(goal_revision=1))]
        data=assemble(run,controls,[],[],[],[])
        self.assertEqual(data['frames'][0]['stage'],'unrecorded')
        self.assertEqual(data['summary']['cause_counts'],{})


if __name__=='__main__':unittest.main()
