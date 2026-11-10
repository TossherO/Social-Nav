"""Offline review keeps identities and source/host clocks separate."""
import json
from pathlib import Path
import tempfile
import unittest

from navigation.control.review import summarize


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


class ControlReviewTests(unittest.TestCase):
    def test_join_uses_bindings_and_never_invents_write_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_rows(root/'control.jsonl',[dict(stamp_s=100.,clock_domain='runtime_live',sequence_id=4,
                packet_binding='packet',request={'goal':[1.,2.]},proposal=dict(status='candidate',velocity=[.1,0.]))])
            write_rows(root/'planning/cycles.jsonl',[dict(packet_binding='packet',trajectory=dict(
                trajectory=dict(people_check=dict(hard_horizon_s=1.,future_conflict_duration_s=.1))))])
            write_rows(root/'safety.jsonl',[dict(stamp_s=100.01,clock_domain='runtime_live',sequence_id=7,
                decision_complete_mono=30.,proposal_binding='packet',state='stop_request',reason='predicted_people_collision',
                people_check=dict(critical_relative_time_s=1.2))])
            write_rows(root/'chassis/events.jsonl',[
                dict(kind='wire_target',mono=29.,units=[0,0,3],velocity_mps_rps=[0.,0.]),
                dict(kind='authorization',mono=30.01,packet=dict(allowed=False,units=[],expires_mono=30.15,
                    diagnostics=dict(safety_sequence=7))),
                dict(kind='wire_target',mono=30.02,units=[0,0,1],velocity_mps_rps=[0.,0.]),
                dict(kind='feedback',mono=30.03,timestamp_source='host_receive_monotonic',device_measurement_stamp_s=None,
                    feedback=dict(velocity_mps_rps=[.03,0.],mode_raw=1,candidate_checksum_matches=True))])
            result=summarize(root)
            control,safety=result['source_clock_events'];events=result['chassis_monotonic_events']
            self.assertEqual(control['people_check']['hard_horizon_s'],1.)
            self.assertEqual(safety['control_sequence_id'],4)
            self.assertEqual(safety['people_check']['critical_relative_time_s'],1.2)
            self.assertIsNone(events[0]['preceding_authorization'])
            self.assertEqual(events[1]['matched_control_sequence'],4)
            self.assertFalse(events[2]['preceding_authorization']['allowed'])
            self.assertIsNone(events[3]['device_measurement_stamp_s'])
            self.assertIsNone(result['chassis_telemetry_quality'])
            self.assertIsNone(result['run_quality']['safety_telemetry_dropped_max'])

    def test_missing_or_duplicate_identity_is_not_linked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_rows(root/'control.jsonl',[dict(stamp_s=1.,clock_domain='a',sequence_id=1,packet_binding='dup'),
                dict(stamp_s=2.,clock_domain='a',sequence_id=3,packet_binding='dup')])
            write_rows(root/'safety.jsonl',[dict(sequence_id=2,stamp_s=1.,clock_domain='a',state='stop_request'),
                dict(sequence_id=2,stamp_s=2.,clock_domain='a',state='stop_request',proposal_binding='dup')])
            write_rows(root/'chassis/events.jsonl',[dict(kind='authorization',mono=3.,packet=dict(diagnostics=dict(safety_sequence=2)))])
            result=summarize(root)
            self.assertEqual(result['ambiguous_safety_sequences'],[2])
            self.assertIsNone(result['chassis_monotonic_events'][0]['matched_safety_sequence'])
            self.assertEqual(result['ambiguous_packet_bindings'],['dup'])
            matched=[row for row in result['source_clock_events'] if row['kind']=='safety' and row['proposal_binding']=='dup'][0]
            self.assertIsNone(matched['control_sequence_id'])
            self.assertEqual(result['counts']['safety_reasons'],{'reason_missing':2})


if __name__=='__main__':unittest.main()
