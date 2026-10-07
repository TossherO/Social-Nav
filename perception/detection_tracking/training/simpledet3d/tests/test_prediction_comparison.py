import unittest

from tools.compare_wheelchair_predictions import select_balanced_records


class PredictionComparisonTest(unittest.TestCase):
    def test_balanced_selection_is_unique_and_covers_all_groups(self):
        records = []
        for index in range(16):
            record = {
                'dataset_index': index,
                'frame_id': f'scene/{index:04d}',
                'scene_id': 'scene',
                'frame_index': index * 20,
                'manual_centers': 3,
                'single_hits': 2,
                'four_hits': 2,
                'single_misses': 1,
                'four_misses': 1,
                'single_boxes': 3,
                'four_boxes': 3,
                'single_unmatched': 1,
                'four_unmatched': 1,
                'hit_delta': 0,
                'unmatched_delta': 0,
            }
            if index < 4:
                record.update(hit_delta=1, single_hits=3, single_misses=0,
                              four_hits=2, four_misses=1)
            elif index < 8:
                record.update(hit_delta=-1, single_hits=2, single_misses=1,
                              four_hits=3, four_misses=0)
            elif index < 12:
                record.update(unmatched_delta=2, single_unmatched=1,
                              four_unmatched=3)
            records.append(record)

        selected = select_balanced_records(records, count=8, min_frame_gap=10)

        self.assertEqual(len(selected), 8)
        self.assertEqual(len({record['frame_id'] for record in selected}), 8)
        self.assertEqual(
            {record['category'] for record in selected},
            {'single_better', 'four_better', 'single_cleaner_tie', 'uniform'},
        )


if __name__ == '__main__':
    unittest.main()
