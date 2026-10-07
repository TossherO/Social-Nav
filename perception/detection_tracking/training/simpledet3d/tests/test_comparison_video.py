import unittest

from tools.create_wheelchair_comparison_video import aligned_frame_indices


class TestComparisonVideo(unittest.TestCase):
    def test_alignment_keeps_dataset_order(self):
        infos = [{'token': 'scene/0002'}, {'token': 'scene/0001'}]
        single = {'scene/0001': {}, 'scene/0002': {}}
        four = {'scene/0002': {}, 'scene/0001': {}}
        self.assertEqual(
            aligned_frame_indices(infos, single, four),
            [(0, 'scene/0002'), (1, 'scene/0001')],
        )

    def test_alignment_rejects_missing_or_extra_predictions(self):
        infos = [{'token': 'scene/0001'}, {'token': 'scene/0002'}]
        single = {'scene/0001': {}, 'scene/0002': {}}
        four = {'scene/0001': {}, 'scene/extra': {}}
        with self.assertRaisesRegex(ValueError, 'missing=1, extra=1'):
            aligned_frame_indices(infos, single, four)

    def test_alignment_rejects_duplicate_info_tokens(self):
        infos = [{'token': 'same'}, {'token': 'same'}]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            aligned_frame_indices(infos, {'same': {}}, {'same': {}})


if __name__ == '__main__':
    unittest.main()
