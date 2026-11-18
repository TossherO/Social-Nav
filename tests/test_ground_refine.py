"""Synthetic checks for the optional vertical-only floor constraints."""
import unittest
import numpy as np
from perception.mapping_localization.ground_refine import estimate_offsets


class GroundRefineTests(unittest.TestCase):
    def test_time_held_out_floor_deviation_decreases(self):
        times = np.arange(180., dtype=float)
        heights = -1.0 + .15*np.sin(times/25.)
        split = lambda stamp: ((stamp//30).astype(int)%2) == 0
        delta, stats, (ids, stamps, grouped, train) = estimate_offsets(times, times, heights, train=split)
        before = np.median(np.abs(grouped[~train]-stats['target_height_m']))
        after = np.median(np.abs(grouped[~train]+delta[ids[~train]]-stats['target_height_m']))
        self.assertLess(after, before)
        self.assertLess(np.max(np.abs(np.diff(delta))), .03)
        self.assertGreater((~train).sum(), 20)

    def test_rejects_insufficient_and_outside_observations(self):
        times = np.arange(100., dtype=float)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            estimate_offsets(times, times[:10], np.zeros(10))
        with self.assertRaisesRegex(ValueError, 'outside'):
            estimate_offsets(times, np.arange(-1., 80.), np.zeros(81))


if __name__ == '__main__': unittest.main()
