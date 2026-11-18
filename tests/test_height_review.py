"""Revisit analysis keeps position/time constraints and never treats proximity as truth."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from perception.mapping_localization.height_review import review


class HeightReviewTests(unittest.TestCase):
    def test_detects_delayed_nearby_height_disagreement(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);(path/'keyframes').mkdir();(path/'COMPLETE').write_text('complete')
            poses=np.array([[1,0,0,0,0,0,0,0,1],[20,3,0,0,0,0,0,1,1],
                            [100,.5,0,.7,0,0,0,2,1]],float)
            np.savetxt(path/'keyframes/poses.txt',poses)
            value=review(path,radius_m=1.,separation_s=60.)
            self.assertEqual(value['revisit']['pairs'],1)
            self.assertEqual(value['revisit']['pairs_over_0_5m'],1)
            self.assertEqual(value['revisit']['highest'][0]['keyframes'],[0,2])
            self.assertIn('no_independent_ground_truth',value['limitations'])


if __name__=='__main__':unittest.main()
