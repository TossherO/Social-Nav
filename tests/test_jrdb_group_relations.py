"""Raw-label graph contracts for front + near/global BEV review."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from decision.vlm import visualize_jrdb_group_relations as review
from tests.test_jrdb_t1_render import fixture


def row(n, group, interactions=()):
    return {"label_id": f"pedestrian:{n}", "social_group": {"cluster_ID": group},
            "H-interaction": [{"pair": f"pedestrian:{other}", "inter_labels": labels}
                              for other, labels in interactions], "box": [10, 10, 20, 30]}


def three(n, x, y, group=None):
    return {"label_id": f"pedestrian:{n}", "social_group": {"cluster_ID": group},
            "box": {"cx": x, "cy": y, "cz": 0, "l": 1, "w": .6, "h": 1.8, "rot_z": 0}}


class GroupReviewTests(unittest.TestCase):
    def test_membership_is_complete_not_inferred_from_hhi(self):
        graph = review.frame_graph([row(1, 8), row(2, 8), row(3, 8), row(4, 9)], [], [])
        self.assertEqual(len(graph["membership_pairs"]), 3)
        self.assertEqual(graph["hhi_edges"], [])

    def test_reverse_pairs_deduplicated_all_labels_and_scores_retained(self):
        graph = review.frame_graph([
            row(1, 8, [(2, {"conversation": 1}), (2, {"standing together": 2})]),
            row(2, 8, [(1, {"conversation": 3})])], [], [])
        edges = graph["hhi_edges"]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["labels"], ["conversation", "standing together"])
        self.assertTrue(edges[0]["reciprocal"])
        self.assertEqual(len(edges[0]["annotations"]), 3)
        self.assertEqual(edges[0]["annotations"][-1]["inter_labels"]["conversation"], 3)

    def test_cross_group_relation_never_merges_groups(self):
        graph = review.frame_graph([row(1, 3, [(2, {"standing together": 1})]), row(2, 32)], [], [])
        self.assertEqual(graph["membership_pairs"], [])
        self.assertFalse(graph["hhi_edges"][0]["same_group"])
        self.assertFalse(graph["hhi_edges"][0]["reciprocal"])

    def test_stitched_priority_fallback_and_missing_partner_report(self):
        graph = review.frame_graph([row(1, 3, [(9, {"conversation": 1})])],
                                   [row(1, 4), row(2, 5)], [three(1, 1, 1, 6), three(3, 40, 30, 7)])
        actors = graph["actors"]
        self.assertEqual(actors["pedestrian:1"]["group_id"], 3)
        self.assertTrue(actors["pedestrian:1"]["group_conflict"])
        self.assertTrue(review.tag("pedestrian:1", actors).endswith("!"))
        self.assertEqual(actors["pedestrian:2"]["group_source"], "front")
        self.assertEqual(actors["pedestrian:3"]["group_source"], "3d")
        self.assertIsNone(actors["pedestrian:2"]["box_3d"])
        self.assertIn("pair_partner_absent", [w["kind"] for w in graph["warnings"]])

    def test_duplicate_does_not_silently_overwrite(self):
        graph = review.frame_graph([row(1, 3), row(1, 8)], [], [])
        self.assertEqual(graph["actors"]["pedestrian:1"]["group_id"], 3)
        self.assertIn("duplicate_id", [w["kind"] for w in graph["warnings"]])

    def test_bounds_contain_far_boxes_and_ego_with_fixed_axes(self):
        data = {"0.pcd": [three(1, -54, -60)], "1.pcd": [three(2, 10, 17)]}
        bounds = review.sequence_bounds(data)
        for rows in data.values():
            for r in rows:
                corners = review.jrdb.box_corners_xy(r["box"])
                self.assertTrue(((corners[:, 0] >= bounds[0]) & (corners[:, 0] <= bounds[1]) &
                                 (corners[:, 1] >= bounds[2]) & (corners[:, 1] <= bounds[3])).all())
        a = review.bev_xy(0, 0, bounds, 960)
        self.assertLess(review.bev_xy(1, 0, bounds, 960)[1], a[1])
        self.assertLess(review.bev_xy(0, 1, bounds, 960)[0], a[0])
        self.assertAlmostEqual(bounds[1]-bounds[0], bounds[3]-bounds[2])

    def test_renderer_retains_far_and_missing_geometry_actors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence, frames, _, _ = fixture(root)
            calibration = review.jrdb.load_json(root / "calibration/lidars.yaml")["sensor_0"]
            graph = review.frame_graph([row(2, 4, [(9, {"conversation": 1})]), row(9, 4), row(12, 6)],
                                       [row(2, 4)], [three(2, 3, 1, 4), three(9, -54, -60, 4)])
            renderer = review.Renderer(root, sequence, 640, calibration,
                                       review.sequence_bounds({"x": [three(9, -54, -60)]}), lidar=False,
                                       layout="front-near-all")
            result = renderer.render(frames[0], graph)
            self.assertEqual(result.size, (1920, 704))
            self.assertEqual(graph["visible_front_ids"], ["pedestrian:2"])
            self.assertEqual(graph["drawn_bev_ids"], ["pedestrian:2", "pedestrian:9"])
            self.assertIn("pedestrian:12", graph["actors"])
            self.assertEqual(graph["hhi_without_3d"], [])
            self.assertGreater(np.asarray(result).std(), 10)

    def test_two_panels_draw_only_local_bev_but_retain_all_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence, frames, _, _ = fixture(root)
            calibration = review.jrdb.load_json(root / "calibration/lidars.yaml")["sensor_0"]
            graph = review.frame_graph([
                row(2, 4, [(9, {"walking together": 1}), (12, {"conversation": 2})]),
                row(9, 4), row(12, 4)], [row(2, 4)],
                [three(2, 3, 1, 4), three(9, -54, -60, 4), three(15, 4, -3, 7)])
            renderer = review.Renderer(root, sequence, 640, calibration, lidar=False,
                                       local_span=32, legend_types=["walking together", "conversation"])
            result = renderer.render(frames[0], graph)
            self.assertEqual(result.size, (1280, 704))
            self.assertIsNone(renderer.bounds)
            self.assertEqual(graph["drawn_bev_ids"], ["pedestrian:15", "pedestrian:2"])
            self.assertEqual(graph["outside_rendered_bev_ids"], ["pedestrian:9"])
            self.assertEqual(graph["outside_local_bev_ids"], ["pedestrian:9"])
            self.assertEqual(len(graph["actors"]), 4)
            self.assertEqual(graph["hhi_without_3d"], [["pedestrian:12", "pedestrian:2"]])
            self.assertEqual(graph["hhi_outside_rendered_bev"], [["pedestrian:2", "pedestrian:9"]])
            self.assertEqual(renderer.local, (-14, 18, -16, 16))

    def test_layout_validation(self):
        with self.assertRaises(ValueError):
            review.Renderer(Path("/tmp"), "fixture", 640, {}, layout="other")
        with self.assertRaises(ValueError):
            review.Renderer(Path("/tmp"), "fixture", 640, {}, layout="front-near-all")


if __name__ == "__main__":
    unittest.main()
