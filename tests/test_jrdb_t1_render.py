"""T1 geometry, label isolation, time causality and dynamic REL contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import ValidationError
import numpy as np
from PIL import Image

from decision.vlm import jrdb_t1_common as t1
from decision.vlm import visualize_jrdb_t1_gt as gt
from decision.vlm.visualize_jrdb_t1_gt import load_reviewed_label


def fixture(root: Path):
    sequence = "fixture_0"
    frames = ["000000", "000007", "000015"]
    people2, people3 = {}, {}
    for frame in frames:
        people2[f"{frame}.jpg"] = [
            {"label_id": f"pedestrian:{n}", "file_id": "999999.jpg",
             "box": [5 + n * 2, 5, 16, 40], "attributes": {"occlusion": "Fully_visible"},
             "social_group": {"cluster_ID": 4}, "action_label": {"talking": 1}}
            for n in (2, 9)]
        people3[f"{frame}.pcd"] = [
            {"label_id": f"pedestrian:{n}",
             "box": {"cx": 3 + n / 10, "cy": n / 10, "cz": 0,
                     "l": 0.8, "w": 0.5, "h": 1.8, "rot_z": 0}}
            for n in (2, 9, 12)]
        path = root / "images/image_0" / sequence / f"{frame}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (90, 60, 40)).save(path)
    two_path = t1.jrdb.label_path(root, "people", sequence, "0")
    three_path = root / "labels/labels_3d" / f"{sequence}.json"
    for path, data in ((two_path, people2), (three_path, people3)):
        path.parent.mkdir(parents=True, exist_ok=True)
        t1.write_json(path, {"labels": data})
    calibration = {"D": [0, 0, 0, 0, 0],
                   "distorted_img_K": [[70, 0, 40], [0, 70, 30], [0, 0, 1]],
                   "undistorted_img_K": [[70, 0, 40], [0, 70, 30], [0, 0, 1]],
                   "cam2ego": [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]}
    calibration_path = root / "calibration/lidars.yaml"
    calibration_path.parent.mkdir(parents=True)
    t1.write_json(calibration_path, {"sensor_0": calibration})
    return sequence, frames, two_path, three_path


class T1RenderingTests(unittest.TestCase):
    def test_semantic_whitelist_and_duplicate_identity(self):
        row = {"label_id": "p1", "box": [1, 2, 3, 4], "attributes": {"occlusion": "visible"},
               "social_group": {"cluster_ID": 888}, "action_label": {"queue": 1}, "file_id": "future"}
        result = t1.geometry_rows([row], 2)
        self.assertEqual(result, {"p1": {"box": [1., 2., 3., 4.], "occlusion": "visible", "truncated": None}})
        with self.assertRaises(ValueError):
            t1.geometry_rows([row, row], 2)

    def test_time_selection_real_frames_only_and_no_future(self):
        available = [f"{n:06d}" for n in range(31)]
        selected = t1.choose_frames(available, "000015", 15)
        self.assertEqual(selected[0], "000000")
        self.assertIn(selected[1], ("000007", "000008"))
        self.assertEqual(selected[-1], "000015")
        with self.assertRaises(ValueError):
            t1.choose_frames(available, "000005", 15)
        with self.assertRaises(ValueError):
            t1.frame_times(["000000", "000000", "000015"], 15)

    def test_bev_axes_and_undistortion_without_distortion(self):
        origin = t1.bev_xy(0, 0, 448)
        self.assertLess(t1.bev_xy(1, 0, 448)[1], origin[1])
        self.assertLess(t1.bev_xy(0, 1, 448)[0], origin[0])
        calibration = {"D": [0] * 5, "distorted_img_K": np.eye(3).tolist(),
                       "undistorted_img_K": np.eye(3).tolist()}
        np.testing.assert_allclose(t1.undistorted_box([10, 20, 30, 40], calibration, (100, 100)),
                                   [10, 20, 40, 60])

    def test_default_motion_missing_not_zero_and_model_has_no_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            sequence, frames, _, _ = fixture(root)
            window = t1.build_window(root, sequence, frames, lidar=False)
            self.assertEqual(len(window.valid_pid), 2)
            for track in window.tracks():
                self.assertIsNone(track["vel"])
                self.assertIsNone(track["cov"])
                self.assertEqual(track["history"], [])
            serialized = json.dumps(t1.model_payload(window))
            self.assertNotIn("pedestrian", serialized)
            self.assertNotIn("cluster_ID", serialized)
            all_bev = t1.build_window(root, sequence, frames, scope="all-bev", lidar=False)
            self.assertEqual(len(all_bev.valid_pid), 3)
            self.assertEqual(sorted(t["visible"] for t in all_bev.tracks()), [0, 1, 1])

    def test_oracle_pid_stability_and_no_label_leakage_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            sequence, frames, two_path, _ = fixture(root)
            before = t1.build_window(root, sequence, frames, lidar=False)
            image_before = np.asarray(t1.render_frame(before, 2))
            labels = t1.jrdb.load_json(two_path)
            for rows in labels["labels"].values():
                rows.reverse()
                for row in rows:
                    row["social_group"] = {"cluster_ID": 999}
                    row["action_label"] = {"HIDDEN_QUEUE": 1}
                    row["H-interaction"] = {"secret": True}
            t1.write_json(two_path, labels)
            after = t1.build_window(root, sequence, frames, lidar=False)
            self.assertEqual(before.pid_map, after.pid_map)
            self.assertEqual(before.geometry_signature, after.geometry_signature)
            np.testing.assert_array_equal(image_before, np.asarray(t1.render_frame(after, 2)))
            styles, links = t1.relation_layout({"g": [{"m": before.valid_pid, "r": "q", "k": 1}], "u": []})
            self.assertFalse(np.array_equal(image_before, np.asarray(t1.render_frame(before, 2, styles, links))))

    def test_ego_motion_compensated_velocity_not_robot_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            sequence, frames, _, three_path = fixture(root)
            data = t1.jrdb.load_json(three_path)
            poses = {}
            for i, frame in enumerate(frames):
                matrix = np.eye(4)
                matrix[0, 3] = i
                poses[frame] = matrix.tolist()
                for row in data["labels"][f"{frame}.pcd"]:
                    row["box"]["cx"] -= i
            t1.write_json(three_path, data)
            pose_path = Path(directory) / "poses.json"
            t1.write_json(pose_path, {"convention": "world_from_ego", "poses": poses})
            window = t1.build_window(root, sequence, frames, lidar=False, pose_path=pose_path)
            for track in window.tracks():
                np.testing.assert_allclose(track["vel"], [0, 0], atol=1e-8)
                self.assertEqual(len(track["history"]), 2)

    def test_overflow_and_missing_3d_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            sequence, frames, two_path, three_path = fixture(root)
            data3 = t1.jrdb.load_json(three_path)
            data3["labels"]["000015.pcd"] = []
            t1.write_json(three_path, data3)
            window = t1.build_window(root, sequence, frames, lidar=False)
            self.assertEqual(len(window.valid_pid), 2)
            self.assertTrue(all(t["pos"] is None for t in window.tracks()))
            data2 = t1.jrdb.load_json(two_path)
            exemplar = data2["labels"]["000015.jpg"][0]
            data2["labels"]["000015.jpg"] = []
            for i in range(21):
                row = copy.deepcopy(exemplar)
                row["label_id"] = f"person:{i}"
                data2["labels"]["000015.jpg"].append(row)
            t1.write_json(two_path, data2)
            with self.assertRaisesRegex(ValueError, "input_overflow"):
                t1.build_window(root, sequence, frames, lidar=False)

    def test_bundle_integrity_and_label_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            out = Path(directory) / "input"
            sequence, frames, _, _ = fixture(root)
            window = t1.build_window(root, sequence, frames, lidar=False)
            manifest = t1.save_input_bundle(window, out)
            rebuilt, _ = t1.load_bundle(out)
            self.assertEqual(rebuilt.geometry_signature, window.geometry_signature)
            template = out / "annotation_template.json"
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                load_reviewed_label(template, manifest)
            self.assertEqual(load_reviewed_label(template, manifest, True)["review_status"], "draft")
            reviewed = t1.jrdb.load_json(template)
            reviewed.update(review_status="reviewed", reviewer="fixture", reviewed_at="2026-09-30T00:00:00Z")
            t1.write_json(template, reviewed)
            load_reviewed_label(template, manifest)
            reviewed["input_fingerprint"] = "wrong"
            t1.write_json(template, reviewed)
            with self.assertRaisesRegex(ValueError, "exact model inputs"):
                load_reviewed_label(template, manifest)
            Image.new("RGB", (896, 448), (0, 0, 0)).save(out / "image_2.png")
            with self.assertRaisesRegex(ValueError, "Input image changed"):
                t1.load_bundle(out)

    def test_dynamic_rel_validation_and_semantic_order_preserved(self):
        good = {"g": [{"m": [9, 2], "r": "q", "k": 1}], "u": [7]}
        t1.validate_response(good, [2, 7, 9])
        self.assertEqual(good["g"][0]["m"], [9, 2])
        cases = [
            {"g": [{"m": [2, 9], "r": "w", "k": 0}], "u": [9]},
            {"g": [{"m": [9, 2], "r": "u", "k": 0}], "u": []},
            {"g": [{"m": [2, 8], "r": "p", "k": 1}], "u": []},
            {"g": [{"m": [2, 9], "r": "t", "k": 1}], "u": []},
            {"g": [{"m": [2, 9], "r": "t", "k": 0},
                   {"m": [7, 9], "r": "w", "k": 0}], "u": []},
        ]
        for response in cases:
            with self.subTest(response=response), self.assertRaises((ValueError, ValidationError)):
                t1.validate_response(response, [2, 7, 9])

    def test_reviewed_and_draft_cli_outputs_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            input_dir = Path(directory) / "input"
            sequence, frames, _, _ = fixture(root)
            manifest = t1.save_input_bundle(t1.build_window(root, sequence, frames, lidar=False), input_dir)
            label = t1.jrdb.load_json(input_dir / "annotation_template.json")
            members = list(reversed(manifest["model"]["VALID_PID"]))
            label["response"] = {"g": [{"m": members, "r": "p", "k": 1}], "u": []}
            label_path = Path(directory) / "label.json"
            t1.write_json(label_path, label)
            preview = Path(directory) / "preview"
            with patch("sys.argv", ["gt", "--input-dir", str(input_dir), "--labels", str(label_path),
                                    "--allow-draft", "--out", str(preview)]):
                gt.main()
            self.assertTrue((preview / "draft.png").exists())
            self.assertFalse((preview / "assistant_response.json").exists())
            label.update(review_status="reviewed", reviewer="synthetic_test_fixture",
                         reviewed_at="2026-09-30T00:00:00Z")
            t1.write_json(label_path, label)
            truth = Path(directory) / "truth"
            with patch("sys.argv", ["gt", "--input-dir", str(input_dir), "--labels", str(label_path),
                                    "--out", str(truth)]):
                gt.main()
            self.assertTrue((truth / "gt.png").exists())
            self.assertEqual(t1.jrdb.load_json(truth / "assistant_response.json"), label["response"])


if __name__ == "__main__":
    unittest.main()
