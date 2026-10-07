"""Small offline fixtures for JRDB alignment and binary decoders."""

import unittest
from fractions import Fraction
from pathlib import Path
import tempfile
import json

import numpy as np
from PIL import Image

from decision.vlm import jrdb_visualize as viz


class JrdbVisualizationTests(unittest.TestCase):
    def test_label_paths(self):
        root = Path("/data/JRDB")
        self.assertEqual(
            viz.label_path(root, "people", "demo_0", "stitched"),
            root / "labels/labels_2d_activity_social_stitched/demo_0.json",
        )
        self.assertEqual(
            viz.label_path(root, "panoptic", "demo_0", "2"),
            root / "labels/labels_2d_panoptic_CW/demo_0_camera_2.json",
        )
        self.assertIsNone(viz.label_path(root, "reasoning", "demo_0", "2"))
        self.assertEqual(viz.label_path(root, "view3d", "demo_0", "stitched"),
                         root / "labels/labels_3d/demo_0.json")

    def test_frame_key_vs_inner_file_id(self):
        data = {"labels": {
            "000000.jpg": [{"file_id": "000002.jpg"}, {"file_id": "000002.jpg"}],
            "000001.jpg": [{"file_id": "000003.jpg"}],
        }}
        issues = viz.frame_label_issues(data, {"000000", "000001"})
        self.assertEqual(issues["frame_key_difference_from_images"], 0)
        self.assertEqual(issues["file_id_minus_frame_key"], {2: 3})

    def test_coco_mismatches(self):
        data = {"images": [{"id": 1, "file_name": "old/export/demo/000000.jpg", "width": 752, "height": 480}],
                "annotations": [{"image_id": 1, "category_id": 1}],
                "categories": [{"id": 2}]}
        issues = viz.coco_issues(data, (3760, 480))
        self.assertEqual(issues["category_ids_not_in_categories"], [1])
        self.assertEqual(issues["metadata_size_mismatches"], 1)
        self.assertEqual(issues["noncanonical_image_paths"], 1)

    def test_lzf_literal_and_back_reference(self):
        # Literal "abc" followed by a three-byte back-reference to it.
        self.assertEqual(viz.lzf_decompress(b"\x02abc\x20\x02", 6), b"abcabc")
        with self.assertRaises(ValueError):
            viz.lzf_decompress(b"\x02abc", 4)

    def test_uncompressed_coco_rle(self):
        mask = viz.decode_coco_mask({"size": [2, 2], "counts": [1, 2, 1]})
        np.testing.assert_array_equal(mask, [[False, True], [True, False]])
        with self.assertRaises(ValueError):
            viz.decode_coco_mask({"size": [2, 2], "counts": [1, 1]})

    def test_compressed_coco_rle(self):
        # COCO compressed string for [1, 2, 1], including differential third run.
        self.assertEqual(viz.decode_coco_counts("121"), [1, 2, 1])

    def test_video_rate_uses_source_frame_spacing(self):
        self.assertEqual(viz.video_rate(["000000", "000001", "000002"], 1, False), Fraction(15, 1))
        self.assertEqual(viz.video_rate(["000000", "000015", "000030"], 1, True), Fraction(1, 1))
        self.assertEqual(viz.video_rate(["000000"], 15, False), Fraction(1, 1))

    def test_h264_video_writer(self):
        import av

        visual = Image.new("RGB", (500, 150), (10, 20, 30))
        frames = [viz.video_canvas(visual, 320, 240, "fixture_0", f"{n:06d}", "people", False)
                  for n in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.mp4"
            self.assertEqual(viz.encode_mp4(path, frames, 320, 240, Fraction(1, 1)), 2)
            with av.open(str(path)) as container:
                self.assertEqual(len(list(container.decode(video=0))), 2)

    def test_static_visualization_excludes_audit_panel(self):
        scene = Image.new("RGB", (100, 60), (190, 20, 30))
        visual = viz.with_side_panel(scene, "AUDIT", ["changing metadata"])
        self.assertEqual(visual.size, scene.size)
        self.assertEqual(visual.getpixel((99, 59)), (190, 20, 30))

    def test_montage_handles_missing_label_frame(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = Image.new("RGB", (320, 240), (80, 120, 160))
            viz.encode_mp4(root / "people.mp4", [frame, frame], 320, 240, Fraction(1, 1))
            viz.encode_mp4(root / "head.mp4", [frame], 320, 240, Fraction(1, 1))
            (root / "manifest.json").write_text(json.dumps({
                "sequence": "fixture_0", "view": "stitched", "stride": 15,
                "layers": {
                    "people": {"status": "rendered", "source_frame_ids": ["000000", "000015"]},
                    "head": {"status": "rendered", "source_frame_ids": ["000000"]},
                },
            }))
            output = root / "overview.mp4"
            result = viz.render_video_montage(root, output, 320, 240)
            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["source_frame_ids"], ["000000", "000015"])
            with av.open(str(output)) as container:
                self.assertEqual(len(list(container.decode(video=0))), 2)

    def test_montage_crop_removes_letterboxing(self):
        image = Image.new("RGB", (320, 240), (19, 23, 29))
        image.paste((240, 240, 240), (40, 60, 280, 160))
        cropped = viz.crop_video_content(image)
        self.assertLess(cropped.width, image.width)
        self.assertLess(cropped.height, image.height)
        self.assertGreaterEqual(cropped.width, 240)
        self.assertGreaterEqual(cropped.height, 100)

    def test_montage_original_fps_with_sparse_layer(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = Image.new("RGB", (320, 240), (80, 120, 160))
            viz.encode_mp4(root / "people.mp4", [frame] * 3, 320, 240, Fraction(15, 1))
            viz.encode_mp4(root / "panoptic.mp4", [frame], 320, 240, Fraction(1, 1))
            (root / "manifest.json").write_text(json.dumps({
                "sequence": "fixture_0", "view": "stitched", "stride": 1,
                "layers": {
                    "people": {"status": "rendered", "source_frame_ids": ["000000", "000001", "000002"]},
                    "panoptic": {"status": "rendered", "source_frame_ids": ["000000"]},
                },
            }))
            output = root / "overview.mp4"
            result = viz.render_video_montage(root, output, 320, 240)
            self.assertEqual(result["frames"], 3)
            self.assertEqual(result["fps"], 15)
            with av.open(str(output)) as container:
                self.assertEqual(container.streams.video[0].average_rate, Fraction(15, 1))
                self.assertEqual(len(list(container.decode(video=0))), 3)

    def test_virtual_camera_and_box_geometry(self):
        projected, depth = viz.project_virtual_camera(
            np.asarray([[0, 0, 0], [0, 0, 2]], dtype=np.float32), 1600, 900)
        self.assertTrue(np.all(depth > 0))
        np.testing.assert_allclose(projected[0], [800, 450], atol=0.01)
        self.assertLess(projected[1, 1], projected[0, 1])
        corners = viz.box_corners_3d({"cx": 0, "cy": 0, "cz": 1,
                                      "l": 2, "w": 1, "h": 2, "rot_z": 0})
        self.assertEqual(corners.shape, (8, 3))
        np.testing.assert_array_equal(corners[:4, 2], np.zeros(4))
        np.testing.assert_array_equal(corners[4:, 2], np.full(4, 2))


if __name__ == "__main__":
    unittest.main()
