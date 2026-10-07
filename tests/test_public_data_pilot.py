"""Offline fixtures only: these tests make no dataset/model quality claims."""

import csv
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from decision.vlm import public_data_pilot as pilot


def fixture():
    row = {"sample_name": "fixture_scene"}
    for index, field in enumerate(pilot.SINGLE_PATH_FIELDS):
        suffix = "pkl" if field == "odometry_filepath" else "json"
        row[field] = f"front_view/fixture_scene/field_{index}.{suffix}"
    for field in pilot.LIST_PATH_FIELDS:
        row[field] = repr([f"front_view/fixture_scene/{field}_{n}.png" for n in range(3)])
    # Match the real CSV's unquoted list syntax for one field.
    row["raw_img_filepaths"] = "[" + ", ".join(
        f"front_view/fixture_scene/raw_{n}.jpg" for n in range(3)
    ) + "]"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    index_bytes = stream.getvalue().encode()
    parsed = pilot.parse_index(index_bytes)[0]
    assets = {
        name: b'{"fixture":true}' if name.endswith(".json") else b"OFFLINE_FIXTURE"
        for name in pilot.row_paths(parsed)
    }
    assets["test.csv"] = index_bytes
    assets["README.md"] = b"Offline test fixture, not public data."
    info = {
        "sha": "a" * 40,
        "siblings": [{"rfilename": name} for name in assets],
    }
    return parsed, assets, info


class Response(io.BytesIO):
    def __init__(self, data, content_length=None):
        super().__init__(data)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


class PublicDataTests(unittest.TestCase):
    def test_fetcher_does_not_emit_project_release_number(self):
        self.assertNotIn('"manifest_version"', Path(pilot.__file__).read_text(encoding='utf-8'))

    def test_path_list_accepts_json_and_python_literals(self):
        self.assertEqual(pilot.path_list('["a/b.png"]'), ["a/b.png"])
        self.assertEqual(pilot.path_list("['a/b.png']"), ["a/b.png"])

    def test_path_list_accepts_published_csv_unquoted_format(self):
        self.assertEqual(
            pilot.path_list("[front_view/101_Spot_1_155/155.jpg, front_view/101_Spot_1_155/156.jpg]"),
            ["front_view/101_Spot_1_155/155.jpg", "front_view/101_Spot_1_155/156.jpg"],
        )

    def test_unquoted_list_rejects_malformed_and_unsafe_paths(self):
        for value in ("[a/b.jpg,]", "[../x.jpg]", "[/tmp/a.jpg]", "[a/b.jpg;evil]", "[foo()]", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.path_list(value)

    def test_path_list_rejects_code_and_wrong_types(self):
        for value in ("__import__('os').getcwd()", "{}", "[]", "[1]", "['../x']"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.path_list(value)

    def test_unsafe_relative_paths(self):
        for value in ("../a", "/tmp/a", "a/../b", "a//b", "a/./b", "a\\b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.relative_path(value)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "link").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                pilot.local_path(root, "link/file")

    def test_csv_contract(self):
        row, assets, _ = fixture()
        self.assertEqual(pilot.parse_index(assets["test.csv"])[0], row)
        with self.assertRaises(ValueError):
            pilot.parse_index(b"sample_name\nfoo\n")
        header = assets["test.csv"].splitlines(keepends=True)[0]
        with self.assertRaises(ValueError):
            pilot.parse_index(header)

    def test_budget(self):
        budget = pilot.Budget(3)
        budget.consume(3)
        with self.assertRaises(ValueError):
            budget.consume(1)
        self.assertEqual(budget.used, 3)

    def test_receive_enforces_streaming_budget(self):
        with patch.object(pilot, "urlopen", return_value=Response(b"12345")):
            with self.assertRaises(ValueError):
                pilot.receive("https://example.invalid/test", pilot.Budget(4))

    def test_receive_enforces_declared_budget(self):
        with patch.object(pilot, "urlopen", return_value=Response(b"1", content_length=100)):
            with self.assertRaises(ValueError):
                pilot.receive("https://example.invalid/test", pilot.Budget(4))

    def run_fetch(self, root):
        _, assets, info = fixture()

        def respond(request, timeout):
            if "/api/datasets/" in request.full_url:
                return Response(json.dumps(info).encode())
            name = request.full_url.split("/resolve/" + info["sha"] + "/", 1)[1]
            return Response(assets[name])

        args = Namespace(output=str(root), limit=1, max_mib=1, revision="main")
        with (patch.object(pilot, "urlopen", side_effect=respond),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            pilot.fetch(args)
        return args

    def test_fetch_and_audit_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            self.run_fetch(root)
            result = pilot.audit(root)
            self.assertEqual(result["scenes"], 1)
            self.assertEqual(result["model_evaluation"], "NOT_RUN")
            manifest = json.loads((root / "pilot_manifest.json").read_text())
            self.assertFalse(manifest["root_sequence_mapping_verified"])

    def test_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            self.run_fetch(root)
            with (root / "README.md").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(ValueError):
                pilot.audit(root)

    def test_same_size_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            self.run_fetch(root)
            with (root / "README.md").open("r+b") as handle:
                handle.write(b"X")
            with self.assertRaises(ValueError):
                pilot.audit(root)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(output=tmp, limit=1, max_mib=1, revision="main")
            with patch.object(pilot, "urlopen") as network, self.assertRaises(ValueError):
                pilot.fetch(args)
            network.assert_not_called()

    def test_manifest_row_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            self.run_fetch(root)
            path = root / "pilot_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["selected_rows"][0]["raw_img_filepaths"] = ["not/in/index.png"]
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                pilot.audit(root)


if __name__ == "__main__":
    unittest.main()
