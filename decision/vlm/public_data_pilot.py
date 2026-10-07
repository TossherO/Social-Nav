#!/usr/bin/env python3
"""VLM data exploration: download/audit a pinned SocialNav-SUB subset.

Standard library only. Public, unauthenticated HTTPS; no pickle deserialization.
Refuses to overwrite an existing output directory. Failed downloads remain
partial and never receive a completed manifest.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from decision.vlm.project_config import dataset_path

REPO = "michaelmunje/SocialNav-SUB"
HOST = "https://huggingface.co"
SINGLE_PATH_FIELDS = (
    "human_answers_filepath", "human_raw_answers_filepath", "n_people_filepath",
    "human_tracking_filepath", "tracking_id_to_obj_label_filepath",
    "odometry_filepath",
)
LIST_PATH_FIELDS = ("raw_img_filepaths", "bev_img_filepaths")
METADATA = ("README.md", "dataset_cfg.yaml", "goal_info.json", "sample_info.json")


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    if any(ord(c) < 32 for c in value):
        raise ValueError("Control character in path")
    return value


def local_path(root, name):
    name = relative_path(name)
    root = Path(root).resolve()
    target = root.joinpath(*PurePosixPath(name).parts)
    node = root
    for part in PurePosixPath(name).parts:
        node = node / part
        if node.is_symlink():
            raise ValueError(f"Symlink is not permitted: {name}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes output: {name}")
    return target


def path_list(value):
    if not isinstance(value, str):
        raise ValueError("Image paths must be a string-encoded list")
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError) as error:
            # The published CSV also uses [dir/a.jpg, dir/b.jpg], with
            # unquoted paths. Accept only this narrow syntax, never eval it.
            stripped = value.strip()
            if not (stripped.startswith("[") and stripped.endswith("]")):
                raise ValueError("Image paths must be a bracketed list") from error
            parsed = [item.strip() for item in stripped[1:-1].split(",")]
            if not all(re.fullmatch(r"[A-Za-z0-9_./-]+", item) for item in parsed):
                raise ValueError("Invalid unquoted image path list") from error
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Image list must be nonempty")
    return [relative_path(item) for item in parsed]


def parse_index(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required = {"sample_name", *SINGLE_PATH_FIELDS, *LIST_PATH_FIELDS}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"CSV schema changed; required fields: {sorted(required)}")
    rows = list(reader)
    if not rows:
        raise ValueError("Dataset index is empty")
    names = set()
    for row in rows:
        name = row["sample_name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name or "") or name in names:
            raise ValueError(f"Invalid or duplicated sample_name: {name!r}")
        names.add(name)
        for field in SINGLE_PATH_FIELDS:
            relative_path(row[field])
        for field in LIST_PATH_FIELDS:
            row[field] = path_list(row[field])
    return rows


def row_paths(row):
    paths = [row[field] for field in SINGLE_PATH_FIELDS]
    for field in LIST_PATH_FIELDS:
        paths.extend(row[field])
    return paths


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def consume(self, count):
        if count < 0 or self.used + count > self.limit:
            raise ValueError("Download byte budget exceeded; output is partial")
        self.used += count


def receive(url, budget, destination=None, timeout=20):
    request = Request(url, headers={"User-Agent": "SocialNav-public-data-pilot/1"})
    digest = hashlib.sha256()
    count = 0
    chunks = []
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > budget.limit - budget.used:
            raise ValueError("Content-Length exceeds remaining download budget")
        handle = destination.open("xb") if destination is not None else None
        try:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                budget.consume(len(chunk))
                count += len(chunk)
                digest.update(chunk)
                if handle is None:
                    chunks.append(chunk)
                else:
                    handle.write(chunk)
        finally:
            if handle is not None:
                handle.close()
    if destination is None:
        return b"".join(chunks)
    return {"path": destination, "bytes": count, "sha256": digest.hexdigest()}


def fetch(args):
    if args.limit < 1 or args.max_mib < 1:
        raise ValueError("--limit and --max-mib must be positive")
    root = Path(args.output)
    if root.exists() or root.is_symlink():
        raise ValueError("Output already exists; choose a new directory (no overwrite)")
    budget = Budget(args.max_mib * 1024 * 1024)
    info_url = f"{HOST}/api/datasets/{REPO}/revision/{quote(args.revision, safe='')}"
    info = json.loads(receive(info_url, budget))
    revision = info.get("sha", "")
    if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
        raise ValueError("Could not resolve an immutable dataset commit")
    siblings = {relative_path(item["rfilename"]) for item in info["siblings"]}
    base = f"{HOST}/datasets/{REPO}/resolve/{revision}/"
    index_bytes = receive(base + "test.csv", budget)
    rows = parse_index(index_bytes)
    if args.limit > len(rows):
        raise ValueError(f"Requested {args.limit} scenes; index has {len(rows)}")
    selected = sorted(rows, key=lambda row: row["sample_name"])[:args.limit]
    files = {"test.csv"}
    files.update(name for name in METADATA if name in siblings)
    for row in selected:
        prefixes = tuple(f"{folder}/{row['sample_name']}/"
                         for folder in ("front_view", "prompts", "human_labels"))
        files.update(name for name in siblings if name.startswith(prefixes))
        files.update(row_paths(row))
    missing = files - siblings
    if missing:
        raise ValueError(f"Index references files absent from pinned tree: {sorted(missing)}")
    print(f"Resolved {revision}: {len(rows)} index scenes; downloading "
          f"{len(selected)} scene(s), {len(files)} files", file=sys.stderr, flush=True)
    # All network metadata is validated before creating the new output.
    root.mkdir(parents=True, exist_ok=False)
    index_path = local_path(root, "test.csv")
    with index_path.open("xb") as handle:
        handle.write(index_bytes)
    records = [{
        "path": "test.csv", "bytes": len(index_bytes),
        "sha256": hashlib.sha256(index_bytes).hexdigest(),
    }]
    for file_number, name in enumerate(sorted(files - {"test.csv"}), start=2):
        path = local_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = receive(base + quote(name, safe="/"), budget, path)
        result["path"] = name
        records.append(result)
        if file_number % 10 == 0 or file_number == len(files):
            print(f"Downloaded {file_number}/{len(files)} files", file=sys.stderr, flush=True)
    manifest = {
        "repo": REPO, "revision": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "exploratory_smoke_only_not_model_evaluation",
        "original_split": "test", "source_family": "SCAND",
        "root_sequence_mapping_verified": False,
        "index_scene_count": len(rows), "selected_rows": selected,
        "files": records, "network_bytes": budget.used,
        "warning": "Do not train on these scenes. Root-sequence leakage audit remains required.",
    }
    with (root / "pilot_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(audit(root), ensure_ascii=False, indent=2))


def audit(root):
    root = Path(root)
    manifest_path = local_path(root, "pilot_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("repo") != REPO
            or not isinstance(manifest.get("files"), list)
            or not isinstance(manifest.get("selected_rows"), list)):
        raise ValueError("Unsupported manifest")
    records = manifest["files"]
    if not records or not manifest["selected_rows"]:
        raise ValueError("Empty file or scene list")
    names = [relative_path(record["path"]) for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicated manifest paths")
    for record in records:
        path = local_path(root, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing file or size mismatch: {record['path']}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Hash mismatch: {record['path']}")
        # Validate JSON syntax, but never execute/deserialize pickle objects.
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    if "test.csv" not in names:
        raise ValueError("Manifest omits index")
    index = {row["sample_name"]: row
             for row in parse_index(local_path(root, "test.csv").read_bytes())}
    seen = set()
    for row in manifest["selected_rows"]:
        name = row["sample_name"]
        if name in seen or index.get(name) != row:
            raise ValueError("Selected row differs from index or is duplicated")
        seen.add(name)
        if not set(row_paths(row)).issubset(names):
            raise ValueError(f"Missing assets for {name}")
    return {
        "status": "asset_integrity_passed",
        "revision": manifest["revision"], "scenes": len(seen), "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "model_evaluation": "NOT_RUN",
        "root_sequence_leakage_audit": "PENDING",
        "note": "Byte hashes/JSON/index only; image decoding, labels and causality need review.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    downloader = commands.add_parser("fetch")
    downloader.add_argument("--output", required=True)
    downloader.add_argument("--limit", type=int, default=1)
    downloader.add_argument("--max-mib", type=int, default=64)
    downloader.add_argument("--revision", default="main")
    checker = commands.add_parser("audit")
    checker.add_argument("--input", default=dataset_path("socialnav_sub_pilot"),
                         help="Default: repository SocialNav-SUB dataset alias")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch(args)
        else:
            print(json.dumps(audit(args.input), ensure_ascii=False, indent=2))
    except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("No model results were produced. Failed output may be partial; "
              "use a new directory for retry. Check network/access and dataset schema.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
