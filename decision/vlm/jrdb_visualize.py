#!/usr/bin/env python3
"""Audit and render the locally downloaded JRDB 2022 train subset.

Reads JRDB files only. Rendered PNGs, MP4s, and reports must be written outside the
dataset root. Pseudo-3D labels are deliberately shown as raw packed codes:
their bit semantics and upper/lower point ordering are not assumed verified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from decision.vlm.project_config import dataset_path
except ModuleNotFoundError:
    from project_config import dataset_path

DEFAULT_ROOT = dataset_path("jrdb")

VIEWS = ("stitched", "0", "2", "4", "6", "8")
LAYERS = ("people", "actions", "social", "head", "pose", "panoptic", "reasoning", "view3d", "pseudo3d")
OPTIONAL_LAYERS = LAYERS + ("bev3d", "pseudo3d_bev")
FRAME_RE = re.compile(r"^[0-9]{6}$")
SEQ_RE = re.compile(r"^[A-Za-z0-9_-]+$")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def validate_id(sequence: str, frame: str | None = None) -> None:
    if not SEQ_RE.fullmatch(sequence):
        raise ValueError(f"Invalid sequence name: {sequence!r}")
    if frame is not None and not FRAME_RE.fullmatch(frame):
        raise ValueError(f"Frame must have six digits: {frame!r}")


def view_name(view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"View must be one of {VIEWS}")
    return "image_stitched" if view == "stitched" else f"image_{view}"


def label_path(root: Path, family: str, sequence: str, view: str) -> Path | None:
    if family == "people":
        folder = "labels_2d_activity_social_stitched" if view == "stitched" else "labels_2d_activity_social"
        name = f"{sequence}.json" if view == "stitched" else f"{sequence}_image{view}.json"
    elif family in ("head", "pose", "panoptic"):
        folders = {
            "head": ("labels_2d_head_stitched", "labels_2d_head"),
            "pose": ("labels_2d_pose_stitched_coco", "labels_2d_pose_coco"),
            "panoptic": ("labels_2d_panoptic_stitched_CW", "labels_2d_panoptic_CW"),
        }
        folder = folders[family][0 if view == "stitched" else 1]
        name = f"{sequence}.json" if view == "stitched" else f"{sequence}_{'camera_' if family == 'panoptic' else 'image'}{view}.json"
    elif family == "reasoning":
        return root / "labels_2d_stitched_reasoning" / f"{sequence}.json" if view == "stitched" else None
    elif family in ("bev3d", "view3d"):
        return root / "labels" / "labels_3d" / f"{sequence}.json"
    else:
        raise ValueError(f"Unknown family: {family}")
    return root / "labels" / folder / name


def frame_rows(path: Path, frame: str, extension: str) -> list[dict]:
    rows = load_json(path).get("labels", {}).get(f"{frame}.{extension}", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected list at {frame}.{extension}: {path}")
    return rows


def coco_frame(path: Path, frame: str) -> tuple[dict, list[dict], dict[int, dict]]:
    data = load_json(path)
    matches = [row for row in data.get("images", []) if Path(row.get("file_name", "")).name == f"{frame}.jpg"]
    if len(matches) != 1:
        raise ValueError(f"Expected one COCO image for {frame} in {path}, found {len(matches)}")
    image = matches[0]
    rows = [row for row in data.get("annotations", []) if row.get("image_id") == image["id"]]
    categories = {row["id"]: row for row in data.get("categories", [])}
    return image, rows, categories


def frame_file_set(path: Path, suffix: str) -> set[str]:
    return {file.stem for file in path.glob(f"*.{suffix}")}


def frame_label_issues(data: dict, image_frames: set[str]) -> dict:
    labels = data.get("labels", {})
    offsets = Counter()
    malformed = 0
    row_count = 0
    for key, rows in labels.items():
        for row in rows:
            row_count += 1
            file_id = row.get("file_id")
            key_stem = Path(key).stem
            file_stem = Path(str(file_id)).stem
            if key_stem.isdigit() and file_stem.isdigit():
                offsets[int(file_stem) - int(key_stem)] += 1
            else:
                malformed += 1
    return {
        "frame_key_difference_from_images": len({Path(key).stem for key in labels} ^ image_frames),
        "rows": row_count,
        "file_id_minus_frame_key": dict(sorted(offsets.items())),
        "missing_or_unparseable_file_id_rows": malformed,
    }


def coco_frame_set(data: dict) -> set[str]:
    return {Path(row.get("file_name", "")).stem for row in data.get("images", [])}


def coco_issues(data: dict, actual_size: tuple[int, int]) -> dict:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    image_ids = [row.get("id") for row in images]
    category_ids = {row.get("id") for row in categories}
    annotation_ids = {row.get("image_id") for row in annotations}
    annotation_categories = {row.get("category_id") for row in annotations}
    paths = [row.get("file_name", "") for row in images]
    return {
        "image_entries": len(images),
        "annotation_entries": len(annotations),
        "duplicate_image_ids": len(image_ids) - len(set(image_ids)),
        "duplicate_image_frame_names": len(paths) - len(set(Path(name).name for name in paths)),
        "dangling_annotation_image_ids": len(annotation_ids - set(image_ids)),
        "category_ids_not_in_categories": sorted(x for x in annotation_categories - category_ids if x is not None),
        "metadata_size_mismatches": sum((row.get("width"), row.get("height")) != actual_size for row in images),
        "noncanonical_image_paths": sum(len(Path(name).parts) != 3 for name in paths),
    }


def audit_sequence(root: Path, sequence: str) -> dict:
    validate_id(sequence)
    views = {view: frame_file_set(root / "images" / view_name(view) / sequence, "jpg") for view in VIEWS}
    base = views["stitched"]
    if not base:
        raise FileNotFoundError(f"No stitched frames for {sequence}")
    report: dict = {
        "sequence": sequence,
        "frame_count": len(base),
        "image_view_frame_differences": {v: len(frames ^ base) for v, frames in views.items()},
        "lidar_frame_differences": {},
        "frame_key_differences": {},
        "file_id_alignment": {},
        "person_id_alignment": {},
        "coco": {},
        "pseudo3d": {},
    }
    for lidar in ("lower_velodyne", "upper_velodyne"):
        frames = frame_file_set(root / "pointclouds" / lidar / sequence, "pcd")
        report["lidar_frame_differences"][lidar] = len(frames ^ base)
    frame_labels = {}
    for view in VIEWS:
        key = f"people_{view}"
        frame_labels[key] = load_json(label_path(root, "people", sequence, view))
    for family in ("bev3d", "reasoning"):
        frame_labels[family] = load_json(label_path(root, family, sequence, "stitched"))
    for name, data in frame_labels.items():
        issues = frame_label_issues(data, base)
        report["frame_key_differences"][name] = issues.pop("frame_key_difference_from_images")
        report["file_id_alignment"][name] = issues
    people = frame_labels["people_stitched"]["labels"]
    for other_name in ("reasoning", "bev3d"):
        other = frame_labels[other_name]["labels"]
        differing = []
        only_people = 0
        only_other = 0
        for frame in sorted(base):
            left = {row.get("label_id") for row in people.get(f"{frame}.jpg", [])}
            right_suffix = "pcd" if other_name == "bev3d" else "jpg"
            right = {row.get("label_id") for row in other.get(f"{frame}.{right_suffix}", [])}
            if left != right:
                differing.append(frame)
                only_people += len(left - right)
                only_other += len(right - left)
        report["person_id_alignment"][f"people_vs_{other_name}"] = {
            "differing_frames": len(differing), "only_people_instances": only_people,
            "only_other_instances": only_other, "examples": differing[:10],
        }
    for view in VIEWS:
        image_path = root / "images" / view_name(view) / sequence / f"{min(base)}.jpg"
        with Image.open(image_path) as image:
            actual_size = image.size
        for family in ("head", "pose", "panoptic"):
            data = load_json(label_path(root, family, sequence, view))
            key = f"{family}_{view}"
            info = coco_issues(data, actual_size)
            coco_frames = coco_frame_set(data)
            missing = sorted(base - coco_frames)
            info["frame_ids_not_in_images"] = len(coco_frames - base)
            info["image_frames_missing_vs_sensor"] = len(missing)
            info["missing_frame_examples"] = missing[:10]
            report["coco"][key] = info
    bins = root / "labels" / "labels_3d_pseudo_panoptic_CW" / sequence
    bin_frames = frame_file_set(bins, "bin")
    pano_frames = coco_frame_set(load_json(label_path(root, "panoptic", sequence, "stitched")))
    size_errors = []
    for frame in sorted(bin_frames):
        lower = pcd_point_count(root / "pointclouds" / "lower_velodyne" / sequence / f"{frame}.pcd")
        upper = pcd_point_count(root / "pointclouds" / "upper_velodyne" / sequence / f"{frame}.pcd")
        actual = (bins / f"{frame}.bin").stat().st_size
        if actual != 4 * (lower + upper):
            size_errors.append(frame)
    report["pseudo3d"] = {
        "files": len(bin_frames),
        "frame_difference_from_panoptic_stitched": len(bin_frames ^ pano_frames),
        "size_mismatch_frames": size_errors[:20],
        "size_mismatch_count": len(size_errors),
    }
    return report


def pcd_header(stream) -> dict:
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD header ended before DATA")
        text = line.decode("ascii").strip()
        if not text or text.startswith("#"):
            continue
        key, _, rest = text.partition(" ")
        header[key] = rest.strip()
        if key == "DATA":
            return header


def pcd_point_count(path: Path) -> int:
    with path.open("rb") as stream:
        return int(pcd_header(stream)["POINTS"])


def lzf_decompress(data: bytes, expected: int) -> bytes:
    output = bytearray()
    index = 0
    while index < len(data):
        control = data[index]
        index += 1
        if control < 32:
            count = control + 1
            output.extend(data[index:index + count])
            index += count
        else:
            count = control >> 5
            offset = (control & 31) << 8
            if count == 7:
                count += data[index]
                index += 1
            offset += data[index]
            index += 1
            source = len(output) - offset - 1
            if source < 0:
                raise ValueError("Invalid LZF back-reference")
            for _ in range(count + 2):
                output.append(output[source])
                source += 1
        if len(output) > expected:
            raise ValueError("PCD decompression exceeded declared size")
    if len(output) != expected:
        raise ValueError(f"PCD decompression size {len(output)} != {expected}")
    return bytes(output)


def read_pcd_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = pcd_header(stream)
        data_mode = header["DATA"]
        if data_mode != "binary_compressed":
            raise ValueError(f"Unsupported PCD mode {data_mode!r}: {path}")
        compressed_size, raw_size = struct.unpack("<II", stream.read(8))
        packed = stream.read(compressed_size)
        if len(packed) != compressed_size:
            raise ValueError(f"Truncated PCD: {path}")
    raw = lzf_decompress(packed, raw_size)
    fields = header["FIELDS"].split()
    sizes = [int(x) for x in header["SIZE"].split()]
    types = header["TYPE"].split()
    counts = [int(x) for x in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
    points = int(header["POINTS"])
    position = 0
    columns = {}
    for name, size, kind, count in zip(fields, sizes, types, counts):
        dtype = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 2): "<u2", ("U", 4): "<u4", ("I", 4): "<i4"}.get((kind, size))
        if dtype is None:
            raise ValueError(f"Unsupported PCD field {name}: {kind}{size}")
        length = points * count * size
        if name in ("x", "y", "z"):
            columns[name] = np.frombuffer(raw, dtype=dtype, count=points * count, offset=position).reshape(points, count)[:, 0]
        position += length
    if position != raw_size or not all(axis in columns for axis in ("x", "y", "z")):
        raise ValueError(f"PCD fields/size disagree: {path}")
    return np.column_stack([columns[axis] for axis in ("x", "y", "z")]).astype(np.float32)


def decode_coco_counts(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    counts = []
    index = 0
    while index < len(value):
        x = 0
        shift = 0
        while True:
            code = ord(value[index]) - 48
            index += 1
            x |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    x |= -(1 << shift)
                break
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_coco_mask(segmentation: dict) -> np.ndarray:
    height, width = (int(x) for x in segmentation["size"])
    counts = decode_coco_counts(segmentation["counts"])
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    for index, count in enumerate(counts):
        if count < 0 or position + count > flat.size:
            raise ValueError("Invalid COCO RLE run lengths")
        if index % 2:
            flat[position:position + count] = 1
        position += count
    if position != flat.size:
        raise ValueError(f"COCO RLE covers {position} instead of {flat.size} pixels")
    return flat.reshape((height, width), order="F").astype(bool)


def stable_color(value: object) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return tuple(60 + int(x) % 180 for x in digest[:3])


def font(size: int = 18):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box, color, text: str = "", width: int = 3) -> None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    x, y, w, h = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        return
    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
    if text:
        text = text[:90]
        label_font = font(16)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        text_y = max(0, y - (bottom - top) - 6)
        draw.rectangle((x, text_y, x + right - left + 8, text_y + bottom - top + 5), fill=(0, 0, 0))
        draw.text((x + 4, text_y + 2), text, fill=color, font=label_font)


def with_side_panel(image: Image.Image, title: str, lines: list[str]) -> Image.Image:
    """Keep the annotated scene, not the old verbose per-frame metadata sidebar."""
    return image.convert("RGB")


def center_of(box) -> tuple[float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x, y, w, h = (float(value) for value in box)
    return x + w / 2, y + h / 2


def draw_people_link(draw: ImageDraw.ImageDraw, first: tuple[float, float],
                     second: tuple[float, float], image_width: int) -> None:
    if image_width > 2000 and abs(first[0] - second[0]) > image_width / 2:
        shift = image_width if first[0] > second[0] else -image_width
        draw.line((first, (second[0] + shift, second[1])), fill=(255, 210, 55), width=3)
        draw.line(((first[0] - shift, first[1]), second), fill=(255, 210, 55), width=3)
    else:
        draw.line((first, second), fill=(255, 210, 55), width=3)


def render_people_layer(base: Image.Image, rows: list[dict], mode: str) -> Image.Image:
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    centers = {row.get("label_id"): center_of(row.get("box")) for row in rows}
    lines = [f"people={len(rows)}  mode={mode}"]
    if mode == "social":
        drawn_pairs = set()
        for row in rows:
            source = row.get("label_id")
            for relation in row.get("H-interaction", []) or []:
                target = relation.get("pair")
                pair = tuple(sorted((str(source), str(target))))
                if pair in drawn_pairs or centers.get(source) is None or centers.get(target) is None:
                    continue
                drawn_pairs.add(pair)
                draw_people_link(draw, centers[source], centers[target], base.width)
    for row in rows:
        person = str(row.get("label_id", "?"))
        group = (row.get("social_group") or {}).get("cluster_ID", "?")
        color = stable_color(f"group-{group}" if mode == "social" else person)
        if mode == "people":
            label = person
            lines.append(f"{person} box={row.get('box')} no_eval={(row.get('attributes') or {}).get('no_eval')}")
        elif mode == "actions":
            actions = ",".join((row.get("action_label") or {}).keys())
            label = f"{person} {actions[:45]}"
            lines.append(f"{person}: action={row.get('action_label', {})}")
            lines.append(f"  social_activity={row.get('social_activity', {})}")
        else:
            label = f"{person} G{group}"
            interactions = row.get("H-interaction") or []
            peers = [(item.get("pair"), item.get("inter_labels")) for item in interactions]
            lines.append(f"{person} group={row.get('social_group', {})} H-interaction={peers}")
            lines.append(f"  group_info={row.get('group_info', [])} demographics={row.get('demographics_info', [])}")
        draw_box(draw, row.get("box"), color, label)
    return with_side_panel(canvas, mode.upper(), lines)


def render_head_layer(base: Image.Image, rows: list[dict], categories: dict[int, dict]) -> Image.Image:
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    counts = Counter(row.get("category_id") for row in rows)
    for row in rows:
        category = row.get("category_id")
        color = (255, 220, 50) if category == 1 else (235, 70, 235)
        draw_box(draw, row.get("bbox"), color, f"H{category} T{row.get('track_id', '?')}", width=2)
    lines = [f"annotations={len(rows)}", f"category counts={dict(counts)}", f"categories declared={categories}",
             "Both category 1 and 2 are shown; inspect duplicates before training."]
    return with_side_panel(canvas, "HEAD BOXES", lines)


def render_pose_layer(base: Image.Image, rows: list[dict], categories: dict[int, dict]) -> Image.Image:
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    lines = [f"pose instances={len(rows)}", "v: 0 invisible, 1 occluded, 2 visible"]
    skeleton = next((row.get("skeleton", []) for row in categories.values() if row.get("keypoints")), [])
    for row in rows:
        values = row.get("keypoints") or []
        if len(values) % 3 or not values:
            continue
        joints = [tuple(values[index:index + 3]) for index in range(0, len(values), 3)]
        color = stable_color(row.get("track_id", row.get("id")))
        for first, second in skeleton:
            if first < len(joints) and second < len(joints) and joints[first][2] > 0 and joints[second][2] > 0:
                draw.line((joints[first][:2], joints[second][:2]), fill=color, width=3)
        for x, y, visibility in joints:
            if visibility > 0:
                radius = 5 if visibility == 2 else 3
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        visible = [joint for joint in joints if joint[2] > 0]
        if visible:
            draw.text((visible[0][0], visible[0][1]), str(row.get("track_id", "?")), fill=color, font=font(16))
    return with_side_panel(canvas, "POSE", lines)


def render_panoptic_layer(base: Image.Image, rows: list[dict], categories: dict[int, dict]) -> Image.Image:
    pixels = np.asarray(base.convert("RGB")).copy()
    class_counts = Counter()
    things = []
    for row in rows:
        category = row.get("category_id")
        class_name = categories.get(category, {}).get("name", f"id{category}")
        mask = decode_coco_mask(row["segmentation"])
        if mask.shape != pixels.shape[:2]:
            raise ValueError(f"Mask {mask.shape} does not match image {pixels.shape[:2]}")
        area = int(mask.sum())
        if row.get("area") is not None and abs(area - float(row["area"])) > 1:
            raise ValueError(f"RLE area mismatch for annotation {row.get('id')}: {area} vs {row['area']}")
        color = np.asarray(stable_color(f"{category}:{row.get('tracking_id', row.get('id'))}"), dtype=np.float32)
        pixels[mask] = (0.50 * pixels[mask] + 0.50 * color).astype(np.uint8)
        class_counts[class_name] += 1
        if categories.get(category, {}).get("isthing") and area >= 100:
            ys, xs = np.nonzero(mask)
            things.append((area, float(xs.mean()), float(ys.mean()),
                           f"{class_name}:{row.get('tracking_id', row.get('id'))}", tuple(int(x) for x in color)))
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    for _, x, y, label, color in sorted(things, reverse=True)[:30]:
        draw.text((x, y), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0), font=font(13))
    lines = [f"masks={len(rows)}", "Only this annotated frame is shown; no temporal propagation."]
    lines.extend(f"thing {name}: {area} px" for area, _, _, name, _ in sorted(things, reverse=True)[:15])
    lines.extend(f"{name}: {count}" for name, count in class_counts.most_common(38))
    return with_side_panel(canvas, "PANOPTIC (COCO RLE)", lines)


def render_reasoning_layer(base: Image.Image, rows: list[dict]) -> Image.Image:
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    centers = {row.get("label_id"): center_of(row.get("box")) for row in rows}
    lines = [f"reasoning people={len(rows)}", "HHI edges are spatial; HOI pair IDs are shown as raw IDs."]
    edges = set()
    for row in rows:
        source = row.get("label_id")
        for relation in row.get("HHI", []) or []:
            target = relation.get("pair")
            edge = tuple(sorted((str(source), str(target))))
            if edge not in edges and centers.get(source) and centers.get(target):
                edges.add(edge)
                draw_people_link(draw, centers[source], centers[target], base.width)
    for row in rows:
        person = str(row.get("label_id", "?"))
        draw_box(draw, row.get("box"), stable_color(person), person)
        hhi = row.get("HHI") or []
        hoi = row.get("HOI") or []
        lines.append(f"{person}: HHI={[(x.get('pair'), list((x.get('inter_labels') or {}).keys())) for x in hhi][:3]}")
        lines.append(f"  HOI={[(x.get('pair'), list((x.get('inter_labels') or {}).keys())) for x in hoi][:3]}")
    return with_side_panel(canvas, "REASONING", lines)


def load_clouds_in_ego(root: Path, sequence: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    lower = read_pcd_xyz(root / "pointclouds" / "lower_velodyne" / sequence / f"{frame}.pcd")
    upper = read_pcd_xyz(root / "pointclouds" / "upper_velodyne" / sequence / f"{frame}.pcd")
    lower_to_ego, upper_to_ego = lidar_to_ego_matrices(str(root.resolve()))

    def transform(points, matrix):
        return (points.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)

    return transform(lower, lower_to_ego), transform(upper, upper_to_ego)


@lru_cache(maxsize=4)
def lidar_to_ego_matrices(root: str) -> tuple[np.ndarray, np.ndarray]:
    import yaml

    calibration = yaml.safe_load((Path(root) / "calibration" / "lidars.yaml").read_text(encoding="utf-8"))
    upper_to_ego = np.asarray(calibration["lidar"]["upper2ego"], dtype=np.float64)
    lower_to_upper = np.asarray(calibration["lidar"]["lower2upper"], dtype=np.float64)
    return upper_to_ego @ lower_to_upper, upper_to_ego


def box_corners_xy(box: dict) -> np.ndarray:
    cx, cy = float(box["cx"]), float(box["cy"])
    length, width = float(box["l"]), float(box["w"])
    angle = float(box["rot_z"])
    local = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                      [length / 2, width / 2], [-length / 2, width / 2],
                      [-length / 2, -width / 2]])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return local @ rotation.T + np.array([cx, cy])


def set_bev_limits(axis, points: np.ndarray, boxes: list[dict]) -> None:
    centers = np.asarray([[float(row["box"]["cx"]), float(row["box"]["cy"])] for row in boxes if row.get("box")])
    if centers.size:
        valid = centers[np.all(np.isfinite(centers), axis=1)]
        if valid.size:
            bound = max(12.0, float(np.max(np.abs(valid))) + 3)
        else:
            bound = 18.0
    else:
        bound = 18.0
    bound = min(bound, 35.0)
    axis.set_xlim(-bound, bound)
    axis.set_ylim(-bound, bound)
    axis.set_aspect("equal")
    axis.set_xlabel("x (m, forward)")
    axis.set_ylabel("y (m, left)")
    axis.grid(alpha=0.2)


def render_bev3d_layer(root: Path, sequence: str, frame: str, rows: list[dict]) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lower, upper = load_clouds_in_ego(root, sequence, frame)
    figure, axis = plt.subplots(figsize=(12, 11), dpi=140)
    axis.scatter(lower[:, 0], lower[:, 1], s=0.35, c="#5a7390", alpha=0.35, label="lower LiDAR")
    axis.scatter(upper[:, 0], upper[:, 1], s=0.35, c="#acb7c0", alpha=0.30, label="upper LiDAR")
    for row in rows:
        box = row.get("box") or {}
        if not all(key in box for key in ("cx", "cy", "l", "w", "rot_z")):
            continue
        corners = box_corners_xy(box)
        color = tuple(x / 255 for x in stable_color(row.get("label_id")))
        axis.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.8)
        axis.text(float(box["cx"]), float(box["cy"]), str(row.get("label_id", "?")), color=color, fontsize=6)
    axis.scatter([0], [0], marker="+", c="red", s=80, label="calibration ego origin")
    set_bev_limits(axis, np.vstack((lower, upper)), rows)
    axis.set_title(f"{sequence} / {frame}: 3D boxes (frame alignment requires visual review)")
    axis.legend(loc="upper right", markerscale=8)
    figure.tight_layout()
    import io
    stream = io.BytesIO()
    figure.savefig(stream, format="png")
    plt.close(figure)
    stream.seek(0)
    image = Image.open(stream).convert("RGB")
    lines = [f"3D boxes={len(rows)}; raw box coordinates drawn against calibrated ego cloud"]
    for row in rows:
        lines.append(f"{row.get('label_id')}: box={row.get('box')} action={row.get('action_label')}")
        lines.append(f"  group={row.get('social_group')} attributes={row.get('attributes')}")
    return with_side_panel(image, "3D LABELS", lines)


def project_virtual_camera(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed oblique virtual camera, with ego x forward/y left/z up."""
    eye = np.array([-16.0, -18.0, 13.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points, dtype=np.float32) - eye
    depth = relative @ forward
    focal = width / (2 * math.tan(math.radians(70) / 2))
    safe_depth = np.maximum(depth, 0.001)
    pixels = np.column_stack((width / 2 + focal * (relative @ right) / safe_depth,
                              height / 2 - focal * (relative @ up) / safe_depth))
    return pixels, depth


def box_corners_3d(box: dict) -> np.ndarray:
    xy = box_corners_xy(box)[:4]
    center_z, height = float(box["cz"]), float(box["h"])
    return np.vstack((np.column_stack((xy, np.full(4, center_z - height / 2))),
                      np.column_stack((xy, np.full(4, center_z + height / 2))))).astype(np.float32)


def render_3d_perspective_layer(root: Path, sequence: str, frame: str, rows: list[dict]) -> Image.Image:
    import cv2

    width, height = 1600, 900
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = (15, 22, 32)
    # Ground grid and axes establish depth; they are guides, not dataset labels.
    for value in range(-15, 16, 5):
        segments = [np.array([[value, -15, -1], [value, 15, -1]], dtype=np.float32),
                    np.array([[-15, value, -1], [15, value, -1]], dtype=np.float32)]
        for segment in segments:
            projected, depth = project_virtual_camera(segment, width, height)
            if np.all(depth > 0.5):
                cv2.line(canvas, tuple(projected[0].astype(int)), tuple(projected[1].astype(int)),
                         (39, 52, 65), 1, cv2.LINE_AA)
    for end, color, name in (((10, 0, 0), (240, 85, 75), "x forward"),
                             ((0, 10, 0), (95, 220, 130), "y left"),
                             ((0, 0, 5), (95, 155, 245), "z up")):
        projected, depth = project_virtual_camera(np.asarray([[0, 0, 0], end], dtype=np.float32), width, height)
        if np.all(depth > 0.5):
            cv2.arrowedLine(canvas, tuple(projected[0].astype(int)), tuple(projected[1].astype(int)),
                            color, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.putText(canvas, name, tuple(projected[1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)

    lower, upper = load_clouds_in_ego(root, sequence, frame)
    for points, base_color in ((lower, (75, 155, 225)), (upper, (185, 205, 215))):
        if len(points) > 30000:
            points = points[::math.ceil(len(points) / 30000)]
        projected, depth = project_virtual_camera(points, width, height)
        pixel = np.rint(projected).astype(np.int32)
        valid = (np.isfinite(projected).all(axis=1) & (depth > 0.5) & (depth < 70)
                 & (pixel[:, 0] >= 0) & (pixel[:, 0] < width)
                 & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
        pixel, depth = pixel[valid], depth[valid]
        if not len(pixel):
            continue
        brightness = np.clip(1.15 - depth / 85, 0.38, 1.0)
        colors = (np.asarray(base_color)[None, :] * brightness[:, None]).astype(np.uint8)
        far_to_near = np.argsort(depth)[::-1]
        pixel, colors = pixel[far_to_near], colors[far_to_near]
        canvas[pixel[:, 1], pixel[:, 0]] = colors
        inside = (pixel[:, 0] + 1) < width
        canvas[pixel[inside, 1], pixel[inside, 0] + 1] = colors[inside]

    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    for row in rows:
        box = row.get("box") or {}
        if not all(key in box for key in ("cx", "cy", "cz", "h", "l", "w", "rot_z")):
            continue
        corners = box_corners_3d(box)
        projected, depth = project_virtual_camera(corners, width, height)
        color = stable_color(row.get("label_id", "?"))
        for first, second in edges:
            if depth[first] > 0.5 and depth[second] > 0.5:
                cv2.line(canvas, tuple(projected[first].astype(int)), tuple(projected[second].astype(int)),
                         color, 2, cv2.LINE_AA)
        top = projected[4:8].mean(axis=0).astype(int)
        if np.all(depth[4:8] > 0.5) and 0 <= top[0] < width and 0 <= top[1] < height:
            cv2.putText(canvas, str(row.get("label_id", "?")), tuple(top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    image = Image.fromarray(canvas)
    lines = [f"3D perspective: {len(rows)} person boxes; fixed virtual camera (-16,-18,13)m",
             "LiDAR points transformed to ego; boxes drawn from raw coordinates.",
             "Coordinate alignment requires visual review; axes x-forward/y-left/z-up."]
    lines.extend(f"{row.get('label_id')}: {row.get('box')}" for row in rows)
    return with_side_panel(image, "3D PERSPECTIVE", lines)


def render_pseudo3d_bev_layer(root: Path, sequence: str, frame: str) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = root / "labels" / "labels_3d_pseudo_panoptic_CW" / sequence / f"{frame}.bin"
    lower, upper = load_clouds_in_ego(root, sequence, frame)
    packed = np.fromfile(path, dtype="<u4")
    if packed.size != len(lower) + len(upper):
        raise ValueError(f"Pseudo-3D label count {packed.size} differs from LiDAR points {len(lower)+len(upper)}")
    low16 = packed & 0xffff
    figure, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=130)
    orders = (("lower + upper", np.vstack((lower, upper))), ("upper + lower", np.vstack((upper, lower))))
    unique_codes, inverse = np.unique(low16, return_inverse=True)
    palette = np.asarray([stable_color(int(value)) for value in unique_codes], dtype=np.float32) / 255
    colors = palette[inverse]
    for axis, (name, points) in zip(axes, orders):
        axis.scatter(points[:, 0], points[:, 1], s=0.7, c=colors, alpha=0.75, linewidths=0)
        set_bev_limits(axis, points, [])
        axis.set_title(f"Candidate point order: {name}")
    figure.tight_layout()
    import io
    stream = io.BytesIO()
    figure.savefig(stream, format="png")
    plt.close(figure)
    stream.seek(0)
    return Image.open(stream).convert("RGB")


def render_pseudo3d_layer(root: Path, sequence: str, frame: str) -> Image.Image:
    """Show raw pseudo-label codes in two *perspective* point-order candidates."""
    import cv2

    path = root / "labels" / "labels_3d_pseudo_panoptic_CW" / sequence / f"{frame}.bin"
    lower, upper = load_clouds_in_ego(root, sequence, frame)
    packed = np.fromfile(path, dtype="<u4")
    if packed.size != len(lower) + len(upper):
        raise ValueError(f"Pseudo-3D label count {packed.size} differs from LiDAR points {len(lower)+len(upper)}")
    low16 = packed & 0xffff
    codes, inverse = np.unique(low16, return_inverse=True)
    colors = np.asarray([stable_color(int(code)) for code in codes], dtype=np.uint8)[inverse]
    panel_width, panel_height = 1100, 800
    canvas = np.empty((panel_height, panel_width * 2, 3), dtype=np.uint8)
    canvas[:] = (15, 22, 32)
    for panel, (name, points) in enumerate((
            ("candidate: lower + upper", np.vstack((lower, upper))),
            ("candidate: upper + lower", np.vstack((upper, lower))))):
        section = canvas[:, panel * panel_width:(panel + 1) * panel_width]
        for value in range(-15, 16, 5):
            for segment in (np.array([[value, -15, -1], [value, 15, -1]], dtype=np.float32),
                            np.array([[-15, value, -1], [15, value, -1]], dtype=np.float32)):
                projected, depth = project_virtual_camera(segment, panel_width, panel_height)
                if np.all(depth > 0.5):
                    cv2.line(section, tuple(projected[0].astype(int)), tuple(projected[1].astype(int)),
                             (35, 47, 60), 1, cv2.LINE_AA)
        projected, depth = project_virtual_camera(points, panel_width, panel_height)
        pixel = np.rint(projected).astype(np.int32)
        valid = (np.isfinite(projected).all(axis=1) & (depth > 0.5) & (depth < 70)
                 & (pixel[:, 0] >= 0) & (pixel[:, 0] < panel_width)
                 & (pixel[:, 1] >= 0) & (pixel[:, 1] < panel_height))
        pixel, depth, colored = pixel[valid], depth[valid], colors[valid]
        far_to_near = np.argsort(depth)[::-1]
        pixel, colored = pixel[far_to_near], colored[far_to_near]
        section[pixel[:, 1], pixel[:, 0]] = colored
        inside = (pixel[:, 0] + 1) < panel_width
        section[pixel[inside, 1], pixel[inside, 0] + 1] = colored[inside]
        cv2.putText(section, name, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(canvas)


def render_frame(root: Path, sequence: str, frame: str, view: str, layers: list[str], output_dir: Path) -> dict:
    validate_id(sequence, frame)
    image_path = root / "images" / view_name(view) / sequence / f"{frame}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if output_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must not be inside the JRDB data root")
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    results: dict = {"sequence": sequence, "frame": frame, "view": view, "source_image": str(image_path),
                     "layers": {}, "warnings": []}
    prepared: dict = {}
    for layer in layers:
        if layer not in OPTIONAL_LAYERS:
            raise ValueError(f"Unknown layer {layer!r}; choices: {OPTIONAL_LAYERS}")
        try:
            if layer in ("people", "actions", "social"):
                if "people" not in prepared:
                    prepared["people"] = frame_rows(label_path(root, "people", sequence, view), frame, "jpg")
                rows = prepared["people"]
                visual = render_people_layer(base, rows, layer)
                count = len(rows)
            elif layer in ("head", "pose", "panoptic"):
                _, rows, categories = coco_frame(label_path(root, layer, sequence, view), frame)
                visual = {"head": render_head_layer, "pose": render_pose_layer,
                          "panoptic": render_panoptic_layer}[layer](base, rows, categories)
                count = len(rows)
            elif layer == "reasoning":
                path = label_path(root, "reasoning", sequence, view)
                if path is None:
                    raise ValueError("Reasoning labels exist only for stitched view")
                rows = frame_rows(path, frame, "jpg")
                visual = render_reasoning_layer(base, rows)
                count = len(rows)
            elif layer in ("view3d", "bev3d"):
                rows = frame_rows(label_path(root, "bev3d", sequence, view), frame, "pcd")
                visual = (render_3d_perspective_layer if layer == "view3d" else render_bev3d_layer)(
                    root, sequence, frame, rows)
                count = len(rows)
            else:
                visual = (render_pseudo3d_layer if layer == "pseudo3d" else render_pseudo3d_bev_layer)(
                    root, sequence, frame)
                count = int(np.fromfile(root / "labels" / "labels_3d_pseudo_panoptic_CW" / sequence / f"{frame}.bin", dtype="<u4").size)
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"{layer}.png"
            visual.save(target)
            results["layers"][layer] = {"status": "rendered", "count": count, "file": str(target)}
        except (FileNotFoundError, ValueError) as error:
            results["layers"][layer] = {"status": "unavailable", "reason": str(error)}
            results["warnings"].append(f"{layer}: {error}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def video_label_index(root: Path, sequence: str, view: str, layer: str) -> tuple[set[str], dict]:
    """Load one label family once, then index its rows by source frame number."""
    if layer in ("pseudo3d", "pseudo3d_bev"):
        folder = root / "labels" / "labels_3d_pseudo_panoptic_CW" / sequence
        return frame_file_set(folder, "bin"), {}
    if layer == "reasoning" and view != "stitched":
        return set(), {}
    family = "people" if layer in ("people", "actions", "social") else "bev3d" if layer == "view3d" else layer
    data = load_json(label_path(root, family, sequence, view))
    if family in ("people", "reasoning", "bev3d"):
        suffix = "pcd" if family == "bev3d" else "jpg"
        rows = {Path(key).stem: value for key, value in data["labels"].items() if key.endswith(f".{suffix}")}
        return set(rows), {"rows": rows}
    images = {}
    for image in data["images"]:
        frame = Path(image["file_name"]).stem
        if frame in images:
            raise ValueError(f"Duplicate COCO frame name {frame} in {label_path(root, family, sequence, view)}")
        images[frame] = image["id"]
    by_image_id = defaultdict(list)
    for row in data["annotations"]:
        by_image_id[row["image_id"]].append(row)
    categories = {row["id"]: row for row in data["categories"]}
    return set(images), {"images": images, "annotations": by_image_id, "categories": categories}


def video_visual(root: Path, sequence: str, frame: str, view: str,
                 layer: str, base: Image.Image, index: dict) -> Image.Image:
    if layer in ("people", "actions", "social"):
        return render_people_layer(base, index["rows"][frame], layer)
    if layer in ("head", "pose", "panoptic"):
        rows = index["annotations"][index["images"][frame]]
        return {"head": render_head_layer, "pose": render_pose_layer,
                "panoptic": render_panoptic_layer}[layer](base, rows, index["categories"])
    if layer == "reasoning":
        return render_reasoning_layer(base, index["rows"][frame])
    if layer in ("view3d", "bev3d"):
        return (render_3d_perspective_layer if layer == "view3d" else render_bev3d_layer)(
            root, sequence, frame, index["rows"][frame])
    return (render_pseudo3d_layer if layer == "pseudo3d" else render_pseudo3d_bev_layer)(
        root, sequence, frame)


def video_canvas(visual: Image.Image, width: int, height: int,
                 sequence: str, frame: str, layer: str, sparse: bool) -> Image.Image:
    """Fit variable-size still visualizations into an even, fixed-size movie frame."""
    canvas = Image.new("RGB", (width, height), (19, 23, 29))
    available_width, available_height = width - 24, height - 108
    scale = min(available_width / visual.width, available_height / visual.height)
    resized = visual.resize((max(1, round(visual.width * scale)),
                             max(1, round(visual.height * scale))), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((width - resized.width) // 2, 56 + (available_height - resized.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"{sequence}  frame {frame}  {layer}",
              fill=(255, 255, 255), font=font(24))
    note = "ANNOTATED KEYFRAMES ONLY - no propagation" if sparse else "Source frame ID shown; sequence sampled without interpolation"
    draw.text((16, height - 36), note, fill=(255, 200, 90) if sparse else (185, 205, 215), font=font(20))
    return canvas


def video_rate(frames: list[str], stride: int, sparse: bool, source_fps: int = 15) -> Fraction:
    if len(frames) > 1:
        step = statistics.median(int(b) - int(a) for a, b in zip(frames, frames[1:]))
    else:
        step = 15 if sparse else stride
    return Fraction(source_fps, 1) / Fraction(step).limit_denominator(1000)


def encode_mp4(path: Path, images, width: int, height: int, rate: Fraction) -> int:
    import av

    count = 0
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for image in images:
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    return count


def render_sequence_videos(root: Path, sequence: str, view: str, layers: list[str],
                           output_dir: Path, start: str | None = None, end: str | None = None,
                           stride: int = 1, max_frames: int | None = None,
                           width: int = 2560, height: int = 960) -> dict:
    validate_id(sequence, start)
    if end is not None:
        validate_id(sequence, end)
    if start and end and start > end:
        raise ValueError("--start-frame must not exceed --end-frame")
    if stride < 1 or max_frames is not None and max_frames < 1:
        raise ValueError("--stride and --max-frames must be positive")
    if width < 320 or height < 240 or width % 2 or height % 2:
        raise ValueError("Video width/height must be even and at least 320x240")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if output_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must not be inside the JRDB data root")
    image_dir = root / "images" / view_name(view) / sequence
    sensor_frames = sorted(frame_file_set(image_dir, "jpg"))
    if not sensor_frames:
        raise FileNotFoundError(f"No source images in {image_dir}")
    bounded = [frame for frame in sensor_frames if (start is None or frame >= start) and (end is None or frame <= end)]
    sampled = bounded[::stride]
    if not sampled:
        raise ValueError("No image frames in the requested range")
    layers = list(dict.fromkeys(layers))
    for layer in layers:
        if layer not in OPTIONAL_LAYERS:
            raise ValueError(f"Unknown layer {layer!r}; choices: {OPTIONAL_LAYERS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"sequence": sequence, "view": view, "source_frames_in_range": len(bounded),
              "sampled_sensor_frames": len(sampled), "stride": stride, "max_frames_per_layer": max_frames,
              "video_size": [width, height], "source_fps_assumed": 15, "layers": {}}
    for layer in layers:
        available, index = video_label_index(root, sequence, view, layer)
        frames = [frame for frame in sampled if frame in available]
        if max_frames is not None:
            frames = frames[:max_frames]
        if not frames:
            result["layers"][layer] = {"status": "unavailable", "reason": "No annotated frames in selected range/view"}
            continue
        sparse = layer in ("panoptic", "pseudo3d", "pseudo3d_bev")
        rate = video_rate(frames, stride, sparse)
        target = output_dir / f"{layer}.mp4"
        partial = output_dir / f"{layer}.partial.mp4"

        def movie_frames():
            for position, frame in enumerate(frames, 1):
                with Image.open(image_dir / f"{frame}.jpg") as image:
                    base = image.convert("RGB")
                visual = video_visual(root, sequence, frame, view, layer, base, index)
                if position == 1 or position % 100 == 0 or position == len(frames):
                    print(f"{layer}: {position}/{len(frames)} source frame {frame}", file=sys.stderr, flush=True)
                yield video_canvas(visual, width, height, sequence, frame, layer, sparse)

        try:
            encoded = encode_mp4(partial, movie_frames(), width, height, rate)
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        result["layers"][layer] = {"status": "rendered", "file": str(target),
                                     "frames": encoded, "source_frame_ids": frames,
                                     "fps": float(rate), "policy": "annotated_keyframes_only" if sparse else "sampled_dense_frames",
                                     "sampled_frames_without_this_label": len(sampled) - len([f for f in sampled if f in available])}
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def render_video_montage(input_dir: Path, output: Path,
                         cell_width: int = 1280, cell_height: int = 480) -> dict:
    """Synchronize already-rendered layer videos by *source* frame ID."""
    import av

    if cell_width < 320 or cell_height < 240 or cell_width % 2 or cell_height % 2:
        raise ValueError("Montage cell width/height must be even and at least 320x240")
    if output.exists():
        raise FileExistsError(output)
    manifest_path = input_dir / "manifest.json"
    manifest = load_json(manifest_path)
    layer_info = manifest["layers"]
    layers = [layer for layer in OPTIONAL_LAYERS if layer in layer_info]
    if not layers:
        raise ValueError("No layer videos in input manifest")
    all_frames = sorted({frame for layer in layers for frame in layer_info[layer].get("source_frame_ids", [])})
    if not all_frames:
        raise ValueError("No encoded source frames in input manifest")
    columns = min(3, len(layers))
    rows = math.ceil(len(layers) / columns)
    width, height = columns * cell_width, rows * cell_height
    rate = video_rate(all_frames, manifest.get("stride", 1), False)
    partial = output.with_name(output.stem + ".partial.mp4")
    if partial.exists():
        raise FileExistsError(partial)
    output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        decoders = {}
        positions = {}
        for layer in layers:
            info = layer_info[layer]
            positions[layer] = 0
            if info.get("status") != "rendered":
                continue
            path = input_dir / f"{layer}.mp4"
            container = stack.enter_context(av.open(str(path)))
            decoders[layer] = iter(container.decode(video=0))

        def montage_frames():
            for index, frame in enumerate(all_frames, 1):
                canvas = Image.new("RGB", (width, height), (15, 19, 25))
                draw = ImageDraw.Draw(canvas)
                for layer_index, layer in enumerate(layers):
                    column, row = layer_index % columns, layer_index // columns
                    x0, y0 = column * cell_width, row * cell_height
                    info = layer_info[layer]
                    ids = info.get("source_frame_ids", [])
                    position = positions[layer]
                    available = position < len(ids) and ids[position] == frame
                    draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
                                   outline=(65, 78, 90), width=2)
                    draw.text((x0 + 12, y0 + 5), f"{layer.upper()}  |  source frame {frame}",
                              fill=(255, 255, 255), font=font(20))
                    if available:
                        try:
                            decoded = next(decoders[layer])
                        except StopIteration as error:
                            raise ValueError(f"Video for {layer} ended before source frame {frame}") from error
                        positions[layer] += 1
                        image = decoded.to_image().convert("RGB")
                        if layer not in ("pseudo3d", "pseudo3d_bev"):
                            image = crop_video_content(image)
                        image = ImageOps.contain(image, (cell_width - 12, cell_height - 38), Image.Resampling.LANCZOS)
                        canvas.paste(image, (x0 + (cell_width - image.width) // 2,
                                             y0 + 34 + (cell_height - 38 - image.height) // 2))
                    else:
                        draw.text((x0 + 24, y0 + cell_height // 2),
                                  "NO LABEL FOR THIS SOURCE FRAME", fill=(255, 190, 80), font=font(24))
                if index == 1 or index % 25 == 0 or index == len(all_frames):
                    print(f"montage: {index}/{len(all_frames)} source frame {frame}", file=sys.stderr, flush=True)
                yield canvas

        try:
            encoded = encode_mp4(partial, montage_frames(), width, height, rate)
            partial.replace(output)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    result = {"source_manifest": str(manifest_path), "sequence": manifest.get("sequence"),
              "view": manifest.get("view"), "file": str(output), "layers": layers,
              "frames": encoded, "source_frame_ids": all_frames, "fps": float(rate),
              "video_size": [width, height], "cell_size": [cell_width, cell_height],
              "missing_cell_policy": "explicit_no_label_placeholder"}
    result_path = output.with_name(output.stem + "_manifest.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def crop_video_content(image: Image.Image) -> Image.Image:
    """Trim letterboxing from a layer MP4 before placing it in a montage cell."""
    pixels = np.asarray(image)
    bright = pixels.mean(axis=2) > 55
    rows = np.flatnonzero(bright.sum(axis=1) > image.width * 0.15)
    if rows.size == 0:
        return image
    top, bottom = max(0, int(rows[0]) - 4), min(image.height, int(rows[-1]) + 5)
    columns = np.flatnonzero(bright[top:bottom].sum(axis=0) > (bottom - top) * 0.20)
    if columns.size == 0:
        return image.crop((0, top, image.width, bottom))
    left, right = max(0, int(columns[0]) - 4), min(image.width, int(columns[-1]) + 5)
    return image.crop((left, top, right, bottom))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Extracted JRDB train root")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Check frame, COCO, and pseudo-3D alignment")
    audit.add_argument("--sequence", help="One sequence; omit to audit all 27")
    audit.add_argument("--output", type=Path, required=True, help="New JSON report path")
    render = commands.add_parser("render", help="Render one frame into separate per-label PNGs")
    render.add_argument("--sequence", required=True)
    render.add_argument("--frame", required=True, help="Six-digit frame number")
    render.add_argument("--view", choices=VIEWS, default="stitched")
    render.add_argument("--layers", default="all", help="Comma-separated layer names, or all")
    render.add_argument("--output-dir", type=Path, required=True, help="New/empty directory outside dataset root")
    video = commands.add_parser("video", help="Render one whole sequence as separate H.264 MP4s per label layer")
    video.add_argument("--sequence", required=True)
    video.add_argument("--view", choices=VIEWS, default="stitched")
    video.add_argument("--layers", default="all", help="Comma-separated layer names, or all")
    video.add_argument("--start-frame", help="Optional inclusive six-digit source frame")
    video.add_argument("--end-frame", help="Optional inclusive six-digit source frame")
    video.add_argument("--stride", type=int, default=1, help="Sample one source frame every N frames; default 1")
    video.add_argument("--max-frames", type=int, help="Optional maximum encoded frames per layer, useful for previews")
    video.add_argument("--width", type=int, default=2560)
    video.add_argument("--height", type=int, default=960)
    video.add_argument("--output-dir", type=Path, required=True, help="New/empty directory outside dataset root")
    montage = commands.add_parser("montage", help="Combine per-layer MP4s into one synchronized grid video")
    montage.add_argument("--input-dir", type=Path, required=True, help="Directory produced by the video command")
    montage.add_argument("--output", type=Path, required=True, help="New H.264 MP4 path")
    montage.add_argument("--cell-width", type=int, default=1280)
    montage.add_argument("--cell-height", type=int, default=480)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command != "montage" and (not (root / "images" / "image_stitched").is_dir() or not (root / "labels").is_dir()):
        parser.error(f"Not an extracted JRDB root: {root}")
    try:
        if args.command == "audit":
            if args.output.exists():
                raise FileExistsError(args.output)
            if args.output.resolve().is_relative_to(root):
                raise ValueError("Audit output must not be inside the JRDB data root")
            sequences = [args.sequence] if args.sequence else sorted(p.name for p in (root / "images" / "image_stitched").iterdir() if p.is_dir())
            reports = []
            for index, sequence in enumerate(sequences, 1):
                print(f"[{index}/{len(sequences)}] auditing {sequence}", file=sys.stderr, flush=True)
                reports.append(audit_sequence(root, sequence))
            result = {"dataset_root": str(root), "sequences": reports}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")
        elif args.command == "render":
            layers = list(LAYERS) if args.layers == "all" else [part.strip() for part in args.layers.split(",") if part.strip()]
            result = render_frame(root, args.sequence, args.frame, args.view, layers, args.output_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "video":
            layers = list(LAYERS) if args.layers == "all" else [part.strip() for part in args.layers.split(",") if part.strip()]
            result = render_sequence_videos(root, args.sequence, args.view, layers, args.output_dir,
                                            start=args.start_frame, end=args.end_frame, stride=args.stride,
                                            max_frames=args.max_frames, width=args.width, height=args.height)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            result = render_video_montage(args.input_dir, args.output, args.cell_width, args.cell_height)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, IndexError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
