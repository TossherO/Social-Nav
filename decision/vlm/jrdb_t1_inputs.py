"""Geometry-only T1 front RGB/context BEV rendering and typed REL protocol."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from decision.vlm import jrdb_t1_common as t1
    from decision.vlm.project_config import dataset_path
except ModuleNotFoundError:
    import jrdb_t1_common as t1
    from project_config import dataset_path

VLM_ROOT = Path(__file__).resolve().parent
ROOT = VLM_ROOT.parents[1]
CONFIG_DIR = VLM_ROOT / "configs"
PROTOCOL = "jrdb_t1_object"
PROFILES = ("object",)
BG = (246, 248, 250)
INK = (26, 39, 51)
CYAN = (0, 132, 158)


def rel_schema(valid_pid: list[int]) -> dict:
    """XGrammar-compatible schema; membership/exclusivity checked separately.

    uniqueItems and cross-array exclusion are NOT guaranteed by grammar engines.
    Avoid pretending that prompt instructions alone constrain decoding.
    """
    if (len(valid_pid) > 20 or len(set(valid_pid)) != len(valid_pid)
            or any(type(p) is not int or p < 0 or p > 99 for p in valid_pid)):
        raise ValueError("Invalid VALID_PID")
    pid = {"type": "integer", "enum": sorted(valid_pid)} if valid_pid else {"type": "integer"}
    group = {"type": "object", "additionalProperties": False,
             "required": ["m", "r", "k"], "properties": {
                 "m": {"type": "array", "minItems": 2, "maxItems": len(valid_pid), "items": pid},
                 "r": {"type": "string", "enum": ["w", "t", "q", "p", "o", "u"]},
                 "k": {"type": "integer", "enum": [0, 1]}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["g", "u"],
              "properties": {
                  "g": {"type": "array", "maxItems": len(valid_pid) // 2, "items": group},
                  "u": {"type": "array", "maxItems": len(valid_pid), "items": pid}}}
    # For zero/one valid person, the group item schema must still be satisfiable.
    if len(valid_pid) < 2:
        schema["properties"]["g"]["items"] = {"type": "object"}
    return schema


def source_rgb(window: t1.Window, index: int) -> tuple[Image.Image, dict[int, list]]:
    path = window.root / "images/image_0" / window.sequence / f"{window.frames[index]}.jpg"
    with Image.open(path) as opened:
        pixels = np.asarray(opened.convert("RGB"))
    cal = window.calibration
    pixels = cv2.undistort(pixels, np.asarray(cal["distorted_img_K"], float),
                         np.asarray(cal["D"], float), None, np.asarray(cal["undistorted_img_K"], float))
    image = Image.fromarray(pixels)
    boxes = {}
    for identity, row in window.two[index].items():
        if identity not in window.pid_map:
            continue
        box = t1.undistorted_box(row["box"], cal, image.size)
        if box is not None:
            boxes[window.pid_map[identity]] = box
    return image, boxes


def text_tag(draw, xy, text, *, size=20, color=INK, fill=BG):
    face = t1.font(size)
    x, y = xy
    box = draw.textbbox((x, y), text, font=face)
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=fill)
    draw.text((x, y), text, font=face, fill=color)


def fit_image(image: Image.Image, size: tuple[int, int], *, background=BG):
    scale = min(size[0] / image.width, size[1] / image.height)
    dimensions = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    offset = ((size[0] - dimensions[0]) // 2, (size[1] - dimensions[1]) // 2)
    canvas = Image.new("RGB", size, background)
    canvas.paste(image.resize(dimensions, Image.Resampling.LANCZOS), offset)
    return canvas, scale, offset


def display_pid(pid, suffix):
    """Compact result-only group/role tag, without a second text overlay."""
    if not suffix:
        return f"P{pid}"
    parts = suffix.strip().split()
    if parts[0] == "?":
        return f"P{pid}?"
    group = parts[0].split(":")[0]
    return f"P{pid}/{group}" + (" " + " ".join(parts[1:]) if len(parts) > 1 else "")


def tagged_rgb(window, index, size=(800, 512), styles=None):
    source, boxes = source_rgb(window, index)
    canvas, scale, offset = fit_image(source, size)
    draw = ImageDraw.Draw(canvas)
    occupied = []
    for pid, box in sorted(boxes.items()):
        color, suffix = (styles or {}).get(pid, (CYAN, ""))
        x1, y1, x2, y2 = [v * scale + offset[j % 2] for j, v in enumerate(box)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = display_pid(pid, suffix)
        w = math.ceil(draw.textlength(label, font=t1.font(20))) + 6
        candidates = [(x1, y1 - 25), (x2 - w, y2 + 2), (x2 + 3, y1), (x1 - w - 3, y1)]
        best = None
        best_cost = float("inf")
        for x, y in candidates:
            x, y = max(3, min(size[0] - w - 3, x)), max(3, min(size[1] - 25, y))
            rect = (x, y, x + w, y + 24)
            overlaps = sum(max(0, min(rect[2], b[2]) - max(rect[0], b[0]))
                           * max(0, min(rect[3], b[3]) - max(rect[1], b[1])) for b in occupied)
            if overlaps < best_cost:
                best, best_cost = rect, overlaps
        occupied.append(best)
        # The leader ties a displaced tag to the correct person without social edges.
        draw.line(((best[0] + w / 2, best[1] + 12), ((x1 + x2) / 2, y1)), fill=color, width=1)
        text_tag(draw, best[:2], label, color=color)
    return canvas, source, boxes


def bev_extent(window):
    """Same metric extent in all frames, only current valid people, no semantics."""
    points = [(0., 0.)]
    for rows in window.three:
        for identity in window.active:
            box = rows.get(identity, {}).get("box")
            if box is not None:
                points.append((box["cx"], box["cy"]))
    xlo, xhi = min(-2, min(p[0] for p in points) - 1), max(10, max(p[0] for p in points) + 1)
    ylo, yhi = min(-6, min(p[1] for p in points) - 1), max(6, max(p[1] for p in points) + 1)
    span = max(xhi - xlo, yhi - ylo)
    xmid, ymid = (xlo + xhi) / 2, (ylo + yhi) / 2
    return [xmid - span / 2, xmid + span / 2, ymid - span / 2, ymid + span / 2]


def clean_bev(window, index, side=480, styles=None, links=None, extent=None):
    canvas = Image.new("RGB", (side, side), BG)
    draw = ImageDraw.Draw(canvas)
    margin = 34
    xlo, xhi, ylo, yhi = extent or [-2, 10, -6, 6]
    unit = (side - margin * 2) / (xhi - xlo)
    def xy(x, y):
        return margin + (yhi - y) * unit, margin + (xhi - x) * unit
    step = max(2, math.ceil((xhi - xlo) / 6 / 2) * 2)
    for x in range(math.ceil(xlo / step) * step, math.floor(xhi / step) * step + 1, step):
        draw.line((xy(x, ylo), xy(x, yhi)), fill=(219, 225, 231))
        draw.text((5, xy(x, 0)[1] - 7), str(x), fill=INK, font=t1.font(13))
    for y in range(math.ceil(ylo / step) * step, math.floor(yhi / step) * step + 1, step):
        draw.line((xy(xlo, y), xy(xhi, y)), fill=(219, 225, 231))
        draw.text((xy(0, y)[0] - 6, side - 24), str(y), fill=INK, font=t1.font(13))
    rx, ry = xy(0, 0)
    draw.polygon([(rx, ry - 12), (rx - 8, ry + 8), (rx + 8, ry + 8)], fill=INK)
    text_tag(draw, (rx + 12, ry - 5), "R", size=15)
    centers = {}
    for identity in window.active:
        row = window.three[index].get(identity)
        if row is None or (extent is None and not t1.inside_bev(row["box"])):
            continue
        pid = window.pid_map[identity]
        centers[pid] = xy(row["box"]["cx"], row["box"]["cy"])
    for link in links or []:
        members = link["members"]
        linked = [members[:2]] if link["relation"] == "p" and link["directed"] else zip(members, members[1:])
        for a, b in linked:
            if a not in centers or b not in centers:
                continue
            if link["directed"]:
                t1.arrow(draw, centers[a], centers[b], link["color"], 3)
            else:
                draw.line((centers[a], centers[b]), fill=link["color"], width=3)
    occupied = []
    for pid, (x, y) in sorted(centers.items()):
        color = (styles or {}).get(pid, (CYAN, ""))[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        suffix = (styles or {}).get(pid, (CYAN, ""))[1]
        label = display_pid(pid, suffix) if styles else f"P{pid}"
        width = math.ceil(draw.textlength(label, font=t1.font(19))) + 6
        candidates = [(x + 8, y - 20), (x - width - 8, y - 20), (x + 8, y + 8), (x - width - 8, y + 8)]
        for radius in (28, 48, 70):
            candidates.extend([(x + 8, y - radius), (x - width - 8, y + radius)])
        costs = []
        for left, top in candidates:
            left, top = max(3, min(side - width - 3, left)), max(30, min(side - 48, top))
            rect = (left, top, left + width, top + 23)
            overlap = sum(max(0, min(rect[2], b[2]) - max(rect[0], b[0]))
                          * max(0, min(rect[3], b[3]) - max(rect[1], b[1])) for b in occupied)
            costs.append((overlap, abs(left - x) + abs(top - y), rect))
        rect = min(costs, key=lambda v: v[:2])[2]
        occupied.append(rect)
        draw.line(((x, y), (rect[0] + width / 2, rect[1] + 11)), fill=color, width=1)
        text_tag(draw, rect[:2], label, size=19, color=color)
    text_tag(draw, (8, 7), "BEV: x forward, y left (m)", size=15)
    return canvas


def render_focus(window: t1.Window, index: int, *, gallery=False, styles=None, links=None, context_bev=True):
    count = len(window.valid_pid)
    rows = max(1, math.ceil(count / 10)) if gallery else 0
    height = 544 + (32 + rows * 160 if gallery else 0)
    canvas = Image.new("RGB", (1280, height), BG)
    draw = ImageDraw.Draw(canvas)
    rgb, source, boxes = tagged_rgb(window, index, styles=styles)
    canvas.paste(rgb, (0, 32))
    canvas.paste(clean_bev(window, index, styles=styles, links=links,
                           extent=bev_extent(window) if context_bev else None), (800, 32))
    lag = window.times[index] - window.times[-1]
    text_tag(draw, (9, 7), f"FRONT  t={lag:+.2f}s", size=19)
    if gallery:
        text_tag(draw, (9, 552), "DETAILS: same front image; tiles in PID order", size=16)
        for j, pid in enumerate(window.valid_pid):
            left, top = (j % 10) * 128, 576 + (j // 10) * 160
            draw.rectangle((left + 2, top + 2, left + 125, top + 157), outline=(212, 219, 226))
            box = boxes.get(pid)
            if box is not None:
                x1, y1, x2, y2 = box
                pad_x, pad_y = (x2 - x1) * 0.12, (y2 - y1) * 0.05
                crop_box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                            min(source.width, math.ceil(x2 + pad_x)), min(source.height, math.ceil(y2 + pad_y)))
                tile, _, _ = fit_image(source.crop(crop_box), (120, 128))
                canvas.paste(tile, (left + 4, top + 26))
            else:
                draw.text((left + 10, top + 74), "not seen", fill=(129, 139, 150), font=t1.font(15))
            text_tag(draw, (left + 8, top + 5), f"P{pid}", size=18, color=CYAN)
    return canvas


def observations(window):
    images = [source_rgb(window, i) for i in range(3)]
    result = []
    for identity in window.active:
        pid = window.pid_map[identity]
        item = {"pid": pid, "xy": [], "rgb_box_1000": [], "rgb_height_px": []}
        for i, (image, boxes) in enumerate(images):
            box3 = window.three[i].get(identity, {}).get("box")
            item["xy"].append(None if box3 is None else [round(box3["cx"], 2), round(box3["cy"], 2)])
            box = boxes.get(pid)
            item["rgb_box_1000"].append(None if box is None else
                                      [round(v * 1000 / (image.width if j % 2 == 0 else image.height))
                                       for j, v in enumerate(box)])
            item["rgb_height_px"].append(None if box is None else round(box[3] - box[1]))
        result.append(item)
    return sorted(result, key=lambda r: r["pid"])


def near_pair_distances(obs, max_distance=3.5):
    """Geometry only; pairwise ego-rotation/translation-invariant distances.

    A distance gate saves tokens, but is not a group classifier or label hint.
    """
    pairs = []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            distances = [None if x is None or y is None else round(math.dist(x, y), 2)
                         for x, y in zip(a["xy"], b["xy"])]
            if any(d is not None and d <= max_distance for d in distances):
                pairs.append([a["pid"], b["pid"], *distances])
    return pairs


def model_input(window, profile="object"):
    if profile != "object":
        raise ValueError("Only object is active; historical variants are archived")
    if window.scope != "front" or window.poses is not None:
        raise ValueError("The current T1 pilot requires front scope without inferred ego motion")
    obs = observations(window)
    end = window.times[-1]
    fields = {"v": window.valid_pid, "dt": [round(t - end, 6) for t in window.times],
              "ts": window.time_source, "e": [round(v, 2) for v in bev_extent(window)],
              "o": [[r["pid"], r["xy"], r["rgb_box_1000"], r["rgb_height_px"]] for r in obs],
              "d": near_pair_distances(obs)}
    system = (CONFIG_DIR / "prompts/jrdb_t1_rel.txt").read_text(encoding="utf-8").strip()
    user = ("Decide at IMAGE_2 using these observations only.\n"
            + json.dumps(fields, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\nReturn only the required JSON. Use integer PIDs exclusively from v. "
              "g is a list of group objects; each m is a nested integer array.")
    return fields, system, user


def implementation_hashes():
    """Pin the renderer and geometry implementation used to create an input."""
    return {p.name: t1.file_digest(p) for p in
            (Path(__file__), VLM_ROOT / "jrdb_t1_common.py")}


def save_bundle(window, profile, out: Path, *, contact_sheet=False):
    """One geometry-only renderer/protocol; review artifacts never enter HTTP."""
    t1.check_output(window.root, out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output is not empty: {out}")
    fields, system, user = model_input(window, profile)
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(3):
        image = render_focus(window, i, gallery=False, context_bev=True)
        image.save(out / f"image_{i}.png")
        images.append(image)
    if contact_sheet:
        contact = Image.new("RGB", (images[0].width, sum(i.height for i in images)), BG)
        for i, image in enumerate(images):
            contact.paste(image, (0, sum(p.height for p in images[:i])))
        contact.save(out / "input_contact_sheet.png")
    hashes = {f"image_{i}.png": t1.file_digest(out / f"image_{i}.png") for i in range(3)}
    model = {"images": list(hashes), "wire_protocol": "object", "fields": fields,
             "user_prompt": user, "response_schema": rel_schema(window.valid_pid)}
    assets = ["calibration/lidars.yaml"] + [
        f"images/image_0/{window.sequence}/{f}.jpg" for f in window.frames]
    root_ref = "jrdb" if window.root.absolute() == dataset_path("jrdb").absolute() else None
    manifest = {"protocol": PROTOCOL, "profile": profile, "root": str(window.root),
                "dataset_root_ref": root_ref,
                "sequence": window.sequence, "frame_ids": window.frames, "scope": window.scope,
                "timestamps": window.times, "time_source": window.time_source,
                "pid_to_source_id": {str(p): identity for identity, p in window.pid_map.items()},
                "source_assets_sha256": {name: t1.file_digest(window.root / name) for name in assets},
                "source_geometry_signature": window.geometry_signature,
                "source_lidar_enabled": window.lidar, "nominal_fps": window.nominal_fps,
                "source_canvas": [window.width, window.height],
                "implementation_sha256": implementation_hashes(),
                "image_sha256": hashes, "image_sizes": [list(i.size) for i in images],
                "system_prompt_sha256": t1.file_digest(CONFIG_DIR / "prompts/jrdb_t1_rel.txt"),
                "input_fingerprint": t1.digest({"model_input": model, "images": hashes, "system": system}),
                "perception_tier": "P0_oracle_boxes_and_association",
                "status": "development_input_not_reviewed_T1_gold"}
    t1.write_json(out / "model_input.json", model)
    t1.write_json(out / "manifest.json", manifest)
    t1.write_json(out / "annotation_template.json", {
        "input_fingerprint": manifest["input_fingerprint"], "review_status": "draft",
        "reviewer": None, "reviewed_at": None, "label_provenance": None,
        "warning": "Unknown placeholder, NOT gold. Review the exact model-visible input first.",
        "response": {"g": [], "u": window.valid_pid}})
    (out / "system_prompt.txt").write_text(system + "\n", encoding="utf-8")
    return manifest


def load_input(out: Path):
    model = t1.jrdb.load_json(out / "model_input.json")
    manifest = t1.jrdb.load_json(out / "manifest.json")
    for name, expected in manifest["image_sha256"].items():
        if Path(name).name != name or t1.file_digest(out / name) != expected:
            raise ValueError("Input image integrity check failed")
    system = (out / "system_prompt.txt").read_text(encoding="utf-8").strip()
    if t1.digest({"model_input": model, "images": manifest["image_sha256"], "system": system}) != manifest["input_fingerprint"]:
        raise ValueError("Model input changed since rendering")
    return model, manifest


def load_window(out: Path):
    """Rebuild geometry for result rendering and check its source provenance."""
    model, manifest = load_input(out)
    if manifest.get("implementation_sha256") != implementation_hashes():
        raise ValueError("Rendering implementation changed since input generation; use the matching source version or regenerate inputs and review labels")
    root_ref = manifest.get("dataset_root_ref")
    if root_ref is not None and root_ref != "jrdb":
        raise ValueError("Unsupported JRDB dataset root reference")
    root = dataset_path(root_ref) if root_ref is not None else Path(manifest["root"])
    for name, expected in manifest["source_assets_sha256"].items():
        if t1.file_digest(root / name) != expected:
            raise ValueError("Source asset changed since input rendering")
    window = t1.build_window(root, manifest["sequence"], manifest["frame_ids"],
                             scope=manifest["scope"], lidar=manifest.get("source_lidar_enabled", True),
                             fps=manifest.get("nominal_fps", 15),
                             width=manifest.get("source_canvas", [896, 448])[0],
                             height=manifest.get("source_canvas", [896, 448])[1])
    if window.geometry_signature != manifest["source_geometry_signature"]:
        raise ValueError("Source geometry changed since input rendering")
    return window, model, manifest
