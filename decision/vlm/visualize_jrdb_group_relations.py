"""Sequence-level, raw-label group/HHI review; never a VLM input renderer."""

from __future__ import annotations

import argparse
from collections import Counter
import colorsys
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

try:
    from decision.vlm import jrdb_t1_common as t1
    from decision.vlm import jrdb_visualize as jrdb
except ModuleNotFoundError:
    import jrdb_t1_common as t1
    import jrdb_visualize as jrdb


BG = (18, 24, 32)
FG = (224, 231, 240)
CROSS = (255, 91, 99)
UNKNOWN = (164, 175, 187)
SHORT = {
    "standing together": "STAND", "conversation": "TALK",
    "sitting together": "SIT", "eating together": "EAT",
    "walking together": "WALK", "walking toward each other": "APPROACH",
    "looking at robot together": "LOOK-ROBOT",
    "going downstairs together": "DOWNSTAIRS", "going upstairs together": "UPSTAIRS",
    "cycling together": "CYCLE", "hugging": "HUG", "moving together": "MOVE",
    "waving hand together": "WAVE", "looking into sth together": "LOOK-IN",
    "shaking hand": "HANDSHAKE", "looking at sth together": "LOOK-AT",
    "bending together": "BEND", "holding sth together": "HOLD",
    "interaction with door together": "DOOR",
}


@lru_cache(maxsize=16)
def font(size):
    return t1.font(size)


@lru_cache(maxsize=1024)
def group_color(group):
    if group is None:
        return UNKNOWN
    hue = int.from_bytes(hashlib.sha256(str(group).encode()).digest()[:4], "big") / 2**32
    return tuple(round(x * 255) for x in colorsys.hsv_to_rgb(hue, .56, .98))


def pid(identity):
    return "P" + identity.removeprefix("pedestrian:")


def group_of(row):
    return (row.get("social_group") or {}).get("cluster_ID")


def tag(identity, actors):
    group = actors[identity]["group_id"]
    conflict = "!" if actors[identity].get("group_conflict") else ""
    return f"{pid(identity)}/G{group if group is not None else '?'}{conflict}"


def unique_rows(rows, source, warnings):
    result = {}
    for row in rows:
        identity = row["label_id"]
        if identity in result:
            warnings.append({"kind": "duplicate_id", "source": source, "person": identity})
            continue  # Keep the first record and report, rather than silently overwrite.
        result[identity] = row
    return result


def frame_graph(stitched_rows, front_rows, three_rows):
    """Keep membership and HHI separate; merge only reverse duplicate HHI edges."""
    warnings = []
    stitched = unique_rows(stitched_rows, "stitched", warnings)
    front = unique_rows(front_rows, "front", warnings)
    three = unique_rows(three_rows, "3d", warnings)
    actors = {}
    for identity in sorted(stitched.keys() | front.keys() | three.keys()):
        row = stitched.get(identity, front.get(identity, three.get(identity)))
        source = "stitched" if identity in stitched else "front" if identity in front else "3d"
        group = group_of(row)
        actors[identity] = {
            "group_id": group, "group_source": source,
            "group_difficulty": (row.get("social_group") or {}).get("cluster_stat"),
            "box_2d": front.get(identity, {}).get("box"),
            "box_3d": three.get(identity, {}).get("box"),
            "attributes_3d": three.get(identity, {}).get("attributes", {}),
            "group_conflict": False,
        }
        for name, other in (("front", front), ("3d", three)):
            if identity in other and group_of(other[identity]) != group:
                actors[identity]["group_conflict"] = True
                warnings.append({"kind": "group_conflict", "person": identity,
                                 "canonical_source": source, "canonical_group": group,
                                 "other_source": name, "other_group": group_of(other[identity])})
        if actors[identity]["box_3d"] is None:
            warnings.append({"kind": "missing_3d", "person": identity})

    groups = {}
    for identity, actor in actors.items():
        if actor["group_id"] is not None:
            groups.setdefault(actor["group_id"], []).append(identity)
    membership = [list(pair) for members in groups.values() for pair in combinations(members, 2)]
    edges = {}
    # The panorama is authoritative, including people outside image_0. Fallback
    # camera HHI is used only for identities absent from the panorama.
    relation_rows = list(stitched.values()) + [r for i, r in front.items() if i not in stitched]
    for row in relation_rows:
        source = row["label_id"]
        for interaction in row.get("H-interaction", []):
            target = interaction.get("pair")
            if not target or source == target:
                warnings.append({"kind": "invalid_pair", "source": source, "target": target})
                continue
            labels = interaction.get("inter_labels") or {}
            if not labels:
                continue
            pair = tuple(sorted((source, target)))
            edge = edges.setdefault(pair, {"pair": list(pair), "labels": [], "annotations": []})
            edge["annotations"].append({"source": source, "target": target, "inter_labels": labels})
            edge["labels"] = sorted(set(edge["labels"]) | labels.keys())
    for edge in edges.values():
        a, b = edge["pair"]
        ga = actors.get(a, {}).get("group_id")
        gb = actors.get(b, {}).get("group_id")
        edge["same_group"] = None if ga is None or gb is None else ga == gb
        directions = {(r["source"], r["target"]) for r in edge["annotations"]}
        edge["reciprocal"] = (a, b) in directions and (b, a) in directions
        for identity in edge["pair"]:
            if identity not in actors:
                warnings.append({"kind": "pair_partner_absent", "person": identity, "pair": edge["pair"]})
    return {"actors": actors, "membership_pairs": membership,
            "hhi_edges": list(edges.values()), "warnings": warnings}


def sequence_bounds(three):
    points = [np.array([[0., 0.]])]
    for rows in three.values():
        points.extend(jrdb.box_corners_xy(row["box"]) for row in rows)
    all_points = np.concatenate(points)
    if not np.isfinite(all_points).all():
        raise ValueError("Nonfinite 3D box coordinates")
    low, high = all_points.min(axis=0) - 2, all_points.max(axis=0) + 2
    center = (low + high) / 2
    span = max(24., math.ceil(float(max(high - low)) / 2) * 2)
    return (float(center[0] - span / 2), float(center[0] + span / 2),
            float(center[1] - span / 2), float(center[1] + span / 2))


def in_bounds(box, bounds):
    return bounds[0] <= box["cx"] <= bounds[1] and bounds[2] <= box["cy"] <= bounds[3]


def bev_xy(x, y, bounds, side):
    span = side - 124
    unit = span / (bounds[1] - bounds[0])
    return 62 + (bounds[3] - y) * unit, 56 + (bounds[1] - x) * unit


def dashed(draw, a, b, color, width=2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    if length < 1:
        return
    for start in np.arange(0, length, 12):
        end = min(start + 7, length)
        draw.line((tuple(a + (b-a) * start / length), tuple(a + (b-a) * end / length)),
                  fill=color, width=width)


class Labels:
    """Place compact target/edge labels without overlap where space permits."""

    def __init__(self, draw, side, top=48, bottom=None):
        self.draw, self.side = draw, side
        self.top, self.bottom = top, bottom or side - 64
        self.used = []

    def place(self, anchor, text, color, size=19, offset=0):
        face = font(size)
        bb = self.draw.textbbox((0, 0), text, font=face)
        w, h = bb[2] - bb[0] + 8, bb[3] - bb[1] + 8
        x, y = anchor
        candidates = [(x + 7, y - h - 3), (x + 7, y + 6), (x - w - 6, y - h - 3),
                      (x - w - 6, y + 6)]
        for n in range(1, 10):
            candidates.extend([(x + 7, y - (h + 3) * n), (x - w - 6, y + (h + 3) * n)])
        candidates = candidates[offset % 4:] + candidates[:offset % 4]
        best, best_score = None, math.inf
        for a, b in candidates:
            a, b = max(5, min(a, self.side - w - 5)), max(self.top, min(b, self.bottom - h))
            rect = (a, b, a + w, b + h)
            overlap = sum(max(0, min(rect[2], r[2]) - max(rect[0], r[0])) *
                          max(0, min(rect[3], r[3]) - max(rect[1], r[1])) for r in self.used)
            score = overlap * 100 + (a - x)**2 + (b - y)**2
            if score < best_score:
                best, best_score = rect, score
            if overlap == 0:
                break
        self.used.append(best)
        a, b, c, d = best
        if math.hypot(a - x, b - y) > 35:
            self.draw.line((anchor, (max(a, min(x, c)), max(b, min(y, d)))), fill=color, width=1)
        self.draw.rectangle(best, fill=BG)
        self.draw.text((a + 4, b + 4 - bb[1]), text, fill=color, font=face)


def draw_edges(draw, graph, centers, labels, relation_filter=None):
    actors = graph["actors"]
    for a, b in graph["membership_pairs"]:
        if a in centers and b in centers:
            color = group_color(actors[a]["group_id"])
            faint = tuple(round(.45 * c + .55 * BG[n]) for n, c in enumerate(color))
            draw.line((centers[a], centers[b]), fill=faint, width=2)
    deferred = []
    for edge in graph["hhi_edges"]:
        a, b = edge["pair"]
        if a not in centers or b not in centers:
            continue
        color = CROSS if edge["same_group"] is False else group_color(actors[a]["group_id"])
        if edge["same_group"] is False:
            dashed(draw, centers[a], centers[b], color, 4)
        else:
            draw.line((centers[a], centers[b]), fill=color, width=4)
        if relation_filter is None or relation_filter(edge):
            text = "+".join(SHORT.get(t, t) for t in edge["labels"])
            midpoint = tuple((np.asarray(centers[a]) + centers[b]) / 2)
            deferred.append((midpoint, text, color))
    return deferred


class Renderer:
    def __init__(self, root, sequence, side, calibration, bounds=None, lidar=True, local_span=24,
                 layout="front-near", legend_types=None):
        if layout not in ("front-near", "front-near-all"):
            raise ValueError("Unknown layout")
        if layout == "front-near-all" and bounds is None:
            raise ValueError("Global BEV layout requires sequence bounds")
        self.root, self.sequence, self.side = root, sequence, side
        self.calibration, self.bounds, self.lidar = calibration, bounds, lidar
        self.layout = layout
        self.legend_types = sorted(legend_types if legend_types is not None else
                                   [name for name, short in SHORT.items() if short in
                                    {"STAND", "TALK", "SIT", "EAT", "WALK", "APPROACH", "LOOK-ROBOT"}])
        self.local = (2 - local_span / 2, 2 + local_span / 2, -local_span / 2, local_span / 2)
        self.maps = None

    @property
    def width(self):
        return self.side * (3 if self.layout == "front-near-all" else 2)

    @property
    def height(self):
        return self.side + 64

    def front(self, frame, graph):
        pixels = cv2.imread(str(self.root / "images/image_0" / self.sequence / f"{frame}.jpg"))
        if pixels is None:
            raise ValueError(f"Cannot read front image {frame}")
        h, w = pixels.shape[:2]
        if self.maps is None:
            c = self.calibration
            self.maps = cv2.initUndistortRectifyMap(np.array(c["distorted_img_K"], float),
                np.array(c["D"], float), None, np.array(c["undistorted_img_K"], float), (w, h), cv2.CV_32FC1)
        pixels = cv2.cvtColor(cv2.remap(pixels, *self.maps, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        scale = min(self.side / w, (self.side - 112) / h)
        size = (round(w * scale), round(h * scale))
        offset = ((self.side - size[0]) // 2, (self.side - size[1]) // 2)
        image = Image.new("RGB", (self.side, self.side), BG)
        image.paste(Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS), offset)
        boxes, centers = {}, {}
        for identity, actor in graph["actors"].items():
            if actor["box_2d"] is None:
                continue
            box = t1.undistorted_box(actor["box_2d"], self.calibration, (w, h))
            if box is not None:
                boxes[identity] = [v * scale + offset[n % 2] for n, v in enumerate(box)]
                x1, y1, x2, y2 = boxes[identity]
                centers[identity] = ((x1+x2)/2, (y1+y2)/2)
        draw = ImageDraw.Draw(image)
        draw.text((18, 14), f"FRONT / camera 0    frame {frame}", font=font(25), fill=FG)
        labels = Labels(draw, self.side, top=50, bottom=self.side - 16)
        relations = draw_edges(draw, graph, centers, labels)
        for identity, box in boxes.items():
            color = group_color(graph["actors"][identity]["group_id"])
            draw.rectangle(box, outline=color, width=3)
            labels.place((box[0], box[1]), tag(identity, graph["actors"]), color, 22)
        for anchor, text, color in relations:
            labels.place(anchor, text, color, 19)
        return image, centers

    def bev(self, graph, visible, cloud, bounds, overview=False):
        side = self.side
        image = Image.new("RGB", (side, side), BG)
        span = bounds[1] - bounds[0]
        if cloud is not None:
            mask = ((cloud[:, 0] >= bounds[0]) & (cloud[:, 0] <= bounds[1]) &
                    (cloud[:, 1] >= bounds[2]) & (cloud[:, 1] <= bounds[3]) &
                    (cloud[:, 2] >= -2) & (cloud[:, 2] <= 2.5))
            xyz = cloud[mask][::2]
            if len(xyz):
                xy = np.rint(np.column_stack(bev_xy(xyz[:, 0], xyz[:, 1], bounds, side))).astype(int)
                array = np.asarray(image).copy()
                array[xy[:, 1], xy[:, 0]] = (61, 72, 83)
                image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        step = 10 if overview else 4
        for x in np.arange(math.ceil(bounds[0]/step)*step, bounds[1]+.001, step):
            a, b = bev_xy(x, bounds[2], bounds, side), bev_xy(x, bounds[3], bounds, side)
            draw.line((a, b), fill=(40, 49, 60), width=1)
            draw.text((a[0]+3, a[1]-18), f"x={x:g}", fill=(112, 126, 141), font=font(14))
        for y in np.arange(math.ceil(bounds[2]/step)*step, bounds[3]+.001, step):
            draw.line((bev_xy(bounds[0], y, bounds, side), bev_xy(bounds[1], y, bounds, side)),
                      fill=(40, 49, 60), width=1)
        draw.text((18, 14), f"BEV {'ALL / ' if overview else 'NEAR / '}{span:g}m   x forward / y left",
                  font=font(25), fill=FG)
        origin = bev_xy(0, 0, bounds, side)
        for sx in (0, 751):
            optical = np.linalg.inv(np.array(self.calibration["undistorted_img_K"], float)) @ [sx, 218, 1]
            ray = np.array(self.calibration["cam2ego"], float)[:3, :3] @ optical
            ray /= max(np.linalg.norm(ray[:2]), 1e-10)
            end = bev_xy(ray[0]*min(span*.32, 12), ray[1]*min(span*.32, 12), bounds, side)
            dashed(draw, origin, end, (151, 160, 177))
        draw.ellipse((origin[0]-6, origin[1]-6, origin[0]+6, origin[1]+6), fill=FG)
        draw.text((origin[0]+9, origin[1]-15), "R", font=font(18), fill=FG)
        if overview:
            corners = [bev_xy(x, y, bounds, side) for x, y in
                       [(self.local[0], self.local[2]), (self.local[1], self.local[3])]]
            rect = (min(p[0] for p in corners), min(p[1] for p in corners),
                    max(p[0] for p in corners), max(p[1] for p in corners))
            draw.rectangle(rect, outline=(142, 155, 167), width=2)
        centers = {i: bev_xy(a["box_3d"]["cx"], a["box_3d"]["cy"], bounds, side)
                   for i, a in graph["actors"].items() if a["box_3d"] and in_bounds(a["box_3d"], bounds)}
        labels = Labels(draw, side)
        def label_relation(edge):
            return not overview or any(not in_bounds(graph["actors"][i]["box_3d"], self.local)
                                       for i in edge["pair"])
        relations = draw_edges(draw, graph, centers, labels, label_relation)
        for identity, center in centers.items():
            actor = graph["actors"][identity]
            color = group_color(actor["group_id"])
            corners = [bev_xy(x, y, bounds, side) for x, y in jrdb.box_corners_xy(actor["box_3d"])]
            draw.line(corners, fill=color, width=2)
            x, y = center
            draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=2,
                         fill=color if identity in visible else BG)
            # Near actors are labeled in the larger local plot. Avoid duplicate
            # overlapping text in the global plot; all boxes/edges remain drawn.
            if not overview or not in_bounds(actor["box_3d"], self.local):
                labels.place(center, tag(identity, graph["actors"]), color, 19)
        for anchor, text, color in relations:
            labels.place(anchor, text, color, 17)
        if overview:
            missing = [tag(i, graph["actors"]) for i, a in graph["actors"].items() if a["box_3d"] is None]
            if missing:
                text = "NO 3D: " + "  ".join(missing)
                draw.text((18, side-49), text, font=font(17), fill=(255, 204, 109))
        else:
            draw.text((18, side-54), "Filled = in front RGB; hollow = outside front RGB", font=font(19), fill=FG)
            if self.layout == "front-near":
                outside = sum(a["box_3d"] is not None and not in_bounds(a["box_3d"], bounds)
                              for a in graph["actors"].values())
                missing = sum(a["box_3d"] is None for a in graph["actors"].values())
                draw.text((18, side-28), f"Outside BEV: {outside}    NO 3D: {missing}    (IDs in JSON)",
                          font=font(18), fill=(255, 204, 109))
        return image

    def render(self, frame, graph):
        front, visible = self.front(frame, graph)
        cloud = None
        if self.lidar:
            cloud = np.concatenate(jrdb.load_clouds_in_ego(self.root, self.sequence, frame))
        near = self.bev(graph, visible, cloud, self.local)
        panels = [front, near]
        if self.layout == "front-near-all":
            panels.append(self.bev(graph, visible, cloud, self.bounds, True))
        canvas = Image.new("RGB", (self.width, self.height), BG)
        for n, panel in enumerate(panels):
            canvas.paste(panel, (self.side*n, 0))
        draw = ImageDraw.Draw(canvas)
        footer = [
            "P=person ID  G=group ID  !=source conflict | thin=same group | labeled=HHI | red dashed=cross-group HHI | JRDB REFERENCE / NOT T1 GOLD",
            "   ".join(f"{SHORT.get(name, name)}={name}" for name in self.legend_types),
        ]
        for n, text in enumerate(footer):
            size = 20 if n == 0 else 17
            while draw.textlength(text, font=font(size)) > canvas.width-36 and size > 9:
                size -= 1
            draw.text((18, self.side+4+n*29), text, font=font(size), fill=FG)
        graph["visible_front_ids"] = sorted(visible)
        graph["outside_local_bev_ids"] = sorted(i for i, a in graph["actors"].items()
                                                if a["box_3d"] and not in_bounds(a["box_3d"], self.local))
        drawn_bounds = self.bounds if self.layout == "front-near-all" else self.local
        graph["drawn_bev_ids"] = sorted(i for i, a in graph["actors"].items()
                                        if a["box_3d"] and in_bounds(a["box_3d"], drawn_bounds))
        graph["outside_rendered_bev_ids"] = sorted(i for i, a in graph["actors"].items()
                                                   if a["box_3d"] and i not in graph["drawn_bev_ids"])
        graph["hhi_without_3d"] = [e["pair"] for e in graph["hhi_edges"]
                                   if any(not graph["actors"].get(i, {}).get("box_3d") for i in e["pair"])]
        graph["hhi_outside_rendered_bev"] = [e["pair"] for e in graph["hhi_edges"]
                                            if all(graph["actors"].get(i, {}).get("box_3d") for i in e["pair"])
                                            and any(i not in graph["drawn_bev_ids"] for i in e["pair"])]
        return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=jrdb.DEFAULT_ROOT)
    parser.add_argument("--sequence", default="bytes-cafe-2019-02-07_0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15, help="Nominal source FPS, not measured timestamps")
    parser.add_argument("--side", type=int, default=960, help="Panel side; height=side+64")
    parser.add_argument("--layout", choices=("front-near", "front-near-all"), default="front-near",
                        help="Default front RGB + local BEV; opt in to the legacy third global panel")
    parser.add_argument("--local-span", type=float, default=24)
    parser.add_argument("--start-frame")
    parser.add_argument("--end-frame")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--snapshots", default="auto",
                        help="auto: first/one-third/two-thirds/last; or comma-separated source frame IDs")
    args = parser.parse_args()
    jrdb.validate_id(args.sequence, args.start_frame)
    if args.end_frame:
        jrdb.validate_id(args.sequence, args.end_frame)
    if args.fps <= 0 or args.side < 640 or args.side % 2 or not math.isfinite(args.local_span) or args.local_span <= 0:
        parser.error("FPS/span must be positive and side must be even and at least 640")
    t1.check_output(args.root, args.out)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty or new")
    paths = {
        "stitched": args.root / "labels/labels_2d_activity_social_stitched" / f"{args.sequence}.json",
        "front": args.root / "labels/labels_2d_activity_social" / f"{args.sequence}_image0.json",
        "3d": args.root / "labels/labels_3d" / f"{args.sequence}.json",
    }
    data = {k: jrdb.load_json(p)["labels"] for k, p in paths.items()}
    frames = sorted(p.stem for p in (args.root / "images/image_0" / args.sequence).glob("*.jpg")
                    if (args.start_frame is None or p.stem >= args.start_frame)
                    and (args.end_frame is None or p.stem <= args.end_frame))
    if not frames or any(int(b)-int(a) != 1 for a, b in zip(frames, frames[1:])):
        parser.error("Require a nonempty consecutive real-frame range; will not duplicate/interpolate frames")
    for frame in frames:
        for source, labels in data.items():
            key = frame + (".pcd" if source == "3d" else ".jpg")
            if key not in labels:
                parser.error(f"Missing entire annotation frame: {source}/{key}")
    calibration = yaml.safe_load((args.root / "calibration/lidars.yaml").read_text())["sensor_0"]
    bounds = (sequence_bounds({f+".pcd": data["3d"][f+".pcd"] for f in frames})
              if args.layout == "front-near-all" else None)
    legend_types = {name for frame in frames for row in data["stitched"][frame+".jpg"]
                    for interaction in row.get("H-interaction", [])
                    for name in (interaction.get("inter_labels") or {})}
    renderer = Renderer(args.root, args.sequence, args.side, calibration, bounds,
                        not args.no_lidar, args.local_span, args.layout, legend_types)
    cv2.setNumThreads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    snapshots = ({frames[round((len(frames)-1)*fraction)] for fraction in (0, 1/3, 2/3, 1)}
                 if args.snapshots == "auto" else set(args.snapshots.split(",")))
    stats, types = Counter(), Counter()
    started = time.monotonic()
    partial = args.out / "group_relations.partial.mp4"
    def images(audit):
        for n, frame in enumerate(frames):
            graph = frame_graph(data["stitched"][frame+".jpg"], data["front"][frame+".jpg"], data["3d"][frame+".pcd"])
            image = renderer.render(frame, graph)
            graph["frame_id"] = frame
            graph["nominal_seconds"] = int(frame)/args.fps
            stats["actor_frames"] += len(graph["actors"])
            stats["bev_actor_frames"] += len(graph["drawn_bev_ids"])
            stats["off_front_bev_actor_frames"] += len(set(graph["drawn_bev_ids"])-set(graph["visible_front_ids"]))
            stats["outside_rendered_bev_actor_frames"] += len(graph["outside_rendered_bev_ids"])
            stats["membership_pair_frames"] += len(graph["membership_pairs"])
            stats["hhi_pair_frames"] += len(graph["hhi_edges"])
            stats["cross_group_hhi_pair_frames"] += sum(e["same_group"] is False for e in graph["hhi_edges"])
            stats["hhi_pair_frames_without_3d"] += len(graph["hhi_without_3d"])
            stats["hhi_pair_frames_outside_rendered_bev"] += len(graph["hhi_outside_rendered_bev"])
            for warning in graph["warnings"]:
                stats[warning["kind"]] += 1
            for edge in graph["hhi_edges"]:
                types.update(edge["labels"])
            audit.write(json.dumps(graph, ensure_ascii=False, allow_nan=False) + "\n")
            if frame in snapshots:
                image.save(args.out / f"frame_{frame}.png")
            if n % 100 == 0 or n == len(frames)-1:
                elapsed = time.monotonic()-started
                print(f"Rendered {n+1}/{len(frames)} frame={frame} elapsed={elapsed:.1f}s", flush=True)
            yield image
    with (args.out / "frame_annotations.jsonl").open("w", encoding="utf-8") as audit:
        count = jrdb.encode_mp4(partial, images(audit), renderer.width, renderer.height, Fraction(args.fps))
    video = args.out / "group_relations.mp4"
    partial.rename(video)
    manifest = {
        "sequence": args.sequence, "root": str(args.root.resolve()), "video": video.name,
        "num_frames": count, "source_frame_ids": frames, "fps": args.fps,
        "time_source": "nominal_frame_index", "duration_seconds": count/args.fps,
        "width": renderer.width, "height": renderer.height, "layout": args.layout,
        "local_bounds_xy": renderer.local,
        "all_bounds_xy": bounds, "lidar": not args.no_lidar, "stats": dict(stats),
        "legend_types": sorted(legend_types),
        "hhi_types_pair_frames": dict(types), "hhi_short_names": SHORT,
        "sources": {k: str(p.resolve()) for k, p in paths.items()},
        "snapshot_frames": sorted(snapshots & set(frames)),
        "notes": ["Outer JSON frame keys are authoritative; do not shift by inner file_id.",
                  "Stitched group/HHI labels take priority; group conflicts and absent 3D recorded per frame.",
                  "Membership edges and raw HHI are separate, not inferred T1 semantic labels.",
                  "All actor/HHI records retained in JSON, including no_eval/interpolated and outside front/local BEV.",
                  "In front-near mode, only local-range actors are drawn in BEV; outside IDs/pairs recorded separately.",
                  "Missing 3D and outside-range actors are distinguished; no location is invented.",
                  "Optional global BEV uses fixed sequence-wide geometry bounds; not a causal VLM input.",
                  "Points calibrated to ego; original 3D boxes retained. Inspect geometric alignment manually.",
                  "No ego-compensated trajectories or inferred walking velocities are rendered."]}
    t1.write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"video": str(video), "num_frames": count, "fps": args.fps,
                      "stats": dict(stats), "elapsed_seconds": round(time.monotonic()-started, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
