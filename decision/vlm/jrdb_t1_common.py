"""Geometry-only JRDB T1 input rendering, shared with the label-only viewer.

The input path deliberately discards social/action fields. Source identities and
oracle association are audit metadata, never part of the VLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

try:
    from decision.vlm import jrdb_visualize as jrdb
except ModuleNotFoundError:
    import jrdb_visualize as jrdb


PROTOCOL = "jrdb_t1_p0_local"
NEUTRAL = (70, 222, 239)
UNKNOWN = (255, 186, 70)
BACKGROUND = (18, 24, 32)
RELATIONS = {"w": "walk", "t": "talk", "q": "queue", "p": "photo", "o": "other", "u": "rel?"}
PALETTE = [(255, 119, 119), (136, 216, 110), (183, 148, 255),
           (255, 207, 97), (103, 169, 255), (246, 153, 224), (107, 223, 183)]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def check_output(root: Path | None, out: Path, *, input_dirs=()) -> None:
    destination = out.resolve()
    if root is not None and destination.is_relative_to(root.resolve()):
        raise ValueError("Outputs must be outside the original JRDB dataset directory")
    if any(destination.is_relative_to(Path(folder).resolve()) for folder in input_dirs):
        raise ValueError("Outputs must be separate from model inputs")


def geometry_rows(rows: list[dict], dimensions: int) -> dict[str, dict]:
    """Whitelist geometry; semantic fields must not survive this boundary."""
    result = {}
    for row in rows:
        identity = row["label_id"]
        if identity in result:
            raise ValueError(f"Duplicate identity in one frame: {identity}")
        if dimensions == 2:
            box = [float(n) for n in row["box"]]
            if len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) <= 0:
                raise ValueError(f"Invalid 2D box for {identity}")
            attributes = row.get("attributes", {})
            result[identity] = {"box": box, "occlusion": attributes.get("occlusion"),
                                "truncated": attributes.get("truncated")}
        else:
            box = {key: float(row["box"][key])
                   for key in ("cx", "cy", "cz", "l", "w", "h", "rot_z")}
            if not np.isfinite(list(box.values())).all() or min(box[k] for k in ("l", "w", "h")) <= 0:
                raise ValueError(f"Invalid 3D box for {identity}")
            result[identity] = {"box": box}
    return dict(sorted(result.items()))


def inside_bev(box: dict) -> bool:
    return -2 <= box["cx"] <= 10 and -6 <= box["cy"] <= 6


def frame_times(frames: list[str], fps: float, timestamps: dict | None = None) -> list[float]:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("FPS must be finite and positive")
    times = [float(timestamps[f]) if timestamps is not None else int(f) / fps for f in frames]
    if not np.isfinite(times).all() or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Three distinct frames must have strictly increasing finite timestamps")
    return times


def choose_frames(available: list[str], end: str, fps: float, timestamps: dict | None = None) -> list[str]:
    jrdb.validate_id("placeholder", end)
    if end not in available:
        raise ValueError(f"Target frame does not exist: {end}")
    # Restrict the entire selection to past/current frames; never seek forward.
    prior = [f for f in available if int(f) <= int(end)]
    times = frame_times(prior, fps, timestamps)
    target = times[-1]
    chosen = []
    for lag in (1.0, 0.5, 0.0):
        index = min(range(len(times)), key=lambda i: abs(times[i] - (target - lag)))
        if abs(times[index] - (target - lag)) > max(0.04, 0.6 / fps):
            raise ValueError("Insufficient real past frames for a one-second three-frame window")
        chosen.append(prior[index])
    frame_times(chosen, fps, timestamps)
    return chosen


def read_poses(path: Path | None, frames: list[str]) -> list[np.ndarray] | None:
    if path is None:
        return None
    data = jrdb.load_json(path)
    if data.get("convention") != "world_from_ego":
        raise ValueError("Pose file must declare convention=world_from_ego (metres, x/y/z ego axes)")
    poses = []
    for frame in frames:
        matrix = np.asarray(data["poses"][frame], dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
            raise ValueError(f"Invalid rigid world_from_ego pose for frame {frame}")
        poses.append(matrix)
    return poses


def transform_position(box: dict, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    point = np.array([box["cx"], box["cy"], box["cz"], 1.0])
    return (np.linalg.inv(current) @ previous @ point)[:2]


@dataclass
class Window:
    root: Path
    sequence: str
    frames: list[str]
    times: list[float]
    nominal_fps: float
    time_source: str
    scope: str
    width: int
    height: int
    lidar: bool
    two: list[dict]
    three: list[dict]
    pid_map: dict[str, int]
    active: list[str]
    calibration: dict
    poses: list[np.ndarray] | None
    pose_path: Path | None

    @property
    def valid_pid(self) -> list[int]:
        return sorted(self.pid_map[identity] for identity in self.active)

    @property
    def geometry_signature(self) -> str:
        return digest({"frames": self.frames, "times": self.times, "two": self.two,
                       "three": self.three, "calibration": self.calibration,
                       "poses": None if self.poses is None else [p.tolist() for p in self.poses],
                       "scope": self.scope, "pid_map": self.pid_map,
                       "size": [self.width, self.height], "lidar": self.lidar})

    def tracks(self) -> list[dict]:
        result = []
        for identity in self.active:
            box = self.three[-1].get(identity, {}).get("box")
            previous = [i for i in range(2) if identity in self.three[i]]
            history = []
            velocity = None
            if self.poses is not None and box is not None:
                for i in previous:
                    xy = transform_position(self.three[i][identity]["box"], self.poses[i], self.poses[-1])
                    history.append([round(self.times[i] - self.times[-1], 2), *np.round(xy, 2).tolist()])
                if previous:
                    i = previous[-1]
                    xy = transform_position(self.three[i][identity]["box"], self.poses[i], self.poses[-1])
                    velocity = np.round((np.array([box["cx"], box["cy"]]) - xy)
                                        / (self.times[-1] - self.times[i]), 2).tolist()
            seen = [i for i in range(3) if identity in self.two[i] or identity in self.three[i]]
            result.append({"pid": self.pid_map[identity],
                           "pos": None if box is None else [round(box["cx"], 2), round(box["cy"], 2)],
                           "vel": velocity, "cov": None,
                           "age": round(self.times[-1] - self.times[min(seen)], 2),
                           "visible": int(identity in self.two[-1]), "lifecycle": "active",
                           "history": history})
        return sorted(result, key=lambda row: row["pid"])


def build_window(root: Path, sequence: str, frames: list[str], *, fps=15.0,
                 timestamps=None, scope="front", width=896, height=448,
                 lidar=True, pose_path: Path | None = None) -> Window:
    if len(frames) != 3:
        raise ValueError("T1 requires exactly three real frames")
    for frame in frames:
        jrdb.validate_id(sequence, frame)
    if any(int(a) >= int(b) for a, b in zip(frames, frames[1:])):
        raise ValueError("Frame IDs must be strictly increasing")
    if scope not in ("front", "all-bev"):
        raise ValueError("Scope must be front or all-bev")
    if width < 640 or height < 320 or width != 2 * height:
        raise ValueError("Canvas must have two square panels: width=2*height, height>=320")
    two_data = jrdb.load_json(jrdb.label_path(root, "people", sequence, "0"))["labels"]
    three_data = jrdb.load_json(root / "labels/labels_3d" / f"{sequence}.json")["labels"]
    # Outer dictionary keys are authoritative: inner file_id can be offset by +2.
    two = [geometry_rows(two_data[f"{f}.jpg"], 2) for f in frames]
    three = [geometry_rows(three_data[f"{f}.pcd"], 3) for f in frames]
    for f in frames:
        paths = [root / "images/image_0" / sequence / f"{f}.jpg"]
        if lidar:
            paths += [root / "pointclouds" / layer / sequence / f"{f}.pcd"
                      for layer in ("lower_velodyne", "upper_velodyne")]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    active = set(two[-1])
    if scope == "all-bev":
        active.update(identity for identity, row in three[-1].items() if inside_bev(row["box"]))
    if len(active) > 20:
        raise ValueError(f"input_overflow: {len(active)} active persons (>20); no silent truncation")
    # Causal first-appearance order, independent of all social group annotations.
    pid_map = {}
    for i in range(3):
        candidates = set(two[i])
        if scope == "all-bev":
            candidates.update(identity for identity, row in three[i].items() if inside_bev(row["box"]))
        for identity in sorted(candidates):
            if identity not in pid_map:
                pid_map[identity] = len(pid_map)
    if len(pid_map) > 100:
        raise ValueError("input_overflow: window requires more than 100 distinct short PIDs")
    calibration_path = root / "calibration/lidars.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))["sensor_0"]
    poses = read_poses(pose_path, frames)
    return Window(root.absolute(), sequence, frames, frame_times(frames, fps, timestamps), fps,
                  "provided_seconds" if timestamps is not None else "nominal_frame_index",
                  scope, width, height, lidar, two, three, pid_map, sorted(active), calibration,
                  poses, None if pose_path is None else pose_path.resolve())


def font(size: int):
    try:
        return ImageFont.truetype(jrdb.FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def badge(draw: ImageDraw.ImageDraw, xy: tuple, text: str, color=NEUTRAL, size=16) -> None:
    typeface = font(size)
    x, y = xy
    bounds = draw.textbbox((x, y), text, font=typeface)
    draw.rectangle((bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2), fill=BACKGROUND)
    draw.text((x, y), text, fill=color, font=typeface)


def undistorted_box(box: list, calibration: dict, image_size: tuple[int, int]) -> list | None:
    x, y, width, height = box
    # Sample all edges, not just corners: distortion can curve the box boundary.
    fraction = np.linspace(0, 1, 17)
    points = np.vstack((np.column_stack((x + width * fraction, np.full(17, y))),
                        np.column_stack((x + width * fraction, np.full(17, y + height))),
                        np.column_stack((np.full(17, x), y + height * fraction)),
                        np.column_stack((np.full(17, x + width), y + height * fraction))))
    points = cv2.undistortPoints(points.reshape(-1, 1, 2),
                                 np.asarray(calibration["distorted_img_K"], float),
                                 np.asarray(calibration["D"], float),
                                 P=np.asarray(calibration["undistorted_img_K"], float)).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite undistorted box coordinates")
    lo = np.maximum(points.min(axis=0), 0)
    hi = np.minimum(points.max(axis=0), np.array(image_size) - 1)
    return None if np.any(hi <= lo) else [*lo.tolist(), *hi.tolist()]


def rgb_panel(window: Window, index: int, styles: dict | None = None) -> Image.Image:
    image_path = window.root / "images/image_0" / window.sequence / f"{window.frames[index]}.jpg"
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("RGB"))
    calibration = window.calibration
    pixels = cv2.undistort(pixels, np.asarray(calibration["distorted_img_K"], float),
                          np.asarray(calibration["D"], float), None,
                          np.asarray(calibration["undistorted_img_K"], float))
    original_size = (pixels.shape[1], pixels.shape[0])
    side = window.height
    scale = min(side / original_size[0], side / original_size[1])
    resized = (round(original_size[0] * scale), round(original_size[1] * scale))
    offset = ((side - resized[0]) // 2, (side - resized[1]) // 2)
    canvas = Image.new("RGB", (side, side), BACKGROUND)
    canvas.paste(Image.fromarray(pixels).resize(resized, Image.Resampling.LANCZOS), offset)
    draw = ImageDraw.Draw(canvas)
    for identity, row in window.two[index].items():
        if identity not in window.pid_map:
            continue
        pid = window.pid_map[identity]
        box = undistorted_box(row["box"], calibration, original_size)
        if box is None:
            continue
        x1, y1, x2, y2 = [v * scale + offset[j % 2] for j, v in enumerate(box)]
        color, suffix = (styles or {}).get(pid, (NEUTRAL, ""))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = f"{pid}{suffix}"
        label_width = draw.textlength(label, font=font(16))
        badge(draw, (max(4, min(x1, side - label_width - 5)), max(offset[1], y1 - 21)), label, color)
    lag = window.times[index] - window.times[-1]
    badge(draw, (8, 8), f"RGB  t{lag:+.2f}s", (219, 229, 238), 15)
    return canvas


def bev_xy(x: float, y: float, side: int) -> tuple[float, float]:
    margin = 22
    unit = (side - 2 * margin) / 12
    return margin + (6 - y) * unit, margin + (10 - x) * unit


def arrow(draw: ImageDraw.ImageDraw, first, second, color, width=2) -> None:
    draw.line((first, second), fill=color, width=width)
    delta = np.asarray(second, float) - np.asarray(first, float)
    length = np.linalg.norm(delta)
    if length < 3:
        return
    direction = delta / length
    perpendicular = np.array([-direction[1], direction[0]])
    tip = np.asarray(second, float)
    base = tip - direction * min(9, length / 3)
    draw.polygon([tuple(tip), tuple(base + perpendicular * 4), tuple(base - perpendicular * 4)], fill=color)


def relation_layout(response: dict) -> tuple[dict, list[dict]]:
    styles = {}
    links = []
    for n, group in enumerate(response["g"]):
        color = PALETTE[n % len(PALETTE)]
        relation = RELATIONS[group["r"]]
        for i, pid in enumerate(group["m"]):
            role = ""
            if group["r"] == "q" and group["k"]:
                role = f" #{i + 1}"
            elif group["r"] == "p" and group["k"]:
                role = " cam" if i == 0 else " subj" if i == 1 else ""
            styles[pid] = (color, f" G{n}:{relation}{role}")
        links.append({"members": group["m"], "color": color,
                      "label": f"G{n}:{relation}", "directed": bool(group["k"]),
                      "relation": group["r"]})
    for pid in response["u"]:
        styles[pid] = (UNKNOWN, " ?")
    return styles, links


def bev_panel(window: Window, index: int, styles: dict | None = None,
              links: list[dict] | None = None) -> Image.Image:
    side = window.height
    canvas = Image.new("RGB", (side, side), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for x in range(-2, 11, 2):
        draw.line((bev_xy(x, -6, side), bev_xy(x, 6, side)), fill=(42, 51, 63))
    for y in range(-6, 7, 2):
        draw.line((bev_xy(-2, y, side), bev_xy(10, y, side)), fill=(42, 51, 63))
    if window.lidar:
        lower, upper = jrdb.load_clouds_in_ego(window.root, window.sequence, window.frames[index])
        points = np.vstack((lower, upper))
        keep = (np.isfinite(points).all(axis=1) & (points[:, 0] >= -2) & (points[:, 0] <= 10)
                & (np.abs(points[:, 1]) <= 6) & (points[:, 2] >= -2) & (points[:, 2] <= 2.5))
        points = points[keep]
        if len(points):
            unit = (side - 44) / 12
            pixels = np.rint(np.column_stack((22 + (6 - points[:, 1]) * unit,
                                              22 + (10 - points[:, 0]) * unit))).astype(int)
            buffer = np.asarray(canvas).copy()
            buffer[pixels[:, 1], pixels[:, 0]] = (92, 103, 117)
            canvas = Image.fromarray(buffer)
            draw = ImageDraw.Draw(canvas)
    # Camera FOV rays from calibrated optical rays. This is not a free-space cone.
    K = np.asarray(window.calibration["undistorted_img_K"], float)
    cam2ego = np.asarray(window.calibration["cam2ego"], float)
    with Image.open(window.root / "images/image_0" / window.sequence / f"{window.frames[index]}.jpg") as source:
        source_width = source.width
    for horizontal in (0, source_width - 1):
        ray = cam2ego[:3, :3] @ (np.linalg.inv(K) @ np.array([horizontal, K[1, 2], 1]))
        ray /= np.linalg.norm(ray[:2])
        start = cam2ego[:2, 3]
        # Draw short ray segments, clipping at the displayed range.
        for distance in np.arange(0, 12, 0.25):
            a, b = start + ray[:2] * distance, start + ray[:2] * (distance + 0.15)
            if -2 <= a[0] <= 10 and abs(a[1]) <= 6 and -2 <= b[0] <= 10 and abs(b[1]) <= 6:
                draw.line((bev_xy(*a, side), bev_xy(*b, side)), fill=(62, 89, 108))
    # This is a virtual marker, not a claim about JRDB robot/chair footprint.
    robot = [bev_xy(x, y, side) for x, y in ((-0.4, -0.3), (-0.4, 0.3), (0.4, 0.3), (0.4, -0.3))]
    draw.polygon(robot, fill=(189, 199, 210), outline=(245, 247, 250))
    arrow(draw, bev_xy(0, 0, side), bev_xy(0.9, 0, side), (245, 247, 250))
    centers = {}
    for identity, row in window.three[index].items():
        if identity not in window.pid_map or not inside_bev(row["box"]):
            continue
        pid = window.pid_map[identity]
        box = row["box"]
        centers[pid] = bev_xy(box["cx"], box["cy"], side)
        color, suffix = (styles or {}).get(pid, (NEUTRAL, ""))
        if window.poses is not None:
            history = [transform_position(window.three[j][identity]["box"], window.poses[j], window.poses[index])
                       for j in range(index) if identity in window.three[j]]
            path = [bev_xy(*p, side) for p in history if -2 <= p[0] <= 10 and abs(p[1]) <= 6]
            if path:
                draw.line(path + [centers[pid]], fill=color, width=1)
                # Backward-looking finite difference after ego-motion compensation.
                j = max(j for j in range(index) if identity in window.three[j])
                delta = (np.array([box["cx"], box["cy"]]) - history[-1]) / (window.times[index] - window.times[j])
                end = np.array([box["cx"], box["cy"]]) + delta * 0.5
                if -2 <= end[0] <= 10 and abs(end[1]) <= 6:
                    arrow(draw, centers[pid], bev_xy(*end, side), color, 1)
        corners = jrdb.box_corners_xy(box)
        draw.line([bev_xy(float(x), float(y), side) for x, y in corners], fill=color, width=2)
        cx, cy = centers[pid]
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)
    for link in links or []:
        members = [pid for pid in link["members"] if pid in centers]
        if link["relation"] == "p" and link["directed"]:
            # Only the first two roles are defined; extra members are not
            # automatically labelled as photographed subjects.
            first, second = link["members"][:2]
            pairs = [(first, second)] if first in centers and second in centers else []
        else:
            # Do not bridge a missing/cropped member and invent an adjacency.
            pairs = [(a, b) for a, b in zip(link["members"], link["members"][1:])
                     if a in centers and b in centers]
        for first, second in pairs:
            if link["directed"]:
                arrow(draw, centers[first], centers[second], link["color"], 3)
            else:
                draw.line((centers[first], centers[second]), fill=link["color"], width=3)
        if len(members) >= 2:
            midpoint = np.asarray([centers[pid] for pid in members]).mean(axis=0)
            label_width = draw.textlength(link["label"], font=font(14))
            badge(draw, (max(4, min(midpoint[0] + 10, side - label_width - 4)),
                         max(28, min(midpoint[1] + 13, side - 23))), link["label"], link["color"], 14)
    for pid, (cx, cy) in centers.items():
        color, suffix = (styles or {}).get(pid, (NEUTRAL, ""))
        # Group/role names are in RGB; keep BEV labels compact around dense crowds.
        short_suffix = " ?" if suffix == " ?" else ""
        if " #" in suffix:
            short_suffix = " #" + suffix.rsplit(" #", 1)[1]
        elif suffix.endswith(" cam"):
            short_suffix = " cam"
        elif suffix.endswith(" subj"):
            short_suffix = " subj"
        badge(draw, (min(side - 34, max(4, cx + 4)), max(28, cy - 20)), f"{pid}{short_suffix}", color, 16)
    badge(draw, (8, 8), "BEV  x^ y<  12m", (219, 229, 238), 15)
    return canvas


def render_frame(window: Window, index: int, styles=None, links=None) -> Image.Image:
    canvas = Image.new("RGB", (window.width, window.height), BACKGROUND)
    canvas.paste(rgb_panel(window, index, styles), (0, 0))
    canvas.paste(bev_panel(window, index, styles, links), (window.height, 0))
    return canvas


def model_payload(window: Window) -> dict:
    return {"SCHEMA": "social_nav_prompt", "MODE": "REL", "INPUT_PROTOCOL": PROTOCOL,
            "NOW_TS": round(window.times[-1], 6), "FRAME_TS": [round(t, 6) for t in window.times],
            "TIME_SOURCE": window.time_source, "VALID_PID": window.valid_pid,
            "TRACKS": window.tracks(), "DECISION_HISTORY": [], "ROUTE_CONTEXT": None,
            "SCOPE": window.scope,
            "MOTION_AVAILABLE": window.poses is not None,
            "BEV_CROPPED_PID": sorted(window.pid_map[i] for i in window.active
                                      if i in window.three[-1] and not inside_bev(window.three[-1][i]["box"]))}


def system_prompt() -> str:
    path = Path(__file__).resolve().parent / "configs/prompts/jrdb_t1_p0_rel.txt"
    return path.read_text(encoding="utf-8").strip()


def user_prompt(payload: dict) -> str:
    return ("按 social_nav_prompt 的 REL 规则判断 IMAGE_2 时刻；仅返回含 g、u 的 JSON。\n"
            "三张真实图按时间升序；实际时间见 FRAME_TS，未必恰好相隔 0.50s。\n"
            "每张 BEV 均以该帧机器人为原点，x 前/y 左；TRACKS 在末帧机器人坐标。\n"
            "青色框和编号只表示几何与关联，不表示社会群组。BEV 点云不是可通行地图。\n"
            "front 仅评估末帧前向相机标注人员；all-bev 另含 BEV 内的视场外人员。\n"
            "vel/cov=null、history=[] 表示信息缺失，不表示静止或零误差。"
            "age 只是本窗口内的观测跨度。FRAME_TS 为 nominal_frame_index 时不是传感器时间戳。\n"
            "只引用 VALID_PID；历史帧独有编号不属于本次输出。"
            "视场外/证据不足允许归属未知 u，已知成员但关系不清用 r=u。\n"
            + json.dumps(payload, ensure_ascii=False, allow_nan=False))


def save_input_bundle(window: Window, out: Path) -> dict:
    check_output(window.root, out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new --out directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    image_hashes = {}
    for i in range(3):
        name = f"image_{i}.png"
        render_frame(window, i).save(out / name)
        image_hashes[name] = file_digest(out / name)
    payload = model_payload(window)
    prompt = user_prompt(payload)
    system = system_prompt()
    (out / "user_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (out / "system_prompt.txt").write_text(system + "\n", encoding="utf-8")
    # Three separate model images; contact sheet is for human inspection only.
    strip = Image.new("RGB", (window.width, 3 * window.height), BACKGROUND)
    for i in range(3):
        with Image.open(out / f"image_{i}.png") as image:
            strip.paste(image, (0, i * window.height))
    strip.save(out / "input_contact_sheet.png")
    fingerprint = digest({"images": image_hashes, "payload": payload, "prompt": prompt, "system": system})
    source_assets = ["calibration/lidars.yaml"]
    for frame in window.frames:
        source_assets.append(f"images/image_0/{window.sequence}/{frame}.jpg")
        if window.lidar:
            source_assets.extend(f"pointclouds/{layer}/{window.sequence}/{frame}.pcd"
                                 for layer in ("lower_velodyne", "upper_velodyne"))
    manifest = {"protocol": PROTOCOL, "input_fingerprint": fingerprint,
                "root": str(window.root), "sequence": window.sequence, "frame_ids": window.frames,
                "timestamps": window.times, "time_source": window.time_source,
                "nominal_fps": window.nominal_fps, "scope": window.scope,
                "width": window.width, "height": window.height, "lidar": window.lidar,
                "ego_poses": None if window.pose_path is None else str(window.pose_path),
                "perception_tier": "P0_oracle_boxes_and_association",
                "geometry_signature": window.geometry_signature,
                "pid_to_source_id": {str(pid): identity for identity, pid in window.pid_map.items()},
                "source_assets_sha256": {name: file_digest(window.root / name) for name in source_assets},
                "image_sha256": image_hashes, "model": payload,
                "quality": {"missing_3d_pid": sorted(window.pid_map[i] for i in window.active if i not in window.three[-1]),
                            "bev_cropped_pid": payload["BEV_CROPPED_PID"],
                            "historical_only_pid": sorted(set(window.pid_map.values()) - set(window.valid_pid)),
                            "excluded_current_3d_by_scope": sorted(set(window.three[-1]) - set(window.active)),
                            "motion_available": window.poses is not None,
                            "main_protocol_exact": False,
                            "notes": ["Front-scope and nominal timestamps are explicit public-pilot variants.",
                                      "JRDB boxes/association are oracle; no real detector/tracker is evaluated.",
                                      "No occupancy/free-space, route, covariance or full track lifetime is inferred.",
                                      "Raw JRDB 3D box coordinates are used as ego; cross-sensor alignment requires scene review."]}}
    write_json(out / "manifest.json", manifest)
    write_json(out / "model_input.json", {"images": list(image_hashes), "fields": payload,
                                         "system_prompt": system, "user_prompt": prompt})
    write_json(out / "annotation_template.json", {
        "input_fingerprint": fingerprint, "review_status": "draft", "reviewer": None,
        "reviewed_at": None, "label_provenance": "manual_review_of_model_input_required",
        "response": {"g": [], "u": window.valid_pid},
        "note": "All-unknown is an unreviewed placeholder, NOT a training/test ground truth."})
    return manifest


def load_bundle(out: Path) -> tuple[Window, dict]:
    manifest = jrdb.load_json(out / "manifest.json")
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unsupported input protocol")
    timestamps = dict(zip(manifest["frame_ids"], manifest["timestamps"]))
    window = build_window(Path(manifest["root"]), manifest["sequence"], manifest["frame_ids"],
                          fps=manifest["nominal_fps"], timestamps=timestamps,
                          scope=manifest["scope"], width=manifest["width"], height=manifest["height"],
                          lidar=manifest["lidar"],
                          pose_path=None if manifest["ego_poses"] is None else Path(manifest["ego_poses"]))
    window.time_source = manifest["time_source"]
    if window.geometry_signature != manifest["geometry_signature"]:
        raise ValueError("Source geometry/calibration/poses changed since rendering")
    for name, expected in manifest["image_sha256"].items():
        if file_digest(out / name) != expected:
            raise ValueError(f"Input image changed since rendering: {name}")
    for name, expected in manifest["source_assets_sha256"].items():
        if file_digest(window.root / name) != expected:
            raise ValueError(f"Source image/pointcloud/calibration changed: {name}")
    payload = model_payload(window)
    model_file = jrdb.load_json(out / "model_input.json")
    if (model_file != {"images": list(manifest["image_sha256"]), "fields": payload,
                      "system_prompt": system_prompt(), "user_prompt": user_prompt(payload)}
            or (out / "user_prompt.txt").read_text(encoding="utf-8").strip() != user_prompt(payload)
            or (out / "system_prompt.txt").read_text(encoding="utf-8").strip() != system_prompt()):
        raise ValueError("Saved model input/prompts changed since rendering")
    fingerprint = digest({"images": manifest["image_sha256"], "payload": payload,
                          "prompt": user_prompt(payload), "system": system_prompt()})
    if fingerprint != manifest["input_fingerprint"]:
        raise ValueError("Input payload/fingerprint mismatch")
    return window, manifest


def validate_response(response: dict, valid_pid: list[int]) -> None:
    from jsonschema import Draft202012Validator

    schema_path = Path(__file__).resolve().parent / "configs/schemas/social_nav_prompt.json"
    schema = jrdb.load_json(schema_path)
    Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                         "$ref": "#/$defs/REL"}).validate(response)
    if len(valid_pid) > 20 or len(valid_pid) != len(set(valid_pid)):
        raise ValueError("Invalid VALID_PID set")
    claimed = []
    minima = []
    for group in response["g"]:
        members = group["m"]
        if group["k"] == 0 and members != sorted(members):
            raise ValueError("Unordered groups (k=0) must have ascending member PIDs")
        if group["r"] == "p" and group["k"] == 1 and members[2:] != sorted(members[2:]):
            raise ValueError("Extra photo members must have ascending PIDs after the two roles")
        minima.append(min(members))
        claimed.extend(members)
    if minima != sorted(minima) or response["u"] != sorted(response["u"]):
        raise ValueError("Groups/unknown PIDs are not in canonical order")
    claimed.extend(response["u"])
    if any(type(pid) is not int for pid in claimed):
        raise ValueError("PID must be an integer, not a string/float/bool")
    if any(type(group["k"]) is not int for group in response["g"]):
        raise ValueError("Role/order flag must be an integer")
    if len(claimed) != len(set(claimed)):
        raise ValueError("A PID occurs in multiple groups or in both g and u")
    if not set(claimed).issubset(valid_pid):
        raise ValueError("Response references PID outside current VALID_PID")


def native_reference(window: Window) -> tuple[dict, list[dict], dict]:
    """Return native group overlays plus raw facts, NOT projected T1 answers."""
    path = jrdb.label_path(window.root, "people", window.sequence, "0")
    # Stitched labels provide native membership for all-bev off-camera actors.
    stitched = jrdb.load_json(jrdb.label_path(window.root, "people", window.sequence, "stitched"))
    rows = {r["label_id"]: r for r in stitched["labels"][f"{window.frames[-1]}.jpg"]}
    rows.update({r["label_id"]: r for r in jrdb.load_json(path)["labels"][f"{window.frames[-1]}.jpg"]})
    groups = {}
    facts = []
    for identity in window.active:
        row = rows.get(identity)
        pid = window.pid_map[identity]
        if row is None:
            facts.append({"pid": pid, "source_id": identity, "native_annotation": None})
            continue
        group = row.get("social_group") or {}
        key = group.get("cluster_ID")
        facts.append({"pid": pid, "source_id": identity, "social_group": group,
                      "actions": row.get("action_label"), "social_activity": row.get("social_activity"),
                      "H_interaction": row.get("H-interaction")})
        # ID zero is not assumed to mean singleton; preserve the native field.
        if key is not None:
            groups.setdefault(str(key), []).append(pid)
    styles = {pid: (UNKNOWN, " ?native") for pid in window.valid_pid}
    links = []
    for n, (key, members) in enumerate(sorted(groups.items())):
        color = PALETTE[n % len(PALETTE)]
        for pid in members:
            styles[pid] = (color, f" JRDB-G{key}")
        if len(members) >= 2:
            links.append({"members": sorted(members), "color": color, "label": f"JRDB-G{key}",
                          "directed": False, "relation": None})
    return styles, links, {"kind": "native_jrdb_reference_not_t1_gold", "facts": facts,
                           "visible_group_subsets": groups,
                           "warning": "Native group/action labels may use wider context than the model inputs; do not auto-map standing together to q/t/o."}
