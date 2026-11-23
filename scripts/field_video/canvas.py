"""Compose the 3D view, semantic floor map, camera and Agent events."""

import math
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIZE = (1920, 1080)
LOCAL_SIZE = (1200, 1000)
SIDE_SIZE = (720, 540)
MAP_TOP = 46
MAP_BOTTOM = 504
MAP_SIZE = (SIDE_SIZE[0], MAP_BOTTOM - MAP_TOP)
POSE_MAX_AGE = 1.25
EVIDENCE_SECONDS = 5
TIME_ZONE = timezone(timedelta(hours=8))

BG = "#0d1722"
RULE = "#263b4c"
TEXT = "#f0f5fa"
MUTED = "#a1b5c8"
TEAL = "#5ee3c9"
AMBER = "#ffc078"
MAP_BG = "#f2f6f5"
MAP_BORDER = "#d1dfdf"
MAP_INK = "#263b4d"
MAP_MUTED = "#4c6574"
MAP_TRAIL = "#007e81"
MAP_INTENT = "#a0520b"
MAP_BLOCKED = "#ab382b"
REGION_EDGE = "#269255"
# Match the static occupancy palette in decision/agent/tools/spatial.py.
MAP_OCCUPANCY = np.array(((246, 246, 239), (169, 177, 186), (58, 63, 71)), dtype=np.uint8)


@lru_cache(maxsize=24)
def _font(size, bold=False):
    weight = "Bold" if bold else "Regular"
    return ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-" + weight + ".ttc",
                              size, index=2)


def _text_width(font, value):
    if hasattr(font, "getlength"):
        return font.getlength(value)
    return font.getsize(value)[0]


def _text(draw, xy, text, *, font, fill, anchor="lt", **style):
    if hasattr(font, "getbbox"):
        draw.text(xy, text, font=font, fill=fill, anchor=anchor, **style)
        return
    # Pillow 7 accepts anchor but ignores it.
    x, y = xy
    width, height = font.getsize(text)
    left, top = font.getoffset(text)
    x -= {"l": left, "m": (left + width) / 2, "r": width}[anchor[0]]
    y -= top if anchor[1] == "t" else (top + height) / 2
    draw.text((x, y), text, font=font, fill=fill, **style)


@lru_cache(maxsize=512)
def _fit_text(value, size, width, bold=False):
    font = _font(size, bold)
    if _text_width(font, value) <= width:
        return value
    while value and _text_width(font, value + "…") > width:
        value = value[:-1]
    return value + "…"


def _opening_lines(instruction):
    width, size = 855, 23
    if _text_width(_font(size, True), instruction) <= width:
        return [instruction]
    first = _fit_text(instruction, size, width, True)[:-1]
    return [first, _fit_text(instruction[len(first):].lstrip(), size, width, True)]


def _duration(seconds):
    minutes, remainder = divmod(int(abs(seconds)), 60)
    sign = "−" if seconds < 0 else "+"
    return f"{sign}{minutes:02d}:{remainder:02d}"


def _clock(stamp):
    if stamp >= 1_000_000_000:
        return datetime.fromtimestamp(stamp, TIME_ZONE).strftime("%H:%M:%S")
    return _duration(stamp)


def _floor_title(floor):
    names = {"floor1": "一楼", "floor2": "二楼", "1": "一楼", "2": "二楼"}
    return names.get(str(floor), str(floor))


def _connected(first, second):
    return (first["floor"] == second["floor"] and first.get("segment") == second.get("segment")
            and second["t"] - first["t"] <= POSE_MAX_AGE
            and math.hypot(second["x"] - first["x"], second["y"] - first["y"]) <= 3)


def _capsule(draw, x, y, text, *, size=16, bold=False, fill=BG, ink=TEXT, align="left"):
    font = _font(size, bold)
    width = math.ceil(_text_width(font, text)) + 20
    if align == "right":
        x -= width
    draw.rectangle((x, y, x + width, y + size + 12), fill=fill)
    _text(draw, (x + 10, y + 6), text, font=font, fill=ink)


def _media(image, title):
    panel = Image.new("RGB", SIDE_SIZE, BG)
    if image is not None:
        scale = min(panel.width / image.width, panel.height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        panel.paste(image.resize(size, Image.BILINEAR),
                    ((panel.width - size[0]) // 2, (panel.height - size[1]) // 2))
    draw = ImageDraw.Draw(panel)
    _capsule(draw, 14, 13, title, bold=True)
    if image is None:
        _text(draw, (panel.width // 2, panel.height // 2), "无同期相机画面",
              font=_font(20), fill=MUTED, anchor="mm")
    return panel


def _dash(draw, first, second, color, width=3):
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = math.hypot(dx, dy)
    for start in range(0, math.ceil(length), 13):
        end = min(start + 7, length)
        draw.line((first[0] + dx * start / length, first[1] + dy * start / length,
                   first[0] + dx * end / length, first[1] + dy * end / length),
                  fill=color, width=width)


def _heading(draw, x, y, yaw):
    forward = math.cos(yaw), -math.sin(yaw)
    side = -forward[1], forward[0]
    triangle = [
        (x + along * forward[0] + cross * side[0], y + along * forward[1] + cross * side[1])
        for along, cross in ((12, 0), (-6, -6), (-3, 0), (-6, 6))
    ]
    draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=MAP_BG, outline=MAP_TRAIL, width=2)
    draw.polygon(triangle, fill=MAP_TRAIL)


def _label(draw, occupied, x, y, text, color, *, bold=False, region=False):
    font = _font(16, bold)
    width, height = math.ceil(_text_width(font, text)) + 6, 23
    offsets = [(11, -11), (-width - 11, -11), (-width / 2, -31),
               (-width / 2, 12), (11, 10), (-width - 11, 10), (11, -34), (-width - 11, -34)]
    if region:
        offsets.insert(0, (-width / 2, -11))
        offsets += [(45, -11), (-width - 45, -11), (-width / 2, 30), (-width / 2, -53),
                    (44, 35), (-width - 44, 35), (44, -55), (-width - 44, -55),
                    (-width / 2, 57), (-width / 2, -79)]
    for dx, dy in offsets:
        left, top = round(x + dx), round(y + dy)
        box = (left, top, left + width, top + height)
        if (box[0] < 4 or box[1] < 1 or box[2] > MAP_SIZE[0] - 4 or box[3] > MAP_SIZE[1] - 1
                or any(box[0] < right and box[2] > other_left and box[1] < bottom and box[3] > other_top
                       for other_left, other_top, right, bottom in occupied)):
            continue
        if (region and (abs(dx + width / 2) > 25 or abs(dy + 11) > 25)
                or not region and abs(dx) != 11 and abs(dy) > 11):
            draw.line((x, y, max(left, min(x, box[2])), max(top, min(y, box[3]))),
                      fill=REGION_EDGE, width=1)
        _text(draw, (left + 3, top + 2), text, font=font, fill=color,
              stroke_width=2, stroke_fill=MAP_BG)
        occupied.append(box)
        return


def _edge_marker(draw, x, y, color):
    """Return a label anchor for a target beyond the fixed crop."""
    cx, cy = MAP_SIZE[0] / 2, MAP_SIZE[1] / 2
    dx, dy = x - cx, y - cy
    ratio = min((cx - 15) / max(abs(dx), 1), (cy - 15) / max(abs(dy), 1))
    x, y = cx + dx * ratio, cy + dy * ratio
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    draw.polygon(((x + 8 * ux, y + 8 * uy),
                  (x - 6 * ux + 6 * uy, y - 6 * uy - 6 * ux),
                  (x - 6 * ux - 6 * uy, y - 6 * uy + 6 * ux)), fill=color)
    return x, y


class _SemanticMap:
    def __init__(self, floor, poses, annotations, routes, occupancy=None):
        self.floor = floor
        actual = [(p["x"], p["y"]) for p in poses]
        targets = [point for route in routes for point in route["points"]]
        annotation_points = [item["pose"][:2] for item in annotations.get("points", [])]
        points = actual or targets or annotation_points or [(0, 0)]
        xs, ys = zip(*points)
        self.center = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.scale = min((MAP_SIZE[0] - 28) / max(14, max(xs) - min(xs) + 8),
                         (MAP_SIZE[1] - 24) / max(12, max(ys) - min(ys) + 8))
        if occupancy is None:
            self.background = Image.new("RGB", MAP_SIZE, tuple(MAP_OCCUPANCY[1]))
        else:
            cols, rows = np.meshgrid(np.arange(MAP_SIZE[0]) + .5, np.arange(MAP_SIZE[1]) + .5)
            x = self.center[0] + (cols - MAP_SIZE[0] / 2) / self.scale
            y = self.center[1] - (rows - MAP_SIZE[1] / 2) / self.scale
            cells = occupancy.at(np.stack((x, y), axis=-1))
            self.background = Image.fromarray(MAP_OCCUPANCY[cells])
        draw = ImageDraw.Draw(self.background)
        self.regions = []
        for item in annotations.get("regions", []):
            shape = [self.project(x, y) for x, y in item["polygon"]]
            mask = Image.new("1", MAP_SIZE)
            ImageDraw.Draw(mask).polygon(shape, fill=1)
            bounds = mask.getbbox()
            if bounds is None:
                continue
            draw.line(shape + [shape[0]], fill=REGION_EDGE, width=2)
            x = (bounds[0] + bounds[2]) / 2
            y = (bounds[1] + bounds[3]) / 2
            self.regions.append((x, y, item["name"]))
        self.points = []
        for item in annotations.get("points", []):
            x, y = self.project(*item["pose"][:2])
            if 8 <= x <= MAP_SIZE[0] - 8 and 8 <= y <= MAP_SIZE[1] - 8:
                self.points.append((x, y, item["name"]))

        self.panel = Image.new("RGB", SIDE_SIZE, MAP_BG)
        draw = ImageDraw.Draw(self.panel)
        _text(draw, (17, 13), f"{_floor_title(floor)}全局地图", font=_font(22, True), fill=MAP_INK)
        _text(draw, (680, 13), "+Y", font=_font(15, True), fill=MAP_MUTED, anchor="rt")
        draw.line((691, 35, 691, 15), fill=MAP_MUTED, width=2)
        draw.polygon(((691, 11), (686, 20), (696, 20)), fill=MAP_MUTED)
        draw.line((0, MAP_TOP - 1, self.panel.width, MAP_TOP - 1), fill=MAP_BORDER)
        draw.line((0, MAP_BOTTOM, self.panel.width, MAP_BOTTOM), fill=MAP_BORDER)
        y = MAP_BOTTOM + 18
        draw.line((17, y, 42, y), fill=MAP_TRAIL, width=4)
        _text(draw, (51, y), "实测轨迹", font=_font(15), fill=MAP_INK, anchor="lm")
        _dash(draw, (145, y), (170, y), MAP_INTENT)
        _text(draw, (179, y), "目标方向", font=_font(15), fill=MAP_INK, anchor="lm")
        for x, cell, label in ((278, 0, "空闲"), (346, 2, "占据"), (414, 1, "未知")):
            draw.rectangle((x, y - 6, x + 12, y + 6), fill=tuple(MAP_OCCUPANCY[cell]), outline=MAP_MUTED)
            _text(draw, (x + 19, y), label, font=_font(15), fill=MAP_INK, anchor="lm")
        scale = 5 * self.scale
        x = 699 - scale
        draw.line((x, y + 7, 699, y + 7), fill=MAP_MUTED, width=2)
        for end in (x, 699):
            draw.line((end, y + 3, end, y + 9), fill=MAP_MUTED, width=2)
        _text(draw, ((x + 699) / 2, y - 5), "5 m", font=_font(13), fill=MAP_MUTED, anchor="mm")

    def project(self, x, y):
        return (MAP_SIZE[0] / 2 + (x - self.center[0]) * self.scale,
                MAP_SIZE[1] / 2 - (y - self.center[1]) * self.scale)

    def render(self, stamp, sections, pose, route):
        map_image = self.background.copy()
        draw = ImageDraw.Draw(map_image)
        for points, times in sections:
            count = bisect_right(times, stamp)
            if count > 1:
                draw.line(points[:count], fill=MAP_BG, width=7)
                draw.line(points[:count], fill=MAP_TRAIL, width=4)

        occupied = []
        labels_used = set()
        route_points = []
        if route and route["points"]:
            points = [self.project(*point) for point in route["points"]]
            labels = route.get("labels", [])
            start_index = min(route.get("waypoint_index", 0), len(points) - 1)
            intent = points[start_index:]
            if pose:
                intent = [self.project(pose["x"], pose["y"])] + intent
            blocked_index = route.get("blocked_index")
            for segment, (first, second) in enumerate(zip(intent, intent[1:])):
                target_index = start_index + segment + (pose is None)
                color = MAP_BLOCKED if (target_index == blocked_index or
                                        route.get("blocked") and blocked_index is None) else MAP_INTENT
                _dash(draw, first, second, color)
            for index, (x, y) in enumerate(points):
                blocked = index == blocked_index
                label = labels[index] if index < len(labels) and labels[index] else (
                    "停车目标" if index == len(points) - 1 else f"途经点 {index + 1}")
                color = MAP_BLOCKED if blocked else MAP_INTENT
                if index < start_index:
                    color = MAP_MUTED
                status = " · 受阻" if blocked else ""
                if 9 <= x <= MAP_SIZE[0] - 9 and 9 <= y <= MAP_SIZE[1] - 9:
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=MAP_BG, outline=color, width=2)
                    if blocked:
                        draw.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                        draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)
                    elif index == len(points) - 1:
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
                else:
                    x, y = _edge_marker(draw, x, y, color)
                    status += " · 图外"
                route_points.append((x, y, label + status, color))
                labels_used.add(label)
                occupied.append((x - 9, y - 9, x + 9, y + 9))

        semantic_points = []
        for x, y, label in self.points:
            if any(math.hypot(x - rx, y - ry) < 9 and label in labels_used
                   for rx, ry, _, _ in route_points):
                continue
            draw.rectangle((x - 3, y - 3, x + 3, y + 3), fill=MAP_MUTED)
            semantic_points.append((x, y, label))
            occupied.append((x - 5, y - 5, x + 5, y + 5))
        if pose:
            x, y = self.project(pose["x"], pose["y"])
            _heading(draw, x, y, pose["yaw"])
            occupied.append((x - 14, y - 14, x + 14, y + 14))
        for x, y, label, color in route_points:
            _label(draw, occupied, x, y, label, color, bold=True)
        for x, y, label in semantic_points:
            _label(draw, occupied, x, y, label, MAP_INK)
        for x, y, label in self.regions:
            if label not in labels_used:
                _label(draw, occupied, x, y, label, MAP_MUTED, region=True)

        panel = self.panel.copy()
        panel.paste(map_image, (0, MAP_TOP))
        if pose is None:
            _capsule(ImageDraw.Draw(panel), 630, 8, "定位暂不可用", fill="#e1ebeb",
                     ink=MAP_MUTED, align="right")
        return panel


class Canvas:
    def __init__(self, data):
        from .local_view import LocalView

        self.local = LocalView(data, size=LOCAL_SIZE)
        self.instruction = _opening_lines(data["instruction"])
        self.mode = "Agent 决策 · 人工手柄驾驶" if data.get("driving_mode") == "manual" else "Agent 决策"
        self.events = sorted(data["events"], key=lambda item: item["t"])
        self.stages = sorted(data.get("stages", []), key=lambda item: item["t"])
        self.poses = sorted(data["poses"], key=lambda item: item["t"])
        self.evidence = sorted(data.get("evidence", []), key=lambda item: item["t"])
        self.event_times = [item["t"] for item in self.events]
        self.stage_times = [item["t"] for item in self.stages]
        self.pose_times = [item["t"] for item in self.poses]
        self.evidence_times = [item["t"] for item in self.evidence]
        self.floor_changes = sorted(data.get("floor_changes", []), key=lambda item: item["t"])
        self.floor_times = [item["t"] for item in self.floor_changes]
        self.routes = defaultdict(list)
        self.references = defaultdict(list)
        for route in sorted(data.get("routes", []), key=lambda item: item["t"]):
            self.routes[route["floor"]].append(route)
        for reference in sorted(data.get("references", []), key=lambda item: item["t"]):
            self.references[reference["floor"]].append(reference)
        self.route_times = {floor: [item["t"] for item in rows] for floor, rows in self.routes.items()}
        self.reference_times = {floor: [item["t"] for item in rows] for floor, rows in self.references.items()}
        self.completion_time = max((item["t"] for item in self.events if item["kind"] == "complete"),
                                   default=None)

        by_floor = defaultdict(list)
        ranges = [0] if self.poses else []
        for index, pose in enumerate(self.poses):
            by_floor[pose["floor"]].append(pose)
            if index and not _connected(self.poses[index - 1], pose):
                ranges.append(index)
        self.section_starts = ranges
        annotations = data.get("annotations", {})
        floors = dict.fromkeys([*data.get("maps", {}), *annotations, *by_floor, *self.routes])
        self.maps = {floor: _SemanticMap(floor, by_floor[floor], annotations.get(floor, {}),
                                         self.routes[floor], self.local.maps.get(floor)) for floor in floors}
        self.first_floor = self.poses[0]["floor"] if self.poses else next(iter(floors), None)
        self.sections = defaultdict(list)
        for left, right in zip(ranges, ranges[1:] + [len(self.poses)]):
            section = self.poses[left:right]
            floor = section[0]["floor"]
            projected = [self.maps[floor].project(p["x"], p["y"]) for p in section]
            self.sections[floor].append((projected, self.pose_times[left:right]))

    def _pose_index(self, t):
        index = bisect_right(self.pose_times, t) - 1
        if index < 0 or t - self.pose_times[index] > POSE_MAX_AGE:
            return None
        return index

    def has_localization(self, t):
        return self._pose_index(t) is not None

    @lru_cache(maxsize=16)
    def _evidence_frame(self, path):
        with Image.open(path) as image:
            return _media(image.convert("RGB"), "归档相机证据")

    def _draw_events(self, draw, t):
        draw.rectangle((0, LOCAL_SIZE[1], LOCAL_SIZE[0] - 1, SIZE[1] - 1), fill=BG)
        draw.line((0, LOCAL_SIZE[1], LOCAL_SIZE[0] - 1, LOCAL_SIZE[1]), fill=RULE)
        _text(draw, (17, 1014), "Agent 事件", font=_font(17, True), fill=TEAL)
        event_index = bisect_right(self.event_times, t) - 1
        rows = []
        while event_index >= 0 and len(rows) < 2:
            event = self.events[event_index]
            previous = self.events[event_index - 1] if event_index > 0 else None
            if (event["kind"] == "accepted" and previous and previous["kind"] == "decision"
                    and event["t"] - previous["t"] <= 1):
                rows.append((previous, "  ·  导航已接受"))
                event_index -= 2
            else:
                rows.append((event, ""))
                event_index -= 1
        if not rows:
            stage = bisect_right(self.stage_times, t) - 1
            text = self.stages[stage]["text"] if stage >= 0 else "等待现场事件"
            _text(draw, (143, 1015), _fit_text(text, 19, 1025), font=_font(19), fill=MUTED)
        for index, (event, suffix) in enumerate(rows):
            y = 1014 + 32 * index
            color = AMBER if event["kind"] in {"blocked", "cancelled", "cancel", "warning"} else TEAL
            _text(draw, (143, y + 2), _clock(event["t"]), font=_font(15),
                  fill=color if index == 0 else MUTED)
            text = _fit_text(event["text"] + suffix, 20, 942, index == 0)
            _text(draw, (230, y), text, font=_font(20, index == 0), fill=TEXT if index == 0 else MUTED)

    def frame(self, t, rgb, *, elapsed, skipped=0.0, scan_xyz=None, scan_frame=None, scan_stamp=None,
              frozen_at=None, bookend=False):
        observed = t if frozen_at is None else frozen_at
        last_index = bisect_right(self.pose_times, observed) - 1
        floor = self.poses[last_index]["floor"] if last_index >= 0 else self.first_floor
        floor_index = bisect_right(self.floor_times, observed) - 1
        if floor_index >= 0:
            floor = self.floor_changes[floor_index]["floor"]
        pose_index = self._pose_index(observed)
        pose = self.poses[pose_index] if pose_index is not None else None
        trail = []
        snapshot_not_before = self.floor_times[floor_index] if floor_index >= 0 else -math.inf
        if pose is not None and pose["floor"] == floor:
            start = self.section_starts[bisect_right(self.section_starts, pose_index) - 1]
            trail = self.poses[start:pose_index + 1]
            snapshot_not_before = max(snapshot_not_before, trail[0]["t"])
        else:
            pose = None

        route_index = bisect_right(self.route_times.get(floor, []), t) - 1
        route = self.routes[floor][route_index] if route_index >= 0 else None
        if route and not route["points"]:
            route = None
        reference_index = bisect_right(self.reference_times.get(floor, []), t) - 1
        reference = self.references[floor][reference_index] if reference_index >= 0 else None
        if reference is not None and (route is None or t >= reference["until"]):
            reference = None

        result = Image.new("RGB", SIZE, BG)
        result.paste(self.local.render(observed, pose, trail, route, scan_xyz, scan_frame, scan_stamp,
                                       reference, snapshot_not_before=snapshot_not_before), (0, 0))
        if floor in self.maps:
            panel = self.maps[floor].render(observed, self.sections[floor], pose, route)
            result.paste(panel, (LOCAL_SIZE[0], 0))
        else:
            _text(ImageDraw.Draw(result), (1560, 270), "暂无楼层语义数据", font=_font(20),
                  fill=MUTED, anchor="mm")
        evidence_index = bisect_right(self.evidence_times, t) - 1
        if evidence_index >= 0 and t - self.evidence_times[evidence_index] < EVIDENCE_SECONDS:
            media = self._evidence_frame(self.evidence[evidence_index]["path"])
        else:
            title = "前置相机" if frozen_at is None else f"前置相机 · 定格于 {_clock(frozen_at)}"
            media = _media(rgb, title)
        result.paste(media, (LOCAL_SIZE[0], SIDE_SIZE[1]))

        draw = ImageDraw.Draw(result)
        if elapsed <= 3:
            for index, line in enumerate(self.instruction):
                _capsule(draw, 15, 13 + 32 * index, line, size=23, bold=True)
        _capsule(draw, LOCAL_SIZE[0] - 14, 13, f"{_clock(t)}  {_duration(elapsed)}",
                 size=17, ink=MUTED, align="right")
        complete = self.completion_time is not None and self.completion_time <= t
        if elapsed <= 3 or frozen_at is not None or bookend or complete:
            _capsule(draw, LOCAL_SIZE[0] - 14, 46, self.mode, ink=TEAL, align="right")
        if skipped:
            _capsule(draw, LOCAL_SIZE[0] - 14, 78, f"跳过 {skipped:.1f} 秒", ink=AMBER, align="right")
        self._draw_events(draw, t)
        return result
