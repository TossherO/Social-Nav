"""Read the recorded Agent timeline for a field demonstration."""

import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import yaml

from .reference import load_references


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        return [
            (stamp, kind, json.loads(value))
            for stamp, kind, value in db.execute("SELECT wall_s, kind, value FROM events ORDER BY seq")
        ]


def _load_maps(demo_dir, metadata, floors):
    assets = metadata["assets"]
    if isinstance(assets, str):
        (floor,) = floors
        assets = {floor: assets}
    maps = {}
    for floor, path in assets.items():
        static_map = (demo_dir / path / "static_map").resolve()
        with (static_map / "map.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        image = static_map / config["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        preset = static_map.parent / "interactive.yaml"
        motion_profile = None
        if preset.is_file():
            motion_profile = yaml.safe_load(preset.read_text(encoding="utf-8")).get("motion_profile")
        manifest_path = static_map / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        maps[floor] = dict(
            image=image, origin=config["origin"], resolution=float(config["resolution"]),
            negate=config.get("negate", 0), occupied_thresh=config.get("occupied_thresh", .65),
            free_thresh=config.get("free_thresh", .196), source=static_map,
            map_id=floors[floor]["localization_map_id"], frame_id=floors[floor]["frame_id"],
            motion_profile=motion_profile, snapshot_id=manifest.get("snapshot_id"),
            layers_sha256=manifest.get("layers_sha256"),
        )
    return maps


def _load_poses(records, floors, start):
    floor_by_map = {data["localization_map_id"]: floor for floor, data in floors.items()}
    samples = {}
    for wall_s, kind, data in records:
        if kind not in {"trajectory_snapshot", "manual_route_trace_samples"}:
            continue
        for segment in data["segments"]:
            association = segment["map_binding"]
            floor = floor_by_map.get(association.get("map_id"))
            if not association.get("valid") or floor is None or segment["clock_domain"] != "runtime_live":
                continue
            for sample in segment["samples"]:
                stamp = sample["stamp_s"]
                if start <= stamp <= wall_s:
                    x, y, yaw = sample["pose"]
                    samples[stamp] = (floor, segment["binding"], x, y, yaw, segment["local_identity"], association)

    poses = []
    last_context = None
    last_stamp = -math.inf
    segment_number = 0
    for stamp, (floor, binding, x, y, yaw, identity, association) in sorted(samples.items()):
        context = (floor, binding)
        if context != last_context or stamp - last_stamp > 2.0:
            segment_number += 1
        poses.append(dict(t=stamp, x=x, y=y, yaw=yaw, floor=floor, segment=str(segment_number),
                          local_identity=identity, map_binding=association))
        last_context = context
        last_stamp = stamp
    return poses


def _in_polygon(point, polygon):
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _route_labels(points, annotations):
    labels = []
    for index, point in enumerate(points):
        label = next((
            item["name"] for item in annotations["points"]
            if math.hypot(point[0] - item["pose"][0], point[1] - item["pose"][1]) < 0.08
        ), "")
        if not label and index == len(points) - 1:
            label = next((
                item["name"] for item in annotations.get("regions", [])
                if _in_polygon(point, item["polygon"])
            ), "停车目标")
        labels.append(label or f"途经点{index + 1}")
    return labels


_BLOCK_REASONS = {
    "goal_footprint_occupied": "目标处的车体占地范围被占用",
    "start_footprint_occupied": "起点处的车体占地范围被占用",
    "returned_path_no_curvature_route": "参考路径不满足车体转弯约束",
}


def _decision_text(points, labels, previous):
    if len(points) > 1:
        if len(points) == 2:
            if previous and previous["labels"][0] != labels[0] and previous["points"][-1] == points[-1]:
                return f"Agent 改走{labels[0]}，仍前往{labels[-1]}停车"
            return f"Agent 选择先经过{labels[0]}，再到{labels[-1]}停车"
        return f"Agent 提交 {len(points) - 1} 个途经点和停车目标"
    if previous and previous["points"][-1] != points[-1]:
        x, y = points[-1]
        old_x, old_y = previous["points"][-1]
        return f"Agent 调整停车目标，距上一次目标 {math.hypot(x - old_x, y - old_y):.2f} 米"
    return "Agent 选择停车目标" if labels[-1] == "停车目标" else f"Agent 选择前往{labels[-1]}"


def load_demo(demo_dir: Path, *, planning_python=None) -> dict:
    """Return time-ordered events, map poses and current accepted targets in Unix seconds."""
    demo_dir = demo_dir.resolve()
    metadata = _read_json(demo_dir / "demo.json")
    start = datetime.fromisoformat(metadata["capture_start"]).timestamp()
    records = _read_events(demo_dir / metadata["agent_session"])
    floors = next(data["floors"] for _, kind, data in records if kind == "task_asset_provenance")
    maps = _load_maps(demo_dir, metadata, floors)
    annotations = {floor: data["annotations"] for floor, data in floors.items()}
    poses = _load_poses(records, floors, start)
    floor = poses[0]["floor"] if poses else next(iter(floors))
    floor_changes = [dict(t=start, floor=floor)]

    events, routes, evidence = [], [], []
    stage_specs = metadata.get("stages", [])
    stage_order = {item["id"]: index for index, item in enumerate(stage_specs)}
    stages = []
    tool_intent = None
    submission = None
    active = None
    active_handle = None
    previous = {}

    def event(stamp, kind, text):
        events.append(dict(t=stamp, kind=kind, text=text))

    def clear_route(stamp):
        routes.append(dict(t=stamp, floor=floor, points=[], labels=[]))

    for stamp, kind, data in records:
        if kind == "tool_intent":
            tool = data["tool"]
            tool_intent = None
            submission = None
            if tool in {"navigate", "navigate_route"}:
                tool_intent = (stamp, data["arguments"])
            elif tool == "cancel_navigation":
                event(stamp, "decision", "Agent 请求撤销当前导航目标")
        elif kind == "navigation_intent" and tool_intent and tool_intent[1]["goal_id"] == data["goal_id"]:
            intent_stamp, arguments = tool_intent
            tool_intent = None
            selected_pose = tuple(data["selected_pose"])
            target = selected_pose[:2]
            points = [tuple(point[:2]) for point in arguments.get("waypoints", [])] or [target]
            points[-1] = target
            labels = _route_labels(points, floors[floor]["annotations"])
            prior = previous.get((floor, data["goal_id"]))
            if prior and len(points) == 1 and labels == ["停车目标"] and prior["points"] != points:
                labels = ["调整后的停车点"]
            event(intent_stamp, "decision", _decision_text(points, labels, prior))
            submission = (data["handle"], points, labels, selected_pose)
        elif (kind == "navigation_ack" and data["status"] == "running"
              and submission and submission[0] == data["handle"]):
            active_handle, points, labels, selected_pose = submission
            active = dict(t=stamp, floor=floor, points=points, labels=labels, goal_id=data["goal_id"],
                          handle=active_handle, waypoint_index=0, selected_pose=selected_pose)
            routes.append(active)
            event(stamp, "accepted", "导航接受途经点和停车目标" if len(points) > 1 else "导航接受停车目标")
            previous[(floor, data["goal_id"])] = active
            submission = None
        elif kind == "navigation_blocked" and active and active_handle == data["handle"]:
            blockage = data.get("blockage") or {}
            index = blockage.get("route_index")
            label = "当前参考路线"
            active = dict(active, t=stamp, blocked=True)
            active.pop("blocked_index", None)
            if data["reason"] == "goal_footprint_occupied":
                if index is None and len(active["points"]) == 1:
                    index = 0
                if index is not None and 0 <= index < len(active["points"]):
                    active["blocked_index"] = index
                    label = active["labels"][index]
            routes.append(active)
            reason = _BLOCK_REASONS.get(data["reason"], "参考路线检查失败")
            event(stamp, "blocked", f"导航反馈{label}受阻，{reason}")
        elif kind == "navigation_unblocked" and active and active_handle == data["handle"]:
            if active.get("blocked"):
                active = dict(active, t=stamp)
                active.pop("blocked")
                active.pop("blocked_index", None)
                routes.append(active)
            event(stamp, "stage", "导航反馈当前参考路线恢复")
        elif kind == "navigation_cancel_confirmed":
            clear_route(stamp)
            active = None
            active_handle = None
            event(stamp, "stage", "导航确认原目标已撤销")
        elif kind == "manual_route_progress" and active:
            visits = data["waypoint_visits"]
            if visits:
                visit = visits[-1]
                index = visit["index"]
                if (visit["handle"] == active_handle and index == active["waypoint_index"]
                        and index + 1 < len(active["points"])):
                    arrived = visit["stamp_s"]
                    if active["t"] <= arrived <= stamp:
                        event(arrived, "arrived", f"现场定位显示已到达{active['labels'][index]}")
                        active = dict(active, t=arrived, waypoint_index=index + 1)
                        routes.append(active)
        elif kind == "navigation_completed" and active and data["completed"][-1]["handle"] == active_handle:
            clear_route(stamp)
            label = active["labels"][-1]
            active = None
            active_handle = None
            reached = data["completed"][-1]
            suffix = "并停稳" if reached.get("stopped") else ""
            event(stamp, "arrived", f"现场定位显示已到达{label}{suffix}")
        elif kind == "elevator_observation_started" and data["result"]["status"] == "accepted":
            event(stamp, "stage", "进入人工辅助乘梯阶段")
        elif kind == "manual_floor_rebound":
            transition = data["transition"]
            if (transition["to_floor"] in maps and transition["association"]["valid"]):
                clear_route(stamp)
                floor = transition["to_floor"]
                floor_changes.append(dict(t=stamp, floor=floor))
                active = None
                active_handle = None
                event(stamp, "stage", "目的楼层定位已接通")
        elif kind == "visual_review":
            outcome = {"partly_satisfied": "，画面部分满足条件", "not_satisfied": "，画面未满足条件"}
            event(stamp, "decision", "Agent 复核停车画面" + outcome.get(data["result"], ""))
        elif kind == "tool_result":
            tool = data["call"]["tool"]
            result = data["value"]
            if tool == "record_task" and result["status"] == "recorded" and len(stage_specs) > 1 and not stages:
                first_goal = result["goals"][0]["requirement"]["goal_id"]
                if first_goal == stage_specs[0]["id"]:
                    stages.append(dict(t=max(stamp, start), text=stage_specs[0]["instruction"]))
            elif tool == "complete_subtask" and result["status"] == "completed":
                if result["task_status"] == "succeeded":
                    event(stamp, "complete", "Agent 确认任务完成")
                elif result["goal_id"] in stage_order:
                    next_index = stage_order[result["goal_id"]] + 1
                    event(stamp, "stage", f"Agent 确认第 {next_index} 阶段完成")
                    if result["task_status"] == "ready" and 0 < next_index == len(stages) < len(stage_specs):
                        stages.append(dict(t=stamp, text=stage_specs[next_index]["instruction"]))

    for key, item in metadata.get("rgb_evidence", {}).items():
        path = demo_dir / item["path"]
        if path.is_file():
            stamp = item["source_stamp_s"]
            evidence.append(dict(t=stamp, path=path))
            text = {
                "first_stop": "首次停车后的前置相机画面", "adjusted_stop": "调整车位后的前置相机画面",
            }.get(key, "归档相机画面")
            event(stamp, "visual", text)

    timeline = {
        "instruction": metadata["instruction"], "driving_mode": metadata["driving_mode"],
        "bag": demo_dir / metadata["bag"], "capture_start": start,
        "events": sorted(events, key=lambda item: item["t"]), "poses": poses,
        "routes": sorted(routes, key=lambda item: item["t"]), "maps": maps,
        "annotations": annotations, "floor_changes": floor_changes,
        "evidence": sorted(evidence, key=lambda item: item["t"]),
    }
    if stages:
        timeline["stages"] = stages
    end = datetime.fromisoformat(metadata["capture_finalized"]).timestamp()
    timeline["references"], timeline["reference_failures"] = load_references(
        demo_dir, timeline, end, planning_python,
    )
    return timeline
