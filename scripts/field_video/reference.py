"""Reconstruct field-video reference geometry from archived cycles or static maps."""

import json
import math
import os
import re
import subprocess
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _leg_goal(route):
    index = route["waypoint_index"]
    return route["selected_pose"] if index == len(route["points"]) - 1 else route["points"][index]


def _read_envelopes(directory, bindings):
    envelopes = {}
    for binding in bindings:
        path = directory / "references" / f"{binding}.json"
        if path.is_file():
            envelopes[binding] = json.loads(path.read_text(encoding="utf-8"))
    if bindings - envelopes.keys():
        for path in (directory / "replies").glob("*.json"):
            reply = json.loads(path.read_text(encoding="utf-8")).get("result") or {}
            binding = reply.get("binding")
            if binding in bindings and binding not in envelopes:
                envelopes[binding] = reply
    return envelopes


def _archived_cycles(demo_dir):
    for path in demo_dir.rglob("cycles.jsonl"):
        if path.parent.name != "planning":
            continue
        with path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
        bindings = {row.get("reference_binding") for row in rows}
        bindings = {key for key in bindings if isinstance(key, str) and re.fullmatch("[a-f0-9]{64}", key)}
        if not bindings:
            continue
        config = path.parent / "config.json"
        if config.is_file():
            max_age = json.loads(config.read_text(encoding="utf-8"))["reference"]["max_age_s"]
        else:
            from navigation.planning.reference import load_config
            max_age = load_config()["max_age_s"]
        envelopes = _read_envelopes(path.parent, bindings)
        for row, next_row in zip(rows, [*rows[1:], {}]):
            stamp, duration = row["stamp_s"], row["dt"]
            if (row.get("reference_binding") in bindings and math.isfinite(stamp)
                    and isinstance(duration, (int, float)) and 0 < duration <= .5):
                expires = min(next_row.get("stamp_s", stamp + duration), stamp + .5)
                if stamp < expires:
                    yield row, expires, max_age, envelopes.get(row["reference_binding"])


def _archived_points(row, envelope, route, map_info, pose, max_age):
    from navigation.planning.continuity import content_valid
    from perception.environment.context import fingerprint

    binding = row["reference_binding"]
    if (envelope.get("binding") != binding
            or fingerprint({key: value for key, value in envelope.items() if key != "binding"}) != binding):
        return None
    result = envelope["result"]
    source = result["input_binding"]
    stamp = envelope["as_of"]
    if (not content_valid(result) or result["status"] != "path_found" or not result["current_path_valid"]
            or envelope["clock_domain"] != "runtime_live" or source["clock_domain"] != "runtime_live"
            or source["as_of"] != stamp or stamp < route["t"] or not 0 <= row["stamp_s"] - stamp <= max_age):
        return None
    handle, index = route["handle"], route["waypoint_index"]
    request = result["request"]
    goal = _leg_goal(route)
    requested = tuple(request["goal"])
    association = source.get("association") or {}
    static_map = source.get("static_map") or {}
    association_fields = ("map_id", "map_frame", "map_revision", "association_session", "association_generation")
    if (request["task_id"] != f"manual-preview:{handle['task_id']}:{handle['call_id']}"
            or request["request_id"] != f"{handle['call_id']}:{index}"
            or request["goal_revision"] != 100 * handle["goal_revision"] + index
            or request["mode"] != "mapped" or request["target_reference"] != "base_link"
            or request["frame_id"] != result["frame_id"] or result["frame_id"] != map_info["frame_id"]
            or not map_info["snapshot_id"] or not map_info["layers_sha256"]
            or request["map_snapshot_id"] != map_info["snapshot_id"]
            or any(static_map.get(key) != map_info[key] for key in ("snapshot_id", "layers_sha256"))
            or len(requested) not in (2, len(goal))
            or any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-6) for a, b in zip(requested, goal))
            or not association.get("valid") or association.get("map_id") != map_info["map_id"]
            or any(association.get(key) != pose["map_binding"].get(key) for key in association_fields)
            or (source["local_session"], source["local_generation"]) != tuple(pose["local_identity"][:2])):
        return None
    points = [tuple(point[:2]) for point in result["path"]]
    if (len(points) < 2 or any(len(point) != 2 or not all(map(math.isfinite, point)) for point in points)
            or math.dist(points[-1], requested[:2]) > 1e-6):
        return None
    return points


class StaticReference:
    """Reuse the read-only planning configuration and verified archived static geometry."""

    def __init__(self):
        self.planners = {}
        self.assets = {}

    def plan(self, route, map_info, pose):
        from navigation.planning.footprint import VehicleGeometry, swept_check
        from navigation.planning.inputs import load_config, load_input
        from navigation.planning.motion import load_motion_config
        from navigation.planning.planner import Planner

        profile = map_info["motion_profile"]
        if profile not in self.planners:
            config = load_config()
            if profile is not None:
                motion = load_motion_config(profile=profile)
                config["motion_constraints"] = {key: motion[key] for key in config["motion_constraints"]}
            native = _ROOT / "outputs/runtime/build/planning/libsocial_nav_pose_search.so"
            if native.is_file():
                config["pose_search"]["native_library"] = str(native)
            geometry = VehicleGeometry.historical(config["geometry_reference"])
            self.planners[profile] = Planner(config, geometry)
        planner = self.planners[profile]

        source = map_info["source"]
        if source not in self.assets:
            _, self.assets[source] = load_input(static_map=source)
        asset = self.assets[source]
        if (asset.snapshot.metadata.get("map_id") != map_info["map_id"]
                or asset.snapshot.metadata["frame_id"] != map_info["frame_id"]):
            raise ValueError("静态图与现场定位地图不匹配")
        handle = route["handle"]
        index = route["waypoint_index"]
        goal = _leg_goal(route)
        request = dict(
            request_id=f"{handle['call_id']}:{index}", task_id=handle["task_id"],
            goal_revision=handle["goal_revision"], mode="static", frame_id=map_info["frame_id"],
            map_snapshot_id=asset.manifest["snapshot_id"], algorithm="hybrid_astar", target_reference="base_link",
            start=[pose[key] for key in ("x", "y", "yaw")], goal=goal,
            goal_behavior="stop" if index == len(route["points"]) - 1 else "pass_through",
        )
        result = planner.query(request, asset=asset)
        if result["status"] != "path_found" or len(result["path"]) < 2:
            raise ValueError(result["reason"] or result["status"])
        # The global search may relax conditional obstacles. Keep the archived
        # occupancy intact when certifying a path to draw over that same map.
        check = swept_check(asset.snapshot, planner.geometry, result["path"], planner.config["safety_margin_m"],
                            planner.config["max_path_samples"], prefix=False)
        if check["status"] != "free":
            raise ValueError(f"静态图路径校验失败 ({check['status']})")
        return [tuple(point[:2]) for point in result["path"]]


def _compute_references(demo_dir, timeline, end):
    poses = timeline["poses"]
    stamps = [pose["t"] for pose in poses]
    cycles = sorted(_archived_cycles(demo_dir), key=lambda item: item[0]["stamp_s"])
    cycle_stamps = [row["stamp_s"] for row, _, _, _ in cycles]
    offline = StaticReference()
    references, failures = [], []

    def current_pose(stamp, floor):
        index = bisect_right(stamps, stamp) - 1
        if index >= 0 and poses[index]["floor"] == floor and stamp - poses[index]["t"] <= 1.:
            return poses[index]
        return None

    routes = timeline["routes"]
    for route, next_route in zip(routes, [*routes[1:], {"t": end}]):
        stamp, floor = route["t"], route["floor"]
        until = min(next_route["t"], end)
        if not route["points"] or route.get("blocked") or until <= stamp:
            continue
        map_info = timeline["maps"][floor]
        first_cycle = bisect_left(cycle_stamps, stamp)
        last_cycle = bisect_left(cycle_stamps, until)
        if last_cycle > first_cycle:
            for row, expires, max_age, envelope in cycles[first_cycle:last_cycle]:
                tick = row["stamp_s"]
                pose = current_pose(tick, floor)
                if envelope is None or pose is None:
                    continue
                try:
                    points = _archived_points(row, envelope, route, map_info, pose, max_age)
                except (KeyError, TypeError, ValueError):
                    continue
                if points:
                    expires = min(expires, envelope["as_of"] + max_age, until)
                    if tick < expires:
                        references.append(dict(t=tick, until=expires, floor=floor, points=points, source="archived"))
            continue
        pose = current_pose(stamp, floor)
        if pose is None:
            failures.append(dict(t=stamp, floor=floor, reason="缺少同层且时间匹配的定位起点"))
            continue
        try:
            points = offline.plan(route, map_info, pose)
        except ImportError as error:
            failures.append(dict(t=stamp, floor=floor, reason=f"规划解释器缺少依赖 ({error})"))
            break
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            failures.append(dict(t=stamp, floor=floor, reason=str(error)))
            continue
        references.append(dict(t=stamp, until=until, floor=floor, points=points, source="offline"))
    return references, failures


def load_references(demo_dir, timeline, end, planning_python=None):
    """Return intervals of floor-local reference XY in the recording's Unix clock."""
    python = planning_python or os.environ.get("SOCIAL_NAV_PYTHON")
    if not python:
        return _compute_references(demo_dir, timeline, end)
    inputs = {key: timeline[key] for key in ("poses", "routes", "maps")}
    try:
        worker = subprocess.run(
            [python, "-m", "scripts.field_video.reference"], cwd=_ROOT, check=True, capture_output=True,
            encoding="utf-8", input=json.dumps([demo_dir, inputs, end], default=str, allow_nan=False),
        )
        return tuple(json.loads(worker.stdout))
    except subprocess.CalledProcessError as error:
        lines = error.stderr.strip().splitlines()
        reason = lines[-1] if lines else f"退出状态 {error.returncode}"
    except (OSError, ValueError) as error:
        reason = str(error)
    return [], [dict(t=timeline["capture_start"], floor=None, reason=f"参考路线计算失败 ({reason})")]


if __name__ == "__main__":
    directory, data, finalized = json.load(sys.stdin)
    print(json.dumps(_compute_references(Path(directory), data, finalized), ensure_ascii=False, allow_nan=False))
