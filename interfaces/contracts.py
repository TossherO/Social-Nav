"""Shared records: perception has an adapter; NOT integrated with the current VLM.

Metres/radians/seconds, explicit frames/clocks, missing values as None. These
typing records perform no validation, transformations, scheduling or safety
enforcement. They are NOT wire messages, ROS messages or executable modules.
Existing object request fields/Schema/fingerprints remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

@dataclass(frozen=True)
class Header:
    observation_id: str
    stamp_s: float
    frame_id: str
    clock_domain: str
    sequence_id: str
    revision: int
    timestamp_source: str = "unknown"


@dataclass(frozen=True)
class TrackIdentity:
    track_id: str
    lifecycle_generation: int


@dataclass(frozen=True)
class Box3D:
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True)
class Detection3D:
    box: Box3D
    class_name: str
    score: float


@dataclass(frozen=True)
class PersonTrack:
    identity: TrackIdentity
    state: Literal["observed", "predicted", "lost"]
    last_observed_stamp_s: float
    box: Box3D | None
    velocity_xyz_mps: tuple[float, float, float] | None
    covariance: tuple[float, ...] | None
    rgb_box_xyxy_px: tuple[float, float, float, float] | None
    confirmed: bool = False
    score: float | None = None
    covariance_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoseEstimate:
    header: Header
    parent_frame: str
    child_frame: str
    parent_from_child_4x4: tuple[tuple[float, ...], ...]
    covariance: tuple[float, ...] | None
    health: Literal["valid", "degraded", "invalid"]


@dataclass(frozen=True)
class MapReference:
    frame_id: str
    map_id: str
    revision: int
    asset_uri: str
    layer: Literal["global", "local", "temporary"]


@dataclass(frozen=True)
class PerceptionSnapshot:
    header: Header
    tracks: tuple[PersonTrack, ...]
    track_mapping_revision: int
    ego_pose: PoseEstimate | None
    maps: tuple[MapReference, ...]
    rgb_asset_uris: tuple[str, ...]
    calibration_revision: str | None
    provenance: str
    health: Literal["valid", "degraded", "invalid"] = "invalid"
    reason: str | None = None


@dataclass(frozen=True)
class VLMResultEnvelope:
    observation_header: Header
    request_id: str
    input_fingerprint: str
    task: Literal["REL", "REL_FOLLOW", "GOAL"]
    track_mapping_revision: int
    mode_revision: int
    candidate_revision: int | None
    response: Mapping[str, Any]


@dataclass(frozen=True)
class GoalPose2D:
    frame_id: str
    x_m: float
    y_m: float
    yaw_rad: float | None


@dataclass(frozen=True)
class AgentIntent:
    task_id: str
    mode: Literal["FOLLOW", "NAVIGATE", "DOCK", "SAFE_STOP"]
    mode_revision: int
    objective: str
    goal: GoalPose2D | None
    target_description: str | None


@dataclass(frozen=True)
class PathPlan:
    header: Header
    task_id: str
    mode_revision: int
    map_revision: int
    poses: tuple[GoalPose2D, ...]


@dataclass(frozen=True)
class NavigationFeedback:
    header: Header
    task_id: str
    mode_revision: int
    state: Literal["idle", "executing", "stopped", "succeeded", "failed"]
    reason: str | None
