"""Validated public defaults shared by repository and portable benchmark clients."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


FIELDS = {"mode", "rate_hz", "deadline_s", "limit", "warmups", "max_tokens", "timeout", "constraint"}


def validate(settings):
    if not isinstance(settings, dict) or set(settings) != FIELDS:
        raise ValueError("Expected the complete benchmark defaults, without unknown fields")
    if settings["mode"] not in {"serial", "paced"} or settings["constraint"] not in {"json-schema", "json-object"}:
        raise ValueError("Invalid benchmark mode or output constraint")
    for name, lower, upper in (("limit", 1, 10000), ("warmups", 0, 1), ("max_tokens", 1, 4096)):
        if type(settings[name]) is not int or not lower <= settings[name] <= upper:
            raise ValueError("Invalid benchmark integer: " + name)
    for name in ("rate_hz", "deadline_s", "timeout"):
        value = settings[name]
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid benchmark timing: " + name)
        if name != "rate_hz" and value > 300:
            raise ValueError("Benchmark deadline and timeout must be at most 300 seconds")
    return dict(settings)


def from_project(config):
    if (not isinstance(config, dict) or not isinstance(config.get("inference"), dict)
            or not isinstance(config.get("benchmark"), dict)):
        raise ValueError("Expected public inference and benchmark configuration")
    inference, benchmark = config["inference"], config["benchmark"]
    for name, expected in (("thinking", False), ("temperature", 0), ("max_in_flight", 1),
                           ("automatic_retries", 0), ("prompt_profile", "object")):
        if inference.get(name) != expected:
            raise ValueError("This benchmark does not support changing inference." + name)
    settings = validate({"mode": benchmark["mode"], "rate_hz": inference["rate_hz"],
        "deadline_s": inference["deadline_s"], "limit": benchmark["limit"],
        "warmups": benchmark["warmups"], "max_tokens": inference["max_tokens"],
        "timeout": benchmark["timeout_s"], "constraint": inference["response_constraint"]})
    period = inference["period_s"]
    if (type(period) not in {int, float} or not math.isfinite(period) or period <= 0
            or not math.isclose(period * settings["rate_hz"], 1, rel_tol=1e-9)):
        raise ValueError("inference.period_s and inference.rate_hz must be reciprocal")
    return settings


def load_portable(directory):
    """Read the exported snapshot only after checking its manifest hash."""
    directory = Path(directory)
    name = "benchmark_defaults.json"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw = (directory / name).read_bytes()
    if manifest.get("files", {}).get(name) != hashlib.sha256(raw).hexdigest():
        raise ValueError("Portable benchmark defaults are missing from the manifest or changed")
    exported = json.loads(raw)
    # Old immutable packages may carry an extra format tag; validate the actual
    # settings and asset hash rather than requiring a project release number.
    if (not isinstance(exported, dict) or "settings" not in exported
            or set(exported) - {"settings", "version"}):
        raise ValueError("Expected portable benchmark settings")
    return validate(exported["settings"])
