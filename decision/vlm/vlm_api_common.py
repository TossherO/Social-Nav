"""VLM-domain portable image, Schema, REL and latency utilities; no credentials."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

from jsonschema import Draft202012Validator

SCHEMA_NOTICE = "JSON Schema requested; local validation alone does not prove server hard decoding."


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("Cannot read a required JSON file; check paths and JSON syntax.") from None


def asset_path(bundle, relative):
    if not isinstance(relative, str):
        raise ValueError("Invalid bundle asset path.")
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise ValueError("Missing asset or path outside the bundle.")
    return path


def load_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("native_social_labels_included") is not False
            or manifest.get("credentials_included") is not False
            or manifest.get("wire_protocol") != "object"
            or not is_object_profile(manifest.get("prompt_profile"))
            or not isinstance(manifest.get("cases"), list) or not manifest["cases"]):
        raise ValueError("Unsupported bundle or unsafe label provenance.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing input asset hash manifest.")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid asset SHA-256.")
        if hashlib.sha256(asset_path(bundle, relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Bundle integrity check failed; extract a fresh copy.")

    def checked_asset(relative):
        if not isinstance(relative, str) or relative not in files:
            raise ValueError("Referenced input asset is missing from the hash manifest.")
        return asset_path(bundle, relative)

    system = checked_asset("system_prompt.txt").read_text(encoding="utf-8").strip()
    cases = []
    for entry in manifest["cases"]:
        if not isinstance(entry.get("case_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["case_id"]):
            raise ValueError("Invalid case ID.")
        model = read_json(checked_asset(entry["input_file"]))
        if model.get("wire_protocol") != "object":
            raise ValueError("Only the current object protocol is supported.")
        if set(model["fields"]) != {"v", "dt", "ts", "e", "o", "d"}:
            raise ValueError("Input is outside the geometry-only whitelist.")
        ids = model["fields"]["v"]
        if len(ids) > 20 or len(ids) != len(set(ids)) or any(type(p) is not int or not 0 <= p <= 99 for p in ids):
            raise ValueError("Invalid current PID set.")
        if len(model["images"]) != 3:
            raise ValueError("Each task must have exactly three real images.")
        paths = [checked_asset(relative) for relative in model["images"]]
        if len(set(paths)) != 3:
            raise ValueError("Three distinct image assets are required.")
        Draft202012Validator.check_schema(model["response_schema"])
        fingerprint = hashlib.sha256(json.dumps({
            "system": system, "model": model,
            "images": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()
        cases.append({"entry": entry, "model": model, "paths": paths, "fingerprint": fingerprint,
                      "validator": Draft202012Validator(model["response_schema"])})
    if len({c["entry"]["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs.")
    return manifest, system, cases


def is_object_profile(value):
    """Accept the object profile and numbered labels in immutable old inputs.

    Compatibility is guarded by wire fields, Schema and hashes in load_bundle;
    new manifests always use the unnumbered name.
    """
    return isinstance(value, str) and re.fullmatch(r"object(?:_v[0-9]+)?", value) is not None


def groups(raw):
    return [(g["m"], g["r"], g["k"]) for g in raw["g"]]


def make_body(system, case, *, constraint="json-schema", max_tokens=1024, model="qwen3.8-27b-fp8"):
    content = []
    for i, path in enumerate(case["paths"]):
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower())
        if mime is None:
            raise ValueError("Unsupported image transport format")
        content += [{"type": "text", "text": f"IMAGE_{i}"},
                    {"type": "image_url", "image_url": {
                        "url": "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}]
    content.append({"type": "text", "text": case["model"]["user_prompt"]})
    response_format = {"type": "json_object"}
    if constraint == "json-schema":
        response_format = {"type": "json_schema", "json_schema": {
            "name": "social_nav_REL", "strict": True, "schema": case["model"]["response_schema"]}}
    return {"model": model, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": content}], "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0, "max_tokens": max_tokens,
            "response_format": response_format, "stream": True,
            "stream_options": {"include_usage": True}}


def usage_numbers(data):
    result = {k: data[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
              if type(data.get(k)) is int and data[k] >= 0}
    for field, names in (("prompt_tokens_details", ("cached_tokens", "image_tokens")),
                         ("completion_tokens_details", ("reasoning_tokens",))):
        detail = data.get(field)
        if isinstance(detail, dict):
            result.update({k: detail[k] for k in names if type(detail.get(k)) is int and detail[k] >= 0})
    return result


def read_stream(response, started, timing):
    chunks, usage, finish, model = [], {}, None, None
    reasoning, done = False, False
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        if not payload:
            continue
        event = json.loads(payload)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("Stream error")
        timing.setdefault("first_event_s", time.perf_counter() - started)
        observed_model = event.get("model")
        if isinstance(observed_model, str):
            model = observed_model
        if isinstance(event.get("usage"), dict):
            usage.update(usage_numbers(event["usage"]))
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Multiple completions")
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content") or delta.get("reasoning"):
                reasoning = True  # Do not retain hidden reasoning.
            text = delta.get("content")
            if text:
                if not isinstance(text, str):
                    raise ValueError("Unexpected content type")
                timing.setdefault("first_output_text_s", time.perf_counter() - started)
                timing["last_output_text_s"] = time.perf_counter() - started
                chunks.append(text)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    timing["stream_complete_s"] = time.perf_counter() - started
    if not done or finish is None:
        raise ValueError("Incomplete stream")
    return {"raw_content": "".join(chunks), "usage": usage,
            "finish_reason": finish, "server_model": model,
            "reasoning_content_present": reasoning}


def validate_rel(raw, case):
    case["validator"].validate(raw)
    claimed, minima, canonical = [], [], True
    for members, relation, ordered in groups(raw):
        if type(ordered) is not int:
            raise ValueError("Role/order flag must be an integer")
        if ordered and relation not in {"q", "p"}:
            raise ValueError("Order/role flag incompatible with relation")
        if not ordered and members != sorted(members):
            canonical = False
        if ordered and relation == "p" and members[2:] != sorted(members[2:]):
            canonical = False
        claimed.extend(members)
        minima.append(min(members))
    claimed.extend(raw["u"])
    if any(type(pid) is not int for pid in claimed):
        raise ValueError("PID must be an integer, not a string/float/bool")
    if len(claimed) != len(set(claimed)) or not set(claimed).issubset(case["model"]["fields"]["v"]):
        raise ValueError("Invalid or duplicate membership")
    canonical &= minima == sorted(minima) and raw["u"] == sorted(raw["u"])
    # Like the original decoder, unordered lists may be sorted losslessly;
    # IDs, relationships and q/p role order must never be repaired.
    return bool(canonical)


def distribution(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values), "p50": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1]}


def summary(records, planned, phase_wall, settings):
    submitted = [r for r in records if r["submitted"]]
    valid = [r for r in submitted if r["valid"]]
    on_time = sum(r.get("valid_within_deadline", False) for r in submitted)
    metrics = {"settings": settings, "schema_notice": SCHEMA_NOTICE,
               "planned_arrivals": planned, "processed_arrivals": len(records),
               "submitted": len(submitted), "dropped_busy": sum(not r["submitted"] for r in records),
               "cancelled_after_error": planned - len(records), "valid": len(valid),
               "failed": len(submitted) - len(valid), "valid_within_deadline": on_time,
               "valid_within_deadline_ratio_all_arrivals": on_time / planned,
               "valid_ratio_submitted": len(valid) / len(submitted) if submitted else 0,
               "valid_results_per_s": len(valid) / phase_wall if phase_wall > 0 else 0,
               "phase_wall_s": phase_wall,
               "first_task_without_explicit_warmup": submitted[0] if submitted and settings["warmups"] == 0 else None,
               "latency_all_submitted_s": {}, "latency_valid_only_s": {}, "token_usage": {}}
    for key in ("prepare_s", "first_output_text_s", "stream_complete_s", "api_complete_s",
                "end_to_end_s", "arrival_to_result_s"):
        for name, rows in (("latency_all_submitted_s", submitted), ("latency_valid_only_s", valid)):
            metrics[name][key] = distribution([r["timing"].get(key) for r in rows])
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "image_tokens", "reasoning_tokens"):
        metrics["token_usage"][key] = distribution([r.get("usage", {}).get(key) for r in submitted])
    return metrics


def save_json(path, value, key=""):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if key:
        encoded = encoded.replace(key, "[REDACTED]")
    Path(path).write_text(encoded + "\n", encoding="utf-8")
