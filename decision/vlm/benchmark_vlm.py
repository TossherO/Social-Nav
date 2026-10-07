"""Portable, bounded JRDB benchmark for a manually deployed SGLang endpoint.

No cloud credentials, deployment actions, cache resets, retries or installations.
The packaged copy and repository execution share the same minimal utilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx
from jsonschema import ValidationError

if __package__:
    from . import vlm_api_common as common
    from . import benchmark_defaults as defaults
else:
    import vlm_api_common as common
    import benchmark_defaults as defaults

HERE = Path(__file__).resolve().parent
SCHEMA_NOTICE = ("SGLang JSON Schema requested plus independent local validation; "
                 "the remote app's grammar/speculation configuration is not independently verified.")
SERVER_FIELDS = ("model_path", "served_model_name", "context_length", "max_running_requests",
                 "quantization", "kv_cache_dtype", "mamba_ssm_dtype", "attention_backend",
                 "grammar_backend", "speculative_algorithm", "speculative_draft_model_path",
                 "disable_radix_cache", "allow_auto_truncate", "tp_size")


def load_config(path):
    config = common.read_json(path)
    if not isinstance(config, dict) or set(config) - {"base_url", "api_key", "model", "allow_insecure_http"}:
        raise ValueError("Expected only base_url, api_key, model and allow_insecure_http")
    base = config.get("base_url", "")
    key, model = config.get("api_key", ""), config.get("model", "")
    insecure = config.get("allow_insecure_http", False)
    if not all(isinstance(v, str) for v in (base, key, model)) or type(insecure) is not bool:
        raise ValueError("Invalid server configuration types")
    base, key, model = base.strip().rstrip("/"), key.strip(), model.strip()
    if not base or any(c.isspace() for c in base + key + model):
        raise ValueError("Fill a valid server URL; no whitespace in URL, token or model ID")
    url = urlsplit(base)
    if (not url.hostname or url.scheme not in {"http", "https"} or url.username or url.password
            or url.query or url.fragment or url.port == 0):
        raise ValueError("Use a plain server URL without credentials, query or fragment")
    if url.hostname in {"maas.qianwenaiapi.com", "dashscope.aliyuncs.com", "api.openai.com"}:
        raise ValueError("This runner is for your own GPU server, not a paid official model API")
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"} and not insecure:
        raise ValueError("Public HTTP requires explicit allow_insecure_http=true; prefer HTTPS or an SSH tunnel")
    if url.path.endswith(("/chat/completions", "/models", "/messages")):
        raise ValueError("Supply the API base URL, not an individual API method")
    if not url.path.rstrip("/").endswith("/v1"):
        base += "/v1"
    return {"base_url": base, "api_key": key, "model": model,
            "allow_insecure_http": insecure, "origin": url.scheme + "://" + url.netloc}


def headers(config):
    result = {"Content-Type": "application/json"}
    if config["api_key"]:
        result["Authorization"] = "Bearer " + config["api_key"]
    return result


def choose_model(data, requested=""):
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("The models endpoint did not return an OpenAI-compatible model list")
    ids = sorted({m["id"] for m in data["data"] if isinstance(m, dict)
                  and isinstance(m.get("id"), str) and m["id"]})
    if any(len(m) > 512 or any(c.isspace() for c in m) for m in ids):
        raise ValueError("Unexpected model identifiers")
    if requested:
        if requested not in ids:
            raise ValueError("Configured model not in the endpoint's model list")
        return requested, ids
    matches = [m for m in ids if re.search(r"qwen[-_/]?(?:3[._-]8|38).*27b", m, re.I)]
    if len(matches) != 1:
        raise ValueError("Model selection is ambiguous; set model to the exact ID shown by --check")
    return matches[0], ids


def check_server(client, config, *, info=False):
    result = {"status": "check_failed", "inference_requests": 0, "model_ids": []}
    started = time.perf_counter()
    try:
        response = client.get(config["base_url"] + "/models", headers=headers(config))
        result["http_status"] = response.status_code
        if response.status_code != 200:
            return result
        data = response.json()
        # Record only IDs; never a server error body, URL query, cookie or token.
        result["model_ids"] = sorted({m["id"] for m in data.get("data", [])
                                      if isinstance(m, dict) and isinstance(m.get("id"), str)})
        model, ids = choose_model(data, config["model"])
        result.update(status="ready", selected_model=model, model_ids=ids,
                      checkpoint_identity_verified=False)
        if info:
            # Optional read-only metadata; some app proxies expose /v1 only.
            server = client.get(config["base_url"][:-3] + "/get_server_info", headers=headers(config))
            result["server_info_http_status"] = server.status_code
            if server.status_code == 200:
                meta = server.json()
                args = meta.get("server_args", {})
                if isinstance(args, dict):
                    result["server_args"] = {k: args[k] for k in SERVER_FIELDS if k in args
                                             and (args[k] is None or type(args[k]) in {str, bool, int, float})}
                for name in ("version", "sglang_version"):
                    if isinstance(meta.get(name), str):
                        result[name] = meta[name][:100]
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
        field = "server_info_error_type" if result["status"] == "ready" else "error_type"
        result[field] = type(error).__name__
    finally:
        result["check_s"] = time.perf_counter() - started
    return result


def make_body(system, case, model, *, constraint="json-schema", max_tokens=1024):
    return common.make_body(system, case, model=model, constraint=constraint, max_tokens=max_tokens)


def execute(client, config, system, case, model, constraint, max_tokens, deadline, target=None):
    started = time.perf_counter()
    row = {"case_id": case["entry"]["case_id"], "scene": case["entry"]["scene"],
           "input_fingerprint": case["fingerprint"],
           "pid_count": len(case["model"]["fields"]["v"]), "submitted": True,
           "valid": False, "local_schema_valid": False, "server_hard_schema_verified": False,
           "constraint_requested": constraint, "wire_protocol": "object",
           "status": "request_error", "timing": {}}
    timing, stage, api_started = row["timing"], "prepare", None
    try:
        body = make_body(system, case, model, constraint=constraint, max_tokens=max_tokens)
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        row.update(request_bytes=len(payload), request_sha256=hashlib.sha256(payload).hexdigest())
        timing["prepare_s"] = time.perf_counter() - started
        api_started, stage = time.perf_counter(), "transport"
        with client.stream("POST", config["base_url"] + "/chat/completions", content=payload,
                           headers=headers(config)) as response:
            row["http_status"] = response.status_code
            timing["response_headers_s"] = time.perf_counter() - api_started
            if response.status_code != 200:
                row["status"] = "http_error"
            else:
                stage = "stream"
                reply = common.read_stream(response, api_started, timing)
                row["status"] = "invalid_response"
                # Reject, rather than strip, textual hidden reasoning. Do not log it.
                has_think = bool(re.search(r"</?(?:think|analysis)\b", reply["raw_content"], re.I))
                has_think |= reply["reasoning_content_present"] or reply["usage"].get("reasoning_tokens", 0) > 0
                if has_think:
                    row.update(usage=reply["usage"], finish_reason=reply["finish_reason"],
                               server_model=reply["server_model"], reasoning_content_present=True)
                    stage = "completion_metadata"
                    raise ValueError("Reasoning appeared despite the non-thinking request")
                row.update(reply)
        timing["api_complete_s"] = time.perf_counter() - api_started
        if row["http_status"] == 200:
            validation_started, stage = time.perf_counter(), "json_parse"
            raw = json.loads(row["raw_content"])
            stage = "schema"
            case["validator"].validate(raw)
            row["local_schema_valid"] = True
            stage = "membership"
            row["canonical_order"] = common.validate_rel(raw, case)
            stage = "model_identity"
            if row["server_model"] != model:
                raise ValueError("Response model differs from the discovered/requested model")
            stage = "completion_metadata"
            if row["finish_reason"] != "stop":
                raise ValueError("Response was truncated or did not finish normally")
            canonical = {"g": [], "u": sorted(raw["u"])}
            for members, relation, ordered in common.groups(raw):
                members = sorted(members) if not ordered else list(members)
                if ordered and relation == "p":
                    members[2:] = sorted(members[2:])
                canonical["g"].append([members, relation, ordered])
            canonical["g"].sort(key=lambda g: min(g[0]))
            rel = {"g": [{"m": m, "r": r, "k": k} for m, r, k in canonical["g"]], "u": canonical["u"]}
            timing["validation_s"] = time.perf_counter() - validation_started
            row.update(valid=True, status="success", response=raw, canonical_rel=rel)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, AttributeError, ValidationError) as error:
        row.update(error_type=type(error).__name__, failure_stage=stage)
        if isinstance(error, ValidationError):
            row.update(schema_error_keyword=error.validator, schema_error_path=list(error.absolute_path))
    if api_started is not None:
        timing.setdefault("api_complete_s", time.perf_counter() - api_started)
    finished = time.perf_counter()
    timing["end_to_end_s"] = finished - started
    if target is not None:
        timing.update(arrival_to_result_s=finished - target, start_lag_s=max(0, started - target))
    row["valid_within_deadline"] = bool(row["valid"] and timing.get("arrival_to_result_s", timing["end_to_end_s"]) <= deadline)
    return row, finished


def prime_prefix(client, config, system, model, constraint):
    """Optional one text-only request; never prime a measured image window."""
    body = {"model": model, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": 'Warm-up only: no observations. Return JSON {"g":[],"u":[]}.'}],
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "max_tokens": 32,
            "stream": True, "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"}}
    if constraint == "json-schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "warmup",
            "strict": True, "schema": {"type": "object", "additionalProperties": False,
                "required": ["g", "u"], "properties": {"g": {"type": "array", "maxItems": 0},
                                                        "u": {"type": "array", "maxItems": 0}}}}}
    started, timing = time.perf_counter(), {}
    row = {"phase": "text_only_warmup_NOT_task", "images_sent": 0, "valid": False}
    try:
        with client.stream("POST", config["base_url"] + "/chat/completions", json=body,
                           headers=headers(config)) as response:
            row["http_status"] = response.status_code
            if response.status_code == 200:
                reply = common.read_stream(response, started, timing)
                row.update(usage=reply["usage"], finish_reason=reply["finish_reason"])
                row["valid"] = (reply["server_model"] == model and reply["finish_reason"] == "stop"
                                and not reply["reasoning_content_present"]
                                and reply["usage"].get("reasoning_tokens", 0) == 0
                                and json.loads(reply["raw_content"]) == {"g": [], "u": []})
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        row["error_type"] = type(error).__name__
    timing["end_to_end_s"] = time.perf_counter() - started
    row["timing"] = timing
    return row


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = HERE.parents[1] if __package__ else HERE
    if __package__:
        defaults_path = HERE / "configs/project.json"
        public_config = common.read_json(defaults_path)
        settings = defaults.from_project(public_config)
        configured_bundle = Path(public_config["development"]["bundle"]).expanduser()
        default_bundle = configured_bundle if configured_bundle.is_absolute() else root / configured_bundle
    else:
        defaults_path = HERE / "benchmark_defaults.json"
        settings = defaults.load_portable(HERE)
        default_bundle = HERE
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--config", type=Path, default=root / "configs/vlm_server.local.json" if __package__ else HERE / "server_config.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Assets/public defaults check; no private config, Key or network")
    parser.add_argument("--check", action="store_true", help="GET model list only, no inference")
    parser.add_argument("--server-info", action="store_true", help="Additionally GET sanitized optional server metadata")
    parser.add_argument("--proxy", help="Explicit trusted proxy; environment proxies ignored")
    parser.add_argument("--mode", choices=("serial", "paced"), default=settings["mode"])
    parser.add_argument("--rate-hz", type=float, help="Override arrival rate; without --deadline-s also set deadline=1/rate")
    parser.add_argument("--deadline-s", type=float, help="Result freshness deadline; independent of HTTP timeout")
    parser.add_argument("--limit", type=int, default=settings["limit"], help="Public default is three cases; select more explicitly")
    parser.add_argument("--warmups", type=int, choices=(0, 1), default=settings["warmups"])
    parser.add_argument("--max-tokens", type=int, default=settings["max_tokens"])
    parser.add_argument("--timeout", type=float, default=settings["timeout"])
    parser.add_argument("--constraint", choices=("json-schema", "json-object"), default=settings["constraint"])
    args = parser.parse_args(argv)
    rate_override = args.rate_hz is not None
    if not rate_override:
        args.rate_hz = settings["rate_hz"]
    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0:
        parser.error("Arrival rate must be finite and positive")
    if args.deadline_s is None:
        args.deadline_s = 1 / args.rate_hz if rate_override else settings["deadline_s"]
    try:
        defaults.validate({name: getattr(args, name) for name in defaults.FIELDS})
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run and args.check:
        parser.error("--dry-run and --check are mutually exclusive")
    args.defaults_source = str(defaults_path)
    return args


def run(args):
    manifest, system, cases = common.load_bundle(args.bundle)
    cases = cases[:args.limit]
    if not cases:
        raise ValueError("No cases in the bundle")
    if args.dry_run:
        sizes = [len(json.dumps(make_body(system, c, "qwen3.8-27b-fp8", constraint=args.constraint,
                                         max_tokens=args.max_tokens), ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode()) for c in cases]
        print(json.dumps({"status": "offline_bundle_verified", "network_requests": 0,
            "inference_requests": 0, "available_cases": len(manifest["cases"]), "selected_cases": len(cases),
            "scenes": sorted({c["entry"]["scene"] for c in cases}), "images_per_request": 3,
            "planned_task_requests": len(cases), "optional_text_warmups": args.warmups,
            "constraint": args.constraint, "non_thinking_requested": True,
            "rate_hz": args.rate_hz, "deadline_s": args.deadline_s,
            "max_tokens": args.max_tokens, "defaults_source": args.defaults_source,
            "request_bytes": common.distribution(sizes), "schema_notice": SCHEMA_NOTICE}, indent=2))
        return 0
    config = load_config(args.config)
    out = args.out or args.bundle / "results" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    key = config["api_key"]
    options = {"proxy": args.proxy, "trust_env": False, "follow_redirects": False,
               "timeout": httpx.Timeout(args.timeout, connect=min(10, args.timeout)),
               "limits": httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=60)}
    with httpx.Client(**options) as client:
        check = check_server(client, config, info=args.server_info)
        common.save_json(out / "model_check.json", check, key)
        if args.check:
            safe = json.dumps(check, ensure_ascii=False, indent=2)
            print(safe.replace(key, "[REDACTED]") if key else safe)
            return 0 if check["status"] == "ready" else 1
        if check["status"] != "ready":
            print("Model check failed; no inference sent. See model_check.json and set the exact model ID if needed.")
            return 1
        model = check["selected_model"]
        settings = {"backend": "remote_SGLang_OpenAI_compatible", "model": model,
            "prompt_profile": manifest.get("prompt_profile"), "mode": args.mode,
            "deadline_s": args.deadline_s, "rate_hz": args.rate_hz, "warmups": args.warmups,
            "timeout_s": args.timeout, "defaults_source": args.defaults_source,
            "distinct_cases": len(cases), "automatic_retries": 0, "repeats": 1,
            "max_in_flight": 1, "connection_reuse": True, "proxy_used": bool(args.proxy),
            "constraint_requested": args.constraint, "server_hard_schema_verified": False,
            "thinking": False, "thinking_control": "chat_template_kwargs.enable_thinking=false",
            "temperature": 0, "max_tokens": args.max_tokens, "model_check_excluded_from_task_timing": True,
            "cache_flushed": False, "server_configuration_modified": False,
            "rendering_detection_tracking_included": False,
            "source": "frozen_geometry_only_JRDB_development_NOT_semantic_gold"}
        common.save_json(out / "settings.json", settings, key)
        if args.warmups:
            warmup = prime_prefix(client, config, system, model, args.constraint)
            common.save_json(out / "warmup_00.json", warmup, key)
            if not warmup["valid"]:
                print("Text-only warm-up failed; no image task sent, no automatic retry.")
                return 1
        phase_start = time.perf_counter()
        rows, busy_until = [], phase_start
        with (out / "requests.jsonl").open("x", encoding="utf-8") as log:
            for i, case in enumerate(cases):
                target = phase_start + i / args.rate_hz if args.mode == "paced" else None
                if target is not None and busy_until > target:
                    row = {"case_id": case["entry"]["case_id"], "submitted": False, "valid": False,
                           "status": "dropped_busy", "timing": {}}
                else:
                    if target is not None:
                        time.sleep(max(0, target - time.perf_counter()))
                    row, busy_until = execute(client, config, system, case, model,
                                             args.constraint, args.max_tokens, args.deadline_s, target)
                row["index"] = i
                rows.append(row)
                encoded = json.dumps(row, ensure_ascii=False)
                log.write((encoded.replace(key, "[REDACTED]") if key else encoded) + "\n")
                log.flush()
                timing = row["timing"]
                print(f"{i+1}/{len(cases)} {row['case_id']} {row['status']} "
                      f"complete={timing.get('end_to_end_s')} ttft={timing.get('first_output_text_s')} "
                      f"output={row.get('usage', {}).get('completion_tokens')} "
                      f"cached={row.get('usage', {}).get('cached_tokens')}", flush=True)
                if row["submitted"] and not row["valid"]:
                    print("Stopped on the first failed task; no fallback or automatic retry.")
                    break
        report = common.summary(rows, len(cases), time.perf_counter() - phase_start, settings)
        report["schema_notice"] = SCHEMA_NOTICE
        report["benchmark_scope"] = "client_image_encoding_HTTP_complete_stream_and_local_validation"
        report["limitations"] = ["development_NOT_blind_or_semantic_accuracy_test", "oracle_geometry_and_PID",
            "small_sample_NOT_latency_SLA_or_navigation_loop", "server_config_and_weights_not_independently_verified",
            "no_hidden_reasoning_or_auth_values_logged", "cache_state_not_controlled"]
        common.save_json(out / "summary.json", report, key)
    compact = {k: report[k] for k in ("planned_arrivals", "submitted", "dropped_busy", "valid", "failed",
                                      "valid_within_deadline_ratio_all_arrivals")}
    compact["valid_end_to_end_s"] = report["latency_valid_only_s"]["end_to_end_s"]
    compact["valid_ttft_s"] = report["latency_valid_only_s"]["first_output_text_s"]
    print(json.dumps(compact, indent=2))
    print("Saved model_check.json, settings.json, requests.jsonl and summary.json in:", out)
    return 0 if report["failed"] == 0 and report["cancelled_after_error"] == 0 else 1


def main():
    try:
        return run(arguments())
    except (ValueError, OSError, KeyError, TypeError, ValidationError) as error:
        print("Configuration/bundle/output error:", type(error).__name__)
        print("Check your own server URL, API-port access, exact model ID and a NEW output directory.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
