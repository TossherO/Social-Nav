"""Wheelchair VLM transport and response helpers; no dataset or model loading."""
import base64
import json
from pathlib import Path
import time

from jsonschema import Draft202012Validator

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

def save_json(path, value, key=""):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if key:
        encoded = encoded.replace(key, "[REDACTED]")
    Path(path).write_text(encoded + "\n", encoding="utf-8")


def groups_schema(pids):
    if (not isinstance(pids,list) or any(type(p) is not int or not 0 <= p <= 99 for p in pids)
            or len(set(pids)) != len(pids)):
        raise ValueError('Invalid current PID set')
    if len(pids)<2:
        return dict(type='object',additionalProperties=False,properties={})
    value=dict(type='array',minItems=1,maxItems=len(pids)//2,
               items=dict(type='array',minItems=2,maxItems=len(pids),uniqueItems=True,
                          items=dict(type='integer',enum=pids)))
    return dict(type='object',additionalProperties=False,
                properties={name:value for name in ('walking','talking','photographing','other')})


def validate_groups(raw,pids,schema=None):
    Draft202012Validator(schema or groups_schema(pids)).validate(raw)
    claimed=[]
    for activity,groups in raw.items():
        if activity not in {'walking','talking','photographing','other'}:
            raise ValueError('Unsupported activity')
        for members in groups:
            if len(members)<2 or any(type(p) is not int or p not in pids for p in members):
                raise ValueError('Invalid group membership')
            claimed.extend(members)
    if len(set(claimed)) != len(claimed):
        raise ValueError('A PID may occur only once across all groups')
