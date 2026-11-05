"""Read-only checks for the current self-hosted VLM service; no inference or deployment."""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import vlm_api_common as common
from .project_config import ROOT

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
                args = meta.get("server_args", meta)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/vlm_server.local.json')
    parser.add_argument('--server-info',action='store_true')
    args=parser.parse_args()
    config=load_config(args.config)
    with httpx.Client(timeout=15,trust_env=False,follow_redirects=False) as client:
        result=check_server(client,config,info=args.server_info)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='ready' else 1


if __name__=='__main__':
    raise SystemExit(main())
