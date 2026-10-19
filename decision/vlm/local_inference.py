"""Offline REL inference with local Transformers and token-level XGrammar constraints.

No HTTP client, service startup, automatic downloads, retries or output repair.
GPU dependencies are imported only when constructing the explicitly selected engine.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import benchmark_defaults as defaults
from . import vlm_api_common as common
from .project_config import ROOT, load_config, project_path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def local_messages(body):
    """Translate transport image keys only; preserve bytes, order and every text."""
    messages = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, list):
            content = [({"type": "image", "image": item["image_url"]["url"]}
                        if item["type"] == "image_url" else dict(item)) for item in content]
        messages.append({"role": message["role"], "content": content})
    return messages


def compare_reference(rows, reference, settings):
    """Read answers only after inference; pair exact source and logical request."""
    reference = Path(reference)
    old_settings = common.read_json(reference / "summary.json")["settings"]
    controlled = ("model", "thinking", "temperature", "max_tokens", "constraint_requested")
    if any(old_settings.get(k) != settings[k] for k in controlled):
        raise ValueError("Reference inference settings differ")
    old_rows = [json.loads(line) for line in (reference / "requests.jsonl").read_text().splitlines() if line.strip()]
    old = {row["case_id"]: row for row in old_rows}
    if len(old) != len(old_rows):
        raise ValueError("Duplicate reference case IDs")
    pairs = []
    for row in rows:
        other = old.get(row["case_id"])
        if other is None:
            raise ValueError("Missing reference case: " + row["case_id"])
        if (row["input_fingerprint"] != other.get("input_fingerprint")
                or row["reference_request_sha256"] != other.get("request_sha256")):
            raise ValueError("Reference input/request binding differs")
        pairs.append({"case_id": row["case_id"], "both_valid": bool(row["valid"] and other["valid"]),
                      "identical_raw": row.get("raw_content") is not None and row.get("raw_content") == other.get("raw_content"),
                      "identical_rel": bool(row["valid"] and other["valid"] and row["canonical_rel"] == other["canonical_rel"]),
                      "same_prompt_tokens": row.get("usage", {}).get("prompt_tokens") == other.get("usage", {}).get("prompt_tokens")})
    return {"reference": str(reference), "paired_cases": len(pairs), "cases": pairs,
            "identical_raw": sum(p["identical_raw"] for p in pairs),
            "identical_rel": sum(p["identical_rel"] for p in pairs),
            "reference_read_after_generation": True, "semantic_accuracy_evaluated": False,
            "same_model_execution_or_latency_assumed": False}


class LocalEngine:
    def __init__(self, model_path, options):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        if options.get("gpus") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, options["gpus"]))
        import torch
        import xgrammar as xgr
        from transformers import AutoProcessor, FineGrainedFP8Config, Qwen3_5ForConditionalGeneration

        self.torch, self.xgr = torch, xgr
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check sandbox/device access before changing the environment")
        torch.manual_seed(options["seed"])
        torch.set_float32_matmul_precision("highest")
        self.context_length = options["context_length"]
        source_config = common.read_json(model_path / "config.json")
        if source_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            raise ValueError("Expected the configured Qwen multimodal architecture")
        qconfig = source_config.get("quantization_config", {})
        if qconfig.get("quant_method") != "fp8":
            raise ValueError("Expected the frozen FP8 checkpoint; do not silently change model precision")
        qconfig = dict(qconfig, dequantize=True)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.bfloat16, quantization_config=FineGrainedFP8Config(**qconfig),
            device_map=options["device_map"],
            max_memory={i: f"{options['max_memory_gib']}GiB" for i in range(torch.cuda.device_count())},
            attn_implementation="sdpa").eval()
        if any(str(d) in {"cpu", "disk"} for d in self.model.hf_device_map.values()):
            raise RuntimeError("GPU memory insufficient; implicit CPU/disk offload is not enabled")
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        info = xgr.TokenizerInfo.from_huggingface(self.processor.tokenizer,
                vocab_size=self.model.config.text_config.vocab_size, stop_token_ids=self.eos_ids)
        self.compiler = xgr.GrammarCompiler(info)
        from importlib.metadata import PackageNotFoundError, version
        from transformers.models.qwen3_5 import modeling_qwen3_5
        packages = {n: version(n) for n in ("torch", "transformers", "tokenizers", "accelerate", "xgrammar", "safetensors", "pillow")}
        for name in ("fla-core", "causal-conv1d", "triton"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                pass
        gdn_layers = [module for module in self.model.modules()
                      if isinstance(module, modeling_qwen3_5.Qwen3_5GatedDeltaNet)]
        selected_implementations = {}
        for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
            selected_implementations[name] = dict(Counter(
                f"{function.__module__}.{function.__name__}" if function is not None else "torch.nn.Conv1d"
                for module in gdn_layers for function in [getattr(module, name)]))
        dtype_counts = Counter()
        for parameter in self.model.parameters():
            dtype_counts[str(parameter.dtype)] += parameter.numel()
        self.metadata = {
            "packages": packages,
            "gdn": {"layers": len(gdn_layers), "fast_path_available": modeling_qwen3_5.is_fast_path_available,
                    "selected_implementations": selected_implementations,
                    "selection_metadata_is_not_execution_proof": True},
            "asset_sha256": {n: hashlib.sha256((model_path / n).read_bytes()).hexdigest() for n in
                ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "generation_config.json", "model.safetensors.index.json")},
            "device_map": self.model.hf_device_map,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "loaded_precision": "BF16_dequantized_from_FP8", "attention_backend": "sdpa",
            "parameter_elements_by_dtype": dict(dtype_counts),
            "speculative_decoding": False, "grammar_any_whitespace": True,
            "max_memory_is_weight_placement_budget_not_total_peak_limit": True}

    def generate(self, body):
        from transformers import GenerationConfig
        from xgrammar.contrib.hf import LogitsProcessor
        torch = self.torch
        messages = local_messages(body)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt")
        n = inputs["input_ids"].shape[-1]
        if n + body["max_tokens"] > self.context_length:
            raise ValueError("Prompt plus output budget exceeds context; input is not truncated")
        input_ids = inputs["input_ids"][0].tolist()
        grid = inputs.get("image_grid_thw")
        compiled = self.compiler.compile_json_schema(body["response_format"]["json_schema"]["schema"], any_whitespace=True)
        grammar = LogitsProcessor(compiled)  # Matchers must be new for every generate().
        inputs = inputs.to(self.model.device)
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=body["max_tokens"],
                repetition_penalty=1.0, use_cache=True, eos_token_id=self.eos_ids,
                pad_token_id=self.model.generation_config.pad_token_id,
                bos_token_id=self.model.generation_config.bos_token_id)
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=config, logits_processor=[grammar])
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        duration = time.perf_counter() - started
        generated = result[0, n:].tolist()
        stopped = bool(generated and generated[-1] in self.eos_ids)
        grammar_complete = bool(stopped and grammar.matchers[0].accept_token(generated[-1])
                                and grammar.matchers[0].is_terminated())
        content = generated[:-1] if stopped else generated
        return {"raw_content": self.processor.tokenizer.decode(content, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False), "generated_token_ids": generated,
                "finish_reason": "stop" if stopped else "length", "grammar_terminated": grammar_complete,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_ids_sha256": fingerprint(input_ids),
                "image_grid_thw": grid.tolist() if grid is not None else None,
                "usage": {"prompt_tokens": n, "completion_tokens": len(generated),
                          "image_tokens": input_ids.count(self.model.config.image_token_id)},
                "generate_s": duration}


def arguments(argv=None):
    config = load_config()
    public = defaults.from_project(config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=project_path(config["development"]["bundle"]))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vlm_local.local.json")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs; selected before importing torch")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=public["limit"])
    parser.add_argument("--case-id", action="append", help="Explicit subset of frozen case IDs")
    parser.add_argument("--max-tokens", type=int, default=public["max_tokens"])
    parser.add_argument("--reference", type=Path, help="Existing remote result; read only AFTER generation")
    parser.add_argument("--dry-run", action="store_true", help="Inputs only; no private config, GPU imports or inference")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 10000 or not 1 <= args.max_tokens <= 4096:
        parser.error("Invalid sample or output token budget")
    return args


def run(args, engine_factory=LocalEngine):
    config = load_config()
    manifest, system, cases = common.load_bundle(args.bundle)
    if args.case_id:
        requested = set(args.case_id)
        if not requested.issubset({c["entry"]["case_id"] for c in cases}):
            raise ValueError("Unknown case ID")
        cases = [c for c in cases if c["entry"]["case_id"] in requested]
    cases = cases[:args.limit]
    if args.dry_run:
        print(json.dumps({"backend": "local_transformers", "cases": [c["entry"]["case_id"] for c in cases],
                          "inference_requests": 0, "network_requests": 0, "constraint": "xgrammar_json_schema"}))
        return 0
    if args.out is None:
        raise ValueError("Specify a new --out directory outside frozen inputs and model assets")
    options = dict(config["local_inference"])
    if args.config.exists():
        local = common.read_json(args.config)
        if not isinstance(local, dict) or set(local) - {"model_path", "gpus", "max_memory_gib"}:
            raise ValueError("Local config accepts only model_path, gpus and max_memory_gib")
        options.update(local)
    if args.gpus is not None:
        if not re.fullmatch(r"\d+(?:,\d+)*", args.gpus):
            raise ValueError("GPU IDs must be comma-separated nonnegative integers")
        options["gpus"] = [int(v) for v in args.gpus.split(",")]
    gpus = options.get("gpus")
    if gpus is not None and (not isinstance(gpus, list) or not gpus or any(type(g) is not int or g < 0 for g in gpus) or len(set(gpus)) != len(gpus)):
        raise ValueError("Invalid GPU selection")
    if type(options["max_memory_gib"]) is not int or options["max_memory_gib"] <= 0:
        raise ValueError("Invalid per-GPU memory budget")
    model_path = (args.model_path or project_path(options.get("model_path", config["model"]["path"]))).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError("Set the actual local model directory")
    out = args.out.resolve()
    if any(out.is_relative_to(p.resolve()) for p in (args.bundle, model_path)):
        raise ValueError("Output must be outside frozen input/model assets")
    out.mkdir(parents=True, exist_ok=False)
    settings = {"backend": "local_transformers", "model": config["model"]["served_name"],
                "model_path": str(model_path), "thinking": False, "temperature": 0,
                "constraint_requested": "json-schema", "max_tokens": args.max_tokens,
                "started_at": datetime.now(timezone.utc).isoformat(), "runtime": options,
                "inference_network_requests": 0, "automatic_retries": 0,
                "source": "frozen_geometry_only_JRDB_development_NOT_semantic_gold"}
    common.save_json(out / "settings.json", settings)
    rows = []
    try:
        started = time.perf_counter()
        engine = engine_factory(model_path, options)
        settings.update(load_s=time.perf_counter() - started, engine=engine.metadata)
        common.save_json(out / "settings.json", settings)
        with (out / "requests.jsonl").open("x", encoding="utf-8") as log:
            for case in cases:
                body = common.make_body(system, case, model=settings["model"], max_tokens=args.max_tokens)
                row = {"case_id": case["entry"]["case_id"], "scene": case["entry"]["scene"],
                       "input_fingerprint": case["fingerprint"], "reference_request_sha256": fingerprint(body),
                       "reference_http_body_not_sent": True, "submitted": True, "valid": False,
                       "status": "error", "local_schema_valid": False, "timing": {}}
                start = time.perf_counter()
                stage = "generate"
                try:
                    row.update(engine.generate(body))
                    row["timing"]["generate_s"] = row.pop("generate_s")
                    stage = "completion"
                    if row["finish_reason"] != "stop" or not row["grammar_terminated"]:
                        raise ValueError("Truncated or unfinished constrained generation")
                    if re.search(r"</?(?:think|analysis)\b", row["raw_content"], re.I):
                        raise ValueError("Reasoning appeared despite non-thinking request")
                    stage = "json_schema"
                    raw = json.loads(row["raw_content"])
                    case["validator"].validate(raw)
                    row["local_schema_valid"] = True
                    stage = "membership"
                    row["canonical_order"] = common.validate_rel(raw, case)
                    row.update(response=raw, canonical_rel=common.canonical_rel(raw), valid=True, status="success")
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error)[:1000], failure_stage=stage)
                row["timing"]["end_to_end_s"] = time.perf_counter() - start
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps({"case_id": row["case_id"], "valid": row["valid"],
                                  "completion_tokens": row.get("usage", {}).get("completion_tokens")}), flush=True)
                if not row["valid"]:
                    break
        summary = {"settings": settings, "planned_cases": len(cases), "submitted": len(rows),
                   "valid": sum(row["valid"] for row in rows), "cancelled_after_error": len(cases) - len(rows),
                   "semantic_accuracy_evaluated": False, "online_deadline_evaluated": False,
                   "latency_valid_only_s": {"end_to_end_s": common.distribution([r["timing"]["end_to_end_s"] for r in rows if r["valid"]])}}
        if args.reference:
            summary["reference_comparison"] = compare_reference(rows, args.reference, settings)
        common.save_json(out / "summary.json", summary)
        return 0 if len(rows) == len(cases) and all(row["valid"] for row in rows) else 1
    except Exception as error:
        common.save_json(out / "summary.json", {"settings": settings, "submitted": len(rows),
                         "valid": sum(r["valid"] for r in rows), "error_type": type(error).__name__, "error": str(error)[:2000]})
        raise


def main(argv=None):
    try:
        return run(arguments(argv))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Local inference stopped: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
