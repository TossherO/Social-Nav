"""Local engine for frozen wheelchair VLM requests.

Only constructs GPU dependencies after resource selection and explicit fallback
reason. No downloads, HTTP requests, retries, source-label reads or output repair.
"""
from collections import Counter
import hashlib
import json
import os
import time

from . import vlm_api_common as common

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
