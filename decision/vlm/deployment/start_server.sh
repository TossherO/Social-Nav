#!/usr/bin/env bash
set -Eeuo pipefail
service_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$service_root/runtime_env.sh"

# Link JIT extensions against the installed driver; do not load CUDA stubs at runtime.
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"

sglang_profile="${SGLANG_PROFILE:-dspark}"
prefill_chunk=2048
case "$sglang_profile" in
  mtp|mtp_previous)
    memory_fraction=0.80
    if [[ "$sglang_profile" == mtp ]]; then
      prefill_chunk=8192
    fi
    acceleration_args=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
      --max-mamba-cache-size 6
    )
    ;;
  baseline)
    memory_fraction=0.85
    acceleration_args=()
    ;;
  dspark)
    memory_fraction=0.85
    prefill_chunk=8192
    draft_model_dir="${SGLANG_DRAFT_MODEL_DIR:-$SGLANG_ROOT/models/Qwen3.8-27B-DSpark}"
    test -f "$draft_model_dir/config.json"
    test -f "$draft_model_dir/model.safetensors"
    export SGLANG_RAGGED_VERIFY_MODE=static
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    acceleration_args=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path "$draft_model_dir"
      --speculative-draft-model-quantization unquant
      --speculative-draft-attention-backend flashinfer
      --speculative-dspark-block-size 7
      --speculative-num-steps 1
      --speculative-eagle-topk 1
      --max-mamba-cache-size 5
      --max-total-tokens 8192
      --disable-prefill-cuda-graph
    )
    ;;
  *)
    echo 'SGLANG_PROFILE must be dspark, mtp, mtp_previous or baseline' >&2
    exit 2
    ;;
esac

test -x "$TRITON_PTXAS_PATH"
test -f "$SGLANG_MODEL_DIR/config.json"
exec "$CONDA_PREFIX/bin/python" -m sglang.launch_server \
  --model-path "$SGLANG_MODEL_DIR" \
  --served-model-name qwen3.8-27b-fp8 \
  --host 127.0.0.1 --port "${SGLANG_PORT:-30000}" \
  --tp-size 2 \
  --dtype bfloat16 \
  --context-length 8192 \
  --max-running-requests 1 \
  --mem-fraction-static "$memory_fraction" \
  --chunked-prefill-size "$prefill_chunk" \
  --attention-backend triton \
  --linear-attn-backend triton \
  --fp8-gemm-backend triton \
  --mm-attention-backend sdpa \
  --sampling-backend pytorch \
  --mamba-ssm-dtype float32 \
  --grammar-backend xgrammar \
  --reasoning-parser qwen3 \
  --enable-cache-report \
  "${acceleration_args[@]}" \
  "$@"
