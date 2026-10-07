#!/usr/bin/env bash
# 在单机 8 张 GPU 上启动 VoxelRCNN CODa 分布式训练。

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 限制每个 DDP/DataLoader 进程创建的 CPU 线程数，避免线程过量竞争。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# 以下变量均可在启动脚本前通过同名环境变量覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CFG_FILE="${CFG_FILE:-configs/voxelrcnn_coda.yaml}"
PRECISION="${PRECISION:-fp32}"
WORKERS="${WORKERS:-2}"
EXTRA_TAG="${EXTRA_TAG:-8gpu_fp32_8frame}"
LOG_GPU_MEMORY="${LOG_GPU_MEMORY:-0}"

EXTRA_ARGS=()
if [[ "${LOG_GPU_MEMORY}" == "1" ]]; then
    EXTRA_ARGS+=(--log_gpu_memory)
elif [[ "${LOG_GPU_MEMORY}" != "0" ]]; then
    echo "[ERROR] LOG_GPU_MEMORY 只能取 0 或 1，当前值: ${LOG_GPU_MEMORY}" >&2
    exit 1
fi

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    tools/train.py \
    --cfg_file "${CFG_FILE}" \
    --precision "${PRECISION}" \
    --workers "${WORKERS}" \
    --extra_tag "${EXTRA_TAG}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
