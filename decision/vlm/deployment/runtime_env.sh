#!/usr/bin/env bash
# Source from the service root; keep the model service separate from the client.
export SGLANG_ROOT="${SGLANG_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
sglang_conda_root="${SGLANG_CONDA_ROOT:-$HOME/miniconda3}"
sglang_conda_env="${SGLANG_CONDA_ENV:-$sglang_conda_root/envs/sglang-cu129}"
set +u
source "$sglang_conda_root/etc/profile.d/conda.sh"
conda activate "$sglang_conda_env"
set -u
export SGLANG_MODEL_DIR="${SGLANG_MODEL_DIR:-$SGLANG_ROOT/models/Qwen3.8-27B-FP8}"
export CUDA_VISIBLE_DEVICES="${SGLANG_GPUS:-2,3}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME CUDA_PATH
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_CCBIN="$CXX"
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib"
export TRITON_CACHE_DIR="$SGLANG_ROOT/cache/triton"
export HF_HOME="$SGLANG_ROOT/cache/huggingface"
export MAX_JOBS=4
