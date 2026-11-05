#!/usr/bin/env bash
set -eo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ENVIRONMENT_CUDA_ARCH:?Set ENVIRONMENT_CUDA_ARCH to the target compute capability, e.g. 86}"
cmake -S "$PROJECT/perception/environment/native" \
  -B "$PROJECT/outputs/perception/build/environment_cuda" \
  -DCMAKE_BUILD_TYPE=Release -DSOCIAL_NAV_CUDA_ARCH="$ENVIRONMENT_CUDA_ARCH" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PROJECT/outputs/perception/ros1_ws/devel/lib"
cmake --build "$PROJECT/outputs/perception/build/environment_cuda" -j"${BUILD_JOBS:-2}"
