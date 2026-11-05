#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SOCIAL_NAV_CUDA_ARCH:?Set the target compute capability, e.g. 86}"
BUILD_ROOT="$PROJECT/outputs/runtime/build"
CPU_PREFIX="${SOCIAL_NAV_GTSAM_PREFIX:-$PROJECT/outputs/perception/dependencies/gtsam}"
HOST_COMPILER="${SOCIAL_NAV_CXX:-/usr/bin/g++}"
CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cmake -S "$PROJECT/perception/mapping_localization/native" -B "$BUILD_ROOT/localization" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$HOST_COMPILER" -DCMAKE_PREFIX_PATH="$CPU_PREFIX" \
  -DSOCIAL_NAV_BUILD_LIO_CUDA="${SOCIAL_NAV_BUILD_LIO_CUDA:-ON}" -DSOCIAL_NAV_CUDA_ARCH="$SOCIAL_NAV_CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" -DCMAKE_CUDA_HOST_COMPILER="${CUDAHOSTCXX:-$HOST_COMPILER}"
cmake --build "$BUILD_ROOT/localization" --parallel "${BUILD_JOBS:-2}"
cmake -S "$PROJECT/perception/environment/native" -B "$BUILD_ROOT/environment" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$HOST_COMPILER" -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DSOCIAL_NAV_CUDA_ARCH="$SOCIAL_NAV_CUDA_ARCH"
cmake --build "$BUILD_ROOT/environment" --parallel "${BUILD_JOBS:-2}"
cmake -S "$PROJECT/navigation/planning/native" -B "$BUILD_ROOT/planning" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$HOST_COMPILER" -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DSOCIAL_NAV_BUILD_CUDA=ON -DSOCIAL_NAV_CUDA_ARCH="$SOCIAL_NAV_CUDA_ARCH"
cmake --build "$BUILD_ROOT/planning" --parallel "${BUILD_JOBS:-2}"
cmake -S "$PROJECT/navigation/control/native" -B "$BUILD_ROOT/dwa" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$HOST_COMPILER" -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DSOCIAL_NAV_CUDA_ARCH="$SOCIAL_NAV_CUDA_ARCH"
cmake --build "$BUILD_ROOT/dwa" --parallel "${BUILD_JOBS:-2}"
