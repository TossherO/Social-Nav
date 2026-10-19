#!/usr/bin/env bash
set -Eeuo pipefail
service_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$service_root/runtime_env.sh"

export SGLANG_SRC="$SGLANG_ROOT/build/sglang"
export CARGO_HOME="$SGLANG_ROOT/build/cargo"
export RUSTUP_HOME="$SGLANG_ROOT/build/rustup"
export RUSTUP_TOOLCHAIN=1.92.0
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC"
export PYO3_PYTHON="$CONDA_PREFIX/bin/python"
export PROTOC="$CONDA_PREFIX/bin/protoc"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$CONDA_PREFIX/lib"
export CARGO_HTTP_TIMEOUT=120

test -d "$SGLANG_SRC/.git"
test -x "$PROTOC"
test -x "$CARGO_HOME/bin/cargo"
mkdir -p "$SGLANG_ROOT/build/wheelhouse" "$SGLANG_ROOT/logs"

SGLANG_BUILD_RUST_EXTS=all \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.19 \
python -m build --wheel --no-isolation "$SGLANG_SRC/python" \
  --outdir "$SGLANG_ROOT/build/wheelhouse" \
  2>&1 | tee "$SGLANG_ROOT/logs/build.log"
