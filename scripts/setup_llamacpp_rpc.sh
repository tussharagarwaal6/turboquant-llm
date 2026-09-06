#!/bin/bash
# Build llama.cpp from source in WSL2 with the CUDA and RPC backends.
#
# WHY THIS EXISTS
#   The prebuilt binary at ~/.local/bin/llama (from llama.app/install.sh) already
#   has CUDA and the full --spec-* speculative-decoding flags, so the default
#   kat-npu mode (bundled MTP head on CUDA) needs NO build at all.
#   It is compiled WITHOUT the RPC backend, though, so attaching a remote
#   drafting device (--rpc / --spec-draft-device RPC0) requires this build.
#
#   Only run this if you want the experimental NPU drafting path. Measurements on
#   this machine show the NPU drafts far slower than the CUDA target decodes, so
#   the RPC path is not expected to be a win. See README "KAT NPU drafter".
#
# Installs into ~/llamacpp-cuda and leaves ~/.local/bin/llama untouched so the
# Qwythos profile keeps working.
#
# Usage:
#   bash scripts/setup_llamacpp_rpc.sh
#   JOBS=8 bash scripts/setup_llamacpp_rpc.sh

set -euo pipefail

SRC_DIR="${SRC_DIR:-$HOME/llamacpp-cuda}"
JOBS="${JOBS:-$(nproc)}"
# RTX 5080 is Blackwell sm_120.
CUDA_ARCH="${CUDA_ARCH:-120}"

echo "llama.cpp CUDA+RPC source build"
echo "  SRC_DIR=$SRC_DIR"
echo "  JOBS=$JOBS"
echo "  CUDA_ARCH=$CUDA_ARCH"
echo

# apt installs nvcc under /usr/local/cuda/bin, but that directory is only on
# PATH after sourcing ~/.bashrc. Non-interactive shells (e.g. wsl bash -lc from
# PowerShell) skip .bashrc, so detect the standard install location here.
if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_bin in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [[ -x "$cuda_bin/nvcc" ]]; then
      export PATH="$cuda_bin:$PATH"
      echo "  Added $cuda_bin to PATH (nvcc was not on PATH)"
      break
    fi
  done
fi

missing=()
command -v cmake >/dev/null 2>&1 || missing+=("cmake")
command -v git   >/dev/null 2>&1 || missing+=("git")
command -v nvcc  >/dev/null 2>&1 || missing+=("nvcc (cuda-toolkit)")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing build prerequisites: ${missing[*]}" >&2
  cat >&2 <<'EOF'

Install them with:

  sudo apt-get update
  sudo apt-get install -y build-essential cmake git

For nvcc, install the CUDA toolkit for WSL2 (~3 GB; do NOT install a Linux
NVIDIA driver inside WSL, only the toolkit):

  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

EOF
  exit 1
fi

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "Cloning llama.cpp into $SRC_DIR ..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC_DIR"
fi

cd "$SRC_DIR"

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF

cmake --build build --config Release --target llama-server llama-bench -j "$JOBS"

BIN="$SRC_DIR/build/bin/llama-server"
if [[ ! -x "$BIN" ]]; then
  echo "Build finished but $BIN is missing" >&2
  exit 1
fi

echo
echo "Built: $BIN"
echo "RPC support:"
if "$BIN" --help 2>&1 | grep -q -- '--rpc'; then
  echo "  --rpc present"
else
  echo "  WARNING: --rpc missing; GGML_RPC did not take effect" >&2
fi
echo
echo "Use it with:  LLAMA_BIN=$BIN SPEC_MODE=npu bash scripts/serve_kat_npu.sh"
