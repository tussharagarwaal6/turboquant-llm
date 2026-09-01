#!/bin/bash
# One-time TabbyAPI + ExLlamaV3 install for EXL3 models (KAT-Coder).
set -euo pipefail

TABBY_DIR="${TABBY_DIR:-$HOME/tabbyAPI}"
PYTHON="${PYTHON:-python3.12}"
UV="${UV:-uv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

if [[ ! -d "$TABBY_DIR/.git" ]]; then
  echo "Cloning TabbyAPI into $TABBY_DIR ..."
  git clone --depth 1 https://github.com/theroyallab/tabbyAPI.git "$TABBY_DIR"
else
  echo "TabbyAPI repo already present at $TABBY_DIR"
fi

cd "$TABBY_DIR"

if [[ -d venv && ! -f venv/bin/activate ]]; then
  echo "Removing incomplete TabbyAPI venv ..."
  rm -rf venv
fi

if [[ ! -f venv/bin/activate ]]; then
  if command -v "$UV" >/dev/null 2>&1; then
    echo "Creating venv with uv ..."
    "$UV" venv venv --python 3.12
  else
    if ! "$PYTHON" -m venv venv 2>/dev/null; then
      echo "Failed to create venv. Install python3.12-venv:" >&2
      echo "  sudo apt-get install -y python3.12-venv" >&2
      echo "Or install uv: https://docs.astral.sh/uv/" >&2
      exit 1
    fi
  fi
fi

if [[ ! -f venv/bin/activate ]]; then
  echo "venv creation failed: venv/bin/activate missing" >&2
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if command -v "$UV" >/dev/null 2>&1; then
  "$UV" pip install -U pip wheel
  echo "Installing TabbyAPI with CUDA 12 wheels (ExLlamaV3) ..."
  "$UV" pip install -U ".[cu12]"
else
  pip install -U pip wheel
  echo "Installing TabbyAPI with CUDA 12 wheels (ExLlamaV3) ..."
  pip install -U ".[cu12]"
fi

echo
echo "TabbyAPI ready. Launch KAT with:"
echo "  bash /mnt/c/dev/turboquant-llm/scripts/serve_kat.sh --context 16384"
