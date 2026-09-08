#!/bin/bash
# Download Gemma 4 26B A4B-it GGUF trunk + mmproj for llama.cpp multimodal serving.
set -euo pipefail

VENV="${VENV:-$HOME/turboquant-llm/.venv}"
MODEL_DIR="${GEMMA4_GGUF_DIR:-$HOME/models/gemma4}"
GGUF_REPO="${GGUF_REPO:-ggml-org/gemma-4-26B-A4B-it-GGUF}"

# ggml-org ships Q4_0 (~14.6 GB); fits 16 GB VRAM with mmproj + KV headroom.
TEXT_GGUF="${TEXT_GGUF:-gemma-4-26B-A4B-it-Q4_0.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-mmproj-gemma-4-26B-A4B-it-Q8_0.gguf}"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

mkdir -p "$MODEL_DIR"

echo "Downloading Gemma 4 26B A4B GGUF to $MODEL_DIR ..."
hf download "$GGUF_REPO" \
  "$TEXT_GGUF" \
  "$MMPROJ_GGUF" \
  --local-dir "$MODEL_DIR"

echo
echo "Download complete."
echo "  Model dir:  $MODEL_DIR"
echo "  Text GGUF:  $MODEL_DIR/$TEXT_GGUF"
echo "  mmproj:     $MODEL_DIR/$MMPROJ_GGUF"
echo
echo "Link for serve script: bash scripts/link_gemma4_gguf.sh"
echo "Start server:          bash scripts/switch_model.sh gemma4 --context 100000"
