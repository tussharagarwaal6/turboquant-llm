#!/bin/bash
# Download Qwen3.5-4B GGUF trunk + mmproj for llama.cpp (text + vision).
#
# The safetensors checkpoint (Qwen/Qwen3.5-4B) is for vLLM/transformers.
# llama.cpp needs GGUF — this script fetches Q4_K_M + the vision projector.
set -euo pipefail

VENV="${VENV:-$HOME/turboquant-llm/.venv}"
MODEL_DIR="${QWEN35_GGUF_DIR:-$HOME/models/qwen35-4b}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3.5-4B-GGUF}"
TEXT_GGUF="${TEXT_GGUF:-Qwen3.5-4B-Q4_K_M.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-mmproj-F16.gguf}"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

mkdir -p "$MODEL_DIR"

echo "Downloading Qwen3.5-4B GGUF to $MODEL_DIR ..."
hf download "$GGUF_REPO" "$TEXT_GGUF" "$MMPROJ_GGUF" --local-dir "$MODEL_DIR"

echo
echo "Download complete."
echo "  Model dir: $MODEL_DIR"
echo "  Text GGUF: $MODEL_DIR/$TEXT_GGUF"
echo "  mmproj:    $MODEL_DIR/$MMPROJ_GGUF"
echo
echo "Link for serve script: bash scripts/link_qwen35_4b_gguf.sh"
echo "Start server:          bash scripts/switch_model.sh qwen35-4b --context 262144"
