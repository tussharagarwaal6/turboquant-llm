#!/bin/bash
# Download Qwythos-9B GGUF (Q4_K_M + mmproj) and prefetch tokenizer/config.
set -euo pipefail

VENV="$HOME/turboquant-llm/.venv"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwythos}"
GGUF_REPO="${GGUF_REPO:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF}"
BASE_REPO="${BASE_REPO:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}"
FALLBACK_BASE="${FALLBACK_BASE:-Qwen/Qwen3.5-9B}"

TEXT_GGUF="Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
MMPROJ_GGUF="mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf"

source "$VENV/bin/activate"

mkdir -p "$MODEL_DIR"

echo "Downloading GGUF weights to $MODEL_DIR ..."
hf download "$GGUF_REPO" \
  "$TEXT_GGUF" \
  "$MMPROJ_GGUF" \
  --local-dir "$MODEL_DIR"

echo "Prefetching tokenizer + config from $BASE_REPO ..."
if hf download "$BASE_REPO" \
  config.json tokenizer.json tokenizer_config.json generation_config.json \
  special_tokens_map.json vocab.json merges.txt chat_template.jinja 2>/dev/null; then
  echo "Tokenizer/config cached from $BASE_REPO"
else
  echo "WARN: $BASE_REPO unavailable; prefetching $FALLBACK_BASE instead." >&2
  hf download "$FALLBACK_BASE" \
    config.json tokenizer.json tokenizer_config.json generation_config.json \
    special_tokens_map.json vocab.json merges.txt chat_template.jinja
fi

echo
echo "Download complete."
echo "  Model dir:  $MODEL_DIR"
echo "  Text GGUF:  $MODEL_DIR/$TEXT_GGUF"
echo "  Vision:     $MODEL_DIR/$MMPROJ_GGUF"
echo
echo "Start server: bash scripts/serve_qwythos.sh"
