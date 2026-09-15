#!/bin/bash
# Serve Qwen3.5-4B from GGUF on llama.cpp — dense 4B, native 262k context.
#
# Tuned for RTX 5080 16 GB: full GPU layers, Q4 KV cache, single parallel slot.
# Hybrid Gated DeltaNet + attention keeps KV growth sub-linear vs pure transformers.
#
# Usage:
#   bash scripts/serve_qwen35_4b.sh
#   bash scripts/serve_qwen35_4b.sh --context 262144
#   bash scripts/switch_model.sh qwen35-4b --context 262144

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="${QWEN35_GGUF_DIR:-$HOME/models/qwen35-4b}"
TARGET_GGUF="${TARGET_GGUF:-$MODEL_DIR/Qwen3.5-4B-Q4_K_M.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-$MODEL_DIR/mmproj-F16.gguf}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-4b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-262144}"
PARALLEL="${PARALLEL:-1}"
MAX_CTX="${MAX_CTX:-262144}"

N_GPU_LAYERS="${N_GPU_LAYERS:-all}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
LOAD_MODE="${LOAD_MODE:-none}"
CACHE_RAM="${CACHE_RAM:-}"

LLAMA_PREBUILT="${LLAMA_PREBUILT:-$HOME/.local/bin/llama}"

usage() {
  cat <<EOF
Usage: bash scripts/serve_qwen35_4b.sh [options]

  --context N     context size (default: $CTX_SIZE; native max $MAX_CTX)
  --port N        listen port (default: $PORT)
  -h, --help      show this help

Defaults target full VRAM residency on a 16 GB card at 262k context:
  N_GPU_LAYERS=all, KV cache q4_0/q4_0, PARALLEL=1, --flash-attn on.

Override examples:
  CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 bash scripts/serve_qwen35_4b.sh --context 131072
  TARGET_GGUF=~/models/qwen35-4b-local/Qwen3.5-4B-Q4_K_M.gguf bash scripts/serve_qwen35_4b.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)    CTX_SIZE="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        CTX_SIZE="$1"; shift
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ "$CTX_SIZE" -gt "$MAX_CTX" ]]; then
  echo "WARN: CTX_SIZE=$CTX_SIZE exceeds native max $MAX_CTX; clamping." >&2
  CTX_SIZE="$MAX_CTX"
fi

# Long contexts need an uncapped host prompt cache for slot reuse.
if [[ "$CTX_SIZE" -ge 131072 && -z "$CACHE_RAM" ]]; then
  CACHE_RAM="-1"
fi

if [[ ! -f "$TARGET_GGUF" ]]; then
  echo "Missing GGUF: $TARGET_GGUF" >&2
  echo "Run: bash scripts/download_qwen35_4b.sh && bash scripts/link_qwen35_4b_gguf.sh" >&2
  exit 1
fi

if [[ ! -f "$MMPROJ_GGUF" ]]; then
  echo "Missing mmproj GGUF: $MMPROJ_GGUF" >&2
  echo "Vision (browser screenshots) requires the projector. Run:" >&2
  echo "  bash scripts/download_qwen35_4b.sh" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_PREBUILT" ]]; then
  echo "Missing llama binary: $LLAMA_PREBUILT" >&2
  echo "Install: curl -LsSf https://llama.app/install.sh | sh" >&2
  exit 1
fi

mkdir -p "$REPO_DIR/media"

args=(
  --model "$TARGET_GGUF"
  --mmproj "$MMPROJ_GGUF"
  --alias "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX_SIZE"
  --parallel "$PARALLEL"
  --n-gpu-layers "$N_GPU_LAYERS"
  --flash-attn on
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --load-mode "$LOAD_MODE"
  --jinja
  --reasoning off
  --spec-type none
  --image-min-tokens 1024
  --temp 0.6
  --top-p 0.95
  --top-k 20
)

if [[ -n "$CACHE_RAM" ]]; then
  args+=(--cache-ram "$CACHE_RAM")
fi

echo "Qwen3.5-4B server config:"
echo "  BINARY            = $LLAMA_PREBUILT serve"
echo "  TARGET_GGUF       = $TARGET_GGUF"
echo "  MMPROJ_GGUF       = $MMPROJ_GGUF"
echo "  SERVED_MODEL_NAME = $SERVED_MODEL_NAME"
echo "  HOST:PORT         = $HOST:$PORT"
echo "  CTX_SIZE          = $CTX_SIZE"
echo "  N_GPU_LAYERS      = $N_GPU_LAYERS"
echo "  KV cache          = $CACHE_TYPE_K / $CACHE_TYPE_V"
echo "  PARALLEL          = $PARALLEL"
echo "  LOAD_MODE         = $LOAD_MODE"
if [[ -n "$CACHE_RAM" ]]; then echo "  CACHE_RAM         = $CACHE_RAM"; fi
echo "  multimodal        = enabled (--mmproj)"
echo "  reasoning         = preserve (Qwen3.5 thinking tags)"
echo "  speculation       = disabled"
echo

exec "$LLAMA_PREBUILT" serve "${args[@]}"
