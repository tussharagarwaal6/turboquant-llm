#!/bin/bash
# Qwen3.5-9B Q6_K: verified RTX 5080 Laptop 16 GB profile, with vision.
set -euo pipefail

MODEL_DIR="${QWEN35_9B_GGUF_DIR:-$HOME/models/qwen35-9b}"
TARGET_GGUF="${TARGET_GGUF:-$MODEL_DIR/Qwen_Qwen3.5-9B-Q6_K.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-$MODEL_DIR/mmproj-Qwen_Qwen3.5-9B-f16.gguf}"
LLAMA_PREBUILT="${LLAMA_PREBUILT:-$HOME/.local/bin/llama}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-100000}"
MAX_CTX=262144
PARALLEL="${PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-all}"
TENSOR_OVERRIDE="${TENSOR_OVERRIDE:-token_embd.weight=CUDA0}"
FIT="${FIT:-off}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
LOAD_MODE="${LOAD_MODE:-none}"
CACHE_RAM="${CACHE_RAM:-}"
REASONING="${REASONING:-on}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
LOG_VERBOSITY="${LOG_VERBOSITY:-4}"

usage() {
  cat <<EOF
Usage: bash scripts/serve_qwen35_9b.sh [options]

  --context N          context size (default: $CTX_SIZE; native max $MAX_CTX)
  --port N             listen port (default: $PORT)
  --reasoning M        on|off|auto (default: $REASONING)
  --reasoning-format F response thinking format (default: $REASONING_FORMAT)
  -h, --help           show this help

Default verified 16 GB GPU profile:
  Q6_K weights, 100000 context, reasoning on, vision enabled,
  all GPU layers including embeddings, Q8/Q8 KV cache, one parallel slot,
  flash attention on, batch 512/256, fit off, speculation disabled.
  Context sizes above 100000 need a new VRAM fit check.

Examples:
  bash scripts/switch_model.sh qwen35-9b
  bash scripts/switch_model.sh qwen35-9b --context 100000 --reasoning on
  bash scripts/serve_qwen35_9b.sh --port 8001 --reasoning off
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context|--port|--reasoning|--reasoning-format)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 1; }
      case "$1" in
        --context) CTX_SIZE="$2" ;;
        --port) PORT="$2" ;;
        --reasoning) REASONING="$2" ;;
        --reasoning-format) REASONING_FORMAT="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then CTX_SIZE="$1"; shift;
      else echo "Unknown argument: $1" >&2; usage >&2; exit 1; fi ;;
  esac
done

[[ "$CTX_SIZE" =~ ^[0-9]+$ && "$CTX_SIZE" -gt 0 && "$CTX_SIZE" -le "$MAX_CTX" ]] || {
  echo "Context must be between 1 and $MAX_CTX tokens." >&2; exit 1;
}
[[ "$PORT" =~ ^[0-9]+$ && "$PORT" -gt 0 && "$PORT" -le 65535 ]] || {
  echo "Port must be between 1 and 65535." >&2; exit 1;
}
case "$REASONING" in on|off|auto) ;; *) echo "Reasoning must be on, off or auto." >&2; exit 1 ;; esac
if [[ "$CTX_SIZE" -ge 131072 && -z "$CACHE_RAM" ]]; then CACHE_RAM=-1; fi
[[ -f "$TARGET_GGUF" ]] || { echo "Missing Q6_K model: $TARGET_GGUF" >&2; exit 1; }
[[ -f "$MMPROJ_GGUF" ]] || { echo "Missing matching vision projector: $MMPROJ_GGUF" >&2; exit 1; }
[[ -x "$LLAMA_PREBUILT" ]] || { echo "Missing llama binary: $LLAMA_PREBUILT" >&2; exit 1; }

args=(
  --model "$TARGET_GGUF" --mmproj "$MMPROJ_GGUF" --alias "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" --parallel "$PARALLEL"
  --n-gpu-layers "$N_GPU_LAYERS" --override-tensor "$TENSOR_OVERRIDE" --fit "$FIT"
  --flash-attn on --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V"
  --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" --load-mode "$LOAD_MODE"
  --jinja --reasoning "$REASONING" --reasoning-format "$REASONING_FORMAT"
  --spec-type none --image-min-tokens 1024 --temp 0.6 --top-p 0.95 --top-k 20
  --log-verbosity "$LOG_VERBOSITY"
)
if [[ -n "$CACHE_RAM" ]]; then args+=(--cache-ram "$CACHE_RAM"); fi

echo "Qwen3.5-9B Q6_K server config:"
echo "  BINARY            = $LLAMA_PREBUILT serve"
echo "  TARGET_GGUF       = $TARGET_GGUF"
echo "  MMPROJ_GGUF       = $MMPROJ_GGUF"
echo "  SERVED_MODEL_NAME = $SERVED_MODEL_NAME"
echo "  HOST:PORT         = $HOST:$PORT"
echo "  CTX_SIZE          = $CTX_SIZE"
echo "  N_GPU_LAYERS      = $N_GPU_LAYERS"
echo "  TENSOR_OVERRIDE   = $TENSOR_OVERRIDE"
echo "  FIT               = $FIT"
echo "  KV cache          = $CACHE_TYPE_K / $CACHE_TYPE_V"
echo "  PARALLEL          = $PARALLEL"
echo "  BATCH/UBATCH      = $BATCH_SIZE / $UBATCH_SIZE"
echo "  LOAD_MODE         = $LOAD_MODE"
if [[ -n "$CACHE_RAM" ]]; then echo "  CACHE_RAM         = $CACHE_RAM"; fi
echo "  multimodal        = enabled (--mmproj)"
echo "  reasoning         = $REASONING ($REASONING_FORMAT format)"
echo "  speculation       = disabled"
echo
exec "$LLAMA_PREBUILT" serve "${args[@]}"
