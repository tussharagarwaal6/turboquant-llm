#!/bin/bash
# Launch Qwythos-9B GGUF via llama.cpp OpenAI-compatible server on port 8000.
# Official path for this model (text + vision via mmproj). Use when vLLM GGUF
# plugin cannot load qwen3_5 yet.
#
# Stop other GPU servers first:
#   bash scripts/kill_gpu.sh
#
# Examples:
#   bash scripts/serve_qwythos_llama.sh --context 16384
#   LLAMA_BIN=$HOME/.local/bin/llama bash scripts/serve_qwythos_llama.sh

set -euo pipefail

APP_DIR=/mnt/c/dev/turboquant-llm
VENV="${VENV:-$HOME/turboquant-llm/.venv}"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/qwythos}"
export TEXT_GGUF="${TEXT_GGUF:-Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf}"
export MMPROJ_GGUF="${MMPROJ_GGUF:-mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwythos-9b}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"
export CTX_SIZE="${CTX_SIZE:-16384}"
export LLAMA_BIN="${LLAMA_BIN:-$HOME/.local/bin/llama}"
export N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
export PARALLEL="${PARALLEL:-}"
export CACHE_RAM="${CACHE_RAM:-}"
export ENABLE_CONTEXT_COMPACTION="${ENABLE_CONTEXT_COMPACTION:-1}"
export BACKEND_PORT="${BACKEND_PORT:-8002}"

# Qwythos GGUF ships YaRN for up to 1,048,576 tokens (see model card).
export MAX_CTX="${MAX_CTX:-1048576}"

usage() {
  cat <<'EOF'
Usage: serve_qwythos_llama.sh [OPTIONS] [CONTEXT_LENGTH]

Start Qwythos via llama.cpp on http://0.0.0.0:8000.

Options:
  -c, --context LENGTH       Context size (default 16384; use 1048576 for 1M YaRN)
  -p, --port PORT            Listen port (default 8000)
  -h, --help                 Show this help

Environment:
  PARALLEL=1                 Recommended for 1M (single slot)
  CACHE_RAM=-1               No RAM cap on prompt/KV cache (needed for long ctx)
  ENABLE_CONTEXT_COMPACTION=1  Summarize old turns when context nears limit (default on)
  BACKEND_PORT=8002          llama.cpp port when compaction proxy listens on PORT

Prerequisites:
  llama.cpp CLI installed (https://llama.app/install.sh)
  bash scripts/download_qwythos.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--context|--ctx-size)
      export CTX_SIZE="$2"
      shift 2
      ;;
    -p|--port)
      export PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        export CTX_SIZE="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

MODEL_PATH="$MODEL_DIR/$TEXT_GGUF"
MMPROJ_PATH="$MODEL_DIR/$MMPROJ_GGUF"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "Missing llama binary: $LLAMA_BIN" >&2
  echo "Install: curl -LsSf https://llama.app/install.sh | sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing GGUF: $MODEL_PATH" >&2
  echo "Run: bash scripts/download_qwythos.sh" >&2
  exit 1
fi
if [[ ! -f "$MMPROJ_PATH" ]]; then
  echo "Missing mmproj: $MMPROJ_PATH" >&2
  echo "Run: bash scripts/download_qwythos.sh" >&2
  exit 1
fi

mkdir -p "$APP_DIR/media"

if [[ "$CTX_SIZE" -gt "$MAX_CTX" ]]; then
  echo "WARN: CTX_SIZE=$CTX_SIZE exceeds model max $MAX_CTX; clamping." >&2
  CTX_SIZE="$MAX_CTX"
fi

# Long-context defaults: one slot so the full ctx-size applies per request.
if [[ "$CTX_SIZE" -ge 8192 ]]; then
  PARALLEL="${PARALLEL:-1}"
fi
if [[ "$CTX_SIZE" -ge 262144 ]]; then
  CACHE_RAM="${CACHE_RAM:--1}"
fi
PARALLEL="${PARALLEL:-4}"
CACHE_RAM="${CACHE_RAM:-8192}"

echo "Qwythos llama.cpp server config:"
echo "  LLAMA_BIN=$LLAMA_BIN"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  MMPROJ_PATH=$MMPROJ_PATH"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  HOST=$HOST"
echo "  PORT=$PORT"
echo "  CTX_SIZE=$CTX_SIZE"
echo "  N_GPU_LAYERS=$N_GPU_LAYERS"
echo "  PARALLEL=$PARALLEL"
echo "  CACHE_RAM=$CACHE_RAM"
if [[ "$CTX_SIZE" -ge 262144 ]]; then
  echo "  NOTE: 1M-class context uses CPU RAM for KV offload; ensure enough free RAM."
fi

_llama_serve_args=(
  --model "$MODEL_PATH"
  --mmproj "$MMPROJ_PATH"
  --alias "$SERVED_MODEL_NAME"
  --host "$HOST"
  --ctx-size "$CTX_SIZE"
  --parallel "$PARALLEL"
  --cache-ram "$CACHE_RAM"
  --n-gpu-layers "$N_GPU_LAYERS"
  --temp 0.6
  --top-p 0.95
  --top-k 20
  --repeat-penalty 1.05
  --reasoning-preserve
)

_wait_for_backend() {
  local url="http://127.0.0.1:${BACKEND_PORT}/v1/models"
  local attempt
  for attempt in $(seq 1 120); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for llama.cpp on port $BACKEND_PORT" >&2
  return 1
}

_compaction_enabled() {
  case "${ENABLE_CONTEXT_COMPACTION,,}" in
    0|false|no|off) return 1 ;;
    *) return 0 ;;
  esac
}

if _compaction_enabled; then
  echo "  ENABLE_CONTEXT_COMPACTION=1 (proxy :$PORT -> llama :$BACKEND_PORT)"
  export COMPACTION_BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}/v1"
  export MAX_MODEL_LEN="$CTX_SIZE"

  "$LLAMA_BIN" serve \
    "${_llama_serve_args[@]}" \
    --port "$BACKEND_PORT" &
  LLAMA_PID=$!
  trap 'kill "$LLAMA_PID" 2>/dev/null || true' EXIT

  if ! _wait_for_backend; then
    kill "$LLAMA_PID" 2>/dev/null || true
    exit 1
  fi

  cd "$APP_DIR" || exit 1
  exec "$VENV/bin/python" -m uvicorn app.compaction_proxy:app --host "$HOST" --port "$PORT"
fi

exec "$LLAMA_BIN" serve \
  "${_llama_serve_args[@]}" \
  --port "$PORT"
