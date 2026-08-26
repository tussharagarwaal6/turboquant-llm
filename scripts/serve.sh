#!/bin/bash
# Launch TurboQuant vLLM (OpenAI-compatible). Venv: ~/turboquant-llm/.venv
#
# Context length (examples):
#   bash scripts/serve.sh --context 32768
#   MAX_MODEL_LEN=32768 bash scripts/serve.sh
#   bash scripts/serve.sh 32768

VENV="$HOME/turboquant-llm/.venv"
APP_DIR=/mnt/c/dev/turboquant-llm

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B-AWQ}"
export KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-8.0}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export AUTO_FIT_MAX_MODEL_LEN="${AUTO_FIT_MAX_MODEL_LEN:-true}"

usage() {
  cat <<'EOF'
Usage: serve.sh [OPTIONS] [CONTEXT_LENGTH]

Start the TurboQuant vLLM server on http://0.0.0.0:8000.

Options:
  -c, --context LENGTH       Max context window (tokens), sets MAX_MODEL_LEN
      --max-model-len LENGTH Same as --context
  -g, --gpu-mem FRAC         GPU memory fraction (GPU_MEMORY_UTILIZATION)
  -k, --kv-offload GIB       CPU KV offload buffer in GiB (KV_OFFLOADING_SIZE)
      --no-auto-fit          Fail instead of reducing context when VRAM is tight
  -h, --help                 Show this help

Examples:
  bash scripts/serve.sh --context 32768
  bash scripts/serve.sh 8192
  MAX_MODEL_LEN=16384 bash scripts/serve.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--context|--max-model-len)
      export MAX_MODEL_LEN="$2"
      shift 2
      ;;
    -g|--gpu-mem)
      export GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    -k|--kv-offload)
      export KV_OFFLOADING_SIZE="$2"
      shift 2
      ;;
    --no-auto-fit)
      export AUTO_FIT_MAX_MODEL_LEN=false
      shift
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
        export MAX_MODEL_LEN="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

# Auto-tune VRAM/RAM for large contexts unless the user set these explicitly.
if [[ -z "${GPU_MEMORY_UTILIZATION+x}" ]]; then
  if [[ "$MAX_MODEL_LEN" -gt 32768 ]]; then
    export GPU_MEMORY_UTILIZATION=0.95
  elif [[ "$MAX_MODEL_LEN" -gt 16384 ]]; then
    export GPU_MEMORY_UTILIZATION=0.92
  else
    export GPU_MEMORY_UTILIZATION=0.88
  fi
fi
if [[ -z "${KV_OFFLOADING_SIZE+x}" ]]; then
  if [[ "$MAX_MODEL_LEN" -le 8192 ]]; then
    export KV_OFFLOADING_SIZE=8.0
  else
    export KV_OFFLOADING_SIZE=$(awk "BEGIN {v=$MAX_MODEL_LEN/2048; print (v>8?v:8)}")
  fi
fi

echo "TurboQuant server config:"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN (requested; may auto-fit down if VRAM tight)"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE (CPU/system RAM spillover)"
echo "  AUTO_FIT_MAX_MODEL_LEN=$AUTO_FIT_MAX_MODEL_LEN"
echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "  MODEL_ID=$MODEL_ID"

cd "$APP_DIR" || exit 1
exec "$VENV/bin/python" -m uvicorn app.server:app --host 0.0.0.0 --port 8000
