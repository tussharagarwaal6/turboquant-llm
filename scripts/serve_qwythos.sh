#!/bin/bash
# Launch Qwythos-9B GGUF on port 8000.
#
# Default runtime is llama.cpp (official, supports vision). vLLM GGUF plugin
# currently fails with "Unknown gguf model_type: qwen3_5" — set
# QWYTHOS_RUNTIME=vllm to attempt vLLM when the plugin adds Qwen3.5 support.
#
# Stop other GPU servers first:
#   bash scripts/kill_gpu.sh
#
# Examples:
#   bash scripts/serve_qwythos.sh --context 16384
#   QWYTHOS_RUNTIME=vllm bash scripts/serve_qwythos.sh

set -euo pipefail

APP_DIR=/mnt/c/dev/turboquant-llm
RUNTIME="${QWYTHOS_RUNTIME:-llama}"

if [[ "$RUNTIME" == "llama" ]]; then
  exec bash "$APP_DIR/scripts/serve_qwythos_llama.sh" "$@"
fi

if [[ "$RUNTIME" != "vllm" ]]; then
  echo "Unknown QWYTHOS_RUNTIME=$RUNTIME (use llama or vllm)" >&2
  exit 1
fi

VENV="$HOME/turboquant-llm/.venv"
APP_DIR=/mnt/c/dev/turboquant-llm

export MODEL_DIR="${MODEL_DIR:-$HOME/models/qwythos}"
export TEXT_GGUF="${TEXT_GGUF:-Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwythos-9b}"
export PORT="${PORT:-8000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export TOKENIZER="${TOKENIZER:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}"
export HF_CONFIG_PATH="${HF_CONFIG_PATH:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}"
export GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$APP_DIR/media}"

export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":200704,\"max_pixels\":1003520}}"
export LIMIT_MM="${LIMIT_MM:-{\"image\":2,\"video\":{\"count\":0}}}"

export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-true}"

# WSL2 workarounds (same as serve_vl.sh).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"

usage() {
  cat <<'EOF'
Usage: serve_qwythos.sh [OPTIONS] [CONTEXT_LENGTH]

Start Qwythos-9B GGUF on http://0.0.0.0:8000 (vLLM + vllm-gguf-plugin).

Options:
  -c, --context LENGTH       Max context window (tokens), sets MAX_MODEL_LEN
      --max-model-len LENGTH Same as --context
  -g, --gpu-mem FRAC         GPU memory fraction (GPU_MEMORY_UTILIZATION)
  -p, --port PORT            Listen port (default 8000)
  -h, --help                 Show this help

Environment overrides:
  MODEL_DIR, TEXT_GGUF, SERVED_MODEL_NAME, TOKENIZER, HF_CONFIG_PATH
  MM_PROCESSOR_KWARGS, LIMIT_MM, ALLOWED_LOCAL_MEDIA_PATH
  CLEAR_VLLM_COMPILE_CACHE=1  Remove stale torch compile cache before start

Prerequisites:
  pip install -r requirements-gguf.txt
  bash scripts/download_qwythos.sh

Examples:
  bash scripts/serve_qwythos.sh --context 16384
  CLEAR_VLLM_COMPILE_CACHE=1 bash scripts/serve_qwythos.sh
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

MODEL_PATH="$MODEL_DIR/$TEXT_GGUF"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing GGUF: $MODEL_PATH" >&2
  echo "Run: bash scripts/download_qwythos.sh" >&2
  exit 1
fi

if [[ -z "${GPU_MEMORY_UTILIZATION+x}" ]]; then
  export GPU_MEMORY_UTILIZATION=0.92
fi

if [[ "${CLEAR_VLLM_COMPILE_CACHE:-0}" == "1" ]]; then
  echo "Clearing stale vLLM torch compile cache..."
  rm -rf "$HOME/.cache/vllm/torch_compile_cache"
fi

EXTRA_ARGS=(--generation-config "$GENERATION_CONFIG")
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${ENABLE_TOOL_CALLING}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "${REASONING_PARSER:-}" ]]; then
  EXTRA_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

echo "Qwythos GGUF server config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  PORT=$PORT"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "  TOKENIZER=$TOKENIZER"
echo "  HF_CONFIG_PATH=$HF_CONFIG_PATH"
echo "  REASONING_PARSER=$REASONING_PARSER"
echo "  TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
echo "  ALLOWED_LOCAL_MEDIA_PATH=$ALLOWED_LOCAL_MEDIA_PATH"
echo "  MM_PROCESSOR_KWARGS=$MM_PROCESSOR_KWARGS"
echo "  LIMIT_MM=$LIMIT_MM"
echo "  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "  VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER"
if ((${#EXTRA_ARGS[@]})); then
  echo "  EXTRA_ARGS=${EXTRA_ARGS[*]}"
fi

mkdir -p "$ALLOWED_LOCAL_MEDIA_PATH"
cd "$APP_DIR" || exit 1

exec "$VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tokenizer "$TOKENIZER" \
  --hf-config-path "$HF_CONFIG_PATH" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt "$LIMIT_MM" \
  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  "${EXTRA_ARGS[@]}"
