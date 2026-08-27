#!/bin/bash
# Launch Qwen2.5-VL multimodal server (image + video + text) via vLLM OpenAI API.
# No TurboQuant — uses awq_marlin on sm_120 (Blackwell). Port 8001 by default.
#
# Stop the text-only TurboQuant server on :8000 first (VRAM is shared):
#   bash scripts/kill_gpu.sh
#
# Examples:
#   bash scripts/serve_vl.sh --context 16384
#   PERF_PROFILE=speed bash scripts/serve_vl.sh
#   CLEAR_VLLM_COMPILE_CACHE=1 bash scripts/serve_vl.sh

VENV="$HOME/turboquant-llm/.venv"
APP_DIR=/mnt/c/dev/turboquant-llm

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-vl}"
export PORT="${PORT:-8001}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$APP_DIR/media}"
export PERF_PROFILE="${PERF_PROFILE:-balanced}"
export GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

# Cap visual tokens per image (256*28*28 .. 1280*28*28). Dominant VRAM knob for VL models.
export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":200704,\"max_pixels\":1003520}}"

# Model max is 14 frames; 32 was ignored and wasted encoder profiling budget.
export LIMIT_MM="${LIMIT_MM:-{\"image\":2,\"video\":{\"count\":1,\"num_frames\":14,\"width\":640,\"height\":640}}}"

# fps belongs here, not in mm-processor-kwargs; Qwen2.5-VL samples video by fps.
export MEDIA_IO_KWARGS="${MEDIA_IO_KWARGS:-{\"video\":{\"fps\":1,\"backend\":\"opencv\"}}}"

# WSL2: FlashInfer JIT sampler cannot build against pip CUDA fragments.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# WSL2 GPU passthrough does not expose UVA, which the V2 model runner requires.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"

# Cursor and other clients send tool_choice="auto"; vLLM needs these flags enabled.
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-true}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

_apply_perf_profile() {
  case "$PERF_PROFILE" in
    speed)
      export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":200704,\"max_pixels\":802816}}"
      export LIMIT_MM="${LIMIT_MM:-{\"image\":2,\"video\":{\"count\":0}}}"
      ;;
    balanced)
      ;;
    *)
      echo "Unknown PERF_PROFILE=$PERF_PROFILE (use balanced or speed)" >&2
      exit 1
      ;;
  esac
}

_setup_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]]; then
    return
  fi
  local nvcc_path
  nvcc_path="$("$VENV/bin/python" - <<'PY' 2>/dev/null || true
import os
from pathlib import Path
try:
    import nvidia
except ImportError:
    raise SystemExit
for root in nvidia.__path__:
    for candidate in sorted(Path(root).glob("cu*")):
        if (candidate / "bin" / "nvcc").is_file():
            print(candidate)
            raise SystemExit
PY
)"
  if [[ -n "$nvcc_path" ]]; then
    export CUDA_HOME="$nvcc_path"
    export PATH="$CUDA_HOME/bin:$VENV/bin:${PATH}"
  fi
}

usage() {
  cat <<'EOF'
Usage: serve_vl.sh [OPTIONS] [CONTEXT_LENGTH]

Start the Qwen2.5-VL multimodal server on http://0.0.0.0:8001.

Options:
  -c, --context LENGTH       Max context window (tokens), sets MAX_MODEL_LEN
      --max-model-len LENGTH Same as --context
  -g, --gpu-mem FRAC         GPU memory fraction (GPU_MEMORY_UTILIZATION)
  -p, --port PORT            Listen port (default 8001)
  -h, --help                 Show this help

Environment overrides:
  MODEL_ID, SERVED_MODEL_NAME, MAX_NUM_SEQS, ALLOWED_LOCAL_MEDIA_PATH
  MM_PROCESSOR_KWARGS, LIMIT_MM, MEDIA_IO_KWARGS, PERF_PROFILE (balanced|speed)
  CLEAR_VLLM_COMPILE_CACHE=1  Remove stale torch compile cache before start
  GENERATION_CONFIG=vllm       Use vLLM sampling defaults (not model's greedy top_p)

If startup OOMs: lower MM_PROCESSOR_KWARGS max_pixels, then add ENFORCE_EAGER=1.

Examples:
  bash scripts/serve_vl.sh --context 16384
  PERF_PROFILE=speed bash scripts/serve_vl.sh
  CLEAR_VLLM_COMPILE_CACHE=1 bash scripts/serve_vl.sh --gpu-mem 0.95
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

_apply_perf_profile

if [[ -z "${GPU_MEMORY_UTILIZATION+x}" ]]; then
  # 0.95 fails on WSL2 RTX 5080 (only ~14.7 GiB free at startup); 0.92 is the safe max.
  export GPU_MEMORY_UTILIZATION=0.92
fi

_setup_cuda_home

if [[ "${CLEAR_VLLM_COMPILE_CACHE:-0}" == "1" ]]; then
  echo "Clearing stale vLLM torch compile cache..."
  rm -rf "$HOME/.cache/vllm/torch_compile_cache"
fi

EXTRA_ARGS=(--generation-config "$GENERATION_CONFIG")
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "${ENABLE_TOOL_CALLING}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

echo "Multimodal VL server config:"
echo "  MODEL_ID=$MODEL_ID"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  PORT=$PORT"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "  PERF_PROFILE=$PERF_PROFILE"
echo "  GENERATION_CONFIG=$GENERATION_CONFIG"
echo "  ALLOWED_LOCAL_MEDIA_PATH=$ALLOWED_LOCAL_MEDIA_PATH"
echo "  MM_PROCESSOR_KWARGS=$MM_PROCESSOR_KWARGS"
echo "  LIMIT_MM=$LIMIT_MM"
echo "  MEDIA_IO_KWARGS=$MEDIA_IO_KWARGS"
echo "  CUDA_HOME=${CUDA_HOME:-unset}"
echo "  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "  VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER"
echo "  ENABLE_TOOL_CALLING=$ENABLE_TOOL_CALLING"
echo "  TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
if ((${#EXTRA_ARGS[@]})); then
  echo "  EXTRA_ARGS=${EXTRA_ARGS[*]}"
fi

mkdir -p "$ALLOWED_LOCAL_MEDIA_PATH"
cd "$APP_DIR" || exit 1

exec "$VENV/bin/vllm" serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --quantization awq_marlin \
  --limit-mm-per-prompt "$LIMIT_MM" \
  --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
  --media-io-kwargs "$MEDIA_IO_KWARGS" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  "${EXTRA_ARGS[@]}"
