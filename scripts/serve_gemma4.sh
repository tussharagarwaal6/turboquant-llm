#!/bin/bash
# Serve Gemma 4 26B A4B MoE from GGUF on llama.cpp with vision (mmproj).
#
# Same long-context MoE profile as scripts/serve_kat_npu.sh, plus --mmproj for
# text + image input. No speculative decoding (no bundled MTP in default build).
#
# Usage:
#   bash scripts/serve_gemma4.sh
#   bash scripts/serve_gemma4.sh --context 100000
#   bash scripts/serve_gemma4.sh --context 100000 --reasoning off
#   bash scripts/switch_model.sh gemma4 --context 100000 --reasoning off

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="${GEMMA4_GGUF_DIR:-$HOME/models/gemma4}"
TARGET_GGUF="${TARGET_GGUF:-$MODEL_DIR/gemma-4-26B-A4B-it-Q4_0.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-$MODEL_DIR/mmproj-gemma-4-26B-A4B-it-Q8_0.gguf}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4-26b-a4b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-16384}"
PARALLEL="${PARALLEL:-1}"

_n_cpu_moe_set="${N_CPU_MOE:+yes}"
_n_gpu_layers_set="${N_GPU_LAYERS:+yes}"

N_CPU_MOE="${N_CPU_MOE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"

CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
LOAD_MODE="${LOAD_MODE:-none}"
CACHE_RAM="${CACHE_RAM:-}"
REASONING="${REASONING:-off}"

LLAMA_PREBUILT="${LLAMA_PREBUILT:-$HOME/.local/bin/llama}"

usage() {
  cat <<EOF
Usage: bash scripts/serve_gemma4.sh [options]

  --context N     context size (default: $CTX_SIZE)
  --port N        listen port (default: $PORT)
  --n-cpu-moe N   MoE experts on CPU (only with numeric N_GPU_LAYERS)
  --reasoning M   llama.cpp reasoning mode (default: $REASONING)
                  Use 'off' so action JSON is not swallowed by thinking.
                  Override with REASONING=auto (or another llama.cpp mode)
                  to restore native Gemma thinking.
  -h, --help      show this help

At --context 65536 and above, N_GPU_LAYERS defaults to 'all' and N_CPU_MOE to a
measured value (12 up to 100k, 16 beyond), matching serve_kat_npu.sh.

For maximum speed at the cost of some KV precision:
  CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 N_CPU_MOE=10 bash scripts/serve_gemma4.sh --context 100000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)    CTX_SIZE="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --n-cpu-moe)  N_CPU_MOE="$2"; _n_cpu_moe_set="yes"; shift 2 ;;
    --reasoning)  REASONING="$2"; shift 2 ;;
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

if [[ "$CTX_SIZE" -ge 131072 && -z "$CACHE_RAM" ]]; then
  CACHE_RAM="-1"
fi

if [[ "$CTX_SIZE" -ge 65536 ]]; then
  if [[ -z "$_n_gpu_layers_set" ]]; then
    N_GPU_LAYERS="all"
  fi
  if [[ -z "$_n_cpu_moe_set" && "$N_GPU_LAYERS" != "auto" ]]; then
    if [[ "$CTX_SIZE" -le 100352 ]]; then
      N_CPU_MOE=12
    else
      N_CPU_MOE=16
    fi
  fi
fi

if [[ ! -f "$TARGET_GGUF" ]]; then
  echo "Missing trunk GGUF: $TARGET_GGUF" >&2
  echo "Run: bash scripts/download_gemma4.sh && bash scripts/link_gemma4_gguf.sh" >&2
  exit 1
fi

if [[ ! -f "$MMPROJ_GGUF" ]]; then
  echo "Missing mmproj GGUF: $MMPROJ_GGUF" >&2
  echo "Run: bash scripts/download_gemma4.sh && bash scripts/link_gemma4_gguf.sh" >&2
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
  --reasoning "$REASONING"
  --spec-type none
)

if [[ -n "$N_CPU_MOE" ]]; then
  if [[ "$N_GPU_LAYERS" == "auto" ]]; then
    echo "WARN: omitting --n-cpu-moe $N_CPU_MOE (conflicts with N_GPU_LAYERS=auto)." >&2
    echo "      Set N_GPU_LAYERS=all to pin MoE offload manually." >&2
  else
    args+=(--n-cpu-moe "$N_CPU_MOE")
  fi
fi

if [[ -n "$CACHE_RAM" ]]; then
  args+=(--cache-ram "$CACHE_RAM")
fi

echo "Gemma 4 26B A4B server config:"
echo "  BINARY            = $LLAMA_PREBUILT serve"
echo "  TARGET_GGUF       = $TARGET_GGUF"
echo "  MMPROJ_GGUF       = $MMPROJ_GGUF"
echo "  SERVED_MODEL_NAME = $SERVED_MODEL_NAME"
echo "  HOST:PORT         = $HOST:$PORT"
echo "  CTX_SIZE          = $CTX_SIZE"
if [[ -n "$N_CPU_MOE" && "$N_GPU_LAYERS" != "auto" ]]; then
  echo "  N_CPU_MOE         = $N_CPU_MOE"
else
  echo "  N_CPU_MOE         = auto-fit"
fi
echo "  N_GPU_LAYERS      = $N_GPU_LAYERS"
if [[ "$CTX_SIZE" -ge 65536 && -z "$_n_cpu_moe_set" && -z "$_n_gpu_layers_set" ]]; then
  echo "                      (long-context profile; override with N_CPU_MOE / N_GPU_LAYERS)"
fi
echo "  KV cache          = $CACHE_TYPE_K / $CACHE_TYPE_V"
echo "  LOAD_MODE         = $LOAD_MODE"
if [[ -n "$CACHE_RAM" ]]; then echo "  CACHE_RAM         = $CACHE_RAM"; fi
echo "  multimodal        = enabled (--mmproj)"
echo "  reasoning         = $REASONING"
echo "  speculation       = disabled"
echo

exec "$LLAMA_PREBUILT" serve "${args[@]}"
