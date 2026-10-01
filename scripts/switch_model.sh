#!/bin/bash
# Switch between Qwen3 TurboQuant, Qwythos GGUF, KAT-Coder, Gemma 4, Qwen3.5-4B/9B,
# and Llama 3.3 70B CPU on :8000.
set -euo pipefail

APP_DIR=/mnt/c/dev/turboquant-llm
cd "$APP_DIR" || exit 1

usage() {
  cat <<'EOF'
Usage: switch_model.sh MODEL [SERVE_ARGS...]

MODEL:
  qwen      Start Qwen3-14B-AWQ + TurboQuant (serve.sh)
  qwythos   Start Qwythos-9B GGUF + vision (serve_qwythos.sh)
  kat       Start KAT-Coder EXL3 via TabbyAPI (serve_kat.sh)
  kat-npu   Start KAT-Coder GGUF via llama.cpp with speculative decoding
            (serve_kat_npu.sh; SPEC_MODE=cuda|none|npu, default cuda)
  gemma4    Start Gemma 4 26B A4B MoE GGUF via llama.cpp + vision (serve_gemma4.sh)
  qwen35-4b Start Qwen3.5-4B GGUF via llama.cpp, native 262k context (serve_qwen35_4b.sh)
  qwen35-9b Start Qwen3.5-9B Q6_K + vision, 100k context, reasoning on (serve_qwen35_9b.sh)
  llama33   Start Llama 3.3 70B Instruct GGUF via llama.cpp, hybrid GPU+CPU (serve_llama33.sh)

Examples:
  bash scripts/switch_model.sh qwen --context 32768
  bash scripts/switch_model.sh qwythos --context 16384
  bash scripts/switch_model.sh kat --context 16384
  bash scripts/switch_model.sh kat-npu --context 16384
  bash scripts/switch_model.sh gemma4 --context 100000
  bash scripts/switch_model.sh gemma4 --context 100000 --reasoning off
  bash scripts/switch_model.sh qwen35-4b --context 262144
  bash scripts/switch_model.sh qwen35-9b
  bash scripts/switch_model.sh qwen35-9b --context 100000 --reasoning on
  bash scripts/switch_model.sh llama33 --context 8192
  CPU_ONLY=1 bash scripts/switch_model.sh llama33 --context 16384
  SPEC_MODE=none bash scripts/switch_model.sh kat-npu
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

MODEL="$1"
shift

# Help and unknown models must not stop the active server.
launch_model() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == -h || "$arg" == --help ]]; then exec bash "$@"; fi
  done
  if ! command -v nvidia-smi >/dev/null 2>&1 && [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
  fi
  bash scripts/kill_gpu.sh
  exec bash "$@"
}

case "$MODEL" in
  qwen|qwen3)
    echo "Starting Qwen3 TurboQuant on :8000 ..."
    launch_model scripts/serve.sh "$@"
    ;;
  qwythos|qwy)
    echo "Starting Qwythos GGUF on :8000 ..."
    launch_model scripts/serve_qwythos.sh "$@"
    ;;
  kat|kat-coder)
    echo "Starting KAT-Coder EXL3 (TabbyAPI) on :8000 ..."
    launch_model scripts/serve_kat.sh "$@"
    ;;
  kat-npu|kat-gguf)
    echo "Starting KAT-Coder GGUF (llama.cpp, SPEC_MODE=${SPEC_MODE:-cuda}) on :8000 ..."
    launch_model scripts/serve_kat_npu.sh "$@"
    ;;
  gemma4|gemma)
    echo "Starting Gemma 4 26B A4B MoE (llama.cpp + vision) on :8000 ..."
    launch_model scripts/serve_gemma4.sh "$@"
    ;;
  qwen35-4b|qwen35|qwen3.5)
    echo "Starting Qwen3.5-4B GGUF (llama.cpp, 262k native) on :8000 ..."
    launch_model scripts/serve_qwen35_4b.sh "$@"
    ;;
  qwen35-9b|qwen35-9b-q6|qwen3.5-9b)
    echo "Starting Qwen3.5-9B Q6_K (llama.cpp + vision, 100k context, reasoning on) on :8000 ..."
    launch_model scripts/serve_qwen35_9b.sh "$@"
    ;;
  llama33|llama3.3|llama-70b|llama33-70b)
    echo "Starting Llama 3.3 70B Instruct (llama.cpp, hybrid GPU+CPU) on :8000 ..."
    launch_model scripts/serve_llama33.sh "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    usage >&2
    exit 1
    ;;
esac
