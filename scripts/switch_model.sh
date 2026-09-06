#!/bin/bash
# Switch between Qwen3 TurboQuant, Qwythos GGUF, and KAT-Coder (EXL3 or GGUF) on port 8000.
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

Examples:
  bash scripts/switch_model.sh qwen --context 32768
  bash scripts/switch_model.sh qwythos --context 16384
  bash scripts/switch_model.sh kat --context 16384
  bash scripts/switch_model.sh kat-npu --context 16384
  SPEC_MODE=none bash scripts/switch_model.sh kat-npu
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

MODEL="$1"
shift

bash scripts/kill_gpu.sh

case "$MODEL" in
  qwen|qwen3)
    echo "Starting Qwen3 TurboQuant on :8000 ..."
    exec bash scripts/serve.sh "$@"
    ;;
  qwythos|qwy)
    echo "Starting Qwythos GGUF on :8000 ..."
    exec bash scripts/serve_qwythos.sh "$@"
    ;;
  kat|kat-coder)
    echo "Starting KAT-Coder EXL3 (TabbyAPI) on :8000 ..."
    exec bash scripts/serve_kat.sh "$@"
    ;;
  kat-npu|kat-gguf)
    echo "Starting KAT-Coder GGUF (llama.cpp, SPEC_MODE=${SPEC_MODE:-cuda}) on :8000 ..."
    exec bash scripts/serve_kat_npu.sh "$@"
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
