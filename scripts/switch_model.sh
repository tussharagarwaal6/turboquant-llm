#!/bin/bash
# Switch between Qwen3 TurboQuant and Qwythos GGUF on port 8000.
set -euo pipefail

APP_DIR=/mnt/c/dev/turboquant-llm
cd "$APP_DIR" || exit 1

usage() {
  cat <<'EOF'
Usage: switch_model.sh MODEL [SERVE_ARGS...]

MODEL:
  qwen      Start Qwen3-14B-AWQ + TurboQuant (serve.sh)
  qwythos   Start Qwythos-9B GGUF + vision (serve_qwythos.sh)

Examples:
  bash scripts/switch_model.sh qwen --context 32768
  bash scripts/switch_model.sh qwythos --context 16384
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
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    usage >&2
    exit 1
    ;;
esac
