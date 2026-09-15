#!/bin/bash
# Benchmark Llama 3.3 70B decode speed at different GPU offload settings.
#
# Restarts are manual — run one config, probe, then try the next.
#
# Usage:
#   bash scripts/bench_llama33.sh              # probe running server
#   N_GPU_LAYERS=auto bash scripts/switch_model.sh llama33 --context 8192
#   bash scripts/bench_llama33.sh --label auto-8k

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${LLAMA33_MODEL:-llama33-70b}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
N_PREDICT="${N_PREDICT:-64}"
LABEL="${LABEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --n-predict) N_PREDICT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/bench_llama33.sh [--label NAME] [--prompt-tokens N] [--n-predict N]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

args=(--model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" --n-predict "$N_PREDICT")
if [[ -n "$LABEL" ]]; then
  args+=(--label "$LABEL")
fi

python3 "$REPO_DIR/scripts/probe_llama_speed.py" "${args[@]}"
