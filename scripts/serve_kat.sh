#!/bin/bash
# Launch KAT-Coder-V2.5-Dev EXL3 via TabbyAPI + ExLlamaV3 on port 8000.
#
# NOTE: EXL3 checkpoints require ExLlamaV3 (TabbyAPI). They cannot load on the
# vLLM TurboQuant path (turboquant_k8v4). TabbyAPI uses Q8 KV cache instead.
#
# Model: P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq (~24 GB, 5.08 bpw + MTP)
#
# One-time setup:
#   bash scripts/setup_tabbyapi.sh
#
# Examples:
#   bash scripts/serve_kat.sh --context 16384
#   bash scripts/switch_model.sh kat --context 8192

set -euo pipefail

APP_DIR=/mnt/c/dev/turboquant-llm
TABBY_DIR="${TABBY_DIR:-$HOME/tabbyAPI}"
HF_REPO_CACHE="/mnt/c/Users/Tusshar Agarwaal/.cache/huggingface/hub/models--P4pps3n--KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq"
MODEL_DIR="${KAT_MODEL_DIR:-$HOME/models/kat-exl3}"
MODEL_NAME="KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq"
PORT="${PORT:-8000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"

usage() {
  cat <<'EOF'
Usage: serve_kat.sh [OPTIONS] [CONTEXT_LENGTH]

Start KAT-Coder EXL3 (TabbyAPI) on http://0.0.0.0:8000.

Options:
  -c, --context LENGTH       Max context (rounded down to multiple of 256)
  -p, --port PORT            Listen port (default 8000)
  -h, --help                 Show this help

Prerequisites:
  bash scripts/setup_tabbyapi.sh
  hf download P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--context|--max-model-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
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
        MAX_SEQ_LEN="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

# TabbyAPI requires cache_size / max_seq_len to be a multiple of 256.
REQUESTED_SEQ_LEN="$MAX_SEQ_LEN"
MAX_SEQ_LEN=$(( (MAX_SEQ_LEN / 256) * 256 ))
if [[ "$MAX_SEQ_LEN" -lt 256 ]]; then
  MAX_SEQ_LEN=256
fi
if [[ "$MAX_SEQ_LEN" != "$REQUESTED_SEQ_LEN" ]]; then
  echo "WARN: TabbyAPI requires context multiple of 256; using ${MAX_SEQ_LEN} (requested ${REQUESTED_SEQ_LEN})" >&2
fi

ensure_tabbyapi() {
  if [[ -f "$TABBY_DIR/venv/bin/activate" ]] && [[ -x "$TABBY_DIR/venv/bin/python" ]]; then
    return 0
  fi
  echo "TabbyAPI venv missing or incomplete — running setup ..."
  bash "$APP_DIR/scripts/setup_tabbyapi.sh"
}

ensure_tabbyapi

SNAPSHOT="$(find "$HF_REPO_CACHE/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)"
if [[ -z "$SNAPSHOT" || ! -f "$SNAPSHOT/model-00001-of-00003.safetensors" ]]; then
  echo "Missing EXL3 weights. Download with:" >&2
  echo "  hf download P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
ln -sfn "$SNAPSHOT" "$MODEL_DIR/$MODEL_NAME"

MODEL_CONFIG="$APP_DIR/config/kat_tabby_model.yml"
SYSMEM_KV_CACHE="${SYSMEM_KV_CACHE:-0}"
if [[ "$MAX_SEQ_LEN" -ge 65536 ]]; then
  MODEL_CONFIG="$APP_DIR/config/kat_tabby_model_long.yml"
  SYSMEM_KV_CACHE="${SYSMEM_KV_CACHE:-16384}"
  echo "Long-context mode: KV spillover via system RAM (sysmem_kv_cache=${SYSMEM_KV_CACHE} MiB)." >&2
  echo "NOTE: Intel GPU offload is not supported by ExLlamaV3/TabbyAPI (NVIDIA CUDA only)." >&2
fi

cp "$MODEL_CONFIG" "$MODEL_DIR/$MODEL_NAME/tabby_config.yml"

# Patch context into per-model overrides.
sed -i "s/max_seq_len: .*/max_seq_len: ${MAX_SEQ_LEN}/" "$MODEL_DIR/$MODEL_NAME/tabby_config.yml"
sed -i "s/cache_size: .*/cache_size: ${MAX_SEQ_LEN}/" "$MODEL_DIR/$MODEL_NAME/tabby_config.yml"

TABBY_CONFIG="$TABBY_DIR/config.yml"
sed "s|MODEL_DIR_PLACEHOLDER|${MODEL_DIR}|g; \
  s/max_seq_len: .*/max_seq_len: ${MAX_SEQ_LEN}/; \
  s/cache_size: .*/cache_size: ${MAX_SEQ_LEN}/; \
  s/port: .*/port: ${PORT}/; \
  s/sysmem_kv_cache: .*/sysmem_kv_cache: ${SYSMEM_KV_CACHE}/" \
  "$APP_DIR/config/tabby_kat_config.yml" > "$TABBY_CONFIG"

echo "KAT-Coder EXL3 (TabbyAPI) config:"
echo "  model_dir=$MODEL_DIR"
echo "  model_name=$MODEL_NAME"
echo "  max_seq_len=$MAX_SEQ_LEN"
echo "  cache_mode=Q8 (ExLlamaV3 KV quant — not vLLM TurboQuant)"
echo "  sysmem_kv_cache=${SYSMEM_KV_CACHE} MiB (0 = GPU-only KV tier)"
echo "  port=$PORT"
echo "  snapshot=$SNAPSHOT"

cd "$TABBY_DIR"
if [[ ! -f venv/bin/activate ]]; then
  echo "TabbyAPI setup failed: $TABBY_DIR/venv/bin/activate not found" >&2
  exit 1
fi
# shellcheck disable=SC1091
source venv/bin/activate
exec python main.py --config "$TABBY_CONFIG"
