#!/bin/bash
# Serve Llama 3.3 70B Instruct from GGUF on llama.cpp.
#
# Default: hybrid GPU+CPU on RTX 5080 (n-gpu-layers auto) for usable decode speed.
# Pure CPU (0.5 tok/s class) is available with CPU_ONLY=1.
#
# Usage:
#   bash scripts/serve_llama33.sh
#   bash scripts/serve_llama33.sh --context 8192
#   CPU_ONLY=1 bash scripts/switch_model.sh llama33 --context 16384

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="${LLAMA33_GGUF_DIR:-$HOME/models/llama33}"
QUANT="${QUANT:-Q5_K_M}"
BASE="Llama-3.3-70B-Instruct-${QUANT}"

resolve_target_gguf() {
  local dir="$1" quant="$2"
  local base="Llama-3.3-70B-Instruct-${quant}"
  local single="$dir/${base}.gguf"
  local split="$dir/${base}/${base}-00001-of-00002.gguf"
  if [[ -n "${TARGET_GGUF:-}" ]]; then
    echo "$TARGET_GGUF"
  elif [[ -f "$single" ]]; then
    echo "$single"
  elif [[ -f "$split" ]]; then
    echo "$split"
  else
    echo "$split"
  fi
}

TARGET_GGUF="$(resolve_target_gguf "$MODEL_DIR" "$QUANT")"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama33-70b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"
MAX_CTX="${MAX_CTX:-131072}"

CPU_ONLY="${CPU_ONLY:-0}"
_n_gpu_layers_set="${N_GPU_LAYERS:+yes}"

# Hybrid default: fit as many of the 80 layers as 16 GB VRAM allows.
# Override: N_GPU_LAYERS=35, CPU_ONLY=1 (force 0), etc.
if [[ "$CPU_ONLY" == "1" ]]; then
  N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
else
  N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
fi

# Ultra 9 275HX: 24 physical cores, no hyperthreading — nproc is correct.
THREADS="${THREADS:-$(nproc)}"
THREADS_BATCH="${THREADS_BATCH:-$THREADS}"

SKIP_BOOST="${SKIP_BOOST:-0}"

# KV / load tuned for hybrid: q4 KV leaves VRAM for more GPU layers.
_cache_k_set="${CACHE_TYPE_K:+yes}"
_cache_v_set="${CACHE_TYPE_V:+yes}"
if [[ "$CPU_ONLY" == "1" ]]; then
  LOAD_MODE="${LOAD_MODE:-mmap}"
  CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
  CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
else
  # mmap: do not copy the full ~40 GB GGUF into RSS; GPU takes hot layers.
  LOAD_MODE="${LOAD_MODE:-mmap}"
  CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
  CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
fi

MIN_WSL_TOTAL_GB="${MIN_WSL_TOTAL_GB:-48}"
MIN_WSL_TOTAL_HYBRID_GB="${MIN_WSL_TOTAL_HYBRID_GB:-32}"
HEADROOM_GB="${HEADROOM_GB:-6}"
MIN_HOST_FREE_GB="${MIN_HOST_FREE_GB:-8}"
MAX_HOST_USED_PCT="${MAX_HOST_USED_PCT:-88}"

LLAMA_PREBUILT="${LLAMA_PREBUILT:-$HOME/.local/bin/llama}"

usage() {
  cat <<EOF
Usage: bash scripts/serve_llama33.sh [options]

  --context N     context size (default: $CTX_SIZE; native max $MAX_CTX)
  --port N        listen port (default: $PORT)
  -h, --help      show this help

Default profile: hybrid GPU (N_GPU_LAYERS=auto) on RTX 5080 — target ~10-20 tok/s decode.
Pure CPU is ~0.5 tok/s on this hardware; use only if GPU must stay free.

Environment:
  CPU_ONLY=1                Force CPU-only (N_GPU_LAYERS=0, slow decode)
  N_GPU_LAYERS=auto|N       GPU layer offload (default: auto unless CPU_ONLY=1)
  QUANT=Q5_K_M              GGUF quant filename suffix
  TARGET_GGUF=...           Override full GGUF path
  THREADS / THREADS_BATCH   CPU threads (default: nproc)
  LOAD_MODE                 mmap (default; avoids duplicating 40 GB in RSS)
  SKIP_BOOST=1              Skip boost_ram.sh
  CACHE_TYPE_K/V            q4_0 hybrid default; q8_0 CPU-only default

WSL memory (CPU-only): MemTotal >= ${MIN_WSL_TOTAL_GB} GB — run scripts/setup_wsl_memory.ps1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)    CTX_SIZE="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
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

if [[ "$CTX_SIZE" -gt "$MAX_CTX" ]]; then
  echo "WARN: CTX_SIZE=$CTX_SIZE exceeds native max $MAX_CTX; clamping." >&2
  CTX_SIZE="$MAX_CTX"
fi

if [[ "$CTX_SIZE" -ge 65536 ]]; then
  if [[ -z "$_cache_k_set" ]]; then CACHE_TYPE_K="q4_0"; fi
  if [[ -z "$_cache_v_set" ]]; then CACHE_TYPE_V="q4_0"; fi
fi

if [[ ! -f "$TARGET_GGUF" ]]; then
  echo "Missing GGUF: $TARGET_GGUF" >&2
  echo "Run: bash scripts/download_llama33.sh && bash scripts/link_llama33_gguf.sh" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_PREBUILT" ]]; then
  echo "Missing llama binary: $LLAMA_PREBUILT" >&2
  echo "Install: curl -LsSf https://llama.app/install.sh | sh" >&2
  exit 1
fi

if [[ "$N_GPU_LAYERS" != "0" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARN: nvidia-smi not found; falling back to CPU_ONLY=1." >&2
  N_GPU_LAYERS=0
  CPU_ONLY=1
  LOAD_MODE="${LOAD_MODE:-mmap}"
fi

_mem_kb() {
  awk -v k="$1" '$1 == k ":" { print $2; exit }' /proc/meminfo
}

estimate_kv_gb() {
  local ctx="$1" k_type="$2" v_type="$3"
  local k_bytes v_bytes
  case "$k_type" in
    q8_0) k_bytes=1 ;;
    q4_0|q4_1) k_bytes=0.5625 ;;
    f16) k_bytes=2 ;;
    *) k_bytes=1 ;;
  esac
  case "$v_type" in
    q8_0) v_bytes=1 ;;
    q4_0|q4_1) v_bytes=0.5625 ;;
    f16) v_bytes=2 ;;
    *) v_bytes=1 ;;
  esac
  awk -v ctx="$ctx" -v kb="$k_bytes" -v vb="$v_bytes" \
    'BEGIN { kv = 80 * ctx * 8 * 128 * (kb + vb); printf "%.2f", kv / 1024 / 1024 / 1024 }'
}

weight_gb() {
  local path="$1" parent bytes
  parent="$(dirname "$path")"
  if [[ "$parent" != "$MODEL_DIR" && "$parent" != "." ]]; then
    bytes="$(du -sb "$parent" 2>/dev/null | awk '{print $1}')"
  else
    bytes="$(du -b "$path" 2>/dev/null | awk '{print $1}')"
  fi
  if [[ -n "$bytes" && "$bytes" -gt 0 ]]; then
    awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }'
  else
    echo "50.0"
  fi
}

preflight_host() {
  local ps1 win_ps1 rc=0
  ps1="$REPO_DIR/scripts/check_host_memory.ps1"
  if ! command -v powershell.exe >/dev/null 2>&1 || [[ ! -f "$ps1" ]]; then
    return 0
  fi
  win_ps1="$(wsl_to_win_path "$ps1")"
  echo "Host RAM preflight:"
  if ! out="$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$win_ps1" \
      -MinFreeGB "$MIN_HOST_FREE_GB" -MaxUsedPct "$MAX_HOST_USED_PCT" 2>&1)"; then
    rc=1
  fi
  while IFS= read -r line; do
    [[ "$line" =~ ^HOST_ ]] && echo "  $line"
  done <<< "$out"
  if [[ "$rc" -ne 0 ]]; then
    echo >&2
    echo "ERROR: Windows host memory is under pressure (pagefile thrashing likely)." >&2
    echo "  - Stop llama33 and close heavy apps (Chrome, other models)" >&2
    echo "  - Re-run: powershell -ExecutionPolicy Bypass -File scripts\\setup_wsl_memory.ps1" >&2
    echo "  - Then: wsl --shutdown  and reopen WSL (targets 48 GB WSL on 64 GB host)" >&2
    exit 1
  fi
}

wsl_to_win_path() {
  local p="$1"
  if command -v wslpath >/dev/null 2>&1; then
    wslpath -w "$p"
    return
  fi
  if [[ "$p" =~ ^/mnt/([a-zA-Z])/(.*)$ ]]; then
    local drive="${BASH_REMATCH[1]}"
    local rest="${BASH_REMATCH[2]}"
    printf '%s:\\%s\n' "$(echo "$drive" | tr '[:lower:]' '[:upper:]')" "${rest//\//\\}"
    return
  fi
  echo "$p"
}

preflight_ram() {
  local total_kb avail_kb total_gb avail_gb w_gb kv_gb need_gb min_total cpu_frac
  total_kb="$(_mem_kb MemTotal)"
  avail_kb="$(_mem_kb MemAvailable)"
  total_gb="$(awk "BEGIN { printf \"%.1f\", $total_kb / 1024 / 1024 }")"
  avail_gb="$(awk "BEGIN { printf \"%.1f\", $avail_kb / 1024 / 1024 }")"
  w_gb="$(weight_gb "$TARGET_GGUF")"
  kv_gb="$(estimate_kv_gb "$CTX_SIZE" "$CACHE_TYPE_K" "$CACHE_TYPE_V")"

  if [[ "$N_GPU_LAYERS" == "0" ]]; then
    min_total="$MIN_WSL_TOTAL_GB"
    need_gb="$(awk -v w="$w_gb" -v kv="$kv_gb" -v h="$HEADROOM_GB" \
      'BEGIN { printf "%.1f", w + kv + h }')"
  else
    min_total="$MIN_WSL_TOTAL_HYBRID_GB"
    # Hybrid + mmap: CPU holds un-offloaded layers only (~35-45% of weights).
    cpu_frac="0.40"
    need_gb="$(awk -v w="$w_gb" -v kv="$kv_gb" -v h="$HEADROOM_GB" -v f="$cpu_frac" \
      'BEGIN { printf "%.1f", w * f + kv + h }')"
  fi

  echo "RAM preflight:"
  echo "  WSL MemTotal      = ${total_gb} GB (need >= ${min_total} GB)"
  echo "  WSL MemAvailable  = ${avail_gb} GB"
  echo "  Weights (GGUF)    = ${w_gb} GB"
  echo "  KV estimate       = ${kv_gb} GB (ctx=$CTX_SIZE, $CACHE_TYPE_K/$CACHE_TYPE_V)"
  echo "  Required avail    = ${need_gb} GB"

  if awk -v t="$total_gb" -v min="$min_total" 'BEGIN { exit !(t < min) }'; then
    echo >&2
    echo "ERROR: WSL MemTotal (${total_gb} GB) is below ${min_total} GB." >&2
    echo "Run: powershell -ExecutionPolicy Bypass -File scripts\\setup_wsl_memory.ps1" >&2
    echo "Then: wsl --shutdown  and reopen WSL." >&2
    exit 1
  fi

  if awk -v a="$avail_gb" -v n="$need_gb" 'BEGIN { exit !(a < n) }'; then
    echo >&2
    echo "ERROR: MemAvailable (${avail_gb} GB) < required (${need_gb} GB)." >&2
    echo "Close other apps, ensure boost ran, or lower --context." >&2
    exit 1
  fi

  if [[ "$N_GPU_LAYERS" != "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU VRAM          = $(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
  fi
}

preflight_host

if [[ "$SKIP_BOOST" != "1" ]]; then
  bash "$REPO_DIR/scripts/boost_ram.sh"
  echo
fi

preflight_ram

args=(
  --model "$TARGET_GGUF"
  --alias "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX_SIZE"
  --parallel "$PARALLEL"
  --n-gpu-layers "$N_GPU_LAYERS"
  --threads "$THREADS"
  --threads-batch "$THREADS_BATCH"
  --flash-attn on
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --load-mode "$LOAD_MODE"
  --jinja
  --spec-type none
)

mode_label="hybrid GPU+CPU"
if [[ "$N_GPU_LAYERS" == "0" ]]; then
  mode_label="CPU-only (expect ~0.5 tok/s decode)"
fi

echo "Llama 3.3 70B server config:"
echo "  BINARY            = $LLAMA_PREBUILT serve"
echo "  TARGET_GGUF       = $TARGET_GGUF"
echo "  SERVED_MODEL_NAME = $SERVED_MODEL_NAME"
echo "  HOST:PORT         = $HOST:$PORT"
echo "  CTX_SIZE          = $CTX_SIZE"
echo "  MODE              = $mode_label"
echo "  N_GPU_LAYERS      = $N_GPU_LAYERS"
echo "  THREADS           = $THREADS (batch=$THREADS_BATCH)"
echo "  KV cache          = $CACHE_TYPE_K / $CACHE_TYPE_V"
echo "  LOAD_MODE         = $LOAD_MODE"
echo "  PARALLEL          = $PARALLEL"
echo "  speculation       = disabled"
if [[ "$N_GPU_LAYERS" != "0" ]]; then
  echo "  tip               = lower --context frees VRAM for more GPU layers (faster decode)"
fi
echo

exec "$LLAMA_PREBUILT" serve "${args[@]}"
