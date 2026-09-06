#!/bin/bash
# Serve KAT-Coder from GGUF on llama.cpp with speculative decoding.
#
# Three speculation modes, selected with SPEC_MODE or --spec-mode:
#
#   cuda  (default)  Use the MTP head bundled inside the KAT GGUF, running on the
#                    RTX 5080 alongside the trunk. No second process, no NPU.
#   none             No speculation. Baseline for measuring the others.
#   npu              EXPERIMENTAL. Draft on the Intel NPU (or iGPU) in a Windows
#                    rpc-server process, attached over TCP as device RPC0.
#
# Why cuda is the default: on this machine the NPU decodes a 0.6B model at
# ~10 tok/s and the iGPU at ~20 tok/s, while the KAT trunk on CUDA already
# decodes faster than that. A drafter must be several times FASTER than the
# target to pay for itself, so the NPU cannot help here. See README.
#
# The EXL3/TabbyAPI path (scripts/serve_kat.sh) is unaffected by this script.
#
# Usage:
#   bash scripts/serve_kat_npu.sh
#   bash scripts/serve_kat_npu.sh --context 32768 --port 8000
#   SPEC_MODE=none bash scripts/serve_kat_npu.sh
#   SPEC_MODE=npu  bash scripts/serve_kat_npu.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="${KAT_GGUF_DIR:-$HOME/models/kat-gguf}"
TARGET_GGUF="${TARGET_GGUF:-$MODEL_DIR/KAT-Coder-V2.5-Dev_Q3_K_M_imatrix_MTP.gguf}"
SPEC_DRAFT_GGUF="${SPEC_DRAFT_GGUF:-$MODEL_DIR/mtp-Qwen3.6-35B-A3B-Q4_0.gguf}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kat-coder-npu}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-16384}"
PARALLEL="${PARALLEL:-1}"

# Record whether the caller pinned the offload split before defaults land, so the
# long-context profile further down never overrides an explicit choice.
_n_cpu_moe_set="${N_CPU_MOE:+yes}"
_n_gpu_layers_set="${N_GPU_LAYERS:+yes}"

# MoE expert offload. Empty = let auto-fit decide (required when N_GPU_LAYERS=auto).
# Set explicitly only with a numeric N_GPU_LAYERS, e.g. N_GPU_LAYERS=40 N_CPU_MOE=16.
N_CPU_MOE="${N_CPU_MOE:-}"
# Let llama.cpp fit layers to free VRAM (required at long ctx).
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"

CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# --load-mode supersedes the old --no-mmap. llama.cpp warns that mmap combined
# with MoE tensor overrides to CPU hurts throughput, so default to none.
LOAD_MODE="${LOAD_MODE:-none}"

# KV spill to CPU RAM for contexts that do not fit in VRAM (unset = llama default 8192 MiB).
CACHE_RAM="${CACHE_RAM:-}"

SPEC_MODE="${SPEC_MODE:-cuda}"
SPEC_N_MAX="${SPEC_N_MAX:-2}"

RPC_PORT="${RPC_PORT:-50052}"
# In WSL2 the Windows host is the default gateway.
WIN_HOST="${WIN_HOST:-$(ip route show default 2>/dev/null | awk '{print $3}' | head -1)}"

# Prebuilt binary has CUDA + spec-decode but NOT the RPC backend; the source
# build from scripts/setup_llamacpp_rpc.sh adds RPC for SPEC_MODE=npu.
LLAMA_PREBUILT="${LLAMA_PREBUILT:-$HOME/.local/bin/llama}"
LLAMA_RPC_BUILD="${LLAMA_RPC_BUILD:-$HOME/llamacpp-cuda/build/bin/llama-server}"

usage() {
  cat <<EOF
Usage: bash scripts/serve_kat_npu.sh [options]

  --context N       context size (default: $CTX_SIZE)
  --port N          listen port (default: $PORT)
  --spec-mode MODE  cuda | none | npu (default: $SPEC_MODE)
  --n-cpu-moe N     MoE layers on CPU; only with numeric N_GPU_LAYERS (default: auto-fit)
  --spec-n-max N    max drafted tokens per step (default: $SPEC_N_MAX)
  -h, --help        show this help

At --context 65536 and above, N_GPU_LAYERS defaults to 'all' and N_CPU_MOE to a
measured value (12 up to 100k, 16 beyond) instead of letting llama.cpp auto-fit,
which nearly doubles decode speed. Setting either variable disables the profile.

For maximum speed at the cost of some KV precision:
  CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 N_CPU_MOE=10 bash scripts/serve_kat_npu.sh --context 100000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)    CTX_SIZE="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --spec-mode)  SPEC_MODE="$2"; shift 2 ;;
    --n-cpu-moe)  N_CPU_MOE="$2"; _n_cpu_moe_set="yes"; shift 2 ;;
    --spec-n-max) SPEC_N_MAX="$2"; shift 2 ;;
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

# --cache-ram sizes the host-side prompt cache that lets a later request reuse a
# previous prompt's KV instead of reprocessing it. It is not KV spill: the active
# context always lives in VRAM. Long contexts make each cached state large, so
# lift the 8192 MiB cap or the cache stops being able to hold even one of them.
if [[ "$CTX_SIZE" -ge 131072 && -z "$CACHE_RAM" ]]; then
  CACHE_RAM="-1"
fi

# Long-context profile.
#
# Past ~64k, `--n-gpu-layers auto` balances its VRAM budget by moving whole
# layers to the CPU, attention included. That is the expensive half of the model
# to move: only a few experts fire per token, but every token needs every
# attention layer. Keeping all layers on the GPU and streaming just the MoE
# experts is much faster at the same VRAM.
#
# Measured on a 16 GB RTX 5080 laptop, Q8 KV, spec-mode cuda, 17.7k-token prompt:
#
#   ctx     config                          decode      VRAM
#   100k    auto-fit (previous default)      11 tok/s    14.5 GB
#   100k    all layers + --n-cpu-moe 12      19 tok/s    14.9 GB
#   200k    all layers + --n-cpu-moe 16      18 tok/s    15.0 GB
#
# The values below leave >1 GiB of VRAM free. That headroom is not optional: on
# Windows the driver pages VRAM out to host RAM instead of failing, and a card
# filled to the brim drops prompt eval from ~1400 to ~25 tok/s with no error.
if [[ "$CTX_SIZE" -ge 65536 ]]; then
  if [[ -z "$_n_gpu_layers_set" ]]; then
    N_GPU_LAYERS="all"
  fi
  if [[ -z "$_n_cpu_moe_set" && "$N_GPU_LAYERS" != "auto" ]]; then
    # Weights dominate VRAM here, not the KV cache: doubling the context from
    # 100k to 200k costs well under a gigabyte, so the split barely moves.
    if [[ "$CTX_SIZE" -le 100352 ]]; then
      N_CPU_MOE=12
    else
      N_CPU_MOE=16
    fi
  fi
fi

case "$SPEC_MODE" in
  cuda|none|npu) ;;
  *) echo "Invalid --spec-mode: $SPEC_MODE (expected cuda, none or npu)" >&2; exit 1 ;;
esac

if [[ ! -f "$TARGET_GGUF" ]]; then
  echo "Missing target GGUF: $TARGET_GGUF" >&2
  echo "Run: bash scripts/link_kat_gguf.sh" >&2
  exit 1
fi

# ---------------------------------------------------------- binary selection --
if [[ "$SPEC_MODE" == "npu" ]]; then
  if [[ ! -x "$LLAMA_RPC_BUILD" ]]; then
    echo "SPEC_MODE=npu needs a llama.cpp build with the RPC backend." >&2
    echo "Missing: $LLAMA_RPC_BUILD" >&2
    echo "Run: bash scripts/setup_llamacpp_rpc.sh" >&2
    exit 1
  fi
  LLAMA_CMD=("$LLAMA_RPC_BUILD")
else
  if [[ ! -x "$LLAMA_PREBUILT" ]]; then
    echo "Missing llama binary: $LLAMA_PREBUILT" >&2
    echo "Install: curl -LsSf https://llama.app/install.sh | sh" >&2
    exit 1
  fi
  LLAMA_CMD=("$LLAMA_PREBUILT" serve)
fi

# ------------------------------------------------------------ base arguments --
args=(
  --model "$TARGET_GGUF"
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
)

# --n-cpu-moe sets tensor_buft_overrides, which blocks common_fit_params when
# N_GPU_LAYERS=auto. Only pass it alongside an explicit layer count.
if [[ -n "$N_CPU_MOE" ]]; then
  if [[ "$N_GPU_LAYERS" == "auto" ]]; then
    echo "WARN: omitting --n-cpu-moe $N_CPU_MOE (conflicts with N_GPU_LAYERS=auto)." >&2
    echo "      Set N_GPU_LAYERS=<N> to pin MoE offload manually." >&2
  else
    args+=(--n-cpu-moe "$N_CPU_MOE")
  fi
fi

if [[ -n "$CACHE_RAM" ]]; then
  args+=(--cache-ram "$CACHE_RAM")
fi

# ------------------------------------------------------------- speculation ----
case "$SPEC_MODE" in
  none)
    args+=(--spec-type none)
    ;;

  cuda)
    # The KAT GGUF ships its own nextn layer (41 blocks = 40 + 1 head), so
    # draft-mtp needs no separate draft file here.
    args+=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_N_MAX")
    ;;

  npu)
    if [[ ! -f "$SPEC_DRAFT_GGUF" ]]; then
      echo "Missing draft GGUF: $SPEC_DRAFT_GGUF" >&2
      echo "Run: bash scripts/link_kat_gguf.sh" >&2
      exit 1
    fi

    # A mismatched MTP head loads fine but drafts tokens that are always
    # rejected, so verify the trunk/head geometry before burning a startup.
    if ! python3 "$REPO_DIR/scripts/gguf_compat.py" check "$TARGET_GGUF" "$SPEC_DRAFT_GGUF"; then
      echo "Refusing to start with an incompatible draft head." >&2
      exit 1
    fi

    if [[ -z "$WIN_HOST" ]]; then
      echo "Could not determine the Windows host IP for RPC." >&2
      echo "Set it explicitly: WIN_HOST=<ip> SPEC_MODE=npu bash scripts/serve_kat_npu.sh" >&2
      exit 1
    fi

    if ! timeout 5 bash -c "exec 3<>/dev/tcp/${WIN_HOST}/${RPC_PORT}" 2>/dev/null; then
      echo "No RPC server at ${WIN_HOST}:${RPC_PORT}." >&2
      echo "Start it on Windows first:" >&2
      echo "  powershell -ExecutionPolicy Bypass -File scripts\\npu_drafter.ps1" >&2
      echo "Then re-run. Diagnose with: bash scripts/check_npu_rpc.sh" >&2
      exit 1
    fi

    # --rpc must precede the device flags so RPC0 exists when they are parsed.
    # --device CUDA0 is explicit so trunk layers are never placed on RPC0.
    args=(--rpc "${WIN_HOST}:${RPC_PORT}" --device CUDA0 "${args[@]}")
    args+=(
      --spec-type draft-mtp
      --spec-draft-model "$SPEC_DRAFT_GGUF"
      --spec-draft-device RPC0
      --spec-draft-n-max "$SPEC_N_MAX"
    )
    ;;
esac

echo "KAT GGUF server config:"
echo "  BINARY            = ${LLAMA_CMD[*]}"
echo "  TARGET_GGUF       = $TARGET_GGUF"
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
echo "  SPEC_MODE         = $SPEC_MODE"
case "$SPEC_MODE" in
  cuda) echo "  speculation       = bundled MTP head on CUDA, n_max=$SPEC_N_MAX" ;;
  npu)  echo "  speculation       = $(basename "$SPEC_DRAFT_GGUF") on RPC0 (${WIN_HOST}:${RPC_PORT}), n_max=$SPEC_N_MAX" ;;
  none) echo "  speculation       = disabled" ;;
esac
echo

exec "${LLAMA_CMD[@]}" "${args[@]}"
