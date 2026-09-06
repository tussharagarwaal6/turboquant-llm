#!/bin/bash
# Benchmark KAT-Coder GGUF decode throughput across speculation configurations.
#
# Starts scripts/serve_kat_npu.sh once per configuration, sends the same coding
# prompt, and records prompt-eval and decode tok/s plus the draft acceptance rate
# reported by the server. Each configuration is a cold start, so a full sweep
# takes a while -- the 18 GB trunk has to be read every time.
#
# Configurations:
#   none   no speculation (the baseline every other number is judged against)
#   cuda   MTP head bundled in the KAT GGUF, drafting on the RTX 5080
#   npu    draft head on the Intel NPU via RPC (needs the Windows drafter and
#          the RPC-enabled build; skipped automatically if unavailable)
#
# N_GPU_LAYERS defaults to `all` here, not `auto`: serve_kat_npu.sh drops
# --n-cpu-moe when layers are auto-fit, which would silently make MOE_SWEEP a
# no-op and have every row report the same auto-chosen split.
#
# PROMPT_TOKENS prepends synthetic filler so decode is measured at a realistic KV
# occupancy. Measuring at an almost-empty context flatters every configuration
# equally and hides how each one behaves at the context you actually run.
#
# PROFILE=short|long|max selects a context size, prompt length, and n_cpu_moe
# range that brackets the optimum for that context: short=8k, long=100k, max=200k.
#
# Usage:
#   bash scripts/bench_kat_npu.sh
#   PROFILE=long REPEATS=5 bash scripts/bench_kat_npu.sh
#   PROFILE=max MODES=cuda REPEATS=3 bash scripts/bench_kat_npu.sh
#   PROFILE=max CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 bash scripts/bench_kat_npu.sh
#   MODES="none cuda" bash scripts/bench_kat_npu.sh
#   SPEC_N_SWEEP="1 2 3" MOE_SWEEP="8 16 24" bash scripts/bench_kat_npu.sh

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${PORT:-8000}"

# PROFILE picks a context size, a prompt length to measure at, and an n_cpu_moe
# range known to bracket the optimum for that context, so the common sweeps are
# one word instead of five variables. Anything set explicitly still wins.
case "${PROFILE:-}" in
  "")
    ;;
  short)
    CTX_SIZE="${CTX_SIZE:-8192}"
    PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
    MOE_SWEEP="${MOE_SWEEP:-12 16 20}"
    ;;
  long)
    CTX_SIZE="${CTX_SIZE:-100000}"
    PROMPT_TOKENS="${PROMPT_TOKENS:-16000}"
    MOE_SWEEP="${MOE_SWEEP:-10 12 14}"
    ;;
  max)
    CTX_SIZE="${CTX_SIZE:-200000}"
    PROMPT_TOKENS="${PROMPT_TOKENS:-16000}"
    MOE_SWEEP="${MOE_SWEEP:-14 16 18}"
    ;;
  *)
    echo "Unknown PROFILE: $PROFILE (expected short, long or max)" >&2
    exit 1
    ;;
esac

CTX_SIZE="${CTX_SIZE:-8192}"
N_PREDICT="${N_PREDICT:-256}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
# Decode tok/s varies by several tok/s between identical requests, mostly from
# GPU clocks and whatever else is using the card. Probe more than once per cold
# start -- repeats are cheap, reloading the 18 GB trunk is not.
REPEATS="${REPEATS:-1}"

# A configuration that fills VRAM does not fail: the driver pages weights back
# and forth to host RAM and everything crawls (prompt eval drops from ~1500 to
# ~25 tok/s, so a single probe can take 15 minutes). Skip those rows instead of
# waiting for them, and leave room for the desktop, which shares this GPU.
SPILL_MARGIN_MIB="${SPILL_MARGIN_MIB:-600}"

# Passed through to serve_kat_npu.sh so a sweep can cover KV precision and read
# the trunk from a local ext4 copy instead of /mnt/c (much faster cold starts).
N_GPU_LAYERS="${N_GPU_LAYERS:-all}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
TARGET_GGUF="${TARGET_GGUF:-}"

MODES="${MODES:-none cuda npu}"
SPEC_N_SWEEP="${SPEC_N_SWEEP:-2}"
MOE_SWEEP="${MOE_SWEEP:-16}"

RESULTS="${RESULTS:-$REPO_DIR/.bench_kat_npu.tsv}"
LOG_DIR="${LOG_DIR:-/tmp/kat_bench}"
mkdir -p "$LOG_DIR"

_npu_available() {
  local rpc_build="${LLAMA_RPC_BUILD:-$HOME/llamacpp-cuda/build/bin/llama-server}"
  local win_host rpc_port
  rpc_port="${RPC_PORT:-50052}"
  win_host="${WIN_HOST:-$(ip route show default 2>/dev/null | awk '{print $3}' | head -1)}"
  [[ -x "$rpc_build" ]] || return 1
  [[ -n "$win_host" ]] || return 1
  timeout 5 bash -c "exec 3<>/dev/tcp/${win_host}/${rpc_port}" 2>/dev/null || return 1
  return 0
}

_wait_ready() {
  local i
  for ((i = 0; i < STARTUP_TIMEOUT; i++)); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    # Bail out early if the server already died.
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

run_one() {
  local mode="$1" spec_n="$2" n_moe="$3"
  local tag="${mode}_n${spec_n}_moe${n_moe}_c${CTX_SIZE}_${CACHE_TYPE_K}"
  local log="$LOG_DIR/$tag.log"

  echo "=============================================================="
  echo "mode=$mode spec_n_max=$spec_n n_cpu_moe=$n_moe ctx=$CTX_SIZE kv=$CACHE_TYPE_K"
  echo "=============================================================="

  bash "$REPO_DIR/scripts/kill_gpu.sh" >/dev/null 2>&1
  sleep 3

  local started_at
  started_at=$(date +%s)

  SPEC_MODE="$mode" \
  SPEC_N_MAX="$spec_n" \
  N_CPU_MOE="$n_moe" \
  N_GPU_LAYERS="$N_GPU_LAYERS" \
  CACHE_TYPE_K="$CACHE_TYPE_K" \
  CACHE_TYPE_V="$CACHE_TYPE_V" \
  TARGET_GGUF="$TARGET_GGUF" \
  CTX_SIZE="$CTX_SIZE" \
  PORT="$PORT" \
    bash "$REPO_DIR/scripts/serve_kat_npu.sh" >"$log" 2>&1 &
  SERVER_PID=$!

  if ! _wait_ready; then
    echo "  SERVER FAILED TO START (see $log)"
    tail -5 "$log" | sed 's/^/    /'
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t0\tFAILED\tFAILED\tFAILED\tFAILED\tFAILED\tFAILED\n' \
      "$mode" "$spec_n" "$n_moe" "$CTX_SIZE" "$CACHE_TYPE_K" >>"$RESULTS"
    return 1
  fi

  local load_s=$(( $(date +%s) - started_at ))
  # KV and weight buffers are allocated during load, so this reading is the real
  # VRAM high-water mark for the configuration.
  local vram vram_total
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

  if (( vram_total - vram < SPILL_MARGIN_MIB )); then
    echo "  SKIPPED: only $(( vram_total - vram )) MiB VRAM free; this would page to host RAM."
    echo "           Raise n_cpu_moe, shorten the context, or quantise the KV cache."
    printf '%s\t%s\t%s\t%s\t%s\t0\tSPILL\tSPILL\tSPILL\tSPILL\t%s\t%s\n' \
      "$mode" "$spec_n" "$n_moe" "$CTX_SIZE" "$CACHE_TYPE_K" "$vram" "$load_s" >>"$RESULTS"
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 3
    return 0
  fi

  echo "  server up in ${load_s}s (VRAM ${vram} MiB); sending prompt ..."

  local rep
  for ((rep = 1; rep <= REPEATS; rep++)); do
    local metrics
    # The label must be non-empty: read strips leading tabs, so an empty first
    # field would shift every column left by one.
    metrics=$(python3 "$REPO_DIR/scripts/probe_llama_speed.py" \
      --base-url "http://127.0.0.1:${PORT}" \
      --label "$tag" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --n-predict "$N_PREDICT")

    local _label prompt_n pp ntok tg accept
    IFS=$'\t' read -r _label prompt_n pp ntok tg accept <<<"$metrics"

    echo "  rep $rep/$REPEATS  prompt-eval $pp tok/s ($prompt_n tok) | decode $tg tok/s ($ntok tok) | acceptance $accept"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$spec_n" "$n_moe" "$CTX_SIZE" "$CACHE_TYPE_K" "$rep" \
      "$prompt_n" "$pp" "$tg" "$accept" "$vram" "$load_s" >>"$RESULTS"
  done

  kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  sleep 3
  return 0
}

printf 'mode\tspec_n_max\tn_cpu_moe\tctx\tkv\trep\tprompt_n\tprompt_tps\tdecode_tps\tacceptance\tvram_mib\tload_s\n' >"$RESULTS"

for mode in $MODES; do
  if [[ "$mode" == "npu" ]] && ! _npu_available; then
    echo "Skipping mode=npu: no RPC-enabled build or no drafter on the RPC port."
    echo "  (see scripts/setup_llamacpp_rpc.sh and scripts/npu_drafter.ps1)"
    continue
  fi

  if [[ "$mode" == "none" ]]; then
    # Draft depth is meaningless without speculation; sweep MoE offload only.
    for n_moe in $MOE_SWEEP; do
      run_one "$mode" 1 "$n_moe"
    done
  else
    for spec_n in $SPEC_N_SWEEP; do
      for n_moe in $MOE_SWEEP; do
        run_one "$mode" "$spec_n" "$n_moe"
      done
    done
  fi
done

bash "$REPO_DIR/scripts/kill_gpu.sh" >/dev/null 2>&1

echo
echo "=============================================================="
echo "Results ($RESULTS)"
echo "=============================================================="
column -t -s $'\t' "$RESULTS"

# Report the median rather than the mean: the first repeat of every cold start
# is slower while GPU clocks ramp, and an occasional outlier would drag a mean
# far enough to pick the wrong configuration.
echo
echo "Median decode tok/s per configuration:"
awk -F'\t' '
  NR > 1 && $9 != "FAILED" && $9 != "SPILL" {
    key = $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5
    n[key]++
    tps[key, n[key]] = $9
    vram[key] = $11
  }
  END {
    printf "mode\tspec_n\tn_cpu_moe\tctx\tkv\treps\tmedian_tps\tvram_mib\n"
    for (key in n) {
      cnt = n[key]
      for (i = 1; i <= cnt; i++) { v[i] = tps[key, i] }
      for (i = 2; i <= cnt; i++) {
        x = v[i]
        for (j = i - 1; j >= 1 && v[j] > x; j--) { v[j + 1] = v[j] }
        v[j + 1] = x
      }
      med = (cnt % 2) ? v[(cnt + 1) / 2] : (v[cnt / 2] + v[cnt / 2 + 1]) / 2
      printf "%s\t%d\t%.2f\t%s\n", key, cnt, med, vram[key]
    }
  }
' "$RESULTS" | column -t -s $'\t'
echo
echo "Keep a speculation mode only if its decode tok/s beats mode=none."
echo
echo "Lower n_cpu_moe means fewer experts streamed over PCIe, but only up to the"
echo "point where VRAM overflows. A row whose vram_mib is close to the card total"
echo "and whose decode collapsed is a silent spill, not the optimum -- the fastest"
echo "setting is the lowest n_cpu_moe that still leaves the card some headroom."
