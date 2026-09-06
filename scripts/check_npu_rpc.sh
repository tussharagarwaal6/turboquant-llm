#!/bin/bash
# Verify that the Windows-side OpenVINO drafting device is reachable from WSL2.
#
# Checks, in order:
#   1. the Windows host IP as seen from WSL (NAT default gateway)
#   2. TCP reachability of the rpc-server port (firewall / listener)
#   3. whether the local llama binary was built with the RPC backend
#   4. whether an RPC device actually enumerates as RPC0
#
# Usage:
#   bash scripts/check_npu_rpc.sh
#   WIN_HOST=172.20.0.1 RPC_PORT=50052 bash scripts/check_npu_rpc.sh

set -uo pipefail

RPC_PORT="${RPC_PORT:-50052}"
WIN_HOST="${WIN_HOST:-$(ip route show default 2>/dev/null | awk '{print $3}' | head -1)}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llamacpp-cuda/build/bin/llama-server}"

fail=0

echo "Windows host (WSL default gateway): ${WIN_HOST:-<unresolved>}"
if [[ -z "$WIN_HOST" ]]; then
  echo "  FAIL: could not resolve the Windows host IP" >&2
  exit 1
fi

echo -n "TCP ${WIN_HOST}:${RPC_PORT}: "
if timeout 5 bash -c "exec 3<>/dev/tcp/${WIN_HOST}/${RPC_PORT}" 2>/dev/null; then
  echo "reachable"
else
  echo "BLOCKED"
  echo "  Start the drafter on Windows:  powershell -File scripts\\npu_drafter.ps1" >&2
  echo "  If it is running, allow the port (elevated PowerShell):" >&2
  echo "    New-NetFirewallRule -DisplayName llamacpp-rpc-${RPC_PORT} -Direction Inbound -Action Allow -Protocol TCP -LocalPort ${RPC_PORT} -RemoteAddress 172.16.0.0/12" >&2
  fail=1
fi

echo -n "llama binary with RPC backend: "
if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "not built ($LLAMA_BIN)"
  echo "  The default CUDA+MTP mode does not need it. For the NPU path run:" >&2
  echo "    bash scripts/setup_llamacpp_rpc.sh" >&2
  fail=1
elif "$LLAMA_BIN" --help 2>&1 | grep -q -- '--rpc'; then
  echo "yes"
  echo "Device enumeration:"
  "$LLAMA_BIN" --rpc "${WIN_HOST}:${RPC_PORT}" --list-devices 2>&1 | sed 's/^/  /'
  if "$LLAMA_BIN" --rpc "${WIN_HOST}:${RPC_PORT}" --list-devices 2>&1 | grep -q 'RPC0'; then
    echo "  RPC0 present"
  else
    echo "  FAIL: RPC0 did not enumerate" >&2
    fail=1
  fi
else
  echo "no (built without GGML_RPC)"
  fail=1
fi

echo
if [[ "$fail" -eq 0 ]]; then
  echo "All checks passed; SPEC_MODE=npu is usable."
else
  echo "Some checks failed. SPEC_MODE=cuda (the default) is unaffected."
fi
exit "$fail"
