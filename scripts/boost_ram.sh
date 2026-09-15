#!/bin/bash
# Best-effort RAM reclaim in WSL + Windows before loading large CPU models.
#
# Called by serve_llama33.sh. Never fails the caller — always prints status.
#
# Usage:
#   bash scripts/boost_ram.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

_drop_caches() {
  sync
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    # Never prompt for a password during automated boost.
    if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
      return 0
    fi
  fi
  echo "  drop_caches skipped (non-interactive; no passwordless sudo)"
}

_mem_kb() {
  local key="$1"
  awk -v k="$key" '$1 == k ":" { print $2; exit }' /proc/meminfo
}

_print_wsl_ram() {
  local label="$1"
  local total avail
  total="$(_mem_kb MemTotal)"
  avail="$(_mem_kb MemAvailable)"
  printf '%s: MemTotal=%.1f GB  MemAvailable=%.1f GB\n' \
    "$label" \
    "$(awk "BEGIN { printf \"%.1f\", $total / 1024 / 1024 }")" \
    "$(awk "BEGIN { printf \"%.1f\", $avail / 1024 / 1024 }")"
}

echo "WSL RAM boost (best-effort)..."
_print_wsl_ram "Before"

_drop_caches

_print_wsl_ram "After (WSL)"

# Invoke Windows booster when running under WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
  win_script="$REPO_DIR/scripts/boost_ram.ps1"
  win_script_ps="$(wsl_to_win_path "$win_script")"
  if command -v powershell.exe >/dev/null 2>&1 && [[ -f "$win_script" ]]; then
    echo
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$win_script_ps" || true
  else
    echo "Windows RAM boost skipped (powershell.exe or boost_ram.ps1 unavailable)"
  fi
fi
