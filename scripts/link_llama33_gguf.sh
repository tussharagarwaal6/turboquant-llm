#!/bin/bash
# Link Llama 3.3 70B Instruct GGUF from the Windows HF cache into ~/models/llama33/.
#
# Usage:
#   bash scripts/link_llama33_gguf.sh
#   QUANT=Q5_K_S bash scripts/link_llama33_gguf.sh

set -euo pipefail

HF_HUB="${HF_HUB:-/mnt/c/Users/Tusshar Agarwaal/.cache/huggingface/hub}"
MODEL_DIR="${LLAMA33_GGUF_DIR:-$HOME/models/llama33}"
REPO="${GGUF_REPO:-bartowski/Llama-3.3-70B-Instruct-GGUF}"
QUANT="${QUANT:-Q5_K_M}"
BASE="Llama-3.3-70B-Instruct-${QUANT}"

is_split_quant() {
  case "$1" in
    Q5_K_M|Q5_K_L|Q6_K|Q6_K_L|Q8_0|f16) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$MODEL_DIR"

_cache_dir() {
  local repo="$1"
  echo "$HF_HUB/models--${repo//\//--}"
}

_snapshot() {
  local repo="$1"
  local base
  base="$(_cache_dir "$repo")/snapshots"
  [[ -d "$base" ]] || return 1
  find "$base" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1
}

link_one() {
  local repo="$1" file="$2"
  local snap src dest_dir

  snap="$(_snapshot "$repo" || true)"
  if [[ -z "$snap" || ! -f "$snap/$file" ]]; then
    return 1
  fi

  src="$snap/$file"
  dest_dir="$MODEL_DIR/$(dirname "$file")"
  mkdir -p "$dest_dir"
  ln -sfn "$src" "$MODEL_DIR/$file"
  printf '  linked %-60s %s\n' "$file" "$(du -Lh "$src" 2>/dev/null | cut -f1)"
  return 0
}

if [[ ! -d "$HF_HUB" ]]; then
  echo "HF cache not found: $HF_HUB" >&2
  exit 1
fi

echo "Linking Llama 3.3 70B ($QUANT) into $MODEL_DIR"
rc=0

if is_split_quant "$QUANT"; then
  for shard in "${BASE}-00001-of-00002.gguf" "${BASE}-00002-of-00002.gguf"; do
    rel="${BASE}/${shard}"
    if [[ -f "$MODEL_DIR/$rel" ]]; then
      echo "  present $rel ($(du -Lh "$MODEL_DIR/$rel" | cut -f1))"
    elif ! link_one "$REPO" "$rel"; then
      echo "  Missing: $REPO/$rel" >&2
      rc=1
    fi
  done
else
  file="${BASE}.gguf"
  if [[ -f "$MODEL_DIR/$file" ]]; then
    echo "  present $file ($(du -Lh "$MODEL_DIR/$file" | cut -f1))"
  elif ! link_one "$REPO" "$file"; then
    echo "  Missing: $REPO/$file" >&2
    rc=1
  fi
fi

if [[ "$rc" -ne 0 ]]; then
  echo "  Download with: bash scripts/download_llama33.sh" >&2
  exit 1
fi

echo
echo "Done. Start with:"
echo "  bash scripts/switch_model.sh llama33 --context 8192"
