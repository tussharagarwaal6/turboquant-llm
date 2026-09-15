#!/bin/bash
# Link Qwen3.5-4B GGUF from the Windows HF cache into ~/models/qwen35-4b/.
#
# Downloads nothing. Mirrors scripts/link_gemma4_gguf.sh.
#
# Usage:
#   bash scripts/link_qwen35_4b_gguf.sh

set -euo pipefail

HF_HUB="${HF_HUB:-/mnt/c/Users/Tusshar Agarwaal/.cache/huggingface/hub}"
MODEL_DIR="${QWEN35_GGUF_DIR:-$HOME/models/qwen35-4b}"

REPO="${GGUF_REPO:-unsloth/Qwen3.5-4B-GGUF}"
TEXT_FILE="${TEXT_GGUF:-Qwen3.5-4B-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_GGUF:-mmproj-F16.gguf}"

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
  local snap src

  snap="$(_snapshot "$repo" || true)"
  if [[ -z "$snap" || ! -f "$snap/$file" ]]; then
    return 1
  fi

  src="$snap/$file"
  ln -sfn "$src" "$MODEL_DIR/$file"
  printf '  linked %-52s %s\n' "$file" "$(du -Lh "$src" 2>/dev/null | cut -f1)"
  return 0
}

if [[ ! -d "$HF_HUB" ]]; then
  echo "HF cache not found: $HF_HUB" >&2
  echo "Override with HF_HUB=/path/to/huggingface/hub" >&2
  exit 1
fi

echo "Linking Qwen3.5-4B GGUF into $MODEL_DIR"
rc=0

if [[ -f "$MODEL_DIR/$TEXT_FILE" ]]; then
  echo "  present $TEXT_FILE ($(du -Lh "$MODEL_DIR/$TEXT_FILE" | cut -f1))"
elif ! link_one "$REPO" "$TEXT_FILE"; then
  echo "  Missing: $REPO/$TEXT_FILE" >&2
  rc=1
fi

if [[ -f "$MODEL_DIR/$MMPROJ_FILE" ]]; then
  echo "  present $MMPROJ_FILE ($(du -Lh "$MODEL_DIR/$MMPROJ_FILE" | cut -f1))"
elif ! link_one "$REPO" "$MMPROJ_FILE"; then
  echo "  Missing: $REPO/$MMPROJ_FILE" >&2
  rc=1
fi

if [[ "$rc" -ne 0 ]]; then
  echo "  Download with: bash scripts/download_qwen35_4b.sh" >&2
  exit 1
fi

echo
echo "Done. Start with:"
echo "  bash scripts/switch_model.sh qwen35-4b --context 262144"
echo
ls -lL "$MODEL_DIR" | tail -n +2
