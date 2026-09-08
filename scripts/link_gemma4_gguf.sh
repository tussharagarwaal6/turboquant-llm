#!/bin/bash
# Link Gemma 4 26B A4B GGUF trunk + mmproj from the Windows HF cache into ~/models/gemma4/.
#
# Downloads nothing. Weights live in the Windows HF cache; this only resolves
# snapshot directories and creates symlinks, mirroring scripts/link_kat_gguf.sh.
#
# Usage:
#   bash scripts/link_gemma4_gguf.sh

set -euo pipefail

HF_HUB="${HF_HUB:-/mnt/c/Users/Tusshar Agarwaal/.cache/huggingface/hub}"
MODEL_DIR="${GEMMA4_GGUF_DIR:-$HOME/models/gemma4}"

REPO="ggml-org/gemma-4-26B-A4B-it-GGUF"
TEXT_FILE="gemma-4-26B-A4B-it-Q4_0.gguf"
MMPROJ_FILE="mmproj-gemma-4-26B-A4B-it-Q8_0.gguf"

# Fallback repo if ggml-org Q4_0 is not cached (imatrix IQ4_XS, ~13 GB).
FALLBACK_REPO="batiai/Gemma-4-26B-A4B-it-GGUF"
FALLBACK_TEXT="google-gemma-4-26B-A4B-it-IQ4_XS.gguf"
FALLBACK_MMPROJ="mmproj-Q6_K.gguf"

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

echo "Linking Gemma 4 GGUF weights into $MODEL_DIR"
rc=0

if ! link_one "$REPO" "$TEXT_FILE"; then
  echo "  primary trunk not cached: $REPO/$TEXT_FILE" >&2
  if link_one "$FALLBACK_REPO" "$FALLBACK_TEXT"; then
    echo "  using fallback trunk: $FALLBACK_TEXT"
    TEXT_FILE="$FALLBACK_TEXT"
  else
    echo "  Download with: bash scripts/download_gemma4.sh" >&2
    echo "    hf download $REPO $TEXT_FILE $MMPROJ_FILE" >&2
    rc=1
  fi
fi

if ! link_one "$REPO" "$MMPROJ_FILE"; then
  if link_one "$FALLBACK_REPO" "$FALLBACK_MMPROJ"; then
    echo "  using fallback mmproj: $FALLBACK_MMPROJ"
    MMPROJ_FILE="$FALLBACK_MMPROJ"
  else
    echo "  missing mmproj: $MMPROJ_FILE" >&2
    echo "  Download with: bash scripts/download_gemma4.sh" >&2
    rc=1
  fi
fi

if [[ "$rc" -ne 0 ]]; then
  exit 1
fi

echo
echo "Done. Set for serve if using fallback filenames:"
echo "  TARGET_GGUF=$MODEL_DIR/$TEXT_FILE"
echo "  MMPROJ_GGUF=$MODEL_DIR/$MMPROJ_FILE"
echo
echo "Contents:"
ls -lL "$MODEL_DIR" | tail -n +2
