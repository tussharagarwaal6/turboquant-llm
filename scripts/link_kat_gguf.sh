#!/bin/bash
# Link the cached KAT GGUF trunk and MTP draft heads into ~/models/kat-gguf/.
#
# Downloads nothing. The weights live in the Windows HF cache; this only resolves
# the snapshot directories and creates symlinks, mirroring the approach used by
# scripts/serve_kat.sh for the EXL3 checkpoint.
#
# Usage:
#   bash scripts/link_kat_gguf.sh

set -euo pipefail

HF_HUB="${HF_HUB:-/mnt/c/Users/Tusshar Agarwaal/.cache/huggingface/hub}"
MODEL_DIR="${KAT_GGUF_DIR:-$HOME/models/kat-gguf}"

TRUNK_REPO="offmonreal/KAT-Coder-V2.5-Dev-MaxQuality-MTP-GGUF"
TRUNK_FILE="KAT-Coder-V2.5-Dev_Q3_K_M_imatrix_MTP.gguf"

DRAFT_REPO="ggml-org/Qwen3.6-35B-A3B-GGUF"
DRAFT_FILES=("mtp-Qwen3.6-35B-A3B-Q4_0.gguf" "mtp-Qwen3.6-35B-A3B-Q8_0.gguf")

mkdir -p "$MODEL_DIR"

# repo_id -> models--org--name
_cache_dir() {
  local repo="$1"
  echo "$HF_HUB/models--${repo//\//--}"
}

# Newest snapshot dir for a repo (HF keeps one per revision).
_snapshot() {
  local repo="$1"
  local base
  base="$(_cache_dir "$repo")/snapshots"
  [[ -d "$base" ]] || return 1
  find "$base" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1
}

link_one() {
  local repo="$1" file="$2" required="$3"
  local snap src

  snap="$(_snapshot "$repo" || true)"
  if [[ -z "$snap" || ! -f "$snap/$file" ]]; then
    if [[ "$required" == "required" ]]; then
      echo "MISSING: $file" >&2
      echo "  Download it on Windows with:" >&2
      echo "    hf download $repo $file" >&2
      return 1
    fi
    echo "  skip (not cached): $file"
    return 0
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

echo "Linking KAT GGUF weights into $MODEL_DIR"
rc=0
link_one "$TRUNK_REPO" "$TRUNK_FILE" required || rc=1
for f in "${DRAFT_FILES[@]}"; do
  # Draft heads are only needed for the experimental NPU/RPC drafting path.
  link_one "$DRAFT_REPO" "$f" optional || rc=1
done

if [[ "$rc" -ne 0 ]]; then
  echo >&2
  echo "One or more required files are missing; see messages above." >&2
  exit 1
fi

echo
echo "Done. Contents:"
ls -lL "$MODEL_DIR" | tail -n +2
