#!/usr/bin/env bash
# install.sh — copy SM86 patches over an existing vllm install.
#
# Usage:
#   ./install.sh /path/to/vllm-env/lib/python3.X/site-packages/vllm
#
# Default target: ./vllm-env in the current directory.

set -euo pipefail

TARGET="${1:-./vllm-env/lib/python3.12/site-packages/vllm}"

if [ ! -d "$TARGET" ]; then
  echo "ERROR: target directory '$TARGET' does not exist." >&2
  echo "Pass the path to your vllm/ site-packages dir as the first argument." >&2
  exit 1
fi

if [ ! -f "$TARGET/utils/deep_gemm.py" ]; then
  echo "ERROR: '$TARGET' does not look like a vllm install (no utils/deep_gemm.py)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"

if [ ! -d "$PATCH_DIR" ]; then
  echo "ERROR: patches/ not found next to this script." >&2
  exit 1
fi

echo "Patching vllm at: $TARGET"
echo

backed_up=0
patched=0
added=0
for f in $(find "$PATCH_DIR" -type f -name '*.py' | sort); do
  rel="${f#$PATCH_DIR/}"
  dst="$TARGET/$rel"
  if [ ! -f "$dst" ]; then
    # NEW file (e.g. K11 flash_mla_decode_sm86.py, K12
    # fp8_paged_mqa_logits_sm86.py). These are SM86-only kernels that
    # do not exist in upstream vllm; copy as additions, no backup.
    mkdir -p "$(dirname "$dst")"
    cp "$f" "$dst"
    echo "  ADD   $rel"
    added=$((added + 1))
    continue
  fi
  if [ ! -f "$dst.sm86-orig" ]; then
    cp -p "$dst" "$dst.sm86-orig"
    backed_up=$((backed_up + 1))
  fi
  cp "$f" "$dst"
  echo "  PATCH $rel"
  patched=$((patched + 1))
done

echo
echo "Done. $patched modified, $added added, $backed_up original(s) saved as *.sm86-orig"
echo
echo "To use: launch vllm with the wrapper at wrapper-vllm-deepseek.sh"
echo "(set VLLM_BIN and MODEL_PATH at the top of the wrapper for your setup)."
echo
echo "Override flags:"
echo "  VLLM_SM86_DEEPSEEK_V4_REF=1   force enable (auto on SM<90)"
echo "  VLLM_SM86_DEEPSEEK_V4_REF=0   force disable"
echo "  VLLM_SM86_K11=1               enable old single-program FlashMLA decode (superseded by K13)"
echo "  VLLM_SM86_K12=1               enable experimental fused paged-MQA logits"
echo "  VLLM_SM86_K13=0               disable split-K FlashMLA decode (ON by default; 3.1x, +17% E2E)"
echo "  (K11 + K12 default OFF, K13 default ON; see README for tradeoffs)"
