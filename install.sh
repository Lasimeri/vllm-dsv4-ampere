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
for f in $(find "$PATCH_DIR" -type f -name '*.py' | sort); do
  rel="${f#$PATCH_DIR/}"
  dst="$TARGET/$rel"
  if [ ! -f "$dst" ]; then
    echo "  SKIP  $rel  (not present in target — vllm version mismatch?)"
    continue
  fi
  if [ ! -f "$dst.sm86-orig" ]; then
    cp -p "$dst" "$dst.sm86-orig"
    backed_up=$((backed_up + 1))
  fi
  cp "$f" "$dst"
  echo "  PATCH $rel"
done

echo
echo "Done. $backed_up file(s) had their original saved as *.sm86-orig"
echo
echo "To use: launch vllm with the wrapper at wrapper-vllm-deepseek.sh"
echo "(adjust MODEL_PATH and venv path inside it for your setup)."
echo
echo "Patches auto-activate when torch.cuda.get_device_capability() < (9,0)."
echo "To force on/off: VLLM_SM86_DEEPSEEK_V4_REF=1 / =0"
