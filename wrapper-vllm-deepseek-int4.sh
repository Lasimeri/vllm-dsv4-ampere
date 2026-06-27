#!/usr/bin/env bash
# Serve the INT4 (W4A16 AutoRound) DeepSeek-V4-Flash on Ampere SM86.
#
# Model:  Intel/DeepSeek-V4-Flash-W4A16-AutoRound  (~145 GiB, int4 GPTQ experts
#         via auto_round:auto_gptq packing + selective bf16 layers).
# Unlike the deepseek-ai FP4+FP8 checkpoint, the routed experts here run the
# hardware Marlin int4 kernel ("Using 'MARLIN' WNA16 MoE backend"), so the
# decode floor is the sparse-MLA/indexer pyref, not weight dequant.
#
# Requires the INT4/AutoRound patches applied (see README "INT4 / AutoRound").
set -euo pipefail

# vllm CLI (or activate the venv first).
VLLM_BIN="${VLLM_BIN:-$(command -v vllm 2>/dev/null)}"
if [ -z "$VLLM_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "ERROR: vllm CLI not found. Set VLLM_BIN=/path/to/vllm or activate the venv." >&2
  exit 1
fi
# The venv bin must be on PATH so FlashInfer's runtime JIT can find `ninja`.
VENV_BIN="$(dirname "$VLLM_BIN")"

export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export PATH="${VENV_BIN}:${CUDA_HOME}/bin:$PATH"
# FlashInfer JIT must use clang on hosts where gcc>=15 rejects torch's headers.
export CXX="${CXX:-clang++}" CC="${CC:-clang}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"   # Xet CAS can stall; classic HTTP is reliable
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_RPC_TIMEOUT=600000
export NVIDIA_TF32_OVERRIDE=0
export VLLM_SM86_DEEPSEEK_V4_REF=1   # auto on SM<90; explicit for clarity

MODEL_PATH="${MODEL_PATH:-Intel/DeepSeek-V4-Flash-W4A16-AutoRound}"

# offload 8 GiB/GPU + ctx 32k keeps more weights resident than the 14.81/100k
# default (the 145 GiB int4 model is smaller than the FP4+FP8 one): ~3.84 tok/s
# on 8x RTX 3080 20GB vs ~2.7 at offload 14.81. Tune to your VRAM.
exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name DeepSeek-V4-Flash \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --max-num-seqs 1 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --override-generation-config '{"temperature": 1.0, "top_p": 1.0}' \
  --host 0.0.0.0 --port 8000
