#!/usr/bin/env bash
# vllm-deepseek: day-0 wrapper for DeepSeek-V4-Flash (deepseek-ai/DeepSeek-V4-Flash)
# Reference: https://vllm.ai/blog/deepseek-v4 (official vLLM day-0 launch post)
#
# HARDWARE NOTE (read before launching):
#   This box is 8x RTX 3080 20GB (Ampere SM 8.6, 160 GB total VRAM).
#   DeepSeek-V4-Flash is ~158 GB on disk in mixed FP4+FP8+BF16. Even ignoring
#   activations/KV cache, weights barely fit, and Ampere has no native FP8/FP4
#   tensor cores, so --kv-cache-dtype fp8 and use_fp4_indexer_cache will fall
#   back to emulation or refuse to load on this hardware. Expect this script
#   to either OOM, run extremely slowly, or fail on FP8 kernel dispatch.
#   Recommended target: H100/H200/B200 class hardware, or a quantized rebuild.
#
# Configure for your setup:
#   VLLM_BIN     path to the vllm CLI (default: `which vllm`)
#   MODEL_PATH   HF id or local snapshot directory (default: deepseek-ai/DeepSeek-V4-Flash)
#                vLLM will download from HF on first run unless MODEL_PATH points to a local mirror.

# Find the vllm binary; override via VLLM_BIN env var if needed.
VLLM_BIN="${VLLM_BIN:-$(command -v vllm 2>/dev/null)}"
if [ -z "$VLLM_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "ERROR: vllm CLI not found. Set VLLM_BIN=/path/to/vllm or activate the venv first." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export PATH="${CUDA_HOME}/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_RPC_TIMEOUT=600000
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# Disable TF32 at the driver level for IEEE-correct FP32 residual math
# (norms, softmax fallbacks). Matmul still runs in BF16/FP8/FP4 per dtype.
export NVIDIA_TF32_OVERRIDE=0
# Force Marlin for MXFP4 MoE. On SM86 this is the only backend that supports
# kMxfp4Static (DeepGEMM FP4 needs SM100+, TRTLLM needs SM90+). Marlin SASS
# fallback target on SM86 is sm_80 (no sm_86 cubin in _moe_C.abi3.so).
export VLLM_MXFP4_USE_MARLIN=1

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "DeepSeek-V4-Flash" \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 14.81 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --max-num-seqs 1 \
  --max-model-len 100000 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --override-generation-config '{"temperature": 1.0, "top_p": 1.0}' \
  --host 0.0.0.0 \
  --port 8000
