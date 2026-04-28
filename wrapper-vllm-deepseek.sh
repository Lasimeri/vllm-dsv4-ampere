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
# MODEL_PATH: prefer a local checkpoint at /llms/archive/deepseek-v4-flash if
# you mirror weights locally; otherwise vLLM will pull from HF on first run.

export CUDA_HOME=/opt/cuda
export PATH="/opt/cuda/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_RPC_TIMEOUT=600000
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# Disable TF32 at the driver level for IEEE-correct FP32 residual math
# (norms, softmax fallbacks). Matmul still runs in BF16/FP8/FP4 per dtype.
export NVIDIA_TF32_OVERRIDE=0

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
[ -d "/llms/archive/deepseek-v4-flash" ] && MODEL_PATH="/llms/archive/deepseek-v4-flash"

exec /home/lasi/vllm-env/bin/vllm serve "$MODEL_PATH" \
  --served-model-name "DeepSeek-V4-Flash" \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 14.81 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager \
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
