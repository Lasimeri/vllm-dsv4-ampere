#!/usr/bin/env bash
# Serve Intel/DeepSeek-V4-Flash-W4A16-AutoRound on 8x RTX 3080 (SM86) using the
# vllm-dsv4-ampere SM86 pyref/Triton patch set on vLLM base c2fb0133.
# Adapted from vllm-dsv4-ampere/wrapper-vllm-deepseek.sh (model + paths swapped).
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
# Include the venv bin so FlashInfer's runtime JIT can find `ninja`.
export PATH="/home/lasimeri/vllm-sm86/.venv/bin:${CUDA_HOME}/bin:$PATH"
# FlashInfer JIT must use clang (gcc-15.3 rejects torch 2.11 headers).
export CXX=clang++ CC=clang
export HF_HUB_DISABLE_XET=1                      # Xet CAS is broken on this host
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_RPC_TIMEOUT=600000
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NVIDIA_TF32_OVERRIDE=0
export VLLM_MXFP4_USE_MARLIN=1                   # harmless for W4A16; needed if any MXFP4
# SM86 pyref auto-enables for capability < (9,0); explicit for clarity:
export VLLM_SM86_DEEPSEEK_V4_REF=1
# K13 split-K fused FlashMLA sparse-decode kernel is ON by default (3.1x over
# the pyref at the TP=8 serve shape; +17% decode tok/s). Set =0 to fall back.
export VLLM_SM86_K13="${VLLM_SM86_K13:-1}"
# K12 bf16 fused paged-MQA logits (indexer) kernel, ON by default (4.4x over the
# pyref at H=64; +2% on top of K13). Warm-up at profiling time avoids the OOM;
# a failure-latch degrades cleanly to pyref. Set =0 to fall back.
export VLLM_SM86_K12="${VLLM_SM86_K12:-1}"
# FULL decode cudagraph capture (default ON): +28% decode tok/s (4.61 -> 5.89
# INT4; 4.77 -> 5.87 FP8). Needs two enablers, both default ON: single-stream
# attention (the cross-stream events block capture) and the compressor's
# capture-safe Triton dispatch for small/decode batches (the pyref .nonzero()
# is capture-illegal). Set VLLM_SM86_NO_MULTISTREAM=0 and/or CG_MODE=PIECEWISE
# to revert.
export VLLM_SM86_NO_MULTISTREAM="${VLLM_SM86_NO_MULTISTREAM:-1}"
export CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
# Sparse MLA prefill on SM86 is vectorized by default (batched bf16 tensor-core
# bmm; ~63x over the old per-token Python loop -> the TTFT win). No env needed.
# VLLM_SM86_PREFILL_LOOP=1 forces the slow per-token reference (parity fallback);
# VLLM_SM86_PREFILL_BLOCK_T tunes the query-tile size (default 512, caps the
# gather footprint).
export VLLM_TORCH_PROFILER_DIR=/home/lasimeri/vllm-sm86/prof

VLLM=/home/lasimeri/vllm-sm86/.venv/bin/vllm
MODEL="${MODEL_PATH:-Intel/DeepSeek-V4-Flash-W4A16-AutoRound}"

exec "$VLLM" serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation-config "{\"cudagraph_mode\": \"${CG_MODE:-PIECEWISE}\"}" \
  --max-num-seqs 1 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --override-generation-config '{"temperature": 1.0, "top_p": 1.0}' \
  --host 0.0.0.0 --port 8000 \
  "$@"
