#!/usr/bin/env bash
# MTP speculative-decode variant of serve_sm86.sh. DeepSeek-V4-Flash ships a
# 1-layer MTP head (num_nextn_predict_layers=1, mtp.0.* int4 weights); the draft
# reuses the same SM86-patched MLA/indexer kernels. num_speculative_tokens
# overridable via SPEC_TOK (default 1).
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export PATH="/home/lasimeri/vllm-sm86/.venv/bin:${CUDA_HOME}/bin:$PATH"
export CXX=clang++ CC=clang
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_RPC_TIMEOUT=600000
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NVIDIA_TF32_OVERRIDE=0
export VLLM_MXFP4_USE_MARLIN=1
export VLLM_SM86_DEEPSEEK_V4_REF=1
export VLLM_SM86_K13="${VLLM_SM86_K13:-1}"
export VLLM_SM86_K12="${VLLM_SM86_K12:-1}"

VLLM=/home/lasimeri/vllm-sm86/.venv/bin/vllm
MODEL="${MODEL_PATH:-Intel/DeepSeek-V4-Flash-W4A16-AutoRound}"
SPEC_TOK="${SPEC_TOK:-1}"
# GPU mem util is lower than the base script: the MTP head weights + draft KV
# cache need headroom on top of the already-tight 0.95.
GPU_UTIL="${GPU_UTIL:-0.90}"

exec "$VLLM" serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --cpu-offload-gb 8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation-config "{\"cudagraph_mode\": \"${CG_MODE:-PIECEWISE}\"}" \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOK}}" \
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
