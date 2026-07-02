#!/usr/bin/env bash
# Launch DSv4-Flash on 8x RTX 3080 (SM86) ampere backend, 80/20 offload, eager.
# Fixes under test: indexer per-head ReLU + attn_sink post-hoc correction.
set -u
cd /home/lasimeri/vllm-dsv4-main
export PATH="/home/lasimeri/vllm-dsv4-main/.venv/bin:$HOME/.local/bin:/opt/cuda/bin:$PATH"
export CUDA_HOME=/opt/cuda
export CXX=clang++ CC=clang
export HF_HUB_DISABLE_XET=1
export VLLM_SM86_NAN_PROBE=${NAN_PROBE:-0}
# Default sink OFF: matches upstream xpu (sink-less BF16 kernel) and produced
# cleaner continuations than the post-hoc fold in A/B; set 1 to re-enable.
export VLLM_SM86_SINK=${VLLM_SM86_SINK:-0}
export VLLM_SM86_FP8_BF16=${VLLM_SM86_FP8_BF16:-0}
EP_FLAG="--enable-expert-parallel"
[ "${EXPERT_PARALLEL:-1}" = "0" ] && EP_FLAG=""
PC_FLAG=""
[ "${PREFIX_CACHE:-1}" = "0" ] && PC_FLAG="--no-enable-prefix-caching"
ASYNC_FLAG=""
[ "${ASYNC_SCHED:-1}" = "0" ] && ASYNC_FLAG="--no-async-scheduling"
# EAGER=1 restores --enforce-eager; default uses breakable cudagraphs
# (auto-enabled for DeepseekV4ForCausalLM via VLLM_USE_BREAKABLE_CUDAGRAPH).
# CG_MODE=FULL captures the whole decode step as one graph
# (FULL_DECODE_ONLY): requires the capture-safe SM86 decode path and
# disables the breakable wrapper.
EAGER_FLAG=""
[ "${EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"
# DCP=N enables decode context parallel (requires a2a comm backend).
DCP_FLAG=""
[ "${DCP:-1}" -gt 1 ] && DCP_FLAG="--decode-context-parallel-size ${DCP} --dcp-comm-backend a2a"
CC_FLAG=""
if [ "${CG_MODE:-BREAKABLE}" = "FULL" ]; then
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
  CC_FLAG='--compilation-config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi
exec .venv/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
  --tensor-parallel-size 8 $EP_FLAG $PC_FLAG $ASYNC_FLAG $DCP_FLAG --kv-cache-dtype fp8 --block-size 256 \
  --gpu-memory-utilization 0.88 --cpu-offload-gb 15 --max-num-seqs 1 \
  --max-model-len "${MAXLEN:-32768}" \
  --max-num-batched-tokens 2048 $EAGER_FLAG $CC_FLAG \
  --host 0.0.0.0 --port 8001
