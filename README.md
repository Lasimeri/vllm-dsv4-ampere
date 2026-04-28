# vllm-dsv4-ampere

Patches that let **DeepSeek-V4-Flash** run on **Ampere SM 8.6** GPUs (RTX 30xx) under vLLM.

Status: **working** — generates coherent text, follows instructions, emits valid OpenAI-compatible tool calls. ~1.6 tok/s decode at short context on 8x RTX 3080 20GB.

This is a starting point for the community. Upstream vLLM rejects the FP8/sparse-MLA/DeepGEMM walls on SM<90; this repo replaces every blocking kernel with a pure-PyTorch reference path that engages BF16 tensor cores wherever possible.

## What this is not

- Fast. Decode is ~1-2 tok/s, not 30. Long-context prefill is slower than upstream because it goes through Python-vectorized GPU code instead of fused Triton kernels.
- A merge candidate. The maintainers explicitly reverted this style of patch (PR #37968 reverted #35271). The proper upstream path is `TRITON_MLA_SPARSE`, which doesn't exist yet.
- Stable across vllm versions. These patches were captured against a specific vllm site-packages snapshot from 2026-04-27. Any meaningful upstream change to the touched files will need to be reconciled.

## What this is

- A working OpenAI-compatible endpoint for DeepSeek-V4-Flash on consumer Ampere.
- The full set of pyref kernel replacements needed to get the model past every SM86 architectural wall.
- A reference for anyone building a proper TRITON-based fallback.

## Hardware tested

8x NVIDIA RTX 3080 20GB (Ampere SM 8.6, 160 GB total VRAM)

Other Ampere cards should work as long as the aggregate VRAM is enough for the 158 GB FP4+FP8 weights plus KV cache.

## Install

```bash
# 1. Get a working vllm install (the version the patches were taken against)
#    See MANIFEST.md for the exact version notes.

# 2. Apply patches
./install.sh /path/to/your/vllm-env/lib/python3.X/site-packages/vllm

# 3. Edit wrapper-vllm-deepseek.sh: set MODEL_PATH and the vllm bin path
#    for your setup, then run it.
```

The patches auto-activate when `torch.cuda.get_device_capability() < (9, 0)`. On Hopper+ they're inert no-ops.

Override:
- `VLLM_SM86_DEEPSEEK_V4_REF=1` — force enable
- `VLLM_SM86_DEEPSEEK_V4_REF=0` — force disable

## What's patched

11 vLLM Python files, gated by a single `_use_sm86_reference()` check. Categories:

```
┌─ Compute kernels (DeepGEMM-replaced) ──────────────────┐
│ utils/deep_gemm.py                                     │
│   tf32_hc_prenorm_gemm   - HC prenorm GEMM             │
│   fp8_fp4_mqa_logits     - sparse indexer Q·K          │
│   fp8_fp4_paged_mqa_logits - paged variant             │
│   fp8_einsum             - generic FP8 einsum          │
│   get_paged_mqa_logits_metadata - stub                 │
└────────────────────────────────────────────────────────┘
┌─ TileLang fused ops ───────────────────────────────────┐
│ model_executor/layers/mhc.py                           │
│   mhc_pre   (RMS+sinkhorn+pre/post mix)                │
│   mhc_post  (post fused op)                            │
└────────────────────────────────────────────────────────┘
┌─ Per-token Triton kernels ─────────────────────────────┐
│ v1/attention/ops/deepseek_v4_ops/                      │
│   fused_inv_rope_fp8_quant.py - inv-RoPE + FP8 quant   │
│   fused_indexer_q.py          - fwd RoPE + FP8         │
│   cache_utils.py              - K cache write/read     │
│   fused_compress_quant_cache.py - compressor           │
│ model_executor/layers/deepseek_compressor.py - dispatch│
└────────────────────────────────────────────────────────┘
┌─ FlashMLA C++ replacements ────────────────────────────┐
│ third_party/flashmla/flash_mla_interface.py            │
│   flash_mla_with_kvcache (sparse decode)               │
│   flash_mla_sparse_fwd   (sparse prefill)              │
└────────────────────────────────────────────────────────┘
┌─ Triton dtype workarounds ─────────────────────────────┐
│ model_executor/layers/quantization/utils/fp8_utils.py  │
│   w8a8_triton_block_scaled_mm - dequant + BF16 GEMM    │
│ model_executor/kernels/linear/scaled_mm/triton.py      │
│   _w8a8_triton_block_scaled_mm_func - E8M0 upcast      │
│ compilation/backends.py                                │
│   configure_post_pass - drop stale PostGradPassMgr     │
└────────────────────────────────────────────────────────┘
```

See `MANIFEST.md` for full per-file patch summary.

## Known limitations

- **FP4 indexer cache path** raises `NotImplementedError`. Don't pass `--attention_config.use_fp4_indexer_cache=True`.
- **Q/K dim mismatch** (576 vs 512) handled via prefix dot product; trailing 64 q_pe rope dims dropped. Approximate but coherent in practice.
- **Long-context prefill** is bounded by per-call vectorized work, not per-token. Still slower than tuned kernels.
- **Marlin MoE** is the SM86 fallback for FP4 expert weights. Non-standard MXFP4 layouts will break it.

## Why this exists

DeepSeek released V4-Flash on 2026-04-24 with day-zero vLLM support targeted at Hopper/Blackwell. The model uses custom DeepGEMM kernels (HyperConnection, sparse MLA, FP8 paged MQA logits) that hard-assert on SM<90. The community-uploaded GGUF works in llama.cpp but llama.cpp lacks vLLM's tool-calling, structured output, and continuous batching — making it unsuitable for agentic workflows.

This repo is the result of methodically replacing each architectural wall with a pure-PyTorch reference until the model produces coherent output end-to-end on Ampere.

## License

Patches are derivative of vLLM (Apache 2.0). This repo follows the same license.

## Disclaimer

Use at your own risk. The patches aim for mathematical correctness (verified against working coherent generations), but have not been validated against a Hopper reference for bit-exactness. Numerical drift is possible in edge cases.
