DeepSeek-V4-Flash on Ampere SM 8.6, vLLM patch snapshot
========================================================

Snapshot taken: 2026-04-29 ~16:00 CDT
Working state:  generates coherent text, follows instructions,
                emits valid tool_calls, ~1.94 tok/s sustained at
                short context (Vichy prompt, finish=stop).
Hardware:       8x RTX 3080 20GB (Ampere SM 8.6, 160 GB total VRAM)
Model:          deepseek-ai/DeepSeek-V4-Flash (FP4 + FP8 + BF16 mixed)
vLLM base:      0.1.dev15830+g8d599d76a (commit 8d599d76a)

Throughput history
------------------
  0.88 tok/s   initial coherent build (2026-04-25)
  1.64 tok/s   prior snapshot (2026-04-27): all FP32 to BF16 wins,
               per-token Python loops vectorized, batched scatter/gather
  1.67 tok/s   pyref-only baseline (2026-04-29 morning)
  1.94 tok/s   K6 + K7 + K10-dequant Triton kernels (this snapshot)

Diff vs the 2026-04-27 patch set
--------------------------------
* K6 `_fused_inv_rope_fp8_quant_sm86_triton`
  (v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py):
    haosdent pattern: Triton emits scaled fp32, eager `.to(fp8e4m3fn)`.
    4.36x kernel-isolated, bit-exact parity vs pyref.
* K7 `_fused_indexer_q_rope_quant_sm86_triton`
  (v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py):
    Same template as K6. 7.38x.
* K10 (partial) `_dequant_fp8_kv_slot_kernel`
  (third_party/flashmla/flash_mla_interface.py):
    Replaces just the K-cache dequant inner kernel inside
    `_flash_mla_decode_pyref` (NOT the full FlashAttention rewrite).
    7.01x kernel-isolated, ~8% Python time saved per profile.
    Critical fix during dev: cache shape (12330, 64, 1, 584) has
    block stride 37440 (64 bytes padding per block). Pass
    `cache.stride(0)` directly; do not infer from `.shape[-1]`.
* Eight opaque-op wrappers around the existing SM86 pyrefs +
  Triton kernels via `direct_register_custom_op`, with proper
  `mutates_args` and `fake_impl`. Enables the compile path to
  treat them as stable boundaries:
    - vllm::deepseek_v4_quant_insert_k_cache_sm86
    - vllm::deepseek_v4_dequant_gather_k_cache_sm86
    - vllm::deepseek_v4_compress_sparse_attn_sm86
    - vllm::deepseek_v4_compress_indexer_attn_sm86
    - vllm::deepseek_v4_fused_inv_rope_fp8_quant_sm86
    - vllm::deepseek_v4_fused_indexer_q_rope_quant_sm86
    - vllm::per_token_group_quant_fp8_sm86
    - vllm::per_token_group_quant_fp8_packed_deepgemm_sm86
* `config/compilation.py`: 9 op names added to `_attention_ops`
  (the 8 above plus `vllm::deepseek_v4_fp8_einsum`) so vLLM's
  splitting_ops list auto-includes them when PIECEWISE compile
  is enabled.
* `_fp32_to_fp8_e4m3fn_byte` helper added in
  `model_executor/layers/quantization/utils/fp8_utils.py`
  (alongside the existing `_fp8_e4m3fn_byte_to_bf16`). Validated
  bit-exact for 254/254 valid bytes; 99.91% on random fp32.

Plus all the patches from the prior snapshots: FlashMLA sparse
decode/prefill pyrefs, compressor pyrefs, fp8_einsum, w8a8
block-scaled MM, E8M0 upcasts, Marlin MoE for MXFP4 experts,
HC prenorm GEMM, MHC TileLang replacements.

cudagraph status
----------------
PIECEWISE cudagraph capture succeeds and reaches ~2.57 tok/s
(+33% over the eager 1.94), but produces deterministic garbage
output ("Vr 1, ... Chinese tokens repeating"). Token 0 (prefill)
is correct; token 1 onward (first replayed decode step) is wrong
in a deterministic way regardless of K10 Triton vs pyref, prefix
caching, async scheduling, and copy_inputs flags.

Active investigation suspects fresh-allocation-inside-forward in
`v1/attention/backends/mla/flashmla_sparse.py:870`
(`attn_out = q.new_empty(...)` produces a different address every
call, breaking the captured graph's recorded read pointers).
This is NOT YET FIXED in this snapshot. Wrapper ships
`--enforce-eager` as the working state.

How to restore
--------------
./install.sh /path/to/your/vllm-env/lib/python3.X/site-packages/vllm
cp wrapper-vllm-deepseek.sh /home/lasi/bin/vllm-deepseek
chmod +x /home/lasi/bin/vllm-deepseek

Envvar override
---------------
VLLM_SM86_DEEPSEEK_V4_REF=0  -> disable all pyrefs
VLLM_SM86_DEEPSEEK_V4_REF=1  -> force pyrefs on SM>=90 (testing)
Default: auto-enabled when device capability < (9, 0).

VLLM_MXFP4_USE_MARLIN=1      -> required on SM86; only Marlin
                                supports kMxfp4Static below SM90.

Patches will be wiped on any vllm-env pip reinstall/upgrade.
