DeepSeek-V4-Flash on Ampere SM 8.6 — vectorized vLLM patch snapshot
====================================================================

Snapshot taken: 2026-04-27 ~21:24 CDT
Working state:  generates coherent text, follows instructions,
                emits valid tool_calls, ~1.64 tok/s at short context.
Hardware:       8x RTX 3080 20GB (Ampere SM 8.6, 160 GB total VRAM)
Model:          deepseek-ai/DeepSeek-V4-Flash (FP4 + FP8 Mixed)

Throughput:     ~1.9x faster than the initial coherent build
                (0.88 → 1.64 tok/s on the vichi prompt).

Diff vs vllm-deepseek-v4-sm86-working-2026-04-27 (the prior backup):
  • All FP32 → BF16 wins: mqa_logits, paged_mqa_logits, mhc_post,
    dequant_kv_slots
  • cache_utils pyrefs: per-token Python loops → batched scatter/gather
  • compressor pyrefs (head=512 sparse + head=128 indexer): same
  • flashmla _dequant_fp8_kv_slots: per-N loop → batched gather
  • _tf32_hc_prenorm_gemm_pyref: full GEMM in split 0 (handles
    non-divisible num_split)

Plus all the patches from the earlier "working" snapshot (FlashMLA
sparse decode/prefill pyrefs, compressor pyrefs, fused_inv_rope_fp8,
fused_indexer_q_rope_quant, fp8_einsum, w8a8 block-scaled MM, E8M0
upcasts, etc.).

How to restore
--------------
cp -a vllm-pkg/. /home/lasi/vllm-env/lib/python3.12/site-packages/vllm/
cp wrapper-vllm-deepseek.sh /home/lasi/bin/vllm-deepseek
chmod +x /home/lasi/bin/vllm-deepseek

Envvar override
---------------
VLLM_SM86_DEEPSEEK_V4_REF=0  → disable all pyrefs
VLLM_SM86_DEEPSEEK_V4_REF=1  → force pyrefs on SM>=90 (testing)
Default: auto-enabled when device capability < (9,0).

Patches will be wiped on any vllm-env pip reinstall/upgrade.
