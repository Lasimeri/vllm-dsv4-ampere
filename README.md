# vllm-dsv4-ampere

Patches that let **DeepSeek-V4-Flash** run on **Ampere SM 8.6** GPUs (RTX 30xx) under vLLM.

Status: **working with PIECEWISE cudagraph capture**. Generates coherent text, follows instructions, emits valid OpenAI-compatible tool calls. **~2.58 tok/s** sustained decode at short context on 8x RTX 3080 20GB (vLLM 0.1.dev15830+g8d599d76a, PIECEWISE compile + capture). Eager-mode fallback runs at ~2.01 tok/s.

This is a starting point for the community. Upstream vLLM rejects FP8/sparse-MLA/DeepGEMM support on SM<90 ("SM80 support better lives in a fork", per @youkaichao on PR #40906). This repo replaces every blocking kernel with either a pure-PyTorch reference path or a hand-written SM86 Triton kernel that engages BF16 tensor cores.

## What this is not

- Fast in absolute terms. ~2.58 tok/s, not 30. Long-context prefill is also slower than upstream because much of the path runs through Python-vectorized GPU code instead of fused Triton kernels.
- A merge candidate. Maintainers explicitly closed analogous PRs for SM80 (#40906) and reverted earlier sparse-MLA patches (#37968 reverted #35271).
- Stable across vllm versions. Patches were captured against vllm `0.1.dev15830+g8d599d76a` on 2026-04-29. Any meaningful upstream change to the touched files will need to be reconciled.

## What this is

- A working OpenAI-compatible endpoint for DeepSeek-V4-Flash on consumer Ampere.
- The full set of pyref kernel replacements needed to get the model past every SM86 architectural wall, plus three hand-tuned SM86 Triton kernels (K6, K7, K10-dequant) that recover ~16% throughput vs the pure-pyref baseline.
- A reference for anyone building a proper SM86 Triton fallback or porting to A100/SM80.

## Hardware tested

8x NVIDIA RTX 3080 20GB (Ampere SM 8.6, 160 GB total VRAM).

Other Ampere cards should work as long as the aggregate VRAM is enough for the 158 GB FP4+FP8 weights plus KV cache (we run with `--cpu-offload-gb 14.81`).

## Install

```bash
# 1. Install vllm at the matching version
#    pip install vllm==0.1.dev15830+g8d599d76a   (or compatible nightly)

# 2. Apply patches
./install.sh /path/to/your/vllm-env/lib/python3.X/site-packages/vllm

# 3. Edit wrapper-vllm-deepseek.sh: set MODEL_PATH and the vllm bin path
#    for your setup, then run it.
```

The pyref paths auto-activate when `torch.cuda.get_device_capability() < (9, 0)`. On Hopper+ they are inert no-ops.

Override:
- `VLLM_SM86_DEEPSEEK_V4_REF=1` -- force enable
- `VLLM_SM86_DEEPSEEK_V4_REF=0` -- force disable

Required envvar on SM86:
- `VLLM_MXFP4_USE_MARLIN=1` -- Marlin is the only MoE backend that supports `kMxfp4Static` below SM90 (DeepGEMM FP4 needs SM100+, TRTLLM needs SM90+). Wrapper sets this automatically.

## What's patched

12 vLLM Python files total. Categories:

```
[ Compute kernels (DeepGEMM-replaced) ]
  utils/deep_gemm.py
    tf32_hc_prenorm_gemm        - HC prenorm GEMM
    fp8_fp4_mqa_logits          - sparse indexer Q.K
    fp8_fp4_paged_mqa_logits    - paged variant
    fp8_einsum                  - generic FP8 einsum
    get_paged_mqa_logits_metadata - stub

[ TileLang fused ops ]
  model_executor/layers/mhc.py
    mhc_pre   (RMS+sinkhorn+pre/post mix)
    mhc_post  (post fused op)

[ Triton + pyref ops, all gated to SM<90 ]
  v1/attention/ops/deepseek_v4_ops/
    fused_inv_rope_fp8_quant.py    - K6 SM86 Triton + pyref
    fused_indexer_q.py             - K7 SM86 Triton + pyref
    cache_utils.py                 - K cache write/read
    fused_compress_quant_cache.py  - compressor (pyref active)
  model_executor/layers/deepseek_compressor.py - dispatch

[ FlashMLA C++ replacements ]
  third_party/flashmla/flash_mla_interface.py
    flash_mla_with_kvcache (sparse decode pyref + K10 Triton dequant)
    flash_mla_sparse_fwd   (sparse prefill pyref)

[ Triton dtype workarounds ]
  model_executor/layers/quantization/utils/fp8_utils.py
    w8a8_triton_block_scaled_mm   - dequant + BF16 GEMM
    _fp32_to_fp8_e4m3fn_byte      - byte-pack helper
    per_token_group_quant_fp8     - opaque-op wrapped
  model_executor/kernels/linear/scaled_mm/triton.py
    _w8a8_triton_block_scaled_mm_func - E8M0 upcast
  compilation/backends.py
    configure_post_pass - drop stale PostGradPassMgr

[ Compile-pass integration ]
  config/compilation.py
    _attention_ops: 9 SM86 op names added so PIECEWISE
                    splitting_ops auto-picks them
```

See `MANIFEST.md` for the full per-file changelog including version-to-version diffs.

## Hand-tuned SM86 Triton kernels

Three kernels were written by hand for SM86 (no native fp8e4nv, no TMA, no FP4) following haosdent's pattern from PR #40906 (Triton emits scaled fp32, eager casts to fp8 outside the compile region):

- **K6** -- inv-RoPE + fp8 quant -- 4.36x kernel-isolated
  `v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`
- **K7** -- indexer-Q RoPE + fp8 quant -- 7.38x kernel-isolated
  `v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`
- **K10-dequant** -- fp8 K-cache dequant inner kernel -- 7.01x kernel-isolated
  `third_party/flashmla/flash_mla_interface.py`

Aggregate E2E gain (eager): 1.67 -> 2.01 tok/s (+20%, accounting for cache-stable allocation reuse from the cudagraph fix).
With PIECEWISE capture: 1.67 -> 2.58 tok/s (+54%).

K8/K9 (compressor) Triton ports were attempted and reverted: bit-exact parity but 44% E2E regression because per-token launch overhead dominates when most tokens early-exit on the compress_ratio condition. The reference Triton code is preserved in-file but dispatch points back to the pyref.

A full K10 FlashAttention-2-style decode kernel is still future work; the current K10 only swaps the FP8 dequant inner kernel.

## PIECEWISE cudagraph fix

The repo ships PIECEWISE compile + capture as the default mode (configured in `wrapper-vllm-deepseek.sh`). Capture works because `per_token_group_quant_fp8_sm86` (and its packed-deepgemm sibling) now route their `(data, scale)` outputs through a shape-keyed persistent buffer cache.

Diagnosis was reached by replacing the gated debug assertion in `vllm/compilation/cuda_graph.py:341` with an always-on diagnostic that logged the runnable identity, batch descriptor, and shape of every input whose `data_ptr()` differed between capture and replay. A single 5-token Vichy probe surfaced 688 mismatched-input segments across 8 worker ranks, all with the same fingerprint: index 0 was a `(1, N) float8_e4m3fn` tensor and index 2 was a `(1, N/128) float32` tensor (groups of 128, one fp32 scale per group). That signature uniquely identifies the per-token-group quantizer's `(data, scale)` return tuple.

Each call to `per_token_group_quant_fp8_sm86` was allocating fresh tensors. PIECEWISE binds segment inputs to live `data_ptr()` recorded at capture time (see `vllm/compilation/cuda_graph.py:279-282`), and `cudagraph_copy_inputs=True` only handles tensors flagged as symbolic-shape (which static-shape decode tensors are not, see `vllm/compilation/backends.py:1274-1279`). So every replay read from the warmup-time address while the live tensor sat at a new address. The fix: a `_stable_buf_pair(data_ref, scale_ref)` helper keyed by `(shape, stride, dtype, device)` that reuses persistent buffers across calls, with a single `.copy_()` to populate them from the freshly-computed result. Patch lives in `patches/model_executor/layers/quantization/utils/fp8_utils.py` near the SM86 op registrations.

Eager-mode behavior is unaffected (the cache reuse is harmless when capture is off; net effect is a small bf16 memcpy per quant call).

## Known limitations

- **FP4 indexer cache path** raises `NotImplementedError`. Don't pass `--attention_config.use_fp4_indexer_cache=True`.
- **Q/K dim mismatch** (576 vs 512) handled via prefix dot product; trailing 64 q_pe rope dims dropped. Approximate but coherent in practice.
- **Long-context prefill** is bounded by per-call vectorized work, not per-token. Still slower than tuned kernels.
- **Marlin MoE** is the SM86 fallback for FP4 expert weights. Non-standard MXFP4 layouts will break it.

## Why this exists

DeepSeek released V4-Flash on 2026-04-24 with day-zero vLLM support targeted at Hopper/Blackwell. The model uses custom DeepGEMM kernels (HyperConnection, sparse MLA, FP8 paged MQA logits) that hard-assert on SM<90. The community-uploaded GGUF works in llama.cpp but llama.cpp lacks vLLM's tool-calling, structured output, and continuous batching, making it unsuitable for agentic workflows.

This repo is the result of methodically replacing each architectural wall with a pure-PyTorch reference until the model produces coherent output end-to-end on Ampere, then writing hand-tuned SM86 Triton kernels for the highest-leverage hot paths.

## License

Patches are derivative of vLLM (Apache 2.0). This repo follows the same license.

## Disclaimer

Use at your own risk. The patches aim for mathematical correctness (verified against working coherent generations and parity tests against pyrefs), but have not been validated against a Hopper reference for bit-exactness. Numerical drift is possible in edge cases.
