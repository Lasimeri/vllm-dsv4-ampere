# DeepSeek-V4-Flash Ampere (SM8x) backend + DCP — port onto vLLM main

Goal: run DeepSeek-V4-Flash sparse MLA on 8x RTX 3080 (SM86) against **current
main** (5c4db60f), with **DCP** working, by adding an `ampere/` backend to
`vllm/models/deepseek_v4/` (mirroring `xpu/`). Then stay synced with main.

Worktree: /home/lasimeri/vllm-dsv4-main (branch `ampere-dsv4-dcp`).
Source kernels to port: the working Triton SM86 set on branch `sm86-snapshot`
in ~/vllm-sm86 (K12 indexer logits, K13 split-K sparse MLA decode, compressor
capture-safe dispatch, fp8 einsum, fused inv-rope). All Python/Triton — no
custom .cu, so build is `VLLM_USE_PRECOMPILED=1` (fast).

## Architecture (mapped)
- Registry: `DeepseekV4ForCausalLM` -> `vllm.models.deepseek_v4`. `__init__.py`
  dispatches by platform (rocm/xpu/else=nvidia). ADD: CUDA cap < (9,0) -> `ampere/`.
- Backend provides: `model.py`, `mtp.py`, an Attention class, and a sparse
  decode kernel. `xpu/` template: `xpu_sparse.py` (attention) +
  `xpu_sparse_decode_fp8.py` (dequant fp8->bf16 + `triton_bf16_mla_sparse_interface`).
- Shared (reuse as-is): `common/ops/*` (cache_utils, fused_compress_quant_cache,
  fused_indexer_q, fused_inv_rope_fp8_quant), `attention.py`, `sparse_mla.py`,
  `compressor.py`, `quant_config.py`.
- DCP: handled in framework MLA backend (`vllm/v1/attention/backends/mla/`,
  per PR #45426 for flashmla sparse). Our sparse decode kernel must return
  softmax-LSE so the framework does the cross-rank merge; indexer builds
  metadata from `dcp_local_seq_lens`. Plus Workstream A (hybrid multi-block
  scheduler glue) for V4-Flash's C4/C128 layout.

## Workstreams
- [ ] W1: `ampere/` scaffold + `__init__.py` dispatch (cap<9.0).
- [ ] W2: port model.py (from nvidia/model.py, swap cutedsl->Triton kernels).
- [ ] W3: port mtp.py.
- [ ] W4: ampere sparse decode (K13) + indexer logits (K12) as backend ops,
      adapted to main's common/ layout + kernel interfaces.
- [ ] W5: DCP — LSE return from K13 + indexer dcp_local metadata (adopt #45426
      pattern) + Workstream A hybrid scheduler.
- [ ] W6: build (VLLM_USE_PRECOMPILED=1, separate .venv) + load + coherence.
- [ ] W7: DCP coherence A/B (dcp=1 vs dcp=8) + long-context capacity.

## Status log
- Worktree + study done. Mapping nvidia/xpu/framework surfaces next.
