# vllm-dsv4-ampere

Patches that let **DeepSeek-V4-Flash** run on **Ampere SM 8.6** GPUs (RTX 30xx) under vLLM.

Status: **working with PIECEWISE cudagraph capture**. Generates coherent text, follows instructions, emits valid OpenAI-compatible tool calls. Decode at short context on 8x RTX 3080 20GB: **4.61 tok/s** on `int4-autoround-support` (Intel INT4 AutoRound + K12 + K13), **2.6 tok/s** on `master` (deepseek-ai FP4+FP8). See the [Branches](#branches) table for the per-branch breakdown. Eager-mode fallback on `master` runs at ~2.01 tok/s.

This is a starting point for the community. Upstream vLLM rejects FP8/sparse-MLA/DeepGEMM support on SM<90 ("SM80 support better lives in a fork", per @youkaichao on PR #40906). This repo replaces every blocking kernel with either a pure-PyTorch reference path or a hand-written SM86 Triton kernel that engages BF16 tensor cores.

## Branches

All branches target 8x RTX 3080 20GB (SM 8.6), TP=8, PIECEWISE cudagraph capture. Pick by which checkpoint and which vLLM base you have.

| Branch | Checkpoint | vLLM base | Decode tok/s (short ctx) | Notes |
| --- | --- | --- | --- | --- |
| `master` | `deepseek-ai/DeepSeek-V4-Flash` (FP4 + FP8) | main @ 2026-04-27 (PR #40860), pinned `0.1.dev15830+g8d599d76a` | **2.6** PIECEWISE / 2.01 eager | Original SM86 patch set: pyref + K6/K7/K10 Triton kernels. |
| `compat/vllm-0.20.1` | `deepseek-ai/DeepSeek-V4-Flash` | `vllm==0.20.1` (PyPI) | not yet E2E-validated | Patches apply cleanly + all `torch.ops.vllm.*` ops register; full inference unverified (see issue #3). |
| `int4-autoround-support` | `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` (INT4 W4A16) | main @ `c2fb0133` (2026-04-30), clang build | **4.61** PIECEWISE (3.84 baseline) | INT4 AutoRound model: routed experts run the hardware Marlin int4 kernel; adds the K13 split-K FlashMLA decode (+17%) and the K12 bf16 indexer kernel (+2%). This is the current development branch. |

The **int4-autoround-support** branch is the fastest because (a) the AutoRound experts use hardware Marlin int4 (no software dequant) and (b) the K13 split-K FlashMLA decode + K12 bf16 indexer kernels fill the SMs at the TP=8 decode shape (3.84 -> 4.51 -> 4.61, +20% total). The remaining decode floor on every branch is the sparse-MLA/indexer dequant + softmax, which has no native Ampere kernel.

> **Note on the vLLM base commit:** the repo above pins `0.1.dev15830+g8d599d76a` (commit `8d599d76a`) which was an intermediate hash on the upstream PR #40860 ("DeepSeek V4 Rebased") staging tree. That commit was squash-merged into vllm-project/vllm `main` on 2026-04-27 and the original hash garbage-collected, so it no longer resolves on github.com/vllm-project/vllm. **Functionally equivalent state**: any vllm-project/vllm `main` checkout from 2026-04-27 onwards (or the v0.20.x patch line) is structurally compatible. Apply the patches in `patches/` on top.

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
# 1. Install vllm-project/vllm at any main-branch commit from
#    2026-04-27 onwards (after PR #40860 "DeepSeek V4 Rebased" merged).
#    Building from source is recommended since the dev wheel for the
#    exact base commit is no longer on PyPI.
#      git clone https://github.com/vllm-project/vllm
#      cd vllm && git checkout <main-after-2026-04-27>
#      pip install -e . --no-build-isolation
#    A v0.20.x patch-line release works equivalently.

# 2. Apply patches (overlays the SM86 pyref + Triton replacements)
git clone https://github.com/Lasimeri/vllm-dsv4-ampere
cd vllm-dsv4-ampere
./install.sh /path/to/your/vllm-env/lib/python3.X/site-packages/vllm

# 3. Edit wrapper-vllm-deepseek.sh: set VLLM_BIN (or activate the
#    venv) and MODEL_PATH for your setup, then run it.
cp wrapper-vllm-deepseek.sh ~/bin/vllm-deepseek
chmod +x ~/bin/vllm-deepseek
~/bin/vllm-deepseek
```

The pyref paths auto-activate when `torch.cuda.get_device_capability() < (9, 0)`. On Hopper+ they are inert no-ops.

### Env-var overrides

| Var | Default | Effect |
| --- | --- | --- |
| `VLLM_SM86_DEEPSEEK_V4_REF` | auto | `1` force enable on Hopper+ (testing); `0` force disable on Ampere |
| `VLLM_SM86_K11` | `0` | `1` enable OLD single-program fused FlashMLA decode (superseded by K13; single-program grid uses 1-4 SMs and loses to cuBLAS -- kept for reference) |
| `VLLM_SM86_K12` | `1` | `0` disable the fused bf16 paged-MQA logits (indexer) kernel. bf16 tensor-core dot (fp32 accumulate, fp32 dequant scale): 4.40x over the pyref at the real H=64 serve shape, and marginally MORE faithful than the pyref (top-512 overlap vs the fp32-true ranking >= pyref). +2% E2E on top of K13 (4.51 -> 4.61). A profiling-time warm-up loads the CUDA module before the KV cache packs the GPU (fixes the prior first-call OOM), and a failure-latch degrades cleanly to the pyref. ON by default. |
| `VLLM_SM86_K13` | `1` | `0` disable the split-K (FlashDecoding) FlashMLA decode kernel. K13 is the realized split-K rework of K11: partial-state kernel over (B, H-tile, K-split) + log-sum-exp combine, filling the SMs. **3.1x over pyref at the TP=8 serve shape (460->148 us/call), +17% E2E decode (3.84 -> 4.51 tok/s)**, bf16 ULP parity. ON by default. |
| `VLLM_MXFP4_USE_MARLIN` | required `1` | Wrapper sets this. Marlin is the only MoE backend that supports `kMxfp4Static` below SM90 (DeepGEMM FP4 needs SM100+, TRTLLM needs SM90+) |

## What's patched

14 vLLM Python files total (12 modified upstream files + 2 new SM86-only files). Categories:

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
    flash_mla_with_kvcache (sparse decode pyref + K10 Triton dequant
                            + optional K11 fused decode dispatch)
    flash_mla_sparse_fwd   (sparse prefill pyref)
  third_party/flashmla/flash_mla_decode_sm86.py    [NEW, SM86-only]
    K11 fused FlashMLA decode -- experimental, default off

[ Standalone SM86 kernels ]
  utils/fp8_paged_mqa_logits_sm86.py               [NEW, SM86-only]
    K12 fused paged-MQA logits -- experimental, default off

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

Six kernels written by hand for SM86 (no native fp8e4nv, no TMA, no FP4). The shipped five: K6/K7/K10/K12 follow haosdent's pattern from PR #40906 (Triton emits bf16/scaled-fp32, casts outside the compile region); K13 is a FlashDecoding split-K attention kernel:

| Kernel | Path | Status | Speedup |
| --- | --- | --- | --- |
| **K6** -- inv-RoPE + fp8 quant | `v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` | shipped, on by default | 4.36x kernel-isolated |
| **K7** -- indexer-Q RoPE + fp8 quant | `v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` | shipped, on by default | 7.38x kernel-isolated |
| **K10-dequant** -- fp8 K-cache dequant inner | `third_party/flashmla/flash_mla_interface.py` | shipped, on by default | 7.01x kernel-isolated |
| **K11** -- full fused FlashMLA decode (single-program) | `third_party/flashmla/flash_mla_decode_sm86.py` | superseded by K13, default OFF | parity correct (max diff 1e-3); E2E -71% because grid is (1, B*H_block) so only 1-4 SMs run while cuBLAS uses all 68. Realized as K13. `VLLM_SM86_K11=1` to try the old path. |
| **K12** -- fused bf16 paged-MQA logits | `utils/fp8_paged_mqa_logits_sm86.py` | **shipped, ON by default** | bf16 tensor-core dot, fp32 accumulate, fp32 dequant scale. 4.40x over the pyref at the real H=64 serve shape (72 vs 319 us/call), top-512 overlap with the fp32-true ranking >= pyref. +2% E2E on top of K13 (4.51 -> 4.61). `warmup_k12()` at profiling time loads the module before the cache packs the GPU (fixes the first-call OOM); a failure-latch degrades cleanly to pyref. `VLLM_SM86_K12=0` to disable. |
| **K13** -- split-K FlashMLA decode | `third_party/flashmla/flash_mla_decode_sm86.py` (`_flash_mla_decode_sm86_splitk`) | **shipped, ON by default** | The split-K rework K11 needed. Partial kernel grid (B, H-tile, K-split) runs online-softmax over K-slices into scratch; combine kernel merges via log-sum-exp. Fills the SMs (16 programs vs K11's 1-4 at the TP=8 serve shape). **3.1x over pyref (460->148 us/call), +17% E2E (3.84 -> 4.51 tok/s)**, bf16 ULP parity. `VLLM_SM86_K13=0` to disable. |

Aggregate E2E gain from K6 + K7 + K10 (eager): 1.67 -> 2.01 tok/s (+20%, accounting for cache-stable allocation reuse from the cudagraph fix).
With PIECEWISE capture: 1.67 -> 2.6 tok/s (+55%).

### Why K11 / K12 are experimental

- **K11 -> K13 (resolved)**: cuBLAS is heavily tuned for the small-GEMM shapes that DSv4's sparse decode produces, and uses all 68 SMs via internal split-K. A single fused Triton program (K11) at decode shape uses 1-4 SMs and loses despite saving global-memory traffic. The fix shipped as **K13**: FlashDecoding-style 2-kernel split-K (partial-state scratch over (B, H-tile, K-split) + log-sum-exp reduction). At the TP=8 serve shape the partial grid is 16 programs (vs 1-4), and K13 runs 3.1x faster than the pyref and +17% E2E. Parity harness: `test_k13.py`.
- **K12 (resolved, now shipped ON)**: three issues, three fixes. (1) The original bf16 kernel reshuffled the order-sensitive top-K via bf16 reduction noise -- fixed by keeping the dequant scale in fp32 until the bf16 cast, which makes the kernel's top-512 overlap with the fp32-true ranking >= the pyref's (the kernel is now the *more* faithful path, not the noisy one). (2) A true-fp32 dot was tried for max fidelity but is ~0.5x the pyref at the real H=64 serve shape because SM86 has no fp32 tensor-core path -- reverted to a bf16 tensor-core dot (fp32 accumulate), which is 4.40x the pyref. (3) JIT-loading the module at gpu-mem-util 0.95 OOM'd on the first decode call -- fixed by `warmup_k12()`, which forces the compile+module-load during the profiling forward (KV cache not yet allocated = headroom); a failure-latch additionally degrades any launch error to a clean pyref run instead of a 5x per-call-exception storm. Net +2% E2E on top of K13.

K8/K9 (compressor) Triton ports were attempted and reverted: bit-exact parity but 44% E2E regression because per-token launch overhead dominates when most tokens early-exit on the compress_ratio condition. The reference Triton code is preserved in-file but dispatch points back to the pyref.

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

## INT4 / AutoRound (W4A16) variant

The original patch set targets `deepseek-ai/DeepSeek-V4-Flash` (FP4 experts + FP8
+ BF16). These patches **also** run the INT4 quant, `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`
(~145 GiB; experts int4 via `auto_round:auto_gptq`, plus selective bf16 layers).
Most changes are backward-compatible (they branch on checkpoint structure), so the
same tree serves both checkpoints.

**Base commit:** captured/validated against vLLM `c2fb0133` (2026-04-30) — 1 day
past the original snapshot, so a few base-drift fixes are included below.

**Key finding:** the int4 experts run the **hardware Marlin kernel**
(`Using 'MARLIN' WNA16 MoE backend`) — weight dequant is *not* the bottleneck.
The decode floor is the sparse-MLA + indexer pyref (V4's DSA has no Ampere kernel).
Measured **~3.84 tok/s** on 8x RTX 3080 20GB (PIECEWISE, `--cpu-offload-gb 8`,
ctx 32k), coherent generation + valid tool calls.

### Build note (gcc >= 15)

vLLM's pre-stable-ABI `_C` target pulls torch 2.11's full C++ headers, which
**gcc-15.3 rejects** (`List_inl.h` two-phase template lookup). Build the base vLLM
with **clang** as the CUDA host compiler — clang accepts the headers and CUDA 13.x
accepts clang (GNU libstdc++, ABI-compatible with the gcc-built torch):

```bash
CC=clang CXX=clang++ NVCC_CCBIN=clang++ TORCH_CUDA_ARCH_LIST=8.6 \
  uv pip install -e . --no-build-isolation
```

### Additional patches (on top of the base set)

New:
- `model_executor/models/deepseek_v4.py` — default `scale_fmt` to `ue8m0` when the
  checkpoint's `quantization_config` omits it (AutoRound does).
- `model_executor/layers/sparse_attn_indexer.py` — let the DeepGEMM guard pass when
  `_use_sm86_reference()` is active.
- `model_executor/layers/deepseek_v4_attention.py` — o-proj branches to a bf16 path
  (opaque op `deepseek_v4_o_proj_bf16_sm86`, registered in `_attention_ops`) when
  `wo_a` is bf16 (AutoRound) instead of fp8.
- `v1/attention/ops/flashmla.py` — route FlashMLA entry points to the SM86 pyref
  interface when `_flashmla_C` is absent (instead of stubbing to `_raise`).

Updated:
- `third_party/flashmla/flash_mla_interface.py` — guard the unconditional
  `import vllm._flashmla_C` (not built on SM86; pyref paths don't use it).
- `model_executor/layers/deepseek_compressor.py` — use the precomputed gate output
  (c2fb0133 moved the gate GEMM into `attn_gemm_parallel_execute`; the snapshot
  version re-applied it -> shape error).
- `config/compilation.py` — add `PassConfig.fuse_mla_dual_rms_norm`,
  `CompilationConfig.encoder_compilation_time` (base-drift), and the new o-proj op
  to `_attention_ops`.
- `compilation/backends.py` — accept the `is_encoder` kwarg (base-drift).

### Runtime dependency fix (not a vLLM patch)

`prometheus_fastapi_instrumentator` 500s every request on this vLLM
(`_IncludedRouter` has no `.path`). One-line fix in its
`routing.py` `_get_route_name` (2 sites): `route_name = getattr(route, "path", None)`.

Run with `wrapper-vllm-deepseek-int4.sh` (set `VLLM_BIN` / `MODEL_PATH`).

## License

Patches are derivative of vLLM (Apache 2.0). This repo follows the same license.

## Disclaimer

Use at your own risk. The patches aim for mathematical correctness (verified against working coherent generations and parity tests against pyrefs), but have not been validated against a Hopper reference for bit-exactness. Numerical drift is possible in edge cases.
