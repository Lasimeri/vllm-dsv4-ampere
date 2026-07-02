# vllm-dsv4-ampere

**DeepSeek-V4-Flash on Ampere (SM 8.6, RTX 30xx) under vLLM.**

Upstream vLLM does not support FP8 / sparse-MLA / DeepGEMM on SM < 90 and has
declined SM80-class patches ("SM80 support better lives in a fork", PR #40906).
This repo is that fork's patch set: every Hopper-only kernel on the
DeepSeek-V4-Flash path (cutedsl, DeepGEMM, FlashMLA-sparse, TileLang MHC,
native fp8 casts) replaced with an SM86-compatible Triton kernel or a
PyTorch reference implementation.

## Status

Current flagship is **`main-port/`**: a vendor-style `ampere/` hardware
backend for **current vLLM main** (`vllm/models/deepseek_v4/ampere/`,
0.23.x line), following the same layout as the upstream `nvidia/`, `amd/`
and `xpu/` backends.

Verified on 8x RTX 3080 20GB, TP=8, the full `deepseek-ai/DeepSeek-V4-Flash`
FP4+FP8 checkpoint, 80/20 CPU/GPU weight offload (`--cpu-offload-gb 15`),
32k context:

- **Coherent, deterministic output** (greedy decoding is bit-identical across
  requests), OpenAI-compatible completions + chat endpoints.
- **Breakable CUDA graphs enabled** (upstream's PIECEWISE-equivalent path for
  models without `torch.compile` support). The decode path is fully
  capture-safe: no host syncs, no per-token Python loops.
- **Decode ~4.4 tok/s** at short context under this deliberately
  offload-heavy config (weights stream over PCIe via UVA every step).
  KV pool: 424,060 tokens.

| Configuration (main-port, 8x3080, 80/20 offload) | Decode tok/s |
| --- | --- |
| eager, host syncs in decode path | 2.3–2.7 |
| eager, sync-free decode | 4.0 |
| breakable cudagraph (current default) | **4.3–4.4** |

## Repo layout

| Path | What |
| --- | --- |
| `main-port/vllm/` | Overlay for **current vLLM main**: the `ampere/` backend + supporting edits. |
| `main-port/launch_e4.sh` | Reference launch script with all env toggles. |
| `patches/`, `install.sh`, `serve_*.sh`, `wrapper-*.sh` | **Legacy**: the original patch set against vLLM main @ 2026-04 (`c2fb0133` era). Inventory in `MANIFEST.md`; kernel-by-kernel docs (K6–K13, vectorized prefill) in git history. |
| `test_*.py` | Parity harnesses for the legacy kernels. |
| `DCP-PLAN.md`, `DCP-FINDINGS.md` | Decode-context-parallel investigation notes. |

Legacy branches (older vLLM bases, kept for reference):

| Branch | Checkpoint | Decode tok/s | Notes |
| --- | --- | --- | --- |
| `int4-autoround-support` | Intel W4A16 AutoRound | 5.89 (FULL capture) | Hardware Marlin int4 experts + K12/K13 SM86 kernels + vectorized prefill (TTFT 6.9–10.2x); vLLM main @ `c2fb0133`. |
| `master` (pre-main-port history) | FP4+FP8 | 2.6 (PIECEWISE) | Original pyref patch set; vLLM main @ 2026-04-27. |
| `compat/vllm-0.20.1` | FP4+FP8 | unvalidated | Patches apply; E2E unverified. |

## Quickstart (main-port)

```bash
# 1. vLLM main (0.23.x line), Python-only build
git clone https://github.com/vllm-project/vllm && cd vllm
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 2. Overlay the ampere backend
cp -r /path/to/vllm-dsv4-ampere/main-port/vllm/ ./

# 3. Serve (TP8, fp8 KV, 80/20 offload, breakable cudagraphs)
bash /path/to/vllm-dsv4-ampere/main-port/launch_e4.sh
```

Env toggles understood by the launch script / backend:

| Var | Default | Effect |
| --- | --- | --- |
| `EAGER=1` | off | restore `--enforce-eager` (disables cudagraphs) |
| `EXPERT_PARALLEL=0` | on | disable expert parallelism |
| `PREFIX_CACHE=0` | on | disable prefix caching |
| `ASYNC_SCHED=0` | on | disable async scheduling |
| `VLLM_SM86_SINK=1` | off | enable the post-hoc attention-sink fold |
| `VLLM_SM86_NAN_PROBE=1` | off | per-layer numerics probes (debug) |
| `VLLM_SM86_IDX_PYREF=1` | off | torch reference for indexer decode logits (debug) |
| `VLLM_SM86_KINS_TORCH=1` | off | torch reference for the K-cache insert (debug) |
| `VLLM_SM86_INDEXER_TILE_ELEMS` | 24000000 | element budget for the indexer prefill logits tile (bounds the `[M,H,N]` transient at long prefills) |

## What the ampere backend replaces

| Hopper-only component | SM86 replacement |
| --- | --- |
| cutedsl compress/store, fused indexer-Q | Triton store + torch RoPE/quant reference |
| DeepGEMM `fp8_paged_mqa_logits` (indexer decode) | Triton paged MQA-logits kernel: manual e4m3 decode, per-head ReLU, static grid with in-kernel early exit (capture-safe) |
| DeepGEMM `fp8_mqa_logits` (indexer prefill) | bf16 torch reference, tiled over query rows |
| FlashMLA sparse decode/prefill | fp8-dequant gather + portable BF16 sparse-MLA Triton kernel |
| Triton `fp8e4nv` casts (unsupported on SM8x) | hand-rolled e4m3fn encode/decode bit manipulation (verified bit-exact vs torch) |
| TileLang MHC (hyperconnection) kernels | torch/triton fallbacks (TileLang assumes Hopper-class features) |
| fp8 block-quant `wo_a` consumed by the o-proj einsum | Marlin repack bypass keeps the raw block-fp8 weight |
| Trained per-head attention sink (BF16 kernel has no sink slot) | optional post-hoc LSE fold (`VLLM_SM86_SINK=1`) |

## Two bugs worth knowing about (if you port this elsewhere)

1. **`.reshape(-1)` on a KV-cache view silently drops writes.** Per-layer KV
   caches are non-contiguous slices of vLLM's unified pool (measured
   `stride(0)=1,039,680` vs 37,376 row bytes). `reshape` falls back to a
   copy, so a scatter through it never reaches the cache — the model then
   attends over unwritten pool memory: coherent-looking garbage or NaN
   depending on physical block placement, non-deterministic at temperature 0.
   Write through strided advanced indexing, or use raw-pointer Triton kernels
   with `tensor.stride(0)`.

2. **`-inf` sentinels NaN-poison online softmax on fully-masked tiles.**
   If a BLOCK_N tile of sparse indices is entirely `-1` before any valid
   entry has been seen, `exp2(-inf - -inf) = NaN` in the rescale. Fixed-width
   index layouts (empty top-k region at short context) hit this on every
   leading tile. Use a large finite sentinel (`-1e30`); bogus partial sums
   are zeroed by the first valid tile's rescale.

## Known limits / roadmap

- Prefill on main-port is slow (Python-vectorized gather paths); decode was
  the priority. The legacy branch's vectorized-prefill treatment (~10x TTFT)
  has not been re-ported yet.
- Remaining decode levers: MTP speculative decoding (head ships with the
  checkpoint; `ampere/mtp.py` is ported but untested), fused fp8
  sparse-decode attention (skip the bf16 workspace round-trip), offload
  placement policy.
- DCP (decode context parallel) for expanded context: notes in
  `DCP-PLAN.md` / `DCP-FINDINGS.md`; blocked upstream on compression+DCP
  (`indexer.py` forbids `dcp>1` with `compress_ratio>1`).
- Not a merge candidate for upstream vLLM (SM < 90 explicitly out of scope);
  maintained here as a community fork.

## Hardware

Tested: 8x NVIDIA RTX 3080 20GB (SM 8.6). Other Ampere cards should work if
aggregate VRAM + `--cpu-offload-gb` covers the ~158 GB FP4+FP8 weights plus
KV cache. A100 (SM 8.0) is untested but the capability gates admit it.

## Credits

Built with AI assistance (Claude); every change validated end-to-end on the
target hardware — greedy-decoding determinism checks, cross-implementation
kernel diffs, and live serving.
