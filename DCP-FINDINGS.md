# DCP (decode context parallel) for DeepseekV4-sparse on SM86 — investigation

Goal: shard the KV cache across the 8 TP ranks so single-sequence context
scales ~8x instead of the MLA latent being replicated on every rank.

## Result: blocked by a fundamental compression-vs-uniform-group tension in vLLM core

Three blockers hit empirically, in order:

1. **SWA assert** (`kv_cache_interface.py:444`, `SlidingWindowSpec.max_memory_usage_bytes`):
   `assert dcp==1 "DCP not support sliding window"`. **FIXED** (phase 1a) — gated
   behind `VLLM_SM86_DCP`, SWA stays replicated. Phase 1b threads a per-group
   `dcp_exempt` flag so the SWA group is not round-robin sharded
   (`block_table.py`, `gpu_input_batch.py`, `gpu_model_runner.py`).

2. **Multi-group CP rejection** (`kv_cache_utils.py:596`, `resolve_kv_cache_block_sizes`):
   `raise "Hybrid KV cache groups with multiple block sizes do not support
   context parallelism"`. DeepseekV4 is irreducibly hybrid — compressor C4
   (block 256, storage 64), C128 (storage 2), indexer, and SWA (block 64) are
   distinct specs → multiple groups. CP requires a single group.
   **Bypassable** with `--disable-hybrid-kv-cache-manager` (→
   `unify_hybrid_kv_cache_specs`, which explicitly knows DeepseekV4 and
   collapses everything to one MLAAttentionSpec type at block 256).

3. **Disable-hybrid destroys the compression (the real blocker).** Unifying to a
   single group forces one uniform page size = the MAX page, so the 128x/4x
   compressed layers get allocated at full page. Measured under
   `--disable-hybrid --decode-context-parallel-size 8`:
   - Pool: 3.08 GiB, **79,872 tokens (2.44x @ 32k)** = ~40 KB/token/rank.
   - A model where compression AND dcp-sharding both held predicts ~1.2 KB/token
     → ~2.6M tokens. The 33x gap = compression lost, sharding not reflected in
     pool sizing.
   - Output was **garbage** ("not least theDow ,54packaging") — the runtime DID
     shard the sparse KV round-robin, but the sparse attention has no cross-rank
     gather (each rank attends to its 1/8 shard). Phase 2 (top-k merge + KV
     all-reduce + reuse K13) would fix correctness, but capacity is already
     worse than the compressed baseline, so disable-hybrid is a dead end.

## Conclusion

DeepseekV4's memory efficiency COMES FROM the hybrid multi-block-size
compression; vLLM's context parallelism requires a single uniform group, which
destroys that compression. They are mutually exclusive in current vLLM.

**A real DCP that keeps compression requires extending vLLM core CP to hybrid
multi-block-size groups** — the scheduler block-size alignment
(`resolve_kv_cache_block_sizes`), block hashing, prefix caching, and per-group
DCP block-table sharding all assume uniform/single group under CP. Multi-week
upstream feature.

**For context room now, use `CPU_OFFLOAD_GB=15` (80/20 offload):** compression
intact, KV pool grows to 9.81 GiB → **202,139 tokens, 6.17x @ 32k** — 2.5x more
usable context than disable-hybrid+DCP, with correct output. See serve_sm86.sh.

## Phase 1 code (this branch)

Correct and would be reused by a full solution; kept behind `VLLM_SM86_DCP` so
default behavior is unchanged. Does NOT make DCP usable on its own (blocker 3).
