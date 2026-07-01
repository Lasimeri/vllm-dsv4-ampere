# DCP for DeepseekV4-sparse WITH compression — project plan

Objective: shard the compressed KV across 8 ranks so single-sequence context
scales ~8x, KEEPING the 4x/128x compression (unlike --disable-hybrid, which
unifies to max page and destroys it). See DCP-FINDINGS.md for why the cheap
paths fail. All work gated behind `VLLM_SM86_DCP` so default serving is
unchanged.

## Workstreams & milestones (each has a rack test gate)

- [x] **P1a** SlidingWindowSpec DCP-safe (no assert). — done 1805b45
- [x] **P1b** per-group `dcp_exempt` in block tables (SWA replicated). — done 6f096b5
- [ ] **A. Scheduler multi-group CP alignment** (core). Allow `len(groups)>1 &&
      dcp>1` in `resolve_kv_cache_block_sizes`: scheduler_block_size =
      LCM(block sizes) * dcp * pcp; keep hybrid groups (compression intact).
      GATE: engine inits, pool shows ~8x tokens AT COMPRESSED KB/token
      (contrast: disable-hybrid gave 40 KB/tok). Prefix caching off first.
- [ ] **B. Per-group sharded block tables verified** (P1b) with real block
      sizes (256 MLA / 64 SWA / indexer). GATE: slot_mapping round-robin
      correct per group; no OOB.
- [ ] **C. Sparse attention cross-shard combine** (model-level, DeepseekV4).
      Each rank scores its local shard (indexer/K12) -> all-gather local
      top-512 -> global top-512 -> all-reduce selected KV latents into
      replicated [512, latent] -> reuse K13 unchanged. Capture-safe (fixed
      512, masked scatter). GATE: greedy decode IDENTICAL to dcp=1 (A/B).
- [ ] **D. Shard the indexer cache too** (full 8x). GATE: capacity + still
      identical greedy.
- [ ] **E. Prefix caching + block hashing under CP** (hash_block_size = GCD;
      verify cross-rank consistency) OR document it stays off under DCP.
- [ ] **F. Perf + capacity validation**: measure single-seq max context and
      tps vs 80/20; update README + serve script.

## Test protocol
Launch: `VLLM_SM86_DCP=1 bash serve_sm86.sh --decode-context-parallel-size 8`
(hybrid KEPT). Correctness A/B: same greedy prompt vs a dcp=1 server, outputs
must match token-for-token. Capacity: read "GPU KV cache size" + KB/token.

## Status log
- (start) Workstream A in progress.
