#!/usr/bin/env python
"""Parity + microbench for the SM86 sparse-prefill FlashMLA reference.

Compares the vectorized `_flash_mla_prefill_pyref` against the per-token
`_flash_mla_prefill_pyref_loop` ground truth at DeepSeek-V4-Flash prefill
shapes (h_q=64, d_qk=576, d_v=512, topk=512). No model load required.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python test_prefill.py
"""
import os
import time

import torch

from vllm.third_party.flashmla.flash_mla_interface import (
    _flash_mla_prefill_pyref,
    _flash_mla_prefill_pyref_loop,
)

DEV = "cuda"
H_Q = 64
D_QK = 576
D_V = 512
TOPK = 512
SM_SCALE = D_QK ** -0.5


def make_case(s_q, s_kv, *, use_topk_len, with_invalid, with_sink, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(s_q, H_Q, D_QK, device=DEV, dtype=torch.bfloat16, generator=g)
    kv = torch.randn(s_kv, 1, D_QK, device=DEV, dtype=torch.bfloat16, generator=g)
    # Random attended positions in [0, s_kv); slot 0 always valid so no token
    # degenerates to an all-masked (nan) row.
    idx = torch.randint(0, s_kv, (s_q, 1, TOPK), device=DEV, dtype=torch.int32,
                        generator=g)
    if with_invalid:
        # Sprinkle out-of-range / negative sentinels (~15% of slots, never slot 0).
        mask = torch.rand(s_q, 1, TOPK, device=DEV, generator=g) < 0.15
        mask[..., 0] = False
        idx = torch.where(mask, torch.full_like(idx, -1), idx)
    topk_length = None
    if use_topk_len:
        topk_length = torch.randint(1, TOPK + 1, (s_q,), device=DEV,
                                    dtype=torch.int32, generator=g)
    attn_sink = None
    if with_sink:
        attn_sink = torch.randn(H_Q, device=DEV, dtype=torch.float32, generator=g)
    return q, kv, idx, topk_length, attn_sink


def diff(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    denom = b.abs().clamp_min(1e-3)
    return d.max().item(), (d / denom).max().item()


def run_parity():
    print("== PARITY (vectorized vs per-token loop) ==")
    cases = [
        dict(use_topk_len=False, with_invalid=False, with_sink=False),
        dict(use_topk_len=True,  with_invalid=False, with_sink=False),
        dict(use_topk_len=False, with_invalid=True,  with_sink=False),
        dict(use_topk_len=True,  with_invalid=True,  with_sink=True),
    ]
    ok = True
    for i, c in enumerate(cases):
        q, kv, idx, tkl, sink = make_case(384, 4096, seed=i, **c)
        o_ref, ml_ref, lse_ref = _flash_mla_prefill_pyref_loop(
            q, kv, idx, SM_SCALE, D_V, sink, tkl, None)
        o_vec, ml_vec, lse_vec = _flash_mla_prefill_pyref(
            q, kv, idx, SM_SCALE, D_V, sink, tkl, None)
        od_a, od_r = diff(o_vec, o_ref)
        ld_a, _ = diff(lse_vec, lse_ref)
        md_a, _ = diff(ml_vec, ml_ref)
        # bf16 matmul ULP across two GEMM shapes -> ~1e-2 abs on O(1) outputs.
        good = od_a < 3e-2 and ld_a < 5e-2 and md_a < 5e-2
        ok &= good
        print(f"  case{i} {c} -> out|Δ|={od_a:.4f} (rel {od_r:.3f}) "
              f"lse|Δ|={ld_a:.4f} max|Δ|={md_a:.4f}  {'OK' if good else 'FAIL'}")
    print("  PARITY:", "PASS" if ok else "FAIL")
    return ok


def bench(s_q, s_kv, iters=20):
    q, kv, idx, tkl, sink = make_case(s_q, s_kv, seed=99, use_topk_len=True,
                                      with_invalid=True, with_sink=True)

    def timed(fn, n):
        # warmup
        fn(q, kv, idx, SM_SCALE, D_V, sink, tkl, None)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn(q, kv, idx, SM_SCALE, D_V, sink, tkl, None)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1e3  # ms

    loop_n = max(1, iters // 5)
    t_loop = timed(_flash_mla_prefill_pyref_loop, loop_n)
    t_vec = timed(_flash_mla_prefill_pyref, iters)
    print(f"  s_q={s_q:5d} s_kv={s_kv:6d}: loop={t_loop:8.2f} ms  "
          f"vec={t_vec:7.2f} ms  speedup={t_loop / t_vec:6.1f}x")


def run_bench():
    print(f"== MICROBENCH (BLOCK_T={os.environ.get('VLLM_SM86_PREFILL_BLOCK_T','512')}) ==")
    for s_q, s_kv in [(512, 4096), (2048, 8192), (4096, 16384)]:
        bench(s_q, s_kv)


if __name__ == "__main__":
    print(f"torch {torch.__version__}  dev {torch.cuda.get_device_name(0)}")
    ok = run_parity()
    run_bench()
    raise SystemExit(0 if ok else 1)
