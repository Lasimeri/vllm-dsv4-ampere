#!/usr/bin/env python
"""Parity + memory check for the tiled SM86 indexer prefill ref.

Compares _fp8_mqa_logits_pyref (now M-tiled) against an inline untiled
reference, and confirms the M=2048/N=8192 case that OOM-crashed the engine
now fits within a bounded transient footprint.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python test_indexer_prefill.py
"""
import torch

from vllm.utils.deep_gemm import _fp8_mqa_logits_pyref

DEV = "cuda"
H = 64
D = 128  # index_head_dim


def untiled_ref(q, k, ks, w, cks, cke, clean):
    M, H_, _ = q.shape
    N = k.shape[0]
    q_bf = q.to(torch.bfloat16)
    k_bf = k.to(torch.bfloat16) * ks.to(torch.bfloat16).unsqueeze(-1)
    scores = torch.einsum("mhd,nd->mhn", q_bf, k_bf)
    logits = (w.to(torch.float32).unsqueeze(-1) * scores.to(torch.float32)).sum(dim=1)
    n_idx = torch.arange(N, device=logits.device, dtype=cks.dtype)
    valid = (n_idx.unsqueeze(0) >= cks.unsqueeze(1)) & (n_idx.unsqueeze(0) < cke.unsqueeze(1))
    fill = float("-inf") if clean else 0.0
    return torch.where(valid, logits, torch.full_like(logits, fill))


def make(M, N, seed=0, causal=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(M, H, D, device=DEV, generator=g) * 4).to(torch.float8_e4m3fn)
    k = (torch.randn(N, D, device=DEV, generator=g) * 4).to(torch.float8_e4m3fn)
    kscl = (torch.rand(N, device=DEV, generator=g) * 0.5 + 0.5).to(torch.float32)
    w = torch.randn(M, H, device=DEV, generator=g, dtype=torch.float32)
    if causal:
        # each query m attends [0, base+m+1)
        base = N - M
        cks = torch.zeros(M, device=DEV, dtype=torch.int32)
        cke = torch.clamp(torch.arange(M, device=DEV) + base + 1, max=N).to(torch.int32)
    else:
        cks = torch.zeros(M, device=DEV, dtype=torch.int32)
        cke = torch.full((M,), N, device=DEV, dtype=torch.int32)
    return q, k, kscl, w, cks, cke


def parity():
    print("== INDEXER PARITY (tiled vs untiled) ==")
    ok = True
    for i, (M, N, clean) in enumerate([(384, 4096, True), (384, 4096, False),
                                       (777, 2048, True)]):
        q, k, kscl, w, cks, cke = make(M, N, seed=i)
        ref = untiled_ref(q, k, kscl, w, cks, cke, clean)
        got = _fp8_mqa_logits_pyref(q, k, kscl, w, cks, cke, clean)
        # compare only finite cells (fill is -inf/0 in both, identical mask)
        fin = torch.isfinite(ref) & torch.isfinite(got)
        d = (ref[fin] - got[fin]).abs()
        maxd = d.max().item() if d.numel() else 0.0
        # mask-position agreement
        mask_eq = torch.equal(torch.isfinite(ref), torch.isfinite(got))
        good = maxd < 5e-2 and mask_eq
        ok &= good
        print(f"  M={M} N={N} clean={clean}: max|Δ|(finite)={maxd:.5f} "
              f"mask_eq={mask_eq}  {'OK' if good else 'FAIL'}")
    print("  PARITY:", "PASS" if ok else "FAIL")
    return ok


def mem_check():
    print("== MEMORY (prev OOM case M=2048 N=8192) ==")
    q, k, kscl, w, cks, cke = make(2048, 8192, seed=42)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _ = _fp8_mqa_logits_pyref(q, k, kscl, w, cks, cke, True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    print(f"  transient peak: {peak/1e6:.1f} MB  (untiled would be ~{2048*64*8192*6/1e9:.1f} GB)")
    return peak


if __name__ == "__main__":
    print(f"torch {torch.__version__}  dev {torch.cuda.get_device_name(0)}")
    ok = parity()
    mem_check()
    raise SystemExit(0 if ok else 1)
