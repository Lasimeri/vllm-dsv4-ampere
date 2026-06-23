"""Indexer paged-MQA-logits kernel (K12, fp32) vs pyref vs fp32 truth.

Both q and K are fp8_e4m3 (exact to fp32). The fp32 kernel should match a plain
fp32 torch einsum; the bf16 pyref carries ~5e-2 reduction noise. What matters
for the sparse indexer is top-K *overlap* with the true ranking.
"""
import os, time
import torch

os.environ.setdefault("VLLM_SM86_DEEPSEEK_V4_REF", "1")

from vllm.utils.fp8_paged_mqa_logits_sm86 import _fp8_paged_mqa_logits_sm86_triton
from vllm.utils.deep_gemm import _fp8_paged_mqa_logits_pyref

dev = "cuda"
torch.manual_seed(0)
B, next_n, D = 1, 1, 128
block_size = 64
TOPK = 512


def build(ctx, H):
    M = B * next_n
    nb = (ctx + block_size - 1) // block_size + 1
    # fp8 q and K
    q = (torch.randn(B, next_n, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    cache = torch.zeros(nb, block_size, 1, D + 4, dtype=torch.uint8, device=dev)
    kf8 = (torch.randn(nb, block_size, D, device=dev) * 0.7).to(torch.float8_e4m3fn)
    cache[..., 0, :D] = kf8.view(torch.uint8)
    scale = (torch.rand(nb, block_size, 1, device=dev) * 0.5 + 0.5).to(torch.float32)
    cache[..., 0, D:D + 4] = scale.view(torch.uint8).reshape(nb, block_size, 4)
    weights = (torch.randn(M, H, device=dev)).to(torch.float32)
    ctx_t = torch.tensor([ctx], dtype=torch.int32, device=dev)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, nb)
    return q, cache, weights, ctx_t, bt, scale, kf8


def fp32_truth(q, kf8, scale, weights, ctx, H):
    # exact fp8->fp32 dequant, fp32 einsum, fp32 head-weighted sum
    qf = q.to(torch.float32).reshape(H, D)
    nb = kf8.shape[0]
    kflat = (kf8.to(torch.float32) * scale).reshape(nb * block_size, D)[:ctx]
    scores = torch.einsum("hd,nd->hn", qf, kflat)            # [H, ctx]
    logits = (weights.reshape(H, 1) * scores).sum(0)          # [ctx]
    return logits


def topk_overlap(a, b, k):
    ia = torch.topk(a, k).indices
    ib = torch.topk(b, k).indices
    return len(set(ia.tolist()) & set(ib.tolist())) / k


for H in (8, 64):
    for ctx in (2048, 8192):
        q, cache, weights, ctx_t, bt, scale, kf8 = build(ctx, H)
        Lk = _fp8_paged_mqa_logits_sm86_triton(q, cache, weights, ctx_t, bt, ctx, False)[0, :ctx]
        Lp = _fp8_paged_mqa_logits_pyref(q, cache, weights, ctx_t, bt, ctx, False)[0, :ctx]
        Lt = fp32_truth(q, kf8, scale, weights, ctx, H)
        dk = (Lk - Lt).abs().max().item()
        dp = (Lp - Lt).abs().max().item()
        ov_k = topk_overlap(Lk, Lt, TOPK)
        ov_p = topk_overlap(Lp, Lt, TOPK)
        print(f"H={H:2d} ctx={ctx:5d} | max|Δ vs truth|  kernel={dk:.2e} pyref={dp:.2e} "
              f"| top{TOPK} overlap vs truth  kernel={ov_k:.4f} pyref={ov_p:.4f}")

print("\n=== speed (us/call) ===")
for H in (8, 64):
    q, cache, weights, ctx_t, bt, scale, kf8 = build(8192, H)
    def bench(fn, n=100):
        for _ in range(10):
            fn(q, cache, weights, ctx_t, bt, 8192, False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            fn(q, cache, weights, ctx_t, bt, 8192, False)
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
    uk = bench(_fp8_paged_mqa_logits_sm86_triton)
    up = bench(_fp8_paged_mqa_logits_pyref)
    print(f"H={H:2d} ctx=8192 | kernel={uk:7.1f}  pyref={up:7.1f}  ({up/uk:.2f}x)")
