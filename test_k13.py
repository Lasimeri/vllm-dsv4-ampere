"""Standalone parity + microbench for K13 (split-K) vs K11 (single-program)
SM86 FlashMLA sparse decode. Both consume the identical FP8 cache layout, so
identical inputs must yield identical out+lse if K13's split/merge is correct.
"""
import os, time
import torch

os.environ.setdefault("VLLM_SM86_DEEPSEEK_V4_REF", "1")

from vllm.third_party.flashmla.flash_mla_decode_sm86 import (
    _flash_mla_decode_sm86_triton,   # K11
    _flash_mla_decode_sm86_splitk,   # K13
    NOPE_DIM, BF16_DIM, TOKEN_DATA_SIZE, TOKEN_SCALE_DIM,
)
from vllm.third_party.flashmla.flash_mla_interface import _flash_mla_decode_pyref

dev = "cuda"
torch.manual_seed(0)

H_q, D_QK, D_V, TOPK = 64, 576, 512, 512
B = 1
block_size = 64
num_blocks = 128                       # 8192 slots, covers topk indices
head_bytes = TOKEN_DATA_SIZE + TOKEN_SCALE_DIM   # 584


def build_cache(num_blocks, block_size):
    """Random but NaN-free FP8 cache in the SM86 packed layout."""
    flat = torch.zeros(num_blocks, block_size * head_bytes, dtype=torch.uint8, device=dev)
    for p in range(block_size):
        d0 = p * TOKEN_DATA_SIZE
        # NOPE fp8: small bf16 -> e4m3fn -> bytes (no NaN)
        nope = (torch.randn(num_blocks, NOPE_DIM, device=dev) * 0.7).to(torch.float8_e4m3fn)
        flat[:, d0:d0 + NOPE_DIM] = nope.view(torch.uint8)
        # BF16: small bf16 -> 2 bytes LE
        bf = (torch.randn(num_blocks, BF16_DIM, device=dev) * 0.7).to(torch.bfloat16)
        bf_bytes = bf.view(torch.uint8).reshape(num_blocks, BF16_DIM, 2)
        flat[:, d0 + NOPE_DIM:d0 + TOKEN_DATA_SIZE] = bf_bytes.reshape(num_blocks, BF16_DIM * 2)
        # Scales: UE8M0 exponent bytes ~127
        s0 = block_size * TOKEN_DATA_SIZE + p * TOKEN_SCALE_DIM
        flat[:, s0:s0 + TOKEN_SCALE_DIM] = torch.randint(
            124, 131, (num_blocks, TOKEN_SCALE_DIM), dtype=torch.uint8, device=dev)
    return flat.reshape(num_blocks, block_size, 1, head_bytes)


def run(fn, q, k_cache, indices, topk_len, sink, extra):
    out = torch.zeros(B, 1, H_q, D_V, dtype=torch.bfloat16, device=dev)
    ek, eidx, elen = extra
    o, lse = fn(
        q=q, k_cache=k_cache, head_dim_v=D_V, indices=indices,
        topk_length=topk_len, attn_sink=sink, softmax_scale=D_QK ** -0.5,
        extra_k_cache=ek, extra_indices_in_kvcache=eidx, extra_topk_length=elen,
        out=out,
    )
    return o.clone(), lse.clone()


def case(name, topk_len=None, sink=None, with_extra=False):
    q = (torch.randn(B, 1, H_q, D_QK, device=dev) * 0.5).to(torch.bfloat16)
    k_cache = build_cache(num_blocks, block_size)
    indices = torch.randint(0, num_blocks * block_size, (B, TOPK), dtype=torch.int32, device=dev)
    extra = (None, None, None)
    if with_extra:
        ek = build_cache(num_blocks, block_size)
        eidx = torch.randint(0, num_blocks * block_size, (B, 64), dtype=torch.int32, device=dev)
        extra = (ek, eidx, None)

    o11, l11 = run(_flash_mla_decode_sm86_triton, q, k_cache, indices, topk_len, sink, extra)
    o12, l12 = run(_flash_mla_decode_sm86_splitk, q, k_cache, indices, topk_len, sink, extra)

    od = (o11.float() - o12.float()).abs()
    ld = (l11.float() - l12.float()).abs()
    print(f"[{name:22s}] out max|Δ|={od.max():.3e} mean={od.mean():.3e} | "
          f"lse max|Δ|={ld.max():.3e} | nan11={torch.isnan(o11).any().item()} "
          f"nan12={torch.isnan(o12).any().item()}")
    return od.max().item()


print("=== PARITY: K13 split-K vs K11 single-program ===")
mx = 0
mx = max(mx, case("main only"))
mx = max(mx, case("topk_len=300", topk_len=torch.tensor([300], dtype=torch.int32, device=dev)))
mx = max(mx, case("sink", sink=torch.randn(H_q, device=dev)))
mx = max(mx, case("extra", with_extra=True))
mx = max(mx, case("sink+extra+tlen",
                  topk_len=torch.tensor([400], dtype=torch.int32, device=dev),
                  sink=torch.randn(H_q, device=dev), with_extra=True))
print(f"=> worst out max|Δ| = {mx:.3e}  ({'PASS' if mx < 5e-2 else 'FAIL'})")


def bench(fn, q, k_cache, indices, n=200):
    out = torch.zeros(B, 1, H_q, D_V, dtype=torch.bfloat16, device=dev)
    args = dict(k_cache=k_cache, head_dim_v=D_V, indices=indices, topk_length=None,
                attn_sink=None, softmax_scale=D_QK ** -0.5, extra_k_cache=None,
                extra_indices_in_kvcache=None, extra_topk_length=None, out=out)
    for _ in range(20):
        fn(q=q, **args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(q=q, **args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us/call


for HB in (8, 64):  # 8 = TP=8 per-rank serve shape; 64 = single-GPU
    print(f"\n=== MICROBENCH (us/call, TOPK=512, H={HB}) ===")
    q = (torch.randn(B, 1, HB, D_QK, device=dev) * 0.5).to(torch.bfloat16)
    k_cache = build_cache(num_blocks, block_size)
    indices = torch.randint(0, num_blocks * block_size, (B, TOPK), dtype=torch.int32, device=dev)

    def benchH(fn, n=200):
        out = torch.zeros(B, 1, HB, D_V, dtype=torch.bfloat16, device=dev)
        a = dict(k_cache=k_cache, head_dim_v=D_V, indices=indices, topk_length=None,
                 attn_sink=None, softmax_scale=D_QK ** -0.5, extra_k_cache=None,
                 extra_indices_in_kvcache=None, extra_topk_length=None, out=out)
        for _ in range(20):
            fn(q=q, **a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(q=q, **a)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6

    usref = benchH(_flash_mla_decode_pyref)
    us11 = benchH(_flash_mla_decode_sm86_triton)
    us12 = benchH(_flash_mla_decode_sm86_splitk)
    print(f"pyref (production) : {usref:8.1f} us/call")
    print(f"K11 single-program : {us11:8.1f} us/call   ({usref/us11:.2f}x vs pyref)")
    print(f"K13 split-K        : {us12:8.1f} us/call   ({usref/us12:.2f}x vs pyref)")
