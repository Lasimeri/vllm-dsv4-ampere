"""K12: fp8_paged_mqa_logits Triton kernel for SM86.

Replaces _fp8_paged_mqa_logits_pyref (utils/deep_gemm.py:576) with a single
fused Triton kernel:
  - eliminates the per-batch host-sync `int(ctx_per_batch[b].item())` so the
    op becomes cudagraph-FULL_DECODE_ONLY-safe in principle
  - fuses FP8 dequant + per-token fp32 scale + Q@K^T einsum + per-head
    weighted reduction into one pass over the K cache
  - parallelises across (M, K-tile) so the entire context length is split
    across the GPU's 68 SMs instead of running a Python for-batch loop

Cache layout (matches the SM86 reference at deep_gemm.py:576):
  kv_cache: [num_blocks, block_size, 1, D+4] uint8
    - bytes [0:D]    : D float8_e4m3fn K values
    - bytes [D:D+4]  : float32 fp32 dequant scale (single scalar per slot)

Logits shape:
  [B*next_n, max_model_len] float32. Positions outside ctx keep the wrapper
  fill (clean_logits=True -> -inf, else 0.0).

Decode call site uses small H_q (TP=8 with 64 heads = 8 per rank), small D
(128), large max_model_len (100k). The K-tile parallelism is what unlocks
the SM utilisation cuBLAS+pyref couldn't reach via Python's per-batch loop.
"""
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _fp8_e4m3fn_byte_to_bf16,
)


@triton.jit
def _fp8_paged_mqa_logits_sm86_kernel(
    Q_ptr,             # bf16 [M, H, D]
    Q_m_stride, Q_h_stride,
    KV_CACHE_ptr,      # uint8 paged [num_blocks, block_size, 1, D+4]
    KV_block_stride,   # bytes per block (cache.stride(0))
    KV_token_stride,   # bytes per token slot (cache.stride(1))
    BT_ptr,            # int32 [B, max_blocks]
    BT_b_stride,
    CTX_ptr,           # int32 [B]  (caller pre-amaxes if 2D)
    WEIGHTS_ptr,       # f32 [M, H]
    W_m_stride,
    LOGITS_ptr,        # f32 [M, MAX_MODEL_LEN]
    L_m_stride,
    M,                 # = B * next_n
    H_q,
    MAX_MODEL_LEN,
    NEXT_N,
    BLOCK_SIZE,
    BT_max_idx,        # = block_tables.shape[1]
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = pid_m // NEXT_N

    # ---- Load Q [BLOCK_H, D] ----
    h_off = tl.arange(0, BLOCK_H)
    h_mask = h_off < H_q
    d_off = tl.arange(0, D_C)

    q_ptrs = Q_ptr + pid_m * Q_m_stride + h_off[:, None] * Q_h_stride + d_off[None, :]
    # fp32 Q (host passes fp32). fp8 K dequants exactly to fp32, so the whole
    # logit is computed at full precision -- the truest top-K ranking. The bf16
    # path (what shipped as the experimental K12) injected ~5e-2 reduction-order
    # noise that reshuffled the order-sensitive sparse top-K.
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    # ---- Load weights [BLOCK_H] f32 ----
    w_ptrs = WEIGHTS_ptr + pid_m * W_m_stride + h_off
    weights = tl.load(w_ptrs, mask=h_mask, other=0.0)

    # ---- Load ctx[b] (int32; on-device, no host sync) ----
    ctx = tl.load(CTX_ptr + b).to(tl.int32)

    # ---- Position tile ----
    pos_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    pos_mask = (pos_off < ctx) & (pos_off < MAX_MODEL_LEN)
    safe_pos = tl.where(pos_mask, pos_off, 0)

    block_idx = safe_pos // BLOCK_SIZE
    in_block = safe_pos % BLOCK_SIZE
    bt_in_range = block_idx < BT_max_idx
    full_mask = pos_mask & bt_in_range
    safe_block_idx = tl.where(full_mask, block_idx, 0)

    # ---- Physical block id from block_table ----
    bt_ptrs = BT_ptr + b * BT_b_stride + safe_block_idx
    block_phys = tl.load(bt_ptrs, mask=full_mask, other=0).to(tl.int64)

    # ---- Per-slot cache byte base ----
    k_byte_base = (
        block_phys * KV_block_stride.to(tl.int64)
        + in_block.to(tl.int64) * KV_token_stride.to(tl.int64)
    )

    # ---- Load D fp8 bytes per slot [BLOCK_N, D] ----
    k_byte_ptrs = KV_CACHE_ptr + k_byte_base[:, None] + d_off[None, :]
    k_bytes = tl.load(k_byte_ptrs, mask=full_mask[:, None], other=0)
    # fp8_e4m3 (3-bit mantissa) -> bf16 is exact -> fp32 is exact.
    k_f = _fp8_e4m3fn_byte_to_bf16(k_bytes).to(tl.float32)  # [BLOCK_N, D] fp32

    # ---- Load 4 bytes of fp32 scale per slot, repack manually ----
    # The scale lives at offset D in the slot. We load 4 individual bytes
    # and repack to int32 then bitcast to fp32. (Triton can't easily do an
    # aligned 4-byte load through a uint8 ptr, so we use 4 byte loads.)
    s0 = (tl.load(KV_CACHE_ptr + k_byte_base + (D_C + 0), mask=full_mask, other=0).to(tl.int32)) & 0xFF
    s1 = (tl.load(KV_CACHE_ptr + k_byte_base + (D_C + 1), mask=full_mask, other=0).to(tl.int32)) & 0xFF
    s2 = (tl.load(KV_CACHE_ptr + k_byte_base + (D_C + 2), mask=full_mask, other=0).to(tl.int32)) & 0xFF
    s3 = (tl.load(KV_CACHE_ptr + k_byte_base + (D_C + 3), mask=full_mask, other=0).to(tl.int32)) & 0xFF
    s_packed = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
    scale_f32 = s_packed.to(tl.float32, bitcast=True)  # [BLOCK_N] f32
    # Apply the dequant scale in fp32 (no bf16 downcast) so K is exact.
    k_f = k_f * scale_f32[:, None]

    # ---- scores = Q @ (K_scaled)^T in true fp32 (input_precision="ieee"
    # forces the IEEE fp32 path, not TF32, so the ranking is deterministic
    # and matches a plain torch fp32 einsum). ----
    scores = tl.dot(q, tl.trans(k_f), out_dtype=tl.float32, input_precision="ieee")
    scores = tl.where(full_mask[None, :], scores, 0.0)

    # ---- weighted sum across H: logits_tile[n] = sum_h(w[h] * scores[h, n]) ----
    weighted = scores * weights[:, None]
    weighted = tl.where(h_mask[:, None], weighted, 0.0)
    logits_tile = tl.sum(weighted, axis=0)  # [BLOCK_N] f32

    # ---- store ----
    out_ptrs = LOGITS_ptr + pid_m * L_m_stride + pos_off
    tl.store(out_ptrs, logits_tile, mask=pos_mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    # tl.dot M dim min 16
    return max(p, 16)


def _fp8_paged_mqa_logits_sm86_triton(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """K12: fused fp8_paged_mqa_logits Triton kernel for SM86. Same contract
    as _fp8_paged_mqa_logits_pyref."""
    B, next_n, H, D = q_values.shape
    M = B * next_n

    block_size = kv_cache.shape[1]
    cache_dim = kv_cache.shape[-1]
    if cache_dim != D + 4:
        raise RuntimeError(
            f"k12: kv_cache last dim {cache_dim} != D+4 ({D+4}); layout mismatch"
        )

    fill = float("-inf") if clean_logits else 0.0
    logits = torch.full(
        (M, max_model_len), fill, dtype=torch.float32, device=q_values.device
    )

    # Pre-amax across next_n if context_lens is 2D, matching pyref.
    if context_lens.dim() == 2:
        ctx_per_batch = context_lens.amax(dim=-1).to(torch.int32).contiguous()
    else:
        ctx_per_batch = context_lens.to(torch.int32).contiguous()

    # Q -> fp32 [M, H, D] contiguous (full-precision logit path)
    q_bf = q_values.to(torch.float32).reshape(M, H, D).contiguous()

    # Cache byte strides
    kv_block_stride = kv_cache.stride(0)
    kv_token_stride = kv_cache.stride(1)

    # Block tables
    block_tables = block_tables.contiguous().to(torch.int32)
    bt_b_stride = block_tables.stride(0)
    bt_max_idx = block_tables.shape[1]

    # Weights [M, H] f32 contiguous
    weights_f = weights.to(torch.float32).reshape(M, H).contiguous()

    BLOCK_H = _next_pow2(H)
    # 64 (not 128): the fp32 K tile is [BLOCK_N, D] -- BLOCK_N=64 keeps it at
    # 32 KB (same as the old bf16 128-tile) and shrinks per-launch scratch,
    # which matters under the packed gpu-memory-utilization=0.95 serve budget.
    BLOCK_N = 64

    grid = (M, triton.cdiv(max_model_len, BLOCK_N))

    _fp8_paged_mqa_logits_sm86_kernel[grid](
        q_bf,
        q_bf.stride(0), q_bf.stride(1),
        kv_cache,
        kv_block_stride,
        kv_token_stride,
        block_tables,
        bt_b_stride,
        ctx_per_batch,
        weights_f,
        weights_f.stride(0),
        logits,
        logits.stride(0),
        M,
        H,
        max_model_len,
        next_n,
        block_size,
        bt_max_idx,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        D_C=D,
        num_warps=4,
    )

    return logits


def _k12_enabled() -> bool:
    """K12 fused paged-MQA logits (indexer).

    History: the original bf16 kernel was 3.6x kernel-isolated but injected
    ~5e-2 reduction-order noise vs the bf16 pyref, reshuffling the sparse
    indexer's order-sensitive top-K and dropping MTP draft acceptance
    (~87% -> ~80%). This version computes the logit in **true fp32** (exact
    fp8->fp32 dequant, fp32 scale, IEEE fp32 dot): it is the most accurate
    ranking available, not a noisy approximation of the bf16 pyref, so the
    top-K it produces is at least as good as the pyref's.

    At the TP=8 serve shape (H_q=8/rank) the fp32 kernel is 4.42x faster than
    the pyref in isolation (71 vs 314 us/call) AND has perfect top-512 overlap
    with the fp32-true ranking (the pyref misses 1-3 of the true top-512 to
    bf16 noise -- the fp32 kernel is strictly the more faithful path).

    Default OFF. Reason is NOT numerics (those are now ideal) but memory: at the
    production gpu-memory-utilization=0.95 packing, JIT-compiling + loading the
    Triton module into an already-95%-full GPU on the first decode call throws
    'Triton Error [CUDA]: out of memory', and the per-call fallback to pyref
    then costs ~5x (4.51 -> 0.9 tok/s). The pyref needs no GPU-resident module.
    Enable with VLLM_SM86_K12=1 AND give headroom (e.g. --gpu-memory-utilization
    0.90, or a warm-up launch before the cache fills). Large H_q (TP=1) is also
    slower here since the fp32 dot has no tensor-core path.
    """
    forced = os.environ.get("VLLM_SM86_K12", "").strip()
    if forced in ("1", "true", "True"):
        return True
    return False
