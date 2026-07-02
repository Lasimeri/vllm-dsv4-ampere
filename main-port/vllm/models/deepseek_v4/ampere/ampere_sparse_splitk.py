# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K (FlashDecoding) BF16 sparse-MLA decode for SM86.

``triton_bf16_mla_sparse_interface`` runs one program per (token, head-tile).
At batch-1 decode with 8 local heads that is a single program on a 68-SM GPU.
This variant partitions the index columns across ``n_splits`` programs, each
running a local online-softmax over its slice into unnormalized (m, l, acc)
scratch; a combine kernel merges the splits with the exact log-sum-exp
reduction. Same contract and numerics as the base kernel (finite -1e30
sentinel, -1 index holes allowed anywhere).
"""

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton


@triton.jit
def _bf16_sparse_splitk_partial(
    q_ptr,  # [T, H, D] bf16
    kv_ptr,  # [seq_kv, D] bf16 (flat workspace)
    idx_ptr,  # [T, W] int32
    part_m_ptr,  # [T, S, BLOCK_H] f32
    part_l_ptr,  # [T, S, BLOCK_H] f32
    part_acc_ptr,  # [T, S, BLOCK_H, D] f32
    seq_kv,
    h_q,
    sm_scale,  # already folded with LOG2E
    idx_stride_t,
    q_stride_t,
    q_stride_h,
    SPLIT_LEN: tl.constexpr,
    W: tl.constexpr,  # index row width
    D: tl.constexpr,  # 512
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)

    h_off = tl.arange(0, BLOCK_H)
    mask_h = h_off < h_q
    d_off = tl.arange(0, D)

    q = tl.load(
        q_ptr + pid_t * q_stride_t + h_off[:, None] * q_stride_h + d_off[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - 1e30
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    k_lo = pid_s * SPLIT_LEN
    for k_local in range(0, SPLIT_LEN, BLOCK_N):
        cols = k_lo + k_local + tl.arange(0, BLOCK_N)
        col_mask = cols < W
        idx = tl.load(idx_ptr + pid_t * idx_stride_t + cols, mask=col_mask, other=-1)
        mask_kv = (idx >= 0) & (idx < seq_kv)

        k = tl.load(
            kv_ptr + idx[None, :].to(tl.int64) * D + d_off[:, None],
            mask=mask_kv[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k.to(q.dtype)) * sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -1e30)

        v = tl.load(
            kv_ptr + idx[:, None].to(tl.int64) * D + d_off[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )

        n_m = tl.maximum(tl.max(qk, 1), m_i)
        re_scale = tl.exp2(m_i - n_m)
        p = tl.exp2(qk - n_m[:, None])
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * re_scale + tl.sum(p, 1)
        m_i = n_m

    n_splits = tl.num_programs(1)
    base = pid_t * n_splits * BLOCK_H + pid_s * BLOCK_H
    tl.store(part_m_ptr + base + h_off, m_i, mask=mask_h)
    tl.store(part_l_ptr + base + h_off, l_i, mask=mask_h)
    acc_base = (pid_t * n_splits + pid_s) * BLOCK_H * D
    tl.store(
        part_acc_ptr + acc_base + h_off[:, None] * D + d_off[None, :],
        acc,
        mask=mask_h[:, None],
    )


@triton.jit
def _bf16_sparse_splitk_combine(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    out_ptr,  # [T, H, D] bf16
    lse_ptr,  # [T, H] f32
    maxl_ptr,  # [T, H] f32
    h_q,
    out_stride_t,
    out_stride_h,
    lse_stride_t,
    N_SPLITS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    LOGE2: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h_off = tl.arange(0, BLOCK_H)
    mask_h = h_off < h_q

    base_t = pid_t * N_SPLITS * BLOCK_H

    # Pass 1: global max and merged denominator across splits (exact LSE merge).
    m_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1e30
    for s in tl.static_range(N_SPLITS):
        m_s = tl.load(part_m_ptr + base_t + s * BLOCK_H + h_off, mask=mask_h, other=-1e30)
        m_max = tl.maximum(m_max, m_s)
    l_tot = tl.zeros([BLOCK_H], dtype=tl.float32)
    for s in tl.static_range(N_SPLITS):
        m_s = tl.load(part_m_ptr + base_t + s * BLOCK_H + h_off, mask=mask_h, other=-1e30)
        l_s = tl.load(part_l_ptr + base_t + s * BLOCK_H + h_off, mask=mask_h, other=0.0)
        l_tot += l_s * tl.exp2(m_s - m_max)
    l_safe = tl.maximum(l_tot, 1e-30)

    # Pass 2: merge accumulators, tiled over the value dim.
    for dv0 in tl.static_range(0, D, BLOCK_DV):
        dv = dv0 + tl.arange(0, BLOCK_DV)
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
        for s in tl.static_range(N_SPLITS):
            m_s = tl.load(
                part_m_ptr + base_t + s * BLOCK_H + h_off, mask=mask_h, other=-1e30
            )
            a_s = tl.load(
                part_acc_ptr
                + (pid_t * N_SPLITS + s) * BLOCK_H * D
                + h_off[:, None] * D
                + dv[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            acc += a_s * tl.exp2(m_s - m_max)[:, None]
        out = acc / l_safe[:, None]
        tl.store(
            out_ptr + pid_t * out_stride_t + h_off[:, None] * out_stride_h + dv[None, :],
            out.to(tl.bfloat16),
            mask=mask_h[:, None],
        )

    max_logits = m_max * LOGE2
    lse = max_logits + tl.log2(l_safe) * LOGE2
    tl.store(lse_ptr + pid_t * lse_stride_t + h_off, lse, mask=mask_h)
    tl.store(maxl_ptr + pid_t * lse_stride_t + h_off, max_logits, mask=mask_h)


def triton_bf16_mla_sparse_splitk(
    q: torch.Tensor,  # [T, H, D]
    kv: torch.Tensor,  # [seq_kv, 1, D]
    indices: torch.Tensor,  # [T, 1, W]
    sm_scale: float,
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in split-K replacement for triton_bf16_mla_sparse_interface
    (decode shape: block_dpe=0, d_v == dim_qk == 512)."""
    T, H, D = q.shape
    assert d_v == D, "split-K decode supports d_v == dim_qk only"
    idx_2d = indices.squeeze(1)
    W = idx_2d.shape[1]
    kv_2d = kv.squeeze(1)
    assert kv_2d.stride(0) == D and kv_2d.stride(1) == 1

    BLOCK_N = 16
    BLOCK_H = max(16, triton.next_power_of_2(H))
    n_tiles = triton.cdiv(W, BLOCK_N)
    # Fill the SMs: one split per tile, capped so combine stays cheap.
    n_splits = max(1, min(n_tiles, 64))
    tiles_per_split = triton.cdiv(n_tiles, n_splits)
    SPLIT_LEN = tiles_per_split * BLOCK_N
    n_splits = triton.cdiv(W, SPLIT_LEN)

    dev = q.device
    part_m = torch.empty(T, n_splits, BLOCK_H, dtype=torch.float32, device=dev)
    part_l = torch.empty(T, n_splits, BLOCK_H, dtype=torch.float32, device=dev)
    part_acc = torch.empty(
        T, n_splits, BLOCK_H, D, dtype=torch.float32, device=dev
    )
    out = torch.empty(T, H, D, dtype=q.dtype, device=dev)
    lse = torch.empty(T, H, dtype=torch.float32, device=dev)
    max_logits = torch.empty(T, H, dtype=torch.float32, device=dev)

    _bf16_sparse_splitk_partial[(T, n_splits)](
        q,
        kv_2d,
        idx_2d,
        part_m,
        part_l,
        part_acc,
        seq_kv=kv_2d.shape[0],
        h_q=H,
        sm_scale=sm_scale * LOG2E,
        idx_stride_t=idx_2d.stride(0),
        q_stride_t=q.stride(0),
        q_stride_h=q.stride(1),
        SPLIT_LEN=SPLIT_LEN,
        W=W,
        D=D,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    _bf16_sparse_splitk_combine[(T,)](
        part_m,
        part_l,
        part_acc,
        out,
        lse,
        max_logits,
        h_q=H,
        out_stride_t=out.stride(0),
        out_stride_h=out.stride(1),
        lse_stride_t=lse.stride(0),
        N_SPLITS=n_splits,
        D=D,
        BLOCK_H=BLOCK_H,
        BLOCK_DV=128,
        LOGE2=LOGE2,
    )
    return out, max_logits, lse
