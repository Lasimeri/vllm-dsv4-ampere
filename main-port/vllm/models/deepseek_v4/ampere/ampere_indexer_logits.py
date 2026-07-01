# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM8x (Ampere) torch references for the sparse-indexer MQA logits.

DeepGEMM's ``fp8_fp4_{paged_,}mqa_logits`` assert "Unsupported architecture"
on SM8x. Ampere has no native fp8, so these dequantize the fp8 q/K to bf16 and
run the MQA logits as a bf16 tensor-core einsum (fp8->bf16 is exact: 4-bit
mantissa fits bf16's 7). Ported from the c2fb0133 SM86 build.
"""

import os

import torch


def _fp8_mqa_logits_pyref(
    q_values: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """SM8x fp8 reference for fp8_fp4_mqa_logits (prefill; fp8 q, fp8 k).

    q_values: [M, H, D] float8_e4m3fn; k_packed: [N, D] float8_e4m3fn;
    k_scales: [N] float32; weights: [M, H] float32. Returns [M, N] float32.
    """
    M, H, D = q_values.shape
    N = k_packed.shape[0]
    k_bf = k_packed.to(torch.bfloat16) * k_scales.to(torch.bfloat16).unsqueeze(-1)
    out = torch.empty(M, N, dtype=torch.float32, device=q_values.device)
    n_idx = torch.arange(N, device=q_values.device, dtype=cu_seqlen_ks.dtype)
    fill = float("-inf") if clean_logits else 0.0
    # Tile over query rows so H*BM*N stays under a fixed element budget
    # (the full [M,H,N] + its fp32 copy is multi-GB for a long prefill chunk).
    budget = int(os.environ.get("VLLM_SM86_INDEXER_TILE_ELEMS", str(24_000_000)))
    BM = max(1, min(M, budget // max(1, H * N)))
    for m0 in range(0, M, BM):
        m1 = min(m0 + BM, M)
        q_bf = q_values[m0:m1].to(torch.bfloat16)
        scores = torch.einsum("mhd,nd->mhn", q_bf, k_bf)
        logits = (
            weights[m0:m1].to(torch.float32).unsqueeze(-1) * scores.to(torch.float32)
        ).sum(dim=1)
        valid = (n_idx.unsqueeze(0) >= cu_seqlen_ks[m0:m1].unsqueeze(1)) & (
            n_idx.unsqueeze(0) < cu_seqlen_ke[m0:m1].unsqueeze(1)
        )
        out[m0:m1] = torch.where(valid, logits, torch.full_like(logits, fill))
    return out


def _fp8_paged_mqa_logits_pyref(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """SM8x fp8 reference for fp8_fp4_paged_mqa_logits (decode; fp8 q + fp8 paged KV).

    q_values: [B, next_n, H, D] float8_e4m3fn;
    kv_cache: [num_blocks, block_size, 1, D+4] uint8 (D fp8 K + 4 fp32 scale);
    weights: [B*next_n, H] float32; context_lens: [B] or [B, next_n] int32;
    block_tables: [B, max_blocks] int32. Returns [B*next_n, max_model_len] f32.
    """
    B, next_n, H, D = q_values.shape
    M = B * next_n
    block_size = kv_cache.shape[1]
    fill = float("-inf") if clean_logits else 0.0
    logits = torch.full(
        (M, max_model_len), fill, dtype=torch.float32, device=q_values.device
    )
    q_bf = q_values.to(torch.bfloat16).reshape(M, H, D)
    if context_lens.dim() == 2:
        ctx_per_batch = context_lens.amax(dim=-1)
    else:
        ctx_per_batch = context_lens
    cache_dim = kv_cache.shape[-1]
    assert cache_dim == D + 4, (
        f"kv_cache last dim {cache_dim} != D+4 ({D + 4}); layout mismatch"
    )
    for b in range(B):
        ctx = int(ctx_per_batch[b].item())
        if ctx <= 0:
            continue
        n_blocks = (ctx + block_size - 1) // block_size
        bt = block_tables[b, :n_blocks].to(torch.long)
        rows = kv_cache[bt].reshape(n_blocks * block_size, cache_dim)[:ctx].contiguous()
        k_bytes = rows[:, :D].contiguous()
        scale_bytes = rows[:, D : D + 4].contiguous()
        scales = scale_bytes.view(torch.float32).squeeze(-1).to(torch.bfloat16)
        k_bf = k_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16) * scales.unsqueeze(-1)
        m_start = b * next_n
        m_end = m_start + next_n
        scores = torch.einsum("mhd,nd->mhn", q_bf[m_start:m_end], k_bf)
        w_b = weights[m_start:m_end].to(torch.float32)
        logits[m_start:m_end, :ctx] = (
            w_b.unsqueeze(-1) * scores.to(torch.float32)
        ).sum(dim=1)
    return logits
