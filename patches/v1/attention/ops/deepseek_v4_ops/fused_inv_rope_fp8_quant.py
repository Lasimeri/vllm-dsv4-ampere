# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused inverse RoPE + block-scaled FP8 quantization kernel for DeepseekV4 attention.

Output scale format is pre-transformed (MN-major TMA-aligned; FP32 on SM90,
INT32-packed UE8M0 on SM100) so fp8_einsum skips transform_sf_into_required_layout.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def _fused_inv_rope_fp8_quant_pyref(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SM86 reference for the fused inv-RoPE + per-block FP8 quant op.

    Mirrors the Triton kernel:
      • Inverse RoPE on last rope_dim slots of each head:
            out_even[k] = x_even[k] * cos[k] + x_odd[k] * sin[k]
            out_odd[k]  = x_odd[k]  * cos[k] - x_even[k] * sin[k]
      • Block absmax scaling per quant_group_size, rounded to power of 2
        (E8M0). Cast to fp8_e4m3fn.
      • Output buffers laid out (G, T_aligned, d) strided, then transposed
        to return (T, G, d) views — matches the kernel's output contract.
    """
    from vllm.utils.deep_gemm import get_tma_aligned_size

    T, H, D = o.shape
    G, HG = n_groups, heads_per_group
    assert H == G * HG
    assert D == nope_dim + rope_dim
    assert rope_dim % 2 == 0

    o_r = o.reshape(T, G, HG, D).to(torch.float32)
    rope = o_r[..., nope_dim:]
    rope_e = rope[..., 0::2]
    rope_o = rope[..., 1::2]

    cs = cos_sin_cache[positions.long()]
    cos_v = cs[..., : rope_dim // 2][:, None, None, :]
    sin_v = cs[..., rope_dim // 2 :][:, None, None, :]

    new_e = rope_e * cos_v + rope_o * sin_v
    new_o = rope_o * cos_v - rope_e * sin_v
    rotated = torch.stack([new_e, new_o], dim=-1).flatten(-2)

    o_full = o_r.clone()
    o_full[..., nope_dim:] = rotated

    d = HG * D
    chunks = d // quant_group_size
    o_per_group = o_full.reshape(T, G, d)
    o_blocks = o_per_group.reshape(T, G, chunks, quant_group_size)

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    eps = 1e-10

    block_absmax = o_blocks.abs().amax(dim=-1).clamp_min(eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))  # [T, G, chunks]

    scales_exp = scales.unsqueeze(-1).expand(T, G, chunks, quant_group_size)
    quant = torch.clamp(o_blocks / scales_exp, -fp8_max, fp8_max)
    fp8_vals = quant.reshape(T, G, d).to(fp8_dtype)

    tma_T = get_tma_aligned_size(T, 4)
    fp8_buf = torch.empty(
        (G, tma_T, d), dtype=fp8_dtype, device=o.device
    )
    fp8_buf[:, :T] = fp8_vals.permute(1, 0, 2)
    if tma_T > T:
        fp8_buf[:, T:].zero_()
    out_fp8 = fp8_buf[:, :T].transpose(0, 1).contiguous()

    if tma_aligned_scales:
        packed_k = (chunks + 3) // 4
        scale_storage = torch.zeros(
            G * packed_k * tma_T, dtype=torch.int32, device=o.device
        )
        scale_buf = scale_storage.as_strided(
            (G, T, packed_k),
            (packed_k * tma_T, 1, tma_T),
        )
        scale_bits = scales.view(torch.int32)
        ue8m0 = (scale_bits >> 23) & 0xFF
        if chunks % 4 != 0:
            pad = 4 - (chunks % 4)
            ue8m0 = torch.cat(
                [ue8m0, torch.zeros(T, G, pad, dtype=ue8m0.dtype, device=ue8m0.device)],
                dim=-1,
            )
        ue8m0_grp = ue8m0.reshape(T, G, packed_k, 4)
        shifts = torch.tensor([0, 8, 16, 24], device=o.device, dtype=torch.int32)
        packed = (ue8m0_grp * (1 << shifts)).sum(dim=-1)
        scale_buf.copy_(packed.permute(1, 0, 2))
    else:
        scale_storage = torch.zeros(
            G * chunks * tma_T, dtype=torch.float32, device=o.device
        )
        scale_buf = scale_storage.as_strided(
            (G, T, chunks),
            (chunks * tma_T, 1, tma_T),
        )
        scale_buf.copy_(scales.permute(1, 0, 2))

    out_scale = scale_buf.transpose(0, 1)
    return out_fp8, out_scale


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + block-scaled FP8 quantization.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] with cos||sin.
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        tma_aligned_scales: Output INT32 packed UE8M0 for SM100 (True)
                            or FP32 for SM90 (False).

    Returns:
        o_fp8: [T, G, D] float8_e4m3fn, strides (D, T*D, 1).
        o_scale: Pre-transformed scale tensor for fp8_einsum.
    """
    try:
        from vllm.utils.deep_gemm import _use_sm86_reference
        if _use_sm86_reference():
            return _fused_inv_rope_fp8_quant_pyref(
                o, positions, cos_sin_cache,
                n_groups, heads_per_group,
                nope_dim, rope_dim, quant_group_size,
                tma_aligned_scales,
            )
    except Exception:
        pass

    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=fp8_dtype,
        device=o.device,
    )

    tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
    if tma_aligned_scales:
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_buf = torch.empty(
            n_groups * packed_sf_k * tma_aligned_T,
            dtype=torch.int32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, packed_sf_k),
            (packed_sf_k * tma_aligned_T, 1, tma_aligned_T),
        )
    else:
        scale_buf = torch.empty(
            n_groups * num_scale_blocks * tma_aligned_T,
            dtype=torch.float32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, num_scale_blocks),
            (num_scale_blocks * tma_aligned_T, 1, tma_aligned_T),
        )

    common_args = dict(
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_stages=1,
        launch_pdl=False,
    )

    grid = (tma_aligned_T, n_groups * heads_per_group)
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        **common_args,
        num_warps=1,
    )

    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)
