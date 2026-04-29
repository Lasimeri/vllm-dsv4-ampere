from typing import Optional, Tuple
import dataclasses
import os

import torch
import triton
import triton.language as tl

import vllm._flashmla_C
flash_mla_cuda = torch.ops._flashmla_C

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _fp8_e4m3fn_byte_to_bf16,
)


@triton.jit
def _dequant_fp8_kv_slot_kernel(
    cache_ptr,            # uint8 paged cache [num_blocks, block_stride] flat
    slot_indices_ptr,     # int64 [N] (caller pre-converted to int64 with mask handling)
    out_ptr,              # bf16 [N, 512]
    valid_mask_ptr,       # bool [N]
    N,
    block_size: tl.constexpr,
    block_stride: tl.constexpr,
    max_slot: tl.constexpr,
    NOPE_DIM: tl.constexpr,         # 448
    BF16_DIM: tl.constexpr,         # 64
    QUANT_BLOCK: tl.constexpr,      # 64
    N_NOPE_BLOCKS: tl.constexpr,    # 7
    TOKEN_DATA_SIZE: tl.constexpr,  # 576
    TOKEN_SCALE_DIM: tl.constexpr,  # 8
    OUT_DIM: tl.constexpr,          # 512
):
    """One program per K slot. Reads paged-cache bytes for the slot,
    dequants 448 fp8 to bf16 via byte-unpack + UE8M0 scale, concatenates
    the 64 bf16 stored portion, writes [512] bf16 row. Zeros invalid slots."""
    n_idx = tl.program_id(0)
    if n_idx >= N:
        return

    slot = tl.load(slot_indices_ptr + n_idx)
    is_valid = (slot >= 0) & (slot < max_slot)

    safe_slot = tl.where(is_valid, slot, 0)
    block_idx = safe_slot // block_size
    pos_in_block = safe_slot % block_size

    cache_block_base = block_idx.to(tl.int64) * block_stride
    data_base = cache_block_base + pos_in_block * TOKEN_DATA_SIZE
    scale_base = (
        cache_block_base
        + block_size * TOKEN_DATA_SIZE
        + pos_in_block * TOKEN_SCALE_DIM
    )

    # Load 8 scale bytes (7 real + 1 pad). Pad byte unused; scaling
    # below applies only to first 7 groups since NOPE has 7×64 dims.
    scale_offsets = tl.arange(0, 8)
    scale_bytes = tl.load(cache_ptr + scale_base + scale_offsets)

    # Convert UE8M0 byte → fp32 scale = 2^(byte-127).
    scales_f32_8 = tl.exp2(scale_bytes.to(tl.float32) - 127.0)

    # Output buffer for this row, fp32 then cast to bf16 at end.
    # NOPE_DIM = 7 * 64 = 448; pad to 8*64 = 512 with zero scaling.
    PADDED_NOPE: tl.constexpr = 8 * QUANT_BLOCK
    nope_offsets = tl.arange(0, PADDED_NOPE)
    nope_mask = nope_offsets < NOPE_DIM
    fp8_bytes = tl.load(cache_ptr + data_base + nope_offsets, mask=nope_mask, other=0)
    nope_bf16 = _fp8_e4m3fn_byte_to_bf16(fp8_bytes).to(tl.float32)

    # Apply per-block scale: reshape to [8, QUANT_BLOCK]; pad-block scale
    # is unused (zeroed via nope_mask anyway).
    nope_2d = tl.reshape(nope_bf16, (8, QUANT_BLOCK))
    scales_2d = tl.reshape(scales_f32_8, (8, 1))
    nope_scaled = nope_2d * scales_2d
    nope_padded_flat = tl.reshape(nope_scaled, (PADDED_NOPE,))

    # Load 64 bf16 values from offset NOPE_DIM (128 bytes = 64 bf16).
    bf16_offsets = tl.arange(0, BF16_DIM)
    bf16_byte_offsets = NOPE_DIM + bf16_offsets * 2  # 2 bytes per bf16
    # Load as i16 then bitcast.
    bf16_lo = tl.load(cache_ptr + data_base + bf16_byte_offsets).to(tl.int32)
    bf16_hi = tl.load(cache_ptr + data_base + bf16_byte_offsets + 1).to(tl.int32)
    bf16_bits = (bf16_hi << 8) | bf16_lo
    bf16_bits_i16 = bf16_bits.to(tl.int16)
    bf16_vals = bf16_bits_i16.to(tl.bfloat16, bitcast=True).to(tl.float32)

    # Apply valid mask (zero out invalid slots).
    valid_f = is_valid.to(tl.float32)
    nope_padded_flat = nope_padded_flat * valid_f
    bf16_vals = bf16_vals * valid_f

    # Write output: [NOPE_DIM] then [BF16_DIM] = 512 total. Padded rows
    # past NOPE_DIM are masked.
    out_row = out_ptr + n_idx * OUT_DIM
    tl.store(out_row + nope_offsets, nope_padded_flat.to(tl.bfloat16), mask=nope_mask)
    tl.store(out_row + NOPE_DIM + bf16_offsets, bf16_vals.to(tl.bfloat16))

    # Write valid mask.
    tl.store(valid_mask_ptr + n_idx, is_valid)


def _dequant_fp8_kv_slots_sm86_triton(
    cache: torch.Tensor,
    slot_indices: torch.Tensor,
    head_dim_v: int = 512,
    head_dim_rope: int = 64,
    num_groups: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM_86 Triton fast path for _dequant_fp8_kv_slots.
    Same contract as the pyref; assumes V4 packed cache layout."""
    NOPE_DIM = 448
    BF16_DIM = 64
    QUANT_BLOCK = 64
    N_NOPE_BLOCKS = NOPE_DIM // QUANT_BLOCK
    TOKEN_DATA_SIZE = NOPE_DIM + BF16_DIM * 2
    TOKEN_SCALE_DIM = 8
    OUT_DIM = NOPE_DIM + BF16_DIM

    num_blocks, block_size, num_kv_heads, head_bytes = cache.shape
    max_slot = num_blocks * block_size

    # CRITICAL: cache may have padding between blocks (alignment). Use the
    # actual memory stride between blocks, not the logical content size.
    # Per-byte address: cache_data + block_idx * cache.stride(0) + ...
    block_stride = cache.stride(0)

    slot_indices = slot_indices.contiguous().to(torch.int64)
    N = slot_indices.shape[0]
    if N == 0:
        out = torch.zeros((0, OUT_DIM), dtype=torch.bfloat16, device=cache.device)
        valid_mask = torch.zeros((0,), dtype=torch.bool, device=cache.device)
        return out, valid_mask

    out = torch.empty((N, OUT_DIM), dtype=torch.bfloat16, device=cache.device)
    valid_mask = torch.empty((N,), dtype=torch.bool, device=cache.device)

    _dequant_fp8_kv_slot_kernel[(N,)](
        cache,                            # pass original (kernel walks via raw byte offsets)
        slot_indices,
        out,
        valid_mask,
        N,
        block_size=block_size,
        block_stride=block_stride,
        max_slot=max_slot,
        NOPE_DIM=NOPE_DIM,
        BF16_DIM=BF16_DIM,
        QUANT_BLOCK=QUANT_BLOCK,
        N_NOPE_BLOCKS=N_NOPE_BLOCKS,
        TOKEN_DATA_SIZE=TOKEN_DATA_SIZE,
        TOKEN_SCALE_DIM=TOKEN_SCALE_DIM,
        OUT_DIM=OUT_DIM,
        num_warps=4,
    )
    return out, valid_mask


def _flashmla_use_sm86_reference() -> bool:
    """Force pure-PyTorch reference for FlashMLA's sparse decode/prefill.
    Auto-enabled on SM<90; envvar VLLM_SM86_DEEPSEEK_V4_REF override."""
    forced = os.environ.get("VLLM_SM86_DEEPSEEK_V4_REF", "").strip()
    if forced in ("0", "false", "False"):
        return False
    if forced in ("1", "true", "True"):
        return True
    try:
        return torch.cuda.get_device_capability(0) < (9, 0)
    except Exception:
        return False


def _dequant_fp8_kv_slots(
    cache: torch.Tensor,
    slot_indices: torch.Tensor,
    head_dim_v: int = 512,
    head_dim_rope: int = 64,
    num_groups: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequant flat-indexed K slots from V4 packed FP8 cache to bf16.

    V4 cache layout per block (block_stride bytes total):
      [0 : block_size * 576]  per-token data, 576 bytes each:
                                [0:448]  448 fp8_e4m3fn (7 groups of 64)
                                [448:576] 64 bf16 (128 bytes)
      [block_size * 576 : ...] per-token UE8M0 scales, 8 bytes each
                                (7 real + 1 padding)

    Returns:
      k_bf16:    [N, 512] bf16 (448 dequant fp8 ⊕ 64 bf16, zeroed for invalid)
      valid_mask:[N] bool
    """
    NOPE_DIM = 448
    BF16_DIM = 64
    QUANT_BLOCK = 64
    N_NOPE_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 7
    TOKEN_DATA_SIZE = NOPE_DIM + BF16_DIM * 2  # 576
    TOKEN_SCALE_DIM = 8

    num_blocks, block_size, num_kv_heads, head_bytes = cache.shape
    max_slot = num_blocks * block_size
    valid_mask = (slot_indices >= 0) & (slot_indices < max_slot)
    safe_idx = torch.where(
        valid_mask, slot_indices, torch.zeros_like(slot_indices)
    ).long()

    block_idx = safe_idx // block_size
    pos_in_block = safe_idx % block_size
    N = safe_idx.shape[0]
    device = cache.device

    cache_flat = cache.reshape(num_blocks, -1)  # [num_blocks, block_stride] uint8
    block_stride = cache_flat.shape[-1]
    cache_flat_1d = cache_flat.reshape(-1)

    arange_data = torch.arange(TOKEN_DATA_SIZE, device=device)
    arange_scale = torch.arange(N_NOPE_BLOCKS, device=device)

    data_base = block_idx * block_stride + pos_in_block * TOKEN_DATA_SIZE
    data_idx = (data_base.unsqueeze(-1) + arange_data).flatten()
    data_bytes = cache_flat_1d[data_idx].reshape(N, TOKEN_DATA_SIZE)

    scale_base = (
        block_idx * block_stride
        + block_size * TOKEN_DATA_SIZE
        + pos_in_block * TOKEN_SCALE_DIM
    )
    scale_idx = (scale_base.unsqueeze(-1) + arange_scale).flatten()
    scale_bytes = cache_flat_1d[scale_idx].reshape(N, N_NOPE_BLOCKS)

    fp8_bytes = data_bytes[:, :NOPE_DIM].contiguous()
    bf16_bytes = data_bytes[:, NOPE_DIM:NOPE_DIM + BF16_DIM * 2].contiguous()

    fp8_vals_bf = fp8_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16)
    bf16_vals = bf16_bytes.view(torch.bfloat16)
    scales_bf = torch.pow(
        2.0, scale_bytes.to(torch.float32) - 127.0
    ).to(torch.bfloat16)
    fp8_grp = fp8_vals_bf.view(N, N_NOPE_BLOCKS, QUANT_BLOCK) * scales_bf.unsqueeze(-1)
    nope_bf16 = fp8_grp.reshape(N, NOPE_DIM)

    out = torch.cat([nope_bf16, bf16_vals], dim=-1)
    out = out * valid_mask.unsqueeze(-1).to(torch.bfloat16)
    return out, valid_mask


def _flash_mla_decode_pyref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM86 reference for sparse-decode FlashMLA.

    q:       [B, S_q, H_q, D_qk] bf16
    indices: [B, S_q, topk] int32 (flat positions in k_cache)
    out:     [B, S_q, H_q, head_dim_v] bf16 (filled in-place)
    Returns: (out, lse)
    """
    B, S_q, H_q, D_qk = q.shape
    head_dim_rope = D_qk - head_dim_v
    out_view = out.reshape(B, S_q, H_q, head_dim_v)
    lse = torch.zeros(B, H_q, S_q, dtype=torch.float32, device=q.device)
    sink = (
        attn_sink.to(torch.float32) if attn_sink is not None else None
    )

    # Normalize indices shape to [B, S_q, topk]
    if indices.dim() == 4:
        indices = indices[:, :, 0, :]
    elif indices.dim() == 3 and indices.shape[1] != S_q and indices.shape[1] == 1:
        indices = indices.expand(B, S_q, indices.shape[-1])
    if extra_indices_in_kvcache is not None and extra_indices_in_kvcache.dim() == 4:
        extra_indices_in_kvcache = extra_indices_in_kvcache[:, :, 0, :]
    if (
        extra_indices_in_kvcache is not None
        and extra_indices_in_kvcache.dim() == 3
        and extra_indices_in_kvcache.shape[1] == 1
        and S_q != 1
    ):
        extra_indices_in_kvcache = extra_indices_in_kvcache.expand(
            B, S_q, extra_indices_in_kvcache.shape[-1]
        )

    for b in range(B):
        for s in range(S_q):
            swa_idx = indices[b, s] if indices.dim() == 3 else indices[b]
            swa_len = (
                int(topk_length[b].item())
                if topk_length is not None
                else swa_idx.shape[0]
            )
            swa_idx_v = swa_idx[:swa_len]
            try:
                k_swa, mask_swa = _dequant_fp8_kv_slots_sm86_triton(
                    k_cache, swa_idx_v, head_dim_v, head_dim_rope
                )
            except Exception:
                k_swa, mask_swa = _dequant_fp8_kv_slots(
                    k_cache, swa_idx_v, head_dim_v, head_dim_rope
                )

            if extra_k_cache is not None and extra_indices_in_kvcache is not None:
                ex_idx = (
                    extra_indices_in_kvcache[b, s]
                    if extra_indices_in_kvcache.dim() == 3
                    else extra_indices_in_kvcache[b]
                )
                ex_len = (
                    int(extra_topk_length[b].item())
                    if extra_topk_length is not None
                    else ex_idx.shape[0]
                )
                ex_idx_v = ex_idx[:ex_len]
                try:
                    k_ex, mask_ex = _dequant_fp8_kv_slots_sm86_triton(
                        extra_k_cache, ex_idx_v, head_dim_v, head_dim_rope
                    )
                except Exception:
                    k_ex, mask_ex = _dequant_fp8_kv_slots(
                        extra_k_cache, ex_idx_v, head_dim_v, head_dim_rope
                    )
                K = torch.cat([k_swa, k_ex], dim=0)
                mask = torch.cat([mask_swa, mask_ex], dim=0)
            else:
                K = k_swa
                mask = mask_swa

            M = K.shape[0]
            if M == 0:
                out_view[b, s].zero_()
                continue

            q_bs = q[b, s].to(torch.bfloat16)  # [H_q, D_qk]
            # Q-dim (576) > K-dim (512) for V4: dot only the matching prefix.
            # The trailing q_pe dims would normally pair with k_pe stored
            # separately; in the SM86 pyref we approximate by truncating.
            common = min(q_bs.shape[-1], K.shape[-1])
            logits = (q_bs[..., :common] @ K[..., :common].transpose(-1, -2)).to(
                torch.float32
            ) * softmax_scale
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

            max_logit = logits.amax(dim=-1, keepdim=True)
            max_logit = torch.where(
                torch.isinf(max_logit), torch.zeros_like(max_logit), max_logit
            )
            exp_logits = (logits - max_logit).exp()
            denom = exp_logits.sum(dim=-1, keepdim=True)
            if sink is not None:
                sink_term = (sink.view(H_q, 1) - max_logit).exp()
                denom = denom + sink_term
            probs = (exp_logits / denom).to(torch.bfloat16)

            V = K[:, :head_dim_v]
            out_bs = (probs @ V).to(out.dtype)
            out_view[b, s].copy_(out_bs)
            lse[b, :, s] = (max_logit.squeeze(-1) + denom.squeeze(-1).log()).to(
                torch.float32
            )

    return out, lse


def _flash_mla_prefill_pyref(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM86 reference for sparse-prefill FlashMLA.

    q:       [s_q, h_q, d_qk] bf16
    kv:      [s_kv, h_kv, d_qk] bf16  (already dequant by caller)
    indices: [s_q, h_kv, topk] int32 (positions in kv tensor)

    Returns: (out, max_logits, lse)
      out:        [s_q, h_q, d_v] bf16
      max_logits: [s_q, h_q] float32
      lse:        [s_q, h_q] float32
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv = kv.shape[0], kv.shape[1]
    assert h_kv == 1, "MLA: h_kv must be 1"
    if out is None:
        out = torch.empty(s_q, h_q, d_v, dtype=q.dtype, device=q.device)
    max_logits = torch.zeros(s_q, h_q, dtype=torch.float32, device=q.device)
    lse = torch.zeros(s_q, h_q, dtype=torch.float32, device=q.device)
    sink = attn_sink.to(torch.float32) if attn_sink is not None else None

    kv2 = kv[:, 0, :]  # [s_kv, d_qk]

    for t in range(s_q):
        idx = indices[t, 0]
        topk = idx.shape[0]
        if topk_length is not None:
            valid_n = int(topk_length[t].item())
        else:
            valid_n = topk
        idx_v = idx[:valid_n]
        valid_mask = (idx_v >= 0) & (idx_v < s_kv)
        safe_idx = torch.where(
            valid_mask, idx_v, torch.zeros_like(idx_v)
        ).long()

        K = kv2[safe_idx]  # [valid_n, d_qk] bf16
        K = K * valid_mask.unsqueeze(-1).to(K.dtype)

        q_t = q[t].to(torch.bfloat16)  # [h_q, d_qk]
        common = min(q_t.shape[-1], K.shape[-1])
        logits = (q_t[..., :common] @ K[..., :common].transpose(-1, -2)).to(
            torch.float32
        ) * sm_scale
        logits = logits.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))

        max_l = logits.amax(dim=-1)
        max_l_safe = torch.where(
            torch.isinf(max_l), torch.zeros_like(max_l), max_l
        )
        exp_logits = (logits - max_l_safe.unsqueeze(-1)).exp()
        denom = exp_logits.sum(dim=-1)
        probs = (exp_logits / denom.unsqueeze(-1)).to(torch.bfloat16)

        V = K[:, :d_v]
        out_t = (probs @ V).to(out.dtype)
        # attn_sink rescale: out *= exp(lse) / (exp(lse) + exp(attn_sink))
        lse_t = max_l_safe + denom.log()
        if sink is not None:
            sink_h = sink.view(h_q)
            scale = (lse_t.exp() / (lse_t.exp() + sink_h.exp())).to(out.dtype)
            out_t = out_t * scale.unsqueeze(-1)
        out[t].copy_(out_t)
        max_logits[t] = max_l_safe
        lse[t] = lse_t

    return out, max_logits, lse

@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None   # (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    num_splits: Optional[torch.Tensor] = None                # (1), dtype torch.int32.


def get_mla_metadata(
    *args,
    **kwargs
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        This function does not need any arguments, but we keep *args and **kwargs to be compatible with the old interface.

    Return:
        A tuple. Due to historical reasons, we return a tuple of (FlashMLASchedMeta, None) now. Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
                Different modes (including fp8/bf16, and sparsity) has different KV cache layouts. See comments below for details.
                The KV cache must be contiguously valid for sparse attention on sm100. Here "contiguously valid" means that every byte, from the very beginning of the KV cache, till the last byte in the KV cache, is valid memory address to visit (i.e. won't IMA). In other words, the KV cache could be a slice of a larger array, but cannot be a list of disjoint memory blocks.
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32. Can be None when sparse attention is used.
        cache_seqlens: (batch_size), torch.int32. Can be None when sparse attention is used.
        head_dim_v: Head_dim of v. Must be 512
        sched_meta: FlashMLASchedMeta, return by get_mla_metadata. You may reuse the same sched_meta across different invocations, but only when the tensor shapes and the values of cache_seqlens, topk_length, and extra_topk_length remain the same.
        num_splits_placeholder: must be "None" (to be compatible with the old interface).
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim_k).
        causal: bool. Whether to apply causal attention mask. Only valid for dense attention
        is_fp8_kvcache: bool.
        indices: (batch_size, seq_len_q, topk). KV indices when sparse attention is enabled.
                    Pay attention that indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * block_size + (the offset of token t among the page block),
                    where t is the k-th token of the j-th q-sequence in the i-th batch.
        attn_sink: Optional[torch.Tensor], (num_heads_q, ), torch.float32. If presented, the final output will be scaled by exp(lse) / (exp(lse) + exp(attn_sink)). Have no affect on the returned softmax_lse. +inf will cause the result to become 0.
        extra_k_cache and extra_indices_in_kvcache: If provided, will attend to these extra tokens in addition to those in k_cache and indices_in_kvcache. Their format requirements are the same as k_cache and indices_in_kvcache respectively.
        topk_length/extra_topk_length: (batch_size, ), torch.int32. If provided, only the leftmost topk_length indices will be processed. Useful when the actual topk for different queries are different so that we can save some computation, compared to masking.
        out: Optional pre-allocated output tensor with shape (batch_size, seq_len_q, num_heads_q, head_dim_v), same dtype as q, and contiguous. If provided, the result will be written into this buffer to avoid allocation. For dense attention, only num_heads_k == 1 (MLA) is supported.

    For DeepSeek V3, DeepSeek V3.1, and DeepSeek V3.2:
        head_dim should be 576 while head_dim_v should be 512.
        In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
            - The shape of the tensor `k_cache` is (num_blocks, page_block_size, num_heads_k, head_dim), and num_heads_k must be 1.
            - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
            - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
            - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    sched_meta = tile_scheduler_metadata
    indices_in_kvcache = indices
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"

    topk = indices_in_kvcache.shape[-1] if indices_in_kvcache is not None else None
    extra_k_page_block_size = extra_k_cache.shape[1] if extra_k_cache is not None else None
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # SM86 reference dispatch (sparse decode only — dense decode falls through).
    if topk is not None and _flashmla_use_sm86_reference():
        if out is None:
            out = torch.empty(
                q.shape[0], q.shape[1], q.shape[2], head_dim_v,
                dtype=q.dtype, device=q.device,
            )
        return _flash_mla_decode_pyref(
            q=q, k_cache=k_cache, head_dim_v=head_dim_v,
            indices=indices_in_kvcache, topk_length=topk_length,
            attn_sink=attn_sink, softmax_scale=softmax_scale,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            out=out,
        )

    if not sched_meta.have_initialized:
        # Sanity check. We only perform sanity check during the first invocation to save CPU time.
        if indices_in_kvcache is not None:
            assert not causal, "causal must be False when indices_in_kvcache is not None (i.e. sparse attention is enabled)"
            
        # Initialize the tile scheduler metadata during the first invocation.
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],

            causal,
            is_fp8_kvcache,
            topk,

            extra_k_page_block_size,
            extra_topk,
        )
    else:
        # Check whether the input arguments are consistent with sched_meta
        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + helper_msg
        assert sched_meta.config.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        assert sched_meta.config.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + helper_msg
        assert sched_meta.config.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        assert sched_meta.config.causal == causal, "sched_meta.config.causal must be equal to causal." + helper_msg
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." + helper_msg
        assert sched_meta.config.topk == topk, "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." + helper_msg
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." + helper_msg
        assert sched_meta.config.extra_topk == extra_topk, "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." + helper_msg

    if topk is not None:
        # Sparse attention
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True when sparse attention is enabled"
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, out
        )
    else:
        # Dense attention
        assert indices_in_kvcache is None and attn_sink is None and extra_k_cache is None and extra_indices_in_kvcache is None and topk_length is None and extra_topk_length is None, "indices_in_kvcache, attn_sink, extra_k_cache, extra_indices_in_kvcache, topk_length and extra_topk_length must be None when dense attention is used."
        assert block_table is not None and cache_seqlens is not None, "block_table and cache_seqlens must be provided when dense attention is used."
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.dense_decode_fwd(
            q, k_cache, head_dim_v,
            cache_seqlens, block_table,
            softmax_scale, causal,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            out
        )
    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits
    return (out, lse)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.
        out: optional pre-allocated output tensor with shape [s_q, h_q, d_v], bfloat16, contiguous on the last dim. If provided, the result will be written into this buffer to avoid allocation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    if _flashmla_use_sm86_reference():
        return _flash_mla_prefill_pyref(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
        )
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
    )
    return results


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if out is None:
        out = torch.empty(qo_total_len, num_qo_heads, head_dim_vo, device=q.device, dtype=q.dtype)
    if lse is None:
        # Make lse contiguous on seqlen dim
        lse = torch.empty(num_qo_heads, qo_total_len, device=q.device, dtype=torch.float32).T

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_fwd(
        workspace_buffer,
        q,
        k,
        v,
        cu_seqlens_qo,
        cu_seqlens_kv,
        out,
        lse,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return out, lse


def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    # TODO: fix bwd GQA
    if num_qo_heads != num_kv_heads:
        raise ValueError(f"SM100 bwd doesn't support GQA now. num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}.")

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if dq is None:
        dq = torch.empty(qo_total_len, num_qo_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.empty(kv_total_len, num_kv_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dv is None:
        dv = torch.empty(kv_total_len, num_kv_heads, head_dim_vo, device=q.device, dtype=q.dtype)

    max_seqlen_qo_aligned = (max_seqlen_qo + 7) // 8 * 8
    bs = cu_seqlens_qo.shape[0] - 1
    workspace_bytes = 0
    workspace_bytes += 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2  # sum_OdO and scaled_lse
    if num_qo_heads != num_kv_heads:
        workspace_bytes += 2 * kv_total_len * num_qo_heads * (head_dim_qk + head_dim_vo)  # dKV_acc
    workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_bwd(
        workspace_buffer,
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_qo,
        cu_seqlens_kv,
        dq,
        dk,
        dv,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return dq, dk, dv


class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        is_varlen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ):
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2], qkv[:, :, head_dim_qk * 2:],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:],
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )
