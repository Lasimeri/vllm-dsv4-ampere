"""K11: fused FlashMLA sparse-decode Triton kernel for SM86.

Replaces the per-(B,S) Python loop in _flash_mla_decode_pyref with a single
Triton kernel that streams FP8 K cache tiles, dequants inline (UE8M0 scale +
bf16 RoPE), runs online FlashAttention softmax, and writes [H_q, D_V] bf16
output plus [H_q] f32 LSE per batch row.

Decode-only: assumes seq_len_q == 1. Caller should fall back to pyref for
prefill / chunked-prefill where S_q > 1.

Approximation preserved from the pyref: Q is truncated from D_QK_FULL=576 to
D_V=512 to match the SM86 cache layout. Mainline FlashMLA pairs Q_pe[64] with
a separately-stored K_pe; the SM86 packed cache does not contain K_pe, so the
trailing 64 Q dims are dropped. This is the same accuracy compromise the pyref
already makes.
"""
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _fp8_e4m3fn_byte_to_bf16,
)


# Cache layout constants (must match pyref / K10).
NOPE_DIM = 448
BF16_DIM = 64
QUANT_BLOCK = 64
N_NOPE_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 7
TOKEN_DATA_SIZE = NOPE_DIM + BF16_DIM * 2  # 576
TOKEN_SCALE_DIM = 8  # 7 real + 1 pad
D_V_DEFAULT = 512

# Triton arange must be power-of-2. We pad NOPE 448 -> 512 and N_NOPE_BLOCKS
# 7 -> 8 with masked loads (extra positions read 0). The padded contributions
# to all dots are zero, so the math is unchanged.
NOPE_PADDED = 512
N_NOPE_BLOCKS_PAD = 8


@triton.jit
def _flash_mla_decode_sm86_kernel(
    Q_ptr,
    Q_b_stride, Q_h_stride,
    K_CACHE_ptr,
    K_block_stride,
    K_block_size,
    K_max_slot,
    IDX_ptr,
    IDX_b_stride,
    TOPK_LEN_ptr,
    EK_CACHE_ptr,
    EK_block_stride,
    EK_block_size,
    EK_max_slot,
    EIDX_ptr,
    EIDX_b_stride,
    ELEN_ptr,
    SINK_ptr,
    OUT_ptr,
    OUT_b_stride, OUT_h_stride,
    LSE_ptr,
    LSE_b_stride,
    softmax_scale,
    H_q,
    HAS_TOPK_LEN: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_E_LEN: tl.constexpr,
    HAS_SINK: tl.constexpr,
    TOPK_MAIN: tl.constexpr,
    TOPK_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NOPE_DIM_C: tl.constexpr,         # 448 (real)
    NOPE_PADDED_C: tl.constexpr,      # 512 (pow2 pad)
    BF16_DIM_C: tl.constexpr,         # 64
    QUANT_BLOCK_C: tl.constexpr,      # 64
    N_NOPE_BLOCKS_C: tl.constexpr,    # 7 (real)
    N_NOPE_BLOCKS_PAD_C: tl.constexpr, # 8 (pow2 pad)
    TOKEN_DATA_SIZE_C: tl.constexpr,  # 576
    TOKEN_SCALE_DIM_C: tl.constexpr,  # 8
):
    NEG_INF_SENT = -1.0e9

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H_q

    # ---- Load Q (NOPE first 448 + RoPE next 64; pad NOPE to 512) ----
    # nope_off goes 0..511; positions 448..511 zeroed via mask.
    nope_off = tl.arange(0, NOPE_PADDED_C)        # 512 pow2
    nope_real_mask = nope_off < NOPE_DIM_C        # True for 0..447
    bf_off = tl.arange(0, BF16_DIM_C)             # 64 pow2

    q_nope_ptrs = (Q_ptr + pid_b * Q_b_stride
                   + h_off[:, None] * Q_h_stride
                   + nope_off[None, :])
    q_nope = tl.load(
        q_nope_ptrs,
        mask=h_mask[:, None] & nope_real_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    q_bf_ptrs = (Q_ptr + pid_b * Q_b_stride
                 + h_off[:, None] * Q_h_stride
                 + (NOPE_DIM_C + bf_off[None, :]))
    q_bf = tl.load(q_bf_ptrs, mask=h_mask[:, None], other=0.0).to(tl.bfloat16)

    # ---- Online softmax state ----
    m_i = tl.full([BLOCK_H], NEG_INF_SENT, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_nope = tl.zeros([BLOCK_H, NOPE_PADDED_C], dtype=tl.float32)
    acc_bf = tl.zeros([BLOCK_H, BF16_DIM_C], dtype=tl.float32)

    if HAS_TOPK_LEN:
        valid_main = tl.load(TOPK_LEN_ptr + pid_b).to(tl.int32)
    else:
        valid_main = TOPK_MAIN

    # 8 scales (7 valid + 1 pad). 8 is pow2.
    scale_off_8 = tl.arange(0, N_NOPE_BLOCKS_PAD_C)

    # ---- Main K loop ----
    for k_start in range(0, TOPK_MAIN, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_topk_mask = k_off < TOPK_MAIN

        slot_ptrs = IDX_ptr + pid_b * IDX_b_stride + k_off
        slots = tl.load(slot_ptrs, mask=k_topk_mask, other=-1).to(tl.int64)
        in_prefix = k_off < valid_main
        valid = in_prefix & k_topk_mask & (slots >= 0) & (slots < K_max_slot)
        safe_slot = tl.where(valid, slots, tl.zeros_like(slots))

        block_idx = safe_slot // K_block_size
        pos = safe_slot % K_block_size
        cache_block_base = block_idx * K_block_stride.to(tl.int64)
        data_base = cache_block_base + pos * TOKEN_DATA_SIZE_C
        scale_base = (cache_block_base + K_block_size * TOKEN_DATA_SIZE_C
                      + pos * TOKEN_SCALE_DIM_C)

        # NOPE 448 fp8 bytes (load 512 padded; 448..511 read 0)
        fp8_byte_ptrs = K_CACHE_ptr + data_base[:, None] + nope_off[None, :]
        fp8_bytes = tl.load(
            fp8_byte_ptrs,
            mask=valid[:, None] & nope_real_mask[None, :],
            other=0,
        )
        fp8_bf = _fp8_e4m3fn_byte_to_bf16(fp8_bytes)  # [BLOCK_K, 512] bf16

        # 8 UE8M0 scales (7 real + 1 pad). Pad-group fp8 values are 0,
        # so scale * 0 = 0 regardless of pad scale value.
        s_byte_ptrs = K_CACHE_ptr + scale_base[:, None] + scale_off_8[None, :]
        s_bytes = tl.load(s_byte_ptrs, mask=valid[:, None], other=127)
        scales_bf = tl.exp2(s_bytes.to(tl.float32) - 127.0).to(tl.bfloat16)

        # Apply scales in bf16: reshape (BLOCK_K, 8, 64) * (BLOCK_K, 8, 1)
        fp8_3d = tl.reshape(
            fp8_bf, (BLOCK_K, N_NOPE_BLOCKS_PAD_C, QUANT_BLOCK_C)
        )
        fp8_scaled = fp8_3d * scales_bf[:, :, None]
        kn_bf = tl.reshape(fp8_scaled, (BLOCK_K, NOPE_PADDED_C))

        # 64 bf16 from offset NOPE_DIM (byte-pair bitcast)
        bf_byte_lo = K_CACHE_ptr + data_base[:, None] + (NOPE_DIM_C + bf_off[None, :] * 2)
        bf_byte_hi = bf_byte_lo + 1
        bf_lo = tl.load(bf_byte_lo, mask=valid[:, None], other=0).to(tl.int32)
        bf_hi = tl.load(bf_byte_hi, mask=valid[:, None], other=0).to(tl.int32)
        bf_bits = ((bf_hi << 8) | bf_lo).to(tl.int16)
        bf_bf16 = bf_bits.to(tl.bfloat16, bitcast=True)

        # logits = (Q_nope @ K_nope^T + Q_bf @ K_bf^T) * sm_scale
        logits_nope = tl.dot(q_nope, tl.trans(kn_bf), out_dtype=tl.float32)
        logits_bf = tl.dot(q_bf, tl.trans(bf_bf16), out_dtype=tl.float32)
        s = (logits_nope + logits_bf) * softmax_scale
        s = tl.where(valid[None, :], s, NEG_INF_SENT)

        # Online softmax update
        m_row = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_row)
        alpha = tl.exp(m_i - m_new)
        p_raw = tl.exp(s - m_new[:, None])
        p = tl.where(valid[None, :], p_raw, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_bf = p.to(tl.bfloat16)
        acc_nope = (acc_nope * alpha[:, None]
                    + tl.dot(p_bf, kn_bf, out_dtype=tl.float32))
        acc_bf = (acc_bf * alpha[:, None]
                  + tl.dot(p_bf, bf_bf16, out_dtype=tl.float32))
        m_i = m_new

    # ---- Extra K loop ----
    if HAS_EXTRA:
        if HAS_E_LEN:
            valid_extra = tl.load(ELEN_ptr + pid_b).to(tl.int32)
        else:
            valid_extra = TOPK_EXTRA

        for k_start in range(0, TOPK_EXTRA, BLOCK_K):
            k_off = k_start + tl.arange(0, BLOCK_K)
            k_topk_mask = k_off < TOPK_EXTRA

            slot_ptrs = EIDX_ptr + pid_b * EIDX_b_stride + k_off
            slots = tl.load(slot_ptrs, mask=k_topk_mask, other=-1).to(tl.int64)
            in_prefix = k_off < valid_extra
            valid = in_prefix & k_topk_mask & (slots >= 0) & (slots < EK_max_slot)
            safe_slot = tl.where(valid, slots, tl.zeros_like(slots))

            block_idx = safe_slot // EK_block_size
            pos = safe_slot % EK_block_size
            cache_block_base = block_idx * EK_block_stride.to(tl.int64)
            data_base = cache_block_base + pos * TOKEN_DATA_SIZE_C
            scale_base = (cache_block_base + EK_block_size * TOKEN_DATA_SIZE_C
                          + pos * TOKEN_SCALE_DIM_C)

            fp8_byte_ptrs = EK_CACHE_ptr + data_base[:, None] + nope_off[None, :]
            fp8_bytes = tl.load(
                fp8_byte_ptrs,
                mask=valid[:, None] & nope_real_mask[None, :],
                other=0,
            )
            fp8_bf = _fp8_e4m3fn_byte_to_bf16(fp8_bytes)

            s_byte_ptrs = EK_CACHE_ptr + scale_base[:, None] + scale_off_8[None, :]
            s_bytes = tl.load(s_byte_ptrs, mask=valid[:, None], other=127)
            scales_bf = tl.exp2(s_bytes.to(tl.float32) - 127.0).to(tl.bfloat16)

            fp8_3d = tl.reshape(
                fp8_bf, (BLOCK_K, N_NOPE_BLOCKS_PAD_C, QUANT_BLOCK_C)
            )
            fp8_scaled = fp8_3d * scales_bf[:, :, None]
            kn_bf = tl.reshape(fp8_scaled, (BLOCK_K, NOPE_PADDED_C))

            bf_byte_lo = EK_CACHE_ptr + data_base[:, None] + (NOPE_DIM_C + bf_off[None, :] * 2)
            bf_byte_hi = bf_byte_lo + 1
            bf_lo = tl.load(bf_byte_lo, mask=valid[:, None], other=0).to(tl.int32)
            bf_hi = tl.load(bf_byte_hi, mask=valid[:, None], other=0).to(tl.int32)
            bf_bits = ((bf_hi << 8) | bf_lo).to(tl.int16)
            bf_bf16 = bf_bits.to(tl.bfloat16, bitcast=True)

            logits_nope = tl.dot(q_nope, tl.trans(kn_bf), out_dtype=tl.float32)
            logits_bf = tl.dot(q_bf, tl.trans(bf_bf16), out_dtype=tl.float32)
            s = (logits_nope + logits_bf) * softmax_scale
            s = tl.where(valid[None, :], s, NEG_INF_SENT)

            m_row = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_row)
            alpha = tl.exp(m_i - m_new)
            p_raw = tl.exp(s - m_new[:, None])
            p = tl.where(valid[None, :], p_raw, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            p_bf = p.to(tl.bfloat16)
            acc_nope = (acc_nope * alpha[:, None]
                        + tl.dot(p_bf, kn_bf, out_dtype=tl.float32))
            acc_bf = (acc_bf * alpha[:, None]
                      + tl.dot(p_bf, bf_bf16, out_dtype=tl.float32))
            m_i = m_new

    # ---- attn_sink: l_i += exp(sink_h - m_i) ----
    if HAS_SINK:
        sink_vals = tl.load(SINK_ptr + h_off, mask=h_mask, other=NEG_INF_SENT)
        sink_term = tl.exp(sink_vals - m_i)
        l_i = l_i + sink_term

    # ---- Finalize ----
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out_nope = (acc_nope / l_safe[:, None]).to(tl.bfloat16)
    out_bf = (acc_bf / l_safe[:, None]).to(tl.bfloat16)
    lse = m_i + tl.log(l_safe)

    # Store first 448 of acc_nope (last 64 always 0, mask them off)
    out_nope_ptrs = (OUT_ptr + pid_b * OUT_b_stride
                     + h_off[:, None] * OUT_h_stride
                     + nope_off[None, :])
    tl.store(
        out_nope_ptrs, out_nope,
        mask=h_mask[:, None] & nope_real_mask[None, :],
    )
    # Store 64 bf at offsets 448..511
    out_bf_ptrs = (OUT_ptr + pid_b * OUT_b_stride
                   + h_off[:, None] * OUT_h_stride
                   + (NOPE_DIM_C + bf_off[None, :]))
    tl.store(out_bf_ptrs, out_bf, mask=h_mask[:, None])

    lse_ptrs = LSE_ptr + pid_b * LSE_b_stride + h_off
    tl.store(lse_ptrs, lse, mask=h_mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    # tl.dot requires M >= 16 on the input tile. Cap at 16 to maximise
    # H-parallelism on small SM counts (RTX 3080 = 68 SMs).
    return min(max(p, 16), 16)


def _round_up(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k


def _flash_mla_decode_sm86_triton(
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
    """K11: fused FlashMLA sparse decode on SM86. Decode-only (S_q == 1)."""
    B, S_q, H_q, D_QK_FULL = q.shape
    if S_q != 1:
        raise RuntimeError("k11: S_q must be 1; caller should fall back to pyref")
    if head_dim_v != D_V_DEFAULT:
        raise RuntimeError(f"k11: D_V must be 512, got {head_dim_v}")

    D_V = head_dim_v

    # Normalize indices to [B, topk_main]
    if indices.dim() == 4:
        indices_n = indices[:, 0, 0, :]
    elif indices.dim() == 3:
        indices_n = indices[:, 0, :]
    else:
        indices_n = indices
    indices_n = indices_n.contiguous().to(torch.int32)
    if indices_n.shape[0] != B:
        if indices_n.shape[0] == 1:
            indices_n = indices_n.expand(B, indices_n.shape[-1]).contiguous()
        else:
            raise RuntimeError(
                f"k11: indices shape {indices_n.shape} incompatible with B={B}"
            )
    topk_main = indices_n.shape[-1]

    num_blocks, k_block_size, num_kv_heads, head_bytes = k_cache.shape
    k_max_slot = num_blocks * k_block_size
    k_block_stride = k_cache.stride(0)

    # Extras
    has_extra = (extra_k_cache is not None and extra_indices_in_kvcache is not None)
    if has_extra:
        eidx = extra_indices_in_kvcache
        if eidx.dim() == 4:
            eidx = eidx[:, 0, 0, :]
        elif eidx.dim() == 3:
            eidx = eidx[:, 0, :]
        eidx = eidx.contiguous().to(torch.int32)
        if eidx.shape[0] != B:
            if eidx.shape[0] == 1:
                eidx = eidx.expand(B, eidx.shape[-1]).contiguous()
            else:
                raise RuntimeError(
                    f"k11: extra indices shape {eidx.shape} incompatible with B={B}"
                )
        topk_extra = eidx.shape[-1]
        e_num_blocks, ek_block_size, _, _ = extra_k_cache.shape
        ek_max_slot = e_num_blocks * ek_block_size
        ek_block_stride = extra_k_cache.stride(0)
    else:
        eidx = indices_n  # placeholder, unused via constexpr
        topk_extra = 64
        ek_block_size = 1
        ek_block_stride = 1
        ek_max_slot = 0

    has_sink = attn_sink is not None
    has_topk_len = topk_length is not None
    has_e_len = has_extra and (extra_topk_length is not None)

    # Always pass valid pointers (gated by constexpr; but Triton compiles the
    # parameter type regardless).
    sink_arg = (attn_sink if has_sink
                else torch.zeros(H_q, dtype=torch.float32, device=q.device))
    tlen_arg = (topk_length if has_topk_len
                else torch.zeros(B, dtype=torch.int32, device=q.device))
    elen_arg = (extra_topk_length if has_e_len
                else torch.zeros(B, dtype=torch.int32, device=q.device))
    ek_arg = extra_k_cache if has_extra else k_cache

    # Reshape views: [B, S_q=1, H, D] -> [B, H, D]
    q_view = q.reshape(B, H_q, D_QK_FULL)
    out_view = out.reshape(B, H_q, D_V)
    lse = torch.empty(B, H_q, dtype=torch.float32, device=q.device)

    BLOCK_H = _next_pow2(H_q)
    # BLOCK_K must be pow2 and at least 16 for tl.dot. 32 keeps shared-mem
    # footprint under SM_86's 100 KB limit (the [BLOCK_K, 512] bf16 tile is
    # the dominant transient; 64 -> 128 KB exceeds, 32 -> 32 KB fits).
    BLOCK_K = 32

    # Round topk to multiple of BLOCK_K so the constexpr loop covers it
    TOPK_MAIN_C = max(_round_up(topk_main, BLOCK_K), BLOCK_K)
    TOPK_EXTRA_C = max(_round_up(topk_extra, BLOCK_K), BLOCK_K)

    # Pad indices if rounded up
    if topk_main < TOPK_MAIN_C:
        pad = torch.full((B, TOPK_MAIN_C - topk_main), -1,
                         dtype=torch.int32, device=q.device)
        indices_n = torch.cat([indices_n, pad], dim=-1)
    if has_extra and topk_extra < TOPK_EXTRA_C:
        pad = torch.full((B, TOPK_EXTRA_C - topk_extra), -1,
                         dtype=torch.int32, device=q.device)
        eidx = torch.cat([eidx, pad], dim=-1)

    grid = (B, triton.cdiv(H_q, BLOCK_H))

    _flash_mla_decode_sm86_kernel[grid](
        q_view,
        q_view.stride(0), q_view.stride(1),
        k_cache,
        k_block_stride,
        k_block_size,
        k_max_slot,
        indices_n,
        indices_n.stride(0),
        tlen_arg,
        ek_arg,
        ek_block_stride,
        ek_block_size,
        ek_max_slot,
        eidx,
        eidx.stride(0),
        elen_arg,
        sink_arg,
        out_view,
        out_view.stride(0), out_view.stride(1),
        lse,
        lse.stride(0),
        softmax_scale,
        H_q,
        HAS_TOPK_LEN=has_topk_len,
        HAS_EXTRA=has_extra,
        HAS_E_LEN=has_e_len,
        HAS_SINK=has_sink,
        TOPK_MAIN=TOPK_MAIN_C,
        TOPK_EXTRA=TOPK_EXTRA_C,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        NOPE_DIM_C=NOPE_DIM,
        NOPE_PADDED_C=NOPE_PADDED,
        BF16_DIM_C=BF16_DIM,
        QUANT_BLOCK_C=QUANT_BLOCK,
        N_NOPE_BLOCKS_C=N_NOPE_BLOCKS,
        N_NOPE_BLOCKS_PAD_C=N_NOPE_BLOCKS_PAD,
        TOKEN_DATA_SIZE_C=TOKEN_DATA_SIZE,
        TOKEN_SCALE_DIM_C=TOKEN_SCALE_DIM,
        num_warps=4,
    )

    # Pyref returns lse with shape [B, H_q, S_q]. We have [B, H_q]; add S_q dim.
    return out, lse.unsqueeze(-1)


def _k11_enabled() -> bool:
    """K11 fused decode is OFF by default: parity is correct but single-program
    grid loses to cuBLAS at decode-shape GEMMs. Re-enable explicitly with
    VLLM_SM86_K11=1 once the split-K (FlashDecoding) variant lands."""
    forced = os.environ.get("VLLM_SM86_K11", "").strip()
    if forced in ("1", "true", "True"):
        return True
    return False
