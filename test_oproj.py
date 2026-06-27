#!/usr/bin/env python
"""Validate the fused bf16 inv-rope o-projection vs the old fp8 round-trip path.

new  = deepseek_v4_o_proj_inv_rope_bf16_sm86   (inv-rope bf16 -> bf16 einsum)
old  = fused_inv_rope_fp8_quant -> o_proj_bf16_sm86 (inv-rope -> fp8 -> bf16)
ref  = inv-rope + einsum in fp32 (ground truth)

Expect: new ~ old (within fp8 quant error => behavior preserved), and
new strictly closer to ref than old (the fp8 round-trip is pure loss).

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python test_oproj.py
"""
import torch

from vllm.model_executor.layers.deepseek_v4_attention import (
    deepseek_v4_o_proj_bf16_sm86,
    deepseek_v4_o_proj_inv_rope_bf16_sm86,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant import (
    _fused_inv_rope_fp8_quant_pyref,
)

DEV = "cuda"
T = 1            # decode
HG = 8           # heads per group
G = 1            # groups
H = G * HG
NOPE, ROPE = 512, 64
D = NOPE + ROPE  # 576
O_LORA = 512
QGS = 128


def inv_rope_fp32(o, positions, cache):
    o_r = o.reshape(T, G, HG, D).to(torch.float32)
    rope = o_r[..., NOPE:]
    re, ro = rope[..., 0::2], rope[..., 1::2]
    cs = cache[positions.long()]
    cv = cs[..., : ROPE // 2][:, None, None, :]
    sv = cs[..., ROPE // 2 :][:, None, None, :]
    ne = re * cv + ro * sv
    no = ro * cv - re * sv
    rot = torch.stack([ne, no], dim=-1).flatten(-2)
    of = o_r.clone()
    of[..., NOPE:] = rot
    return of.reshape(T, G, HG * D)  # fp32 [T,G,d]


def main():
    g = torch.Generator(device=DEV).manual_seed(0)
    o = torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16, generator=g)
    positions = torch.randint(0, 4096, (T,), device=DEV, dtype=torch.int64, generator=g)
    cache = torch.randn(8192, ROPE, device=DEV, dtype=torch.float32, generator=g)
    wo_a = torch.randn(G * O_LORA, HG * D, device=DEV, dtype=torch.bfloat16, generator=g) * 0.05

    # ref: fp32 inv-rope + fp32 einsum
    o_bgd_f32 = inv_rope_fp32(o, positions, cache)
    wo_a_3d_f32 = wo_a.to(torch.float32).view(G, O_LORA, HG * D)
    z_ref = torch.einsum("bhr,hdr->bhd", o_bgd_f32, wo_a_3d_f32)

    # new fused op
    z_new = torch.empty(T, G, O_LORA, device=DEV, dtype=torch.bfloat16)
    deepseek_v4_o_proj_inv_rope_bf16_sm86(
        o, positions, cache, wo_a, z_new, G, HG, NOPE, ROPE, O_LORA)

    # old path: inv-rope+fp8 quant, then bf16 o-proj (dequant + einsum)
    o_fp8, o_scale = _fused_inv_rope_fp8_quant_pyref(
        o, positions, cache, G, HG, NOPE, ROPE, QGS, tma_aligned_scales=False)
    z_old = torch.empty(T, G, O_LORA, device=DEV, dtype=torch.bfloat16)
    deepseek_v4_o_proj_bf16_sm86(o_fp8, o_scale, wo_a, z_old, G, O_LORA)

    def err(z):
        d = (z.float() - z_ref).abs()
        return d.max().item(), (d / z_ref.abs().clamp_min(1e-3)).mean().item()

    new_max, new_rel = err(z_new)
    old_max, old_rel = err(z_old)
    new_vs_old = (z_new.float() - z_old.float()).abs().max().item()
    print(f"z_ref scale: |mean|={z_ref.abs().mean():.4f} max={z_ref.abs().max():.4f}")
    print(f"new vs ref : max|Δ|={new_max:.4f}  mean_rel={new_rel:.4f}")
    print(f"old vs ref : max|Δ|={old_max:.4f}  mean_rel={old_rel:.4f}  (fp8 round-trip)")
    print(f"new vs old : max|Δ|={new_vs_old:.4f}  (behavior preserved if small)")
    better = new_max <= old_max and new_rel <= old_rel
    print(f"new more faithful than old: {better}")
    # behavior preserved: new within fp8 error of old; and new closer to truth
    ok = better and new_vs_old < 5 * old_max
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    print(f"torch {torch.__version__}  dev {torch.cuda.get_device_name(0)}")
    main()
