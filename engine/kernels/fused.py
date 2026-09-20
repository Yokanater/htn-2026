"""Small fused kernels that keep the reference's BF16 rounding points.

Every intermediate the reference materialises as a BF16 tensor is rounded to BF16 here too
(`_bf`), and fp32 math uses correctly-rounded division and libdevice exp, matching the
PyTorch CUDA kernels these replace.
"""

import torch
import triton
import triton.language as tl
try:
    from triton.language.extra import libdevice
except ImportError:  # older layout
    from triton.language.extra.cuda import libdevice


@triton.jit
def _bf(x):
    """Round an fp32 value to BF16 and back: the reference stores this intermediate in BF16."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    T, eps,
    stride_qkv,                       # row stride of qkv [B*T, 6144]
    stride_qb, stride_qh, stride_qt,  # q_out [B, 32, T, 128] strides
    stride_cb, stride_ch, stride_cn,  # cache [B, 8, cap, 128] strides
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, POS_MODE: tl.constexpr,
):
    """One program per (token row, head in [0, NQ + 2*NKV)).
    Position of token t of sequence b: POS_MODE 0 -> t, 1 -> pos[0] + t, 2 -> pos[b] + t."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    b = row // T
    t = row % T
    if POS_MODE == 2:
        pos = tl.load(pos_ptr + b).to(tl.int32) + t
    elif POS_MODE == 1:
        pos = tl.load(pos_ptr).to(tl.int32) + t
    else:
        pos = t
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, HALF)
    src = qkv_ptr + row * stride_qkv + head * D

    if head >= NQ + NKV:                                   # V: copy into the cache as is
        vh = head - NQ - NKV
        dst = v_cache_ptr + b * stride_cb + vh * stride_ch + pos * stride_cn
        tl.store(dst + d, tl.load(src + d))
        tl.store(dst + HALF + d, tl.load(src + HALF + d))
    else:
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + HALF + d).to(tl.float32)
        var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
        r = tl.math.rsqrt(var + eps)
        if head < NQ:
            w_ptr = qn_ptr
        else:
            w_ptr = kn_ptr
        w1 = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
        # Qwen3RMSNorm: weight * normed.to(bf16)  -> BF16 tensor
        y1 = _bf(_bf(x1 * r) * w1)
        y2 = _bf(_bf(x2 * r) * w2)
        c1 = tl.load(cos_ptr + pos * D + d).to(tl.float32)
        c2 = tl.load(cos_ptr + pos * D + HALF + d).to(tl.float32)
        s1 = tl.load(sin_ptr + pos * D + d).to(tl.float32)
        s2 = tl.load(sin_ptr + pos * D + HALF + d).to(tl.float32)
        # q*cos + rotate_half(q)*sin, rotate_half(q) = cat(-q2, q1); each product is a BF16 tensor
        o1 = _bf(y1 * c1) + _bf(-y2 * s1)
        o2 = _bf(y2 * c2) + _bf(y1 * s2)
        if head < NQ:
            dst = q_out_ptr + b * stride_qb + head * stride_qh + t * stride_qt
        else:
            kh = head - NQ
            dst = k_cache_ptr + b * stride_cb + kh * stride_ch + pos * stride_cn
        tl.store(dst + d, o1.to(tl.bfloat16))
        tl.store(dst + HALF + d, o2.to(tl.bfloat16))


def qk_norm_rope_cache(qkv, q_norm_w, k_norm_w, cos, sin, pos, q_out, k_cache, v_cache, B, T, eps,
                       pos_mode: int, q_strides=None):
    """qkv [B*T, 6144] BF16 -> q_out normed + rotated, indexed (b, head, t) through q_strides
    (default: q_out is [B, 32, T, 128]); K normed + rotated and V written into the caches
    [B, 8, cap, 128] at position t (pos_mode 0), pos[0] + t (1) or pos[b] + t (2).
    cos/sin are the [cap, 128] BF16 tables from the reference rotary module."""
    nkv, D = k_cache.shape[1], k_cache.shape[3]
    nq = qkv.shape[1] // D - 2 * nkv
    sb, sh, st = q_strides or (q_out.stride(0), q_out.stride(1), q_out.stride(2))
    _qk_norm_rope_kernel[(B * T, nq + 2 * nkv)](
        qkv, q_norm_w, k_norm_w, cos, sin, pos, q_out, k_cache, v_cache,
        T, eps, qkv.stride(0), sb, sh, st,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        NQ=nq, NKV=nkv, D=D, POS_MODE=pos_mode, num_warps=1,
    )


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, inter, stride_gu, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < inter
    g = tl.load(gu_ptr + row * stride_gu + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * stride_gu + inter + cols, mask=m, other=0.0).to(tl.float32)
    # torch silu (opmath fp32): x / (1 + exp(-x)), stored BF16; then BF16 * BF16 -> BF16
    s = _bf(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))
    tl.store(out_ptr + row * inter + cols, (s * u).to(tl.bfloat16), mask=m)


def silu_mul(gu, inter, out=None):
    """gu [M, 2*inter] (gate | up) -> silu(gate) * up, [M, inter] BF16."""
    M = gu.shape[0]
    if out is None:
        out = torch.empty((M, inter), dtype=gu.dtype, device=gu.device)
    BLOCK = 1024
    _silu_mul_kernel[(M, triton.cdiv(inter, BLOCK))](gu, out, inter, gu.stride(0), BLOCK=BLOCK, num_warps=4)
    return out
