"""Residual/norm and SwiGLU fusions with native BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _residual_norm(X, BRANCH, W, RESIDUAL, NORMAL, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    mask = col < 2560
    offset = row * 2560 + col
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    branch = tl.load(BRANCH + offset, mask, 0).to(tl.float32)
    residual = (x + branch).to(tl.bfloat16)
    value = residual.to(tl.float32)
    variance = tl.sum(value * value, 0) / 2560
    normed = (value * tl.math.rsqrt(variance + EPS)).to(tl.bfloat16)
    weight = tl.load(W + col, mask, 0).to(tl.float32)
    normal = normed.to(tl.float32) * weight
    tl.store(RESIDUAL + offset, residual, mask)
    tl.store(NORMAL + offset, normal, mask)


def residual_norm(x, branch, norm):
    residual = torch.empty_like(x)
    normal = torch.empty_like(x)
    _residual_norm[(x.numel() // 2560,)](
        x, branch, norm.weight, residual, normal, norm.variance_epsilon,
        num_warps=16, enable_fp_fusion=False,
    )
    return residual, normal


@triton.jit
def _swiglu(PACKED, OUT, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // 9728
    col = index % 9728
    mask = row < ROWS
    gate = tl.load(PACKED + row * 19456 + col, mask, 0).to(tl.float32)
    up = tl.load(PACKED + row * 19456 + 9728 + col, mask, 0).to(tl.float32)
    # Match PyTorch's expf and division, then materialize the SiLU BF16 cast.
    activated = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16)
    tl.store(OUT + index, activated.to(tl.float32) * up, mask)


def swiglu(packed):
    rows = packed.numel() // 19456
    out = torch.empty((*packed.shape[:-1], 9728),
                      dtype=packed.dtype, device=packed.device)
    _swiglu[(triton.cdiv(rows * 9728, 256),)](
        packed, out, rows, 256, num_warps=4, enable_fp_fusion=False,
    )
    return out
