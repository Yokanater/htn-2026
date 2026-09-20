"""Qwen3's RMSNorm in Triton, written to match the reference exactly.

``rms_norm_rows`` normalises rows that may sit inside a wider packed row (the per-head
q/k norm reads 128-wide heads straight out of the packed qkv projection), writing a
contiguous output. Arithmetic is exactly Qwen3RMSNorm's: fp32 reduce, normalise in fp32,
cast to BF16, *then* multiply by the BF16 weight.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_rows_kernel(x_ptr, w_ptr, y_ptr, outer_stride, heads, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    src = (row // heads) * outer_stride + (row % heads) * n_cols + cols

    # The reference reduces in fp32 over the whole row. Masked lanes load as zero.
    x = tl.load(x_ptr + src, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)

    # Cast placement: the reference ends ``self.weight * hidden_states.to(input_dtype)``, so
    # the normalised value is rounded to BF16 *before* the weight multiply. Keeping the
    # product in fp32 is a different function and can move a logit past the tie margin.
    # BF16 x BF16 is computed as the exact fp32 product rounded once to BF16 (as torch does).
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = normed.to(tl.bfloat16).to(tl.float32) * weight
    tl.store(y_ptr + row * n_cols + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rms_norm_rows(x: torch.Tensor, weight: torch.Tensor, eps: float, heads: int = 1,
                  out: torch.Tensor | None = None) -> torch.Tensor:
    """RMSNorm of ``weight.numel()``-wide rows.

    ``x`` is [M, W] with unit last stride; each of its M rows holds ``heads`` consecutive
    rows of width N = weight.numel() starting at column 0 (W >= heads*N). Returns
    [M, heads, N] contiguous (or writes ``out`` of that many elements).
    """
    n_cols = weight.numel()
    m = x.shape[0]
    assert x.stride(-1) == 1
    if out is None:
        out = torch.empty((m, heads, n_cols), dtype=x.dtype, device=x.device)
    block = triton.next_power_of_2(n_cols)
    _rms_norm_rows_kernel[(m * heads,)](
        x, weight, out, x.stride(0), heads, n_cols, eps,
        BLOCK=block, num_warps=max(1, min(16, block // 256)),
    )
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, matching ``Qwen3RMSNorm.forward``."""
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    return rms_norm_rows(rows, weight, eps).reshape(shape)
