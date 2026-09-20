"""Small-batch BF16 linear, FP32 split-K partials and reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    steps: tl.constexpr = (K + BK * SPLIT - 1) // (BK * SPLIT)
    k = split * steps * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(steps):
        x = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + k[:, None],
                    (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
        k += BK
    tl.store(OUT + split * M * N + m[:, None] * N + n[None, :],
             acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(PARTIAL, OUT, SIZE: tl.constexpr, SPLIT: tl.constexpr,
            BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT)
    values = tl.load(PARTIAL + split[:, None] * SIZE + col[None, :],
                     col[None, :] < SIZE, 0)
    tl.store(OUT + col, tl.sum(values, 0), col < SIZE)


def linear(x, weight, split=4):
    shape = x.shape
    m = x.numel() // shape[-1]
    n, k = weight.shape
    output = torch.empty((*shape[:-1], n), dtype=x.dtype, device=x.device)
    partial = output if split == 1 else torch.empty(
        (split, m, n), dtype=torch.float32, device=x.device,
    )
    _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64), split)](
        x, weight, partial, m, n, k, split, 16, 64, 64,
        num_warps=4, num_stages=3,
    )
    if split != 1:
        _reduce[(triton.cdiv(m * n, 256),)](
            partial, output, m * n, split, 256, num_warps=4,
        )
    return output
