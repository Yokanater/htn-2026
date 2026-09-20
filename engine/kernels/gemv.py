"""Skinny GEMMs for decode (M = batch rows <= 64) with fused epilogues.

  out = x @ W.T                          EPI_NONE
  out = res + bf16(x @ W.T)              EPI_RES    (residual add, BF16 like the reference)
  out = bf16(silu(bf16(g))) * bf16(u)    EPI_SILU   (g, u = x @ Wg.T, x @ Wu.T from a packed [2I, K] W)
  tok = argmax(bf16(x @ W.T))            lm_head_argmax (lowest index on exact ties)

Accumulation is fp32 (tensor-core dot), split-K partials are summed in a fixed order; only the
reduction order differs from cuBLAS. Every BF16 rounding point of the reference is kept.
Configs are tuned per (shape, batch) at warmup by timing a few candidates.
"""

import torch
import triton
import triton.language as tl
try:
    from triton.language.extra import libdevice
except ImportError:  # older layout
    from triton.language.extra.cuda import libdevice

EPI_NONE, EPI_RES, EPI_SILU = 0, 1, 2


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _row_rstd(x_base, m_ok, K, eps, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Qwen3RMSNorm's rsqrt(mean(x*x) + eps) per row, fp32 (reduction order differs only)."""
    rk = tl.arange(0, BLOCK_K)
    ss = tl.zeros([BLOCK_M], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_base + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    return tl.math.rsqrt(ss / K + eps)


@triton.jit
def _normed(x, rstd, nw_ptr, k):
    """weight * bf16(x * rstd), rounded to BF16: exactly the reference's cast placement."""
    xn = (x.to(tl.float32) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(nw_ptr + k).to(tl.float32)
    return (xn * w[None, :]).to(tl.bfloat16)


@triton.jit
def _epilogue(acc, acc2, res_ptrs, out_ptrs, mask, EPI: tl.constexpr):
    if EPI == 1:
        y = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32) + _bf(acc)
    elif EPI == 2:
        g = _bf(acc)
        y = _bf(tl.math.div_rn(g, 1.0 + libdevice.exp(-g))) * _bf(acc2)
    else:
        y = acc
    tl.store(out_ptrs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, out_ptr, res_ptr, part_ptr, nw_ptr, eps,
                 M, N, K, stride_x, stride_w, stride_out, stride_res, k_chunk, pair_off,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 EPI: tl.constexpr, SPLIT: tl.constexpr, NORM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    n_ok = rn < N
    k_lo = pid_k * k_chunk
    k_hi = tl.minimum(k_lo + k_chunk, K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    acc2 = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    x_base = x_ptr + rm[:, None] * stride_x
    w_base = w_ptr + rn[:, None] * stride_w
    if NORM:
        rstd = _row_rstd(x_base, m_ok, K, eps, BLOCK_M, BLOCK_K)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        k = k0 + rk
        x = tl.load(x_base + k[None, :], mask=m_ok[:, None], other=0.0)
        if NORM:
            x = _normed(x, rstd, nw_ptr, k)
        w = tl.load(w_base + k[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        if EPI == 2:
            w2 = tl.load(w_base + pair_off * stride_w + k[None, :], mask=n_ok[:, None], other=0.0)
            acc2 = tl.dot(x, tl.trans(w2), acc2)
    if SPLIT:
        mask = m_ok[:, None] & n_ok[None, :]
        pp = part_ptr + (pid_k * M + rm[:, None]) * (2 * N) + rn[None, :]
        tl.store(pp, acc, mask=mask)
        if EPI == 2:
            tl.store(pp + N, acc2, mask=mask)
    else:
        mask = m_ok[:, None] & n_ok[None, :]
        _epilogue(acc, acc2, res_ptr + rm[:, None] * stride_res + rn[None, :],
                  out_ptr + rm[:, None] * stride_out + rn[None, :], mask, EPI)


@triton.jit
def _splitk_reduce_kernel(part_ptr, out_ptr, res_ptr, M, N, n_split, stride_out, stride_res,
                          BLOCK_N: tl.constexpr, EPI: tl.constexpr):
    m = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = rn < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    acc2 = tl.zeros([BLOCK_N], tl.float32)
    for s in range(0, n_split):
        pp = part_ptr + (s * M + m) * (2 * N) + rn
        acc += tl.load(pp, mask=n_ok, other=0.0)
        if EPI == 2:
            acc2 += tl.load(pp + N, mask=n_ok, other=0.0)
    _epilogue(acc, acc2, res_ptr + m * stride_res + rn, out_ptr + m * stride_out + rn, n_ok, EPI)


@triton.jit
def _lm_argmax_partial_kernel(x_ptr, w_ptr, pval_ptr, pidx_ptr, nw_ptr, eps, M, N, K, stride_x, stride_w,
                              n_tiles, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                              NORM: tl.constexpr):
    pid = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    x_base = x_ptr + rm[:, None] * stride_x
    if NORM:
        rstd = _row_rstd(x_base, m_ok, K, eps, BLOCK_M, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        x = tl.load(x_base + k[None, :], mask=m_ok[:, None], other=0.0)
        if NORM:
            x = _normed(x, rstd, nw_ptr, k)
        w = tl.load(w_ptr + rn[:, None] * stride_w + k[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    logits = tl.where(n_ok[None, :], _bf(acc), float("-inf"))   # the reference's logits are BF16
    best = tl.max(logits, axis=1)
    idx = tl.min(tl.where(logits == best[:, None], rn[None, :], 2147483647), axis=1)
    tl.store(pval_ptr + rm * n_tiles + pid, best, mask=m_ok)
    tl.store(pidx_ptr + rm * n_tiles + pid, idx, mask=m_ok)


@triton.jit
def _lm_argmax_reduce_kernel(pval_ptr, pidx_ptr, tok_ptr, n_tiles, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    ok = r < n_tiles
    v = tl.load(pval_ptr + m * n_tiles + r, mask=ok, other=float("-inf"))
    i = tl.load(pidx_ptr + m * n_tiles + r, mask=ok, other=2147483647)
    best = tl.max(v, axis=0)
    tl.store(tok_ptr + m, tl.min(tl.where(v == best, i, 2147483647), axis=0).to(tl.int64))


def _block_m(M):
    return max(16, triton.next_power_of_2(M))


class Gemv:
    """out[M, N] (+epilogue) = x[M, K] @ w[N_w, K].T for a fixed M, with a tuned config."""

    CANDIDATES = [  # BLOCK_N, BLOCK_K, num_warps, num_stages
        (32, 128, 4, 4), (64, 128, 4, 4), (16, 256, 4, 3), (32, 256, 8, 3),
    ]

    def __init__(self, M, N, K, epi, device, sm_count, cfg=None, norm_eps=None):
        """norm_eps: fuse Qwen3RMSNorm of x (its weight passed per call) into the prologue."""
        self.M, self.N, self.K, self.epi = M, N, K, epi
        self.norm_eps = norm_eps
        self.device, self.sm = device, sm_count
        self.part = None
        self.set_config(cfg or self.CANDIDATES[0])

    def set_config(self, cfg):
        bn, bk, warps, stages = cfg
        self.cfg = cfg
        tiles = triton.cdiv(self.N, bn)
        split = max(1, min(8, round(2 * self.sm / tiles)))
        steps = triton.cdiv(self.K, bk)
        split = min(split, steps)
        chunk = triton.cdiv(steps, split) * bk
        self.split = triton.cdiv(self.K, chunk)
        self.chunk, self.tiles = chunk, tiles
        if self.split > 1:
            self.part = torch.empty((self.split, self.M, 2 * self.N), dtype=torch.float32, device=self.device)

    def __call__(self, x, w, out, res=None, norm_w=None):
        assert self.K % self.cfg[1] == 0 and x.stride(-1) == 1 and w.stride(-1) == 1
        assert (norm_w is not None) == (self.norm_eps is not None)
        bn, bk, warps, stages = self.cfg
        res_t = res if res is not None else out
        split = self.split > 1
        _gemv_kernel[(self.tiles, self.split)](
            x, w, out, res_t, self.part if split else out,
            norm_w if norm_w is not None else w, self.norm_eps or 0.0,
            self.M, self.N, self.K, x.stride(0), w.stride(0), out.stride(0), res_t.stride(0),
            self.chunk, self.N,
            BLOCK_M=_block_m(self.M), BLOCK_N=bn, BLOCK_K=bk, EPI=self.epi, SPLIT=split,
            NORM=self.norm_eps is not None, num_warps=warps, num_stages=stages,
        )
        if split:
            _splitk_reduce_kernel[(self.M, triton.cdiv(self.N, 256))](
                self.part, out, res_t, self.M, self.N, self.split, out.stride(0), res_t.stride(0),
                BLOCK_N=256, EPI=self.epi, num_warps=4,
            )
        return out

    def tune(self, x, w, out, res=None, norm_w=None, iters=20):
        """Time each candidate (compiles it) and keep the fastest valid one."""
        best = None
        for cfg in self.CANDIDATES:
            if self.K % cfg[1]:
                continue
            try:
                self.set_config(cfg)
                for _ in range(3):
                    self(x, w, out, res, norm_w)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    self(x, w, out, res, norm_w)
                e.record()
                e.synchronize()
                t = s.elapsed_time(e) / iters
                if best is None or t < best[0]:
                    best = (t, cfg)
            except Exception:
                continue
        if best is None:
            raise RuntimeError(f"no gemv config compiled for M={self.M} N={self.N} K={self.K}")
        self.set_config(best[1])
        return best


class LmHeadArgmax:
    """tok[M] = argmax over vocab of bf16(x[M, K] @ w[V, K].T); lowest index on exact ties."""

    CANDIDATES = [(64, 128, 4, 4), (128, 128, 8, 3), (32, 256, 4, 3)]

    def __init__(self, M, V, K, device, norm_eps=None):
        self.M, self.V, self.K, self.device = M, V, K, device
        self.norm_eps = norm_eps
        self.set_config(self.CANDIDATES[0])

    def set_config(self, cfg):
        self.cfg = cfg
        self.tiles = triton.cdiv(self.V, cfg[0])
        self.pval = torch.empty((self.M, self.tiles), dtype=torch.float32, device=self.device)
        self.pidx = torch.empty((self.M, self.tiles), dtype=torch.int32, device=self.device)

    def __call__(self, x, w, tok, norm_w=None):
        assert (norm_w is not None) == (self.norm_eps is not None)
        bn, bk, warps, stages = self.cfg
        _lm_argmax_partial_kernel[(self.tiles,)](
            x, w, self.pval, self.pidx, norm_w if norm_w is not None else w, self.norm_eps or 0.0,
            self.M, self.V, self.K, x.stride(0), w.stride(0), self.tiles,
            BLOCK_M=_block_m(self.M), BLOCK_N=bn, BLOCK_K=bk, NORM=self.norm_eps is not None,
            num_warps=warps, num_stages=stages,
        )
        _lm_argmax_reduce_kernel[(self.M,)](self.pval, self.pidx, tok, self.tiles,
                                            BLOCK=triton.next_power_of_2(self.tiles), num_warps=8)
        return tok

    def tune(self, x, w, tok, norm_w=None, iters=10):
        best = None
        for cfg in self.CANDIDATES:
            try:
                self.set_config(cfg)
                for _ in range(2):
                    self(x, w, tok, norm_w)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    self(x, w, tok, norm_w)
                e.record()
                e.synchronize()
                t = s.elapsed_time(e) / iters
                if best is None or t < best[0]:
                    best = (t, cfg)
            except Exception:
                continue
        if best is None:
            raise RuntimeError("no lm-head config compiled")
        self.set_config(best[1])
        return best
