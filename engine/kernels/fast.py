"""T5 decode kernels: five launches per layer, no separate norm / reduce / merge / RoPE kernels.

  embed_ss      x = embed[tok]; ss[0] = sum(x^2) (per row); zero this step's other ss slots
  per layer:
    gemm qkv    rstd from ss (no prologue pass) -> BF16 normed x on load -> x @ Wqkv.T
    attn        q/k norm + RoPE + KV-cache write + split-K GQA flash decode + merge (last split)
    gemm o      x += BF16(attn @ Wo.T); ss_next += sum(x_new^2)          (residual epilogue)
    gemm gu     rstd from ss -> normed x -> silu(BF16 g) * BF16 u          (SwiGLU epilogue)
    gemm down   x += BF16(act @ Wd.T); ss_next += sum(x_new^2)
  lm_head       rstd from ss -> normed x -> BF16 logits -> argmax (lowest index); last CTA writes
                the token and advances the position

Split-K partials are reduced by the last-arriving CTA of each output tile (a scalar atomic ticket,
debug_barrier before it, L2 (.cg) loads after it) - no extra launch. RMSNorm statistics are exact
sums of squares of the BF16 residual stream accumulated by the kernel that produces it (fp32
atomics: order-dependent at the ulp level, like any reduction reorder). Every BF16 rounding point of
transformers 4.51.3 is kept.

Each kernel body is a `*_unit` device function over one work unit, shared with the persistent
megakernel (kernels/mega.py); CG=True makes activation loads bypass L1 (data produced by other CTAs
inside the same launch).
"""

import math

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
def _ld(ptrs, mask, other, CG: tl.constexpr):
    if CG:
        return tl.load(ptrs, mask=mask, other=other, cache_modifier=".cg")
    else:
        return tl.load(ptrs, mask=mask, other=other)


# ------------------------------------------------------------------------------------ embedding
@triton.jit
def _embed_unit(m, tok_ptr, emb_ptr, x_ptr, ss_ptr, M, H, n_slots, BLOCK_H: tl.constexpr):
    t = tl.load(tok_ptr + m)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        ok = h < H
        v = tl.load(emb_ptr + t * H + h, mask=ok, other=0.0)
        tl.store(x_ptr + m * H + h, v, mask=ok)
        vf = v.to(tl.float32)
        acc += vf * vf
    tl.store(ss_ptr + m, tl.sum(acc, axis=0))
    for s in range(1, n_slots):                               # this step's accumulators start at 0
        tl.store(ss_ptr + s * M + m, 0.0)


@triton.jit
def _embed_ss_kernel(tok_ptr, emb_ptr, x_ptr, ss_ptr, M, H, n_slots, BLOCK_H: tl.constexpr):
    _embed_unit(tl.program_id(0), tok_ptr, emb_ptr, x_ptr, ss_ptr, M, H, n_slots, BLOCK_H)


def embed_ss(tok, emb, x, ss):
    """tok int64[M]; emb [V, H]; x [M, H] out; ss fp32 [n_slots, M] (slot 0 written, others zeroed)."""
    M, H = x.shape
    _embed_ss_kernel[(M,)](tok, emb, x, ss, M, H, ss.shape[0], BLOCK_H=1024, num_warps=4)


# ------------------------------------------------------------------------------------ skinny GEMM
@triton.jit
def _epilogue(acc, acc2, rm, rn, m_ok, n_ok, out_ptr, res_ptr, ss_out_ptr, stride_out,
              EPI: tl.constexpr, SS_OUT: tl.constexpr, CG: tl.constexpr):
    mask = m_ok[:, None] & n_ok[None, :]
    offs = rm[:, None] * stride_out + rn[None, :]
    if EPI == 1:
        y = _ld(res_ptr + offs, mask, 0.0, CG).to(tl.float32) + _bf(acc)
    elif EPI == 2:
        g = _bf(acc)
        y = _bf(tl.math.div_rn(g, 1.0 + libdevice.exp(-g))) * _bf(acc2)
    else:
        y = acc
    yb = y.to(tl.bfloat16)
    tl.store(out_ptr + offs, yb, mask=mask)
    if SS_OUT:
        yf = tl.where(mask, yb.to(tl.float32), 0.0)
        tl.atomic_add(ss_out_ptr + rm, tl.sum(yf * yf, axis=1), mask=m_ok, sem="relaxed")


@triton.jit
def _gemm_unit(pid_n, pid_k, x_ptr, w_ptr, out_ptr, res_ptr, part_ptr, ticket_ptr, nw_ptr, ss_in_ptr, ss_out_ptr,
               eps, M, N, K, stride_x, stride_w, stride_out, k_chunk, n_split, pair_off,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               EPI: tl.constexpr, NORM: tl.constexpr, SS_OUT: tl.constexpr, CG: tl.constexpr):
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    n_ok = rn < N
    if NORM:  # Qwen3RMSNorm statistics were accumulated by the producer of x
        rstd = tl.math.rsqrt(_ld(ss_in_ptr + rm, m_ok, 0.0, CG) / K + eps)
    k_lo = pid_k * k_chunk
    k_hi = tl.minimum(k_lo + k_chunk, K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    acc2 = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    x_base = x_ptr + rm[:, None] * stride_x
    w_base = w_ptr + rn[:, None] * stride_w
    for k0 in range(k_lo, k_hi, BLOCK_K):
        k = k0 + rk
        x = _ld(x_base + k[None, :], m_ok[:, None], 0.0, CG)
        if NORM:  # weight * bf16(x * rstd), rounded to BF16 (the reference's cast placement)
            xn = (x.to(tl.float32) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            x = (xn * tl.load(nw_ptr + k).to(tl.float32)[None, :]).to(tl.bfloat16)
        w = tl.load(w_base + k[None, :], mask=n_ok[:, None], other=0.0, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
        if EPI == 2:
            w2 = tl.load(w_base + pair_off * stride_w + k[None, :], mask=n_ok[:, None], other=0.0,
                         eviction_policy="evict_first")
            acc2 = tl.dot(x, tl.trans(w2), acc2)
    if n_split == 1:
        _epilogue(acc, acc2, rm, rn, m_ok, n_ok, out_ptr, res_ptr, ss_out_ptr, stride_out, EPI, SS_OUT, CG)
    else:
        mask = m_ok[:, None] & n_ok[None, :]
        pp = part_ptr + (pid_k * M + rm[:, None]) * (2 * N) + rn[None, :]
        tl.store(pp, acc, mask=mask)
        if EPI == 2:
            tl.store(pp + N, acc2, mask=mask)
        tl.debug_barrier()                                    # every thread's partial is issued
        ticket = tl.atomic_add(ticket_ptr + pid_n, 1, sem="acq_rel")
        if ticket == n_split - 1:                             # last CTA of this tile reduces
            acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            acc2 = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for s in range(0, n_split):
                ps = part_ptr + (s * M + rm[:, None]) * (2 * N) + rn[None, :]
                acc += tl.load(ps, mask=mask, other=0.0, cache_modifier=".cg")
                if EPI == 2:
                    acc2 += tl.load(ps + N, mask=mask, other=0.0, cache_modifier=".cg")
            _epilogue(acc, acc2, rm, rn, m_ok, n_ok, out_ptr, res_ptr, ss_out_ptr, stride_out, EPI, SS_OUT, CG)
            tl.atomic_xchg(ticket_ptr + pid_n, 0)             # ready for the next use


@triton.jit
def _gemm_kernel(x_ptr, w_ptr, out_ptr, res_ptr, part_ptr, ticket_ptr, nw_ptr, ss_in_ptr, ss_out_ptr,
                 eps, M, N, K, stride_x, stride_w, stride_out, k_chunk, n_split, pair_off,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 EPI: tl.constexpr, NORM: tl.constexpr, SS_OUT: tl.constexpr):
    _gemm_unit(tl.program_id(0), tl.program_id(1), x_ptr, w_ptr, out_ptr, res_ptr, part_ptr, ticket_ptr, nw_ptr,
               ss_in_ptr, ss_out_ptr, eps, M, N, K, stride_x, stride_w, stride_out, k_chunk, n_split, pair_off,
               BLOCK_M, BLOCK_N, BLOCK_K, EPI, NORM, SS_OUT, False)


def _block_m(M):
    return max(16, triton.next_power_of_2(M))


def split_plan(N, K, bn, bk, split):
    """(k_chunk, n_split, tiles) for a requested split-K factor."""
    steps = K // bk
    split = max(1, min(split, steps))
    chunk = triton.cdiv(steps, split) * bk
    return chunk, triton.cdiv(K, chunk), triton.cdiv(N, bn)


class FastGemm:
    """out[M, N] (+epilogue) = norm?(x)[M, K] @ w.T, split-K reduced by the last CTA per tile."""

    BASE = [(32, 128, 4, 4), (64, 128, 4, 4), (16, 256, 4, 3), (32, 256, 8, 3)]  # BN, BK, warps, stages
    SPLITS = [1, 2, 3, 4, 6, 8]

    def __init__(self, M, N, K, epi, device, norm=False, ss_out=False):
        self.M, self.N, self.K, self.epi, self.norm, self.ss_out = M, N, K, epi, norm, ss_out
        self.device = device
        self.eps = 1e-6
        self.set_config((32, 128, 4, 4, 1))

    def set_config(self, cfg):
        bn, bk, warps, stages, split = cfg
        self.chunk, self.split, self.tiles = split_plan(self.N, self.K, bn, bk, split)
        self.cfg = (bn, bk, warps, stages, split)
        self.part = torch.empty((max(self.split, 1), self.M, 2 * self.N), dtype=torch.float32, device=self.device)
        self.ticket = torch.zeros((self.tiles,), dtype=torch.int32, device=self.device)

    def __call__(self, x, w, out, res=None, norm_w=None, ss_in=None, ss_out=None):
        bn, bk, warps, stages, _ = self.cfg
        assert self.K % bk == 0 and x.stride(-1) == 1 and w.stride(-1) == 1 and out.stride(-1) == 1
        assert (norm_w is not None) == self.norm and (ss_out is not None) == self.ss_out
        _gemm_kernel[(self.tiles, self.split)](
            x, w, out, res if res is not None else out, self.part, self.ticket,
            norm_w if norm_w is not None else w, ss_in if ss_in is not None else self.part,
            ss_out if ss_out is not None else self.part,
            self.eps, self.M, self.N, self.K, x.stride(0), w.stride(0), out.stride(0),
            self.chunk, self.split, self.N,
            BLOCK_M=_block_m(self.M), BLOCK_N=bn, BLOCK_K=bk, EPI=self.epi, NORM=self.norm,
            SS_OUT=self.ss_out, num_warps=warps, num_stages=stages,
        )
        return out

    def candidates(self):
        for base in self.BASE:
            if self.K % base[1]:
                continue
            for split in self.SPLITS:
                if split <= self.K // base[1]:
                    yield base + (split,)


# ------------------------------------------------------------------------------------ attention
@triton.jit
def _attn_unit(b, kvh, split,
               qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr, k_ptr, v_ptr, acc_ptr, ml_ptr, ticket_ptr, out_ptr,
               stride_qkv, stride_kb, stride_kh, stride_kn, n_kv_heads, n_splits, chunk, scale, eps,
               GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, QPAD: tl.constexpr,
               SPLITS_P2: tl.constexpr, CG: tl.constexpr):
    """One (sequence b, KV head, split) unit. q/k per-head RMSNorm + RoPE from the packed qkv row,
    writes the new token's K/V into the cache (the split owning `pos`), flash-decodes its key range
    (P rounded to BF16 before P@V, like flash), and the last split to finish merges."""
    HALF: tl.constexpr = D // 2
    pos = tl.load(pos_ptr).to(tl.int32)
    rows = tl.arange(0, QPAD)
    row_ok = rows < GROUP
    d = tl.arange(0, HALF)
    dd = tl.arange(0, D)
    n_q_heads = n_kv_heads * GROUP
    qh = kvh * GROUP + rows
    row_base = qkv_ptr + b * stride_qkv
    c1 = tl.load(cos_ptr + pos * D + d).to(tl.float32)
    c2 = tl.load(cos_ptr + pos * D + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + pos * D + d).to(tl.float32)
    s2 = tl.load(sin_ptr + pos * D + HALF + d).to(tl.float32)

    # q heads of this group: Qwen3RMSNorm(128) -> BF16, then RoPE with BF16 intermediates
    qp = row_base + qh[:, None] * D
    x1 = _ld(qp + d[None, :], row_ok[:, None], 0.0, CG).to(tl.float32)
    x2 = _ld(qp + HALF + d[None, :], row_ok[:, None], 0.0, CG).to(tl.float32)
    r = tl.math.rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D + eps)
    w1 = tl.load(qn_ptr + d).to(tl.float32)
    w2 = tl.load(qn_ptr + HALF + d).to(tl.float32)
    y1 = _bf(_bf(x1 * r[:, None]) * w1[None, :])
    y2 = _bf(_bf(x2 * r[:, None]) * w2[None, :])
    q1 = (_bf(y1 * c1[None, :]) + _bf(-y2 * s1[None, :])).to(tl.bfloat16)
    q2 = (_bf(y2 * c2[None, :]) + _bf(y1 * s2[None, :])).to(tl.bfloat16)

    start = split * chunk
    owns = (start <= pos) & (pos < start + chunk)
    end = tl.minimum(start + chunk, pos)                       # cached keys; `pos` itself is new
    m_i = tl.full([QPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([QPAD], tl.float32)
    acc = tl.zeros([QPAD, D], tl.float32)
    base = b * stride_kb + kvh * stride_kh
    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + offs_n
        n_ok = n < end
        kp = k_ptr + base + n[:, None] * stride_kn
        k1 = tl.load(kp + d[None, :], mask=n_ok[:, None], other=0.0)
        k2 = tl.load(kp + HALF + d[None, :], mask=n_ok[:, None], other=0.0)
        s = (tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))) * scale
        s = tl.where(n_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + base + n[:, None] * stride_kn + dd[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if owns:
        # the new token: k = RoPE(RMSNorm(k)), v as is; store for later steps, attend from registers
        kp_new = row_base + (n_q_heads + kvh) * D
        kx1 = _ld(kp_new + d, d < HALF, 0.0, CG).to(tl.float32)
        kx2 = _ld(kp_new + HALF + d, d < HALF, 0.0, CG).to(tl.float32)
        kr = tl.math.rsqrt((tl.sum(kx1 * kx1, axis=0) + tl.sum(kx2 * kx2, axis=0)) / D + eps)
        kw1 = tl.load(kn_ptr + d).to(tl.float32)
        kw2 = tl.load(kn_ptr + HALF + d).to(tl.float32)
        ky1 = _bf(_bf(kx1 * kr) * kw1)
        ky2 = _bf(_bf(kx2 * kr) * kw2)
        kn1 = _bf(_bf(ky1 * c1) + _bf(-ky2 * s1))
        kn2 = _bf(_bf(ky2 * c2) + _bf(ky1 * s2))
        vp_new = row_base + (n_q_heads + n_kv_heads + kvh) * D
        vn = _ld(vp_new + dd, dd < D, 0.0, CG)
        dst = base + pos * stride_kn
        tl.store(k_ptr + dst + d, kn1.to(tl.bfloat16))
        tl.store(k_ptr + dst + HALF + d, kn2.to(tl.bfloat16))
        tl.store(v_ptr + dst + dd, vn)
        s_new = (tl.sum(q1.to(tl.float32) * kn1[None, :], axis=1)
                 + tl.sum(q2.to(tl.float32) * kn2[None, :], axis=1)) * scale
        m_new = tl.maximum(m_i, s_new)
        alpha = tl.exp(m_i - m_new)
        p_new = tl.exp(s_new - m_new)
        l_i = l_i * alpha + p_new
        acc = acc * alpha[:, None] + _bf(p_new)[:, None] * vn.to(tl.float32)[None, :]
        m_i = m_new

    part = (b * n_q_heads + qh) * n_splits + split
    tl.store(acc_ptr + part[:, None] * D + dd[None, :], acc, mask=row_ok[:, None])
    tl.store(ml_ptr + part * 2, m_i, mask=row_ok)
    tl.store(ml_ptr + part * 2 + 1, l_i, mask=row_ok)
    tl.debug_barrier()
    tk = ticket_ptr + b * n_kv_heads + kvh
    ticket = tl.atomic_add(tk, 1, sem="acq_rel")
    if ticket == n_splits - 1:                                 # last split merges the group
        sp = tl.arange(0, SPLITS_P2)
        sp_ok = sp < n_splits
        prow = (b * n_q_heads + qh) * n_splits
        mm = tl.load(ml_ptr + (prow[:, None] + sp[None, :]) * 2, mask=row_ok[:, None] & sp_ok[None, :],
                     other=float("-inf"), cache_modifier=".cg")
        m_max = tl.max(mm, axis=1)
        num = tl.zeros([QPAD, D], tl.float32)
        den = tl.zeros([QPAD], tl.float32)
        for s_i in range(0, n_splits):
            ms = tl.load(ml_ptr + (prow + s_i) * 2, mask=row_ok, other=float("-inf"), cache_modifier=".cg")
            ls = tl.load(ml_ptr + (prow + s_i) * 2 + 1, mask=row_ok, other=0.0, cache_modifier=".cg")
            ws = tl.where(ms == float("-inf"), 0.0, tl.exp(ms - m_max))
            a_s = tl.load(acc_ptr + (prow + s_i)[:, None] * D + dd[None, :], mask=row_ok[:, None], other=0.0,
                          cache_modifier=".cg")
            num += a_s * ws[:, None]
            den += ls * ws
        den = tl.where(row_ok, den, 1.0)
        o = num / den[:, None]
        tl.store(out_ptr + (b * n_q_heads + qh)[:, None] * D + dd[None, :], o.to(tl.bfloat16), mask=row_ok[:, None])
        tl.atomic_xchg(tk, 0)


@triton.jit
def _attn_fused_kernel(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr, k_ptr, v_ptr, acc_ptr, ml_ptr, ticket_ptr, out_ptr,
    stride_qkv, stride_kb, stride_kh, stride_kn, n_kv_heads, n_splits, chunk, scale, eps,
    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, QPAD: tl.constexpr, SPLITS_P2: tl.constexpr,
):
    _attn_unit(tl.program_id(0), tl.program_id(1), tl.program_id(2),
               qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr, k_ptr, v_ptr, acc_ptr, ml_ptr, ticket_ptr, out_ptr,
               stride_qkv, stride_kb, stride_kh, stride_kn, n_kv_heads, n_splits, chunk, scale, eps,
               GROUP, D, BLOCK_N, QPAD, SPLITS_P2, False)


def attn_split_plan(batch, n_kv_heads, capacity, sm_count, block_n=64, splits=None):
    if splits is None:
        splits = max(1, min(-(-2 * sm_count // (batch * n_kv_heads)), -(-capacity // block_n)))
    chunk = -(-capacity // splits)
    chunk = -(-chunk // block_n) * block_n
    return chunk, -(-capacity // chunk)


class FastAttention:
    """Fused decode attention for a fixed (batch, capacity); see _attn_unit."""

    def __init__(self, batch, n_kv_heads, capacity, device, sm_count, block_n=64, splits=None):
        self.batch, self.n_kv, self.capacity, self.block_n = batch, n_kv_heads, capacity, block_n
        self.chunk, self.splits = attn_split_plan(batch, n_kv_heads, capacity, sm_count, block_n, splits)
        nq = n_kv_heads * 4
        self.acc = torch.empty((batch * nq * self.splits, 128), dtype=torch.float32, device=device)
        self.ml = torch.empty((batch * nq * self.splits, 2), dtype=torch.float32, device=device)
        self.ticket = torch.zeros((batch * n_kv_heads,), dtype=torch.int32, device=device)
        self.scale = 1.0 / math.sqrt(128)

    def __call__(self, qkv, q_norm_w, k_norm_w, cos, sin, pos, k_cache, v_cache, out, eps):
        """qkv [B, 6144] BF16 (row stride any); pos int64[1] = position of the new token;
        out [B, 32, 128] BF16."""
        B = self.batch
        _attn_fused_kernel[(B, self.n_kv, self.splits)](
            qkv, q_norm_w, k_norm_w, cos, sin, pos, k_cache, v_cache, self.acc, self.ml, self.ticket, out,
            qkv.stride(0), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.n_kv, self.splits, self.chunk, self.scale, eps,
            GROUP=4, D=128, BLOCK_N=self.block_n, QPAD=16, SPLITS_P2=max(2, triton.next_power_of_2(self.splits)),
            num_warps=4, num_stages=2,
        )
        return out


# ------------------------------------------------------------------------------------ LM head
@triton.jit
def _lm_unit(pid, x_ptr, w_ptr, pval_ptr, pidx_ptr, ticket_ptr, tok_ptr, pos_ptr, nw_ptr, ss_in_ptr, eps,
             M, N, K, stride_x, stride_w, n_tiles,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
             INC_POS: tl.constexpr, CG: tl.constexpr):
    """One vocabulary tile; returns True in the CTA that finished the global argmax."""
    rm = tl.arange(0, BLOCK_M)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    n_ok = rn < N
    rstd = tl.math.rsqrt(_ld(ss_in_ptr + rm, m_ok, 0.0, CG) / K + eps)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    x_base = x_ptr + rm[:, None] * stride_x
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        x = _ld(x_base + k[None, :], m_ok[:, None], 0.0, CG)
        xn = (x.to(tl.float32) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        x = (xn * tl.load(nw_ptr + k).to(tl.float32)[None, :]).to(tl.bfloat16)
        w = tl.load(w_ptr + rn[:, None] * stride_w + k[None, :], mask=n_ok[:, None], other=0.0,
                    eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
    logits = tl.where(n_ok[None, :], _bf(acc), float("-inf"))   # the reference's logits are BF16
    best = tl.max(logits, axis=1)
    idx = tl.min(tl.where(logits == best[:, None], rn[None, :], 2147483647), axis=1)
    tl.store(pval_ptr + rm * n_tiles + pid, best, mask=m_ok)
    tl.store(pidx_ptr + rm * n_tiles + pid, idx, mask=m_ok)
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel")
    last = ticket == n_tiles - 1
    if last:                                                    # last tile: global argmax per row
        for m in range(0, M):
            bv = tl.full([BLOCK_T], float("-inf"), tl.float32)
            bi = tl.full([BLOCK_T], 2147483647, tl.int32)
            for t0 in range(0, n_tiles, BLOCK_T):
                t = t0 + tl.arange(0, BLOCK_T)
                ok = t < n_tiles
                v = tl.load(pval_ptr + m * n_tiles + t, mask=ok, other=float("-inf"), cache_modifier=".cg")
                i = tl.load(pidx_ptr + m * n_tiles + t, mask=ok, other=2147483647, cache_modifier=".cg")
                better = (v > bv) | ((v == bv) & (i < bi))
                bv = tl.where(better, v, bv)
                bi = tl.where(better, i, bi)
            gmax = tl.max(bv, axis=0)
            tl.store(tok_ptr + m, tl.min(tl.where(bv == gmax, bi, 2147483647), axis=0).to(tl.int64))
        if INC_POS:
            tl.store(pos_ptr, tl.load(pos_ptr) + 1)
        tl.atomic_xchg(ticket_ptr, 0)
    return last


@triton.jit
def _lm_kernel(x_ptr, w_ptr, pval_ptr, pidx_ptr, ticket_ptr, tok_ptr, pos_ptr, nw_ptr, ss_in_ptr, eps,
               M, N, K, stride_x, stride_w, n_tiles,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
               INC_POS: tl.constexpr):
    _lm_unit(tl.program_id(0), x_ptr, w_ptr, pval_ptr, pidx_ptr, ticket_ptr, tok_ptr, pos_ptr, nw_ptr, ss_in_ptr,
             eps, M, N, K, stride_x, stride_w, n_tiles, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_T, INC_POS, False)


class FastLmHead:
    CANDIDATES = [(64, 128, 4, 4), (128, 128, 8, 3), (32, 256, 4, 3)]

    def __init__(self, M, V, K, device):
        self.M, self.V, self.K, self.device = M, V, K, device
        self.eps = 1e-6
        self.ticket = torch.zeros((1,), dtype=torch.int32, device=device)
        self.set_config(self.CANDIDATES[0])

    def set_config(self, cfg):
        self.cfg = cfg
        self.tiles = triton.cdiv(self.V, cfg[0])
        self.pval = torch.empty((self.M, self.tiles), dtype=torch.float32, device=self.device)
        self.pidx = torch.empty((self.M, self.tiles), dtype=torch.int32, device=self.device)

    def candidates(self):
        return list(self.CANDIDATES)

    def __call__(self, x, w, tok, norm_w, ss_in, pos, inc_pos):
        bn, bk, warps, stages = self.cfg
        _lm_kernel[(self.tiles,)](
            x, w, self.pval, self.pidx, self.ticket, tok, pos, norm_w, ss_in, self.eps,
            self.M, self.V, self.K, x.stride(0), w.stride(0), self.tiles,
            BLOCK_M=_block_m(self.M), BLOCK_N=bn, BLOCK_K=bk, BLOCK_T=1024, INC_POS=inc_pos,
            num_warps=warps, num_stages=stages,
        )
        return tok
