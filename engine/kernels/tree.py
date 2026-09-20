"""Token-tree speculative verification kernels (exact: every emitted token is the model's argmax).

Each sequence b verifies a tree of T nodes in one forward. Node 0 is the last emitted token (its K/V
not yet cached); node i has a parent, a depth d_i (node 0: depth 0) and an ancestor-or-self bitmask
anc[b, i] (bit j set if node j is on the path root..i). Node i:
  - is rotated at absolute position base[b] + d_i,
  - writes its K/V into cache slot base[b] + i (a scratch slot; see compaction below),
  - attends to every cached key < base[b] and to block slot base[b] + j iff bit j of anc[b, i].
After the host picks the accepted path (root, n_1, .., n_a), `compact` moves the K/V of n_k from slot
base + idx(n_k) to base + k (idx(n_k) >= k, so an in-order copy never clobbers a later source).
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _tree_qk_kernel(qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, base_ptr, depth_ptr,
                    q_out_ptr, k_cache_ptr, v_cache_ptr, T, eps, stride_qkv,
                    stride_cb, stride_ch, stride_cn,
                    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr):
    """One program per (row = b*T + t, head). q_out is [B, T, NQ, D] contiguous."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    b = row // T
    t = row % T
    base = tl.load(base_ptr + b).to(tl.int32)
    rpos = base + tl.load(depth_ptr + row).to(tl.int32)        # RoPE position
    slot = base + t                                             # scratch cache slot
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, HALF)
    src = qkv_ptr + row * stride_qkv + head * D
    if head >= NQ + NKV:
        vh = head - NQ - NKV
        dst = v_cache_ptr + b * stride_cb + vh * stride_ch + slot * stride_cn
        tl.store(dst + d, tl.load(src + d))
        tl.store(dst + HALF + d, tl.load(src + HALF + d))
    else:
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + HALF + d).to(tl.float32)
        r = tl.math.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        if head < NQ:
            w_ptr = qn_ptr
        else:
            w_ptr = kn_ptr
        w1 = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
        y1 = _bf(_bf(x1 * r) * w1)
        y2 = _bf(_bf(x2 * r) * w2)
        c1 = tl.load(cos_ptr + rpos * D + d).to(tl.float32)
        c2 = tl.load(cos_ptr + rpos * D + HALF + d).to(tl.float32)
        s1 = tl.load(sin_ptr + rpos * D + d).to(tl.float32)
        s2 = tl.load(sin_ptr + rpos * D + HALF + d).to(tl.float32)
        o1 = _bf(y1 * c1) + _bf(-y2 * s1)
        o2 = _bf(y2 * c2) + _bf(y1 * s2)
        if head < NQ:
            dst = q_out_ptr + (row * NQ + head) * D
        else:
            dst = k_cache_ptr + b * stride_cb + (head - NQ) * stride_ch + slot * stride_cn
        tl.store(dst + d, o1.to(tl.bfloat16))
        tl.store(dst + HALF + d, o2.to(tl.bfloat16))


@triton.jit
def _tree_attn_partial_kernel(q_ptr, k_ptr, v_ptr, base_ptr, anc_ptr, acc_ptr, ml_ptr,
                              stride_kb, stride_kh, stride_kn, n_kv_heads, n_splits, chunk, scale,
                              T: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                              QPAD: tl.constexpr, N_RB: tl.constexpr):
    """Rows of a program: (t = r // GROUP, head kvh*GROUP + r % GROUP) for r in its row block (at
    most 32 rows: Triton 3.1 aborts compiling 64-row Hopper MMAs with a register operand here).
    Keys: cache slots < base are visible to all rows; block slot base + j is visible to row t iff bit
    j of anc[b, t]."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2) // N_RB
    rb = tl.program_id(2) % N_RB
    base = tl.load(base_ptr + b).to(tl.int32)
    rows = rb * QPAD + tl.arange(0, QPAD)
    dims = tl.arange(0, D)
    t = rows // GROUP
    row_ok = rows < T * GROUP
    qh = kvh * GROUP + rows % GROUP
    n_q_heads = n_kv_heads * GROUP
    qrow = (b * T + t) * n_q_heads + qh
    q = tl.load(q_ptr + qrow[:, None] * D + dims[None, :], mask=row_ok[:, None], other=0.0)
    anc = tl.load(anc_ptr + b * T + t, mask=row_ok, other=0)

    start = split * chunk
    end = tl.minimum(start + chunk, base + T)
    m_i = tl.full([QPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([QPAD], tl.float32)
    acc = tl.zeros([QPAD, D], tl.float32)
    kvbase = b * stride_kb + kvh * stride_kh
    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + offs_n
        n_ok = n < end
        kv_off = kvbase + n[:, None] * stride_kn + dims[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        j = n - base                                            # block index (negative: cached key)
        in_block = (j >= 0)[None, :]
        bit = (anc[:, None] >> tl.maximum(j, 0)[None, :]) & 1
        visible = n_ok[None, :] & ((~in_block) | (bit == 1))
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off, mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    part = qrow * n_splits + split
    tl.store(acc_ptr + part[:, None] * D + dims[None, :], acc, mask=row_ok[:, None])
    tl.store(ml_ptr + part * 2, m_i, mask=row_ok)
    tl.store(ml_ptr + part * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _tree_merge_kernel(acc_ptr, ml_ptr, out_ptr, n_splits, SPLITS: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    sp = tl.arange(0, SPLITS)
    sp_ok = sp < n_splits
    m = tl.load(ml_ptr + (row * n_splits + sp) * 2, mask=sp_ok, other=float("-inf"))
    l = tl.load(ml_ptr + (row * n_splits + sp) * 2 + 1, mask=sp_ok, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_max))
    dims = tl.arange(0, D)
    acc = tl.load(acc_ptr + (row * n_splits + sp)[:, None] * D + dims[None, :], mask=sp_ok[:, None], other=0.0)
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0)
    tl.store(out_ptr + row * D + dims, (num / den).to(out_ptr.dtype.element_ty))


@triton.jit
def _compact_kernel(k_ptr, v_ptr, base_ptr, src_ptr, cnt_ptr, stride_kl, stride_kb, stride_kh, stride_kn,
                    T, NKV: tl.constexpr, D: tl.constexpr):
    """Program (layer, b, k): for 1 <= k < cnt[b], cache[base + k] <- cache[base + src[b, k]].
    Runs k in increasing order inside one program per (layer, b) to keep the in-place copy safe."""
    layer = tl.program_id(0)
    b = tl.program_id(1)
    cnt = tl.load(cnt_ptr + b)
    base = tl.load(base_ptr + b).to(tl.int32)
    h = tl.arange(0, NKV)
    dd = tl.arange(0, D)
    off_hd = h[:, None] * stride_kh + dd[None, :]
    lb = layer.to(tl.int64) * stride_kl + b.to(tl.int64) * stride_kb   # > 2^31 elements at large B*cap
    for k in range(1, cnt):
        s = tl.load(src_ptr + b * T + k).to(tl.int32)
        if s != k:
            src = lb + (base + s) * stride_kn + off_hd
            dst = lb + (base + k) * stride_kn + off_hd
            tl.store(k_ptr + dst, tl.load(k_ptr + src))
            tl.store(v_ptr + dst, tl.load(v_ptr + src))


class TreeAttention:
    def __init__(self, batch, T, n_kv_heads, capacity, device, sm_count, block_n=64):
        self.batch, self.T, self.n_kv, self.block_n = batch, T, n_kv_heads, block_n
        splits = max(1, min(-(-2 * sm_count // (batch * n_kv_heads)), -(-capacity // block_n)))
        chunk = -(-capacity // splits)
        chunk = -(-chunk // block_n) * block_n
        self.splits, self.chunk = -(-capacity // chunk), chunk
        rows = batch * T * n_kv_heads * 4
        self.acc = torch.empty((rows * self.splits, 128), dtype=torch.float32, device=device)
        self.ml = torch.empty((rows * self.splits, 2), dtype=torch.float32, device=device)
        self.scale = 1.0 / math.sqrt(128)
        rows_total = triton.next_power_of_2(T * 4)
        self.qpad = max(16, min(32, rows_total))              # <= 32 rows per program (see kernel)
        self.n_rb = max(1, rows_total // self.qpad)

    def __call__(self, q, k_cache, v_cache, base, anc, out):
        """q, out [B, T, 32, 128] BF16; base int64[B]; anc int32[B, T]."""
        B = self.batch
        _tree_attn_partial_kernel[(B, self.n_kv, self.splits * self.n_rb)](
            q, k_cache, v_cache, base, anc, self.acc, self.ml,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), self.n_kv, self.splits, self.chunk, self.scale,
            T=self.T, GROUP=4, D=128, BLOCK_N=self.block_n, QPAD=self.qpad, N_RB=self.n_rb,
            num_warps=4, num_stages=2,
        )
        _tree_merge_kernel[(B * self.T * self.n_kv * 4,)](
            self.acc, self.ml, out, self.splits, SPLITS=max(2, triton.next_power_of_2(self.splits)), D=128,
            num_warps=4,
        )
        return out


def tree_qk(qkv, q_norm_w, k_norm_w, cos, sin, base, depth, q_out, k_cache, v_cache, B, T, eps):
    nkv, D = k_cache.shape[1], k_cache.shape[3]
    nq = qkv.shape[1] // D - 2 * nkv
    _tree_qk_kernel[(B * T, nq + 2 * nkv)](
        qkv, q_norm_w, k_norm_w, cos, sin, base, depth, q_out, k_cache, v_cache, T, eps, qkv.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), NQ=nq, NKV=nkv, D=D, num_warps=1,
    )


def compact(k_all, v_all, base, src, cnt, T):
    """k_all/v_all [L, B, 8, cap, 128]; base int64[B]; src int32[B, T]; cnt int32[B]."""
    L, B = k_all.shape[0], k_all.shape[1]
    _compact_kernel[(L, B)](k_all, v_all, base, src, cnt, k_all.stride(0), k_all.stride(1), k_all.stride(2),
                            k_all.stride(3), T, NKV=k_all.shape[2], D=k_all.shape[4], num_warps=4)
