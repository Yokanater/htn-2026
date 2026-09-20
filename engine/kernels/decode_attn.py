"""Grouped-query flash-decoding for one new token per sequence.

Graph-safe: the number of valid keys is read from a device tensor (``pos + 1``), so one
captured launch serves every decode step. The cache has fixed capacity; slots at or beyond
the valid length are never read.

Arithmetic follows the flash-attention forward that the reference's SDPA dispatches to:
scores are BF16 x BF16 dot products accumulated in fp32, softmax runs in fp32 with online
rescaling, the row sum uses fp32 probabilities, and the probabilities are rounded to BF16
before the P @ V product (fp32 accumulate). The final normalisation is in fp32, then BF16.
Only the order of the reduction over keys differs (split-K + merge).
"""

import math

import torch
import triton
import triton.language as tl

GROUP = 4       # query heads per KV head
HEAD_DIM = 128
BLOCK_N = 64
QPAD = 16       # tl.dot needs >= 16 rows; the 4 real query heads are padded


@triton.jit
def _attn_partial_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, acc_ptr, ml_ptr,
    stride_kb, stride_kh, stride_kn,
    n_kv_heads, n_splits, chunk, scale,
    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, QPAD: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    length = tl.load(pos_ptr).to(tl.int32) + 1

    rows = tl.arange(0, QPAD)
    dims = tl.arange(0, D)
    row_ok = rows < GROUP
    qh = kvh * GROUP + rows                                   # query head ids
    n_q_heads = n_kv_heads * GROUP
    q = tl.load(q_ptr + (b * n_q_heads + qh)[:, None] * D + dims[None, :], mask=row_ok[:, None], other=0.0)

    start = split * chunk
    end = tl.minimum(start + chunk, length)
    m_i = tl.full([QPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([QPAD], tl.float32)
    acc = tl.zeros([QPAD, D], tl.float32)
    base = b * stride_kb + kvh * stride_kh
    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + offs_n
        n_ok = n < end
        kv_off = base + n[:, None] * stride_kn + dims[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale                     # [QPAD, BLOCK_N] fp32
        s = tl.where(n_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off, mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    part = ((b * n_q_heads + qh) * n_splits + split)
    tl.store(acc_ptr + part[:, None] * D + dims[None, :], acc, mask=row_ok[:, None])
    tl.store(ml_ptr + part * 2, m_i, mask=row_ok)
    tl.store(ml_ptr + part * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _attn_merge_kernel(acc_ptr, ml_ptr, out_ptr, n_splits, SPLITS: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)                                     # b * n_q_heads + head
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


class DecodeAttention:
    """Preallocated split-K decode attention for a fixed (batch, capacity)."""

    def __init__(self, batch, n_kv_heads, capacity, device, sm_count):
        self.batch, self.n_kv, self.capacity = batch, n_kv_heads, capacity
        target = 2 * sm_count
        splits = max(1, min(-(-target // (batch * n_kv_heads)), -(-capacity // BLOCK_N)))
        chunk = -(-capacity // splits)
        chunk = -(-chunk // BLOCK_N) * BLOCK_N
        self.splits = -(-capacity // chunk)
        self.chunk = chunk
        nq = n_kv_heads * GROUP
        self.acc = torch.empty((batch * nq * self.splits, HEAD_DIM), dtype=torch.float32, device=device)
        self.ml = torch.empty((batch * nq * self.splits, 2), dtype=torch.float32, device=device)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def __call__(self, q, k_cache, v_cache, pos, out):
        """q [B, 32, 128] contiguous BF16; caches [B, 8, capacity, 128]; pos int64[1] = index
        of the token just written; out [B, 32, 128] BF16."""
        B = self.batch
        _attn_partial_kernel[(B, self.n_kv, self.splits)](
            q, k_cache, v_cache, pos, self.acc, self.ml,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.n_kv, self.splits, self.chunk, self.scale,
            GROUP=GROUP, D=HEAD_DIM, BLOCK_N=BLOCK_N, QPAD=QPAD, num_warps=4, num_stages=2,
        )
        _attn_merge_kernel[(B * self.n_kv * GROUP,)](
            self.acc, self.ml, out, self.splits,
            SPLITS=max(2, triton.next_power_of_2(self.splits)), D=HEAD_DIM, num_warps=4,
        )
        return out


@triton.jit
def _attn_verify_partial_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, acc_ptr, ml_ptr,
    stride_kb, stride_kh, stride_kn,
    n_kv_heads, n_splits, chunk, scale,
    T: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, QPAD: tl.constexpr,
):
    """T queries per sequence at positions pos[b] .. pos[b]+T-1 (causal). Row r of a program is
    (t = r // GROUP, query head kvh*GROUP + r % GROUP); it sees keys <= pos[b] + t."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    base_pos = tl.load(pos_ptr + b).to(tl.int32)

    rows = tl.arange(0, QPAD)
    dims = tl.arange(0, D)
    t = rows // GROUP
    row_ok = rows < T * GROUP
    qh = kvh * GROUP + rows % GROUP
    n_q_heads = n_kv_heads * GROUP
    qrow = (b * T + t) * n_q_heads + qh                       # q / out layout [B, T, NQ, D]
    q = tl.load(q_ptr + qrow[:, None] * D + dims[None, :], mask=row_ok[:, None], other=0.0)
    limit = base_pos + t                                       # last visible key per row

    start = split * chunk
    end = tl.minimum(start + chunk, base_pos + T)
    m_i = tl.full([QPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([QPAD], tl.float32)
    acc = tl.zeros([QPAD, D], tl.float32)
    base = b * stride_kb + kvh * stride_kh
    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + offs_n
        n_ok = n < end
        kv_off = base + n[:, None] * stride_kn + dims[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        visible = n_ok[None, :] & (n[None, :] <= limit[:, None])
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)   # rows with nothing visible yet
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


class VerifyAttention:
    """Causal attention for T new tokens per sequence at per-sequence positions (graph-safe)."""

    def __init__(self, batch, T, n_kv_heads, capacity, device, sm_count):
        self.batch, self.T, self.n_kv, self.capacity = batch, T, n_kv_heads, capacity
        target = 2 * sm_count
        splits = max(1, min(-(-target // (batch * n_kv_heads)), -(-capacity // BLOCK_N)))
        chunk = -(-capacity // splits)
        chunk = -(-chunk // BLOCK_N) * BLOCK_N
        self.splits = -(-capacity // chunk)
        self.chunk = chunk
        rows = batch * T * n_kv_heads * GROUP
        self.acc = torch.empty((rows * self.splits, HEAD_DIM), dtype=torch.float32, device=device)
        self.ml = torch.empty((rows * self.splits, 2), dtype=torch.float32, device=device)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)
        self.qpad = max(16, triton.next_power_of_2(T * GROUP))

    def __call__(self, q, k_cache, v_cache, pos, out):
        """q, out [B, T, 32, 128] contiguous BF16; pos int64[B] = position of token 0."""
        B = self.batch
        _attn_verify_partial_kernel[(B, self.n_kv, self.splits)](
            q, k_cache, v_cache, pos, self.acc, self.ml,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            self.n_kv, self.splits, self.chunk, self.scale,
            T=self.T, GROUP=GROUP, D=HEAD_DIM, BLOCK_N=BLOCK_N, QPAD=self.qpad, num_warps=4, num_stages=2,
        )
        _attn_merge_kernel[(B * self.T * self.n_kv * GROUP,)](
            self.acc, self.ml, out, self.splits,
            SPLITS=max(2, triton.next_power_of_2(self.splits)), D=HEAD_DIM, num_warps=4,
        )
        return out
