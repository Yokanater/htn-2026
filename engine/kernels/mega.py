"""T6: one persistent kernel per decode step (the whole 36-layer forward + LM head + argmax).

Grid = one CTA per SM, all resident. Work is a fixed sequence of ops
  embed, 36 x (qkv, attention, o_proj, gate/up, down), lm_head
each split into units assigned round-robin to CTAs (unit u -> CTA u % P). A CTA with units in an op
first waits until the previous op has completed all its units (a counter in global memory: arrive =
debug_barrier + release atomic, wait = acquire atomic spin), then runs them. Ops depend only on
earlier ops and every CTA is resident, so the schedule cannot deadlock; every spin is still bounded
(LIMIT polls) and sets an error flag instead of hanging, which the engine's warmup check turns into
a fallback. Counters never reset: this launch's targets are (epoch + 1) * units, and the CTA that
finishes the LM head advances the epoch.

Unit bodies are exactly T5's (kernels/fast.py), so the arithmetic is identical; activations produced
by other CTAs in this launch are loaded through L2 (.cg) since L1 is not coherent across SMs.
"""

import math

import torch
import triton
import triton.language as tl

from kernels.fast import EPI_NONE, EPI_RES, EPI_SILU, _attn_unit, _block_m, _embed_unit, _gemm_unit, _lm_unit
from kernels.fast import attn_split_plan, split_plan


@triton.jit
def _wait(cnt_ptr, target, err_ptr, LIMIT: tl.constexpr):
    """Spin until the counter reaches target. Bounded, and once any CTA has flagged an error every
    later wait returns at once, so a broken schedule costs milliseconds, never the warmup budget."""
    c = tl.atomic_add(cnt_ptr, 0, sem="acquire")
    e = tl.atomic_add(err_ptr, 0, sem="relaxed")
    it = 0
    while (c < target) & (it < LIMIT) & (e == 0):
        c = tl.atomic_add(cnt_ptr, 0, sem="acquire")
        e = tl.atomic_add(err_ptr, 0, sem="relaxed")
        it += 1
    if c < target:
        tl.atomic_xchg(err_ptr, 1)


@triton.jit
def _arrive(cnt_ptr):
    tl.debug_barrier()                                          # the unit's stores are all issued
    tl.atomic_add(cnt_ptr, 1, sem="release")


@triton.jit
def _mega_kernel(
    tok_ptr, pos_ptr, x_ptr, ss_ptr, qkv_ptr, attn_ptr, act_ptr,
    emb_ptr, wqkv_ptr, ln1_ptr, qn_ptr, kn_ptr, wo_ptr, ln2_ptr, wgu_ptr, wd_ptr, fn_ptr, lm_ptr,
    cos_ptr, sin_ptr, kc_ptr, vc_ptr,
    part_ptr, gticket_ptr, aacc_ptr, aml_ptr, aticket_ptr, pval_ptr, pidx_ptr, lticket_ptr,
    done_ptr, epoch_ptr, err_ptr,
    M, H, I, V, NQKV, QS, NL, n_slots, n_kv,
    stride_kl, stride_kb, stride_kh, stride_kn, eps, scale,
    qkv_chunk, qkv_split, o_chunk, o_split, gu_chunk, gu_split, d_chunk, d_split,
    a_chunk, a_splits, lm_tiles,
    BLOCK_M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BN_LM: tl.constexpr, BK_LM: tl.constexpr,
    QPAD: tl.constexpr, SPLITS_P2: tl.constexpr, BLOCK_H: tl.constexpr, LIMIT: tl.constexpr,
):
    c = tl.program_id(0)
    P = tl.num_programs(0)
    ep = tl.load(epoch_ptr) + 1                                 # completions to reach this launch

    # op 0: embedding + norm statistics for layer 0
    for u in range(c, M, P):
        _embed_unit(u, tok_ptr, emb_ptr, x_ptr, ss_ptr, M, H, n_slots, BLOCK_H)
        _arrive(done_ptr)
    prev = 0
    prev_n = M

    qkv_tiles = tl.cdiv(NQKV, BN)
    o_tiles = tl.cdiv(H, BN)
    gu_tiles = tl.cdiv(I, BN)
    d_tiles = tl.cdiv(H, BN)
    n_qkv = qkv_tiles * qkv_split
    n_attn = M * n_kv * a_splits
    n_o = o_tiles * o_split
    n_gu = gu_tiles * gu_split
    n_d = d_tiles * d_split
    for l in range(0, NL):
        op = 1 + 5 * l
        l64 = l.to(tl.int64)                                    # layer offsets exceed 2^31 elements
        kc = kc_ptr + l64 * stride_kl
        vc = vc_ptr + l64 * stride_kl
        # qkv = RMSNorm_1(x) @ Wqkv.T
        if c < n_qkv:
            _wait(done_ptr + prev, ep * prev_n, err_ptr, LIMIT)
            for u in range(c, n_qkv, P):
                _gemm_unit(u // qkv_split, u % qkv_split, x_ptr, wqkv_ptr + l64 * NQKV * H, qkv_ptr, qkv_ptr,
                           part_ptr, gticket_ptr, ln1_ptr + l64 * H, ss_ptr + (2 * l) * M, ss_ptr,
                           eps, M, NQKV, H, H, H, NQKV, qkv_chunk, qkv_split, NQKV,
                           BLOCK_M, BN, BK, 0, True, False, True)
                _arrive(done_ptr + op)
        # attention (+ q/k norm, RoPE, KV write, merge)
        if c < n_attn:
            _wait(done_ptr + op, ep * n_qkv, err_ptr, LIMIT)
            for u in range(c, n_attn, P):
                b = u // (n_kv * a_splits)
                rem = u % (n_kv * a_splits)
                _attn_unit(b, rem // a_splits, rem % a_splits,
                           qkv_ptr, qn_ptr + l64 * 128, kn_ptr + l64 * 128, cos_ptr, sin_ptr, pos_ptr, kc, vc,
                           aacc_ptr, aml_ptr, aticket_ptr, attn_ptr,
                           NQKV, stride_kb, stride_kh, stride_kn, n_kv, a_splits, a_chunk, scale, eps,
                           4, 128, 64, QPAD, SPLITS_P2, True)
                _arrive(done_ptr + op + 1)
        # x += o_proj(attn); statistics for post_attention_layernorm
        if c < n_o:
            _wait(done_ptr + op + 1, ep * n_attn, err_ptr, LIMIT)
            for u in range(c, n_o, P):
                _gemm_unit(u // o_split, u % o_split, attn_ptr, wo_ptr + l64 * H * QS, x_ptr, x_ptr,
                           part_ptr, gticket_ptr, ln2_ptr, ss_ptr, ss_ptr + (2 * l + 1) * M,
                           eps, M, H, QS, QS, QS, H, o_chunk, o_split, H,
                           BLOCK_M, BN, BK, 1, False, True, True)
                _arrive(done_ptr + op + 2)
        # act = silu(g) * u of RMSNorm_2(x)
        if c < n_gu:
            _wait(done_ptr + op + 2, ep * n_o, err_ptr, LIMIT)
            for u in range(c, n_gu, P):
                _gemm_unit(u // gu_split, u % gu_split, x_ptr, wgu_ptr + l64 * 2 * I * H, act_ptr, act_ptr,
                           part_ptr, gticket_ptr, ln2_ptr + l64 * H, ss_ptr + (2 * l + 1) * M, ss_ptr,
                           eps, M, I, H, H, H, I, gu_chunk, gu_split, I,
                           BLOCK_M, BN, BK, 2, True, False, True)
                _arrive(done_ptr + op + 3)
        # x += down(act); statistics for the next input_layernorm (or the final norm)
        if c < n_d:
            _wait(done_ptr + op + 3, ep * n_gu, err_ptr, LIMIT)
            for u in range(c, n_d, P):
                _gemm_unit(u // d_split, u % d_split, act_ptr, wd_ptr + l64 * H * I, x_ptr, x_ptr,
                           part_ptr, gticket_ptr, ln2_ptr, ss_ptr, ss_ptr + (2 * l + 2) * M,
                           eps, M, H, I, I, I, H, d_chunk, d_split, H,
                           BLOCK_M, BN, BK, 1, False, True, True)
                _arrive(done_ptr + op + 4)
        prev = op + 4
        prev_n = n_d

    # final RMSNorm + LM head + argmax; the CTA that completes it advances pos and the epoch
    op_lm = 1 + 5 * NL
    if c < lm_tiles:
        _wait(done_ptr + prev, ep * prev_n, err_ptr, LIMIT)
        for u in range(c, lm_tiles, P):
            last = _lm_unit(u, x_ptr, lm_ptr, pval_ptr, pidx_ptr, lticket_ptr, tok_ptr, pos_ptr, fn_ptr,
                            ss_ptr + (2 * NL) * M, eps, M, V, H, H, H, lm_tiles,
                            BLOCK_M, BN_LM, BK_LM, 1024, True, True)
            if last:
                tl.store(epoch_ptr, ep)
            _arrive(done_ptr + op_lm)


def _balanced_split(tiles, steps, programs, max_split=8):
    """Split-K factor whose unit count spreads most evenly over the persistent CTAs."""
    best = (0.0, 1)
    for s in range(1, min(max_split, steps) + 1):
        units = tiles * s
        waves = -(-units // programs)
        eff = units / (waves * programs)
        if eff > best[0] + 1e-9:
            best = (eff, s)
    return best[1]


class MegaStep:
    """Owns the scratch and launch arguments of the persistent decode-step kernel for one state."""

    BN, BK, BN_LM, BK_LM = 32, 128, 64, 128

    def __init__(self, eng, st, programs=None, limit=1 << 18, num_warps=4, num_stages=3):
        dev = st.k_all.device
        B, H, I, V = st.B, eng.embed.shape[1], eng.inter, eng.lm_head.shape[0]
        NQKV, QS = eng.w_qkv_all.shape[1], eng.w_o_all.shape[2]
        self.eng, self.st = eng, st
        self.P = programs or eng.sm_count
        self.limit, self.num_warps, self.num_stages = limit, num_warps, num_stages
        steps_h, steps_qs, steps_i = H // self.BK, QS // self.BK, I // self.BK
        plans = {}
        for name, n, k, steps in (("qkv", NQKV, H, steps_h), ("o", H, QS, steps_qs),
                                  ("gu", I, H, steps_h), ("d", H, I, steps_i)):
            s = _balanced_split(triton.cdiv(n, self.BN), steps, self.P)
            plans[name] = split_plan(n, k, self.BN, self.BK, s)          # (chunk, split, tiles)
        self.plans = plans
        self.a_chunk, self.a_splits = attn_split_plan(B, 8, st.capacity, self.P)
        self.lm_tiles = triton.cdiv(V, self.BN_LM)
        max_part = max(p[1] * 2 * n for p, n in ((plans["qkv"], NQKV), (plans["o"], H), (plans["gu"], I),
                                                 (plans["d"], H)))
        self.part = torch.empty((max_part * B,), dtype=torch.float32, device=dev)
        self.gticket = torch.zeros((max(p[2] for p in plans.values()),), dtype=torch.int32, device=dev)
        nq = 8 * 4
        self.aacc = torch.empty((B * nq * self.a_splits, 128), dtype=torch.float32, device=dev)
        self.aml = torch.empty((B * nq * self.a_splits, 2), dtype=torch.float32, device=dev)
        self.aticket = torch.zeros((B * 8,), dtype=torch.int32, device=dev)
        self.pval = torch.empty((B, self.lm_tiles), dtype=torch.float32, device=dev)
        self.pidx = torch.empty((B, self.lm_tiles), dtype=torch.int32, device=dev)
        self.lticket = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.done = torch.zeros((2 + 5 * len(eng.layers),), dtype=torch.int32, device=dev)
        self.epoch = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.err = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.scale = 1.0 / math.sqrt(128)

    def describe(self):
        p = self.plans
        return (f"P={self.P} splits qkv={p['qkv'][1]} o={p['o'][1]} gu={p['gu'][1]} d={p['d'][1]} "
                f"attn={self.a_splits} lm_tiles={self.lm_tiles}")

    def __call__(self):
        eng, st, p = self.eng, self.st, self.plans
        B = st.B
        _mega_kernel[(self.P,)](
            st.tok, st.pos, st.x, st.ss, st.qkv_buf, st.attn_out, st.act_buf,
            eng.embed, eng.w_qkv_all, eng.ln1_all, eng.qn_all, eng.kn_all, eng.w_o_all, eng.ln2_all,
            eng.w_gu_all, eng.w_d_all, eng.final_norm, eng.lm_head,
            st.cos, st.sin, st.k_all, st.v_all,
            self.part, self.gticket, self.aacc, self.aml, self.aticket, self.pval, self.pidx, self.lticket,
            self.done, self.epoch, self.err,
            B, eng.embed.shape[1], eng.inter, eng.lm_head.shape[0], eng.w_qkv_all.shape[1], eng.w_o_all.shape[2],
            len(eng.layers), st.ss.shape[0], 8,
            st.k_all.stride(0), st.k_all.stride(1), st.k_all.stride(2), st.k_all.stride(3), eng.eps, self.scale,
            p["qkv"][0], p["qkv"][1], p["o"][0], p["o"][1], p["gu"][0], p["gu"][1], p["d"][0], p["d"][1],
            self.a_chunk, self.a_splits, self.lm_tiles,
            BLOCK_M=_block_m(B), BN=self.BN, BK=self.BK, BN_LM=self.BN_LM, BK_LM=self.BK_LM,
            QPAD=16, SPLITS_P2=max(2, triton.next_power_of_2(self.a_splits)), BLOCK_H=1024, LIMIT=self.limit,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
