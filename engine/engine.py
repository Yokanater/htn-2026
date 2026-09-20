"""Qwen3 4B greedy decode engine.

Loads the checkpoint with Transformers (so tied weights and dtypes are exactly the
reference's) and runs its own forward: packed QKV and gate/up weights, a static per-layer KV
cache, SDPA flash prefill, a Triton split-K GQA decode attention that reads a device-side
position, and CUDA graphs over both prefill and the decode step. Arithmetic follows
transformers 4.51.3 modeling_qwen3 operation by operation; only reduction order differs.

Tiers, fastest first. On the first call of a shape (the platform's untimed warmup) the engine
validates tiers against the loaded native model by teacher-forcing their own tokens, falling
back a tier on any exception, non-finite logit, or margin above MARGIN_LIMIT, and keeps the
faster of the two best passing tiers:
  T6  one persistent kernel per decode step (all layers + LM head), T5's arithmetic, counter-based
      dependencies between ops, bounded spins
  T5  five kernels per layer: norm statistics carried in epilogues, split-K reduced by the last CTA
      (no reduce launches), attention with fused q/k-norm + RoPE + KV write + merge, LM head +
      argmax + position advance in one launch
  T4  T3 + Triton skinny-GEMM decode (residual / SwiGLU epilogues, LM head + argmax), optionally
      with exact prompt-lookup speculative decoding
  T3  CUDA graphs + fused Triton kernels
  T2  CUDA graphs + Triton norm / decode attention, torch elementwise ops
  T1  eager, torch ops only (reference-style SDPA over the cache)
  T0  the starter's native Transformers loop
Diagnostics go to stderr with an "[engine]" prefix, only during that warmup call.
"""

import gc
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kernels.decode_attn import DecodeAttention, VerifyAttention
    from kernels.rmsnorm import rms_norm_rows
    _KERNEL_ERR = None
except Exception as _exc:  # no Triton: the torch-only tiers still work
    _KERNEL_ERR = repr(_exc)[:200]
try:
    from kernels.fused import qk_norm_rope_cache, silu_mul
    _FUSED_ERR = None
except Exception as _exc:
    _FUSED_ERR = repr(_exc)[:200]
try:
    from kernels.gemv import EPI_NONE, EPI_RES, EPI_SILU, Gemv, LmHeadArgmax
    _GEMV_ERR = None
except Exception as _exc:
    _GEMV_ERR = repr(_exc)[:200]
try:
    from kernels import fast as _fast
    _FAST_ERR = None
except Exception as _exc:
    _FAST_ERR = repr(_exc)[:200]
try:
    from kernels.mega import MegaStep
    _MEGA_ERR = None
except Exception as _exc:
    _MEGA_ERR = repr(_exc)[:200]
try:
    from kernels import tree as _tree
    from spec import SeqDraft, Tree
    _TREE_ERR = None
except Exception as _exc:
    _TREE_ERR = repr(_exc)[:200]

DEVICE = os.environ.get("ENGINE_DEVICE", "cuda:0")
CUDA = DEVICE.startswith("cuda")
N_HEADS, N_KV, HEAD_DIM = 32, 8, 128
Q_SIZE, KV_SIZE = N_HEADS * HEAD_DIM, N_KV * HEAD_DIM
BF16 = torch.bfloat16

CHECK_STEPS = 32          # self-check length (tokens) on the warmup prompt
CHECK_ROWS = 4            # rows run through the native loop for the divergence report
MARGIN_LIMIT = 1.0        # native's own drift is <= 0.75; the judge's margin is 2.0
PREFILL_TOKENS = 16384    # prefill processes at most this many tokens per row-chunk
FORCE_TIER = int(os.environ.get("ENGINE_TIER", "-1"))
SPEC = os.environ.get("ENGINE_SPEC", "0") != "0"   # prompt-lookup spec: off until tree/lookahead
SPEC_MAX_ROWS = 64        # verify runs B*(k+1) rows through the skinny GEMMs
SPEC_CHECK_TOKENS = 128   # warmup tokens used to validate and time speculative decoding
SPEC_MIN_GAIN = 0.92      # keep spec only if its warmup wall time is below this fraction of plain
WARMUP_SOFT_S = 170.0     # past this many seconds since load start, skip optional warmup work
WARMUP_HARD_S = 200.0     # past this, keep the best tier found so far (load + warmup limit: 300 s)
SPEC_TREE = os.environ.get("ENGINE_TREE", "1") != "0"   # exact token-tree speculation (T5/T6)
TREE_CHECK_TOKENS = 128   # warmup tokens used to validate and time tree speculation
TREE_MIN_GAIN = 0.92      # keep it only if its warmup wall time is below this fraction of plain

GEMV_MAX_B = 64          # T4's skinny-GEMM decode path covers batches up to this

#            graphs  triton  fused  gemv   fast   mega
TIERS = {6: (True, True, True, False, False, True), 5: (True, True, True, False, True, False),
         4: (True, True, True, True, False, False), 3: (True, True, True, False, False, False),
         2: (True, True, False, False, False, False), 1: (False, False, False, False, False, False)}
COMPARE_TIERS = 3         # time up to this many passing tiers at warmup and keep the fastest


def log(msg):
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _repeat_kv(x):
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, N_HEADS // N_KV, s, d).reshape(b, N_HEADS, s, d)


def _time_configs(obj, configs, run, iters=20):
    """Set each config on obj, time `run(obj)` with CUDA events, keep the fastest that works."""
    best = None
    for cfg in configs:
        try:
            obj.set_config(cfg)
            for _ in range(3):
                run(obj)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                run(obj)
            e.record()
            e.synchronize()
            t = s.elapsed_time(e) / iters
            if best is None or t < best[0]:
                best = (t, cfg)
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"no config ran for {type(obj).__name__}")
    obj.set_config(best[1])
    return best


class _NullEvent:
    def record(self, *a):
        pass

    def synchronize(self):
        pass


def _event(timing=False):
    return torch.cuda.Event(enable_timing=timing) if CUDA else _NullEvent()


class _Layer:
    __slots__ = ("ln1", "w_qkv", "q_norm", "k_norm", "w_o", "ln2", "w_gu", "w_down")


class _State:
    """Everything shape-dependent for one (batch, prompt_len, max_new_tokens, tier)."""

    def __init__(self, eng, B, S, N, tier):
        self.B, self.S, self.N, self.tier = B, S, N, tier
        self.graphs, self.triton, self.fused, self.gemv, self.fast, self.mega = TIERS[tier]
        # speculative verify width T = k + 1 (0 = off); B*T rows must fit the skinny GEMMs
        T = 7 if B == 1 else 5
        self.spec_T = T if (self.gemv and SPEC and N > 1 and B * T <= SPEC_MAX_ROWS
                            and time.perf_counter() - eng.t_load0 < WARMUP_SOFT_S - 60) else 0
        self.spec = False
        Tt = 16 if B == 1 else 8
        self.tree_T = Tt if ((self.fast or self.mega) and SPEC_TREE and _TREE_ERR is None and N > 1
                             and B * Tt <= SPEC_MAX_ROWS
                             and time.perf_counter() - eng.t_load0 < WARMUP_SOFT_S - 60) else 0
        self.tree_on = False
        self.tree_graph = None
        cap = self.capacity = S + N + self.spec_T + self.tree_T
        n_layers = len(eng.layers)
        # one stacked cache per K/V (layer-major) so the persistent kernel can index layers
        self.k_all = torch.zeros((n_layers, B, N_KV, cap, HEAD_DIM), dtype=BF16, device=DEVICE)
        self.v_all = torch.zeros((n_layers, B, N_KV, cap, HEAD_DIM), dtype=BF16, device=DEVICE)
        self.k_cache = [self.k_all[i] for i in range(n_layers)]
        self.v_cache = [self.v_all[i] for i in range(n_layers)]
        with torch.inference_mode():
            dummy = torch.empty(1, dtype=BF16, device=DEVICE)
            cos, sin = eng.rotary(dummy, torch.arange(cap, device=DEVICE)[None])  # reference module
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()          # [cap, 128] BF16
        self.ids = torch.zeros((B, S), dtype=torch.int64, device=DEVICE)
        self.tok = torch.zeros((B,), dtype=torch.int64, device=DEVICE)
        self.pos = torch.zeros((1,), dtype=torch.int64, device=DEVICE)
        self.rows = max(1, min(B, PREFILL_TOKENS // S))
        self.attn = DecodeAttention(B, N_KV, cap, DEVICE, eng.sm_count) if self.triton else None
        self.attn_out = torch.empty((B, N_HEADS, HEAD_DIM), dtype=BF16, device=DEVICE)
        self.q_buf = torch.empty((B, N_HEADS, 1, HEAD_DIM), dtype=BF16, device=DEVICE)
        self.host = torch.empty((max(N, 1), B), dtype=torch.int64, pin_memory=CUDA)
        self.ids_host = torch.empty((B, S), dtype=torch.int64, pin_memory=CUDA)
        self.events = [_event() for _ in range(max(N, 1))]
        self.prefill_graph = self.decode_graph = self.verify_graph = None
        self.prefill_logits = None
        self.compile_s = self.capture_s = 0.0
        self.gqa = eng.gqa_flash and self.triton
        if self.gemv:
            self.plans, self.qkv_buf, self.act_buf = eng._make_plans(B)
        if self.fast:
            eng._fast_plans(self)
        if self.mega:
            H, I = eng.embed.shape[1], eng.inter
            self.x = torch.empty((B, H), dtype=BF16, device=DEVICE)
            self.ss = torch.zeros((2 * n_layers + 1, B), dtype=torch.float32, device=DEVICE)
            self.qkv_buf = torch.empty((B, eng.w_qkv_all.shape[1]), dtype=BF16, device=DEVICE)
            self.act_buf = torch.empty((B, I), dtype=BF16, device=DEVICE)
            self.megastep = MegaStep(eng, self)
        if self.tree_T:
            try:
                eng._tree_setup(self)
            except Exception as exc:                    # speculation is optional: never the tier
                log(f"tree speculation disabled at setup: {repr(exc)[:160]}")
                self.tree_T = 0
        if self.spec_T:
            T = self.spec_T
            self.vplans, self.v_qkv, self.v_act = eng._make_plans(B * T)
            self.vattn = VerifyAttention(B, T, N_KV, cap, DEVICE, eng.sm_count)
            self.v_tok = torch.zeros((B, T), dtype=torch.int64, device=DEVICE)
            self.v_pos = torch.zeros((B,), dtype=torch.int64, device=DEVICE)
            self.v_q = torch.empty((B, T, N_HEADS, HEAD_DIM), dtype=BF16, device=DEVICE)
            self.v_attn_out = torch.empty((B, T, N_HEADS, HEAD_DIM), dtype=BF16, device=DEVICE)
            self.v_out = torch.zeros((B * T,), dtype=torch.int64, device=DEVICE)
            self.v_tok_host = torch.empty((B, T), dtype=torch.int64, pin_memory=CUDA)
            self.v_pos_host = torch.empty((B,), dtype=torch.int64, pin_memory=CUDA)
            self.v_out_host = torch.empty((B * T,), dtype=torch.int64, pin_memory=CUDA)
            self.v_event = _event()


class Engine:
    def __init__(self, model_path: str) -> None:
        t0 = self.t_load0 = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.hf = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=BF16, attn_implementation="sdpa", local_files_only=True,
            ).eval().to(DEVICE)
        )
        base = self.hf.model
        cfg = self.hf.config
        self.eps = cfg.rms_norm_eps
        self.inter = cfg.intermediate_size
        self.embed = base.embed_tokens.weight
        self.lm_head = self.hf.lm_head.weight                  # tied: same storage as embed
        self.final_norm = base.norm.weight
        self.rotary = base.rotary_emb
        self.layers = []
        with torch.no_grad():
            # layer-stacked, packed weights; every tier uses views into them (the persistent kernel
            # addresses layer l at a fixed stride). The native model keeps its own copies until the
            # warmup self-check is done.
            NL, H = len(base.layers), cfg.hidden_size
            a0, m0 = base.layers[0].self_attn, base.layers[0].mlp
            nq, nkv, I = a0.q_proj.weight.shape[0], a0.k_proj.weight.shape[0], cfg.intermediate_size
            new = lambda *shape: torch.empty(shape, dtype=BF16, device=DEVICE)  # noqa: E731
            self.w_qkv_all = new(NL, nq + 2 * nkv, H)
            self.w_o_all = new(NL, H, a0.o_proj.weight.shape[1])
            self.w_gu_all = new(NL, 2 * I, H)
            self.w_d_all = new(NL, H, I)
            self.ln1_all, self.ln2_all = new(NL, H), new(NL, H)
            self.qn_all, self.kn_all = new(NL, HEAD_DIM), new(NL, HEAD_DIM)
            for i, layer in enumerate(base.layers):
                at, mlp = layer.self_attn, layer.mlp
                self.w_qkv_all[i, :nq].copy_(at.q_proj.weight)
                self.w_qkv_all[i, nq:nq + nkv].copy_(at.k_proj.weight)
                self.w_qkv_all[i, nq + nkv:].copy_(at.v_proj.weight)
                self.w_o_all[i].copy_(at.o_proj.weight)
                self.w_gu_all[i, :I].copy_(mlp.gate_proj.weight)
                self.w_gu_all[i, I:].copy_(mlp.up_proj.weight)
                self.w_d_all[i].copy_(mlp.down_proj.weight)
                self.ln1_all[i].copy_(layer.input_layernorm.weight)
                self.ln2_all[i].copy_(layer.post_attention_layernorm.weight)
                self.qn_all[i].copy_(at.q_norm.weight)
                self.kn_all[i].copy_(at.k_norm.weight)
                L = _Layer()
                L.ln1, L.ln2 = self.ln1_all[i], self.ln2_all[i]
                L.q_norm, L.k_norm = self.qn_all[i], self.kn_all[i]
                L.w_qkv, L.w_o, L.w_gu, L.w_down = self.w_qkv_all[i], self.w_o_all[i], self.w_gu_all[i], self.w_d_all[i]
                self.layers.append(L)
        self.sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count if CUDA else 1
        self.state = None
        self.tier = None
        self._tuned = {}
        self._tuned_fast = {}
        self.gqa_flash = self._probe_gqa_flash()
        self.load_s = time.perf_counter() - t0
        log(f"loaded in {self.load_s:.1f}s; kernels {'ok' if _KERNEL_ERR is None else 'UNAVAILABLE ' + _KERNEL_ERR}; "
            f"fused {'ok' if _FUSED_ERR is None else 'UNAVAILABLE ' + _FUSED_ERR}; "
            f"T5 {'ok' if _FAST_ERR is None else 'UNAVAILABLE ' + _FAST_ERR}; "
            f"T6 {'ok' if _MEGA_ERR is None else 'UNAVAILABLE ' + _MEGA_ERR}; gqa_flash={self.gqa_flash}")

    def _make_plans(self, M):
        """T4 skinny-GEMM plans for M rows (tuned once per M; tuning compiles every kept config)."""
        H, I = self.embed.shape[1], self.inter
        L = self.layers[0]
        qkv_buf = torch.empty((M, L.w_qkv.shape[0]), dtype=BF16, device=DEVICE)
        act_buf = torch.empty((M, I), dtype=BF16, device=DEVICE)
        plans = {
            "qkv": Gemv(M, L.w_qkv.shape[0], H, EPI_NONE, DEVICE, self.sm_count),
            "o": Gemv(M, H, L.w_o.shape[1], EPI_RES, DEVICE, self.sm_count),
            "gu": Gemv(M, I, H, EPI_SILU, DEVICE, self.sm_count),
            "down": Gemv(M, H, I, EPI_RES, DEVICE, self.sm_count),
            "lm": LmHeadArgmax(M, self.lm_head.shape[0], H, DEVICE),
        }
        cached = self._tuned.get(M)
        if cached is None:
            t0 = time.perf_counter()
            g = torch.Generator(device=DEVICE).manual_seed(0)
            rnd = lambda n: torch.randn((M, n), generator=g, device=DEVICE).to(BF16)  # noqa: E731
            res = torch.zeros((M, H), dtype=BF16, device=DEVICE)
            scratch = torch.empty((M, H), dtype=BF16, device=DEVICE)
            tok = torch.empty((M,), dtype=torch.int64, device=DEVICE)
            jobs = {
                "qkv": lambda pl: pl.tune(rnd(H), L.w_qkv, qkv_buf),
                "o": lambda pl: pl.tune(rnd(L.w_o.shape[1]), L.w_o, scratch, res),
                "gu": lambda pl: pl.tune(rnd(H), L.w_gu, act_buf),
                "down": lambda pl: pl.tune(rnd(I), L.w_down, scratch, res),
                "lm": lambda pl: pl.tune(rnd(H), self.lm_head, tok),
            }
            cached, desc = {}, []
            for k, plan in plans.items():
                t, cfg = jobs[k](plan)
                cached[k] = cfg
                desc.append(f"{k}={cfg[0]}x{cfg[1]}w{cfg[2]}s{cfg[3]}:{t * 1e3:.0f}us")
            self._tuned[M] = cached
            log(f"gemv tuned M={M} in {time.perf_counter() - t0:.1f}s: {' '.join(desc)}")
        for k, plan in plans.items():
            plan.set_config(cached[k])
        return plans, qkv_buf, act_buf

    def _decode_gemv(self, st):
        """T4 decode step (run 2's form): Triton RMSNorm + skinny GEMMs with fused epilogues."""
        B = st.B
        p = st.plans
        x = F.embedding(st.tok, self.embed)
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            h = rms_norm_rows(x, L.ln1, self.eps).view(B, -1)
            qkv = p["qkv"](h, L.w_qkv, st.qkv_buf)
            qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, st.q_buf, kc, vc,
                               B, 1, self.eps, 1)
            a = st.attn(st.q_buf.view(B, N_HEADS, HEAD_DIM), kc, vc, st.pos, st.attn_out)
            p["o"](a.view(B, Q_SIZE), L.w_o, x, res=x)
            h = rms_norm_rows(x, L.ln2, self.eps).view(B, -1)
            act = p["gu"](h, L.w_gu, st.act_buf)
            p["down"](act, L.w_down, x, res=x)
        h = rms_norm_rows(x, self.final_norm, self.eps).view(B, -1)
        p["lm"](h, self.lm_head, st.tok)
        st.pos.add_(1)

    # ------------------------------------------------------------------ T5
    def _fast_gemms(self, M):
        """T5 GEMM set + LM head for M rows; configs (tile, warps, stages, split-K) tuned once per M."""
        H, I, V = self.embed.shape[1], self.inter, self.lm_head.shape[0]
        L = self.layers[0]
        fg = {
            "qkv": _fast.FastGemm(M, L.w_qkv.shape[0], H, _fast.EPI_NONE, DEVICE, norm=True),
            "o": _fast.FastGemm(M, H, L.w_o.shape[1], _fast.EPI_RES, DEVICE, ss_out=True),
            "gu": _fast.FastGemm(M, I, H, _fast.EPI_SILU, DEVICE, norm=True),
            "down": _fast.FastGemm(M, H, I, _fast.EPI_RES, DEVICE, ss_out=True),
        }
        for g in fg.values():
            g.eps = self.eps
        lm = _fast.FastLmHead(M, V, H, DEVICE)
        lm.eps = self.eps
        cached = self._tuned_fast.get(M)
        if cached is None:
            cached = {k: g.cfg for k, g in fg.items()}
            cached["lm"] = lm.cfg
            if CUDA and time.perf_counter() - self.t_load0 < WARMUP_SOFT_S - 60:
                t0 = time.perf_counter()
                gen = torch.Generator(device=DEVICE).manual_seed(0)
                rnd = lambda n: torch.randn((M, n), generator=gen, device=DEVICE).to(BF16)  # noqa: E731
                ss = torch.full((M,), float(H), dtype=torch.float32, device=DEVICE)
                ss_out = torch.zeros((M,), dtype=torch.float32, device=DEVICE)
                scratch = torch.zeros((M, H), dtype=BF16, device=DEVICE)
                qkv_o = torch.empty((M, L.w_qkv.shape[0]), dtype=BF16, device=DEVICE)
                act_o = torch.empty((M, I), dtype=BF16, device=DEVICE)
                tok = torch.zeros((M,), dtype=torch.int64, device=DEVICE)
                posb = torch.zeros((1,), dtype=torch.int64, device=DEVICE)
                jobs = {
                    "qkv": lambda g: g(rnd(H), L.w_qkv, qkv_o, norm_w=L.ln1, ss_in=ss),
                    "o": lambda g: g(rnd(L.w_o.shape[1]), L.w_o, scratch, res=scratch, ss_out=ss_out),
                    "gu": lambda g: g(rnd(H), L.w_gu, act_o, norm_w=L.ln2, ss_in=ss),
                    "down": lambda g: g(rnd(I), L.w_down, scratch, res=scratch, ss_out=ss_out),
                }
                desc = []
                for k, g in fg.items():
                    t, cfg = _time_configs(g, g.candidates(), jobs[k])
                    cached[k] = cfg
                    desc.append(f"{k}={cfg[0]}x{cfg[1]}w{cfg[2]}s{cfg[3]}k{cfg[4]}:{t * 1e3:.0f}us")
                xl = rnd(H)
                t, cfg = _time_configs(lm, lm.candidates(),
                                       lambda m: m(xl, self.lm_head, tok, self.final_norm, ss, posb, False))
                cached["lm"] = cfg
                desc.append(f"lm={cfg[0]}x{cfg[1]}:{t * 1e3:.0f}us")
                log(f"fast GEMMs tuned M={M} in {time.perf_counter() - t0:.1f}s: {' '.join(desc)}")
            self._tuned_fast[M] = cached
        for k, g in fg.items():
            g.set_config(cached[k])
        lm.set_config(cached["lm"])
        return fg, lm

    def _fast_plans(self, st):
        """T5 buffers and kernels for batch B."""
        B, H, I = st.B, self.embed.shape[1], self.inter
        L = self.layers[0]
        st.x = torch.empty((B, H), dtype=BF16, device=DEVICE)
        st.ss = torch.zeros((2 * len(self.layers) + 1, B), dtype=torch.float32, device=DEVICE)
        st.qkv_buf = torch.empty((B, L.w_qkv.shape[0]), dtype=BF16, device=DEVICE)
        st.act_buf = torch.empty((B, I), dtype=BF16, device=DEVICE)
        st.fg, st.flm = self._fast_gemms(B)
        st.fattn = _fast.FastAttention(B, N_KV, st.capacity, DEVICE, self.sm_count)

    def _tree_setup(self, st):
        """Buffers and kernels for token-tree verification of T nodes per sequence (M = B*T rows)."""
        from types import SimpleNamespace
        B, T = st.B, st.tree_T
        M, H, I = B * T, self.embed.shape[1], self.inter
        L = self.layers[0]
        tr = SimpleNamespace()
        tr.fg, tr.lm = self._fast_gemms(M)
        tr.attn = _tree.TreeAttention(B, T, N_KV, st.capacity, DEVICE, self.sm_count)
        i64, i32 = torch.int64, torch.int32
        dev = lambda shape, dt: torch.zeros(shape, dtype=dt, device=DEVICE)  # noqa: E731
        pin = lambda shape, dt: torch.zeros(shape, dtype=dt, pin_memory=CUDA)  # noqa: E731
        tr.tok, tr.depth, tr.base, tr.anc = dev((B, T), i64), dev((M,), i64), dev((B,), i64), dev((B, T), i32)
        tr.c_base, tr.c_src, tr.c_cnt = dev((B,), i64), dev((B, T), i32), dev((B,), i32)
        tr.tok_h, tr.depth_h, tr.base_h, tr.anc_h = pin((B, T), i64), pin((M,), i64), pin((B,), i64), pin((B, T), i32)
        tr.c_base_h, tr.c_src_h, tr.c_cnt_h = pin((B,), i64), pin((B, T), i32), pin((B,), i32)
        tr.out, tr.out_h = dev((M,), i64), pin((M,), i64)
        tr.x = torch.empty((M, H), dtype=BF16, device=DEVICE)
        tr.ss = dev((2 * len(self.layers) + 1, M), torch.float32)
        tr.qkv = torch.empty((M, L.w_qkv.shape[0]), dtype=BF16, device=DEVICE)
        tr.q = torch.empty((B, T, N_HEADS, HEAD_DIM), dtype=BF16, device=DEVICE)
        tr.attn_out = torch.empty((B, T, N_HEADS, HEAD_DIM), dtype=BF16, device=DEVICE)
        tr.act = torch.empty((M, I), dtype=BF16, device=DEVICE)
        tr.dpos = dev((1,), i64)
        tr.event = _event()
        st.tr = tr

    def _verify_tree(self, st):
        """One tree-verification forward (graph-capturable): apply the previous acceptance's KV
        compaction, then run the B*T tree nodes through the model; st.tr.out = argmax per node."""
        tr, B, T = st.tr, st.B, st.tree_T
        M = B * T
        g = tr.fg
        _tree.compact(st.k_all, st.v_all, tr.c_base, tr.c_src, tr.c_cnt, T)
        _fast.embed_ss(tr.tok.view(-1), self.embed, tr.x, tr.ss)
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            qkv = g["qkv"](tr.x, L.w_qkv, tr.qkv, norm_w=L.ln1, ss_in=tr.ss[2 * i])
            _tree.tree_qk(qkv, L.q_norm, L.k_norm, st.cos, st.sin, tr.base, tr.depth, tr.q, kc, vc, B, T, self.eps)
            a = tr.attn(tr.q, kc, vc, tr.base, tr.anc, tr.attn_out)
            g["o"](a.view(M, Q_SIZE), L.w_o, tr.x, res=tr.x, ss_out=tr.ss[2 * i + 1])
            act = g["gu"](tr.x, L.w_gu, tr.act, norm_w=L.ln2, ss_in=tr.ss[2 * i + 1])
            g["down"](act, L.w_down, tr.x, res=tr.x, ss_out=tr.ss[2 * i + 2])
        tr.lm(tr.x, self.lm_head, tr.out, self.final_norm, tr.ss[2 * len(self.layers)], tr.dpos, False)

    def _decode_fast(self, st):
        """T5 decode step: 1 + 5 per layer + 1 launches. Norm statistics live in st.ss:
        slot 2i = input_layernorm of layer i, 2i+1 = post_attention_layernorm, last = final norm."""
        B = st.B
        g = st.fg
        x = st.x
        _fast.embed_ss(st.tok, self.embed, x, st.ss)
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            qkv = g["qkv"](x, L.w_qkv, st.qkv_buf, norm_w=L.ln1, ss_in=st.ss[2 * i])
            a = st.fattn(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, kc, vc, st.attn_out, self.eps)
            g["o"](a.view(B, Q_SIZE), L.w_o, x, res=x, ss_out=st.ss[2 * i + 1])
            act = g["gu"](x, L.w_gu, st.act_buf, norm_w=L.ln2, ss_in=st.ss[2 * i + 1])
            g["down"](act, L.w_down, x, res=x, ss_out=st.ss[2 * i + 2])
        st.flm(x, self.lm_head, st.tok, self.final_norm, st.ss[2 * len(self.layers)], st.pos, True)

    def _verify(self, st):
        """Speculative verify: T tokens per sequence (st.v_tok) at positions st.v_pos[b] + t.
        Writes their K/V and the greedy token after every prefix into st.v_out [B*T]."""
        B, T = st.B, st.spec_T
        p = st.vplans
        x = F.embedding(st.v_tok.view(-1), self.embed)          # [B*T, H], row = b*T + t
        q_strides = (T * Q_SIZE, HEAD_DIM, Q_SIZE)               # v_q is [B, T, 32, 128]
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            h = rms_norm_rows(x, L.ln1, self.eps).view(B * T, -1)
            qkv = p["qkv"](h, L.w_qkv, st.v_qkv)
            qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.v_pos, st.v_q, kc, vc,
                               B, T, self.eps, 2, q_strides=q_strides)
            a = st.vattn(st.v_q, kc, vc, st.v_pos, st.v_attn_out)
            p["o"](a.view(B * T, Q_SIZE), L.w_o, x, res=x)
            h = rms_norm_rows(x, L.ln2, self.eps).view(B * T, -1)
            act = p["gu"](h, L.w_gu, st.v_act)
            p["down"](act, L.w_down, x, res=x)
        h = rms_norm_rows(x, self.final_norm, self.eps).view(B * T, -1)
        p["lm"](h, self.lm_head, st.v_out)

    # ------------------------------------------------------------------ building blocks
    def _norm(self, st, x2d, w, heads=1):
        """Qwen3RMSNorm over rows of width w.numel(); x2d [M, >=heads*N] -> [M, heads, N]."""
        if st.triton:
            return rms_norm_rows(x2d, w, self.eps, heads)
        n = w.numel()
        xf = x2d[:, :heads * n].reshape(x2d.shape[0], heads, n).to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return w * xf.to(BF16)

    def _act(self, st, gu):
        if st.fused:
            return silu_mul(gu, self.inter)
        return F.silu(gu[:, :self.inter]) * gu[:, self.inter:]

    @staticmethod
    def _sdpa_causal(st, q, k, v):
        """Causal prompt attention. With GQA-native flash (torch 2.5 enable_gqa) the 8 KV heads are
        indexed directly - the same arithmetic as the reference's repeat_kv + flash, without the copy."""
        if st.gqa:
            return F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                                  is_causal=True, scale=HEAD_DIM ** -0.5, enable_gqa=True)
        return F.scaled_dot_product_attention(q.contiguous(), _repeat_kv(k), _repeat_kv(v),
                                              is_causal=True, scale=HEAD_DIM ** -0.5)

    @staticmethod
    def _sdpa_last(st, q, k, v):
        if st.gqa:
            return F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                                  scale=HEAD_DIM ** -0.5, enable_gqa=True)
        return F.scaled_dot_product_attention(q.contiguous(), _repeat_kv(k), _repeat_kv(v), scale=HEAD_DIM ** -0.5)

    def _probe_gqa_flash(self):
        """True if torch's flash backend runs GQA natively here and matches repeat_kv exactly."""
        if not CUDA:
            return False
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            g = torch.Generator(device=DEVICE).manual_seed(0)
            q = torch.randn((2, N_HEADS, 70, HEAD_DIM), generator=g, device=DEVICE).to(BF16)
            k = torch.randn((2, N_KV, 70, HEAD_DIM), generator=g, device=DEVICE).to(BF16)
            v = torch.randn((2, N_KV, 70, HEAD_DIM), generator=g, device=DEVICE).to(BF16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=HEAD_DIM ** -0.5, enable_gqa=True)
            ref = F.scaled_dot_product_attention(q, _repeat_kv(k), _repeat_kv(v), is_causal=True, scale=HEAD_DIM ** -0.5)
            return bool(torch.equal(a, ref))
        except Exception:
            return False

    def _prefill(self, st):
        """Prompt forward in row chunks; fills KV [0, S), writes token 0 to st.tok, returns logits."""
        B, S = st.B, st.S
        cos = st.cos[:S][None, None]
        sin = st.sin[:S][None, None]
        last_layer = len(self.layers) - 1
        outs = []
        for r0 in range(0, B, st.rows):
            r1 = min(B, r0 + st.rows)
            b, M = r1 - r0, (r1 - r0) * S
            x = F.embedding(st.ids[r0:r1], self.embed).view(M, -1)
            for i, L in enumerate(self.layers):
                kc, vc = st.k_cache[i][r0:r1], st.v_cache[i][r0:r1]
                h = self._norm(st, x, L.ln1).view(M, -1)
                qkv = F.linear(h, L.w_qkv)
                if st.fused:
                    q = torch.empty((b, N_HEADS, S, HEAD_DIM), dtype=BF16, device=DEVICE)
                    qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, q, kc, vc,
                                       b, S, self.eps, 0)
                else:
                    q = self._norm(st, qkv, L.q_norm, N_HEADS).view(b, S, N_HEADS, HEAD_DIM).transpose(1, 2)
                    k = self._norm(st, qkv[:, Q_SIZE:], L.k_norm, N_KV).view(b, S, N_KV, HEAD_DIM).transpose(1, 2)
                    v = qkv[:, Q_SIZE + KV_SIZE:].view(b, S, N_KV, HEAD_DIM).transpose(1, 2)
                    q = (q * cos) + (_rotate_half(q) * sin)
                    k = (k * cos) + (_rotate_half(k) * sin)
                    kc[:, :, :S].copy_(k)
                    vc[:, :, :S].copy_(v)
                if i == last_layer:
                    # only the last position reaches the LM head: its causal row sees every key
                    a = self._sdpa_last(st, q[:, :, -1:], kc[:, :, :S], vc[:, :, :S])
                    x = x.view(b, S, -1)[:, -1] + F.linear(a.reshape(b, Q_SIZE), L.w_o)
                    h = self._norm(st, x, L.ln2).view(b, -1)
                    x = x + F.linear(self._act(st, F.linear(h, L.w_gu)), L.w_down)
                    break
                a = self._sdpa_causal(st, q, kc[:, :, :S], vc[:, :, :S])
                x = x + F.linear(a.transpose(1, 2).reshape(M, Q_SIZE), L.w_o)
                h = self._norm(st, x, L.ln2).view(M, -1)
                x = x + F.linear(self._act(st, F.linear(h, L.w_gu)), L.w_down)
            last = x
            outs.append(F.linear(self._norm(st, last, self.final_norm).view(b, -1), self.lm_head))
        logits = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
        st.tok.copy_(torch.argmax(logits, dim=-1))
        return logits

    def _decode(self, st, host_pos=None):
        """One token per sequence at st.pos (device) or host_pos (eager T1); advances st.pos."""
        if st.mega:
            return st.megastep()
        if st.fast:
            return self._decode_fast(st)
        if st.gemv:
            return self._decode_gemv(st)
        B = st.B
        x = F.embedding(st.tok, self.embed)
        if host_pos is None:
            cos = st.cos.index_select(0, st.pos)[None]           # [1, 1, 128]
            sin = st.sin.index_select(0, st.pos)[None]
        else:
            cos = st.cos[host_pos:host_pos + 1][None]
            sin = st.sin[host_pos:host_pos + 1][None]
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            h = self._norm(st, x, L.ln1).view(B, -1)
            qkv = F.linear(h, L.w_qkv)
            if st.fused:
                qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, st.q_buf, kc, vc,
                                   B, 1, self.eps, 1)
                q = st.q_buf.view(B, N_HEADS, HEAD_DIM)
            else:
                q = self._norm(st, qkv, L.q_norm, N_HEADS)
                k = self._norm(st, qkv[:, Q_SIZE:], L.k_norm, N_KV)
                v = qkv[:, Q_SIZE + KV_SIZE:].view(B, N_KV, HEAD_DIM)
                q = (q * cos) + (_rotate_half(q) * sin)
                k = (k * cos) + (_rotate_half(k) * sin)
                if host_pos is None:
                    kc.index_copy_(2, st.pos, k[:, :, None])
                    vc.index_copy_(2, st.pos, v[:, :, None])
                else:
                    kc[:, :, host_pos].copy_(k)
                    vc[:, :, host_pos].copy_(v)
            if st.triton:
                a = st.attn(q.contiguous(), kc, vc, st.pos, st.attn_out).view(B, Q_SIZE)
            else:
                n = host_pos + 1
                a = F.scaled_dot_product_attention(
                    q.view(B, N_HEADS, 1, HEAD_DIM).contiguous(), _repeat_kv(kc[:, :, :n]), _repeat_kv(vc[:, :, :n]),
                    scale=HEAD_DIM ** -0.5).transpose(1, 2).reshape(B, Q_SIZE)
            x = x + F.linear(a, L.w_o)
            h = self._norm(st, x, L.ln2).view(B, -1)
            x = x + F.linear(self._act(st, F.linear(h, L.w_gu)), L.w_down)
        logits = F.linear(self._norm(st, x, self.final_norm).view(B, -1), self.lm_head)
        st.tok.copy_(torch.argmax(logits, dim=-1))
        st.pos.add_(1)
        return logits

    # ------------------------------------------------------------------ shapes, graphs
    def _build(self, B, S, N, tier):
        if tier in (4, 5, 6) and B > GEMV_MAX_B:
            tier = 3
        self.state = None
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()
        st = _State(self, B, S, N, tier)
        if st.graphs:
            with torch.inference_mode():
                t0 = time.perf_counter()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):          # JIT-compiles every Triton specialisation, warms cuBLAS
                        self._prefill(st)
                        if N > 1:
                            st.pos.fill_(S)
                            self._decode(st)
                        if st.spec_T:
                            st.v_pos.fill_(S)
                            self._verify(st)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                if st.mega and int(st.megastep.err.item()) != 0:
                    raise RuntimeError(f"T6 dependency spin expired ({st.megastep.describe()})")
                if st.tree_T:
                    try:
                        with torch.cuda.stream(side):
                            for _ in range(2):
                                st.tr.base.fill_(S)
                                self._verify_tree(st)
                        torch.cuda.current_stream().wait_stream(side)
                        torch.cuda.synchronize()
                    except Exception as exc:
                        log(f"tree speculation disabled at warmup: {repr(exc)[:160]}")
                        st.tree_T = 0
                st.compile_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                try:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        st.prefill_logits = self._prefill(st)
                    st.prefill_graph = g
                    if N > 1:
                        st.pos.fill_(S)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            self._decode(st)
                        st.decode_graph = g
                    if st.spec_T:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            self._verify(st)
                        st.verify_graph = g
                    if st.tree_T:
                        try:
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                self._verify_tree(st)
                            st.tree_graph = g
                        except Exception as exc:
                            log(f"tree speculation disabled at capture: {repr(exc)[:160]}")
                            st.tree_T, st.tree_graph = 0, None
                except Exception as exc:
                    log(f"graph capture failed, T{tier} runs eager: {repr(exc)[:160]}")
                    st.prefill_graph = st.decode_graph = st.verify_graph = None
                    st.spec_T = 0
                torch.cuda.synchronize()
                st.capture_s = time.perf_counter() - t0
        self.state = st
        return st

    def _stream(self, st, input_ids, N, timing=None):
        """Greedy stream of N steps on st. Keeps the GPU one step ahead of the host."""
        B, S = st.B, st.S
        if timing is not None:
            ev = [(_event(True), _event(True)) for _ in range(N)]
        with torch.inference_mode():
            st.ids_host.copy_(torch.tensor(input_ids, dtype=torch.int64))
            st.ids.copy_(st.ids_host, non_blocking=True)
            st.pos.fill_(S)
            if timing is not None:
                ev[0][0].record()
            if st.prefill_graph is not None:
                st.prefill_graph.replay()
            else:
                st.prefill_logits = self._prefill(st)
            if timing is not None:
                ev[0][1].record()
            st.host[0].copy_(st.tok, non_blocking=True)
            st.events[0].record()
        for t in range(1, N):
            with torch.inference_mode():
                if timing is not None:
                    ev[t][0].record()
                if st.decode_graph is not None:
                    st.decode_graph.replay()
                elif st.graphs or st.triton:
                    self._decode(st)
                else:
                    self._decode(st, host_pos=S + t - 1)
                if timing is not None:
                    ev[t][1].record()
                st.host[t].copy_(st.tok, non_blocking=True)
                st.events[t].record()
            st.events[t - 1].synchronize()
            yield st.host[t - 1].tolist()
        st.events[N - 1].synchronize()
        if timing is not None and CUDA:
            timing.extend(a.elapsed_time(b) for a, b in ev)
        yield st.host[N - 1].tolist()

    def _stream_spec(self, st, input_ids, N):
        """Exact prompt-lookup speculative decoding (see NOTES.md, Phase 4). Yields exactly N steps."""
        B, S, T = st.B, st.S, st.spec_T
        k = T - 1
        with torch.inference_mode():
            st.ids_host.copy_(torch.tensor(input_ids, dtype=torch.int64))
            st.ids.copy_(st.ids_host, non_blocking=True)
            st.pos.fill_(S)
            st.prefill_graph.replay()
            st.host[0].copy_(st.tok, non_blocking=True)
            st.events[0].record()
        st.events[0].synchronize()
        first = st.host[0].tolist()
        yield first
        if N == 1:
            return
        seqs = [list(p) + [t] for p, t in zip(input_ids, first)]
        idx = [({}, {}) for _ in range(B)]                     # n=3, n=2: n-gram -> index after it
        reg = [1] * B                                          # next n-gram end index to register

        def register(b):
            seq, (i3, i2) = seqs[b], idx[b]
            for j in range(reg[b], len(seq) - 1):             # only n-grams whose continuation exists
                i2[(seq[j - 1], seq[j])] = j + 1
                if j >= 2:
                    i3[(seq[j - 2], seq[j - 1], seq[j])] = j + 1
            reg[b] = len(seq) - 1

        for b in range(B):
            register(b)
        vt, vp, vo = st.v_tok_host, st.v_pos_host, st.v_out_host
        yielded = 1
        while yielded < N:
            rows = []
            for b in range(B):
                seq = seqs[b]
                last = seq[-1]
                if len(seq) - S >= N:                          # done: frozen at a slot inside capacity
                    rows.append([last] * T)
                    vp[b] = S
                    continue
                p = idx[b][0].get((seq[-3], seq[-2], last))
                if p is None:
                    p = idx[b][1].get((seq[-2], last))
                d = seq[p:p + k] if p is not None else []
                d = d + [last] * (k - len(d))
                rows.append([last] + d)
                vp[b] = len(seq) - 1
            vt.copy_(torch.tensor(rows, dtype=torch.int64))
            with torch.inference_mode():
                st.v_tok.copy_(vt, non_blocking=True)
                st.v_pos.copy_(vp, non_blocking=True)
                st.verify_graph.replay()
                vo.copy_(st.v_out, non_blocking=True)
                st.v_event.record()
            st.v_event.synchronize()
            g = vo.tolist()
            for b in range(B):
                seq = seqs[b]
                if len(seq) - S >= N:
                    continue
                gb, row = g[b * T:(b + 1) * T], rows[b]
                a = 0
                while a < k and row[a + 1] == gb[a]:
                    a += 1
                seq.extend(gb[:a + 1])
                register(b)
            while yielded < N and all(len(sq) - S > yielded for sq in seqs):
                yield [sq[S + yielded] for sq in seqs]
                yielded += 1

    def _stream_tree(self, st, input_ids, N):
        """Exact token-tree speculative decoding (NOTES.md, Phase 4 v2). Yields exactly N steps."""
        tr, B, S, T = st.tr, st.B, st.S, st.tree_T
        maxd = min(T - 1, 8)
        with torch.inference_mode():
            st.ids_host.copy_(torch.tensor(input_ids, dtype=torch.int64))
            st.ids.copy_(st.ids_host, non_blocking=True)
            st.pos.fill_(S)
            if st.prefill_graph is not None:
                st.prefill_graph.replay()
            else:
                st.prefill_logits = self._prefill(st)
            st.host[0].copy_(st.tok, non_blocking=True)
            st.events[0].record()
        st.events[0].synchronize()
        first = st.host[0].tolist()
        yield first
        if N == 1:
            return
        seqs = [list(p) + [t] for p, t in zip(input_ids, first)]
        drafts = [SeqDraft(sq) for sq in seqs]
        c_base, c_src, c_cnt = [S] * B, [[0] * T for _ in range(B)], [1] * B
        yielded = 1
        while yielded < N:
            trees, toks, depths, bases, ancs = [], [], [], [], []
            for b in range(B):
                sq = seqs[b]
                if len(sq) - S >= N:                           # done: frozen at a safe slot
                    tree = Tree(sq[-1], T)
                    tree.pad(T)
                    trees.append(None)
                    bases.append(S)
                else:
                    tree = drafts[b].build(T, maxd)
                    trees.append(tree)
                    bases.append(len(sq) - 1)
                toks.append(tree.tokens)
                depths.extend(tree.depth)
                ancs.append([a - (1 << 32) if a >= 1 << 31 else a for a in tree.anc])
            tr.tok_h.copy_(torch.tensor(toks, dtype=torch.int64))
            tr.depth_h.copy_(torch.tensor(depths, dtype=torch.int64))
            tr.base_h.copy_(torch.tensor(bases, dtype=torch.int64))
            tr.anc_h.copy_(torch.tensor(ancs, dtype=torch.int32))
            tr.c_base_h.copy_(torch.tensor(c_base, dtype=torch.int64))
            tr.c_src_h.copy_(torch.tensor(c_src, dtype=torch.int32))
            tr.c_cnt_h.copy_(torch.tensor(c_cnt, dtype=torch.int32))
            with torch.inference_mode():
                for d, h in ((tr.tok, tr.tok_h), (tr.depth, tr.depth_h), (tr.base, tr.base_h), (tr.anc, tr.anc_h),
                             (tr.c_base, tr.c_base_h), (tr.c_src, tr.c_src_h), (tr.c_cnt, tr.c_cnt_h)):
                    d.copy_(h, non_blocking=True)
                if st.tree_graph is not None:
                    st.tree_graph.replay()
                else:
                    self._verify_tree(st)
                tr.out_h.copy_(tr.out, non_blocking=True)
                tr.event.record()
            tr.event.synchronize()
            g = tr.out_h.tolist()
            for b in range(B):
                tree = trees[b]
                if tree is None:
                    c_cnt[b] = 1
                    continue
                emitted, path = drafts[b].update(tree, g[b * T:(b + 1) * T])
                c_base[b] = bases[b]
                c_src[b] = [0] + path + [0] * (T - 1 - len(path))
                c_cnt[b] = len(path) + 1
                seqs[b].extend(emitted)
                drafts[b].register()
            while yielded < N and all(len(sq) - S > yielded for sq in seqs):
                yield [sq[S + yielded] for sq in seqs]
                yielded += 1

    def _decide_tree(self, st, input_ids, N):
        """Warmup-only: validate tree speculation on this prompt; keep it only if exact
        (teacher-forced margin <= MARGIN_LIMIT) and clearly faster than plain decode."""
        n = min(N, TREE_CHECK_TOKENS)
        try:
            steps = list(self._stream_tree(st, input_ids, n))
            assert len(steps) == n and all(len(x) == st.B for x in steps)
            ours = [list(r) for r in zip(*steps)]
            margin, nonarg, _ = self._teacher_force(input_ids, ours)
            walls = []
            for _ in range(2):
                t0 = time.perf_counter()
                for _ in self._stream_tree(st, input_ids, n):
                    pass
                walls.append((time.perf_counter() - t0) * 1e3)
            tree_ms = min(walls)
            plain_ms, _ = self._time_stream(st, input_ids, n)
            st.tree_on = margin <= MARGIN_LIMIT and tree_ms < TREE_MIN_GAIN * plain_ms
            log(f"tree spec T={st.tree_T}: margin={margin:.3f} non_argmax={nonarg} {n} tok: tree={tree_ms:.1f}ms "
                f"plain={plain_ms:.1f}ms -> {'ON' if st.tree_on else 'off'}")
        except Exception as exc:
            st.tree_on = False
            log(f"tree speculation disabled: {repr(exc)[:200]}")

    def _decide_spec(self, st, input_ids, N):
        """Warmup-only: validate speculative decoding on this prompt and keep it only if it is
        exact (teacher-forced margin <= MARGIN_LIMIT) and clearly faster than plain decode."""
        n = min(N, SPEC_CHECK_TOKENS)
        try:
            steps = list(self._stream_spec(st, input_ids, n))
            assert len(steps) == n and all(len(x) == st.B for x in steps)
            ours = [list(r) for r in zip(*steps)]
            margin, nonarg, _ = self._teacher_force(input_ids, ours)
            walls = []
            for _ in range(2):
                t0 = time.perf_counter()
                for _ in self._stream_spec(st, input_ids, n):
                    pass
                walls.append((time.perf_counter() - t0) * 1e3)
            spec_ms = min(walls)
            plain_ms, _ = self._time_stream(st, input_ids, n)
            st.spec = margin <= MARGIN_LIMIT and spec_ms < SPEC_MIN_GAIN * plain_ms
            log(f"spec k={st.spec_T - 1}: margin={margin:.3f} non_argmax={nonarg} {n} tok: spec={spec_ms:.1f}ms "
                f"plain={plain_ms:.1f}ms -> {'ON' if st.spec else 'off'}")
        except Exception as exc:
            st.spec = False
            log(f"spec disabled: {repr(exc)[:200]}")

    # ------------------------------------------------------------------ native reference (T0 + self-check)
    def _native(self, input_ids, N):
        current = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
        cache = None
        with torch.inference_mode():
            for _ in range(N):
                output = self.hf(input_ids=current, past_key_values=cache, use_cache=True,
                                 logits_to_keep=1, return_dict=True)
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist(), output.logits[:, -1, :]

    def _teacher_force(self, input_ids, toks):
        """Replay prompt + toks through native Qwen in one forward (row chunks).
        Returns (max margin, #non-argmax positions, first step with a non-argmax token or -1)."""
        K = len(toks[0])
        S = len(input_ids[0])
        chunk = max(1, 8192 // (S + K))
        worst, nonarg, first = 0.0, 0, -1
        with torch.inference_mode():
            for r0 in range(0, len(input_ids), chunk):
                seqs = [p + t[:-1] for p, t in zip(input_ids[r0:r0 + chunk], toks[r0:r0 + chunk])]
                ids = torch.tensor(seqs, dtype=torch.int64, device=DEVICE)
                logits = self.hf(input_ids=ids, logits_to_keep=K, use_cache=False).logits.float()
                tgt = torch.tensor(toks[r0:r0 + chunk], dtype=torch.int64, device=DEVICE)
                margin = logits.max(-1).values - logits.gather(-1, tgt[..., None])[..., 0]
                if not torch.isfinite(margin).all():
                    return float("inf"), -1, -1
                worst = max(worst, margin.max().item())
                bad = (margin > 0).any(0).nonzero()
                nonarg += int((margin > 0).sum().item())
                if len(bad):
                    step = int(bad[0].item())
                    first = step if first < 0 else min(first, step)
        return worst, nonarg, first

    def _select_tier(self, input_ids, N):
        """Warmup-only: validate tiers best-first on this prompt; keep the faster of the first two
        that pass (a new kernel tier can never make the engine wrong, nor slower than the next one)."""
        t_start = time.perf_counter()
        B, S = len(input_ids), len(input_ids[0])
        K = min(N, CHECK_STEPS)
        rows = input_ids[:CHECK_ROWS]
        native_toks, native_logits0 = [], None
        for step, logits in self._native(rows, K):
            native_toks.append(step)
            if native_logits0 is None:
                native_logits0 = logits.float()
        native_toks = [list(r) for r in zip(*native_toks)]
        if FORCE_TIER >= 0:
            order = [FORCE_TIER] if FORCE_TIER > 0 else []
        elif _KERNEL_ERR is None and CUDA:
            small = B <= GEMV_MAX_B and _FUSED_ERR is None
            order = ([6] if small and _FAST_ERR is None and _MEGA_ERR is None else []) \
                + ([5] if small and _FAST_ERR is None else []) \
                + ([4] if small and _GEMV_ERR is None else []) \
                + ([3] if _FUSED_ERR is None else []) + [2, 1]
        else:
            order = [1]
        passing = []                       # (wall ms for K tokens, tier, per-step GPU timings)
        for n, tier in enumerate(order):
            if passing and time.perf_counter() - self.t_load0 > WARMUP_HARD_S:
                log(f"warmup deadline: stop comparing tiers after {len(passing)} passing")
                break
            try:
                st = self._build(B, S, N, tier)
                steps = list(self._stream(st, input_ids, K))
                ours = [list(r) for r in zip(*steps)]
                pl = st.prefill_logits.float()
                finite = bool(torch.isfinite(pl).all().item())
                dlog = (pl[:len(rows)] - native_logits0).abs().max().item()
                margin, nonarg, first_nonarg = self._teacher_force(input_ids, ours)
                div = next((t for t in range(K) if any(ours[r][t] != native_toks[r][t] for r in range(len(rows)))), -1)
                log(f"check T{tier}: margin={margin:.3f} non_argmax={nonarg}/{B * K} first_non_argmax={first_nonarg} "
                    f"first_diverge_vs_native={div} prefill_logit_maxdiff={dlog:.3f} finite={finite} "
                    f"compile={st.compile_s:.1f}s capture={st.capture_s:.1f}s")
                if st.mega and CUDA and int(st.megastep.err.item()) != 0:
                    finite = False                             # a bounded spin expired: not trustworthy
                    log(f"T6 spin limit hit ({st.megastep.describe()})")
                if finite and margin <= MARGIN_LIMIT:
                    wall, timing = self._time_stream(st, input_ids, K)
                    if st.mega and CUDA and int(st.megastep.err.item()) != 0:
                        log("T6 spin limit hit while timing; dropped")
                        continue
                    passing.append((wall, tier, timing))
                    budget_ok = time.perf_counter() - self.t_load0 < WARMUP_SOFT_S - 40
                    if len(passing) >= COMPARE_TIERS or (len(passing) >= 2 and not budget_ok) \
                            or K == 1 or tier == 1:
                        break
                    continue
                reason = "non-finite logits" if not finite else f"margin {margin:.3f} > {MARGIN_LIMIT}"
            except Exception as exc:
                reason = f"exception {repr(exc)[:200]}"
            nxt = order[n + 1] if n + 1 < len(order) else 0
            log(f"FALLBACK T{tier}->T{nxt}: {reason}")
            self.state = None
        check_s = time.perf_counter() - t_start

        if passing:
            wall, chosen, timing = min(passing)
            if len(passing) > 1:
                log("speed: " + " ".join(f"T{t}={w / K:.3f}ms/tok" for w, t, _ in passing) + f" -> T{chosen}")
            if self.state is None or self.state.tier != chosen:
                self._build(B, S, N, chosen)
            if (self.state.spec_T and self.state.verify_graph is not None
                    and time.perf_counter() - self.t_load0 < WARMUP_SOFT_S):
                self._decide_spec(self.state, input_ids, N)
            if (self.state.tree_T and self.state.tree_graph is not None
                    and time.perf_counter() - self.t_load0 < WARMUP_SOFT_S):
                self._decide_tree(self.state, input_ids, N)
            # free the native model's unpacked copies; our tiers never touch them again
            self.hf = None
            gc.collect()
            if CUDA:
                torch.cuda.empty_cache()
            if timing:
                dec = sorted(timing[1:]) or [0.0]
                host = (wall - sum(timing)) / max(1, K)
                log(f"timing T{chosen} {B}x{S}: prefill={timing[0]:.2f}ms step mean={sum(dec) / len(dec):.3f} "
                    f"p50={dec[len(dec) // 2]:.3f} max={dec[-1]:.3f}ms host_overhead/step={host:.3f}ms "
                    f"stream_wall={wall:.1f}ms for {K} tok")
        else:
            chosen = 0
        self.tier = chosen
        peak = torch.cuda.max_memory_reserved() / 1e9 if CUDA else 0.0
        spec = self.state is not None and (self.state.spec or self.state.tree_on)
        log(f"tier=T{chosen}{'+spec' if spec else ''} shape={B}x{S}x{N} check={check_s:.1f}s "
            f"warmup={time.perf_counter() - t_start:.1f}s load={self.load_s:.1f}s peak={peak:.1f}GB")

    def _time_stream(self, st, input_ids, K):
        """Best-of-two wall time (ms) of a K-token stream, with per-step GPU timings."""
        best = None
        for _ in range(2):
            timing = []
            t0 = time.perf_counter()
            for _ in self._stream(st, input_ids, K, timing):
                pass
            wall = (time.perf_counter() - t0) * 1e3
            if best is None or wall < best[0]:
                best = (wall, timing)
        return best

    # ------------------------------------------------------------------ API
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        N = int(max_new_tokens)
        if N <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        gc_was = gc.isenabled()
        gc.disable()
        try:
            st = self.state
            if self.tier is None or (self.tier > 0 and (st is None or (st.B, st.S, st.N) != (B, S, N))):
                if self.hf is not None:
                    self._select_tier(input_ids, N)       # first call of a shape: the untimed warmup
                    st = self.state
                else:
                    st = self._build(B, S, N, self.tier)
            if self.tier == 0:
                for step, _ in self._native(input_ids, N):
                    yield step
                return
            if st.tree_on:
                yield from self._stream_tree(st, input_ids, N)
            elif st.spec:
                yield from self._stream_spec(st, input_ids, N)
            else:
                yield from self._stream(st, input_ids, N)
        finally:
            if gc_was:
                gc.enable()
