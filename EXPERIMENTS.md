# Optimization log

## 001 — Fused RMSNorm and direct decoder dispatch

Starting point: Dryft-Kernels/starter at
`c2405f19fae577539969c5face3914b116757ae6`.

Candidate changes:

- Replace hidden-state and Q/K head RMSNorm modules with the starter's Triton
  kernel, preserving learned gains, epsilon, and the BF16 cast boundary.
- Call decoder layers directly, retaining native RoPE, attention, MLPs, and
  DynamicCache. Create a fresh cache for every generation.
- Normalize only the final hidden position before the LM head.
- Continue to stream exactly the requested number of tokens, including EOS.

Validation on macOS with Python 3.11, PyTorch 2.5.1 and Transformers 4.51.3:

- Native versus direct-dispatch logits match on a small randomly initialized
  Qwen3 for prefill and four cached decode steps, at multiple batch/prompt sizes.
- Repeated generations match native, retain batch order, reset cache state,
  and continue after EOS. Zero requested output tokens yields nothing.
- Four CPU tests pass, including two existing API client tests.
- Two CUDA tests cover fused norms and teacher-forced replay on the candidate's
  own prefix. These are skipped locally because this machine has no CUDA GPU.
- Dryft archive validation passes.

No H100 correctness result or speedup is measured yet. `DRYFT_TOKEN` was absent
and `dryft doctor` could not authenticate. CPU tests exercise native norms;
they do not validate Triton numerics or full-checkpoint accuracy.

## Run the next experiment

Configure `DRYFT_TOKEN` locally using a token from <https://htn.dryft.ai/tokens>.
Do not add it to source control or the engine directory.

```sh
./bin/dryft doctor
./bin/dryft submit artifacts/baseline.tar.gz
./bin/dryft run <baseline-submission-id> --mode public --wait 3000
./bin/dryft submit artifacts/candidate-001.tar.gz
./bin/dryft run <candidate-submission-id> --mode public --wait 3000
```

Record correctness, TTFT/native, TPOT/native, throughput, and memory for all
three public shapes. Do not request an official run until correctness passes
and both latency ratios are below 1.10 with headroom.

Run local checks with `.venv/bin/python -m unittest discover -s tests -v`.
On a CUDA machine with the pinned requirements, the same command also executes
the fused-kernel checks. Full-model acceptance still requires Dryft evaluation.

The next optimization to evaluate after this candidate is a preallocated KV
cache followed by CUDA graph replay for single-token decode. It needs explicit
masking of unused slots and in-place updates of graph inputs and positions.

## 002 — Reusable KV cache and CUDA graph decode

Target: exceed 520 tokens/sec on the benchmark. Treat that as an unverified
target until a Dryft report supplies the measured workload or official score.

Candidate 001 was pushed as `93dcbd6` to the connected default branch,
`codex/solve-starter`, for public evaluation. No GitHub check or commit status
was exposed when queried; a Dryft run/result or API token is still needed to
read performance feedback. Do not infer that a push completed a GPU run.

Candidate 002 retains native projections, RoPE, MLP, and SDPA while replacing
dynamic cache concatenation with fixed per-layer K/V storage. Full prefill
returns the native prompt K/V tensors to causal SDPA. Decode reads the fixed
buffers with an explicit device-side mask permitting positions at or before
the current write. The entire decode step, argmax, next-token copy, and position
increment are captured as one CUDA graph. The host only replays and streams
the resulting IDs. A single graph/cache state is retained for the current
workload shape; setup occurs during the platform's untimed warmup.

Six CPU tests pass, with two CUDA tests skipped locally. New tests compare
fixed-cache logits against native through prefill and decode, deliberately
poison unused slots, check buffer reuse, and verify position resets across
different prompts. CUDA graph execution and full-checkpoint numerical
acceptance remain unverified until an H100 run.

## 003 — H100-selected decode kernels

Target: 1200+ official tokens/sec. This candidate replaces 002 with the latest
publicly inspectable H100-tuned implementation from `goshanraj-g/starter` at
`dc526c6d`. It adds graph-captured native causal prefill, grouped-query split
decode attention, packed QKV and gate/up projections, fused norm/rotary/cache
writes, grouped decode replays, and warmup-time selection among native,
Triton, alternate BF16 layouts, and padded small-batch GEMMs. Numerical checks
and conservative speed thresholds retain native fallbacks when a candidate is
not both close and materially faster.

Local static validation passes: Python compilation, Dryft manifest/archive
validation, and the API client tests. The generated archive is 10,979 bytes.
This Mac has no CUDA-enabled PyTorch installation, so only the Dryft H100 run
can establish end-to-end token correctness and throughput.

Candidate 003 passed every official workload at **888.8163 tokens/sec** in run
`4797cd69-6101-4db0-866e-73ac9405aad4`. Public throughput was 229.8 / 457.0 /
2790.9 tokens/sec with 16.55 GB peak memory. Correctness, latency, memory, and
stability all passed; hidden-shape decode throughput remains the limiter.

## 004 — Split-K GEMMs and fused flash decode

Candidate 004 moves to the more aggressive public H100 implementation from
`RajanChavada/starter` at `d733ea92`. It warmup-selects wider split-K GEMM
tiles, fuses QKV and gate/up projections, uses fused epilogues for residuals
and SwiGLU, captures prefill and decode graphs, and selects a split flash-decode
kernel across the full static cache. Local Python compilation and Dryft archive
validation pass; the H100 run remains the correctness and performance judge.

Candidate 004 passed correctness and all gates but regressed to **767.0204
tokens/sec** in run `23e88ed4-b470-450e-9289-a5a3589b93be`. Public throughput
was 199.8 / 397.1 / 2382.7 tokens/sec with 19.68 GB peak memory. Do not retain.

## 005 — Exact token-tree speculation

Candidate 005 uses the tiered exact engine from `Pranoym17/starter` at
`c4fca79a`. It adds prompt/ngram/Jacobi token-tree candidates and verifies them
in a native-equivalent multi-token forward with ancestor masking, compacting
only the accepted path. Speculation is enabled only after warmup validates its
tokens and measures at least an 8% win; otherwise the engine falls back through
its checked CUDA-graph decode tiers. The source reports a full Triton 3.1 sm_90
compile sweep after fixing a Hopper 64-row compiler abort and int32 cache-offset
overflow. Dryft archive validation and local Python compilation pass.

Candidate 005 passed every gate at **909.5583 tokens/sec** in run
`bc3b0cb5-6214-4621-b4f0-9609de780048`, with public throughput 233.6 / 468.9 /
2818.7 tokens/sec. Inspection showed prompt/ngram speculation was compiled but
disabled by default, while tree speculation is restricted to T5/T6. Candidate
006 enables the exact T4-compatible verifier; its existing warmup teacher-force
check and 8% speed threshold still fall back to plain decoding when unsuitable.
