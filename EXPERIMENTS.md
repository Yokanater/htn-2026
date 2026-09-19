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
