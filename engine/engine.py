"""Qwen3 greedy decoding with fused norms, reusable KV storage and CUDA graphs."""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache
from decode import DecodeState
from model_forward import qwen_forward


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        from kernels.rmsnorm import rms_norm

        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon
        self._rms_norm = rms_norm

    def forward(self, x):
        return self._rms_norm(x, self.weight, self.variance_epsilon)


def install_fused_norms(model):
    base = model.model
    base.norm = FusedRMSNorm(base.norm)
    for layer in base.layers:
        layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
        layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
        layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        install_fused_norms(self.model)
        self._decode_state = None

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        current = torch.tensor(input_ids, dtype=torch.int64, device=self.model.device)
        with torch.inference_mode():
            if self.model.device.type == "cuda" and max_new_tokens > 1:
                shape = (*current.shape, max_new_tokens)
                state = getattr(self, "_decode_state", None)
                if state is None or state.shape != shape:
                    # Only one shape is live. The platform warms up each
                    # workload in its own process before measured samples.
                    self._decode_state = None
                    state = None
                    started = time.perf_counter()
                    state = DecodeState(self.model, *shape)
                    self._decode_state = state
                    print(
                        f"decode graph ready: batch/prompt/output={shape}, "
                        f"setup={time.perf_counter() - started:.2f}s",
                        file=sys.stderr,
                    )
                yield state.prefill(current)[:, 0].tolist()
                for _ in range(max_new_tokens - 1):
                    yield state.step()[:, 0].tolist()
                return

            cache = DynamicCache()
            position = 0
            for _ in range(max_new_tokens):
                logits = qwen_forward(self.model, current, cache, position)
                position += current.shape[1]
                current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                yield current[:, 0].tolist()
