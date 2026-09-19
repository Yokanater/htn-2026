"""Qwen3 greedy decoding with fused norms and direct decoder-layer dispatch."""

import torch
from transformers import AutoModelForCausalLM, DynamicCache


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


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, first_position):
    """Full unpadded prefill, or one decode token, using a dynamic cache.

    SDPA supplies the causal mask for prefill. Single-token decode can attend
    to the entire initialized cache. This path does not support chunked
    prefill, padded batches, or a fixed-capacity cache.
    """
    base = model.model
    x = base.embed_tokens(input_ids)
    length = input_ids.shape[1]
    cache_position = torch.arange(
        first_position, first_position + length, device=input_ids.device
    )
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = base.rotary_emb(x, position_ids)
    for layer in base.layers:
        x = layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]

    # No later operation uses the other prompt positions. Normalization is
    # independent per token, so only normalize the position sent to the head.
    return model.lm_head(base.norm(x[:, -1:, :]))


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

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        current = torch.tensor(input_ids, dtype=torch.int64, device=self.model.device)
        cache = DynamicCache()
        position = 0
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = qwen_forward(self.model, current, cache, position)
                position += current.shape[1]
                current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                yield current[:, 0].tolist()
