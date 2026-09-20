"""Original prefill projections/norms with exact BF16 rotary/cache fusion."""

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from kernels.prefill_rotary import rotary_store


def prefill(model, input_ids, cache, check_rotary=False):
    if cache.length != 0:
        raise ValueError("Causal prefill requires a reset cache")
    base = model.model
    batch, length = input_ids.shape
    x = base.embed_tokens(input_ids)
    positions = torch.arange(length, device=input_ids.device)
    position_embeddings = base.rotary_emb(x, positions.unsqueeze(0))
    for index, layer in enumerate(base.layers):
        attn = layer.self_attn
        n = layer.input_layernorm(x)
        shape = (batch, length, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(n).view(shape))
        k = attn.k_norm(attn.k_proj(n).view(shape))
        v = attn.v_proj(n).view(shape)
        rotated = rotary_store(q, k, v, position_embeddings,
                               cache.key_tokens[index], cache.value_tokens[index])
        if check_rotary:
            # Called in eager warmup only, never inside CUDA graph capture.
            reference_q, reference_k = apply_rotary_pos_emb(
                q.transpose(1, 2), k.transpose(1, 2), *position_embeddings,
            )
            if not (torch.equal(rotated, reference_q.transpose(1, 2))
                    and torch.equal(cache.key_tokens[index][:, :length],
                                    reference_k.transpose(1, 2))
                    and torch.equal(cache.value_tokens[index][:, :length], v)):
                raise RuntimeError("Prefill rotary/cache fusion differs from native BF16 output")
        a = torch.ops.aten._flash_attention_forward(
            rotated, cache.key_tokens[index][:, :length], cache.value_tokens[index][:, :length],
            None, None, length, length, 0.0, True, False,
            scale=attn.scaling,
        )[0]
        x = x + attn.o_proj(a.reshape(batch, length, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    cache.length = length
    return model.lm_head(base.norm(x[:, -1:, :]))
