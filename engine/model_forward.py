"""Direct Qwen3 layer dispatch, with explicit positions and attention masks."""

import torch


@torch.inference_mode()
def forward_at_positions(model, input_ids, cache, cache_position, attention_mask=None):
    base = model.model
    x = base.embed_tokens(input_ids)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = base.rotary_emb(x, position_ids)
    for layer in base.layers:
        x = layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    # Norm is independent per token. Only the final position reaches the head.
    return model.lm_head(base.norm(x[:, -1:, :]))


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, first_position):
    """Unpadded full prefill or single-token decode with a dynamic cache."""
    cache_position = torch.arange(
        first_position, first_position + input_ids.shape[1], device=input_ids.device
    )
    return forward_at_positions(model, input_ids, cache, cache_position)
