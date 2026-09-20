"""Shared KV storage; eager updates expose only the initialized prefix.

Logical views are [B, Hkv, capacity, D]. Physical storage is token-major so
FlashAttention can view each batch's reserved segment without a layout copy.
"""

import torch


class PrefixCache:
    def __init__(self, config, batch, capacity, device, dtype):
        shape = (batch, capacity, config.num_key_value_heads, config.head_dim)
        self.key_tokens = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        # Masked values must be finite: an uninitialized NaN could contaminate
        # attention even with a zero softmax weight. Zero once during allocation.
        self.value_tokens = [torch.zeros_like(key) for key in self.key_tokens]
        self.keys = [key.transpose(1, 2) for key in self.key_tokens]
        self.values = [value.transpose(1, 2) for value in self.value_tokens]
        self.capacity = capacity
        self.length = 0

    def reset(self):
        # Every exposed slot is overwritten before use; unused capacity stays hidden.
        self.length = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded")
        key = self.keys[layer_idx]
        value = self.values[layer_idx]
        key[:, :, self.length:end, :].copy_(key_states)
        value[:, :, self.length:end, :].copy_(value_states)
        return key[:, :, :end, :], value[:, :, :end, :]
