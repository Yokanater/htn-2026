"""Reusable KV storage and a single-token CUDA graph.

Prefill uses native causal SDPA on the prompt tensors. Decode uses fixed-size
cache tensors and a device-side valid-position mask, so replay never attends
to stale slots or requires a CPU read of the current position.
"""

import torch
from model_forward import forward_at_positions


class KVCache:
    """The update interface used by Qwen3Attention; no generic cache dispatch.

    K/V buffers are contiguous [batch, kv_heads, capacity, head_dim]. Prefill
    starts at position zero and overwrites the prompt prefix in every layer.
    Decode writes one absolute position and returns the full backing buffers;
    its caller must supply a mask excluding positions beyond that write.
    """

    def __init__(self, model, batch, capacity):
        config = model.config
        shape = (batch, config.num_key_value_heads, capacity, config.head_dim)
        self.key_cache = [
            torch.zeros(shape, dtype=model.dtype, device=model.device)
            for _ in model.model.layers
        ]
        self.value_cache = [torch.zeros_like(key) for key in self.key_cache]
        self.prefilling = True

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        key = self.key_cache[layer_idx]
        value = self.value_cache[layer_idx]
        if self.prefilling:
            length = key_states.shape[2]
            key[:, :, :length, :].copy_(key_states)
            value[:, :, :length, :].copy_(value_states)
            # Native prompt tensors avoid exposing unused capacity and retain
            # SDPA's fast causal prefill path without a dense explicit mask.
            return key_states, value_states
        position = cache_kwargs["cache_position"]
        key.index_copy_(2, position, key_states)
        value.index_copy_(2, position, value_states)
        return key, value


class DecodeState:
    def __init__(self, model, batch, prompt_length, output_length, *, capture=True):
        self.model = model
        self.shape = (batch, prompt_length, output_length)
        self.capacity = prompt_length + output_length - 1
        self.cache = KVCache(model, batch, self.capacity)
        self.tokens = torch.zeros((batch, 1), dtype=torch.int64, device=model.device)
        self.position = torch.tensor([prompt_length], device=model.device)
        self.key_positions = torch.arange(self.capacity, device=model.device)
        self.prompt_positions = torch.arange(prompt_length, device=model.device)
        self.graph = None
        if capture:
            self._capture()

    @torch.inference_mode()
    def _decode_step(self):
        # Boolean SDPA mask: True permits attention. Recomputed on the GPU
        # inside the graph, including the slot this step is about to write.
        mask = (self.key_positions <= self.position[0]).view(1, 1, 1, -1)
        logits = forward_at_positions(
            self.model, self.tokens, self.cache, self.position, mask
        )
        self.tokens.copy_(logits[:, -1].argmax(-1, keepdim=True))
        self.position.add_(1)

    @torch.inference_mode()
    def _capture(self):
        self.cache.prefilling = False
        # Initialize CUDA libraries and all decode specializations before
        # capture. Reset the write position even for a two-token output budget.
        stream = torch.cuda.Stream(device=self.model.device)
        stream.wait_stream(torch.cuda.current_stream(self.model.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.position.fill_(self.shape[1])
                self._decode_step()
        torch.cuda.current_stream(self.model.device).wait_stream(stream)
        self.position.fill_(self.shape[1])
        torch.cuda.synchronize(self.model.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._decode_step()

    @torch.inference_mode()
    def prefill(self, input_ids):
        self.cache.prefilling = True
        logits = forward_at_positions(
            self.model, input_ids, self.cache, self.prompt_positions
        )
        self.tokens.copy_(logits[:, -1].argmax(-1, keepdim=True))
        self.position.fill_(self.shape[1])
        self.cache.prefilling = False
        return self.tokens

    @torch.inference_mode()
    def step(self):
        if self.graph is None:
            self._decode_step()
        else:
            self.graph.replay()
        return self.tokens
