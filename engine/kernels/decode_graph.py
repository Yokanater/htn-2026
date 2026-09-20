"""A single-token native Qwen step captured with PyTorch 2.5.1 CUDA graphs.

KV storage, token IDs and absolute position have stable CUDA addresses. Only
their contents change between replays. Host conversion is the caller's job.
"""

import torch
from kernels.qkv_epilogue import qkv_epilogue
from kernels.select_linears import select_linears
from kernels.select_attention import select_attention
from kernels.pointwise import residual_norm, swiglu


class GraphCache:
    def __init__(self, storage, position):
        self.keys = storage.keys
        self.values = storage.values
        self.flat_keys = [key.flatten(0, 1) for key in storage.key_tokens]
        self.flat_values = [value.flatten(0, 1) for value in storage.value_tokens]
        self.position = position

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key = self.keys[layer_idx]
        value = self.values[layer_idx]
        key.index_copy_(2, self.position, key_states)
        value.index_copy_(2, self.position, value_states)
        # Valid lengths are supplied separately to the native FlashAttention op.
        return key, value


class DecodeGraph:
    @torch.inference_mode()
    def __init__(self, model, storage, first_token, first_position, decode_steps):
        self.model = model
        self.tokens = torch.empty_like(first_token)
        self.position = torch.empty(1, dtype=torch.int64, device=first_token.device)
        batch = first_token.shape[0]
        self.capacity = storage.capacity
        self.cu_query = torch.arange(batch + 1, dtype=torch.int32, device=first_token.device)
        self.cu_key = self.cu_query * storage.capacity
        self.valid_lengths = torch.empty(batch, dtype=torch.int32, device=first_token.device)
        self.cache = GraphCache(storage, self.position)
        self.linears = select_linears(model, batch)
        self.attention = select_attention(
            self.cache.flat_keys, self.cache.flat_values, self.cu_query,
            self.cu_key, self.capacity, first_position,
            model.model.layers[0].self_attn.scaling,
        )
        self.chunk_size = min(8, decode_steps)
        self.output = torch.empty((self.chunk_size, batch), dtype=torch.int64,
                                  device=first_token.device)
        self.graphs = {}
        stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            # Initialize cuBLAS/SDPA and allocator paths before capture. Each
            # warmup overwrites the same next slot; prompt slots stay intact.
            for _ in range(3):
                self.reset(first_token, first_position)
                self.step()
        current_stream.wait_stream(stream)
        counts = {self.chunk_size}
        if decode_steps % self.chunk_size:
            counts.add(decode_steps % self.chunk_size)
        for count in sorted(counts):
            self.reset(first_token, first_position)
            stream.wait_stream(current_stream)
            graph = torch.cuda.CUDAGraph()
            # Independent pools avoid imposing a replay order on the tail and
            # full-group graphs. Shared inputs/cache/output have stable addresses.
            with torch.cuda.graph(graph, stream=stream):
                for slot in range(count):
                    self.step()
                    self.output[slot].copy_(self.tokens[:, 0])
            current_stream.wait_stream(stream)
            self.graphs[count] = graph
        self.reset(first_token, first_position)

    def reset(self, first_token, first_position):
        self.tokens.copy_(first_token)
        self.position.fill_(first_position)
        self.valid_lengths.fill_(first_position + 1)

    def step(self):
        base = self.model.model
        x = base.embed_tokens(self.tokens)
        position_ids = self.position.unsqueeze(0)
        position_embeddings = base.rotary_emb(x, position_ids)
        batch = self.tokens.shape[0]
        n = base.layers[0].input_layernorm(x)
        for layer_idx, layer in enumerate(base.layers):
            attn = layer.self_attn
            q = qkv_epilogue(
                self.linears["qkv"](n, attn.qkv_weight), attn, position_embeddings,
                self.position, self.cache.flat_keys[layer_idx],
                self.cache.flat_values[layer_idx], self.capacity,
            )
            # Both native and Triton candidates read the entire device-side
            # valid prefix. Selection and compilation happen only in warmup.
            a = self.attention(
                q, self.cache.flat_keys[layer_idx], self.cache.flat_values[layer_idx],
                self.cu_query, self.cu_key, self.capacity, self.valid_lengths,
                attn.scaling,
            )
            x, n = residual_norm(
                x, self.linears["o"](a.reshape(batch, 1, -1), attn.o_proj.weight),
                layer.post_attention_layernorm,
            )
            mlp = layer.mlp
            activated = swiglu(self.linears["gate_up"](n, mlp.gate_up_weight))
            next_norm = (base.layers[layer_idx + 1].input_layernorm
                         if layer_idx + 1 < len(base.layers) else base.norm)
            x, n = residual_norm(
                x, self.linears["down"](activated, mlp.down_proj.weight), next_norm,
            )
        logits = self.linears["head"](n, self.model.lm_head.weight)
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.add_(1)
        self.valid_lengths.add_(1)
