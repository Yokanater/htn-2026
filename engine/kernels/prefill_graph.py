"""Fixed-shape prefill graph with persistent input/token buffers."""

import torch
from kernels.prefill import prefill


class PrefillGraph:
    @torch.inference_mode()
    def __init__(self, model, cache, input_ids):
        self.input = torch.empty_like(input_ids)
        self.input.copy_(input_ids)
        self.cache = cache
        self.length = input_ids.shape[1]
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            cache.reset()
            logits = prefill(model, self.input, cache, check_rotary=True)
            tokens = logits[:, -1].argmax(-1, keepdim=True)
        current.wait_stream(stream)
        del logits, tokens
        cache.reset()
        with torch.cuda.graph(self.graph, stream=stream):
            logits = prefill(model, self.input, cache)
            self.tokens = logits[:, -1].argmax(-1, keepdim=True)
        current.wait_stream(stream)

    def run(self, input_ids):
        self.input.copy_(input_ids)
        self.graph.replay()
        self.cache.length = self.length
        return self.tokens
