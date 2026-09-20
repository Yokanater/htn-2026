"""Warmup selection: measure the complete weight set per operation."""

from functools import partial
import statistics

import torch
from torch.nn.functional import linear as native_linear

from kernels.linear import linear as triton_linear
from kernels.padded_linear import padded_linear


def graph_time(operation, x, weights):
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for weight in weights:
            operation(x, weight)
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [operation(x, weight) for weight in weights]
    current.wait_stream(stream)
    graph.replay()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    # Keep all graph-owned outputs alive until the last replay completes.
    del graph, outputs
    return statistics.median(times)


def column_linear(x, weight, layouts):
    return native_linear(x, layouts[weight.data_ptr()])


@torch.inference_mode()
def select_linears(model, batch):
    layers = model.model.layers
    categories = {
        "qkv": [layer.self_attn.qkv_weight for layer in layers],
        "o": [layer.self_attn.o_proj.weight for layer in layers],
        "gate_up": [layer.mlp.gate_up_weight for layer in layers],
        "down": [layer.mlp.down_proj.weight for layer in layers],
        "head": [model.lm_head.weight],
    }
    if batch > 32:
        return {name: native_linear for name in categories}
    selected = {}
    base_peak = torch.cuda.max_memory_allocated()
    retained = 0
    limit = int(torch.cuda.get_device_properties(0).total_memory * 0.85)
    for name, weights in categories.items():
        x = torch.randn((batch, 1, weights[0].shape[1]),
                        dtype=weights[0].dtype, device=weights[0].device)
        best = native_linear
        reference = native_linear(x, weights[0])
        best_ms = graph_time(best, x, weights)
        native_ms = best_ms
        column_candidate = None
        extra_bytes = sum(weight.numel() * weight.element_size() for weight in weights)
        if base_peak + retained + extra_bytes < limit:
            layouts = {weight.data_ptr(): weight.t().contiguous().t() for weight in weights}
            column_candidate = partial(column_linear, layouts=layouts)
            actual = column_candidate(x, weights[0])
            if torch.allclose(actual, reference, rtol=0.016, atol=0.002):
                elapsed = graph_time(column_candidate, x, weights)
                if elapsed < best_ms:
                    best, best_ms = column_candidate, elapsed
            del layouts, actual
        candidates = [partial(triton_linear, split=split)
                      for split in ((1,) if name == "head" else (2, 4, 8))]
        if batch < 16:
            candidates.append(padded_linear)
        for candidate in candidates:
            result = candidate(x, weights[0])
            # Detect implementation mistakes before selecting a kernel. End-to-
            # end greedy/teacher-forced validation is still required separately.
            if not torch.allclose(result, reference, rtol=0.016, atol=0.002):
                continue
            elapsed = graph_time(candidate, x, weights)
            if elapsed < best_ms:
                best, best_ms = candidate, elapsed
        selected[name] = best if best_ms < native_ms * 0.95 else native_linear
        if selected[name] is column_candidate:
            retained += extra_bytes
        del column_candidate
    return selected
