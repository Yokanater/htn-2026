"""Warmup-only comparison of exact full-prefix decode attention kernels."""

from functools import partial
import statistics

import torch
from kernels.attention import attention, attention_gqa


def native_attention(q, key, value, cu_query, cu_key, capacity, lengths, scale):
    return torch.ops.aten._flash_attention_forward(
        q, key, value, cu_query, cu_key, 1, capacity, 0.0, False, False,
        scale=scale, seqused_k=lengths,
    )[0]


def measure(operation, query, keys, values, cu_query, cu_key, capacity, lengths, scale):
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for key, value in zip(keys, values):
            operation(query, key, value, cu_query, cu_key, capacity, lengths, scale)
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [operation(query, key, value, cu_query, cu_key, capacity, lengths, scale)
                   for key, value in zip(keys, values)]
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
    del graph, outputs
    return statistics.median(times)


@torch.inference_mode()
def select_attention(keys, values, cu_query, cu_key, capacity, prompt_length, scale):
    batch = cu_query.numel() - 1
    device = keys[0].device
    query = torch.randn((batch, 32, 128), dtype=keys[0].dtype, device=device)
    lengths = torch.full((batch,), prompt_length, dtype=torch.int32, device=device)
    args = (query, keys, values, cu_query, cu_key, capacity, lengths, scale)
    best = native_attention
    native_ms = measure(best, *args)
    best_ms = native_ms
    candidates = [partial(attention, split=4), partial(attention_gqa, split=8)]
    for operation in candidates:
        valid = True
        for length in sorted({1, max(1, prompt_length // 2), prompt_length}):
            lengths.fill_(length)
            for index in (0, len(keys) - 1):
                test_args = (query, keys[index], values[index], cu_query, cu_key,
                             capacity, lengths, scale)
                reference = native_attention(*test_args)
                result = operation(*test_args)
                if not torch.allclose(result, reference, rtol=0.016, atol=0.002):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            lengths.fill_(prompt_length)
            elapsed = measure(operation, *args)
            if elapsed < best_ms:
                best, best_ms = operation, elapsed
    return best if best_ms < native_ms * 0.9 else native_attention
