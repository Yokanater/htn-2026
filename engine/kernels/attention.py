"""Full-prefix split attention; BF16 inputs, FP32 reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, LENGTH, PART, STATS, capacity,
             SPLIT: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    d = tl.arange(0, 128)
    q = tl.load(Q + (batch * 32 + head) * 128 + d).to(tl.float32)
    length = tl.load(LENGTH + batch)
    chunk = ((capacity + BN * SPLIT - 1) // (BN * SPLIT)) * BN
    begin = split * chunk
    maximum = -float("inf")
    denominator = 0.0
    acc = tl.full((128,), 0, tl.float32)
    if begin < length:
        for start in range(begin, begin + chunk, BN):
            token = start + tl.arange(0, BN)
            offset = (batch * capacity + token[:, None]) * 1024 + (head // 4) * 128 + d[None, :]
            key = tl.load(K + offset, token[:, None] < length, 0).to(tl.float32)
            score = tl.sum(key * q[None, :], 1) * SCALE
            score = tl.where(token < length, score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, 0))
            alpha = tl.exp2(maximum - next_maximum)
            probability = tl.exp2(score - next_maximum)
            value = tl.load(V + offset, token[:, None] < length, 0).to(tl.float32)
            # Flash-style BF16 probability product, retaining FP32 numerator
            # and denominator through block and partition combinations.
            rounded = probability.to(tl.bfloat16).to(tl.float32)
            acc = acc * alpha + tl.sum(rounded[:, None] * value, 0)
            denominator = denominator * alpha + tl.sum(probability, 0)
            maximum = next_maximum
    part = (batch * 32 + head) * SPLIT + split
    tl.store(PART + part * 128 + d, acc)
    tl.store(STATS + part * 2, maximum)
    tl.store(STATS + part * 2 + 1, denominator)


@triton.jit
def _combine(PART, STATS, OUT, SPLIT: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, SPLIT)
    d = tl.arange(0, 128)
    offset = row * SPLIT + split
    maximum = tl.load(STATS + offset * 2)
    denominator = tl.load(STATS + offset * 2 + 1)
    weight = tl.exp2(maximum - tl.max(maximum, 0))
    numerator = tl.load(PART + offset[:, None] * 128 + d[None, :])
    result = tl.sum(numerator * weight[:, None], 0) / tl.sum(denominator * weight, 0)
    tl.store(OUT + row * 128 + d, result)


def attention(q, key, value, cu_query, cu_key, capacity, lengths, scale, split=8):
    batch = q.shape[0]
    partial = torch.empty((batch, 32, split, 128), device=q.device, dtype=torch.float32)
    stats = torch.empty((batch, 32, split, 2), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _partial[(batch, 32, split)](
        q, key, value, lengths, partial, stats, capacity,
        split, scale * 1.4426950408889634, 64, num_warps=4,
    )
    _combine[(batch * 32,)](partial, stats, out, split, num_warps=4)
    return out


@triton.jit
def _partial_gqa(Q, K, V, LENGTH, PART, STATS, capacity,
                 SPLIT: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    row = tl.arange(0, 16)
    d = tl.arange(0, 128)
    head = kv_head * 4 + row
    q = tl.load(Q + (batch * 32 + head[:, None]) * 128 + d[None, :],
                row[:, None] < 4, 0)
    length = tl.load(LENGTH + batch)
    chunk = ((capacity + BN * SPLIT - 1) // (BN * SPLIT)) * BN
    begin = split * chunk
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, 128), 0, tl.float32)
    if begin < length:
        for start in range(begin, begin + chunk, BN):
            token = start + tl.arange(0, BN)
            key = tl.load(K + (batch * capacity + token[None, :]) * 1024
                          + kv_head * 128 + d[:, None], token[None, :] < length, 0)
            score = tl.dot(q, key) * SCALE
            score = tl.where(token[None, :] < length, score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, 1))
            alpha = tl.exp2(maximum - next_maximum)
            probability = tl.exp2(score - next_maximum[:, None])
            value = tl.load(V + (batch * capacity + token[:, None]) * 1024
                            + kv_head * 128 + d[None, :], token[:, None] < length, 0)
            acc = tl.dot(probability.to(tl.bfloat16), value, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(probability, 1)
            maximum = next_maximum
    part = (batch * 32 + head) * SPLIT + split
    tl.store(PART + part[:, None] * 128 + d[None, :], acc, row[:, None] < 4)
    tl.store(STATS + part * 2, maximum, row < 4)
    tl.store(STATS + part * 2 + 1, denominator, row < 4)


def attention_gqa(q, key, value, cu_query, cu_key, capacity, lengths, scale, split=8):
    batch = q.shape[0]
    partial = torch.empty((batch, 32, split, 128), device=q.device, dtype=torch.float32)
    stats = torch.empty((batch, 32, split, 2), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _partial_gqa[(batch, 8, split)](
        q, key, value, lengths, partial, stats, capacity,
        split, scale * 1.4426950408889634, 64, num_warps=4, num_stages=2,
    )
    _combine[(batch * 32,)](partial, stats, out, split, num_warps=4)
    return out
