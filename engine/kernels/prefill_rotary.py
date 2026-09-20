"""Prefill rotary and KV store with eager BF16 product boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary(QIN, KIN, VIN, COS, SIN, QOUT, KCACHE, VCACHE,
            TOKENS: tl.constexpr, CAPACITY: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    batch = row // TOKENS
    token = row % TOKENS
    col = tl.arange(0, 128)
    other = (col + 64) % 128
    if head < 32:
        source = QIN + row * 4096 + head * 128
    else:
        source = KIN + row * 1024 + (head - 32) * 128
    x = tl.load(source + col).to(tl.float32)
    rotated = tl.load(source + other).to(tl.float32)
    rotated = tl.where(col < 64, -rotated, rotated)
    cosine = tl.load(COS + token * 128 + col).to(tl.float32)
    sine = tl.load(SIN + token * 128 + col).to(tl.float32)
    left = (x * cosine).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = (left + right).to(tl.bfloat16)
    if head < 32:
        tl.store(QOUT + row * 4096 + head * 128 + col, result)
    else:
        destination = (batch * CAPACITY + token) * 1024 + (head - 32) * 128 + col
        tl.store(KCACHE + destination, result)
        value = tl.load(VIN + row * 1024 + (head - 32) * 128 + col)
        tl.store(VCACHE + destination, value)


def rotary_store(query, key, value, position_embeddings, key_cache, value_cache):
    batch, length = query.shape[:2]
    output = torch.empty_like(query)
    cosine, sine = position_embeddings
    _rotary[(batch * length, 40)](
        query, key, value, cosine, sine, output, key_cache, value_cache,
        length, key_cache.shape[1], num_warps=4, enable_fp_fusion=False,
    )
    return output
