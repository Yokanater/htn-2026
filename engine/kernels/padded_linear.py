"""Expose a tensor-core-friendly row count without changing real BF16 inputs."""

from torch.nn.functional import linear, pad


def padded_linear(x, weight, rows=16):
    shape = x.shape
    batch = x.numel() // shape[-1]
    padded = pad(x.reshape(batch, shape[-1]), (0, 0, 0, rows - batch))
    result = linear(padded, weight)[:batch]
    return result.reshape(*shape[:-1], weight.shape[0])
