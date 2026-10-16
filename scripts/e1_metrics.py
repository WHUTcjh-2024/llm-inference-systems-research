"""Pure E1 measurement helpers, shared by the GPU run and local tests."""

import math
import statistics

import numpy as np


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Expected nonempty values and a percentile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "mean": statistics.mean(values),
    }


def stats_ms(values: list[float]) -> dict:
    result = summary(values)
    return {
        "count": result["count"],
        "p50_ms": result["p50"],
        "p95_ms": result["p95"],
        "mean_ms": result["mean"],
    }


def mean_active_blocks(packed: np.ndarray, possible: int) -> float:
    """Count every signed int32 word as 32 unsigned bits; average over B/Hkv."""
    if packed.dtype != np.int32 or packed.ndim != 4 or packed.shape[2] != 1:
        raise ValueError("Expected [B,Hkv,1,words] int32 bitmask")
    if possible <= 0:
        raise ValueError("Possible block count must be positive")
    words = packed.view(np.uint32).reshape(packed.shape[0], packed.shape[1], -1)
    if words.shape[-1] != (possible + 31) // 32:
        raise ValueError("Bitmask word count differs from possible blocks")
    valid_tail_bits = possible % 32
    if valid_tail_bits:
        invalid_mask = np.uint32(0xFFFFFFFF ^ ((1 << valid_tail_bits) - 1))
        if np.any(words[..., -1] & invalid_mask):
            raise ValueError("Bitmask contains active padding bits")
    counts = (
        np.fromiter(
            (int(value).bit_count() for value in words.flat),
            dtype=np.int32,
            count=words.size,
        )
        .reshape(words.shape)
        .sum(axis=-1)
    )
    if np.any(counts > possible):
        raise ValueError("Bitmask contains active padding bits")
    return float(counts.mean())
