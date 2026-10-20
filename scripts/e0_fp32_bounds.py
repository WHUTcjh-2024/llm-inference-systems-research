"""Conditional FP32 score bounds for the pinned Stage 1 router.

The arithmetic is exact on CPU. Its connection to a GPU kernel still requires
checking that the compiled dot obeys the operation model in e0_contract.md.
This is a correctness prototype, not an efficient GPU summary builder.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from e0_oracle import Node

UNIT_ROUNDOFF = Fraction(1, 2**24)
MIN_NORMAL = Fraction(1, 2**126)
MAX_FINITE = Fraction(float(np.finfo(np.float32).max))


def _fp32_values(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float32 values")
    nonzero = array[array != 0]
    if np.any(np.abs(nonzero.astype(np.float64)) < float(MIN_NORMAL)):
        raise ValueError(f"{name} contains subnormal inputs")
    return array


def _fraction(value: np.float32) -> Fraction:
    return Fraction(float(value))


def fp32_envelope_nodes(
    query: np.ndarray,
    keys: np.ndarray,
    group_size: int,
    *,
    scale: np.float32,
) -> tuple[Node, ...]:
    """Upper-bound x=fl32(fl32(dot(q,k)*scale)*0.5) under the stated model.

    Every dot is assumed to use at most D rounded products and D rounded
    additions, with |fl(y)-y| <= u*|y|+min_normal per operation. The final
    scaling and halving obey the same bound. Backend conformance is unverified.
    """
    q = _fp32_values(query, "query")
    k = _fp32_values(keys, "keys")
    if q.ndim != 1 or k.ndim != 2 or k.shape[1] != q.size or not q.size:
        raise ValueError("Expected query [D] and keys [N,D] with D > 0")
    if not k.shape[0] or group_size < 1:
        raise ValueError("Expected nonempty keys and positive group size")
    if (
        not isinstance(scale, np.float32)
        or not math.isfinite(float(scale))
        or scale <= 0
    ):
        raise ValueError("scale must be a positive finite float32 scalar")
    if Fraction(float(scale)) < MIN_NORMAL:
        raise ValueError("scale must be a normal float32 scalar")

    q_exact = [_fraction(value) for value in q]
    scale_exact = _fraction(scale)
    operations = 2 * q.size
    growth = (1 + UNIT_ROUNDOFF) ** operations
    nodes = []
    for start in range(0, k.shape[0], group_size):
        indices = tuple(range(start, min(start + group_size, k.shape[0])))
        group = k[list(indices)]
        dot_upper = Fraction(0)
        sum_abs_upper = Fraction(0)
        for dim, q_value in enumerate(q_exact):
            column = [_fraction(value) for value in group[:, dim]]
            dot_upper += q_value * (max(column) if q_value >= 0 else min(column))
            sum_abs_upper += abs(q_value) * max(map(abs, column))

        # A full path through D products and additions has at most 2D rounded
        # operations. MIN_NORMAL also covers a flushed subnormal result.
        dot_error = (growth - 1) * sum_abs_upper
        dot_error += operations * MIN_NORMAL * growth
        dot_abs_limit = growth * (sum_abs_upper + operations * MIN_NORMAL)
        if dot_abs_limit >= MAX_FINITE:
            raise ValueError("FP32 dot overflow cannot be ruled out")
        z_abs_limit = (1 + UNIT_ROUNDOFF) * scale_exact * dot_abs_limit
        z_abs_limit += MIN_NORMAL
        if z_abs_limit >= MAX_FINITE:
            raise ValueError("FP32 overflow cannot be ruled out")

        scale_error = scale_exact * dot_error
        scale_error += UNIT_ROUNDOFF * scale_exact * dot_abs_limit + MIN_NORMAL
        upper = (scale_exact * dot_upper + scale_error) / 2 + MIN_NORMAL
        if not -MAX_FINITE < upper < MAX_FINITE:
            raise ValueError("No finite FP32 upper bound")

        # Float conversion may round down. One additional FP32 ULP makes the
        # stored bound strictly larger than the exact rational upper bound.
        rounded = np.float32(float(upper))
        outward = np.nextafter(rounded, np.float32(np.inf))
        if not np.isfinite(outward):
            raise ValueError("No finite FP32 upper bound")
        nodes.append(Node(indices, float(outward)))
    return tuple(nodes)
