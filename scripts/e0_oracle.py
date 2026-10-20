"""CPU-only E0 oracle for the pinned DashAttention three-step router.

This checks the real-number control flow of stage2_block_selection.py at
5e1c55124cd1645b5234a1af12866b7b5f8bfa2e. It does not certify Triton
rounding, GPU speed, or the cost of constructing node bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-5
SENTINEL_X = -5e8  # Stage 1 max starts at -1e9; Stage 2 multiplies by 0.5.
EPS_PRIOR = 1e-20


@dataclass(frozen=True)
class Node:
    indices: tuple[int, ...]
    upper_x: float


@dataclass(frozen=True)
class Step:
    t_before: float
    ff: float
    df: float
    ddf: float
    t_lo: float
    t_hi: float
    t_after: float


@dataclass(frozen=True)
class HeadResult:
    steps: tuple[Step, ...]
    probabilities: np.ndarray
    exact_scores: int
    max_scores: int
    reactivated_nodes: int


def _scores_x(scores_z: np.ndarray) -> np.ndarray:
    values = np.asarray(scores_z, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty 1D vector of finite chunk scores")
    if np.max(values) <= -1e9:
        raise ValueError(
            "All-sentinel or out-of-domain scores require baseline fallback"
        )
    return values * 0.5


def _step(x: np.ndarray, t: float, lo: float, hi: float) -> Step:
    active = x > t
    delta = np.where(active, x - t, 0.0)
    ff = float(np.sum(delta * delta) - 1.0)
    df = float(-2.0 * np.sum(delta))
    ddf = float(2.0 * np.sum(active))
    denom = 2.0 * df * df - ff * ddf
    if not math.isfinite(denom) or denom == 0.0:
        raise ValueError("Degenerate Halley denominator requires baseline fallback")
    proposed = t - (2.0 * ff * df) / denom
    if not math.isfinite(proposed):
        raise ValueError("Nonfinite Halley update requires baseline fallback")
    next_lo = t if ff > 0.0 else lo
    next_hi = t if ff < 0.0 else hi
    good = next_lo - EPS < proposed < next_hi + EPS
    next_t = proposed if good else 0.5 * (next_lo + next_hi)
    return Step(t, ff, df, ddf, next_lo, next_hi, next_t)


def _initial_state(max_x: float) -> tuple[float, float, float]:
    hi = max_x
    lo = hi - 1.0
    return 0.5 * (lo + hi), lo, hi


def solve_full(scores_z: np.ndarray) -> HeadResult:
    """Direct all-score translation of the pinned kernel's Stage 2 loop."""
    x = _scores_x(scores_z)
    t, lo, hi = _initial_state(max(SENTINEL_X, float(np.max(x))))
    steps = []
    for _ in range(3):
        step = _step(x, t, lo, hi)
        steps.append(step)
        t, lo, hi = step.t_after, step.t_lo, step.t_hi
    probabilities = np.square(np.maximum(x - t, 0.0))
    return HeadResult(tuple(steps), probabilities, x.size, x.size, 0)


def _validate_nodes(x: np.ndarray, nodes: tuple[Node, ...]) -> None:
    if not nodes:
        raise ValueError("At least one bound node is required")
    covered = []
    for node in nodes:
        if not node.indices or not math.isfinite(node.upper_x):
            raise ValueError("Nodes need finite upper bounds and nonempty leaves")
        covered.extend(node.indices)
        if any(index < 0 or index >= x.size for index in node.indices):
            raise ValueError("Node index is outside the score vector")
        if node.upper_x < float(np.max(x[list(node.indices)])):
            raise ValueError("Unsafe node upper bound")
    if sorted(covered) != list(range(x.size)):
        raise ValueError("Nodes must partition the chunks exactly once")


def solve_certified(scores_z: np.ndarray, nodes: tuple[Node, ...]) -> HeadResult:
    """Defer exact scores only while a node is provably zero at the current t."""
    x = _scores_x(scores_z)
    _validate_nodes(x, nodes)  # E0 has all scores; a GPU implementation cannot do this.
    cached = np.full(x.shape, np.nan)
    scored = set()
    max_x = SENTINEL_X
    max_scores = 0

    # Best-first maximum certificate: stop when no unseen node can beat best.
    for index in sorted(
        range(len(nodes)), key=lambda i: nodes[i].upper_x, reverse=True
    ):
        node = nodes[index]
        if node.upper_x <= max_x:
            break
        cached[list(node.indices)] = x[list(node.indices)]
        scored.add(index)
        max_scores += len(node.indices)
        max_x = max(max_x, float(np.max(cached[list(node.indices)])))
    if max_x <= SENTINEL_X:
        raise ValueError("Cannot certify a nonsentinel maximum")

    t, lo, hi = _initial_state(max_x)
    steps = []
    previously_deferred = set()
    reactivated = 0
    for _ in range(3):
        deferred = set()
        for index, node in enumerate(nodes):
            if index in scored:
                continue
            if node.upper_x <= t:
                deferred.add(index)
                continue
            if index in previously_deferred:
                reactivated += 1
            cached[list(node.indices)] = x[list(node.indices)]
            scored.add(index)
        # Missing scores have zero contribution because their U <= current t.
        effective = np.where(np.isnan(cached), -np.inf, cached)
        step = _step(effective, t, lo, hi)
        steps.append(step)
        t, lo, hi = step.t_after, step.t_lo, step.t_hi
        previously_deferred = deferred

    # Final prior and mask use t_3, not the threshold before the third update.
    for index, node in enumerate(nodes):
        if index not in scored and node.upper_x > t:
            if index in previously_deferred:
                reactivated += 1
            cached[list(node.indices)] = x[list(node.indices)]
            scored.add(index)
    effective = np.where(np.isnan(cached), -np.inf, cached)
    probabilities = np.square(np.maximum(effective - t, 0.0))
    return HeadResult(
        tuple(steps),
        probabilities,
        int(np.count_nonzero(~np.isnan(cached))),
        max_scores,
        reactivated,
    )


def pack_group(
    heads: tuple[HeadResult, ...], *, estimate_diag: bool, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce GQA mean, optional diagonal prior transform, and signed bitpack."""
    if not heads or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Expected nonempty GQA group and positive finite sigma")
    n = heads[0].probabilities.size
    if any(head.probabilities.size != n for head in heads):
        raise ValueError("GQA heads must have equal chunk counts")
    mean_prob = np.mean(np.stack([head.probabilities for head in heads]), axis=0)
    prior = np.zeros(n + 1, dtype=np.float64)
    if estimate_diag:
        positive = mean_prob > EPS_PRIOR
        if np.any(positive):
            log_g = float(np.mean(np.log(mean_prob[positive])))
            prior[:-1][positive] = np.exp((np.log(mean_prob[positive]) - log_g) / sigma)
    else:
        prior[:-1] = mean_prob
    prior[-1] = 1.0
    words = np.zeros((n + 1 + 31) // 32, dtype=np.uint32)
    for index, active in enumerate([*(mean_prob > 0.0), True]):
        if active:
            words[index // 32] |= np.uint32(1) << np.uint32(index % 32)
    return prior, words.view(np.int32)


def l2_nodes(query: np.ndarray, keys: np.ndarray, group_size: int) -> tuple[Node, ...]:
    """Construct illustrative real-valued ball bounds; not GPU FP32 bounds."""
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    if q.ndim != 1 or k.ndim != 2 or k.shape[1] != q.size or group_size < 1:
        raise ValueError("Expected query [D], keys [N,D], and positive group size")
    nodes = []
    for start in range(0, len(k), group_size):
        indices = tuple(range(start, min(start + group_size, len(k))))
        group = k[list(indices)]
        center = np.mean(group, axis=0)
        radius = float(np.max(np.linalg.norm(group - center, axis=1)))
        center_score = float(np.dot(q, center))
        bound = center_score + float(np.linalg.norm(q)) * radius
        # E0 uses double precision and validates against exact CPU scores.
        # This slack is not a proof of coverage for the pinned tl.dot path.
        slack = 1e-12 * (1.0 + abs(center_score) + abs(bound))
        nodes.append(Node(indices, bound + slack))
    return tuple(nodes)


def run_trials(seed: int, trials: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    totals = {
        "trials": trials,
        "exact_scores": 0,
        "full_scores": 0,
        "reactivated_nodes": 0,
    }
    for _ in range(trials):
        queries = rng.normal(size=(2, 16))
        centers = rng.normal(size=(8, 16))
        keys = np.repeat(centers, 8, axis=0) + 0.05 * rng.normal(size=(64, 16))
        full_heads = []
        certified_heads = []
        for q in queries:
            x = keys @ q
            nodes = l2_nodes(q, keys, 8)
            full = solve_full(x * 2.0)
            certified = solve_certified(x * 2.0, nodes)
            for expected, actual in zip(full.steps, certified.steps):
                np.testing.assert_allclose(
                    tuple(vars(expected).values()),
                    tuple(vars(actual).values()),
                    rtol=0,
                    atol=1e-12,
                )
            np.testing.assert_allclose(
                full.probabilities, certified.probabilities, rtol=0, atol=1e-12
            )
            full_heads.append(full)
            certified_heads.append(certified)
            totals["exact_scores"] += certified.exact_scores
            totals["full_scores"] += full.exact_scores
            totals["reactivated_nodes"] += certified.reactivated_nodes
        for estimate_diag in (False, True):
            full_prior, full_mask = pack_group(
                tuple(full_heads), estimate_diag=estimate_diag, sigma=1e8
            )
            certified_prior, certified_mask = pack_group(
                tuple(certified_heads), estimate_diag=estimate_diag, sigma=1e8
            )
            np.testing.assert_allclose(full_prior, certified_prior, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(full_mask, certified_mask)
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    print(json.dumps(run_trials(args.seed, args.trials), indent=2))
