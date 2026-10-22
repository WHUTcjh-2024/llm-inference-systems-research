"""Synthetic workloads for Experiment B.

Prompt lengths are controlled exactly via prompt_token_ids, so tokenizer
differences do not change the intended shape.
"""

from __future__ import annotations

import random


def build_prompts(
    kind: str,
    n: int,
    prompt_len: int,
    seed: int = 0,
) -> list[dict]:
    """Return a list of TokensPrompt dicts.

    kind:
      uniform     - all prompts the same length
      mixed       - alternating short (64) and long (>=1024) prompts
      preemption  - all prompts long (>=2048), intended for a tight KV budget
    """
    rng = random.Random(seed)

    if kind == "uniform":
        lengths = [prompt_len] * n
    elif kind == "mixed":
        long_len = max(prompt_len, 1024)
        lengths = [64 if i % 2 == 0 else long_len for i in range(n)]
    elif kind == "preemption":
        lengths = [max(prompt_len, 2048)] * n
    else:
        raise ValueError(f"unknown workload kind: {kind}")

    # 1000..30000 is inside the vocabulary of common Qwen/Llama checkpoints.
    return [
        {"prompt_token_ids": [rng.randint(1000, 30000) for _ in range(L)]}
        for L in lengths
    ]
