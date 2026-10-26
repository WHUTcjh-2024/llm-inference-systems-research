#!/usr/bin/env python3
"""Run one workload without instrumentation, for control comparisons.

The instrumented path sets VLLM_ENABLE_V1_MULTIPROCESSING=0 so the patches
can see the engine. This script runs the same workload with the engine core
in a separate process (vLLM default) or in-process, and reports throughput
and per-request TPOT, so the instrumentation's effect on the overlap
phenomenon can be bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--workload", default="mixed", choices=["uniform", "mixed", "preemption"])
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--async-scheduling", default="default", choices=["on", "off", "default"])
    p.add_argument("--chunked-prefill", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--attn-backend", default="FLASHINFER")
    p.add_argument("--engine-multiprocessing", default="on", choices=["on", "off"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-full", action="store_true")
    return p.parse_args()


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def main() -> None:
    args = parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    if args.engine_multiprocessing == "off":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    from workloads import build_prompts

    kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        distributed_executor_backend="uni",
    )
    if args.chunked_prefill:
        kwargs["enable_chunked_prefill"] = True
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.async_scheduling != "default":
        kwargs["async_scheduling"] = args.async_scheduling == "on"
    if args.attn_backend:
        kwargs["attention_backend"] = args.attn_backend

    llm = LLM(**kwargs)
    config = llm.llm_engine.vllm_config
    print(
        f"[config] async_scheduling={config.scheduler_config.async_scheduling} "
        f"executor={config.parallel_config.distributed_executor_backend} "
        f"runner_v2={config.use_v2_model_runner} "
        f"chunked_prefill={config.scheduler_config.enable_chunked_prefill} "
        f"attn_backend={config.attention_config.backend} "
        f"prefix_caching={config.cache_config.enable_prefix_caching} "
        f"engine_multiprocessing={args.engine_multiprocessing}",
        flush=True,
    )

    prompts = build_prompts(args.workload, args.num_prompts, args.prompt_len, args.seed)
    params = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    if args.warmup_full:
        llm.generate(prompts, params, use_tqdm=False)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    generated = sum(len(out.outputs[0].token_ids) for out in outputs)
    digest = hashlib.sha256(
        json.dumps(
            [out.outputs[0].token_ids for out in outputs], separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    tpot_ms = []
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        if metrics is None or len(out.outputs[0].token_ids) <= 1:
            continue
        first = getattr(metrics, "first_token_time", None)
        finished = getattr(metrics, "finished_time", None)
        if first is None or finished is None:
            continue
        tpot_ms.append(
            (finished - first) / (len(out.outputs[0].token_ids) - 1) * 1e3
        )
    print(
        f"[result] elapsed_s={elapsed:.6f} requests={len(outputs)} "
        f"generated_tokens={generated} throughput={generated / elapsed:.3f} "
        f"sha256={digest}"
    )
    if tpot_ms:
        print(
            f"[tpot_ms] median={statistics.median(tpot_ms):.3f} "
            f"p95={p95(tpot_ms):.3f} n={len(tpot_ms)}"
        )


if __name__ == "__main__":
    main()
