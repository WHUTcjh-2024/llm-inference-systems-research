#!/usr/bin/env python3
"""Experiment B: run one vLLM workload with host-phase instrumentation.

The engine must run in this process for the monkey-patches to see calls.
We set VLLM_ENABLE_V1_MULTIPROCESSING=0 and request the "uni" executor
where the installed version supports it.

Examples:
  python bench_b.py --workload uniform --out r.csv
  python bench_b.py --workload mixed --chunked-prefill --out r.csv
  python bench_b.py --workload preemption --num-prompts 128 \
      --gpu-memory-utilization 0.60 --out r.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument(
        "--workload",
        default="mixed",
        choices=["uniform", "mixed", "preemption"],
    )
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument(
        "--async-scheduling",
        default="default",
        choices=["on", "off", "default"],
    )
    p.add_argument("--chunked-prefill", action="store_true")
    p.add_argument(
        "--no-uni-executor",
        action="store_true",
        help="do not request distributed_executor_backend=uni",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--attn-backend", default="FLASHINFER")
    p.add_argument("--out", required=True)
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--no-device-events", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--warmup-full",
        action="store_true",
        help="run the complete workload once before recording",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Must be set before importing vllm. The experiment requires the engine
    # to run in this process; otherwise the monkey-patches never see worker
    # calls and the CSV stays empty.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    from vllm import LLM, SamplingParams

    import instrument_vllm as inst
    from workloads import build_prompts

    rec = inst.Recorder(
        args.out,
        nvtx=args.nvtx,
        device_events=not args.no_device_events,
    )
    patch_status = inst.patch_all(rec)
    for line in patch_status:
        print(f"[patch] {line}", file=sys.stderr)
    for phase in ("schedule", "execute", "attn_metadata", "flashinfer_metadata"):
        if not any(line.startswith(f"{phase}: patched ") for line in patch_status):
            rec.finalize()
            raise RuntimeError(f"Required instrumentation phase is missing: {phase}")

    kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
    )
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if not args.no_uni_executor:
        kwargs["distributed_executor_backend"] = "uni"
    if args.chunked_prefill:
        kwargs["enable_chunked_prefill"] = True
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.async_scheduling != "default":
        kwargs["async_scheduling"] = args.async_scheduling == "on"
    if args.attn_backend:
        kwargs["attention_backend"] = args.attn_backend

    # A rejected experiment setting must fail rather than silently changing
    # the async/chunked-prefill control group.
    llm = LLM(**kwargs)
    config = llm.llm_engine.vllm_config
    actual_async = config.scheduler_config.async_scheduling
    print(
        f"[config] async_scheduling={actual_async} "
        f"executor={config.parallel_config.distributed_executor_backend} "
        f"runner_v2={config.use_v2_model_runner} "
        f"chunked_prefill={config.scheduler_config.enable_chunked_prefill} "
        f"attn_backend={config.attention_config.backend} "
        f"cudagraph_mode={config.compilation_config.cudagraph_mode} "
        f"prefix_caching={config.cache_config.enable_prefix_caching}",
        file=sys.stderr,
    )
    if not config.use_v2_model_runner:
        raise RuntimeError("vLLM did not select the instrumented V2 model runner")
    if args.async_scheduling != "default":
        requested_async = args.async_scheduling == "on"
        if actual_async != requested_async:
            raise RuntimeError("vLLM did not apply the requested async setting")

    prompts = build_prompts(
        args.workload, args.num_prompts, args.prompt_len, args.seed
    )
    params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,
    )

    if args.warmup_full:
        print("[warmup] complete workload", file=sys.stderr)
        llm.generate(prompts, params, use_tqdm=False)
    elif args.warmup > 0:
        warm = prompts[: min(args.warmup, len(prompts))]
        llm.generate(
            warm,
            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
            use_tqdm=False,
        )

    import torch

    rec.start()
    t0 = time.perf_counter()
    torch.cuda.nvtx.range_push("gate0:measure")
    try:
        outputs = llm.generate(prompts, params, use_tqdm=True)
    finally:
        torch.cuda.nvtx.range_pop()
        rec.finalize()
    elapsed = time.perf_counter() - t0
    generated_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    output_digest = hashlib.sha256(
        json.dumps(
            [out.outputs[0].token_ids for out in outputs],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    print(f"[output] requests={len(outputs)} sha256={output_digest}", file=sys.stderr)
    print(
        f"[throughput] elapsed_s={elapsed:.6f} "
        f"generated_tokens={generated_tokens} "
        f"output_tokens_per_s={generated_tokens / elapsed:.3f}",
        file=sys.stderr,
    )

    print(f"[done] rows={rec.rows} csv={args.out}", file=sys.stderr)
    if rec.rows == 0:
        raise RuntimeError(
            "No host rows recorded. Check the engine process and patch status."
        )


if __name__ == "__main__":
    main()
