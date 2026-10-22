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
import os
import sys


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
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_llm(LLM, kwargs: dict):
    """Retry without optional kwargs that this vLLM version rejects."""
    optional = [
        "async_scheduling",
        "enable_chunked_prefill",
        "distributed_executor_backend",
        "max_model_len",
        "trust_remote_code",
    ]
    while True:
        try:
            return LLM(**kwargs)
        except (TypeError, ValueError) as exc:
            msg = str(exc)
            removed = None
            for key in optional:
                if key in kwargs and key in msg:
                    removed = key
                    break
            if removed is None:
                for key in reversed(optional):
                    if key in kwargs:
                        removed = key
                        break
            if removed is None:
                raise
            print(
                f"[warn] LLM() rejected {removed}; retrying without it "
                f"({msg})",
                file=sys.stderr,
            )
            kwargs.pop(removed)


def main() -> None:
    args = parse_args()

    # Must be set before importing vllm. The experiment requires the engine
    # to run in this process; otherwise the monkey-patches never see worker
    # calls and the CSV stays empty.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if args.attn_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attn_backend

    from vllm import LLM, SamplingParams

    import instrument_vllm as inst
    from workloads import build_prompts

    rec = inst.Recorder(
        args.out,
        nvtx=args.nvtx,
        device_events=not args.no_device_events,
    )
    for line in inst.patch_all(rec):
        print(f"[patch] {line}", file=sys.stderr)

    kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
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

    llm = build_llm(LLM, kwargs)

    prompts = build_prompts(
        args.workload, args.num_prompts, args.prompt_len, args.seed
    )
    params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,
    )

    if args.warmup > 0:
        warm = prompts[: min(args.warmup, len(prompts))]
        llm.generate(
            warm,
            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
            use_tqdm=False,
        )

    rec.start()
    llm.generate(prompts, params, use_tqdm=True)
    rec.finalize()

    print(f"[done] rows={rec.rows} csv={args.out}", file=sys.stderr)
    if rec.rows == 0:
        print(
            "[warn] no rows recorded. The engine likely ran in a separate "
            "process. Keep VLLM_ENABLE_V1_MULTIPROCESSING=0, or check the "
            "installed vLLM version for an in-process engine option.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
