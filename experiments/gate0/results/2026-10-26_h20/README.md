# Gate 0 results — 2026-10-26 — NVIDIA H20

Status: **in progress**. Experiment A is complete. Experiment B contains one
valid smoke run; the formal workload matrix and Nsight validation are pending
because the instance was rebooted into no-GPU mode before they could run.
`week1_report.md` is a draft that records the completed evidence and marks
every decision-relevant gap as pending.

## Environment (recorded during the GPU boot)

| item | value |
|---|---|
| GPU | 1 × NVIDIA H20, 97,871 MiB, compute capability 9.0 |
| driver | 580.65.06 |
| CUDA toolkit | 13.0.88 (`nvcc` at `/usr/local/cuda/bin/nvcc`) |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.29.0 (PyPI wheel `vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl`, sha256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`) |
| FlashInfer | 0.6.18 (JIT-compiled SM90 sampling ops on first use) |
| Transformers | 5.20.2 |
| Python | 3.12.3 |
| OS / kernel | Ubuntu 22.04.5 LTS / 5.15.0-130-generic |
| Model | `Qwen/Qwen3-8B`, HF revision `b968826d9c46dd6066d109eabc6255188de91218`, 15.26 GiB safetensors |
| Repo base | `345ba8b` plus the harness changes committed with these results |
| Clocks | not locked |

Serving configuration for the smoke run: V2 model runner, `FLASHINFER`
attention backend, `uni` executor, in-process engine
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`), chunked prefill on, prefix cache off,
`--max-num-seqs 8`, `--max-model-len 4096`, `--gpu-memory-utilization 0.70`,
`async_scheduling=on`.

## Experiment A

Source: `experiments/gate0/exp_a_clone_semantics/`.

- `exp_a_*` — graph with `cudaMallocAsync`/`cudaFreeAsync` nodes. The first
  instantiation succeeds; the second returns `cudaErrorNotSupported`, so this
  clone path cannot be tested for workspace isolation.
- `exp_a_external_*` — graph with an external 4 MiB buffer:
  - shared captured pointer, concurrent: 200/200 runs report the same address
    and 200/200 show corruption;
  - explicitly rebound second buffer, serial: 0/200 corruption, median
    36.641 ms;
  - explicitly rebound second buffer, concurrent: 0/200 corruption, median
    18.322 ms (two 18 ms kernels overlap).
- `clone_external_nodes_summary.txt` — Nsight node timeline: concurrent groups
  overlap 20/20 kernel pairs; the serial group overlaps 0/20.
- Memory deltas at 4 MiB workspace: two execs cost 2.00 MiB at instantiation
  (≈1 MiB each); first replay of e1 and e2 each add 0.00 MiB, i.e. no lazy
  backing allocation was observed at this size.

## Experiment B

`exp_b/smoke/` — one valid smoke run (8 requests, 8 output tokens, mixed
prompts) after the unit fix. It validates the instrumentation; it is not a
formal Gate 0 sample:

- `D_step` median 6.211 ms; `H_res` median 1.405 ms; `H_res/D_step` median
  22.60% (p95 39.62%); `k_actual` median/max 2; GPU inter-step gap ≈0.4 µs
  (0.64% of the measurement span). The host spends ~93% of EngineCore
  iteration wall time blocked in `AsyncOutputFuture.result()` (decode steady
  state ≈79%), i.e. waiting for the sampled-token D2H copy.
- The first step is a 4352-token chunked prefill (`D_step` 462.9 ms,
  `H_res/D_step` 4.7%); the medians above are decode-only
  (`scheduled_tokens <= 8`).

`exp_b/invalid_unitbug/` — the first smoke run, preserved. Its device
timestamps were stored with a 1000× unit error (`elapsed_time` returns ms, the
recorder wrote µs as `ms × 1e6`). `k_actual` and `H_res/D_step` from those
files are invalid; they are kept only as a failure record.

## Not yet measured

- Formal `uniform`, `mixed` and `mixed + async off` runs (3 repetitions each).
- Preemption workload.
- Nsight Systems trace of the mixed workload and cross-check against the
  instrumented timeline.
- Clean (non-instrumented) control with engine multiprocessing on/off.
- `k_res` vs `k_dep` attribution and the Week 1 decision report.

## Next GPU session

The instance was rebooted into no-GPU mode, so the formal matrix has not run.
One command runs the remaining measurement set:

```bash
cd experiments/gate0/exp_b_host_work
MODEL=/root/autodl-tmp/models/Qwen3-8B-hf \
PYTHON=/root/autodl-tmp/gate0-env/bin/python \
REPEATS=3 WITH_NSYS=1 ./run_formal.sh
```

It runs uniform/mixed/mixed-async-off three times each, the clean
multiprocessing control, and Nsight traces, then writes
`results/formal_aggregate.csv`.

Binary profiler traces (`*.nsys-rep`, `*.sqlite`) remain local per the
repository data policy; their summaries are committed.

