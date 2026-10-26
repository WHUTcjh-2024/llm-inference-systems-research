#!/usr/bin/env python3
"""Summarize Gate 0 host instrumentation.

Definitions used here (and the limits of each):

  D_step        device time between the CUDA events around execute_model.
                Because async scheduling returns before the kernels finish,
                the end event is enqueued after the step's kernels, so the
                interval is the main-stream device window of that step.
  step_time     device start-to-start cadence of consecutive steps.
  H_busy        wall time of one EngineCore iteration minus the time it
                spent blocked in AsyncOutputFuture.result(); i.e. host CPU
                work per step after the model forward is graph-replayed.
  H_res         equal to H_busy: schedule, input/metadata preparation,
                graph launch, sampling launch, output update. This is the
                work that remains on the host once the forward is captured.
  k_actual      number of steps whose device window has not finished at the
                instant the host finished enqueueing step s (j <= s).
  GPU idle      positive gaps between consecutive device windows on the main
                stream. It is not attributed to a cause here.

The CSV also carries scheduler snapshots (waiting/running/free KV blocks/
scheduled tokens) so that resource pressure can be separated from dependency
waits when the traces are read together.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


PHASES = (
    "schedule",
    "block_alloc",
    "execute",
    "prepare",
    "attn_metadata",
    "flashinfer_metadata",
    "sample",
    "postprocess",
    "engine_step",
    "engine_step_bq",
    "engine_wait",
    "engine_update",
    "graph_replay",
)
TOP_LEVEL = ("schedule", "execute", "engine_update", "postprocess")
RESOURCES = (
    "waiting",
    "running",
    "free_blocks",
    "scheduled_tokens",
    "running_preempted_ever",
)
FIELDS = (
    "step",
    *PHASES,
    "H_busy_us",
    "H_res_us",
    "unaccounted_us",
    "D_step_us",
    "device_period_us",
    "schedule_period_us",
    "k_actual",
    "k_host_proxy",
    "h_res_over_D",
    "gpu_gap_before_us",
    "device_start_us",
    "device_end_us",
    *RESOURCES,
)


def p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def stat_line(name: str, values: list[float], unit: str = "us") -> str:
    if not values:
        return f"{name:24} n=0"
    return (
        f"{name:24} median={statistics.median(values):9.1f}{unit} "
        f"p95={p95(values):9.1f}{unit} max={max(values):9.1f}{unit} n={len(values)}"
    )


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    host: list[dict] = []
    device: dict[int, tuple[int, int]] = {}
    resources: dict[int, dict[str, int]] = defaultdict(dict)
    k_proxy: dict[int, int] = {}
    epoch_ns = 0

    with open(args.input, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kind = row["kind"]
            if kind == "epoch":
                epoch_ns = int(row["value"])
            elif kind == "host":
                host.append(row)
            elif kind == "device_window":
                device[int(row["step"])] = (int(row["start_ns"]), int(row["end_ns"]))
            elif kind == "resource":
                resources[int(row["step"])][row["phase"]] = int(row["value"])
            elif kind == "k":
                k_proxy[int(row["step"])] = int(row["value"])

    if not host:
        raise SystemExit("No host rows recorded; check patch status and engine process.")
    if not device or not epoch_ns:
        raise SystemExit(
            "No absolute device timeline. Run with device events enabled and a "
            "CUDA-capable process."
        )

    engine_rows = [
        row for row in host if row["phase"] in ("engine_step", "engine_step_bq")
    ]
    if not engine_rows:
        raise SystemExit("No EngineCore step rows; cannot separate host work from waits.")

    wait_intervals = [
        (int(row["start_ns"]), int(row["end_ns"]))
        for row in host
        if row["phase"] == "engine_wait"
    ]

    def phase_intervals(phase: str) -> list[tuple[int, int, int]]:
        return [
            (int(row["start_ns"]), int(row["end_ns"]), int(row["dur_ns"]))
            for row in host
            if row["phase"] == phase
        ]

    phases = {phase: phase_intervals(phase) for phase in PHASES}

    # Per-engine-iteration host accounting. Intervals are attributed by start
    # time so a wait/update that belongs to an older queued step still lands
    # inside the iteration that performed it.
    iterations: list[dict] = []
    for row in sorted(engine_rows, key=lambda r: int(r["start_ns"])):
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        duration = end - start
        wait = sum(overlap(start, end, w0, w1) for w0, w1 in wait_intervals)
        busy = max(0, duration - wait)
        entry = {
            "step": int(row["step"]),
            "start": start,
            "end": end,
            "H_busy_us": busy / 1e3,
            "engine_wait_us": wait / 1e3,
            "engine_total_us": duration / 1e3,
        }
        for phase in PHASES:
            if phase in ("engine_step", "engine_step_bq", "engine_wait"):
                continue
            total = sum(
                overlap(start, end, s0, s1) for s0, s1, _ in phases[phase]
            )
            entry[f"{phase}_us"] = total / 1e3
        top_sum = sum(entry.get(f"{phase}_us", 0.0) for phase in TOP_LEVEL)
        entry["unaccounted_us"] = max(0.0, entry["H_busy_us"] - top_sum)
        iterations.append(entry)

    # Device-derived metrics.
    steps = sorted(device)
    device_start_us = {s: device[s][0] for s in steps}
    device_end_us = {s: device[s][1] for s in steps}
    D_us = {s: device[s][1] - device[s][0] for s in steps}
    device_period_us = {
        s: device_start_us[nxt] - device_start_us[s]
        for s, nxt in zip(steps, steps[1:])
        if device_start_us[nxt] > device_start_us[s]
    }
    gpu_gap_us = {
        s: device_start_us[nxt] - device_end_us[s]
        for s, nxt in zip(steps, steps[1:])
    }

    schedule_start = {
        int(row["step"]): int(row["start_ns"]) for row in host if row["phase"] == "schedule"
    }
    ordered_sched = sorted(schedule_start.items())
    schedule_period_us = {
        step: (nxt - start) / 1e3
        for (step, start), (_, nxt) in zip(ordered_sched, ordered_sched[1:])
        if nxt > start
    }

    execute_end = {
        int(row["step"]): int(row["end_ns"]) for row in host if row["phase"] == "execute"
    }

    # k_actual at the instant step s finished being enqueued.
    k_actual: dict[int, int] = {}
    for s in steps:
        if s not in execute_end:
            continue
        host_us = (execute_end[s] - epoch_ns) / 1e3
        k_actual[s] = sum(
            1 for j in steps if j <= s and device_end_us[j] > host_us
        )

    rows = []
    for entry in iterations:
        s = entry["step"]
        row = {"step": s}
        for phase in PHASES:
            row[f"{phase}_us"] = entry.get(f"{phase}_us", 0.0)
        row["engine_wait_us"] = entry["engine_wait_us"]
        row["H_busy_us"] = entry["H_busy_us"]
        row["H_res_us"] = entry["H_busy_us"]
        row["unaccounted_us"] = entry["unaccounted_us"]
        row["D_step_us"] = D_us.get(s, 0)
        row["device_period_us"] = device_period_us.get(s, 0)
        row["schedule_period_us"] = schedule_period_us.get(s, 0)
        row["k_actual"] = k_actual.get(s, -1)
        row["k_host_proxy"] = k_proxy.get(s, -1)
        row["h_res_over_D"] = (
            entry["H_busy_us"] / D_us[s] if D_us.get(s) else 0.0
        )
        row["gpu_gap_before_us"] = gpu_gap_us.get(s, 0)
        row["device_start_us"] = device_start_us.get(s, 0)
        row["device_end_us"] = device_end_us.get(s, 0)
        for name in RESOURCES:
            row[name] = resources.get(s, {}).get(name, -1)
        rows.append(row)

    if not rows:
        raise SystemExit("No EngineCore iterations; check the engine_step patch.")

    def values(field: str) -> list[float]:
        return [row[field] for row in rows if isinstance(row[field], (int, float)) and row[field] >= 0]

    print(f"steps={len(rows)} measurement_ms={(device_end_us[steps[-1]] - device_start_us[steps[0]]) / 1e3:.1f}")
    print(stat_line("D_step", values("D_step_us")))
    print(stat_line("device_period", values("device_period_us")))
    print(stat_line("schedule_period", values("schedule_period_us")))
    print(stat_line("H_busy (H_res)", values("H_res_us")))
    for phase in TOP_LEVEL + ("prepare", "attn_metadata", "flashinfer_metadata", "sample", "graph_replay", "engine_wait"):
        print(stat_line(phase, values(f"{phase}_us")))
    print(stat_line("unaccounted", values("unaccounted_us")))
    ratios = [
        row["H_res_us"] / row["D_step_us"]
        for row in rows
        if row["D_step_us"] > 0 and row["H_res_us"] >= 0
    ]
    if ratios:
        print(
            f"{'H_res/D_step':24} median={statistics.median(ratios) * 100:8.2f}% "
            f"p95={p95(ratios) * 100:8.2f}% n={len(ratios)}"
        )
    ratios_period = [
        row["H_res_us"] / row["device_period_us"]
        for row in rows
        if row["device_period_us"] > 0 and row["H_res_us"] >= 0
    ]
    if ratios_period:
        print(
            f"{'H_res/device_period':24} median={statistics.median(ratios_period) * 100:8.2f}% "
            f"p95={p95(ratios_period) * 100:8.2f}% n={len(ratios_period)}"
        )
    k_values = [v for v in k_actual.values() if v >= 0]
    if k_values:
        print(
            f"{'k_actual':24} median={statistics.median(k_values):.0f} "
            f"p95={p95(k_values):.0f} max={max(k_values)} n={len(k_values)}"
        )
    gaps = [gap / 1e3 for gap in gpu_gap_us.values() if gap > 0]
    if gaps:
        span_us = device_end_us[steps[-1]] - device_start_us[steps[0]]
        print(
            f"{'gpu_gap':24} count={len(gaps)} total_ms={sum(gaps) / 1e3:.3f} "
            f"fraction={sum(gaps) / span_us * 100:.2f}% "
            f"median_us={statistics.median(gaps):.1f} p95_us={p95(gaps):.1f} "
            f"max_us={max(gaps):.1f}"
        )
    for name in RESOURCES:
        vals = [row[name] for row in rows if row[name] >= 0]
        if vals:
            print(
                f"{name:24} median={statistics.median(vals):.1f} "
                f"min={min(vals)} max={max(vals)}"
            )

    print(
        "\nDefinitions: H_res = EngineCore iteration wall time minus time blocked "
        "in AsyncOutputFuture.result(); D_step = CUDA-event window around "
        "execute_model; k_actual counts steps still unfinished on device when the "
        "host finishes enqueueing step s. GPU gaps are unattributed."
    )

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in FIELDS})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
