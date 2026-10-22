#!/usr/bin/env python3
"""Summarize Gate 0 host instrumentation without making a go/no-go claim.

The CSV records Python method wall times. CUDA events around execute_model
measure a stream window that may include host enqueue gaps, not pure device
execution. The backlog counter ends when execute_model returns, not when the
GPU finishes. Neither is a validated D_step or k_actual measurement.
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
)
FIELDS = (
    "step",
    *(f"{phase}_ns" for phase in PHASES),
    "host_method_sum_ns",
    "device_window_ns",
    "schedule_period_ns",
    "k_host_proxy",
)


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def summarize(rows: list[dict[str, int]], field: str) -> None:
    values = [row[field] / 1e3 for row in rows if row[field] > 0]
    if values:
        print(
            f"{field:25} median={statistics.median(values):9.1f} us "
            f"p95={p95(values):9.1f} us n={len(values)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    phases: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    windows: dict[int, int] = {}
    backlog: dict[int, int] = {}
    schedule_starts: dict[int, int] = {}

    with open(args.input, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            kind = row["kind"]
            if kind == "host":
                phases[step][row["phase"]] += int(row["dur_ns"])
                if row["phase"] == "schedule":
                    schedule_starts[step] = int(row["start_ns"])
            elif kind in {"device_window", "device"}:
                windows[step] = int(row["dur_ns"])
            elif kind == "k":
                backlog[step] = int(row["value"])

    ordered_starts = sorted(schedule_starts.items())
    periods = {
        step: next_start - start
        for (step, start), (_, next_start) in zip(
            ordered_starts, ordered_starts[1:]
        )
        if next_start > start
    }

    rows = []
    for step in sorted(phases):
        values = phases[step]
        row = {"step": step, **{f"{phase}_ns": values.get(phase, 0) for phase in PHASES}}
        row["host_method_sum_ns"] = sum(
            values.get(phase, 0) for phase in ("schedule", "execute", "postprocess")
        )
        row["device_window_ns"] = windows.get(step, 0)
        row["schedule_period_ns"] = periods.get(step, 0)
        row["k_host_proxy"] = backlog.get(step, 0)
        rows.append(row)

    if not rows:
        raise SystemExit("No host steps recorded; check patch status and engine process.")

    print(f"steps={len(rows)}")
    for phase in PHASES:
        summarize(rows, f"{phase}_ns")
    for field in ("host_method_sum_ns", "device_window_ns", "schedule_period_ns"):
        summarize(rows, field)
    proxies = [row["k_host_proxy"] for row in rows]
    print(f"k_host_proxy              median={statistics.median(proxies):.1f} max={max(proxies)}")
    print(
        "EXPLORATORY ONLY: device_window includes possible host enqueue gaps; "
        "k_host_proxy is not GPU in-flight depth. Classify residual host work "
        "and GPU idle with a CUDA/NVTX timeline before applying Gate 0 thresholds."
    )

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
