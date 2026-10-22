#!/usr/bin/env python3
"""Summarize Experiment B.

Definitions (per scheduler step):
  H_step = schedule + execute + postprocess      (top-level host work)
  H_res  = host work that cannot be captured into a graph
           = schedule + (prepare or execute) + sample + postprocess
  H_graph = H_step - H_res
  D_step  = device time measured with CUDA events around execute_model
  step_time = H_step + D_step                    (no-overlap upper bound)
  k       = scheduled - executed, sampled at execute start

The Gate 0 threshold is H_res / step_time >= 5%, i.e. residual host work as
a fraction of the serial step, NOT H_res / H_step (which is usually close
to 1 and says little about the available speedup).

Caveat: this is a heuristic decomposition, not a proof. Print the phase
patch status from bench_b.py and adjust if your vLLM version nests calls
differently.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(0.95 * (len(ys) - 1)))))
    return ys[idx]


def ms(ns: float) -> float:
    return ns / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--graphable",
        nargs="*",
        default=[],
        help="phases to treat as graphable (subtracted from H_res)",
    )
    args = ap.parse_args()

    steps: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    devices: dict[int, int] = {}
    ks: dict[int, int] = {}
    schedule_start: dict[int, int] = {}

    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            kind = row["kind"]
            dur = int(row["dur_ns"])
            if kind == "host":
                steps[step][row["phase"]] += dur
                if row["phase"] == "schedule" and int(row["start_ns"]) > 0:
                    schedule_start[step] = int(row["start_ns"])
            elif kind == "device":
                devices[step] = dur
            elif kind == "k":
                ks[step] = int(row["value"])

    # Measured step period: time between successive scheduler calls.
    # More robust than H_step + D when CUDA events miss the worker stream.
    starts = sorted(schedule_start.items())
    periods: dict[int, int] = {}
    for i, (step, t) in enumerate(starts):
        if i + 1 < len(starts):
            periods[step] = starts[i + 1][1] - t
    if starts:
        known = [p for p in periods.values() if p > 0]
        if known:
            periods[starts[-1][0]] = int(statistics.median(known))

    graphable = set(args.graphable)
    per: list[dict] = []

    for step in sorted(steps):
        d = steps[step]

        h_step = d.get("schedule", 0) + d.get("execute", 0) + d.get("postprocess", 0)
        if h_step == 0:
            h_step = sum(d.values())

        # Build H_res without double counting nested phases.
        h_res = d.get("schedule", 0)
        if "prepare" in d:
            h_res += d["prepare"]
        elif "attn_metadata" in d:
            h_res += d["attn_metadata"]
        else:
            h_res += d.get("execute", 0)

        if "sample" in d:
            h_res += d["sample"]

        # Execute work not covered by prepare/sample.
        if "prepare" in d or "sample" in d:
            covered = d.get("prepare", 0) + d.get("sample", 0)
            h_res += max(0, d.get("execute", 0) - covered)

        h_res += d.get("postprocess", 0)

        for phase in graphable:
            h_res -= d.get(phase, 0)
        h_res = max(0, h_res)

        h_graph = max(0, h_step - h_res)
        d_ns = devices.get(step, 0)
        k = ks.get(step, 0)
        step_time_ns = h_step + d_ns
        period_ns = periods.get(step, 0)

        per.append(
            {
                "step": step,
                "h_step_ns": h_step,
                "h_res_ns": h_res,
                "h_graph_ns": h_graph,
                "d_ns": d_ns,
                "step_time_ns": step_time_ns,
                "step_period_ns": period_ns,
                "k": k,
                "h_res_ratio": h_res / h_step if h_step else 0.0,
                "h_res_over_step": h_res / step_time_ns if step_time_ns else 0.0,
                "h_res_over_period": h_res / period_ns if period_ns else 0.0,
                "h_res_over_d": h_res / d_ns if d_ns else 0.0,
            }
        )

    if not per:
        print("no steps found; check instrumentation and the patch status log")
        return

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in per]

    print(f"steps={len(per)}")
    print(
        f"H_step  median={ms(statistics.median(col('h_step_ns'))):.3f} ms  "
        f"p95={ms(p95(col('h_step_ns'))):.3f} ms"
    )
    print(
        f"H_res   median={ms(statistics.median(col('h_res_ns'))):.3f} ms  "
        f"p95={ms(p95(col('h_res_ns'))):.3f} ms"
    )
    print(
        f"D_step  median={ms(statistics.median(col('d_ns'))):.3f} ms  "
        f"p95={ms(p95(col('d_ns'))):.3f} ms"
    )
    print(
        f"H_res/step   median={statistics.median(col('h_res_over_step')):.4f}  "
        f"p95={p95(col('h_res_over_step')):.4f}"
    )
    period_ratios = [
        r["h_res_over_period"] for r in per if r["step_period_ns"] > 0
    ]
    if period_ratios:
        print(
            f"H_res/period median={statistics.median(period_ratios):.4f}  "
            f"p95={p95(period_ratios):.4f}  (measured scheduler cadence)"
        )
    print(
        f"H_res/H_step median={statistics.median(col('h_res_ratio')):.4f}"
    )
    print(
        f"H_res/D      median={statistics.median(col('h_res_over_d')):.4f}"
    )
    print(
        f"k_actual median={statistics.median(col('k')):.2f}  "
        f"max={max(col('k')):.0f}  p95={p95(col('k')):.2f}"
    )

    if statistics.median(col("d_ns")) == 0:
        print(
            "WARN: device time is zero. CUDA events likely did not record on "
            "the worker stream. Use H_res/period or an nsys run for D."
        )

    if period_ratios:
        ratio = statistics.median(period_ratios)
        denom_name = "step_period"
    else:
        ratio = statistics.median(col("h_res_over_step"))
        denom_name = "step_time"
    kmed = statistics.median(col("k"))

    if ratio < 0.05:
        print(
            f"VERDICT: residual non-graphable host work < 5% of {denom_name} "
            "on this card. Do NOT conclude the idea fails: record absolute us "
            "and retest on A100/A800. H20 prefill is much slower, so the same "
            "host us is a smaller fraction."
        )
    elif kmed <= 1:
        print(
            "VERDICT: k <= 1. Check async scheduling; if k_dep is the cause, "
            "pivot to device-side planning or Plan B (checker)."
        )
    else:
        print(
            f"VERDICT: k > 1 and residual host work >= 5% of {denom_name}. "
            "Proceed to attribution and the two-class comparison "
            "(sync vs clone/SetParams)."
        )

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
            w.writeheader()
            w.writerows(per)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
