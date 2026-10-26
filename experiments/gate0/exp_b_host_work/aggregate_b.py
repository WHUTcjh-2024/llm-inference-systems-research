#!/usr/bin/env python3
"""Aggregate formal Experiment B runs into one table.

Each run contributes its own median/p95; the table then reports the median
across runs (and the observed spread). Decode rows are separated from prefill
rows by `scheduled_tokens <= max_num_seqs`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict


RUN_RE = re.compile(
    r"exp_b_(?P<workload>uniform|mixed|preemption)_async-(?P<async_mode>on|off|default)_"
    r"(?P<stamp>\d{8}_\d{6})\.summary\.csv$"
)


def p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def run_metrics(path: str, max_num_seqs: int) -> dict[str, float] | None:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 5:
        return None

    def floats(name: str) -> list[float]:
        out = []
        for row in rows:
            try:
                out.append(float(row[name]))
            except (KeyError, ValueError):
                pass
        return out

    decode = [
        row
        for row in rows
        if 0 < int(float(row.get("scheduled_tokens", 0))) <= max_num_seqs
    ]
    if len(decode) < 3:
        decode = [row for row in rows if float(row.get("scheduled_tokens", 0)) > 0]

    def col(rows_subset: list[dict], name: str) -> list[float]:
        return [
            float(row[name])
            for row in rows_subset
            if row.get(name, "") not in ("", None) and float(row[name]) >= 0
        ]

    d_vals = col(decode, "D_step_us")
    h_vals = col(decode, "H_res_us")
    ratios = [
        float(row["H_res_us"]) / float(row["D_step_us"])
        for row in decode
        if float(row.get("D_step_us", 0)) > 0 and float(row.get("H_res_us", 0)) >= 0
    ]
    k_vals = col(decode, "k_actual")
    wait_total = sum(col(rows, "engine_wait_us"))
    engine_total = sum(col(rows, "engine_step_us")) + sum(
        col(rows, "engine_step_bq_us")
    )
    starts = col(rows, "device_start_us")
    ends = col(rows, "device_end_us")
    gaps = [max(0.0, float(row.get("gpu_gap_before_us", 0))) for row in rows]
    span = (max(ends) - min(starts)) if starts and ends else 0.0

    return {
        "steps": len(rows),
        "decode_steps": len(decode),
        "D_step_med_us": median(d_vals),
        "D_step_p95_us": p95(d_vals),
        "H_res_med_us": median(h_vals),
        "H_over_D_med": median(ratios) * 100 if ratios else float("nan"),
        "H_over_D_p95": p95(ratios) * 100 if ratios else float("nan"),
        "k_med": median(k_vals),
        "k_max": max(k_vals) if k_vals else float("nan"),
        "wait_frac": wait_total / engine_total * 100 if engine_total else float("nan"),
        "gap_frac": sum(gaps) / span * 100 if span else float("nan"),
        "free_blocks_med": median(col(rows, "free_blocks")),
        "waiting_med": median(col(rows, "waiting")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    parser.add_argument("--out", default="results/formal_aggregate.csv")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.dir, "*.summary.csv"))):
        match = RUN_RE.search(os.path.basename(path))
        if not match:
            continue
        metrics = run_metrics(path, args.max_num_seqs)
        if metrics is None:
            continue
        metrics["run"] = os.path.basename(path)  # type: ignore[assignment]
        groups[(match["workload"], match["async_mode"])].append(metrics)

    if not groups:
        raise SystemExit(f"No formal runs found under {args.dir}")

    fields = [
        "config",
        "runs",
        "decode_steps_med",
        "D_step_med_us",
        "D_step_p95_med_us",
        "H_res_med_us",
        "H_over_D_med_pct",
        "H_over_D_p95_med_pct",
        "k_med",
        "k_max",
        "wait_frac_med_pct",
        "gap_frac_med_pct",
        "free_blocks_med",
        "waiting_med",
    ]
    out_rows = []
    for (workload, async_mode), metrics_list in sorted(groups.items()):
        def agg(name: str) -> str:
            values = [
                float(m[name])  # type: ignore[arg-type]
                for m in metrics_list
                if m[name] == m[name]  # not NaN
            ]
            if not values:
                return "nan"
            if name == "k_max":
                return f"{max(values):.0f}"
            return f"{median(values):.3f}"

        row = {
            "config": f"{workload}_async-{async_mode}",
            "runs": str(len(metrics_list)),
            "decode_steps_med": agg("decode_steps"),
            "D_step_med_us": agg("D_step_med_us"),
            "D_step_p95_med_us": agg("D_step_p95_us"),
            "H_res_med_us": agg("H_res_med_us"),
            "H_over_D_med_pct": agg("H_over_D_med"),
            "H_over_D_p95_med_pct": agg("H_over_D_p95"),
            "k_med": agg("k_med"),
            "k_max": agg("k_max"),
            "wait_frac_med_pct": agg("wait_frac"),
            "gap_frac_med_pct": agg("gap_frac"),
            "free_blocks_med": agg("free_blocks_med"),
            "waiting_med": agg("waiting_med"),
        }
        out_rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print("config,runs,D_step_med,H_res_med,H/D_med,H/D_p95,k_med,k_max,wait%,gap%")
    for row in out_rows:
        print(
            f"{row['config']},{row['runs']},{row['D_step_med_us']},"
            f"{row['H_res_med_us']},{row['H_over_D_med_pct']},"
            f"{row['H_over_D_p95_med_pct']},{row['k_med']},{row['k_max']},"
            f"{row['wait_frac_med_pct']},{row['gap_frac_med_pct']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
