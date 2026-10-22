#!/usr/bin/env python3
"""Summarize GPU activity inside bench_b.py's NVTX measurement range.

This reports device activity and idle gaps, not their causal attribution.
Nsight instrumentation perturbs timing, so compare only like-for-like traces.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics


def quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="nsys export --type sqlite output")
    args = parser.parse_args()

    with sqlite3.connect(args.sqlite) as db:
        ranges = db.execute(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE text = 'gate0:measure' AND end IS NOT NULL"
        ).fetchall()
        if len(ranges) != 1:
            raise SystemExit(f"Expected one gate0:measure range, found {len(ranges)}")
        begin, end = ranges[0]
        intervals = []
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
            intervals.extend(
                (max(start, begin), min(stop, end))
                for start, stop in db.execute(
                    f"SELECT start, end FROM {table} "
                    "WHERE start < ? AND end > ?",
                    (end, begin),
                )
                if min(stop, end) > max(start, begin)
            )

        graph_launches = db.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId = s.id "
            "WHERE s.value LIKE 'cudaGraphLaunch%' "
            "AND r.start >= ? AND r.start < ?",
            (begin, end),
        ).fetchone()[0]

    if not intervals:
        raise SystemExit("No GPU activity in measurement range")
    intervals.sort()
    merged = [list(intervals[0])]
    for start, stop in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])

    busy = sum(stop - start for start, stop in merged)
    gap_ranges = [
        (previous[1], start)
        for previous, (start, _) in zip(merged, merged[1:])
    ]
    gaps = [stop - start for start, stop in gap_ranges]
    active_span = merged[-1][1] - merged[0][0]
    print(f"measure_ms={(end - begin) / 1e6:.3f}")
    print(f"device_active_span_ms={active_span / 1e6:.3f}")
    print(f"gpu_activity_intervals={len(intervals)} graph_launch_api_calls={graph_launches}")
    print(f"gpu_busy_fraction={busy / active_span:.4f}")
    print(f"gpu_internal_idle_ms={(active_span - busy) / 1e6:.3f}")
    if gaps:
        print(
            f"idle_gaps={len(gaps)} median_us={statistics.median(gaps) / 1e3:.2f} "
            f"p95_us={quantile(gaps, 0.95) / 1e3:.2f} "
            f"max_us={max(gaps) / 1e3:.2f}"
        )
        for threshold_us in (10, 50, 100, 500):
            selected = [gap for gap in gaps if gap >= threshold_us * 1000]
            print(
                f"idle_ge_{threshold_us}us count={len(selected)} "
                f"total_ms={sum(selected) / 1e6:.3f}"
            )
        print("largest_idle_gaps_ms_after_measure_start:")
        for start, stop in sorted(
            gap_ranges, key=lambda gap: gap[1] - gap[0], reverse=True
        )[:5]:
            print(f"  {((start + stop) / 2 - begin) / 1e6:.3f} "
                  f"duration_ms={(stop - start) / 1e6:.3f}")


if __name__ == "__main__":
    main()
