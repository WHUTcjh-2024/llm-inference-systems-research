#!/usr/bin/env python3
"""Cross-check the Gate 0 instrumentation against an Nsight Systems trace.

Reports, inside the `gate0:measure` range:
  - GPU kernel/memcpy coverage and idle gaps;
  - CUDA runtime API calls that can block the host (stream/device/event
    synchronize and synchronous memcpy), with total time per name;
  - graph-launch API calls issued while the GPU was already busy (a direct
    sign that the host ran ahead of the device);
  - NVTX phase ranges emitted by bench_b.py (host-side timeline);
  - which host NVTX/API activity overlaps the largest GPU gaps.

This validates placement and overlap. It does not attribute a gap to a cause.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics


def quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="nsys export --type sqlite output")
    parser.add_argument("--steps-csv", help="optional analyze_b per-step CSV")
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

        runtime = []
        if table_exists(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
            runtime = db.execute(
                "SELECT s.value, r.start, r.end "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId = s.id "
                "WHERE r.start >= ? AND r.start < ? AND r.end IS NOT NULL",
                (begin, end),
            ).fetchall()

        nvtx = []
        if table_exists(db, "NVTX_EVENTS"):
            nvtx = db.execute(
                "SELECT text, start, end FROM NVTX_EVENTS "
                "WHERE start >= ? AND start < ? AND end IS NOT NULL "
                "AND text LIKE 'gate0:%'",
                (begin, end),
            ).fetchall()

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
    print(f"gpu_activity_intervals={len(intervals)}")
    print(f"gpu_busy_fraction={busy / active_span:.4f}")
    print(f"gpu_internal_idle_ms={(active_span - busy) / 1e6:.3f}")
    kernel_only = [s for s in intervals]
    print(
        f"gpu_interval_us median={statistics.median([e - s for s, e in kernel_only]) / 1e3:.2f} "
        f"p95={quantile([e - s for s, e in kernel_only], 0.95) / 1e3:.2f}"
    )
    if gaps:
        print(
            f"idle_gaps={len(gaps)} median_us={statistics.median(gaps) / 1e3:.2f} "
            f"p95_us={quantile(gaps, 0.95) / 1e3:.2f} max_us={max(gaps) / 1e3:.2f}"
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
            print(
                f"  at_ms={((start + stop) / 2 - begin) / 1e6:.3f} "
                f"duration_ms={(stop - start) / 1e6:.3f}"
            )

    # Runtime API: blocking calls and graph launches.
    sync_names = ("Synchronize", "cudaMemcpy", "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol")
    sync_rows = [
        row for row in runtime if any(name in row[0] for name in sync_names)
    ]
    graph_rows = [row for row in runtime if "cudaGraphLaunch" in row[0]]
    print(f"runtime_api_calls={len(runtime)}")
    print(
        f"blocking_api_calls={len(sync_rows)} total_ms={sum(r[2] - r[1] for r in sync_rows) / 1e6:.3f}"
    )
    per_name: dict[str, list[int]] = {}
    for name, start, stop in runtime:
        per_name.setdefault(name, []).append(stop - start)
    for name, durations in sorted(
        per_name.items(), key=lambda item: sum(item[1]), reverse=True
    )[:8]:
        print(
            f"  api {name[:48]:48} n={len(durations):5d} "
            f"total_ms={sum(durations) / 1e6:8.3f} "
            f"median_us={statistics.median(durations) / 1e3:8.2f}"
        )

    busy_at_launch = 0
    for name, start, stop in graph_rows:
        if any(a <= start < b for a, b in merged):
            busy_at_launch += 1
    print(
        f"graph_launch_api_calls={len(graph_rows)} issued_while_gpu_busy={busy_at_launch}"
    )

    # NVTX host phases emitted by bench_b.py.
    per_phase: dict[str, list[int]] = {}
    for text, start, stop in nvtx:
        per_phase.setdefault(text, []).append(stop - start)
    for name, durations in sorted(per_phase.items()):
        print(
            f"nvtx {name:24} n={len(durations):5d} "
            f"median_us={statistics.median(durations) / 1e3:9.2f} "
            f"p95_us={quantile(durations, 0.95) / 1e3:9.2f}"
        )

    # What host activity overlaps the largest GPU gaps?
    if gap_ranges:
        print("largest_gap_overlaps:")
        for start, stop in sorted(
            gap_ranges, key=lambda gap: gap[1] - gap[0], reverse=True
        )[:3]:
            overlapping = [
                (text, s, e)
                for text, s, e in nvtx
                if s < stop and e > start
            ]
            api = [
                (name, s, e)
                for name, s, e in runtime
                if s < stop and e > start
            ]
            print(
                f"  gap_ms={(stop - start) / 1e6:.3f} "
                f"nvtx={[(t, (e - s) / 1e3) for t, s, e in overlapping[:4]]} "
                f"api={[(n, (e - s) / 1e3) for n, s, e in api[-4:]]}"
            )

    if args.steps_csv:
        import csv as csvmod

        with open(args.steps_csv, newline="", encoding="utf-8") as handle:
            rows = list(csvmod.DictReader(handle))
        dsteps = [float(r["D_step_us"]) for r in rows if float(r["D_step_us"]) > 0]
        if dsteps:
            print(
                f"csv_steps={len(rows)} csv_D_step_median_us={statistics.median(dsteps):.1f} "
                f"csv_D_step_p95_us={quantile([int(v * 1e3) for v in dsteps], 0.95) / 1e3:.1f}"
            )


if __name__ == "__main__":
    main()
