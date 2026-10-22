#!/usr/bin/env python3
"""Summarize Experiment A results.

Reads the CSV emitted by clone_semantics and prints:
  - workspace-address equality rate
  - race rate
  - wall time per replay
  - a verdict about whether exec clones are usable
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def load(path: str) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["mode"]].append(row)
    return rows


def summarize(mode: str, rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        return
    same = sum(int(r["addr_same"]) for r in rows)
    race = sum(int(r["race"]) for r in rows)
    wall = [float(r["wall_ms"]) for r in rows]
    bad0 = sum(int(r["bad0"]) for r in rows)
    bad1 = sum(int(r["bad1"]) for r in rows)

    print(f"\n[{mode}] n={n}")
    print(f"  workspace address equal: {same}/{n} ({100.0 * same / n:.1f}%)")
    print(f"  runs with corruption:    {race}/{n} ({100.0 * race / n:.1f}%)")
    print(f"  total mismatched ints:   bad0={bad0} bad1={bad1}")
    print(
        "  wall_ms: "
        f"median={statistics.median(wall):.3f} "
        f"p95={quantile(wall, 0.95):.3f} "
        f"min={min(wall):.3f}"
    )


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[idx]


def verdict(rows: dict[str, list[dict]]) -> str:
    conc = rows.get("concurrent", [])
    if not conc:
        return "no concurrent rows; rerun with --mode both"
    n = len(conc)
    same = sum(int(r["addr_same"]) for r in conc) / n
    race = sum(int(r["race"]) for r in conc) / n

    if same > 0.5 and race > 0:
        return (
            "VERDICT: clones share graph workspace and race when concurrent. "
            "Do not use exec clones for k in-flight steps; use SetParams or "
            "single-graph device-versioned ring."
        )
    if same > 0.5 and race == 0:
        return (
            "VERDICT: same address but no corruption observed. Check nsys: "
            "the driver may be serializing the two launches. If they are "
            "serialized, clones do not provide overlap."
        )
    if same <= 0.5 and race == 0:
        return (
            "VERDICT: separate workspaces, no race. Check the stderr meta file "
            "for memory delta: if e2 costs one workspace, clones are isolated "
            "but memory scales with k (viable for k=2 only)."
        )
    return (
        "VERDICT: separate workspaces but corruption observed. Unexpected; "
        "inspect the out buffer and rerun with a larger spin window."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    rows = load(args.input)
    for mode in ("serial", "concurrent"):
        if mode in rows:
            summarize(mode, rows[mode])
    print("\n" + verdict(rows))


if __name__ == "__main__":
    main()
