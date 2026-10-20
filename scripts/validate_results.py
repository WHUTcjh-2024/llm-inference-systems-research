"""Reject incomplete or internally inconsistent E1 output before handoff."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from e1_metrics import mean_active_blocks


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_trace(path: Path, batch_size: int, layers: int) -> None:
    names = ("q_scaled", "k_chunks", "scores", "max_val", "bmask", "prior")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            f"layer{layer:02d}_{name}" for layer in range(layers) for name in names
        }
        require(set(archive.files) == expected, f"{path.name}: trace keys")
        for layer in range(layers):
            q, keys, scores, max_val, mask, prior = (
                archive[f"layer{layer:02d}_{name}"] for name in names
            )
            prefix = f"{path.name}: layer {layer}"
            require(q.shape == (batch_size, 32, 1, 128), f"{prefix}: q shape")
            require(
                keys.ndim == 4
                and keys.shape[:2] == (batch_size, 2)
                and keys.shape[-1] == 128,
                f"{prefix}: keys shape",
            )
            chunks = keys.shape[2]
            require(scores.shape == (batch_size, 32, chunks), f"{prefix}: scores shape")
            require(max_val.shape == (batch_size, 32), f"{prefix}: max shape")
            require(
                mask.shape == (batch_size, 2, 1, (chunks + 32) // 32),
                f"{prefix}: bitmask shape",
            )
            require(
                prior.shape == (batch_size, 2, 1, chunks + 1), f"{prefix}: prior shape"
            )
            require(mask.dtype == np.int32, f"{prefix}: bitmask dtype")
            require(
                all(
                    array.dtype == np.float32
                    for array in (q, keys, scores, max_val, prior)
                ),
                f"{prefix}: float dtypes",
            )
            require(
                all(
                    np.isfinite(array).all()
                    for array in (q, keys, scores, max_val, prior)
                ),
                f"{prefix}: nonfinite values",
            )
            require(
                np.array_equal(max_val, np.maximum(-1e9, np.max(scores, axis=-1))),
                f"{prefix}: max differs from Stage 1 scores",
            )
            require(np.all(prior[..., -1] == 1.0), f"{prefix}: missing diagonal prior")
            diagonal_word, diagonal_bit = divmod(chunks, 32)
            require(
                np.all((mask[..., diagonal_word].view(np.uint32) >> diagonal_bit) & 1),
                f"{prefix}: missing diagonal bit",
            )
            mean_active_blocks(mask, chunks + 1)


def validate(results: dict, lock: dict, directory: Path) -> None:
    require(results.get("status") == "complete", "Run did not complete")
    expected_names = {case["name"] for case in lock["cases"]}
    require(
        set(results.get("cases", {})) == expected_names, "Case set differs from lock"
    )
    require(
        results.get("upstream_commit") == lock["upstream_commit"],
        "Source revision differs",
    )
    require(
        results.get("model_revision") == lock["model_revision"],
        "Model revision differs",
    )
    require(
        results.get("lock_sha256")
        == hashlib.sha256((directory / "e1.lock.json").read_bytes()).hexdigest(),
        "Run lock digest differs",
    )
    decode_tokens = lock["new_tokens"] - 1
    tpot_count = decode_tokens * lock["repetitions"]

    for case in lock["cases"]:
        name = case["name"]
        item = results["cases"][name]
        require(item["status"] == "complete", f"{name}: incomplete")
        require(item["batch_size"] == case["batch_size"], f"{name}: batch mismatch")
        require(
            case["target_tokens"] - 128
            <= item["prompt_tokens"]
            <= case["target_tokens"],
            f"{name}: prompt length",
        )
        require(
            item["prompt_tokens"] + item["new_tokens"] <= 16384,
            f"{name}: context overflow",
        )
        require(item["tpot_ms"]["count"] == tpot_count, f"{name}: TPOT count")
        require(
            len(item["tpot_trials_ms"]) == lock["repetitions"], f"{name}: trial count"
        )
        for trial in item["tpot_trials_ms"]:
            require(len(trial) == decode_tokens, f"{name}: trial length")
            require(
                all(math.isfinite(x) and x > 0 for x in trial), f"{name}: TPOT values"
            )
        for stage in ("stage1", "stage2", "stage3"):
            require(
                item["stage_ms"][stage]["count"] == decode_tokens,
                f"{name}: {stage} count",
            )
            values = item["stage_per_token_ms"][stage]
            require(len(values) == decode_tokens, f"{name}: {stage} length")
            require(
                all(math.isfinite(x) and x > 0 for x in values),
                f"{name}: {stage} values",
            )
        active = item["active"]
        require(active["density"]["count"] == decode_tokens, f"{name}: density count")
        require(
            all(0 <= active["density"][key] <= 1 for key in ("p50", "p95", "mean")),
            f"{name}: density summary",
        )
        require(
            len(active["density_per_token"]) == decode_tokens
            and all(0 <= x <= 1 for x in active["density_per_token"]),
            f"{name}: density values",
        )
        require(
            len(active["active_blocks_per_token"]) == decode_tokens
            and len(active["possible_blocks_per_token"]) == decode_tokens
            and all(
                0 <= selected <= possible
                for selected, possible in zip(
                    active["active_blocks_per_token"],
                    active["possible_blocks_per_token"],
                )
            ),
            f"{name}: block counts",
        )
        require(
            math.isclose(
                active["density"]["p50"],
                np.median(active["density_per_token"]),
                rel_tol=1e-5,
                abs_tol=1e-8,
            ),
            f"{name}: density summary inconsistent",
        )
        for label, values_key in (
            ("router_call_ms", "router_call_per_token_ms"),
            ("router_host_enqueue_ms", "router_host_enqueue_per_token_ms"),
        ):
            require(item[label]["count"] == decode_tokens, f"{name}: {label} count")
            values = item[values_key]
            require(len(values) == decode_tokens, f"{name}: {values_key} length")
            require(
                all(math.isfinite(x) and x > 0 for x in values),
                f"{name}: {values_key} values",
            )
        require(
            item["router_stage_sum_ms"]["count"] == decode_tokens,
            f"{name}: stage sum count",
        )
        share = item["router_call_share_of_tpot_p50_approx"]
        require(math.isfinite(share) and share > 0, f"{name}: router share")
        require(
            math.isclose(
                share,
                np.median(item["router_call_per_token_ms"])
                / np.median(
                    [value for trial in item["tpot_trials_ms"] for value in trial]
                ),
                rel_tol=1e-5,
            ),
            f"{name}: router share inconsistent",
        )
        trace_path = directory / item["trace_file"]
        require(
            trace_path.is_file() and trace_path.stat().st_size > 0,
            f"{name}: trace missing",
        )
        validate_trace(trace_path, case["batch_size"], layers=32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(
        (args.results.parent / "e1.lock.json").read_bytes() == args.lock.read_bytes(),
        "Run lock differs from requested lock",
    )
    validate(results, lock, args.results.parent)
    print("E1 result schema and completeness verified.")


if __name__ == "__main__":
    main()
