"""Ensure archived masks and timing schema reject malformed E1 output."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_results import validate, validate_trace


class TraceValidationTests(unittest.TestCase):
    def test_trace_rejects_float_bitmask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            arrays = {}
            for layer in range(32):
                prefix = f"layer{layer:02d}_"
                arrays[prefix + "q_scaled"] = np.zeros(
                    (1, 32, 1, 128), dtype=np.float32
                )
                arrays[prefix + "k_chunks"] = np.zeros((1, 2, 1, 128), dtype=np.float32)
                arrays[prefix + "scores"] = np.zeros((1, 32, 1), dtype=np.float32)
                arrays[prefix + "max_val"] = np.zeros((1, 32), dtype=np.float32)
                arrays[prefix + "bmask"] = np.full((1, 2, 1, 1), 3, dtype=np.int32)
                arrays[prefix + "prior"] = np.ones((1, 2, 1, 2), dtype=np.float32)
            np.savez_compressed(path, **arrays)
            validate_trace(path, batch_size=1, layers=32)

            arrays["layer00_bmask"] = arrays["layer00_bmask"].astype(np.float32)
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "bitmask dtype"):
                validate_trace(path, batch_size=1, layers=32)

            arrays["layer00_bmask"] = np.ones((1, 2, 1, 1), dtype=np.int32)
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "diagonal bit"):
                validate_trace(path, batch_size=1, layers=32)

            arrays["layer00_bmask"] = np.full((1, 2, 1, 1), 3, dtype=np.int32)
            arrays["layer00_max_val"][0, 0] = 1.0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "max differs"):
                validate_trace(path, batch_size=1, layers=32)

    def test_route_share_must_match_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_bytes = b"test lock bytes"
            (root / "e1.lock.json").write_bytes(lock_bytes)
            (root / "trace-4k-b1.npz").write_bytes(b"test trace")
            lock = {
                "upstream_commit": "source",
                "model_revision": "model",
                "new_tokens": 2,
                "repetitions": 1,
                "cases": [{"name": "4k-b1", "target_tokens": 4096, "batch_size": 1}],
            }
            sample = {"count": 1}
            case = {
                "status": "complete",
                "batch_size": 1,
                "prompt_tokens": 4096,
                "new_tokens": 2,
                "trace_file": "trace-4k-b1.npz",
                "tpot_ms": sample,
                "tpot_trials_ms": [[10.0]],
                "stage_ms": {name: sample for name in ("stage1", "stage2", "stage3")},
                "stage_per_token_ms": {
                    name: [0.5] for name in ("stage1", "stage2", "stage3")
                },
                "router_call_ms": sample,
                "router_call_per_token_ms": [2.0],
                "router_host_enqueue_ms": sample,
                "router_host_enqueue_per_token_ms": [1.0],
                "router_stage_sum_ms": sample,
                "router_call_share_of_tpot_p50_approx": 0.2,
                "active": {
                    "density": {"count": 1, "p50": 0.5, "p95": 0.5, "mean": 0.5},
                    "density_per_token": [0.5],
                    "active_blocks_per_token": [1.0],
                    "possible_blocks_per_token": [2.0],
                },
            }
            results = {
                "status": "complete",
                "upstream_commit": "source",
                "model_revision": "model",
                "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                "cases": {"4k-b1": case},
            }
            with patch("validate_results.validate_trace"):
                validate(results, lock, root)
                case["router_call_share_of_tpot_p50_approx"] = 0.1
                with self.assertRaisesRegex(ValueError, "router share inconsistent"):
                    validate(results, lock, root)


if __name__ == "__main__":
    unittest.main()
