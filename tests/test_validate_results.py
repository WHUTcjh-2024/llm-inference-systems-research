"""Ensure archived masks remain exact integers and malformed traces fail."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_results import validate_trace


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
                arrays[prefix + "bmask"] = np.ones((1, 2, 1, 1), dtype=np.int32)
                arrays[prefix + "prior"] = np.ones((1, 2, 1, 2), dtype=np.float32)
            np.savez_compressed(path, **arrays)
            validate_trace(path, batch_size=1, layers=32)

            arrays["layer00_bmask"] = arrays["layer00_bmask"].astype(np.float32)
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "bitmask dtype"):
                validate_trace(path, batch_size=1, layers=32)


if __name__ == "__main__":
    unittest.main()
