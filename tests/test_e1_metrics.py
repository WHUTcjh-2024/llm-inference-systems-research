"""Local checks for signed bitpacking and summary boundaries."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e1_metrics import mean_active_blocks, percentile, stats_ms, summary


class MetricsTests(unittest.TestCase):
    def test_signed_words_count_all_bits(self) -> None:
        packed = np.array([[[[0, 1]], [[3, -2147483648]]]], dtype=np.int32)
        self.assertEqual(mean_active_blocks(packed, possible=64), 2.0)

    def test_padding_bits_are_rejected(self) -> None:
        packed = np.array([[[[-2147483648]]]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "padding"):
            mean_active_blocks(packed, possible=5)

    def test_small_sample_percentile(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.95), 2.9)
        self.assertEqual(summary([0.25, 0.75])["p50"], 0.5)
        self.assertEqual(stats_ms([1.0, 2.0])["p50_ms"], 1.5)


if __name__ == "__main__":
    unittest.main()
