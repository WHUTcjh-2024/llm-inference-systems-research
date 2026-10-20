"""E0 checks for deferred scoring, reactivation, and GQA output semantics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e0_oracle import (
    HeadResult,
    Node,
    l2_nodes,
    pack_group,
    run_trials,
    solve_certified,
    solve_full,
)


class OracleTests(unittest.TestCase):
    def test_deferred_node_reactivates_after_threshold_drops(self) -> None:
        # t_0 = 0.5; after one Halley step t_1 < 0.3.
        scores = np.array([2.0, 0.6])
        nodes = (Node((0,), 1.0), Node((1,), 0.3))
        full = solve_full(scores)
        certified = solve_certified(scores, nodes)
        self.assertEqual(certified.max_scores, 1)
        self.assertEqual(certified.exact_scores, 2)
        self.assertGreaterEqual(certified.reactivated_nodes, 1)
        self.assertLess(full.steps[0].t_after, 0.3)
        self.assertEqual(full.steps, certified.steps)
        np.testing.assert_array_equal(full.probabilities, certified.probabilities)

    def test_negative_and_tied_maxima(self) -> None:
        scores = np.array([-2.0, -2.0, -5.0, -8.0])
        nodes = (Node((0, 1), -1.0), Node((2, 3), -2.5))
        full = solve_full(scores)
        certified = solve_certified(scores, nodes)
        self.assertEqual(full.steps, certified.steps)
        np.testing.assert_array_equal(full.probabilities, certified.probabilities)

    def test_final_output_rechecks_nodes_after_third_update(self) -> None:
        x = np.array(
            [
                -0.006135453530398895,
                -0.16140805240183703,
                -0.4291075440693802,
                -0.19426540009811016,
                -1.0014253514734064,
                -0.1830054604186567,
                -1.2219961231471341,
                0.27242367126644756,
                -0.15303119157691594,
                0.7249873447533367,
            ]
        )
        nodes = tuple(Node((i,), float(value)) for i, value in enumerate(x))
        full = solve_full(x * 2.0)
        certified = solve_certified(x * 2.0, nodes)
        self.assertLess(full.steps[-1].t_after, x[1])
        self.assertTrue(all(step.t_before >= x[1] for step in full.steps))
        self.assertGreater(full.probabilities[1], 0.0)
        self.assertEqual(full.steps, certified.steps)
        np.testing.assert_array_equal(full.probabilities, certified.probabilities)

    def test_disjoint_gqa_support_and_word_boundary(self) -> None:
        left = np.zeros(33)
        right = np.zeros(33)
        left[0] = 1.0
        right[32] = 4.0
        heads = (HeadResult((), left, 0, 0, 0), HeadResult((), right, 0, 0, 0))
        plain, words = pack_group(heads, estimate_diag=False, sigma=1e8)
        np.testing.assert_array_equal(plain[[0, 32, 33]], [0.5, 2.0, 1.0])
        np.testing.assert_array_equal(words.view(np.uint32), [1, 3])
        scaled, scaled_words = pack_group(heads, estimate_diag=True, sigma=1e8)
        np.testing.assert_array_equal(scaled_words, words)
        self.assertEqual(scaled[-1], 1.0)
        self.assertTrue(np.all(scaled[[0, 32]] > 0.0))
        self.assertTrue(np.all(scaled[1:32] == 0.0))

    def test_bad_bound_and_all_sentinel_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            solve_certified(np.array([2.0, 0.0]), (Node((0,), 0.9), Node((1,), 0.0)))
        with self.assertRaisesRegex(ValueError, "partition"):
            solve_certified(np.array([2.0, 0.0]), (Node((0,), 1.0), Node((0,), 1.0)))
        with self.assertRaisesRegex(ValueError, "sentinel"):
            solve_full(np.array([-1e9, -1e9]))

    def test_clustered_geometry_reduces_exact_scores(self) -> None:
        totals = run_trials(seed=0, trials=100)
        self.assertEqual(totals["full_scores"], 12800)
        self.assertLess(totals["exact_scores"], totals["full_scores"])
        self.assertGreater(totals["reactivated_nodes"], 0)

    def test_l2_bound_covers_cpu_scores(self) -> None:
        query = np.array([1.0, -2.0])
        keys = np.array([[1.0, 1.0], [1.1, 1.0], [-4.0, 2.0]])
        nodes = l2_nodes(query, keys, group_size=2)
        for node in nodes:
            self.assertGreaterEqual(
                node.upper_x, max(float(np.dot(query, keys[i])) for i in node.indices)
            )


if __name__ == "__main__":
    unittest.main()
