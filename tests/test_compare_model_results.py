"""Tests for paired, ladder-clustered model comparisons."""

from __future__ import annotations

import unittest

from llm_coherence.analysis.compare_model_results import (
    COHERENCE_METRICS,
    compare_coherence_summaries,
    holm_adjust,
    paired_ladder_analysis,
)


class PairedModelComparisonTests(unittest.TestCase):
    def test_paired_analysis_uses_ladders_as_resampling_units(self) -> None:
        result = paired_ladder_analysis(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            bootstrap_samples=100,
            randomization_samples=100,
            seed=7,
        )

        self.assertEqual(result["n_paired_ladders"], 4)
        self.assertEqual(result["right_minus_left"], 1.0)
        self.assertEqual(result["difference_cluster_bootstrap_ci95"], [1.0, 1.0])
        self.assertEqual(result["paired_randomization_method"], "exact_sign_flip")
        self.assertEqual(result["paired_randomization_p_value"], 0.125)
        self.assertEqual(result["resampling_unit"], "ladder")

    def test_holm_adjustment_is_monotone_in_rank_order(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})

        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_coherence_comparison_pairs_matching_variation_ids(self) -> None:
        left_rows = []
        right_rows = []
        for index in range(5):
            left_row = {
                "variation_id": f"category_{index}",
                "category": "category",
            }
            right_row = dict(left_row)
            for metric in COHERENCE_METRICS:
                left_row[metric] = 0.4 + index * 0.01
                right_row[metric] = 0.5 + index * 0.01
            left_rows.append(left_row)
            right_rows.append(right_row)

        result = compare_coherence_summaries(
            {
                "aggregate": {"overall": {}},
                "per_variation_set": left_rows,
            },
            {
                "aggregate": {"overall": {}},
                "per_variation_set": right_rows,
            },
            bootstrap_samples=100,
            randomization_samples=100,
            seed=3,
        )

        self.assertEqual(result["n_paired_ladders"], 5)
        self.assertAlmostEqual(
            result["metrics"]["monotonicity_rate"]["right_minus_left"],
            0.1,
        )
        self.assertIsNotNone(
            result["exploratory_monotonicity_by_category"]["category"][
                "exploratory_paired_analysis"
            ]
        )


if __name__ == "__main__":
    unittest.main()
