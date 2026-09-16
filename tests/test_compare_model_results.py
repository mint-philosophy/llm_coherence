"""Tests for paired, ladder-clustered model comparisons."""

from __future__ import annotations

import unittest
from copy import deepcopy

from llm_coherence.analysis.compare_model_results import (
    COHERENCE_METRICS,
    compare_coherence_summaries,
    compare_within_ladder_summaries,
    holm_adjust,
    paired_ladder_analysis,
)


class PairedModelComparisonTests(unittest.TestCase):
    @staticmethod
    def _complete_summary() -> dict:
        row = {
            "variation_id": "category_1",
            "category": "category",
            "n_comparisons": 10,
            "n_monotonic": 5,
            "n_erratic_flips": 5,
            **{metric: 0.5 for metric in COHERENCE_METRICS},
        }
        return {
            "n_tiers": 7,
            "aggregate": {
                "overall": {
                    "n_variation_sets": 1,
                    "n_total_comparisons": 10,
                    "n_tiers": 7,
                    **{metric: 0.5 for metric in COHERENCE_METRICS},
                }
            },
            "per_variation_set": [row],
        }

    def test_different_tier_designs_are_rejected(self) -> None:
        left = self._complete_summary()
        right = deepcopy(left)
        right["n_tiers"] = right["aggregate"]["overall"]["n_tiers"] = 3
        with self.assertRaisesRegex(ValueError, "tier counts differ"):
            compare_coherence_summaries(
                left, right, bootstrap_samples=10, randomization_samples=10, seed=1
            )

    def test_unverifiable_headline_is_not_published(self) -> None:
        left = self._complete_summary()
        right = deepcopy(left)
        right["aggregate"]["overall"]["mean_isotonic_r2"] = 0.0
        report = compare_coherence_summaries(
            left, right, bootstrap_samples=10, randomization_samples=10, seed=1
        )
        metric = report["metrics"]["mean_isotonic_r2"]
        self.assertNotIn("right_headline_aggregate", metric)
        self.assertEqual(metric["right_macro_mean"], 0.5)
        self.assertIn("omitted", metric["headline_aggregate_note"])

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
            left_row.update(
                n_comparisons=100,
                n_monotonic=40 + index,
                n_erratic_flips=40 + index,
            )
            right_row.update(
                n_comparisons=100,
                n_monotonic=50 + index,
                n_erratic_flips=50 + index,
            )
            left_rows.append(left_row)
            right_rows.append(right_row)
        left_overall = {metric: 0.42 for metric in COHERENCE_METRICS}
        right_overall = {metric: 0.52 for metric in COHERENCE_METRICS}
        left_overall.update(n_variation_sets=5, n_total_comparisons=500, n_tiers=7)
        right_overall.update(n_variation_sets=5, n_total_comparisons=500, n_tiers=7)

        result = compare_coherence_summaries(
            {
                "aggregate": {"overall": left_overall},
                "per_variation_set": left_rows,
            },
            {
                "aggregate": {"overall": right_overall},
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

    def test_within_ladder_comparison_rejects_incomplete_coverage(self) -> None:
        left = {
            "n_ladders": 1,
            "n_ladders_expected": 1,
            "n_total_pairs": 41,
            "n_requests_expected": 42,
            "overall_accuracy": 1.0,
            "parse_errors": 0,
            "per_ladder": [{"ladder_id": "category_1", "accuracy": 1.0, "n": 41}],
        }
        right = {
            "n_ladders": 1,
            "n_ladders_expected": 1,
            "n_total_pairs": 42,
            "n_requests_expected": 42,
            "overall_accuracy": 1.0,
            "parse_errors": 0,
            "per_ladder": [{"ladder_id": "category_1", "accuracy": 1.0, "n": 42}],
        }

        with self.assertRaisesRegex(
            ValueError, "left within-ladder summary is incomplete"
        ):
            compare_within_ladder_summaries(
                left,
                right,
                bootstrap_samples=100,
                randomization_samples=100,
                seed=1,
            )

    def test_coherence_comparison_rejects_inconsistent_count_and_rate(self) -> None:
        row = {
            "variation_id": "category_1",
            "category": "category",
            "n_comparisons": 10,
            "n_monotonic": 0,
            "n_erratic_flips": 0,
            **{metric: 0.0 for metric in COHERENCE_METRICS},
        }
        row["monotonicity_rate"] = 1.0
        summary = {
            "aggregate": {
                "overall": {
                    "n_variation_sets": 1,
                    "n_total_comparisons": 10,
                    "n_tiers": 7,
                }
            },
            "per_variation_set": [row],
        }

        with self.assertRaisesRegex(ValueError, "inconsistent monotonicity_rate"):
            compare_coherence_summaries(
                summary,
                summary,
                bootstrap_samples=10,
                randomization_samples=10,
                seed=1,
            )

    def test_within_ladder_comparison_rejects_invalid_accuracy(self) -> None:
        summary = {
            "n_ladders": 1,
            "n_ladders_expected": 1,
            "n_total_pairs": 42,
            "n_requests_expected": 42,
            "overall_accuracy": 2.0,
            "parse_errors": 0,
            "per_ladder": [{"ladder_id": "category_1", "accuracy": 2.0, "n": 42}],
        }

        with self.assertRaisesRegex(ValueError, "invalid accuracy"):
            compare_within_ladder_summaries(
                summary,
                summary,
                bootstrap_samples=10,
                randomization_samples=10,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
