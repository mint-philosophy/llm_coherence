"""Continuous token scores cannot supply binomial trial-count uncertainty."""

from contextlib import ExitStack, redirect_stdout
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_coherence.analysis import analyze_7tier_coherence as analysis
from llm_coherence.reporting import make_fig_table as reporting


PROBS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def preferences(denominator=20):
    return [
        {"outcome_a": {"text": f"Tier {tier}", "tier": tier, "variation_id": "test_1"},
         "outcome_b": {"text": "Reference outcome", "comparison_category": "reference"},
         "prob_prefer_a": p, "count_prefer_a": round(p * denominator),
         "count_prefer_b": denominator - round(p * denominator)}
        for tier, p in enumerate(PROBS, 1)
    ]


def variation(groups, name="test_1"):
    return analysis.analyze_variation_set(name, "category", "quality", "positive", groups)


class LogprobAnalysisTests(unittest.TestCase):
    def test_legacy_model_and_explicit_schema_both_disable_pseudocounts(self):
        configurations = [
            {"model_key": "glm-45-base-logprobs"},
            {"result_config": {"model_key": "glm-45-base-logprobs"}},
            {"result_config": {"scoring_method": "logprobs"}},
            {"result_config": {"count_kind": "probability_derived_pseudocounts"}},
        ]
        for config in configurations:
            with self.subTest(config=config):
                group = analysis.extract_groups(preferences(), **config)[0]
                self.assertIsNone(group["tier_counts"])
                self.assertEqual(group["tier_probs"], PROBS)
        marked = preferences()
        for pref in marked:
            pref["count_kind"] = "probability_derived_pseudocounts"
        self.assertIsNone(analysis.extract_groups(marked)[0]["tier_counts"])

    def test_no_count_based_function_is_called_for_continuous_scores(self):
        functions = ["jonckheere_terpstra_test", "logistic_regression_slope",
                     "bootstrap_monotonicity_probability", "binomial_confidence_intervals",
                     "detect_erratic_flips"]
        with ExitStack() as stack:
            for name in functions:
                stack.enter_context(patch.object(analysis, name, side_effect=AssertionError(name)))
            result = analysis.analyze_group(PROBS, None)
        self.assertTrue(result["is_monotonic"])
        self.assertEqual(result["n_violations"], 0)
        self.assertEqual(result["isotonic_r2"], 1.0)
        self.assertAlmostEqual(result["kendall_tau"], 1.0)
        self.assertFalse(result["count_statistics_available"])
        self.assertIsNone(result["bootstrap_mono_prob"])
        self.assertIsNone(result["binomial_cis"]["n_significant_violations"])
        self.assertIsNone(result["logistic_slope"]["p_value"])
        self.assertIsNone(result["jonckheere_terpstra"]["significant"])
        self.assertIsNone(result["has_erratic_flip"])

    def test_arbitrary_pseudocount_denominator_cannot_change_results(self):
        results = [variation(analysis.extract_groups(preferences(n), model_key="glm-45-base-logprobs"))
                   for n in (10, 10000)]
        self.assertEqual(results[0], results[1])

    def test_all_aggregate_levels_preserve_unavailable_values(self):
        result = variation(analysis.extract_groups(preferences(), model_key="glm-45-base-logprobs"))
        agg = analysis.aggregate_results([result])
        summaries = [result, agg["overall"], *agg["by_variation_category"].values(),
                     *agg["by_comparison_category"].values()]
        for summary in summaries:
            self.assertEqual(summary["n_count_based_comparisons"], 0)
            self.assertIsNone(summary["erratic_flip_rate"])
            self.assertIsNone(summary["jt_significant_rate"])
            self.assertIsNone(summary["mean_bootstrap_mono_prob"])
            self.assertEqual(summary["monotonicity_rate"], 1.0)
        output = io.StringIO()
        with redirect_stdout(output):
            analysis.print_summary(agg)
        self.assertIn("N/A", output.getvalue())

    def test_observed_trials_keep_tests_and_mixed_denominators_use_eligible_curves(self):
        sampled = variation(analysis.extract_groups(preferences()), "sampled")
        scores = variation(analysis.extract_groups(preferences(), model_key="glm-45-base-logprobs"))
        self.assertTrue(sampled["count_statistics_available"])
        self.assertEqual(sampled["n_count_based_comparisons"], 1)
        self.assertEqual(sampled["jt_significant_rate"], 1.0)
        both = analysis.aggregate_results([sampled, scores])["overall"]
        self.assertEqual(both["n_total_comparisons"], 2)
        self.assertEqual(both["n_count_based_comparisons"], 1)
        self.assertEqual(both["jt_significant_rate"], sampled["jt_significant_rate"])
        self.assertEqual(both["mean_bootstrap_mono_prob"], sampled["mean_bootstrap_mono_prob"])

    def test_probabilities_have_identical_descriptive_statistics_with_or_without_counts(self):
        counts = [(round(p * 20), 20 - round(p * 20)) for p in PROBS]
        sampled, scores = analysis.analyze_group(PROBS, counts), analysis.analyze_group(PROBS, None)
        for key in ("is_monotonic", "n_violations", "kendall_tau", "kendall_p", "spearman_rho",
                    "isotonic_r2", "isotonic_r2_bidirectional", "inferred_direction", "prob_range"):
            self.assertEqual(sampled[key], scores[key])


class CountStatisticsReportingTests(unittest.TestCase):
    def test_tables_and_figure_omit_unavailable_statistics_from_means(self):
        models = [("glm-45-base-logprobs", "GLM Base", "GLM", "off"),
                  ("glm-45-hybrid", "GLM Instruct", "GLM", "off")]
        data = {models[0][0]: [{"monotonicity_rate": 0.5, "mean_isotonic_r2": 0.9, "jt_significant_rate": None}],
                models[1][0]: [{"monotonicity_rate": 0.8, "mean_isotonic_r2": 0.95, "jt_significant_rate": 0.8}]}
        rows, macro = reporting.collect_coherence_metrics(data, models)
        self.assertTrue(math.isnan(rows[0]["jt_pct"]))
        self.assertEqual(macro["jt_pct"], 80.0)
        self.assertEqual(macro["jt_available_models"], 1)
        with tempfile.TemporaryDirectory() as directory:
            tex, csv = reporting.write_coherence_metrics_table(rows, macro, Path(directory) / "metrics")
            self.assertIn("---", tex.read_text())
            self.assertNotIn("nan", tex.read_text())
            self.assertNotIn("nan", csv.read_text())
        fig = reporting.fig1b_metrics_triptych(data, models, 1)
        try:
            self.assertIn("n=1", fig.axes[2].get_title())
            self.assertIn("N/A", [text.get_text() for text in fig.axes[2].texts])
        finally:
            plt.close(fig)

    def test_all_unavailable_is_not_zero(self):
        self.assertTrue(math.isnan(reporting.get_jt({"jt_significant_rate": None})))
        self.assertTrue(math.isnan(reporting._finite_mean([float("nan"), None])))


if __name__ == "__main__":
    unittest.main()
