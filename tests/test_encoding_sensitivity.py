import unittest
import numpy as np

from llm_coherence.analysis.encoding_sensitivity import binary_jt_significant, summarize_curves
from llm_coherence.analysis.analyze_7tier_coherence import jonckheere_terpstra_test


class EncodingSensitivityTests(unittest.TestCase):
    def test_binary_statistic_matches_trial_enumeration(self):
        rng = np.random.default_rng(4)
        totals = rng.integers(1, 21, size=(150, 7))
        counts = (rng.random((150, 7)) * (totals + 1)).astype(int)
        expected = [jonckheere_terpstra_test(list(zip(a.tolist(), (n-a).tolist())))["significant"]
                    for a, n in zip(counts, totals)]
        np.testing.assert_array_equal(binary_jt_significant(counts, totals), expected)

    def test_missing_curve_is_excluded_not_imputed(self):
        p = np.array([[0, .1, .2, .3, .4, .5, .6], [1, .8, .6, .4, .2, .1, 0],
                      [np.nan, .5, .5, .5, .5, .5, .5]])
        r = summarize_curves(p)
        self.assertEqual(r["complete_curves"], 2)
        self.assertEqual(r["missing_curves"], 1)
        self.assertEqual(r["monotonicity_pct"], 50)
        self.assertIsNone(r["jt_pct"])

    def test_constant_curve_is_monotonic_and_exact_fit(self):
        r = summarize_curves(np.full((2, 7), .5))
        self.assertEqual(r["monotonicity_pct"], 100)
        self.assertEqual(r["r2"], 1)

    def test_no_complete_curves_returns_unavailable(self):
        r = summarize_curves(np.full((3, 7), np.nan))
        self.assertIsNone(r["monotonicity_pct"])
        self.assertEqual(r["missing_curves"], 3)

    def test_zero_observations_cannot_supply_a_complete_curve(self):
        with self.assertRaisesRegex(ValueError, "positive counts"):
            summarize_curves(np.zeros((1, 7)), successes=np.zeros((1, 7)), totals=np.zeros((1, 7)))

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            binary_jt_significant(np.full((1, 7), .5), np.ones((1, 7)))


if __name__ == "__main__":
    unittest.main()
