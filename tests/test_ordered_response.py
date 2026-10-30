"""Scientific invariants of the order-aware coherent-response repair."""
from __future__ import annotations

import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import expit
from scipy.stats import binom

from llm_coherence.analysis.coherent_baseline import fit_ordered_utilities, ordered_objective
from llm_coherence.analysis.ordered_response import (
    draw_order_counts, fit_ordered_responses, monotonicity_probabilities,
    order_block_pmf, order_objective, simulate_order_monotonicity,
)
from llm_coherence.analysis.repair_noisy_agent import (
    cross_validate_orders, fold_assignment, matched_curve_counts, prediction_metrics, repeat_checks,
    verify_validation_inputs,
)
from llm_coherence.analysis.rebuttal_inputs import digest


class OrderedFitTests(unittest.TestCase):
    def test_analytic_gradient_including_position_offset(self):
        rng = np.random.default_rng(91)
        n = np.full((2, 3, 7, 2), 10.)
        k = rng.binomial(10, .6, size=n.shape)
        parameters = rng.normal(size=2 * 7 + 3 - 1 + 1)
        value, analytic = order_objective(parameters, k, n, 1e-3)
        numerical = []
        for i in range(len(parameters)):
            change = np.zeros_like(parameters)
            change[i] = 1e-5
            above = order_objective(parameters + change, k, n, 1e-3)[0]
            below = order_objective(parameters - change, k, n, 1e-3)[0]
            numerical.append((above - below) / 2e-5)
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(analytic, numerical, atol=2e-10, rtol=1e-7)

    def test_zero_position_recovers_original_pooled_objective_and_fit(self):
        rng = np.random.default_rng(92)
        n = np.full((2, 4, 7, 2), 10.)
        k = rng.binomial(10, np.linspace(.1, .9, 7)[None, None, :, None], size=n.shape)
        parameters = rng.normal(size=2 * 7 + 4 - 1)
        original, gradient = ordered_objective(parameters, k.sum(axis=-1), n.sum(axis=-1), 1e-5)
        adjusted, adjusted_gradient = order_objective(np.r_[parameters, 0.], k, n, 1e-5)
        self.assertAlmostEqual(original, adjusted, places=14)
        np.testing.assert_allclose(gradient, adjusted_gradient[:-1], atol=1e-14)
        old = fit_ordered_utilities(k.sum(axis=-1), n.sum(axis=-1))
        new = fit_ordered_responses(k, n, position_effect=False)
        self.assertTrue(new.diagnostics["optimizer_success"])
        self.assertEqual(new.position_offset, 0)
        np.testing.assert_allclose(old.probabilities, new.probabilities[..., 0], atol=2e-6)
        np.testing.assert_array_equal(new.probabilities[..., 0], new.probabilities[..., 1])

    def test_known_position_effect_recovery_orientation_and_shared_outcomes(self):
        u = np.array([np.linspace(-3, 0, 7), np.linspace(0, 3, 7)])
        v = np.array([0., -1., .5, 2.])
        delta = .8
        p = expit(u[:, None, :, None] - v[None, :, None, None] + delta * np.array([1., -1.]))
        n = np.full(p.shape, 10000)
        k = np.rint(n * p)
        ids = np.array([[f"{li}:{ti}" for ti in range(7)] for li in range(2)])
        ids[1, 0] = ids[0, 6]
        fit = fit_ordered_responses(k, n, outcome_ids=ids, ridge=1e-7)
        self.assertTrue(fit.diagnostics["optimizer_success"])
        self.assertAlmostEqual(fit.position_offset, delta, places=3)
        self.assertLess(np.max(np.abs(fit.probabilities - p)), 2e-4)
        self.assertAlmostEqual(fit.ladder_utilities[0, 6], fit.ladder_utilities[1, 0], places=12)
        self.assertEqual(fit.diagnostics["shared_outcome_groups"], 1)
        self.assertTrue(np.all(np.diff(fit.probabilities, axis=2) >= -1e-14))
        reversed_fit = fit_ordered_responses(k[..., ::-1], n, outcome_ids=ids, ridge=1e-7)
        self.assertAlmostEqual(reversed_fit.position_offset, -delta, places=3)
        np.testing.assert_allclose(reversed_fit.probabilities[..., ::-1], fit.probabilities, atol=1e-8)

    def test_invalid_counts_and_unobserved_orders_rejected(self):
        n = np.ones((1, 2, 7, 2))
        with self.assertRaisesRegex(ValueError, "Invalid observed"):
            fit_ordered_responses(n + 1, n)
        n[..., 1] = 0
        with self.assertRaisesRegex(ValueError, "Both presentation"):
            fit_ordered_responses(n, n)


class ResponseDistributionTests(unittest.TestCase):
    def test_block_pmfs_have_correct_means_variances_and_endpoints(self):
        p, n = np.array([[0., .2], [.7, 1.]]), 10
        support = np.arange(n + 1)
        for rho in (0., .5, 1.):
            pmf = order_block_pmf(p, n, rho)
            np.testing.assert_allclose(pmf.sum(axis=-1), 1, atol=1e-13)
            np.testing.assert_allclose((pmf * support).sum(axis=-1), n * p, atol=1e-12)
            variance = (pmf * (support - n * p[..., None]) ** 2).sum(axis=-1)
            np.testing.assert_allclose(variance, n * p * (1 - p) * (1 + (n - 1) * rho), atol=1e-11)
            self.assertEqual(pmf[0, 0, 0], 1)
            self.assertEqual(pmf[1, 1, n], 1)

    def test_monotonicity_recursion_matches_exhaustive_enumeration(self):
        p = np.array([[.1, .05], [.25, .1], [.4, .2], [.5, .3], [.7, .45], [.8, .6], [.9, .8]])
        pmfs = [np.convolve(binom.pmf([0, 1], 1, a), binom.pmf([0, 1], 1, b)) for a, b in p]
        expected = sum(np.prod([pmfs[t][c] for t, c in enumerate(counts)])
                       for counts in itertools.product(range(3), repeat=7)
                       if all(a <= b for a, b in zip(counts[:-1], counts[1:])))
        for rho in (0., .5, 1.):
            self.assertAlmostEqual(float(monotonicity_probabilities(p, 1, rho)), expected, places=13)

    def test_draws_keep_presentation_orientation_and_correlation_limits(self):
        always_a = np.tile([1., 0.], (200, 7, 1))
        rng = np.random.default_rng(93)
        for rho in (0., .5, 1.):
            k = draw_order_counts(always_a, 10, repeat_correlation=rho, rng=rng)
            self.assertTrue(np.all(k[..., 0] == 10))
            self.assertTrue(np.all(k[..., 1] == 0))
            np.testing.assert_allclose(monotonicity_probabilities(always_a, 10, rho), 1.)
        k = draw_order_counts(np.full((1000, 7, 2), .5), 10, repeat_correlation=1, rng=rng)
        self.assertTrue(np.isin(k, [0, 10]).all())

    def test_exact_curve_probability_matches_full_count_simulation(self):
        p = np.stack([np.linspace(.15, .95, 7), np.linspace(.05, .85, 7)], axis=-1)
        for rho in (0., .5, 1.):
            expected = float(monotonicity_probabilities(p, 10, rho))
            counts = draw_order_counts(np.broadcast_to(p, (30000, 7, 2)), 10,
                                       repeat_correlation=rho, rng=np.random.default_rng(94))
            observed = np.all(np.diff(counts.sum(axis=-1), axis=-1) >= 0, axis=-1).mean()
            self.assertLess(abs(observed - expected), 6 * np.sqrt(expected * (1 - expected) / 30000))
        result = simulate_order_monotonicity(np.broadcast_to(p, (2, 3, 7, 2)), np.ones((2, 3), bool), 10)
        self.assertAlmostEqual(result["exact_expected_rate"], float(monotonicity_probabilities(p, 10, 0)))

    def test_repeat_expectations_allow_different_probabilities_in_each_order(self):
        p = np.broadcast_to([.8, .2], (1, 1, 7, 2))
        n = np.full(p.shape, 2)
        k = np.broadcast_to([2, 0], p.shape)
        row = repeat_checks(p, k, n, np.ones((1, 1), bool))[0]
        self.assertAlmostEqual(row["predicted_mean_order_gap"], .6)
        self.assertAlmostEqual(row["expected_unanimous_group_fraction"], .68)
        self.assertAlmostEqual(row["expected_opposite_unanimous_orders_fraction"], .8 ** 4 + .2 ** 4)
        self.assertAlmostEqual(row["expected_mean_squared_order_gap"], .6 ** 2 + 2 * .8 * .2 / 2)
        self.assertEqual(row["observed_unanimous_group_fraction"], 1.)


class MatchedValidationTests(unittest.TestCase):
    def test_unchanged_inputs_and_reader_can_move_to_a_different_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sources.csv"
            reader = Path(directory) / "noise_response_inputs.py"
            source.write_text("verified input selection\n", encoding="utf-8")
            reader.write_text("# verified response reader\n", encoding="utf-8")
            validation = {"source_files": {"/previous/checkout/sources.csv": digest(source)},
                          "code_files": {"/previous/checkout/noise_response_inputs.py": digest(reader)}}
            verify_validation_inputs(validation, [source], reader)
            source.write_text("changed input selection\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different source inputs"):
                verify_validation_inputs(validation, [source], reader)

    def test_relocated_reader_with_changed_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / "noise_response_inputs.py"
            reader.write_text("# original reader\n", encoding="utf-8")
            validation = {"source_files": {},
                          "code_files": {"/previous/noise_response_inputs.py": digest(reader)}}
            reader.write_text("# different parser\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reader differs"):
                verify_validation_inputs(validation, [], reader)

    def test_ambiguous_recorded_reader_versions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / "noise_response_inputs.py"
            reader.write_text("# original reader\n", encoding="utf-8")
            validation = {"source_files": {},
                          "code_files": {"/one/noise_response_inputs.py": digest(reader),
                                         "/two/noise_response_inputs.py": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "reader differs"):
                verify_validation_inputs(validation, [], reader)

    def test_incomplete_or_unverified_tier_excludes_whole_curve(self):
        labels = np.ones((1, 3, 7, 2, 2), dtype=np.int8)
        codes = np.ones((1, 3, 7), dtype=np.int8)
        labels[0, 1, 2, 0, 0] = -1
        codes[0, 2, 2] = 0
        panel = SimpleNamespace(trials_per_direction=2, probabilities=np.ones((1, 3, 7)),
                                successes=(labels == 1).sum(axis=(-1, -2)), totals=(labels >= 0).sum(axis=(-1, -2)))
        k, n, mask = matched_curve_counts(panel, labels, codes, np.ones((1, 3), bool))
        np.testing.assert_array_equal(mask, [[True, False, False]])
        self.assertEqual(n.sum(), 28)
        self.assertTrue(np.all(k[:, 1:] == 0))
        labels[0, 0, 0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "reconcile"):
            matched_curve_counts(panel, labels, codes, np.ones((1, 3), bool))

    def test_both_models_hold_out_all_answers_on_the_same_curves(self):
        rng = np.random.default_rng(95)
        n = np.full((3, 6, 7, 2), 10)
        k = rng.binomial(n, np.linspace(.15, .85, 7)[None, None, :, None])
        eligible = np.ones((3, 6), bool)
        ids = np.array([[f"{li}:{ti}" for ti in range(7)] for li in range(3)])
        assignment = fold_assignment(eligible.shape, 3, 42)
        changed = k.copy()
        changed[assignment == 0] = n[assignment == 0] - changed[assignment == 0]
        for position in (False, True):
            kwargs = {"position_effect": position, "assignment": assignment}
            first, diagnostics = cross_validate_orders(k, n, eligible, ids, **kwargs)
            second, _ = cross_validate_orders(changed, n, eligible, ids, **kwargs)
            self.assertTrue(all(d["status"] == "complete" for d in diagnostics))
            np.testing.assert_array_equal(first[assignment == 0], second[assignment == 0])
            metrics = prediction_metrics(first, k, n, eligible)
            self.assertEqual(metrics["curves"], 18)
            self.assertEqual(metrics["responses"], 2520)

    def test_unsupported_heldout_ladder_is_reported_and_not_scored(self):
        k, n = np.ones((3, 3, 7, 2)), np.full((3, 3, 7, 2), 2)
        eligible = np.array([[True, False, False], [True, True, True], [True, True, True]])
        assignment = np.array([[0, 1, 2], [0, 1, 2], [1, 2, 0]])
        ids = np.array([[f"{li}:{ti}" for ti in range(7)] for li in range(3)])
        predicted, diagnostics = cross_validate_orders(k, n, eligible, ids,
                                                        position_effect=True, assignment=assignment)
        self.assertEqual(diagnostics[0]["unsupported_test_curves"], 1)
        self.assertTrue(np.isnan(predicted[0, 0]).all())
        self.assertEqual(prediction_metrics(predicted, k, n, eligible)["curves"], 6)


if __name__ == "__main__":
    unittest.main()
