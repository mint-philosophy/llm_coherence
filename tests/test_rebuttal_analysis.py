"""Numerical and provenance checks for the five offline rebuttal analyses."""

import csv
from contextlib import redirect_stderr
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from llm_coherence.analysis.coherent_baseline import (
    draw_protocol_counts, fit_ordered_utilities, increasing_projection,
    isotonic_scores, ordered_objective, simulate_monotonicity, _tied_parameterization,
)
from llm_coherence.analysis.rebuttal_inputs import (
    category_rows, digest, load_panel, read_ladders, read_sources, within_distance_rows,
)
from llm_coherence.analysis.rebuttal_analysis import (
    main, noisy_agent_reference, permutation_reference, shuffle_tiers,
    violation_statistics,
)


class NumericalTests(unittest.TestCase):
    def test_batched_isotonic_matches_sklearn(self):
        rng = np.random.default_rng(7)
        y = np.concatenate([rng.uniform(size=(100, 7)),
                            np.array([[.3] * 7, [0, 0, 1, 1, 0, 0, 1], list(range(7))])])
        expected = np.stack([IsotonicRegression().fit_transform(np.arange(7), row) for row in y])
        np.testing.assert_allclose(increasing_projection(y), expected, atol=1e-14)
        inc, bi = isotonic_scores(np.array([list(range(7)), list(range(6, -1, -1)), [.3] * 7]))
        np.testing.assert_allclose(inc[:2], [1, 0], atol=1e-14)
        np.testing.assert_allclose(bi[:2], [1, 1], atol=1e-14)
        self.assertTrue(np.isnan(inc[2]) and np.isnan(bi[2]))

    def test_ordered_likelihood_gradient_and_fit(self):
        rng = np.random.default_rng(23)
        utilities = np.linspace(-2, 2, 7)[None, :] + np.arange(3)[:, None] * .2
        reference = np.array([0, -.4, .7, 1.2])
        p = expit(utilities[:, None, :] - reference[None, :, None])
        totals = np.full((3, 4, 7), 500)
        successes = rng.binomial(totals, p)
        parameters = rng.uniform(.1, .7, size=24)
        error = check_grad(lambda x: ordered_objective(x, successes, totals, 1e-5)[0],
                           lambda x: ordered_objective(x, successes, totals, 1e-5)[1], parameters)
        self.assertLess(error, 1e-6)
        fit = fit_ordered_utilities(successes, totals)
        self.assertTrue(fit.diagnostics["optimizer_success"])
        self.assertTrue(np.all(np.diff(fit.ladder_utilities, axis=-1) >= 0))
        self.assertEqual(fit.reference_utilities[0], 0)
        self.assertLess(np.sqrt(np.mean((fit.probabilities - p) ** 2)), .03)

    def test_noise_endpoints_repeat_correlation_and_reproducibility(self):
        p = np.tile([0, .1, .3, .5, .7, .9, 1], (100, 1))
        counts = draw_protocol_counts(p, 10, repeat_correlation=1, rng=np.random.default_rng(3))
        self.assertTrue(set(np.unique(counts)) <= {0, 10, 20})
        self.assertTrue(np.all(counts[:, 0] == 0))
        self.assertTrue(np.all(counts[:, -1] == 20))
        a = simulate_monotonicity(p[None], np.ones((1, 100), bool), 10, repetitions=10, seed=1)
        b = simulate_monotonicity(p[None], np.ones((1, 100), bool), 10, repetitions=10, seed=1)
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            draw_protocol_counts(p, 10, repeat_correlation=1.1, rng=np.random.default_rng())

    def test_identical_outcomes_share_utilities_without_reversing_tiers(self):
        rng = np.random.default_rng(29)
        ids = np.array([[f"{li}:{ti}" for ti in range(7)] for li in range(3)])
        ids[1, 1] = ids[0, 4]
        ids[2, 3] = ids[1, 5]
        n = np.full((3, 4, 7), 100)
        # Deliberately inconsistent counts at shared outcomes require the fit
        # to honor equality rather than reproducing independent ladder fits.
        k = rng.binomial(n, np.linspace(.1, .9, 7)[None, None, :])
        fit = fit_ordered_utilities(k, n, outcome_ids=ids)
        self.assertTrue(fit.diagnostics["optimizer_success"])
        self.assertAlmostEqual(fit.ladder_utilities[0, 4], fit.ladder_utilities[1, 1], places=12)
        self.assertAlmostEqual(fit.ladder_utilities[1, 5], fit.ladder_utilities[2, 3], places=12)
        self.assertTrue(np.all(np.diff(fit.ladder_utilities, axis=-1) >= 0))
        self.assertEqual(fit.diagnostics["shared_outcome_groups"], 2)
        transform, initial, _, _ = _tied_parameterization(ids, np.ones((3, 7)), 4)
        error = check_grad(lambda z: ordered_objective(transform @ z, k, n, 1e-5)[0],
                           lambda z: np.asarray(transform.T @ ordered_objective(transform @ z, k, n, 1e-5)[1]), initial)
        self.assertLess(error, 1e-6)

    def test_future_duplicate_equality_cycles_fail_explicitly(self):
        ids = np.array([[f"{li}:{ti}" for ti in range(7)] for li in range(2)])
        ids[1, 0], ids[1, 2] = ids[0, 0], ids[0, 2]
        with self.assertRaisesRegex(ValueError, "equality cycle"):
            fit_ordered_utilities(np.ones((2, 2, 7)), np.full((2, 2, 7), 2), outcome_ids=ids)

    def test_shared_permutation_and_exact_shuffled_floor(self):
        y = np.arange(7) / 10
        curves = np.array([[y, y + .01], [y, y + .02]])
        permuted = shuffle_tiers(curves, np.random.default_rng(1))
        np.testing.assert_allclose(permuted[:, 1] - permuted[:, 0], [[.01] * 7, [.02] * 7])
        all_orders = np.array(list(itertools.permutations(y)))
        self.assertEqual(np.all(np.diff(all_orders, axis=-1) >= 0, axis=-1).sum(), 1)
        flat = np.full((2, 2, 7), .3)
        result, _ = permutation_reference(flat, repetitions=3)
        self.assertEqual(result["observed"]["monotonicity"], 1)
        self.assertEqual(result["permuted"]["monotonicity"]["mean"], 1)
        self.assertIsNone(result["observed"]["bidirectional_r2_nonconstant"])
        self.assertEqual(result["constant_curves"], 4)

    def test_violation_sizes_keep_direction_ties_and_missing_separate(self):
        p = np.array([[[0, .5, .45, .55, .45, .70, .45],
                       [.5] * 7, [np.nan, 0, 0, 0, 0, 0, 0]]])
        stats, bins, transitions, drops, finite = violation_statistics(p)
        self.assertEqual(stats["curves_analyzed"], 2)
        self.assertEqual(stats["curves_missing_probability"], 1)
        self.assertEqual(stats["monotonicity_rate"], .5)
        self.assertEqual(stats["downward_steps"], 3)
        np.testing.assert_allclose(drops[0, 0], [0, .05, 0, .1, 0, .25])
        self.assertEqual([b["downward_steps"] for b in bins], [1, 1, 1, 0, 0])
        self.assertEqual(sum(r["downward_steps"] for r in transitions), 3)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ladder_path = self.root / "ladders.json"
        rows = [{"original_statement_id": f"Example category_{li}", "category": "Example category",
                 "valence": "negative", "variations": [{"tier": t, "text": f"L{li} tier{t}"}
                                                           for t in range(1, 8)]} for li in range(2)]
        self.ladder_path.write_text(json.dumps(rows))
        self.ladders = read_ladders(self.ladder_path)
        self.sources = {}
        self.references = {ri: f"reference {ri}" for ri in range(3)}
        for li, lid in enumerate(self.ladders):
            prefs = []
            for ri in self.references:
                for ti, count in enumerate([2, 5, 7, 10, 13, 16, 18], 1):
                    k = count + ri - 1
                    prefs.append({"outcome_a": {"variation_id": lid, "category": "Example category",
                                                 "text": f"L{li} tier{ti}", "tier": ti},
                                  "outcome_b": {"comparison_id": ri, "text": self.references[ri]},
                                  "count_prefer_a": k, "count_prefer_b": 20 - k,
                                  "prob_prefer_a": k / 20})
            path = self.root / f"raw{li}.json"
            path.write_text(json.dumps({"config": {"model_key": "sampled", "num_trials": 10,
                                                    "include_flipped": True}, "preferences": prefs}))
            self.sources[lid] = {"model_key": "sampled", "ladder_id": lid,
                                 "path": str(path), "sha256": digest(path)}
        self.summary = self.root / "within.json"
        self.summary.write_text(json.dumps({
            "model_key": "sampled", "per_ladder": [{"ladder_id": lid} for lid in self.ladders],
            "n_total_pairs": 83,
            "by_distance": {str(d): {"correct": 4 * (7 - d) - (1 if d == 1 else 0),
                                      "total": 4 * (7 - d) - (1 if d == 1 else 0)} for d in range(1, 7)}}))
        self.summary_record = {"model_key": "sampled", "within_summary": {
            "path": str(self.summary), "sha256": digest(self.summary)}}

    def panel(self, **kwargs):
        return load_panel("sampled", self.sources, self.ladders, expected_references=3,
                          canonical_references=self.references, **kwargs)

    def change_first(self, change, *, refresh_hash=True):
        record = next(iter(self.sources.values()))
        path = Path(record["path"])
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))
        if refresh_hash:
            record["sha256"] = digest(path)

    def test_hash_and_duplicate_tier_rejection(self):
        self.change_first(lambda d: d.update(untracked="edit"), refresh_hash=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.panel()
        self.change_first(lambda d: d["preferences"].append(d["preferences"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate or invalid tier"):
            self.panel()

    def test_wording_requires_opt_in_and_is_excluded_from_calibration(self):
        self.change_first(lambda d: d["preferences"][0]["outcome_a"].update(text="wrong input"))
        with self.assertRaisesRegex(ValueError, "wording mismatches"):
            self.panel()
        panel = self.panel(allow_input_mismatches=True)
        diagnostics, rows, _ = noisy_agent_reference(panel, repetitions=3)
        self.assertEqual(diagnostics["status"], "complete")
        self.assertEqual(diagnostics["eligible_curves"], 3)
        self.assertEqual(diagnostics["complete_curves_excluded_for_input_mismatch"], 3)
        self.assertEqual(len(rows), 7)

    def test_unparseable_probability_is_missing(self):
        self.change_first(lambda d: d["preferences"][0].update(count_prefer_a=0, count_prefer_b=0,
                                                               prob_prefer_a=.5))
        panel = self.panel()
        self.assertTrue(np.isnan(panel.probabilities[0, 0, 0]))
        self.assertEqual(panel.validation["complete_probability_curves"], 5)
        diagnostics, _, _ = noisy_agent_reference(panel, repetitions=3)
        self.assertEqual(diagnostics["eligible_curves"], 5)

    def test_logprobs_are_not_simulated_as_counts(self):
        for record in self.sources.values():
            path = Path(record["path"])
            data = json.loads(path.read_text())
            data["config"]["scoring_method"] = "logprobs"
            for pref in data["preferences"]:
                pref.update(count_prefer_a=0, count_prefer_b=10, prob_prefer_a=.04)
            path.write_text(json.dumps(data))
            record["sha256"] = digest(path)
        panel = self.panel()
        self.assertEqual(panel.probabilities[0, 0, 0], .04)
        diagnostics, rows, fitted = noisy_agent_reference(panel, repetitions=3)
        self.assertEqual(diagnostics["status"], "excluded_continuous_logprobs")
        self.assertEqual(rows, [])
        self.assertIsNone(fitted)

    def test_adjacent_denominator_bounds_and_category_design_counts(self):
        rows = within_distance_rows(self.summary_record, self.ladders)
        self.assertEqual(rows[0]["observed_pairs"], 23)
        self.assertEqual(rows[0]["expected_pairs"], 24)
        self.assertEqual(rows[0]["full_coverage_accuracy_lower"], 23 / 24)
        self.assertEqual(rows[0]["full_coverage_accuracy_upper"], 1)
        counts = category_rows(self.ladders, references=3)[0]
        self.assertEqual(counts["ladders"], 2)
        self.assertEqual(counts["expected_cross_ladder_curves_per_model"], 6)
        self.assertEqual(counts["expected_adjacent_pairs_per_model"], 24)

    def test_cli_exports_without_changing_sources(self):
        raw_csv = self.root / "raw_sources.csv"
        with raw_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["model_key", "ladder_id", "path", "sha256"])
            writer.writeheader()
            writer.writerows(self.sources.values())
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps([self.summary_record]))
        references = self.root / "references.json"
        references.write_text(json.dumps({"comparison_sample": [
            {"sample_index": key, "text": txt} for key, txt in self.references.items()]}))
        before = {path.name: digest(path) for path in self.root.iterdir() if path.is_file()}
        output = self.root / "outputs"
        args = ["--raw-sources", str(raw_csv), "--source-manifest", str(manifest),
                "--ladders", str(self.ladder_path), "--reference-sample", str(references),
                "--output", str(output), "--permutations", "3", "--simulations", "3", "--skip-plots"]
        self.assertEqual(main(args), 0)
        run = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(run["status"], "complete")
        self.assertTrue(run["provisional"])
        with (output / "curves.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 6)
        with (output / "noisy_agent_scenarios.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 7)
        self.assertEqual(before, {name: digest(self.root / name) for name in before})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)
        self.assertEqual(len(read_sources(raw_csv)["sampled"]), 2)


if __name__ == "__main__":
    unittest.main()
