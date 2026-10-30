"""Check response provenance, order orientation, and held-out fitting."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import binom

from llm_coherence.analysis.noise_response_inputs import (
    ABSENT, CONFLICT, keyed_sidecar, legacy_batch_slots, merge_trial,
    modern_batch_slots, parse_labels, prompt_matches, reconcile,
)
from llm_coherence.analysis.rebuttal_inputs import digest
from llm_coherence.analysis.validate_noisy_agent import (
    calibration_bins, cross_validate_curves, prediction_error,
    trial_diagnostics, unanimity_probabilities,
)
from llm_coherence.analysis.noise_validation_plots import opposite_unanimity_bound


class NoiseMathTests(unittest.TestCase):
    def test_flip_orientation_and_no_imputed_choices(self):
        np.testing.assert_array_equal(parse_labels(["A", "B", None], False, 0), [1, 0, -1])
        np.testing.assert_array_equal(parse_labels(["A", "B", None], False, 1), [0, 1, -1])
        self.assertFalse(reconcile(np.array([[1, 1], [0, ABSENT]]),
                                   {"count_prefer_a": 2, "count_prefer_b": 2}))

    def test_order_effect_is_not_mistaken_for_outcome_preference(self):
        # Always answers displayed A: same choice within an order, opposite
        # underlying outcomes across orders. The pooled preference is 0.5.
        labels = np.tile(np.array([[1, 1], [0, 0]]), (1, 1, 7, 1, 1))
        stats, expected, joint = trial_diagnostics(labels, np.full((1, 1, 7), .5),
                                                    np.ones((1, 1), bool), np.ones((1, 1, 7), int))
        self.assertEqual(stats["mean_order_difference"], 1)
        self.assertEqual(stats["observed_unanimous_group_fraction"], 1)
        self.assertEqual(stats["opposite_unanimous_orders_fraction"], 1)
        self.assertAlmostEqual(stats["conditional_random_split_expected_order_mse"], 1 / 3)
        self.assertAlmostEqual(stats["conditional_random_split_expected_unanimity"], 1 / 3)
        self.assertEqual(sum(r["cells"] for r in joint), 7)
        self.assertTrue(all(r["expected_mean_order_difference"] == 0 for r in expected))

    def test_exact_binomial_expectations_against_enumeration(self):
        p, n = .3, 3
        zero, full = unanimity_probabilities(np.array([p]), n, 0)
        self.assertAlmostEqual(zero[0] + full[0], binom.pmf(0, n, p) + binom.pmf(n, n, p))
        expected_mse = sum(((a - b) / n) ** 2 * binom.pmf(a, n, p) * binom.pmf(b, n, p)
                           for a, b in itertools.product(range(n + 1), repeat=2))
        self.assertAlmostEqual(expected_mse, 2 * p * (1 - p) / n)
        for rho in (0, .5, 1):
            zero, full = unanimity_probabilities(np.array([0., .5, 1.]), 10, rho)
            self.assertEqual(zero[0], 1)
            self.assertEqual(full[-1], 1)
        zero, full = unanimity_probabilities(np.array([.3]), 10, 1)
        self.assertEqual(zero[0] + full[0], 1)

    def test_opposite_unanimity_bound_is_attained_at_half_probability(self):
        for n in (1, 2, 10):
            grid = np.linspace(0, 1, 1001)
            probabilities = 2 * (grid * (1 - grid)) ** n
            self.assertAlmostEqual(probabilities.max(), opposite_unanimity_bound(n))
        self.assertAlmostEqual(opposite_unanimity_bound(10), 1 / 524288)

    def test_missing_or_unverified_cells_never_enter_trial_statistics(self):
        labels = np.ones((1, 1, 7, 2, 2), dtype=np.int8)
        labels[0, 0, 0, 0, 0] = -1
        codes = np.ones((1, 1, 7), int)
        codes[0, 0, 1] = 0
        stats, _, _ = trial_diagnostics(labels, np.full((1, 1, 7), .8), np.ones((1, 1), bool), codes)
        self.assertEqual(stats["cells"], 5)
        self.assertAlmostEqual(stats["coverage_fraction"], 5 / 7)

    def test_heldout_curve_answers_do_not_enter_their_fitted_predictions(self):
        rng = np.random.default_rng(43)
        totals = np.full((3, 6, 7), 20)
        successes = rng.binomial(totals, np.linspace(.1, .9, 7)[None, None, :])
        eligible = np.ones((3, 6), bool)
        ids = np.array([[f"{li}:{ti}" for ti in range(7)] for li in range(3)])
        first, assignment, diagnostics = cross_validate_curves(successes, totals, eligible, ids)
        self.assertTrue(all(row["status"] == "complete" for row in diagnostics))
        changed = successes.copy()
        changed[assignment == 0] = totals[assignment == 0] - changed[assignment == 0]
        second, _, _ = cross_validate_curves(changed, totals, eligible, ids)
        np.testing.assert_allclose(first[assignment == 0], second[assignment == 0], atol=0, rtol=0)
        self.assertEqual(prediction_error(first, successes, totals, eligible)["curves"], 18)
        self.assertEqual(sum(row["cells"] for row in calibration_bins(first, successes, totals, eligible, "heldout")), 126)


class TrialProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pref = {"outcome_a": {"text": "tier outcome"}, "outcome_b": {"text": "reference outcome"},
                     "count_prefer_a": 2, "count_prefer_b": 2}
        self.raw = {"config": {"test_name": "ladder", "model_key": "model", "num_trials": 2,
                               "with_reasoning": False}, "metadata": {"model_name_full": "model"},
                    "preferences": [self.pref]}

    def write_jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_unkeyed_rows_are_not_trials_and_duplicate_choices_do_not_multiply(self):
        row = {"custom_id": "unique", "pair_idx": 0, "trial": 0, "direction": "A", "content": "A"}
        path = self.write_jsonl("traces.jsonl", [row, row, {"message_idx": 1, "content": "A"}])
        slots, audit = keyed_sidecar(path, self.raw, {})
        self.assertEqual((slots >= 0).sum(), 1)
        self.assertEqual(audit["unkeyed_rows"], 1)
        self.assertFalse(reconcile(slots[0], self.pref))

    def test_conflicting_duplicate_trials_are_ambiguous(self):
        row = {"custom_id": "unique", "pair_idx": 0, "trial": 0, "direction": "A", "content": "A"}
        path = self.write_jsonl("traces.jsonl", [row, {**row, "content": "B"}])
        slots, _ = keyed_sidecar(path, self.raw, {})
        self.assertEqual(slots[0, 0, 0], CONFLICT)
        merge_trial(slots, (0, 0, 0), 1)
        self.assertEqual(slots[0, 0, 0], CONFLICT)

    def test_one_custom_id_cannot_represent_different_trials(self):
        row = {"custom_id": "unique", "pair_idx": 0, "trial": 0, "direction": "A", "content": "A"}
        path = self.write_jsonl("traces.jsonl", [row, {**row, "trial": 1}])
        with self.assertRaisesRegex(ValueError, "multiple trials"):
            keyed_sidecar(path, self.raw, {})

    def test_prompt_verification_uses_display_order(self):
        body = {"input": "Choose:\n\nOption A:\ntier outcome\n\nOption B:\nreference outcome\n\nPlease answer."}
        self.assertTrue(prompt_matches(body, self.pref, 0))
        self.assertFalse(prompt_matches(body, self.pref, 1))
        body["input"] = body["input"].replace("tier outcome", "old wording")
        self.assertFalse(prompt_matches(body, self.pref, 0))

    def test_modern_batch_uses_stable_ids_and_checks_request_hashes(self):
        comparisons = self.root / "comparisons.json"
        comparisons.write_text(json.dumps({"comparisons": [{"outcome_a": self.pref["outcome_a"],
                                                            "outcome_b": self.pref["outcome_b"]}]}))
        requests = self.write_jsonl("input.jsonl", [{"custom_id": "s0000-c0000-dab-t000"}])
        self.raw["metadata"].update(batch_run_id=self.root.name, comparison_file_sha256=digest(comparisons))
        (self.root / "batch_manifest.json").write_text(json.dumps({
            "model_key": "model", "num_trials": 2, "include_flipped": True,
            "sets": [{"test_name": "ladder", "comparison_path": "comparisons.json", "comparison_file_sha256": digest(comparisons)}],
            "shards": [{"input_file": "input.jsonl", "sha256": digest(requests)}]}))
        (self.root / "batch_jobs.json").write_text(json.dumps({"jobs": [{"output_file": "output.jsonl"}]}))
        rows = [{"custom_id": f"s0000-c0000-d{direction}-t{trial:03d}",
                 "response": {"status_code": 200, "body": {"status": "completed", "output": [
                     {"type": "message", "content": [{"type": "output_text", "text": "A"}]}]}}}
                for direction in ("ab", "ba") for trial in range(2)]
        self.write_jsonl("output.jsonl", list(reversed(rows)))
        slots = modern_batch_slots(self.root, {0: self.raw}, {})[0]
        np.testing.assert_array_equal(slots[0], [[1, 1], [0, 0]])
        self.assertTrue(reconcile(slots[0], self.pref))
        requests.write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            modern_batch_slots(self.root, {0: self.raw}, {})

    def test_legacy_internal_reasoning_can_still_request_bare_letters(self):
        self.raw["config"]["with_reasoning"] = True
        requests, responses = [], []
        for direction in ("A", "B"):
            texts = ["tier outcome", "reference outcome"]
            if direction == "B":
                texts.reverse()
            for trial in range(2):
                cid = f"ladder__0__trial{trial}__dir{direction}"
                requests.append({"custom_id": cid, "body": {"model": "model", "messages": [
                    {"role": "user", "content": f"Option A:\n{texts[0]}\n\nOption B:\n{texts[1]}\n\nPlease respond with only A or B."}]}})
                responses.append({"custom_id": cid, "response": {"status_code": 200, "body": {
                    "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]}}})
        self.write_jsonl("batch_smoke_chunk01.jsonl", requests)
        self.write_jsonl("batch_output_chunk01.jsonl", responses)
        slots = legacy_batch_slots(self.root, {0: self.raw}, {})[0]
        self.assertTrue(reconcile(slots[0], self.pref))


if __name__ == "__main__":
    unittest.main()
