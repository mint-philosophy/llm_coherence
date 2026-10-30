"""Legacy batch boundaries must be proven before response identities are used."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from llm_coherence.analysis.noise_response_inputs import (
    ABSENT, LEGACY_SEQUENTIAL_COMMITS, legacy_sequential_sidecar, recover_trials,
)
from llm_coherence.analysis.rebuttal_inputs import digest


class LegacySequentialRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = {
            "config": {"test_name": "ladder", "num_trials": 10,
                       "include_flipped": True, "with_reasoning": False},
            "metadata": {"git_commit_sha": "e423488dd19af0a30d20ed3395319d168c15eca5"},
            "preferences": [],
        }
        self.rows = []
        self.expected = np.zeros((7, 2, 10), dtype=np.int8)
        for pi in range(7):
            for direction in range(2):
                self.expected[pi, direction, :pi + 1 + direction] = 1
                # Successes complete in arbitrary order, with some transport retries.
                for trial in [3, 9, 0, 8, 2, 4, 7, 1, 6, 5]:
                    chosen = self.expected[pi, direction, trial]
                    letter = "A" if chosen != direction else "B"
                    self.rows.append({"message_idx": trial, "attempt": trial % 3,
                                      "content": letter, "reasoning": None})
            a = int(self.expected[pi].sum())
            self.raw["preferences"].append({
                "outcome_a": {"text": f"tier {pi}", "tier": pi + 1},
                "outcome_b": {"text": "reference", "comparison_id": 1},
                "count_prefer_a": a, "count_prefer_b": 20 - a,
            })

    def write_rows(self, rows):
        path = self.root / "reasoning_traces.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def recover(self, rows=None, raw=None):
        return legacy_sequential_sidecar(self.write_rows(self.rows if rows is None else rows),
                                          self.raw if raw is None else raw, {})

    def assert_rejected(self, reason, *, rows=None, raw=None):
        slots, audit = self.recover(rows, raw)
        self.assertTrue(np.all(slots == ABSENT))
        self.assertEqual(audit, {"legacy_sequential_rejected_" + reason: 1})

    def test_restores_trial_identity_and_display_order_for_all_reviewed_versions(self):
        path = self.write_rows(self.rows)
        for commit in LEGACY_SEQUENTIAL_COMMITS:
            with self.subTest(commit=commit):
                raw = deepcopy(self.raw)
                raw["metadata"]["git_commit_sha"] = commit
                provenance = {}
                slots, audit = legacy_sequential_sidecar(path, raw, provenance)
                np.testing.assert_array_equal(slots, self.expected)
                self.assertEqual(audit["legacy_sequential_recovered_rows"], 140)
                self.assertEqual(provenance[str(path.resolve())], digest(path))

    def test_unknown_or_missing_producer_version_is_not_inferred_from_counts(self):
        for metadata in ({}, {"git_commit_sha": "unknown"}):
            raw = deepcopy(self.raw)
            raw["metadata"] = metadata
            self.assert_rejected("unreviewed_commit", raw=raw)

    def test_missing_and_duplicate_records_reject_entire_file(self):
        self.assert_rejected("incomplete_or_extra_rows", rows=self.rows[:36] + self.rows[37:])
        self.assert_rejected("incomplete_or_extra_rows", rows=self.rows + [self.rows[-1]])

    def test_duplicate_index_even_with_correct_length_and_counts_is_rejected(self):
        rows = deepcopy(self.rows)
        rows[-1]["message_idx"] = rows[-2]["message_idx"]
        self.assert_rejected("invalid_block_indices", rows=rows)

    def test_out_of_range_and_boolean_indices_are_rejected(self):
        for index in (-1, 10, True):
            rows = deepcopy(self.rows)
            rows[0]["message_idx"] = index
            self.assert_rejected("invalid_block_indices", rows=rows)

    def test_late_count_mismatch_does_not_expose_a_plausible_prefix(self):
        raw = deepcopy(self.raw)
        raw["preferences"][-1]["count_prefer_a"] += 1
        self.assert_rejected("saved_count_mismatch", raw=raw)

    def test_keyed_rows_cannot_be_reinterpreted_as_sequential(self):
        rows = deepcopy(self.rows)
        rows[0]["custom_id"] = "explicit-id"
        self.assert_rejected("mixed_schema", rows=rows)

    def test_failed_or_malformed_success_records_are_rejected(self):
        for update in ({"content": None}, {"content": ""}, {"attempt": -1}, {"attempt": True}):
            rows = deepcopy(self.rows)
            rows[0].update(update)
            self.assert_rejected("invalid_success_record", rows=rows)

    def test_unsupported_protocol_is_rejected(self):
        for update in ({"include_flipped": False}, {"num_trials": 2}):
            raw = deepcopy(self.raw)
            raw["config"].update(update)
            self.assert_rejected("unsupported_protocol", raw=raw)

    def test_recorded_unparseable_answer_is_preserved_not_imputed(self):
        rows, raw = deepcopy(self.rows), deepcopy(self.raw)
        # First record is trial 3, originally a reference choice in original order.
        rows[0]["content"] = "neither"
        raw["preferences"][0]["count_prefer_b"] -= 1
        slots, audit = self.recover(rows, raw)
        self.assertEqual(slots[0, 0, 3], -1)
        self.assertEqual((slots < 0).sum(), 1)
        self.assertEqual(audit["legacy_sequential_complete_logs"], 1)

    def setup_pipeline(self, raw=None):
        raw = self.raw if raw is None else raw
        result = self.root / "results.json"
        result.write_text(json.dumps(raw))
        self.write_rows(self.rows)
        panel = SimpleNamespace(model_key="fixture", trials_per_direction=10,
                                probabilities=np.zeros((1, 1, 7)),
                                reference_ids=[1], ladder_ids=["ladder"])
        return panel, {"ladder": {"path": str(result), "sha256": digest(result)}}

    def test_pipeline_records_source_code_and_hash_provenance(self):
        panel, sources = self.setup_pipeline()
        labels, codes, audit = recover_trials(panel, sources)
        np.testing.assert_array_equal(labels[0, 0], self.expected)
        np.testing.assert_array_equal(codes, np.full((1, 1, 7), 6))
        self.assertEqual(audit["coverage"]["cells_source_6"], 7)
        self.assertEqual(len(audit["source_files"]), 2)
        self.assertIn(6, audit["source_codes"])

    def test_pipeline_rejects_same_counts_but_conflicting_order_specific_answers(self):
        raw = deepcopy(self.raw)
        pref = raw["preferences"][0]
        pref["raw_responses_original"] = ["A"] + ["B"] * 9
        pref["raw_responses_flipped"] = ["B"] * 2 + ["A"] * 8
        # Swap two trials without changing the original-order success count.
        pref["raw_responses_original"][0:2] = ["B", "A"]
        panel, sources = self.setup_pipeline(raw)
        with self.assertRaisesRegex(ValueError, "conflicts with verified response identity"):
            recover_trials(panel, sources)


if __name__ == "__main__":
    unittest.main()
