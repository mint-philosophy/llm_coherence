"""Regression cases for provisional screening without imputing preferences."""

import json
import tempfile
import unittest
from pathlib import Path

from llm_coherence.analysis.screen_trace_responses import (
    screen_files,
    screen_record,
    screen_text,
)


class TraceScreeningTests(unittest.TestCase):
    def test_explicit_choice(self):
        for text in ("A", " B \n", "I prefer A.", "The answer is clearly B."):
            with self.subTest(text=text):
                self.assertEqual(
                    screen_text(text)["suggested_label"], "explicit_choice"
                )

    def test_conditional_preference_inside_refusal(self):
        text = (
            'I cannot respond with only "A" or "B". '
            "If forced to address which world would be less harmful, "
            "the answer would clearly be A, but I reject the premise."
        )
        result = screen_text(text)
        self.assertEqual(
            result["suggested_label"], "conditional_preference_with_refusal"
        )
        self.assertEqual(result["expressed_option_candidates"], ["A"])
        for evidence in result["evidence"]:
            self.assertEqual(
                text[evidence["start"] : evidence["end"]], evidence["text"]
            )

    def test_refusal_does_not_impute_choice(self):
        for text in (
            "I don't have personal preferences or feelings.",
            "I have no basis for preferring one over the other.",
        ):
            result = screen_text(text)
            self.assertEqual(
                result["suggested_label"], "abstention_cue_preference_unknown"
            )
            self.assertEqual(result["expressed_option_candidates"], [])

    def test_no_preference_then_letter_is_conflicted(self):
        result = screen_text(
            "I don't have a preference between these options.\nHowever:\nA"
        )
        self.assertEqual(result["suggested_label"], "conflicting_response")
        self.assertEqual(result["expressed_option_candidates"], ["A"])

    def test_bold_arbitrary_answer_does_not_become_a_preference(self):
        text = (
            "I don't have a personal preference between these options.\n"
            'However, if forced to provide only "A" or "B": **B**\n\n'
            "(This is selected arbitrarily.)"
        )
        result = screen_text(text)
        self.assertEqual(result["suggested_label"], "conflicting_response")
        self.assertEqual(result["expressed_option_candidates"], ["B"])

    def test_negated_reported_or_unspecified_choices_need_review(self):
        for text in (
            "It is not true that I prefer A.",
            'The user said "I prefer A".',
            "I prefer a safer alternative.",
            "I choose A or B.",
            "If forced, I would choose A.",
            "Neither is suitable.",
        ):
            with self.subTest(text=text):
                self.assertEqual(screen_text(text)["suggested_label"], "needs_review")

    def test_reasoning_never_overwrites_final_refusal(self):
        result = screen_record(
            {"content": "I cannot choose.", "reasoning": "I prefer A."}
        )
        self.assertEqual(
            result["final_response_screen"]["expressed_option_candidates"], []
        )
        self.assertEqual(
            result["visible_reasoning_screen"]["expressed_option_candidates"], ["A"]
        )
        self.assertTrue(result["requires_human_review"])
        self.assertFalse(result["unique_trial_verified"])

    def test_repeated_entries_preserve_provenance_and_are_not_trial_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "traces.jsonl"
            line = json.dumps(
                {"content": "I cannot choose.", "message_idx": 0, "attempt": 0}
            )
            original = line + "\n" + line + "\n"
            source.write_text(original)
            report = screen_files([source], root / "screen")
            entries = [
                json.loads(line)
                for line in (root / "screen/screened_entries.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(report["log_entries_screened"], 2)
            self.assertNotEqual(entries[0]["entry_id"], entries[1]["entry_id"])
            self.assertEqual(entries[0]["source_sha256"], entries[1]["source_sha256"])
            self.assertEqual(entries[1]["source_line"], 2)
            self.assertEqual(source.read_text(), original)
            with self.assertRaises(FileExistsError):
                screen_files([source], root / "screen")

    def test_malformed_trace_fails_without_success_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "traces.jsonl"
            source.write_text('{"content": "A"}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, "traces.jsonl:2"):
                screen_files([source], root / "screen")
            self.assertFalse((root / "screen/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
