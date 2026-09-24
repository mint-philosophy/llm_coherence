"""Parser failures, refusal wording, and missing judge labels remain distinct."""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_coherence.analysis.annotate_trace_responses import main
from llm_coherence.analysis.compare_trace_parser import compare_parser
from llm_coherence.analysis.trace_annotation import (
    FIELDS,
    load_bundle,
    prepare,
    read_jsonl,
    template,
    write_jsonl,
)


MIXED = 'I don\'t have a preference. You requested "A" or "B": **A**'


class ParserComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "reasoning_traces.jsonl"
        self.examples = [
            ("A", "I cannot choose between these options.", "A", "accepts"),
            ("B", None, "B", "accepts"),
            (MIXED, None, "A", "accepts"),
            ("I cannot choose between these options.", None, "none", "rejects"),
            ("I cannot choose A.", None, "none", "rejects"),
            ("Answer: B", None, "B", "accepts"),
            ("If forced, I would choose B.", None, "conditional_B", "accepts"),
            (None, "I choose A.", "unobservable", "unobservable"),
            ("I decline this comparison. However I select A.", None, "A", "mixed"),
            ("I choose A. Example nine.", None, "A", "accepts"),
            ("I choose B. Example ten.", None, "B", "accepts"),
            ("I am undecided. Example eleven.", None, "unclear", "unclear"),
        ]
        write_jsonl(
            self.source,
            [{"content": c, "reasoning": r} for c, r, _, _ in self.examples],
        )
        self.bundle = self.root / "bundle"
        self.manifest = prepare(
            [self.source],
            self.bundle,
            pilot_size=4,
            validation_size=4,
            triage_size=4,
            seed=42,
        )
        _, self.entries = load_bundle(self.bundle)
        self.by_text = {
            entry["text"]["final_response"]: entry for entry in self.entries.values()
        }
        self.predictions = self.root / "predictions.jsonl"
        self.write_predictions(self.predictions, "judge-one")

    def write_predictions(self, path, name, omit=()):
        records = []
        for content, reasoning, choice, comparison in self.examples:
            entry = self.by_text[content]
            if entry["entry_id"] in omit:
                continue
            row = template(entry, name)
            row["annotator"]["kind"] = "model"
            for channel, text in entry["text"].items():
                if text is None:
                    row["annotation"][channel] = {
                        field: {"label": "unobservable", "evidence": []}
                        for field in FIELDS
                    }
                else:
                    channel_choice = choice if channel == "final_response" else "none"
                    channel_comparison = (
                        comparison if channel == "final_response" else "rejects"
                    )
                    row["annotation"][channel] = {
                        "expressed_choice": {
                            "label": channel_choice,
                            "evidence": [text]
                            if channel_choice not in ("none", "unclear")
                            else [],
                        },
                        "comparison_response": {
                            "label": channel_comparison,
                            "evidence": [text],
                        },
                        "stated_reason": {"label": "none", "evidence": []},
                    }
            records.append(row)
        write_jsonl(path, records)

    def run_comparison(self, name="report", **kwargs):
        output = self.root / name
        report = compare_parser(
            self.bundle,
            kwargs.pop("predictions", [self.predictions]),
            output,
            split=kwargs.pop("split", "all"),
            parser_mode=kwargs.pop("parser_mode", "forced-choice"),
            **kwargs,
        )
        records = read_jsonl(output / "entries.jsonl")
        return report, {row["text"]["final_response"]: row for row in records}

    def test_actual_parser_uses_only_final_response_and_preserves_input_files(self):
        paths = [self.source, self.predictions, self.bundle / "validation_texts.jsonl"]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with (
            patch(
                "llm_coherence.runtime.agents.create_agent",
                side_effect=AssertionError("No API"),
            ),
            patch(
                "llm_coherence.runtime.api_keys.load_api_key",
                side_effect=AssertionError("No credentials"),
            ),
        ):
            report, records = self.run_comparison()
        self.assertEqual(records["A"]["parser_result"], "A")
        self.assertEqual(
            records["A"]["semantic_labels"]["visible_reasoning.comparison_response"],
            "rejects",
        )
        self.assertEqual(records[None]["parser_result"], "unparseable")
        self.assertEqual(report["parser"]["input_channel"], "final_response")
        self.assertFalse(report["parser"]["historical_trial_outcome_verified"])
        self.assertEqual(
            before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        )

    def test_disclaimer_and_bold_choice_are_review_candidates_without_relabelling(self):
        _, records = self.run_comparison()
        mixed = records[MIXED]
        self.assertEqual(mixed["parser_result"], "unparseable")
        self.assertEqual(
            mixed["semantic_labels"]["final_response.expressed_choice"], "A"
        )
        self.assertIn(
            "parser_unparseable_with_annotated_selection", mixed["review_flags"]
        )
        self.assertIn(
            "annotated_selection_with_lexical_abstention_cue", mixed["review_flags"]
        )
        original = next(
            row
            for row in read_jsonl(self.predictions)
            if row["entry_id"] == mixed["entry_id"]
        )
        self.assertEqual(mixed["annotation"], original["annotation"])

    def test_parser_can_extract_a_mentioned_letter_in_a_refusal(self):
        _, records = self.run_comparison()
        refusal = records["I cannot choose A."]
        self.assertEqual(refusal["parser_result"], "A")
        self.assertIn(
            "parser_selection_without_annotated_selection", refusal["review_flags"]
        )
        self.assertIn(
            "parser_selection_with_comparison_objection", refusal["review_flags"]
        )

    def test_conditional_and_mixed_annotations_are_not_converted_to_votes(self):
        report, records = self.run_comparison()
        conditional = records["If forced, I would choose B."]
        self.assertEqual(
            conditional["semantic_labels"]["final_response.expressed_choice"],
            "conditional_B",
        )
        self.assertIn(
            "parser_selection_with_conditional_annotation", conditional["review_flags"]
        )
        mixed = records["I decline this comparison. However I select A."]
        self.assertIn(
            "annotated_selection_with_comparison_objection", mixed["review_flags"]
        )
        self.assertIsNone(report["judges"][0]["semantic_accuracy"])
        self.assertFalse(report["human_validation_performed"])

    def test_missing_annotation_retains_parseable_entry_and_cross_tab_denominator(self):
        self.write_predictions(
            self.predictions, "judge-one", omit=[self.by_text["B"]["entry_id"]]
        )
        report, records = self.run_comparison()
        missing = records["B"]
        self.assertEqual(missing["parser_result"], "B")
        self.assertIsNone(missing["annotation"])
        self.assertEqual(missing["review_flags"], ["missing_annotation"])
        judge = report["judges"][0]
        self.assertEqual(
            (
                judge["selected_entries"],
                judge["annotations_available"],
                judge["annotations_missing"],
            ),
            (12, 11, 1),
        )
        for table in judge["cross_tabs"].values():
            self.assertEqual(sum(sum(counts.values()) for counts in table.values()), 12)
            self.assertEqual(table["B"]["missing_annotation"], 1)
        with (self.root / "report/comparison.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 12)
        csv_missing = next(
            row for row in csv_rows if row["entry_id"] == missing["entry_id"]
        )
        self.assertEqual(csv_missing["final_response.expressed_choice"], "")

    def test_answer_marker_mode_is_explicit_and_changes_only_parser_replay(self):
        _, ordinary = self.run_comparison("ordinary")
        report, markers = self.run_comparison("markers", parser_mode="answer-marker")
        self.assertEqual(markers["A"]["parser_result"], "unparseable")
        self.assertEqual(markers["Answer: B"]["parser_result"], "B")
        self.assertEqual(markers["A"]["annotation"], ordinary["A"]["annotation"])
        self.assertTrue(report["parser"]["with_reasoning"])
        with self.assertRaisesRegex(ValueError, "parser mode"):
            self.run_comparison("bad-mode", parser_mode="guess")

    def test_development_export_excludes_validation_even_with_all_predictions(self):
        report, records = self.run_comparison(split="development")
        expected = set(
            self.manifest["splits"]["pilot"] + self.manifest["splits"]["triage"]
        )
        self.assertEqual({row["entry_id"] for row in records.values()}, expected)
        self.assertFalse(report["includes_validation_entries"])
        self.assertEqual(set(report["selected_entry_ids"]), expected)
        self.assertEqual(
            report["judges"][0]["predictions_outside_selected_split"],
            12 - len(expected),
        )
        for field in report["judges"][0]["cross_tabs"].values():
            self.assertEqual(
                sum(sum(counts.values()) for counts in field.values()), len(expected)
            )

    def test_multiple_judges_do_not_duplicate_parser_counts(self):
        second = self.root / "second.jsonl"
        self.write_predictions(second, "judge-two")
        report, _ = self.run_comparison(predictions=[self.predictions, second])
        all_rows = read_jsonl(self.root / "report/entries.jsonl")
        self.assertEqual(len(all_rows), 24)
        self.assertEqual(sum(report["parser_counts"].values()), 12)
        self.assertEqual(
            Counter(row["judge"] for row in all_rows),
            {"judge-one": 12, "judge-two": 12},
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.run_comparison(
                "duplicate", predictions=[self.predictions, self.predictions]
            )
        self.assertFalse((self.root / "duplicate").exists())

    def test_tampered_predictions_are_rejected_before_export(self):
        predictions = read_jsonl(self.predictions)
        predictions[0]["entry_sha256"] = "wrong"
        write_jsonl(self.predictions, predictions)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.run_comparison()
        self.assertFalse((self.root / "report").exists())

    def test_output_directory_cannot_overwrite_existing_results(self):
        self.run_comparison()
        marker = self.root / "report/report.json"
        original = marker.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_comparison()
        self.assertEqual(marker.read_bytes(), original)

    def test_cli_requires_mode_and_runs_the_offline_comparison(self):
        args = [
            "compare-parser",
            "--bundle",
            str(self.bundle),
            "--predictions",
            str(self.predictions),
            "--split",
            "development",
            "--output-dir",
            str(self.root / "cli"),
        ]
        with (
            patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            main(args)
        self.assertEqual(error.exception.code, 2)
        with patch("sys.stdout", new=io.StringIO()) as stream:
            self.assertEqual(main(args + ["--parser-mode", "forced-choice"]), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(
            sum(result["parser_counts"].values()), result["selected_entries"]
        )

    def test_failed_export_has_no_completion_marker(self):
        with patch(
            "llm_coherence.analysis.compare_trace_parser.write_jsonl",
            side_effect=OSError("disk failure"),
        ):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.run_comparison()
        self.assertFalse((self.root / "report/report.json").exists())


if __name__ == "__main__":
    unittest.main()
