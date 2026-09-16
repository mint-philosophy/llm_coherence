"""Tests for refusal robustness and visible-rationale provenance analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_coherence.analysis.refusal_robustness import (
    _normalize_results_dir,
    analyze_results,
    audit_reasoning_traces,
    summarize_annotations,
    write_annotation_template,
)


def _preference(tier: int, count_a: int, count_b: int, *, linked: bool = False) -> dict:
    preference = {
        "outcome_a": {
            "text": f"Ladder tier {tier}",
            "variation_id": "Test category_1",
            "tier": tier,
            "tier_label": f"tier_{tier}",
            "category": "Test category",
        },
        "outcome_b": {
            "text": "Fixed comparison",
            "comparison_id": 3,
            "comparison_category": "Comparison category",
        },
        "count_prefer_a": count_a,
        "count_prefer_b": count_b,
        "prob_prefer_a": count_a / (count_a + count_b),
        "prob_prefer_b": count_b / (count_a + count_b),
    }
    if count_a + count_b < 4:
        preference["expected_trials"] = 4
        preference["parseable_trials"] = count_a + count_b
        preference["missing_trials"] = 4 - count_a - count_b
        preference["missing_by_reason"] = {"unparseable": 4 - count_a - count_b}
        if linked:
            preference["missing_responses"] = [
                {
                    "custom_id": f"c0002-dab-t{index:03d}",
                    "direction": "AB",
                    "trial_index": index,
                    "reason": "unparseable",
                    "raw_response": "I cannot choose.",
                }
                for index in range(4 - count_a - count_b)
            ]
    return preference


def _write_result(root: Path, *, linked: bool = False) -> Path:
    artifact = root / "phase6b_ladder_Test_category_1"
    artifact.mkdir(parents=True)
    preferences = [
        _preference(1, 0, 4),
        _preference(2, 1, 3),
        _preference(3, 1, 1, linked=linked),
        _preference(4, 2, 2),
        _preference(5, 3, 1),
        _preference(6, 4, 0),
        _preference(7, 4, 0),
    ]
    payload = {
        "schema_version": "1.0",
        "config": {
            "model_key": "test-model",
            "num_trials": 2,
            "include_flipped": True,
            "is_base_model": False,
        },
        "metadata": {"unparseable_count": 2},
        "preferences": preferences,
    }
    (artifact / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return artifact


class RefusalRobustnessTests(unittest.TestCase):
    def test_behavioral_audit_reconciles_counts_and_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root)
            (artifact / "reasoning_traces.jsonl").write_text(
                json.dumps({"message_idx": 0, "attempt": 0, "content": "refusal"})
                + "\n",
                encoding="utf-8",
            )

            report, missing_ids = analyze_results(root, "test-model")
            trace_report, _linked = audit_reasoning_traces(root, missing_ids)

            self.assertEqual(report["coverage"]["expected_trials"], 28)
            self.assertEqual(report["coverage"]["parseable_trials"], 26)
            self.assertEqual(report["coverage"]["missing_trials"], 2)
            self.assertTrue(report["coverage"]["counts_reconcile"])
            self.assertTrue(report["artifact_integrity"]["passed"])
            self.assertEqual(
                report["artifact_integrity"]["probability_fields_checked"], 7
            )
            self.assertEqual(report["coverage"]["affected_cells"], 1)
            self.assertEqual(
                report["affected_cells"][0]["winner_robust_to_missing"], "uncertain"
            )
            self.assertEqual(
                report["monotonicity_sensitivity"]["conditional_on_parseable"]["rate"],
                1.0,
            )
            self.assertEqual(
                report["monotonicity_sensitivity"]["all_missing_prefer_a"]["rate"],
                0.0,
            )
            self.assertEqual(
                report["monotonicity_sensitivity"]["formal_assignment_bounds"],
                {
                    "minimum_monotonic_groups": 0,
                    "minimum_rate": 0.0,
                    "maximum_monotonic_groups": 1,
                    "maximum_rate": 1.0,
                    "interpretation": (
                        "Exact range over every integer allocation of missing A/B "
                        "votes, computed independently within each seven-tier group."
                    ),
                },
            )
            self.assertFalse(trace_report["quantitative_trial_level_analysis_allowed"])
            self.assertIn(
                "missing responses do not expose stable custom_id values",
                trace_report["blocking_reasons"],
            )

    def test_linked_traces_generate_annotation_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root, linked=True)
            trace_rows = [
                {
                    "custom_id": f"c0002-dab-t{index:03d}",
                    "attempt": 0,
                    "content": "I cannot choose.",
                    "reasoning": "Both outcomes have merit.",
                }
                for index in range(2)
            ]
            (artifact / "reasoning_traces.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in trace_rows),
                encoding="utf-8",
            )

            _report, missing_ids = analyze_results(root, "test-model")
            trace_report, linked = audit_reasoning_traces(root, missing_ids)
            template = root / "annotations.jsonl"
            write_annotation_template(template, linked)

            self.assertTrue(trace_report["quantitative_trial_level_analysis_allowed"])
            self.assertEqual(trace_report["linked_missing_trial_ids"], 2)
            self.assertEqual(len(template.read_text(encoding="utf-8").splitlines()), 2)
            template_row = json.loads(
                template.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(template_row["direction"], "AB")
            self.assertEqual(template_row["canonical_outcome_a_text"], "Ladder tier 3")
            self.assertEqual(template_row["prompt_option_b_text"], "Fixed comparison")

    def test_batch_reasoning_summaries_are_linked_and_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root, linked=True)
            trace_rows = [
                {
                    "custom_id": f"c0002-dab-t{index:03d}",
                    "content": "I cannot choose.",
                    "summary": "The outcomes appear incomparable.",
                }
                for index in range(2)
            ]
            (artifact / "reasoning_summaries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in trace_rows),
                encoding="utf-8",
            )

            report, missing_ids = analyze_results(root, "test-model")
            trace_report, linked = audit_reasoning_traces(
                root,
                missing_ids,
                report["coverage"]["missing_records_without_custom_id"],
            )
            template = root / "annotations.jsonl"
            write_annotation_template(template, linked)
            template_rows = [
                json.loads(line)
                for line in template.read_text(encoding="utf-8").splitlines()
            ]

            self.assertTrue(trace_report["quantitative_trial_level_analysis_allowed"])
            self.assertEqual(
                template_rows[0]["reasoning"], "The outcomes appear incomparable."
            )

    def test_mismatched_trace_content_blocks_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root, linked=True)
            trace_rows = [
                {
                    "custom_id": f"c0002-dab-t{index:03d}",
                    "content": "A stale response",
                    "reasoning": "A visible rationale",
                }
                for index in range(2)
            ]
            (artifact / "reasoning_traces.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in trace_rows),
                encoding="utf-8",
            )

            report, missing_trials = analyze_results(root, "test-model")
            trace_report, _linked = audit_reasoning_traces(
                root,
                missing_trials,
                report["coverage"]["missing_records_without_custom_id"],
            )

            self.assertFalse(trace_report["quantitative_trial_level_analysis_allowed"])
            self.assertEqual(trace_report["linked_response_content_mismatches"], 2)

    def test_partial_missing_ids_block_trace_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root, linked=True)
            result_path = artifact / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["preferences"][2]["missing_responses"][1]["custom_id"] = None
            result_path.write_text(json.dumps(result), encoding="utf-8")
            (artifact / "reasoning_traces.jsonl").write_text(
                json.dumps(
                    {
                        "custom_id": "c0002-dab-t000",
                        "attempt": 0,
                        "content": "I cannot choose.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report, missing_ids = analyze_results(root, "test-model")
            trace_report, _linked = audit_reasoning_traces(
                root,
                missing_ids,
                report["coverage"]["missing_records_without_custom_id"],
            )

            self.assertFalse(trace_report["quantitative_trial_level_analysis_allowed"])
            self.assertEqual(trace_report["missing_trials_without_custom_id"], 1)

    def test_absent_response_text_blocks_trace_linkage(self) -> None:
        for missing_field in ("raw_response", "content"):
            with (
                self.subTest(missing_field=missing_field),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                artifact = _write_result(root, linked=True)
                _report, missing_trials = analyze_results(root, "test-model")
                trace_rows = []
                for trial_key, context in missing_trials.items():
                    trace = {
                        "custom_id": trial_key.split(":", 1)[1],
                        "content": "I cannot choose.",
                        "reasoning": "Both outcomes have merit.",
                    }
                    if missing_field == "raw_response":
                        context.pop("raw_response")
                    else:
                        trace.pop("content")
                    trace_rows.append(trace)
                (artifact / "reasoning_traces.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in trace_rows),
                    encoding="utf-8",
                )
                report, _linked = audit_reasoning_traces(root, missing_trials)
                self.assertFalse(report["quantitative_trial_level_analysis_allowed"])
                self.assertEqual(
                    report["linked_trials_without_verifiable_response_content"], 2
                )

    def test_model_root_path_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary) / "model"
            expected = model_root / "ladder_vs_comparison_statements"
            expected.mkdir(parents=True)

            self.assertEqual(
                _normalize_results_dir(model_root, "test-model"), expected.resolve()
            )

    def test_audit_rejects_invalid_missing_reason_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root, linked=True)
            result_path = artifact / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["preferences"][2]["missing_by_reason"] = {
                "unparseable": 3,
                "other": -1,
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid missing_by_reason"):
                analyze_results(root, "test-model")

    def test_audit_rejects_shuffled_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root)
            result_path = artifact / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["preferences"][0], result["preferences"][1] = (
                result["preferences"][1],
                result["preferences"][0],
            )
            result_path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid tier sequence"):
                analyze_results(root, "test-model")

    def test_zero_parseable_cell_is_excluded_from_conditional_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _write_result(root)
            result_path = artifact / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            cell = result["preferences"][2]
            cell.update(
                count_prefer_a=0,
                count_prefer_b=0,
                prob_prefer_a=None,
                prob_prefer_b=None,
                expected_trials=4,
                parseable_trials=0,
                missing_trials=4,
                missing_by_reason={"unparseable": 4},
            )
            result["metadata"]["unparseable_count"] = 4
            result_path.write_text(json.dumps(result), encoding="utf-8")

            report, _missing = analyze_results(root, "test-model")
            conditional = report["monotonicity_sensitivity"]["conditional_on_parseable"]

            self.assertEqual(conditional["groups_with_complete_parseable_estimates"], 0)
            self.assertEqual(
                conditional["groups_unavailable_due_to_zero_parseable_cell"], 1
            )
            self.assertIsNone(conditional["rate"])

    def test_annotation_summary_reports_transitions_and_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coded.jsonl"
            rows = []
            for coder in ("coder_1", "coder_2"):
                rows.extend(
                    [
                        {
                            "trial_key": "ladder:c1",
                            "coder": coder,
                            "reasoning_conclusion": "favors_a",
                            "refusal_reason": "political_or_religious_neutrality",
                            "final_response": "explicit_refusal",
                            "relationship": "reasoning_favors_a_or_b_but_final_refuses",
                        },
                        {
                            "trial_key": "ladder:c2",
                            "coder": coder,
                            "reasoning_conclusion": "favors_b",
                            "refusal_reason": "none",
                            "final_response": "B",
                            "relationship": "reasoning_and_answer_agree",
                        },
                    ]
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = summarize_annotations(path)

            self.assertEqual(summary["annotation_rows"], 4)
            self.assertEqual(summary["unique_trials"], 2)
            self.assertEqual(summary["consensus_trials"], 2)
            self.assertEqual(summary["trials_requiring_adjudication"], [])
            self.assertEqual(
                summary["reasoning_to_final_transitions"]["favors_a->explicit_refusal"],
                1,
            )
            self.assertEqual(
                summary["inter_rater_agreement"]["reasoning_conclusion"]["cohen_kappa"],
                1.0,
            )

    def test_annotation_disagreement_requires_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coded.jsonl"
            common = {
                "trial_key": "ladder:c1",
                "refusal_reason": "political_or_religious_neutrality",
                "final_response": "explicit_refusal",
                "relationship": "reasoning_favors_a_or_b_but_final_refuses",
            }
            rows = [
                {**common, "coder": "coder_1", "reasoning_conclusion": "favors_a"},
                {
                    **common,
                    "coder": "coder_2",
                    "reasoning_conclusion": "unclear",
                    "relationship": "unclear",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = summarize_annotations(path)

            self.assertEqual(summary["consensus_trials"], 0)
            self.assertEqual(summary["trials_requiring_adjudication"], ["ladder:c1"])
            self.assertEqual(summary["reasoning_to_final_transitions"], {})

            with self.assertRaisesRegex(ValueError, "trial coverage does not match"):
                summarize_annotations(path, expected_trial_keys={"ladder:other"})

    def test_annotation_requires_two_fixed_independent_coders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coded.jsonl"
            row = {
                "trial_key": "ladder:c1",
                "coder": "coder_1",
                "reasoning_conclusion": "favors_a",
                "refusal_reason": "none",
                "final_response": "A",
                "relationship": "reasoning_and_answer_agree",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly two independent coders"):
                summarize_annotations(path)

    def test_adjudication_is_used_only_for_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coded.jsonl"
            common = {
                "trial_key": "ladder:c1",
                "reasoning_conclusion": "favors_a",
                "refusal_reason": "none",
                "final_response": "A",
                "relationship": "reasoning_and_answer_agree",
            }
            rows = [
                {**common, "coder": "coder_1"},
                {**common, "coder": "coder_2"},
                {**common, "coder": "coder_3", "adjudicated": True},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "despite coder agreement"):
                summarize_annotations(path)


if __name__ == "__main__":
    unittest.main()
