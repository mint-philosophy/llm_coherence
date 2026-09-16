"""Tests for refusal robustness and visible-rationale provenance analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_coherence.analysis.refusal_robustness import (
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
                    "reason": "unparseable",
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
            self.assertEqual(
                summary["reasoning_to_final_transitions"]["favors_a->explicit_refusal"],
                2,
            )
            self.assertEqual(
                summary["inter_rater_agreement"]["reasoning_conclusion"]["cohen_kappa"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
