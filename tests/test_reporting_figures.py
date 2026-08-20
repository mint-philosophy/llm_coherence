"""Offline regression tests for publication reporting figures."""

from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from llm_coherence.reporting.make_fig_table import (
    collect_combined_headline_rows,
    discover_reporting_model_keys,
    fig7c_within_ladder_accuracy,
    fig_within_ladder_gap,
    summarize_within_ladder_model,
)


class WithinLadderGapFigureTests(unittest.TestCase):
    def test_skips_missing_monotonicity_and_uses_single_column_layout(self) -> None:
        rows = [
            {
                "file_key": "gpt-56-sol-thinking",
                "mono_pct": 85.6,
                "wl_accuracy_pct": 98.8,
            },
            {
                "file_key": "glm-45-base-logprobs",
                "mono_pct": 10.1,
                "wl_accuracy_pct": 99.0,
                "wl_complete_parseable_coverage": False,
            },
            {
                "file_key": "qwen-37-flash-openrouter-thinking",
                "mono_pct": math.nan,
                "wl_accuracy_pct": 98.1,
            },
        ]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            figure = fig_within_ladder_gap(rows)
        self.addCleanup(plt.close, figure)

        self.assertIsNotNone(figure)
        self.assertEqual(tuple(figure.get_size_inches()), (3.3, 3.2))
        self.assertIn("WARNING: skipping Qwen3.7 Flash (on)", stdout.getvalue())

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            ["Reasoning off", "Reasoning on"],
        )
        self.assertEqual(
            [tick.get_text() for tick in figure.axes[0].get_yticklabels()],
            ["GLM-4.5 Base\u2020"],
        )
        self.assertEqual(
            [tick.get_text() for tick in figure.axes[1].get_yticklabels()],
            ["GPT-5.6 Sol"],
        )
        self.assertEqual(
            [text.get_text() for text in figure.axes[0].texts],
            ["99.0", "10.1"],
        )
        self.assertEqual(
            [text.get_text() for text in figure.axes[1].texts],
            ["98.8", "85.6"],
        )
        self.assertEqual(figure.axes[0].get_xlim(), (0.0, 102.0))
        self.assertEqual(figure.axes[0].get_xlim(), figure.axes[1].get_xlim())
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            [
                "Strict ladder\u2013statement monotonicity",
                "Direct tier\u2013pair accuracy",
            ],
        )
        self.assertIn(
            "\u2020 Accuracy uses parseable responses",
            figure.texts[0].get_text(),
        )

    def test_stacked_panels_have_no_vertical_label_overlap(self) -> None:
        rows = [
            {
                "file_key": f"example-model-{index:02d}{suffix}",
                "mono_pct": 10.0 + index * 5.0 + (2.0 if suffix else 0.0),
                "wl_accuracy_pct": 95.0 + index % 5,
            }
            for index in range(12)
            for suffix in ("", "-thinking")
        ]
        figure = fig_within_ladder_gap(rows)
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()

        renderer = figure.canvas.get_renderer()
        self.assertEqual(len(figure.axes), 2)
        for axis in figure.axes:
            boxes = sorted(
                (
                    tick.get_window_extent(renderer)
                    for tick in axis.get_yticklabels()
                ),
                key=lambda box: box.y0,
            )
            self.assertEqual(len(boxes), 12)
            self.assertTrue(
                all(lower.y1 <= upper.y0 for lower, upper in zip(boxes, boxes[1:]))
            )
            self.assertGreaterEqual(min(box.x0 for box in boxes), 0.0)
            self.assertLessEqual(max(box.x1 for box in boxes), figure.bbox.width)


class WithinLadderAccuracyFigureTests(unittest.TestCase):
    def test_existing_paired_accuracy_figure_is_preserved(self) -> None:
        rows = [
            {
                "file_key": "gpt-56-sol",
                "n_ladders": 30,
                "overall_accuracy_pct": 99.0,
                "accuracy_ci95_pct": 0.3,
            },
            {
                "file_key": "glm-45-base-logprobs",
                "n_ladders": 30,
                "overall_accuracy_pct": 98.0,
                "accuracy_ci95_pct": 0.4,
            },
            {
                "file_key": "gpt-56-sol-thinking",
                "n_ladders": 30,
                "overall_accuracy_pct": 98.8,
                "accuracy_ci95_pct": 0.3,
            },
            {
                "file_key": "gpt-56-luna-thinking",
                "n_ladders": 30,
                "overall_accuracy_pct": 98.9,
                "accuracy_ci95_pct": 0.3,
            },
        ]

        figure = fig7c_within_ladder_accuracy(rows)
        self.addCleanup(plt.close, figure)

        self.assertIsNotNone(figure)
        self.assertEqual(tuple(figure.get_size_inches()), (10.0, 5.0))
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(
            [text.get_text() for text in figure.axes[0].get_legend().get_texts()],
            ["reasoning off", "reasoning on"],
        )


class WithinLadderReportingDataTests(unittest.TestCase):
    def test_combined_rows_preserve_accuracy_coverage(self) -> None:
        rows, _macro = collect_combined_headline_rows(
            ["gpt-56-sol"],
            [],
            [{
                "file_key": "gpt-56-sol",
                "n_ladders": 100,
                "overall_accuracy_pct": 95.0,
                "perfect_ladders": 80,
                "n_trials": 3990,
                "n_trials_expected": 4200,
                "n_trials_missing": 210,
                "complete_parseable_coverage": False,
                "accuracy_lower_bound_pct": 90.25,
                "accuracy_upper_bound_pct": 95.25,
            }],
        )

        self.assertEqual(rows[0]["wl_n_trials"], 3990)
        self.assertEqual(rows[0]["wl_n_trials_expected"], 4200)
        self.assertEqual(rows[0]["wl_n_trials_missing"], 210)
        self.assertFalse(rows[0]["wl_complete_parseable_coverage"])
        self.assertEqual(rows[0]["wl_accuracy_lower_bound_pct"], 90.25)
        self.assertEqual(rows[0]["wl_accuracy_upper_bound_pct"], 95.25)

    def test_micro_accuracy_ci_and_coverage_match_reported_estimator(self) -> None:
        row = summarize_within_ladder_model(
            "gpt-56-sol",
            {
                "overall_accuracy": 0.1,
                "n_ladders": 2,
                "n_total_pairs": 10,
                "n_total_pairs_expected": 20,
                "n_requests_missing_from_scoring": 10,
                "complete_parseable_coverage": False,
                "overall_accuracy_bounds": {
                    "lower_missing_incorrect": 0.05,
                    "upper_missing_correct": 0.55,
                },
                "per_ladder": [
                    {"ladder_id": "a_1", "accuracy": 1.0, "n": 1, "valence": "positive"},
                    {"ladder_id": "b_2", "accuracy": 0.0, "n": 9, "valence": "negative"},
                ],
            },
        )

        self.assertEqual(row["overall_accuracy_pct"], 10.0)
        self.assertAlmostEqual(row["accuracy_ci95_pct"], 90.0)
        self.assertEqual(row["n_trials_expected"], 20)
        self.assertEqual(row["n_trials_missing"], 10)
        self.assertFalse(row["complete_parseable_coverage"])
        self.assertEqual(row["accuracy_lower_bound_pct"], 5.0)
        self.assertAlmostEqual(row["accuracy_upper_bound_pct"], 55.0)

    def test_model_discovery_unions_separate_reporting_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results_dir = root / "coherence"
            within_dir = root / "within"
            results_dir.mkdir()
            summary = (
                within_dir
                / "gpt-56-sol-thinking"
                / "within_ladder"
                / "summary.json"
            )
            summary.parent.mkdir(parents=True)
            summary.write_text("{}", encoding="utf-8")

            self.assertIn(
                "gpt-56-sol-thinking",
                discover_reporting_model_keys(results_dir, within_dir),
            )


if __name__ == "__main__":
    unittest.main()
