"""Guard the actual denominators and matched cohorts used in the follow-up plots."""

import tempfile
import unittest
from pathlib import Path
from shutil import copytree

import pandas as pd

from llm_coherence.reporting.followup_figures import (
    export_temperature_details, load_data, load_request_paths, validate_tables,
)


DATA = Path(__file__).resolve().parents[1] / "reports/followup_20261103/data"


class FollowupFigureTests(unittest.TestCase):
    def setUp(self):
        self.ranking, self.temperature, self.curves, self.provenance = load_data(DATA)

    def validate(self):
        validate_tables(self.ranking, self.temperature, self.curves, self.provenance)

    def test_audited_primary_values(self):
        self.assertEqual(int(self.ranking.n_valid_rankings.sum()), 2241)
        ministral = self.ranking.set_index("model_key").loc["ministral-3b-2512-openrouter"]
        self.assertEqual(int(ministral.n_valid_rankings), 86)
        self.assertAlmostEqual(ministral.exact_match_fraction_parsed, 25 / 86)
        passed = (self.temperature.monotonicity_percent * 35 / 100).round().astype(int)
        self.assertEqual(passed.tolist(), [28, 26, 19, 15])

    def test_nonflat_monotonicity(self):
        # flat curves pass strict monotonicity for free and get rarer as temperature rises
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "temperature_details.csv"
            export_temperature_details(self.temperature, self.curves, target)
            rows = pd.read_csv(target)
        self.assertEqual(rows["Flat curves"].tolist(), [9, 6, 5, 5])
        self.assertEqual(rows["Non-flat curves"].tolist(), [26, 29, 30, 30])
        self.assertEqual(rows["Non-flat and strictly monotone"].tolist(), [19, 20, 14, 10])

    def test_request_paths_cover_every_ranking(self):
        paths = load_request_paths(DATA, self.ranking)
        self.assertEqual(int(paths.recovery_runner.sum()), 566)
        for key in ("gpt-56-luna-thinking", "gpt-56-terra-thinking",
                    "gpt-56-sol-thinking", "opus-46-thinking"):
            self.assertEqual(int(paths.loc[key, "recovery_runner"]), 100)
        self.assertEqual(int(paths.loc["gpt-56-luna", "recovery_runner"]), 0)

    def test_rejects_planned_denominator_for_ranking_fraction(self):
        self.ranking.loc[0, "exact_match_fraction_parsed"] = 32 / 100
        with self.assertRaisesRegex(ValueError, "valid rankings"):
            self.validate()

    def test_rejects_duplicate_model(self):
        self.ranking.loc[1, "model_key"] = self.ranking.loc[0, "model_key"]
        with self.assertRaisesRegex(ValueError, "Duplicate ranking"):
            self.validate()

    def test_rejects_including_incomplete_curve(self):
        missing = self.curves.parsed_responses == 139
        self.assertEqual(int(missing.sum()), 1)
        self.curves.loc[missing, "paired_full_response_curve"] = True
        with self.assertRaisesRegex(ValueError, "common-curve mask"):
            self.validate()

    def test_rejects_duplicate_curve(self):
        self.curves.loc[1, ["ladder_id", "comparison_id"]] = self.curves.loc[
            0, ["ladder_id", "comparison_id"]]
        with self.assertRaisesRegex(ValueError, "Duplicate temperature curve"):
            self.validate()

    def test_rejects_changed_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "data"
            copytree(DATA, target)
            with (target / "ranking_summary.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                load_data(target)


if __name__ == "__main__":
    unittest.main()
