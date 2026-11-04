"""Guard the actual denominators and matched cohorts used in the follow-up plots."""

import tempfile
import unittest
from pathlib import Path
from shutil import copytree

from llm_coherence.reporting.followup_figures import load_data, validate_tables


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
