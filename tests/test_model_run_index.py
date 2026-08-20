"""Tests for lightweight model-run index discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_coherence.runtime.model_run_index import model_run_payloads_present


class ModelRunPayloadDetectionTests(unittest.TestCase):
    def test_absent_or_unrelated_output_trees_are_not_model_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reporting_views" / "within_ladder").mkdir(parents=True)

            self.assertFalse(model_run_payloads_present(root))

    def test_incomplete_canonical_run_tree_requires_an_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gpt-56-sol" / "ladder_vs_comparison_statements").mkdir(
                parents=True
            )

            self.assertTrue(model_run_payloads_present(root))


if __name__ == "__main__":
    unittest.main()
