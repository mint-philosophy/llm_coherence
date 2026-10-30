"""Keep published monotonicity tied to the predefined increasing tier order."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from llm_coherence.paths import phase6b_coherence_json_path
from llm_coherence.reporting import make_fig_table as reporting


class ReportingMonotonicityTests(unittest.TestCase):
    def test_tables_categories_and_scatter_agree_on_fixed_direction(self) -> None:
        # A ladder whose responses mostly decrease must not earn credit for
        # reversing the predefined T1 -> T7 ordering. Constant curves pass.
        curves = {
            "reversed": [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                [0.1, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8],
            ],
            "increasing": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
        }
        records = []
        fixed_passes = []
        for category, ladder_curves in curves.items():
            steps = np.diff(ladder_curves, axis=1)
            increasing = np.all(steps >= 0, axis=1)
            decreasing = np.all(steps <= 0, axis=1)
            fixed_passes.extend(increasing)
            records.append({
                "category": category,
                "monotonicity_rate": float(increasing.mean()),
                "monotonicity_rate_inferred_direction": float(
                    decreasing.mean() if category == "reversed" else increasing.mean()
                ),
            })

        model_key = "gpt-54"
        models = [(model_key, "GPT-5.4", "gpt", "off")]
        all_data = {model_key: records}
        expected_pct = 100.0 * float(np.mean(fixed_passes))
        self.assertEqual(expected_pct, 50.0)

        rows, macro = reporting.collect_coherence_metrics(all_data, models)
        with self.subTest(output="headline table"):
            self.assertAlmostEqual(rows[0]["mono_pct"], expected_pct)
            self.assertAlmostEqual(macro["mono_pct"], expected_pct)

        matrix = reporting._build_category_matrix(
            all_data, models, ["reversed", "increasing"]
        )
        with self.subTest(output="category heatmap"):
            np.testing.assert_allclose(matrix, [[0.0, 100.0]])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = phase6b_coherence_json_path(model_key, root)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "aggregate": {"overall": {"monotonicity_rate": expected_pct / 100}},
                "per_variation_set": records,
            }), encoding="utf-8")
            figure = reporting.fig9_pred_util_coherence_scatter(
                root, {model_key: {"mean_test_auc": "0.9"}}
            )
            self.assertIsNotNone(figure)
            try:
                scatter_pct = float(figure.axes[0].collections[0].get_offsets()[0, 0])
                with self.subTest(output="predictive utility scatter"):
                    self.assertAlmostEqual(scatter_pct, expected_pct)
                    self.assertAlmostEqual(scatter_pct, rows[0]["mono_pct"])
            finally:
                plt.close(figure)

    def test_missing_fixed_direction_metric_is_not_reported_as_zero(self) -> None:
        with self.assertRaises(KeyError):
            reporting.get_mono({"monotonicity_rate_inferred_direction": 1.0})


if __name__ == "__main__":
    unittest.main()
