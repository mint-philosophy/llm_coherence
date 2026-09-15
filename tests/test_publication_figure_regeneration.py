"""Regression tests for the portable 9 pt publication-figure workflow."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt

from llm_coherence.reporting.make_fig_table import (
    PUBLICATION_FONT_SIZE_PT,
    _draw_category_heatmap,
    validate_minimum_figure_font_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGENERATION_SCRIPT = (
    REPO_ROOT / "scripts" / "06_reporting" / "regenerate_figures_3_5_9pt.py"
)


def load_regeneration_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "regenerate_publication_figures_9pt",
        REGENERATION_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {REGENERATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PublicationFigureFontTests(unittest.TestCase):
    def test_font_validator_rejects_text_below_nine_points(self) -> None:
        figure, axis = plt.subplots()
        self.addCleanup(plt.close, figure)
        axis.set_title("too small", fontsize=PUBLICATION_FONT_SIZE_PT - 1)

        with self.assertRaisesRegex(ValueError, "below the 9 pt"):
            validate_minimum_figure_font_size(figure)

    def test_category_heatmap_uses_nine_point_text(self) -> None:
        models = [
            ("gpt-56-sol", "", "OpenAI", "off"),
            ("glm-45-hybrid", "", "GLM", "off"),
        ]
        records = [
            {"category": "Life and species", "monotonicity_rate": 0.85},
            {"category": "Religion and spirituality", "monotonicity_rate": 0.50},
        ]
        all_data = {model_key: list(records) for model_key, *_ in models}

        figure = _draw_category_heatmap(
            all_data,
            models,
            ylabel="Reasoning off",
        )
        self.addCleanup(plt.close, figure)

        validate_minimum_figure_font_size(figure)
        self.assertGreaterEqual(figure.get_size_inches()[1], 3.65)


class PortableRegenerationScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_regeneration_script()

    def test_script_contains_no_date_stamped_input_directories(self) -> None:
        source = REGENERATION_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("complete24_qwenon", source)
        self.assertNotRegex(source, r"20\d{6}")

    def test_custom_directories_generate_table_based_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables_dir = root / "tables"
            figures_dir = root / "figures"

            write_csv(
                tables_dir / "table_headline_combined.csv",
                [
                    {
                        "model_key": "gpt-56-sol",
                        "model": "GPT-5.6 Sol (reasoning off)",
                        "reasoning": "off",
                        "tier_x_tier_acc_pct": 99.0,
                        "strict_mono_pct": 85.2,
                    },
                    {
                        "model_key": "gpt-56-sol-thinking",
                        "model": "GPT-5.6 Sol (reasoning on)",
                        "reasoning": "on",
                        "tier_x_tier_acc_pct": 98.8,
                        "strict_mono_pct": 85.6,
                    },
                ],
            )
            write_csv(
                tables_dir / "table_coherence_metrics.csv",
                [
                    {
                        "model_key": "gpt-56-sol",
                        "model": "GPT-5.6 Sol (reasoning off)",
                        "reasoning": "off",
                        "strict_mono_pct": 85.2,
                        "iso_r2_bidirectional": 0.960,
                        "jt_sig_pct": 61.9,
                    },
                    {
                        "model_key": "gpt-56-sol-thinking",
                        "model": "GPT-5.6 Sol (reasoning on)",
                        "reasoning": "on",
                        "strict_mono_pct": 85.6,
                        "iso_r2_bidirectional": 0.972,
                        "jt_sig_pct": 59.3,
                    },
                ],
            )

            result = self.module.main(
                [
                    "--tables-dir",
                    str(tables_dir),
                    "--output-dir",
                    str(figures_dir),
                    "--figure",
                    "within-ladder",
                    "--figure",
                    "triptych",
                ]
            )

            self.assertEqual(result, 0)
            expected = {
                "fig_within_ladder_internal_coherence_bars_9pt.pdf",
                "fig_within_ladder_internal_coherence_bars_9pt.png",
                "fig1b_metrics_triptych_9pt.pdf",
                "fig1b_metrics_triptych_9pt.png",
            }
            self.assertEqual(
                {path.name for path in figures_dir.iterdir()},
                expected,
            )


if __name__ == "__main__":
    unittest.main()
