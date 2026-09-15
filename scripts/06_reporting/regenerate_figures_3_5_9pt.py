#!/usr/bin/env python3
"""Regenerate the dense main-paper figures with a 9 pt minimum font."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from llm_coherence.paths import (
    FIGURES_OUTPUT_DIR,
    LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR,
    TABLES_OUTPUT_DIR,
    resolve_repo_path,
)
from llm_coherence.reporting.make_fig_table import (
    PUBLICATION_FONT_SIZE_PT,
    _build_category_matrix,
    _category_short,
    base_model_label,
    build_model_tuples,
    discover_reporting_model_keys,
    load_all_per_set,
    validate_minimum_figure_font_size,
)

DEFAULT_TABLE_DIR = TABLES_OUTPUT_DIR
DEFAULT_RESULTS_DIR = LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR
DEFAULT_FIGURE_DIR = FIGURES_OUTPUT_DIR

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": PUBLICATION_FONT_SIZE_PT,
        "axes.titlesize": PUBLICATION_FONT_SIZE_PT,
        "axes.labelsize": PUBLICATION_FONT_SIZE_PT,
        "xtick.labelsize": PUBLICATION_FONT_SIZE_PT,
        "ytick.labelsize": PUBLICATION_FONT_SIZE_PT,
        "legend.fontsize": PUBLICATION_FONT_SIZE_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

ORANGE = "#D55E00"
BLUE = "#0072B2"
FAMILY_COLORS = {
    "opus": "#8E63B6",
    "glm": "#F28E2B",
    "llama": "#D58FB3",
    "mistral": "#D9A500",
    "nemotron": "#45A65A",
    "gpt54": "#2387C9",
    "gpt56": "#C58A00",
    "qwen": "#4AA39B",
}

# Distinct tints identify model variants; hatching alone identifies reasoning.
# This mirrors the original reporting palette and keeps matched off/on runs
# visually paired without collapsing Nano/Mini/Standard or Luna/Terra/Sol.
MODEL_COLORS = {
    "gpt-54": "#1B365D",
    "gpt-54-thinking": "#1B365D",
    "gpt-54-mini": "#0072B2",
    "gpt-54-mini-thinking": "#0072B2",
    "gpt-54-nano": "#56B4E9",
    "gpt-54-nano-thinking": "#56B4E9",
    "gpt-56-sol": "#D98E04",
    "gpt-56-sol-thinking": "#D98E04",
    "gpt-56-luna": "#51458A",
    "gpt-56-luna-thinking": "#51458A",
    "gpt-56-terra": "#167C75",
    "gpt-56-terra-thinking": "#167C75",
    "opus-46": "#9467BD",
    "opus-46-thinking": "#9467BD",
    "nemotron-3-super": "#2CA02C",
    "nemotron-3-super-thinking": "#2CA02C",
    "glm-45-hybrid": "#D55E00",
    "glm-45-hybrid-thinking": "#D55E00",
    "glm-45-base-logprobs": "#FFB347",
    "qwen-37-flash-openrouter": "#8C564B",
    "qwen-37-flash-openrouter-thinking": "#8C564B",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["model_key"] != "macro_avg"]


def short_name(model: str) -> str:
    return (
        model.replace(" (reasoning off)", "")
        .replace(" (reasoning on)", "")
        .replace(" (baseline)", "")
        .replace("GLM-4.5-Hybrid", "GLM-4.5 Hybrid")
    )


ORIGINAL_COLUMN_NAMES = {
    "GPT-5.6 Sol": "GPT-5.6 Sol",
    "GPT-5.6 Luna": "GPT-5.6 Luna",
    "GPT-5.6 Terra": "GPT-5.6 Terra",
    "GLM-4.5 Hybrid": "GLM-4.5 Hybrid",
    "GLM-4.5 Base": "GLM-4.5 Base",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Mistral-Small-2603": "Mistral-Small-2603",
    "Ministral-3B-2512": "Ministral-3B-2512",
    "Nemotron-3-Super": "Nemotron-3-Super 120B",
    "Opus-4.6": "Opus-4.6",
    "Qwen-3.7 Flash": "Qwen-3.7 Flash",
    "GPT-5.4-Mini": "GPT-5.4 Mini",
    "GPT-5.4-Nano": "GPT-5.4 Nano",
}

ORIGINAL_TRIPTYCH_NAMES = {
    "Opus-4.6": "Opus-4.6",
    "GLM-4.5 Base": "GLM-4.5 Base",
    "GLM-4.5 Hybrid": "GLM-4.5 Hybrid",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Ministral-3B-2512": "Ministral-3B-2512",
    "Mistral-Small-2603": "Mistral-Small-2603",
    "Nemotron-3-Super": "Nemotron-3-Super 120B",
    "GPT-5.4": "GPT-5.4",
    "GPT-5.4-Mini": "GPT-5.4 Mini",
    "GPT-5.4-Nano": "GPT-5.4 Nano",
    "GPT-5.6 Luna": "GPT-5.6 Luna",
    "GPT-5.6 Sol": "GPT-5.6 Sol",
    "GPT-5.6 Terra": "GPT-5.6 Terra",
    "Qwen-3.7 Flash": "Qwen-3.7 Flash",
}


def original_column_name(row: dict[str, str]) -> str:
    """Return the publication model name used in the paired-bar plot."""
    base = short_name(row["model"])
    return ORIGINAL_COLUMN_NAMES.get(base, base)


def original_triptych_name(row: dict[str, str]) -> str:
    base = short_name(row["model"])
    return f"{ORIGINAL_TRIPTYCH_NAMES.get(base, base)} · {row['reasoning']}"


def family(model_key: str) -> str:
    if model_key.startswith("opus"):
        return "opus"
    if model_key.startswith("glm"):
        return "glm"
    if model_key.startswith("llama"):
        return "llama"
    if model_key.startswith("mistral") or model_key.startswith("ministral"):
        return "mistral"
    if model_key.startswith("nemotron"):
        return "nemotron"
    if model_key.startswith("gpt-54"):
        return "gpt54"
    if model_key.startswith("gpt-56"):
        return "gpt56"
    return "qwen"


def model_color(model_key: str) -> str:
    return MODEL_COLORS.get(model_key, FAMILY_COLORS[family(model_key)])


def save(fig: plt.Figure, stem: str, figure_dir: Path) -> list[Path]:
    """Validate and save one figure as PDF and PNG."""
    validate_minimum_figure_font_size(fig)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = figure_dir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(path)
        paths.append(path)
    plt.close(fig)
    return paths


def make_within_ladder(table_dir: Path, figure_dir: Path) -> list[Path]:
    rows = read_rows(table_dir / "table_headline_combined.csv")
    modes = ["off", "on"]
    grouped = {
        mode: sorted(
            [row for row in rows if row["reasoning"] == mode],
            key=lambda row: -float(row["strict_mono_pct"]),
        )
        for mode in modes
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.35),
        sharex=True,
        gridspec_kw={"width_ratios": [13, 11], "wspace": 1.02},
    )
    for ax, mode in zip(axes, modes):
        mode_rows = grouped[mode]
        y = np.arange(len(mode_rows))
        accuracy = np.array([float(row["tier_x_tier_acc_pct"]) for row in mode_rows])
        mono = np.array([float(row["strict_mono_pct"]) for row in mode_rows])
        # Preserve the original paired-bar geometry.  The taller canvas keeps
        # 9 pt labels readable without changing the visual design.
        ax.barh(y - 0.16, accuracy, height=0.28, color=BLUE)
        ax.barh(y + 0.16, mono, height=0.28, color=ORANGE)
        for yi, acc, mon in zip(y, accuracy, mono):
            ax.text(
                min(acc + 0.65, 106.5),
                yi - 0.16,
                f"{acc:.1f}",
                ha="left",
                va="center",
                color="#333333",
                fontsize=PUBLICATION_FONT_SIZE_PT,
                fontweight="normal",
            )
            ax.text(
                min(mon + 0.60, 99.0),
                yi + 0.16,
                f"{mon:.1f}",
                ha="left",
                va="center",
                color="#444444",
                fontsize=PUBLICATION_FONT_SIZE_PT,
                fontweight="normal",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [original_column_name(row) for row in mode_rows],
            fontsize=PUBLICATION_FONT_SIZE_PT,
        )
        ax.set_ylim(len(mode_rows) - 0.5, -0.5)
        ax.set_xlim(0, 108)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Percentage (%)", fontsize=PUBLICATION_FONT_SIZE_PT)
        ax.set_title(
            f"Reasoning {mode}",
            loc="left",
            fontsize=PUBLICATION_FONT_SIZE_PT,
        )
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Line2D([], [], color=ORANGE, lw=5, label="Strict monotonicity"),
            Line2D([], [], color=BLUE, lw=5, label="Tier-pair accuracy"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=PUBLICATION_FONT_SIZE_PT,
    )
    fig.subplots_adjust(left=0.235, right=0.985, top=0.82, bottom=0.13)
    return save(fig, "fig_within_ladder_internal_coherence_bars_9pt", figure_dir)


def make_category_heatmap(
    reasoning: str,
    results_dir: Path,
    figure_dir: Path,
) -> list[Path]:
    """Category heatmap sized for a two-column AAAI figure."""
    model_keys = discover_reporting_model_keys(results_dir, results_dir)
    models = build_model_tuples(model_keys)
    all_data = load_all_per_set(results_dir, models)
    selected_models = [
        row for row in models if row[3] == reasoning and row[0] in all_data
    ]
    if not selected_models:
        raise RuntimeError(f"No reasoning-{reasoning} models were found")

    first_model_key = selected_models[0][0]
    categories = sorted({record["category"] for record in all_data[first_model_key]})
    category_counts = {
        category: sum(
            record["category"] == category
            for record in all_data[first_model_key]
        )
        for category in categories
    }
    column_labels = [
        f"{_category_short(category)} ($n$={category_counts[category]})"
        for category in categories
    ]
    row_labels = []
    for model_key, _label, _family, _reasoning in selected_models:
        label = base_model_label(model_key).replace(" (baseline)", "")
        label = label.replace("GPT-5.4-Mini", "GPT-5.4 Mini")
        label = label.replace("GPT-5.4-Nano", "GPT-5.4 Nano")
        label = label.replace("GLM-4.5-Hybrid", "GLM-4.5 Hybrid")
        label = label.replace("Nemotron-3-Super", "Nemotron-3-Super 120B")
        row_labels.append(label)

    matrix = _build_category_matrix(all_data, selected_models, categories)
    display_matrix = np.vstack([matrix, np.mean(matrix, axis=0, keepdims=True)])

    figure_height = 4.20 if reasoning == "off" else 3.65
    fig, ax = plt.subplots(figsize=(7.0, figure_height), dpi=300)
    image = ax.imshow(
        display_matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
    )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        column_labels,
        rotation=42,
        ha="right",
        rotation_mode="anchor",
        fontsize=PUBLICATION_FONT_SIZE_PT,
    )
    ax.set_yticks(range(len(row_labels) + 1))
    ax.set_yticklabels(
        row_labels + ["Column mean"],
        fontsize=PUBLICATION_FONT_SIZE_PT,
    )
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_title(
        f"Reasoning {reasoning}",
        fontsize=PUBLICATION_FONT_SIZE_PT,
        pad=4,
    )
    ax.axhline(len(row_labels) - 0.5, color="white", linewidth=2.0)
    ax.axhline(len(row_labels) - 0.5, color="0.25", linewidth=0.8)

    for row_index in range(display_matrix.shape[0]):
        for column_index in range(display_matrix.shape[1]):
            value = float(display_matrix[row_index, column_index])
            color = "white" if value <= 20 or value >= 84 else "#202020"
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                color=color,
                fontsize=PUBLICATION_FONT_SIZE_PT,
                fontweight="bold" if row_index == len(row_labels) else "normal",
            )

    fig.subplots_adjust(left=0.245, right=0.925, bottom=0.31, top=0.91)
    colorbar_axis = fig.add_axes([0.945, 0.31, 0.014, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(
        "Strict monotonicity (%)",
        fontsize=PUBLICATION_FONT_SIZE_PT,
    )
    colorbar.ax.tick_params(labelsize=PUBLICATION_FONT_SIZE_PT)
    stem = (
        "fig7_category_breakdown_off_9pt"
        if reasoning == "off"
        else "fig7b_category_breakdown_on_9pt"
    )
    return save(fig, stem, figure_dir)


def make_category_heatmap_off(
    results_dir: Path,
    figure_dir: Path,
) -> list[Path]:
    """Reasoning-off category heatmap with all reported configurations."""
    return make_category_heatmap("off", results_dir, figure_dir)


def make_category_heatmap_on(
    results_dir: Path,
    figure_dir: Path,
) -> list[Path]:
    """Reasoning-on category heatmap with all reported configurations."""
    return make_category_heatmap("on", results_dir, figure_dir)


def model_order(row: dict[str, str]) -> tuple[int, int, str]:
    key = row["model_key"]
    groups = ["opus", "glm-45-base", "glm-45-hybrid", "llama", "ministral", "mistral", "nemotron", "gpt-54", "gpt-56", "qwen"]
    group_index = next((i for i, prefix in enumerate(groups) if key.startswith(prefix)), 99)
    return group_index, 1 if row["reasoning"] == "on" else 0, key


def make_triptych(table_dir: Path, figure_dir: Path) -> list[Path]:
    rows = sorted(
        read_rows(table_dir / "table_coherence_metrics.csv"),
        key=model_order,
    )
    specs = [
        ("strict_mono_pct", (0, 100), "Strict mono. (%)", "No adjacent-tier violations", "64.4%"),
        ("iso_r2_bidirectional", (0.5, 1.0), "Iso R² (bi)", r"Best monotonic fit ($\uparrow$ or $\downarrow$)", "0.938"),
        ("jt_sig_pct", (50, 95), "J–T sig. (%)", "Ordered trend in raw trials", "75.0%"),
    ]
    y = np.arange(len(rows))
    labels = [original_triptych_name(row) for row in rows]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 5.60),
        sharey=True,
        gridspec_kw={"wspace": 0.16},
    )
    for panel, (ax, (field, limits, xlabel, title, macro)) in enumerate(zip(axes, specs)):
        values = np.array([float(row[field]) for row in rows])
        colors = [model_color(row["model_key"]) for row in rows]
        bars = ax.barh(y, values, height=0.72, color=colors, edgecolor="white")
        for bar, row in zip(bars, rows):
            if row["reasoning"] == "on":
                bar.set_hatch("//")
                bar.set_edgecolor("white")
        ax.set_xlim(*limits)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlabel(xlabel, fontsize=PUBLICATION_FONT_SIZE_PT)
        ax.set_title(
            f"{title}\nMean {macro}",
            fontsize=PUBLICATION_FONT_SIZE_PT,
            pad=6,
            linespacing=1.05,
        )
        ax.axvline(float(macro.rstrip("%")), color="#333333", lw=0.8, ls=":")
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels(labels, fontsize=PUBLICATION_FONT_SIZE_PT)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Patch(facecolor="#888888", edgecolor="white", label="reasoning off"),
            Patch(facecolor="#888888", edgecolor="white", hatch="//", label="reasoning on"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=PUBLICATION_FONT_SIZE_PT,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
    )
    # Match the original compact layout while reserving enough physical space
    # for 9 pt labels at full two-column width.
    fig.subplots_adjust(left=0.265, right=0.995, top=0.84, bottom=0.10)
    return save(fig, "fig1b_metrics_triptych_9pt", figure_dir)


FIGURE_CHOICES = (
    "within-ladder",
    "category-off",
    "category-on",
    "triptych",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate the dense publication figures with 9 pt text."
    )
    parser.add_argument(
        "--tables-dir",
        type=resolve_repo_path,
        default=DEFAULT_TABLE_DIR,
        help="Directory containing table_headline_combined.csv and "
        "table_coherence_metrics.csv (default: results/tables).",
    )
    parser.add_argument(
        "--results-dir",
        type=resolve_repo_path,
        default=DEFAULT_RESULTS_DIR,
        help="Model-run root used for category heatmaps (default: outputs).",
    )
    parser.add_argument(
        "--output-dir",
        type=resolve_repo_path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for generated PDF and PNG files "
        "(default: results/figures).",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=FIGURE_CHOICES,
        dest="figures",
        help="Figure to generate; repeat to select more than one. "
        "The default generates all four figure groups.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = args.figures or list(FIGURE_CHOICES)

    if "within-ladder" in selected:
        make_within_ladder(args.tables_dir, args.output_dir)
    if "category-off" in selected:
        make_category_heatmap_off(args.results_dir, args.output_dir)
    if "category-on" in selected:
        make_category_heatmap_on(args.results_dir, args.output_dir)
    if "triptych" in selected:
        make_triptych(args.tables_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
