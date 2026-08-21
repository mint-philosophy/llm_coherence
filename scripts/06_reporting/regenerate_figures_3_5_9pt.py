#!/usr/bin/env python3
"""Regenerate the two dense main-paper figures with a 9 pt minimum font."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


REPO = Path(__file__).resolve().parents[2]
REPORTING = REPO / "outputs" / "09_reporting"
TABLE_DIR = REPORTING / "tables_original_style_complete24_qwenon_20260820"
FIG_DIR = REPORTING / "figures_original_style_complete24_qwenon_20260820"

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
    "GPT-5.6 Terra": "5.6-Terra",
    "GLM-4.5 Hybrid": "GLM-Hybrid",
    "GLM-4.5 Base": "GLM-4.5 Base",
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "Mistral-Small-2603": "Mistral-S",
    "Ministral-3B-2512": "Ministral-3B",
    "Nemotron-3-Super": "Nemotron",
    "Opus-4.6": "Opus-4.6",
    "Qwen-3.7 Flash": "Qwen-3.7 Flash",
}

ORIGINAL_TRIPTYCH_NAMES = {
    "Opus-4.6": "Opus-4.6",
    "GLM-4.5 Base": "GLM-Base",
    "GLM-4.5 Hybrid": "GLM-Hybrid",
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "Ministral-3B-2512": "Ministral-3B",
    "Mistral-Small-2603": "Mistral-S",
    "Nemotron-3-Super": "Nemotron",
    "GPT-5.4": "5.4",
    "GPT-5.4-Mini": "5.4-Mini",
    "GPT-5.4-Nano": "5.4-Nano",
    "GPT-5.6 Luna": "5.6-Luna",
    "GPT-5.6 Sol": "5.6-Sol",
    "GPT-5.6 Terra": "5.6-Terra",
    "Qwen-3.7 Flash": "Qwen-3.7 Flash",
}


def original_column_name(row: dict[str, str]) -> str:
    """Match the compact labels and coverage daggers in the original plot."""
    base = short_name(row["model"])
    label = ORIGINAL_COLUMN_NAMES.get(base, base)
    if row.get("tier_x_tier_complete_coverage", "True").lower() != "true":
        label += "†"
    return label


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


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(path)
    plt.close(fig)


def make_within_ladder() -> None:
    rows = read_rows(TABLE_DIR / "table_headline_combined.csv")
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
        figsize=(7.0, 4.55),
        sharex=True,
        gridspec_kw={"width_ratios": [13, 11], "wspace": 0.50},
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
                fontsize=9,
                fontweight="normal",
            )
            ax.text(
                min(mon + 0.60, 99.0),
                yi + 0.16,
                f"{mon:.1f}",
                ha="left",
                va="center",
                color="#444444",
                fontsize=9,
                fontweight="normal",
            )
        ax.set_yticks(y)
        ax.set_yticklabels([original_column_name(row) for row in mode_rows], fontsize=9)
        ax.set_ylim(len(mode_rows) - 0.5, -0.5)
        ax.set_xlim(0, 108)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Percentage (%)", fontsize=9)
        ax.set_title(f"Reasoning {mode}", loc="left", fontsize=9)
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
        fontsize=9,
    )
    fig.text(
        0.5,
        0.018,
        "† Parseable-response denominator.",
        ha="center",
        va="bottom",
        color="#666666",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.17)
    save(fig, "fig_within_ladder_internal_coherence_bars_9pt")


def model_order(row: dict[str, str]) -> tuple[int, int, str]:
    key = row["model_key"]
    groups = ["opus", "glm-45-base", "glm-45-hybrid", "llama", "ministral", "mistral", "nemotron", "gpt-54", "gpt-56", "qwen"]
    group_index = next((i for i, prefix in enumerate(groups) if key.startswith(prefix)), 99)
    return group_index, 1 if row["reasoning"] == "on" else 0, key


def make_triptych() -> None:
    rows = sorted(read_rows(TABLE_DIR / "table_coherence_metrics.csv"), key=model_order)
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
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(f"{title}\nMean {macro}", fontsize=9, pad=6, linespacing=1.05)
        ax.axvline(float(macro.rstrip("%")), color="#333333", lw=0.8, ls=":")
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels(labels, fontsize=9)
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
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
    )
    # Match the original compact layout while reserving enough physical space
    # for 9 pt labels at full two-column width.
    fig.subplots_adjust(left=0.165, right=0.995, top=0.84, bottom=0.10)
    save(fig, "fig1b_metrics_triptych_9pt")


if __name__ == "__main__":
    make_within_ladder()
    make_triptych()
