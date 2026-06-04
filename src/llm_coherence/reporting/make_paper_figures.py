"""
Generate AIES 2026 paper figures from coherence analysis JSONs.

Reads per-model coherence JSONs (from analyze_7tier_coherence.py), predictive-
utility CSVs (from predictive_utility.py), and raw results files, then produces
fig1--fig12 as PDF+PNG (figures/) and table_coherence_metrics.{tex,csv} (tables/).

Models are specified via repeatable --model flags or auto-discovered from
the results directory. No hardcoded model list is required.

Usage:
    # Explicit models
    python make_paper_figures.py --model ministral-3b-2512-openrouter --model glm-45-hybrid

    # Auto-discover all models under results/
    python make_paper_figures.py

    # Custom dirs
    python make_paper_figures.py --results-dir results --output-dir ./figures --tables-dir ./tables
"""

import argparse
import csv
import json
import math
import pathlib
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

from llm_coherence.paths import MODEL_RUNS_OUTPUT_DIR, REPORT_OUTPUTS_DIR, REPO_ROOT

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PARAM_VAR = REPO_ROOT

N_TIERS = 7

# Palette cycles when family cannot be inferred.
_DEFAULT_PALETTE = [
    "#0072B2", "#D55E00", "#2CA02C", "#9467BD", "#E377C2",
    "#8C564B", "#17BECF", "#BCBD22", "#7F7F7F", "#FF7F0E",
]

FAMILY_KEYWORDS = {
    "gpt":       "OpenAI",
    "o1":        "OpenAI",
    "o3":        "OpenAI",
    "o4":        "OpenAI",
    "opus":      "Anthropic",
    "claude":    "Anthropic",
    "sonnet":    "Anthropic",
    "haiku":     "Anthropic",
    "nemotron":  "NVIDIA",
    "glm":       "GLM",
    "llama":     "Meta",
    "ministral": "Mistral",
    "mistral":   "Mistral",
    "magistral": "Mistral",
    "deepseek":  "DeepSeek",
    "qwen":      "Qwen",
    "gemini":    "Google",
}

FAMILY_COLORS = {
    "OpenAI":    "#0072B2",
    "Anthropic": "#9467BD",
    "NVIDIA":    "#2CA02C",
    "GLM":       "#D55E00",
    "Meta":      "#E377C2",
    "Mistral":   "#B8860B",
    "DeepSeek":  "#17BECF",
    "Qwen":      "#8C564B",
    "Google":    "#BCBD22",
}

# Per-model tints where a family default would collide or need emphasis.
MODEL_COLOR_OVERRIDES = {
    "glm-45-base-logprobs": "#FFB347",
}

REASON_HATCH = {"on": "//", "off": None}

BH_PASS_FACE = "#888888"
BH_FAIL_EDGE = "#C0392B"


def infer_family(model_key: str) -> str:
    key_lower = model_key.lower()
    for keyword, family in FAMILY_KEYWORDS.items():
        if keyword in key_lower:
            return family
    return "Other"


def infer_reasoning(model_key: str) -> str:
    return "on" if "thinking" in model_key.lower() else "off"


def make_display_label(model_key: str) -> str:
    """Derive a short human-friendly label from a model key."""
    label = model_key.replace("-openrouter", "").replace("_", " ")
    label = re.sub(r"-(\d)", r" \1", label)
    label = label.replace("-", " ").title()
    if len(label) > 28:
        label = label[:26] + ".."
    return label


def build_model_tuples(model_keys: list[str]) -> list[tuple[str, str, str, str]]:
    """Build (file_key, display_label, family, reasoning) tuples from model keys."""
    return [
        (k, make_display_label(k), infer_family(k), infer_reasoning(k))
        for k in model_keys
    ]


def discover_models(results_dir: pathlib.Path) -> list[str]:
    """Auto-discover model keys from subdirectories that contain coherence JSONs."""
    if not results_dir.exists():
        return []
    keys = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        coherence = d / f"phase6b_coherence_{d.name}.json"
        if coherence.exists():
            keys.append(d.name)
    return keys


def find_coherence_json(results_dir: pathlib.Path, model_key: str) -> pathlib.Path | None:
    """Find the coherence analysis JSON for a model under model-scoped results."""
    candidates = [
        results_dir / model_key / f"phase6b_coherence_{model_key}.json",
        results_dir / model_key / f"{model_key}.json",
        results_dir / model_key / f"{model_key}_analysis.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def load_analysis(results_dir: pathlib.Path, model_key: str) -> dict | None:
    path = find_coherence_json(results_dir, model_key)
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean_ci_95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return (float(arr[0]) if n == 1 else 0.0), 0.0
    se = float(arr.std(ddof=1) / math.sqrt(n))
    return float(arr.mean()), 1.96 * se


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


_palette_idx = 0

def get_color(family, reasoning, model_key=None):
    global _palette_idx
    if model_key and model_key in MODEL_COLOR_OVERRIDES:
        base = MODEL_COLOR_OVERRIDES[model_key]
    elif family in FAMILY_COLORS:
        base = FAMILY_COLORS[family]
    else:
        base = _DEFAULT_PALETTE[_palette_idx % len(_DEFAULT_PALETTE)]
        FAMILY_COLORS[family] = base
        _palette_idx += 1
    if reasoning == "on" and not (model_key and model_key in MODEL_COLOR_OVERRIDES):
        rgb = mcolors.to_rgb(base)
        return tuple(min(1.0, c * 1.15) for c in rgb)
    return base


def get_mono(rec):
    return rec.get("monotonicity_rate_inferred_direction", rec.get("monotonicity_rate", 0))


def get_r2(rec):
    return rec.get("mean_isotonic_r2_bidirectional", rec.get("mean_isotonic_r2", 0))


def get_jt(rec):
    return rec.get("jt_significant_rate", 0)


def load_overall_stats(results_dir: pathlib.Path, model_key: str) -> dict:
    """Per-model aggregate stats from coherence JSON overall block."""
    d = load_analysis(results_dir, model_key)
    if not d:
        return {}
    return (d.get("aggregate") or {}).get("overall") or d.get("overall") or {}


def _compact_bar_label(file_key: str) -> str:
    """Short x-axis label for crowded bar charts."""
    labels = {
        "glm-45-base-logprobs": "GLM Base",
        "glm-45-hybrid": "GLM Hybrid",
        "glm-45-hybrid-thinking": "GLM Hyb*",
        "gpt-54": "GPT-54",
        "gpt-54-mini": "GPT-54 Mini",
        "gpt-54-mini-thinking": "GPT-Mini*",
        "gpt-54-nano": "GPT-54 Nano",
        "gpt-54-nano-thinking": "GPT-Nano*",
        "gpt-54-thinking": "GPT-54*",
        "llama-31-8b-instruct-openrouter": "Llama 8B",
        "ministral-3b-2512-openrouter": "Ministral 3B",
        "nemotron-3-super": "Nemotron",
        "nemotron-3-super-thinking": "Nemo*",
        "opus-46": "Opus 4.6",
        "mistral-small-2603-openrouter-thinking": "Mistral Sm*",
    }
    return labels.get(file_key, _short_model_label(file_key, max_len=12))


def _paper_table_model_name(file_key: str, display_label: str) -> str:
    """Publication-ready model name for the coherence metrics table."""
    names = {
        "glm-45-base-logprobs": "GLM-4.5 Base Logprobs",
        "glm-45-hybrid": "GLM-4.5 Hybrid",
        "glm-45-hybrid-thinking": "GLM-4.5 Hybrid Thinking",
        "gpt-54": "GPT-5.4",
        "gpt-54-mini": "GPT-5.4 Mini",
        "gpt-54-mini-thinking": "GPT-5.4 Mini Thinking",
        "gpt-54-nano": "GPT-5.4 Nano",
        "gpt-54-nano-thinking": "GPT-5.4 Nano Thinking",
        "gpt-54-thinking": "GPT-5.4 Thinking",
        "llama-31-8b-instruct-openrouter": "Llama 3.1 8B Instruct",
        "ministral-3b-2512-openrouter": "Ministral 3B",
        "mistral-small-2603-openrouter-thinking": "Mistral Small 3.1 Thinking",
        "nemotron-3-super": "Nemotron 3 Super",
        "nemotron-3-super-thinking": "Nemotron 3 Super Thinking",
        "opus-46": "Claude Opus 4.6",
    }
    return names.get(file_key, display_label)


def collect_coherence_metrics(
    all_data: dict,
    active_models: list,
) -> tuple[list[dict], dict]:
    """Per-model coherence summary rows and macro averages (unweighted across models)."""
    rows = []
    for file_key, label, _family, _reasoning in active_models:
        per_set = all_data[label]
        mono = float(np.mean([get_mono(r) for r in per_set]))
        iso_r2 = float(np.mean([get_r2(r) for r in per_set]))
        jt = float(np.mean([get_jt(r) for r in per_set]))
        rows.append({
            "file_key": file_key,
            "label": label,
            "table_name": _paper_table_model_name(file_key, label),
            "mono_pct": mono * 100.0,
            "iso_r2": iso_r2,
            "jt_pct": jt * 100.0,
            "n_ladders": len(per_set),
        })
    macro = {
        "mono_pct": float(np.mean([r["mono_pct"] for r in rows])),
        "iso_r2": float(np.mean([r["iso_r2"] for r in rows])),
        "jt_pct": float(np.mean([r["jt_pct"] for r in rows])),
    }
    return rows, macro


def print_coherence_metrics_table(rows: list[dict], macro: dict) -> None:
    print(f"\n{'Model':<35} {'Mono%':>7} {'IsoR2':>7} {'JT%':>6} {'N':>4}")
    print("-" * 62)
    for r in rows:
        print(
            f"  {r['label']:<33} {r['mono_pct']:>6.1f}% "
            f"{r['iso_r2']:>6.3f} {r['jt_pct']:>5.1f}% {r['n_ladders']:>4}"
        )
    print("-" * 62)
    print(
        f"  {'MACRO AVG':<33} {macro['mono_pct']:>6.1f}% "
        f"{macro['iso_r2']:>6.3f} {macro['jt_pct']:>5.1f}%"
    )


def _sort_rows_by_mono_desc(rows: list[dict]) -> list[dict]:
    """Best-to-worst strict monotonicity for publication tables."""

    def _key(r: dict) -> tuple:
        mono = r.get("mono_pct", float("nan"))
        if math.isnan(mono):
            return (float("inf"), r.get("table_name", r.get("file_key", "")))
        return (-mono, r.get("table_name", r.get("file_key", "")))

    return sorted(rows, key=_key)


def write_coherence_metrics_table(
    rows: list[dict],
    macro: dict,
    out_base: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write LaTeX and CSV coherence metrics tables."""
    rows = _sort_rows_by_mono_desc(rows)
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    tex_lines = [
        "% Auto-generated by make_paper_figures.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[h!]",
        "  \\centering",
        "  \\caption{Descriptive coherence metrics by model. Strict preference monotonicity is "
        "the fraction of comparison blocks with no adjacent-tier violations (set-inferred direction). "
        "Isotonic $R^2$ (bidirectional) is the mean $\\max(R^2_{\\uparrow}, R^2_{\\downarrow})$ "
        "across blocks. JT sig.\\ is the fraction of blocks with a significant "
        "Jonckheere--Terpstra ordered trend ($\\alpha = 0.05$).}",
        "  \\label{tab:coherence_metrics}",
        "  \\small",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Model & Strict mono (\\%) & Iso.\\ $R^2$ (bi) & JT sig.\\ (\\%) & $N$ ladders \\\\",
        "    \\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"    {r['table_name']:<28} & {r['mono_pct']:4.1f} & {r['iso_r2']:.3f} "
            f"& {r['jt_pct']:4.1f} & {r['n_ladders']:3d} \\\\"
        )
    tex_lines.extend([
        "    \\midrule",
        f"    {'Macro avg':<28} & {macro['mono_pct']:4.1f} & {macro['iso_r2']:.3f} "
        f"& {macro['jt_pct']:4.1f} & --- \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_key", "model", "strict_mono_pct", "iso_r2_bidirectional",
                "jt_sig_pct", "n_ladders",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model_key": r["file_key"],
                "model": r["table_name"],
                "strict_mono_pct": f"{r['mono_pct']:.1f}",
                "iso_r2_bidirectional": f"{r['iso_r2']:.3f}",
                "jt_sig_pct": f"{r['jt_pct']:.1f}",
                "n_ladders": r["n_ladders"],
            })
        writer.writerow({
            "model_key": "macro_avg",
            "model": "Macro avg",
            "strict_mono_pct": f"{macro['mono_pct']:.1f}",
            "iso_r2_bidirectional": f"{macro['iso_r2']:.3f}",
            "jt_sig_pct": f"{macro['jt_pct']:.1f}",
            "n_ladders": "",
        })

    return tex_path, csv_path


def collect_pred_util_metrics(
    results_dir: pathlib.Path,
    pred_util: dict[str, dict],
    active_models: list,
) -> tuple[list[dict], dict]:
    """Per-model predictive-utility rows and pooled/unweighted macro summaries."""
    key_order = [fk for fk, *_ in active_models if fk in pred_util]
    for k in sorted(pred_util):
        if k not in key_order:
            key_order.append(k)

    rows = []
    for file_key in key_order:
        row = pred_util[file_key]
        mono = coherence_aggregate_mono(results_dir, file_key)
        rows.append({
            "file_key": file_key,
            "table_name": _paper_table_model_name(file_key, file_key),
            "mean_auc": _float_csv(row.get("mean_test_auc")),
            "mean_null": _float_csv(row.get("mean_null_auc")),
            "bh_pct": _float_csv(row.get("frac_sig_at_bh_fdr05")) * 100.0,
            "n_ladders": int(_float_csv(row.get("n_sets"), 0)),
            "mono_pct": mono * 100.0 if not math.isnan(mono) else float("nan"),
        })

    pooled_sig, pooled_n = 0, 0
    for file_key in key_order:
        for ps in load_pred_util_per_set(results_dir, file_key):
            pooled_n += 1
            if _parse_sig_bh(ps.get("sig_bh_fdr05")):
                pooled_sig += 1

    xs = [r["mono_pct"] for r in rows if not math.isnan(r["mono_pct"])]
    ys = [r["mean_auc"] for r in rows]
    model_r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else float("nan")

    macro = {
        "mean_auc": float(np.mean([r["mean_auc"] for r in rows])),
        "mean_null": float(np.mean([r["mean_null"] for r in rows])),
        "bh_pct": float(np.mean([r["bh_pct"] for r in rows])),
        "n_models": len(rows),
        "pooled_n": pooled_n,
        "pooled_bh_pct": (pooled_sig / pooled_n * 100.0) if pooled_n else float("nan"),
        "pooled_bh_pass": pooled_sig,
        "model_r_mono_auc": model_r,
    }
    return rows, macro


def write_pred_util_metrics_table(
    rows: list[dict],
    macro: dict,
    out_base: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write LaTeX and CSV predictive-utility summary tables."""
    rows = _sort_rows_by_mono_desc(rows)
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    pooled_n = int(macro["pooled_n"])
    pooled_pct = macro["pooled_bh_pct"]
    pooled_pass = int(macro["pooled_bh_pass"])
    caption = (
        "Predictive utility by model (held-out test AUC vs.\\ tier-label permutation null; "
        "BH FDR $\\alpha = 0.05$ within model). "
        f"Pooled: {pooled_pass}/{pooled_n} ladder--model pairs pass BH "
        f"({pooled_pct:.1f}\\%). "
        "Three pairs were excluded (one wellbeing ladder for GPT-5.4 Mini Thinking, "
        "GPT-5.4 Thinking, and Claude Opus 4.6) because the held-out fold had no "
        "preference variation under split seed~0. "
        f"Model-level $r$ (strict monotonicity vs.\\ mean AUC) $= {macro['model_r_mono_auc']:.2f}$."
    )

    tex_lines = [
        "% Auto-generated by make_paper_figures.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[h!]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:pred_util_metrics}",
        "  \\small",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Model & BH pass (\\%) & Mean AUC & Null AUC & $N$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"    {r['table_name']:<28} & {r['bh_pct']:4.1f} & {r['mean_auc']:.3f} "
            f"& {r['mean_null']:.3f} & {r['n_ladders']:3d} \\\\"
        )
    tex_lines.extend([
        "    \\midrule",
        f"    {'Macro avg (unweighted)':<28} & {macro['bh_pct']:4.1f} "
        f"& {macro['mean_auc']:.3f} & {macro['mean_null']:.3f} & --- \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_key", "model", "bh_pass_pct", "mean_test_auc", "mean_null_auc",
                "strict_mono_pct", "n_analyzable_ladders",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model_key": r["file_key"],
                "model": r["table_name"],
                "bh_pass_pct": f"{r['bh_pct']:.1f}",
                "mean_test_auc": f"{r['mean_auc']:.3f}",
                "mean_null_auc": f"{r['mean_null']:.3f}",
                "strict_mono_pct": f"{r['mono_pct']:.1f}" if not math.isnan(r["mono_pct"]) else "",
                "n_analyzable_ladders": r["n_ladders"],
            })
        writer.writerow({
            "model_key": "macro_avg_unweighted",
            "model": "Macro avg (unweighted)",
            "bh_pass_pct": f"{macro['bh_pct']:.1f}",
            "mean_test_auc": f"{macro['mean_auc']:.3f}",
            "mean_null_auc": f"{macro['mean_null']:.3f}",
            "strict_mono_pct": "",
            "n_analyzable_ladders": "",
        })
        writer.writerow({
            "model_key": "pooled",
            "model": "Pooled (all ladder-model pairs)",
            "bh_pass_pct": f"{macro['pooled_bh_pct']:.1f}",
            "mean_test_auc": "",
            "mean_null_auc": "",
            "strict_mono_pct": "",
            "n_analyzable_ladders": macro["pooled_n"],
        })

    return tex_path, csv_path


def _ascending_indices(values: list[float] | np.ndarray) -> list[int]:
    """Indices that sort values low-to-high (stable on ties)."""
    vals = np.asarray(values, dtype=float)
    return sorted(range(len(vals)), key=lambda i: (vals[i], i))


def _annotate_macro_avg(ax, avg: float, label: str, *, color: str = "#333333"):
    """Dashed macro-average line with label in upper-left (avoids bar overlap)."""
    ax.axhline(avg, color=color, lw=1.2, ls=":", alpha=0.85, zorder=10)
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92),
        zorder=11, clip_on=True,
    )


def _model_bar_chart(
    all_data: dict,
    models: list,
    value_fn,
    ylabel: str,
    ylim: tuple,
    macro_fmt: str,
    macro_scale: float = 1.0,
    suptitle: str | None = None,
    subtitle: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper center",
) -> plt.Figure:
    """Shared bar chart for per-model headline metrics with macro-average line."""
    fig_h = 5.2 if subtitle else 4.8
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=300)
    rows = []
    for file_key, label, family, reasoning in models:
        if label not in all_data:
            continue
        vals = [value_fn(rec) for rec in all_data[label]]
        m, e = mean_ci_95(vals)
        rows.append({
            "bar_label": _compact_bar_label(file_key),
            "mean": m * macro_scale,
            "err": e * macro_scale,
            "color": get_color(family, reasoning, file_key),
            "hatch": REASON_HATCH[reasoning],
        })
    order = _ascending_indices([r["mean"] for r in rows])
    rows = [rows[i] for i in order]
    bar_labels = [r["bar_label"] for r in rows]
    means = [r["mean"] for r in rows]
    errs = [r["err"] for r in rows]
    colors = [r["color"] for r in rows]
    hatches = [r["hatch"] for r in rows]

    xs = np.arange(len(bar_labels))
    bars = ax.bar(xs, means, yerr=errs, color=colors, edgecolor="white", capsize=3, width=0.72)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("white")

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, len(bar_labels) - 0.4)
    ax.margins(x=0.02)

    macro_avg = float(np.mean(means))
    _annotate_macro_avg(ax, macro_avg, f"Avg {macro_fmt.format(macro_avg)}")

    if show_legend:
        from matplotlib.patches import Patch
        used_families = list(dict.fromkeys(
            fam for _, lbl, fam, _ in models if lbl in all_data
        ))
        legend_handles = [Patch(facecolor=FAMILY_COLORS[f], label=f) for f in used_families]
        legend_handles.append(
            Patch(facecolor="gray", hatch="//", edgecolor="white", label="Reasoning ON")
        )
        if legend_loc == "upper center":
            fig.legend(
                handles=legend_handles, loc="upper center", ncol=4,
                frameon=True, fontsize=7.5, framealpha=0.92, edgecolor="#dddddd",
                bbox_to_anchor=(0.5, 0.99),
            )
        else:
            ax.legend(
                handles=legend_handles, loc=legend_loc, frameon=True, fontsize=7.5,
                ncol=2, framealpha=0.92, edgecolor="#dddddd",
            )

    top = 0.78 if (suptitle and show_legend and legend_loc == "upper center") else (
        0.90 if suptitle else 0.95
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.02 if legend_loc == "upper center" else 0.98)
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha="center", fontsize=7.5, color="#444444")

    style_axes(ax)
    fig.subplots_adjust(top=top, bottom=0.28, left=0.10, right=0.98)
    return fig


def _category_short(name: str) -> str:
    """Short category label for heatmaps."""
    short = {
        "AI and human romantic relationships": "AI romance",
        "AI moral patienthood": "AI moral pat.",
        "Global economy": "Global econ.",
        "Global politics and geopolitics": "Global politics",
        "Life and species": "Life/species",
        "Personal accomplishments": "Personal accom.",
        "Personal finances": "Personal finance",
        "Personal freedom and autonomy": "Personal freedom",
        "Religion and spirituality": "Religion",
        "United States economy": "US economy",
        "United States politics and policies": "US politics",
        "Wellbeing of humans": "Human wellbeing",
    }
    return short.get(name, name[:18])


def _model_legend_handles(points: list[dict]) -> list:
    """One legend handle per model point (color + marker + compact label)."""
    handles = []
    for pt in sorted(points, key=lambda p: p.get("table_name", p.get("short", ""))):
        handles.append(Line2D(
            [0], [0],
            marker="^" if pt.get("reasoning") == "on" else "o",
            color="w",
            markerfacecolor=pt.get("color", "#666666"),
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=pt.get("short", pt.get("label", "")),
        ))
    return handles


def _draw_binned_trend_panel(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    accent: str,
    show_bh_fails: bool = True,
    bh_points: list[dict] | None = None,
    title: str = "",
):
    """Clean binned-mean trend (no point cloud)."""
    centers, means, errs, counts = _binned_mean_auc(xs, ys, n_bins=10)
    ax.errorbar(
        centers, means, yerr=errs, fmt="o-", color=accent, lw=2.2,
        markersize=7, capsize=3, capthick=1.0, elinewidth=1.0, zorder=4,
        label="Mean AUC ± 95% CI",
    )
    if len(centers) >= 2:
        z = np.polyfit(centers, means, 1)
        xl = np.linspace(0, 1, 50)
        ax.plot(xl, np.polyval(z, xl), ls="--", color="#D55E00", lw=1.5, alpha=0.85)
    if show_bh_fails and bh_points:
        fails = [p for p in bh_points if not p["sig_bh"]]
        if fails:
            ax.scatter(
                [p["mono"] for p in fails], [p["auc"] for p in fails],
                facecolors="none", edgecolors=BH_FAIL_EDGE, linewidths=1.8,
                s=36, zorder=5, label=f"BH fail ($n$={len(fails)})",
            )
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else float("nan")
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=6)
    ax.text(
        0.03, 0.97, f"$r$ = {r:.2f}, $n$ = {len(xs)}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.9),
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.72, 1.0)
    ax.set_xlabel("Monotonicity rate", fontsize=9)
    ax.set_ylabel("Held-out test AUC", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=7, framealpha=0.92, edgecolor="#dddddd")
    style_axes(ax)


def _draw_tercile_box_panel(ax, xs: np.ndarray, ys: np.ndarray, *, accent: str, title: str = ""):
    """Box plot by monotonicity tercile."""
    edges = np.quantile(xs, [0, 1 / 3, 2 / 3, 1])
    tercile_data = []
    for t in range(3):
        lo, hi = edges[t], edges[t + 1]
        mask = (xs >= lo) & (xs < hi) if t < 2 else (xs >= lo) & (xs <= hi)
        tercile_data.append(ys[mask])
    ax.boxplot(
        tercile_data, positions=[0, 1, 2], widths=0.5, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor="#E8F4FC", edgecolor=accent),
        whiskerprops=dict(color=accent), capprops=dict(color=accent),
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Mid", "High"], fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=6)
    ax.set_ylabel("Held-out test AUC", fontsize=9)
    ax.set_ylim(0.72, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.3)
    style_axes(ax)


def load_all_per_set(results_dir, models):
    data = {}
    skipped = []
    for file_key, label, family, reasoning in models:
        d = load_analysis(results_dir, file_key)
        if d is None:
            skipped.append(file_key)
            continue
        data[label] = d["per_variation_set"]
    if skipped:
        print(f"  Skipped (no coherence JSON): {', '.join(skipped)}")
    return data


# ============================================================
# Fig 1: Headline monotonicity
# ============================================================
def fig1_headline_mono(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    fig = _model_bar_chart(
        all_data, models, get_mono,
        ylabel=(
            f"Strict preference monotonicity (%)\n"
            f"(comparison blocks with no step violations; {n_ladders} ladders)"
        ),
        ylim=(0, 100),
        macro_fmt="{:.1f}%",
        macro_scale=100.0,
        suptitle="Strict preference monotonicity",
        subtitle=(
            "Each bar = mean across ladders. "
            "A block is monotonic only if win-probability is strictly ordered at every adjacent tier."
        ),
    )
    fig.axes[0].axhline(50, color="gray", lw=0.6, ls="--", alpha=0.35)
    return fig


# ============================================================
# Fig 2: Isotonic R²
# ============================================================
def fig2_iso_r2(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    return _model_bar_chart(
        all_data, models, get_r2,
        ylabel=(
            r"Isotonic $R^2$ (bidirectional)"
            f"\n(max $R^2$ of increasing/decreasing fit; {n_ladders} ladders)"
        ),
        ylim=(0.5, 1.0),
        macro_fmt="{:.2f}",
        macro_scale=1.0,
        suptitle=r"Isotonic $R^2$ (bidirectional): coarse monotonic fit",
        subtitle=(
            "Measures how well some monotonic function of tier explains win-probability variance. "
            "Allows local wiggles; not equivalent to strict monotonicity (Fig. 1)."
        ),
    )


# ============================================================
# Fig 2b: Jonckheere–Terpstra significance rate
# ============================================================
def fig2b_jt_significance(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    return _model_bar_chart(
        all_data, models, get_jt,
        ylabel=(
            "JT significant rate (%)\n"
            f"(ordered trend at $\\alpha=0.05$; {n_ladders} ladders)"
        ),
        ylim=(55, 95),
        macro_fmt="{:.1f}%",
        macro_scale=100.0,
        suptitle="Jonckheere--Terpstra trend test",
        subtitle=(
            "Fraction of comparison blocks where later tiers stochastically dominate earlier tiers. "
            "Tests overall trend, not strict stepwise ordering."
        ),
        legend_loc="upper center",
    )


# ============================================================
# Fig 1b: Three-metric comparison (strict mono vs iso R² vs JT)
# ============================================================
# GPT-5.4 scale bars annotated in fig1b (file_key → panel formats).
_FIG1B_GPT54_ANNOT = {
    "gpt-54-nano": {"mono": True, "r2": True, "jt": True},
    "gpt-54-mini": {"mono": True, "r2": True, "jt": True},
    "gpt-54": {"mono": True, "r2": True, "jt": True},
    "gpt-54-mini-thinking": {"mono": True, "r2": True, "jt": False},
    "gpt-54-thinking": {"mono": True, "r2": True, "jt": False},
}
_FIG1B_ANNOT_FMT = {
    "mono": lambda v: f"{v:.1f}%",
    "r2": lambda v: f"{v:.3f}",
    "jt": lambda v: f"{v:.1f}%",
}


def _annotate_fig1b_bar_values(ax, xs, means, active, panel_key: str) -> None:
    """Label GPT-5.4 scale-comparison bars cited in §4.2."""
    for i, (fk, _, _, _) in enumerate(active):
        flags = _FIG1B_GPT54_ANNOT.get(fk)
        if not flags or not flags.get(panel_key):
            continue
        val = means[i]
        label = _FIG1B_ANNOT_FMT[panel_key](val)
        ax.text(
            xs[i], val, label,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            color="#1a1a1a", clip_on=False,
        )


def fig1b_metrics_triptych(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    """Stacked headline metrics (strict mono, iso R², JT) top to bottom."""
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if lbl in all_data]
    if not active:
        return plt.figure()

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), dpi=300, sharex=False)
    specs = [
        (get_mono, 100.0, (0, 100), "Strict mono (%)", "{:.1f}%", "mono",
         "No adjacent-tier violations"),
        (get_r2, 1.0, (0.5, 1.0), r"Iso $R^2$ (bi)", "{:.2f}", "r2",
         r"Best monotonic fit ($\uparrow$ or $\downarrow$)"),
        (get_jt, 100.0, (55, 95), "JT sig. (%)", "{:.1f}%", "jt",
         "Ordered trend in raw trials"),
    ]

    for ax, (fn, scale, ylim, ylabel, fmt, panel_key, short_desc) in zip(axes, specs):
        panel_rows = []
        for fk, label, family, reasoning in active:
            vals = [fn(rec) for rec in all_data[label]]
            panel_rows.append({
                "fk": fk,
                "label": label,
                "family": family,
                "reasoning": reasoning,
                "mean": float(np.mean(vals)) * scale,
                "bar_label": _compact_bar_label(fk),
                "color": get_color(family, reasoning, fk),
                "hatch": REASON_HATCH[reasoning],
            })
        order = _ascending_indices([r["mean"] for r in panel_rows])
        panel_rows = [panel_rows[i] for i in order]
        panel_active = [(r["fk"], r["label"], r["family"], r["reasoning"]) for r in panel_rows]
        means = [r["mean"] for r in panel_rows]
        colors = [r["color"] for r in panel_rows]
        hatches = [r["hatch"] for r in panel_rows]
        bar_labels = [r["bar_label"] for r in panel_rows]
        panel_xs = np.arange(len(panel_rows))

        bars = ax.bar(panel_xs, means, color=colors, edgecolor="white", width=0.72)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("white")

        ax.set_ylabel(ylabel, fontsize=9)
        y_lo, y_hi = ylim
        pad = (y_hi - y_lo) * 0.10
        ax.set_ylim(y_lo, y_hi + pad)
        ax.set_xlim(-0.6, len(panel_rows) - 0.4)
        macro = float(np.mean(means))
        _annotate_macro_avg(ax, macro, f"Avg {fmt.format(macro)}")
        _annotate_fig1b_bar_values(ax, panel_xs, means, panel_active, panel_key)
        ax.set_title(short_desc, fontsize=9, pad=6, loc="left")
        ax.set_xticks(panel_xs)
        ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=7.5)
        style_axes(ax)

    from matplotlib.patches import Patch
    used_families = list(dict.fromkeys(fam for _, _, fam, _ in active))
    legend_handles = [Patch(facecolor=FAMILY_COLORS[f], label=f) for f in used_families]
    legend_handles.append(
        Patch(facecolor="gray", hatch="//", edgecolor="white", label="Reasoning ON (*)")
    )
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=4,
        frameon=True, fontsize=8, framealpha=0.95, edgecolor="#dddddd",
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.suptitle(
        f"Three coherence metrics compared ({n_ladders} ladders; macro avg across models)",
        fontsize=11, y=1.01,
    )
    fig.text(
        0.5, 0.005,
        "Strict monotonicity is the stringent test; isotonic $R^2$ and JT measure coarse "
        "directional trend and can remain high despite frequent local violations.",
        ha="center", fontsize=7.5, color="#444444",
    )
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.38)
    return fig


# ============================================================
# Fig 3: Reasoning lift
# ============================================================
def _find_reasoning_pairs(models: list, all_data: dict) -> list[tuple[str, str, str]]:
    """Auto-detect (off_label, on_label, family) pairs from the model list.

    Matches models that share the same base key (stripping '-thinking') and
    differ only in reasoning mode.
    """
    off_models = {}
    on_models = {}
    for fk, label, family, reasoning in models:
        if label not in all_data:
            continue
        base = fk.removesuffix("-thinking") if fk.endswith("-thinking") else fk
        if reasoning == "on":
            on_models[base] = (label, family)
        else:
            off_models[base] = (label, family)

    pairs = []
    for base in off_models:
        if base in on_models:
            off_label, family = off_models[base]
            on_label, _ = on_models[base]
            pairs.append((off_label, on_label, family))
    return pairs


def fig3_reasoning_lift(all_data: dict, models: list) -> plt.Figure:
    pairs = _find_reasoning_pairs(models, all_data)
    if not pairs:
        return plt.figure()

    pair_rows = []
    for off_label, on_label, family in pairs:
        off_rates = [get_mono(r) for r in all_data[off_label]]
        on_rates = [get_mono(r) for r in all_data[on_label]]
        m_off, e_off = mean_ci_95(off_rates)
        m_on, e_on = mean_ci_95(on_rates)
        pair_rows.append({
            "off_label": off_label,
            "off_mean": m_off * 100,
            "off_err": e_off * 100,
            "on_mean": m_on * 100,
            "on_err": e_on * 100,
            "color_off": FAMILY_COLORS.get(family, "#999999"),
            "color_on": get_color(family, "on"),
        })
    order = _ascending_indices([r["off_mean"] for r in pair_rows])
    pair_rows = [pair_rows[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    width = 0.35
    xs = np.arange(len(pair_rows))

    off_means = [r["off_mean"] for r in pair_rows]
    on_means = [r["on_mean"] for r in pair_rows]
    off_errs = [r["off_err"] for r in pair_rows]
    on_errs = [r["on_err"] for r in pair_rows]
    colors_off = [r["color_off"] for r in pair_rows]
    colors_on = [r["color_on"] for r in pair_rows]

    ax.bar(xs - width/2, off_means, width, yerr=off_errs, color=colors_off,
           edgecolor="white", capsize=3, label="Reasoning OFF")
    ax.bar(xs + width/2, on_means, width, yerr=on_errs, color=colors_on,
           edgecolor="white", capsize=3, label="Reasoning ON", hatch="//")

    for i, (off, on) in enumerate(zip(off_means, on_means)):
        lift = on - off
        sign = "+" if lift >= 0 else ""
        ax.annotate(f"{sign}{lift:.1f}", xy=(i + width/2, on + on_errs[i] + 1),
                    ha="center", fontsize=8, color="black")

    pair_labels = [r["off_label"] for r in pair_rows]
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Monotonicity rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_axes(ax)
    plt.tight_layout()
    return fig


# ============================================================
# Fig 4: Per-set scatter
# ============================================================
def fig4_set_scatter(all_data: dict, models: list) -> plt.Figure:
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if lbl in all_data]
    if not active:
        return plt.figure()

    n = len(active)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8.8, 3.8 * nrows), dpi=300, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    axes_flat = axes.ravel()

    for i, (file_key, label, family, reasoning) in enumerate(active):
        ax = axes_flat[i]
        x = np.asarray([get_mono(rec) for rec in all_data[label]], dtype=float)
        y = np.asarray([get_r2(rec) for rec in all_data[label]], dtype=float)
        color = get_color(family, reasoning, file_key)
        marker = "^" if reasoning == "on" else "o"

        # Raw per-ladder points.
        ax.scatter(
            x, y, color=color, alpha=0.28, s=20, marker=marker, edgecolor="none"
        )
        # Mean reference point for quick reading.
        mx, my = float(np.mean(x)), float(np.mean(y))
        ax.scatter([mx], [my], color=color, s=85, marker=marker,
                   edgecolor="black", linewidth=0.7, zorder=4)
        ax.axvline(mx, color=color, lw=0.8, ls="--", alpha=0.45)
        ax.axhline(my, color=color, lw=0.8, ls="--", alpha=0.45)

        ax.set_title(label, fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.22)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.3, 1.02)
        style_axes(ax)

    # Hide any unused panel slots.
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared labels.
    fig.text(0.5, 0.02, "Per-ladder monotonicity rate (set-inferred direction)",
             ha="center")
    fig.text(0.02, 0.5, r"Per-ladder isotonic $R^2$ (bidirectional)",
             va="center", rotation="vertical")
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    return fig


# ============================================================
# Fig 4b: Aggregate centroid scatter (single panel)
# ============================================================
def fig4b_centroid_scatter(all_data: dict, models: list) -> plt.Figure:
    """Per-model centroids in (monotonicity, isotonic R²) space — no point cloud."""
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if lbl in all_data]
    if not active:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=300)
    centroids = []
    for file_key, label, family, reasoning in active:
        x = np.asarray([get_mono(rec) for rec in all_data[label]], dtype=float)
        y = np.asarray([get_r2(rec) for rec in all_data[label]], dtype=float)
        color = get_color(family, reasoning, file_key)
        marker = "^" if reasoning == "on" else "o"
        mx, my = float(np.mean(x)), float(np.mean(y))
        centroids.append({
            "x": mx, "y": my, "color": color, "marker": marker,
            "short": _compact_bar_label(file_key), "family": family, "reasoning": reasoning,
        })
        ax.scatter(
            mx, my, s=90, color=color, marker=marker,
            edgecolor="black", linewidth=0.6, zorder=4,
        )

    ax.set_xlabel("Strict preference monotonicity (model mean)", fontsize=9)
    ax.set_ylabel(r"Isotonic $R^2$ bidirectional (model mean)", fontsize=9)
    ax.set_xlim(0, 0.88)
    ax.set_ylim(0.82, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.28)

    all_x = np.array([c["x"] for c in centroids])
    all_y = np.array([c["y"] for c in centroids])
    mx, my = float(np.mean(all_x)), float(np.mean(all_y))
    ax.axvline(mx, color="black", lw=0.9, ls=":", alpha=0.45)
    ax.axhline(my, color="black", lw=0.9, ls=":", alpha=0.45)
    ax.text(
        0.03, 0.97,
        f"Macro avg: mono = {mx*100:.0f}%,  iso $R^2$ = {my:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
    )

    legend_pts = [{"color": c["color"], "reasoning": c["reasoning"], "short": c["short"]} for c in centroids]
    ax.legend(
        handles=_model_legend_handles(legend_pts),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, fontsize=7, framealpha=0.95, edgecolor="#dddddd",
        title="Model", title_fontsize=8,
    )
    style_axes(ax)
    fig.subplots_adjust(right=0.72)
    return fig


# ============================================================
# Fig 5: Incoherence
# ============================================================
def fig5_incoherence(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), dpi=300,
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    rows = []
    for file_key, label, family, reasoning in models:
        if label not in all_data:
            continue
        rates = [get_mono(rec) for rec in all_data[label]]
        rows.append({
            "label": label,
            "rates": rates,
            "color": get_color(family, reasoning, file_key),
            "mean": float(np.mean(rates)),
        })
    order = _ascending_indices([r["mean"] for r in rows])
    rows = [rows[i] for i in order]
    labels = [r["label"] for r in rows]
    data_lists = [r["rates"] for r in rows]
    colors = [r["color"] for r in rows]

    bp = axL.boxplot(data_lists, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black"),
                     whiskerprops=dict(color="gray"),
                     capprops=dict(color="gray"),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="red",
                                     markeredgecolor="none", alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")

    axL.set_xticks(range(1, len(labels) + 1))
    axL.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axL.set_ylim(0, 1)
    axL.set_ylabel("Per-ladder monotonicity rate")
    axL.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.4)
    style_axes(axL)

    n_below = [sum(1 for v in dl if v < 0.5) for dl in data_lists]
    axR.bar(range(len(labels)), n_below, color=colors, edgecolor="white")
    axR.set_xticks(range(len(labels)))
    axR.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axR.set_ylabel(f"Ladders with mono < 0.5 (of {n_ladders})")
    for i, n in enumerate(n_below):
        axR.text(i, n + 1, str(n), ha="center", fontsize=8)
    style_axes(axR)
    plt.tight_layout()
    return fig


# ============================================================
# Fig 6: Heatmap
# ============================================================
def fig6_heatmap(all_data: dict, models: list) -> plt.Figure:
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if lbl in all_data]
    first_label = active[0][1]
    set_ids = [rec["variation_id"] for rec in all_data[first_label]]

    M = np.zeros((len(set_ids), len(active)))
    for j, (_, label, _, _) in enumerate(active):
        for i, rec in enumerate(all_data[label]):
            M[i, j] = get_mono(rec)

    row_means = M.mean(axis=1)
    order = np.argsort(-row_means)
    M = M[order]
    sorted_ids = [set_ids[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 14), dpi=300)
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(active)))
    ax.set_xticklabels([lbl for _, lbl, _, _ in active], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(0, len(sorted_ids), 5))
    ax.set_yticklabels([sorted_ids[i][:28] for i in range(0, len(sorted_ids), 5)], fontsize=5)
    ax.set_ylabel("Valence ladders (sorted by cross-model mean)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Monotonicity rate")
    plt.tight_layout()
    return fig


# ============================================================
# Fig 7 / 7b: Category breakdown (reasoning OFF vs ON)
# ============================================================
CATEGORY_HEATMAP_CMAP = "RdYlGn"
CATEGORY_HEATMAP_VMIN = 0
CATEGORY_HEATMAP_VMAX = 100


def _build_category_matrix(
    all_data: dict,
    models_subset: list,
    categories: list[str],
) -> np.ndarray:
    """Rows = models, cols = categories; values = mean strict mono (%)."""
    M = np.zeros((len(models_subset), len(categories)))
    for i, (_, label, _, _) in enumerate(models_subset):
        cat_data: dict[str, list] = defaultdict(list)
        for rec in all_data[label]:
            cat_data[rec["category"]].append(get_mono(rec))
        for j, cat in enumerate(categories):
            M[i, j] = np.mean(cat_data[cat]) * 100 if cat_data[cat] else 0.0
    return M


def _draw_category_heatmap(
    all_data: dict,
    models_subset: list,
    *,
    ylabel: str,
) -> plt.Figure:
    """Category heatmap with per-column fleet means in a bottom summary row."""
    if not models_subset:
        return plt.figure()

    first_label = models_subset[0][1]
    categories = sorted(set(rec["category"] for rec in all_data[first_label]))
    cat_labels = [_category_short(c) for c in categories]
    row_labels = [_compact_bar_label(fk) for fk, _, _, _ in models_subset]
    M = _build_category_matrix(all_data, models_subset, categories)
    col_means = np.mean(M, axis=0)
    M_display = np.vstack([M, col_means.reshape(1, -1)])

    n_model_rows = len(models_subset)
    n_display_rows = n_model_rows + 1
    fig_h = max(4.0, 0.34 * n_display_rows + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_h), dpi=300)

    im = ax.imshow(
        M_display, aspect="auto", cmap=CATEGORY_HEATMAP_CMAP,
        vmin=CATEGORY_HEATMAP_VMIN, vmax=CATEGORY_HEATMAP_VMAX,
    )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_display_rows))
    ax.set_yticklabels(row_labels + ["Column mean"], fontsize=8)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("Category", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    ax.axhline(n_model_rows - 0.5, color="white", linewidth=2.0)
    ax.axhline(n_model_rows - 0.5, color="0.25", linewidth=0.8)

    for i in range(M_display.shape[0]):
        for j in range(M_display.shape[1]):
            val = M_display[i, j]
            txt_color = "white" if val < 35 or val > 75 else "black"
            weight = "bold" if i == n_model_rows else "normal"
            ax.text(
                j, i, f"{val:.0f}", ha="center", va="center",
                fontsize=7, color=txt_color, fontweight=weight,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Strict monotonicity (%)", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.96)
    return fig


def fig7_category_breakdown_off(all_data: dict, models: list) -> plt.Figure:
    """Heatmap: monotonicity by category (reasoning-off models)."""
    models_off = [(fk, lbl, fam, r) for fk, lbl, fam, r in models
                  if r == "off" and lbl in all_data]
    return _draw_category_heatmap(
        all_data, models_off, ylabel="Model (reasoning off)",
    )


def fig7b_category_breakdown_on(all_data: dict, models: list) -> plt.Figure:
    """Heatmap: monotonicity by category (reasoning-on models)."""
    models_on = [(fk, lbl, fam, r) for fk, lbl, fam, r in models
                 if r == "on" and lbl in all_data]
    return _draw_category_heatmap(
        all_data, models_on, ylabel="Model (reasoning on)",
    )


# ============================================================
# Fig 8: Ladder shape curves
# ============================================================
def _find_raw_results(results_dir: pathlib.Path, model_key: str) -> list[pathlib.Path]:
    """Collect raw results.json files under the model-scoped layout.

    Readable per-ladder dirs are phase6b_ladder_<ladder_id>; the legacy hash
    pattern (phase6b_variations_prune_*) is matched too for any unmigrated dirs.
    """
    model_root = results_dir / model_key
    if not model_root.exists():
        return []
    found = sorted(model_root.glob("phase6b_ladder_*/results.json"))
    found += sorted(model_root.glob("phase6b_variations_prune_*/results.json"))
    return found


def fig8_ladder_shapes(all_data: dict, models: list, results_dir: pathlib.Path) -> plt.Figure:
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if lbl in all_data]
    n = len(active)
    ncols = min(4, n) if n else 4
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), dpi=300, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    for idx, (file_key, label, family, reasoning) in enumerate(active):
        ax = axes_flat[idx]
        color = get_color(family, reasoning, file_key)

        tier_probs_all = []
        raw_files = _find_raw_results(results_dir, file_key)
        for path in raw_files:
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            prefs = d.get("preferences", [])
            if not prefs:
                continue

            tier_wins = {}
            tier_total = {}
            valence = prefs[0].get("outcome_a", {}).get("valence", "positive")
            for p in prefs:
                oa, ob = p.get("outcome_a", {}), p.get("outcome_b", {})
                tier_a, tier_b = oa.get("tier"), ob.get("tier")
                ca = p.get("count_prefer_a", 0)
                cb = p.get("count_prefer_b", 0)
                total = ca + cb
                if total == 0:
                    continue
                if tier_a is not None:
                    tier_wins[tier_a] = tier_wins.get(tier_a, 0) + ca
                    tier_total[tier_a] = tier_total.get(tier_a, 0) + total
                if tier_b is not None:
                    tier_wins[tier_b] = tier_wins.get(tier_b, 0) + cb
                    tier_total[tier_b] = tier_total.get(tier_b, 0) + total

            if len(tier_total) < N_TIERS:
                continue
            tiers_sorted = sorted(tier_total.keys())
            probs = np.array([tier_wins[t] / tier_total[t] for t in tiers_sorted])
            if valence == "negative":
                probs = probs[::-1]
            tier_probs_all.append(probs)

        if not tier_probs_all:
            ax.set_title(f"{label} (no data)", fontsize=9)
            continue

        mat = np.array(tier_probs_all)
        means = np.nanmean(mat, axis=0)
        sds = np.nanstd(mat, axis=0)
        x = np.arange(1, N_TIERS + 1)
        ax.fill_between(x, means - sds, means + sds, color=color, alpha=0.2)
        ax.plot(x, means, color=color, lw=2, marker="o", markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{i}" for i in x], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=9)
        style_axes(ax)
        if idx % ncols == 0:
            ax.set_ylabel("P(prefer tier)")

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    return fig


# ============================================================
# Predictive utility loaders
# ============================================================
def _float_csv(val, default=float("nan")):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_pred_util_per_model(results_dir: pathlib.Path) -> dict[str, dict]:
    """Load per-model predictive-utility summaries keyed by model_key."""
    out: dict[str, dict] = {}
    if not results_dir.exists():
        return out
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        path = d / "pred_utility_test" / "per_model_pred_util.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            out[d.name] = rows[0]
    return out


def load_pred_util_per_set(results_dir: pathlib.Path, model_key: str) -> list[dict]:
    """Per-ladder predictive-utility rows for one model."""
    path = results_dir / model_key / "pred_utility_test" / "per_set_pred_util.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def coherence_aggregate_mono(results_dir: pathlib.Path, model_key: str) -> float:
    """Overall monotonicity rate from coherence JSON, or NaN if missing."""
    d = load_analysis(results_dir, model_key)
    if not d:
        return float("nan")
    overall = (d.get("aggregate") or {}).get("overall") or {}
    return _float_csv(overall.get("monotonicity_rate"))


def _short_model_label(model_key: str, max_len: int = 18) -> str:
    """Compact label for scatter annotations."""
    s = model_key.replace("-openrouter", "").replace("-thinking", "*")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _find_pred_util_reasoning_pairs(
    pred_util: dict[str, dict],
) -> list[tuple[str, str, str]]:
    """(off_key, on_key, family) for models with both reasoning modes."""
    off_by_base: dict[str, tuple[str, str]] = {}
    on_by_base: dict[str, str] = {}
    for key in pred_util:
        base = key.removesuffix("-thinking") if key.endswith("-thinking") else key
        family = infer_family(key)
        if infer_reasoning(key) == "on":
            on_by_base[base] = key
        else:
            off_by_base[base] = (key, family)
    pairs = []
    for base, (off_key, family) in off_by_base.items():
        if base in on_by_base:
            pairs.append((off_key, on_by_base[base], family))
    return pairs


# ============================================================
# Fig 9: Predictive AUC vs coherence (model-level)
# ============================================================
def fig9_pred_util_coherence_scatter(
    results_dir: pathlib.Path,
    pred_util: dict[str, dict],
) -> plt.Figure | None:
    points = []
    for key, row in pred_util.items():
        mono = coherence_aggregate_mono(results_dir, key)
        auc = _float_csv(row.get("mean_test_auc"))
        if math.isnan(mono) or math.isnan(auc):
            continue
        family = infer_family(key)
        reasoning = infer_reasoning(key)
        points.append({
            "key": key,
            "short": _compact_bar_label(key),
            "family": family,
            "reasoning": reasoning,
            "color": get_color(family, reasoning, key),
            "mono": mono,
            "auc": auc,
            "bh_pct": _float_csv(row.get("frac_sig_at_bh_fdr05")) * 100.0,
        })
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=300)
    for pt in points:
        marker = "^" if pt["reasoning"] == "on" else "o"
        ax.scatter(
            pt["mono"] * 100, pt["auc"],
            s=85, color=pt["color"], marker=marker,
            edgecolor="black", linewidth=0.5, zorder=3,
        )

    # Highlight GLM Base dissociation: high BH pass despite low monotonicity.
    for pt in points:
        if pt["key"] == "glm-45-base-logprobs":
            ax.annotate(
                f"{pt['bh_pct']:.0f}% BH pass",
                xy=(pt["mono"] * 100, pt["auc"]),
                xytext=(14, 10),
                textcoords="offset points",
                fontsize=7.5,
                ha="left",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92),
                zorder=6,
            )
            break

    xs = [p["mono"] * 100 for p in points]
    ys = [p["auc"] for p in points]
    if len(points) >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(
            0.03, 0.03, f"$r$ = {r:.2f}  ($n$ = {len(points)} models)",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.92),
        )

    ax.set_xlabel("Strict monotonicity (model mean, %)", fontsize=9)
    ax.set_ylabel("Predictive utility: mean held-out AUC", fontsize=9)
    ax.set_xlim(0, 88)
    ax.set_ylim(0.76, 0.98)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.28)
    style_axes(ax)

    ax.legend(
        handles=_model_legend_handles(points),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, fontsize=7, framealpha=0.95, edgecolor="#dddddd",
        title="Model (* = reasoning on)", title_fontsize=8,
    )
    fig.subplots_adjust(right=0.68)
    return fig


# ============================================================
# Fig 10: Reasoning lift on predictive utility (paired bars)
# ============================================================
def fig10_pred_util_reasoning_lift(pred_util: dict[str, dict]) -> plt.Figure | None:
    pairs = _find_pred_util_reasoning_pairs(pred_util)
    if not pairs:
        return None

    pair_rows = []
    for off_key, on_key, family in pairs:
        off_auc = _float_csv(pred_util[off_key].get("mean_test_auc"))
        on_auc = _float_csv(pred_util[on_key].get("mean_test_auc"))
        pair_rows.append({
            "off_key": off_key,
            "off_auc": off_auc,
            "on_auc": on_auc,
            "color_off": FAMILY_COLORS.get(family, "#999999"),
            "color_on": get_color(family, "on"),
        })
    order = _ascending_indices([r["off_auc"] for r in pair_rows])
    pair_rows = [pair_rows[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    width = 0.35
    xs = np.arange(len(pair_rows))

    off_aucs = [r["off_auc"] for r in pair_rows]
    on_aucs = [r["on_auc"] for r in pair_rows]
    colors_off = [r["color_off"] for r in pair_rows]
    colors_on = [r["color_on"] for r in pair_rows]

    ax.bar(xs - width / 2, off_aucs, width, color=colors_off,
           edgecolor="white", label="Reasoning OFF")
    ax.bar(xs + width / 2, on_aucs, width, color=colors_on,
           edgecolor="white", label="Reasoning ON", hatch="//")

    for i, (off, on) in enumerate(zip(off_aucs, on_aucs)):
        lift_pp = (on - off) * 100
        sign = "+" if lift_pp >= 0 else ""
        ymax = max(off, on)
        ax.annotate(
            f"{sign}{lift_pp:.1f}pp",
            xy=(i + width / 2, ymax + 0.008),
            ha="center", fontsize=8, color="black",
        )

    pair_labels = [make_display_label(r["off_key"]) for r in pair_rows]
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean held-out test AUC")
    ax.set_ylim(0.82, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_axes(ax)
    plt.tight_layout()
    return fig


# ============================================================
# Fig 11–12: Per-ladder predictive AUC vs monotonicity
# ============================================================


def _norm_variation_id(vid: str) -> str:
    return vid.replace(" ", "_") if vid else vid


def _parse_sig_bh(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def _load_pred_util_ladder_points(
    results_dir: pathlib.Path,
    models: list,
) -> list[dict]:
    """Per-ladder rows with monotonicity, held-out AUC, BH flag, and model metadata."""
    points: list[dict] = []
    for file_key, label, family, reasoning in models:
        per_set = load_pred_util_per_set(results_dir, file_key)
        analysis = load_analysis(results_dir, file_key)
        if not per_set or not analysis:
            continue
        mono_by_id = {}
        for rec in analysis.get("per_variation_set") or []:
            if not isinstance(rec, dict):
                continue
            vid = _norm_variation_id(rec.get("variation_id", ""))
            mono_by_id[vid] = get_mono(rec)
        for row in per_set:
            vid = row.get("variation_id", "")
            mono = mono_by_id.get(vid)
            auc = _float_csv(row.get("test_auc"))
            if mono is None or math.isnan(auc):
                continue
            points.append({
                "mono": float(mono),
                "auc": float(auc),
                "sig_bh": _parse_sig_bh(row.get("sig_bh_fdr05")),
                "family": family,
                "reasoning": reasoning,
                "label": label,
                "model_key": file_key,
            })
    return points


def _binned_mean_auc(
    xs: np.ndarray,
    ys: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width bins on monotonicity → center, mean AUC, 95% CI half-width, count."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, means, errs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (xs >= lo) & (xs < hi)
        else:
            mask = (xs >= lo) & (xs <= hi)
        sub = ys[mask]
        if len(sub) < 2:
            continue
        m, err = mean_ci_95(sub)
        centers.append(0.5 * (lo + hi))
        means.append(m)
        errs.append(err)
        counts.append(int(mask.sum()))
    return (
        np.asarray(centers),
        np.asarray(means),
        np.asarray(errs),
        np.asarray(counts),
    )


def fig11_pred_util_per_ladder_scatter(
    results_dir: pathlib.Path,
    models: list,
) -> plt.Figure | None:
    """Ladder-level predictive utility vs monotonicity (binned trend + tercile boxes)."""
    points = _load_pred_util_ladder_points(results_dir, models)
    if len(points) < 5:
        return None

    xs = np.asarray([p["mono"] for p in points], dtype=float)
    ys = np.asarray([p["auc"] for p in points], dtype=float)
    n_fail = sum(1 for p in points if not p["sig_bh"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    _draw_binned_trend_panel(
        ax1, xs, ys, accent="#0072B2", show_bh_fails=True, bh_points=points,
        title=f"All models ($n$ = {len(points)}, BH fail = {n_fail})",
    )
    _draw_tercile_box_panel(
        ax2, xs, ys, accent="#0072B2",
        title="By monotonicity tercile",
    )
    fig.suptitle(
        "Coherence and predictive utility align at the ladder level",
        fontsize=11, y=1.02,
    )
    fig.subplots_adjust(top=0.88, wspace=0.28)
    return fig


def fig12_pred_util_reasoning_split(
    results_dir: pathlib.Path,
    models: list,
) -> plt.Figure | None:
    """Reasoning OFF vs ON: binned trend and tercile boxes side by side."""
    points = _load_pred_util_ladder_points(results_dir, models)
    off_pts = [p for p in points if p["reasoning"] == "off"]
    on_pts = [p for p in points if p["reasoning"] == "on"]
    if len(off_pts) < 5 or len(on_pts) < 5:
        return None

    off_x = np.asarray([p["mono"] for p in off_pts], dtype=float)
    off_y = np.asarray([p["auc"] for p in off_pts], dtype=float)
    on_x = np.asarray([p["mono"] for p in on_pts], dtype=float)
    on_y = np.asarray([p["auc"] for p in on_pts], dtype=float)

    off_fail = sum(1 for p in off_pts if not p["sig_bh"])
    on_fail = sum(1 for p in on_pts if not p["sig_bh"])
    off_auc = float(np.mean(off_y))
    on_auc = float(np.mean(on_y))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    _draw_binned_trend_panel(
        axes[0, 0], off_x, off_y, accent="#5C6BC0", show_bh_fails=True, bh_points=off_pts,
        title=f"Reasoning OFF ($n$ = {len(off_pts)}, BH fail = {off_fail})",
    )
    _draw_binned_trend_panel(
        axes[0, 1], on_x, on_y, accent="#0072B2", show_bh_fails=True, bh_points=on_pts,
        title=f"Reasoning ON ($n$ = {len(on_pts)}, BH fail = {on_fail})",
    )
    _draw_tercile_box_panel(axes[1, 0], off_x, off_y, accent="#5C6BC0", title="OFF — tercile")
    _draw_tercile_box_panel(axes[1, 1], on_x, on_y, accent="#0072B2", title="ON — tercile")

    fig.suptitle(
        f"Predictive utility vs coherence: reasoning OFF vs ON  "
        f"(mean AUC {off_auc:.3f} → {on_auc:.3f})",
        fontsize=11, y=1.01,
    )
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.28)
    return fig


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Generate AIES 2026 paper figures")
    ap.add_argument(
        "--model", action="append", default=None,
        help="Model key to include (repeatable). If omitted, auto-discovers "
             "all models under --results-dir that have a coherence JSON.",
    )
    ap.add_argument("--results-dir",
                    default=str(MODEL_RUNS_OUTPUT_DIR),
                    help="Results root containing model-scoped subdirs "
                         "(default: outputs/04_model_runs/)")
    ap.add_argument("--output-dir",
                    default=str(REPORT_OUTPUTS_DIR / "figures"),
                    help="Directory for output figures (default: outputs/06_figures_tables/figures/)")
    ap.add_argument("--tables-dir",
                    default=str(REPORT_OUTPUTS_DIR / "tables"),
                    help="Directory for output tables (default: outputs/06_figures_tables/tables/)")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (PARAM_VAR / results_dir).resolve()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = pathlib.Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_keys = args.model
    else:
        model_keys = discover_models(results_dir)
        if not model_keys:
            print(f"ERROR: no models with coherence JSONs found in {results_dir}")
            return

    models = build_model_tuples(model_keys)

    print(f"Results dir:  {results_dir}")
    print(f"Figures dir:  {out_dir}")
    print(f"Tables dir:   {tables_dir}")
    print(f"Models:       {', '.join(k for k, *_ in models)}")

    print("\nLoading coherence data for available models...")
    all_data = load_all_per_set(results_dir, models)

    if not all_data:
        print("ERROR: no coherence data found. Run analyze_7tier_coherence.py first.")
        return

    first_label = next(iter(all_data))
    n_ladders = len(all_data[first_label])
    active_models = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if lbl in all_data]

    metric_rows, macro = collect_coherence_metrics(all_data, active_models)
    print_coherence_metrics_table(metric_rows, macro)

    table_base = tables_dir / "table_coherence_metrics"
    tex_path, csv_path = write_coherence_metrics_table(metric_rows, macro, table_base)
    print(f"\nWrote {tex_path.name} and {csv_path.name} to {tables_dir}")

    targets = [
        ("fig1_headline_monotonicity", fig1_headline_mono(all_data, models, n_ladders)),
        ("fig2_iso_r2_by_variant", fig2_iso_r2(all_data, models, n_ladders)),
        ("fig2b_jt_significance", fig2b_jt_significance(all_data, models, n_ladders)),
        ("fig1b_metrics_triptych", fig1b_metrics_triptych(all_data, models, n_ladders)),
        ("fig3_reasoning_lift", fig3_reasoning_lift(all_data, models)),
        ("fig4_set_scatter", fig4_set_scatter(all_data, models)),
        ("fig4b_centroid_scatter", fig4b_centroid_scatter(all_data, models)),
        ("fig5_incoherence", fig5_incoherence(all_data, models, n_ladders)),
        ("fig6_coherence_heatmap", fig6_heatmap(all_data, models)),
        ("fig7_category_breakdown_off", fig7_category_breakdown_off(all_data, models)),
        ("fig7b_category_breakdown_on", fig7b_category_breakdown_on(all_data, models)),
        ("fig8_ladder_shapes", fig8_ladder_shapes(all_data, models, results_dir)),
    ]

    pred_util = load_pred_util_per_model(results_dir)
    if pred_util:
        print(f"\nPredictive utility: {len(pred_util)} models with per_model_pred_util.csv")
        pu_rows, pu_macro = collect_pred_util_metrics(results_dir, pred_util, active_models)
        pu_base = tables_dir / "table_pred_util_metrics"
        pu_tex, pu_csv = write_pred_util_metrics_table(pu_rows, pu_macro, pu_base)
        print(
            f"  Pooled BH pass: {pu_macro['pooled_bh_pass']}/{pu_macro['pooled_n']} "
            f"({pu_macro['pooled_bh_pct']:.1f}%)  |  "
            f"Macro mean AUC: {pu_macro['mean_auc']:.3f}  |  "
            f"Model-level r: {pu_macro['model_r_mono_auc']:.2f}"
        )
        print(f"Wrote {pu_tex.name} and {pu_csv.name} to {tables_dir}")
        fig9 = fig9_pred_util_coherence_scatter(results_dir, pred_util)
        if fig9 is not None:
            targets.append(("fig9_pred_util_vs_coherence", fig9))
        fig10 = fig10_pred_util_reasoning_lift(pred_util)
        if fig10 is not None:
            targets.append(("fig10_pred_util_reasoning_lift", fig10))
        fig11 = fig11_pred_util_per_ladder_scatter(results_dir, models)
        if fig11 is not None:
            targets.append(("fig11_pred_util_per_ladder_scatter", fig11))
        fig12 = fig12_pred_util_reasoning_split(results_dir, models)
        if fig12 is not None:
            targets.append(("fig12_pred_util_reasoning_split", fig12))
    else:
        print("\nSkipping fig9–fig12 (no pred_utility_test/ outputs found).")

    print(f"\nGenerating {len(targets)} figures...")
    for name, fig in targets:
        if fig is None:
            continue
        for ext in ("pdf", "png"):
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            print(f"  {path.name} ({path.stat().st_size // 1024}KB)")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
