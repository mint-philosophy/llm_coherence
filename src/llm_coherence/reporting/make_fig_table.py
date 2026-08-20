"""
Generate paper figures and tables from coherence analysis JSONs.

Port of ``parametric_variations/phase6b_experiments/make_fig_table.py``.

Reads per-model coherence JSONs (from analyze_7tier_coherence.py), predictive-
utility CSVs (from predictive_utility.py), within-ladder summary JSONs
(from run_within_ladder_experiment.py --analyze), and raw results files, then
produces fig1--fig12 as PDF+PNG and tables (coherence metrics, within-ladder
accuracy, combined headline, model configs, cost log, category matrix; fig7c).

Usage:
    python -m llm_coherence.reporting.make_fig_table
    python scripts/06_reporting/13_make_fig_table.py
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

# Publication typography.  The methodology pipeline and valence-ladder
# schematics use the same Helvetica/Arial sans-serif stack.  Declaring it here
# prevents Matplotlib from silently falling back to its default DejaVu Sans and
# keeps text editable/searchable in vector exports.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "regular",
    "axes.titleweight": "regular",
    "axes.labelweight": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

from llm_coherence.config import (
    MODEL_CONFIGS,
    ModelConfig,
    model_key_from_results_folder,
    resolve_model_results_dir,
)
from llm_coherence.paths import (
    COHERENCE_TEST_SUBDIR,
    FIGURES_OUTPUT_DIR,
    LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR,
    REPO_ROOT,
    TABLES_OUTPUT_DIR,
    VALIDATION_DATA_DIR,
    phase6b_coherence_json_path,
    resolve_repo_path,
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

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
    "kimi":      "Moonshot",
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
    "Moonshot":  "#000000",
}

# Per-model tints where a family default would collide or need emphasis.
MODEL_COLOR_OVERRIDES = {
    # A stable color identifies the base model; marker shape or hatching—not
    # color—identifies reasoning mode.  This keeps matched off/on variants
    # visually paired while distinguishing model sizes and subfamilies.
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
    "qwen-37-flash": "#8C564B",
    "qwen-37-flash-thinking": "#8C564B",
    "qwen-37-flash-openrouter": "#8C564B",
    "qwen-37-flash-openrouter-thinking": "#8C564B",
    "kimi-k2": "#000000",
    "kimi-k2-openrouter": "#000000",
    "kimi-k2-thinking": "#000000",
    "kimi-k2-openrouter-thinking": "#000000",
    "kimi-k3-openrouter-thinking-low": "#000000",
    "kimi-k3-openrouter-thinking-medium": "#000000",
    "kimi-k3-openrouter-thinking-high": "#000000",
}

REASON_HATCH = {"on": "//", "off": None}

BH_PASS_FACE = "#888888"
BH_FAIL_EDGE = "#C0392B"

_BRAND_TOKENS = {
    "gpt": "GPT",
    "glm": "GLM",
    "llama": "Llama",
    "opus": "Opus",
    "nemotron": "Nemotron",
    "ministral": "Ministral",
    "mistral": "Mistral",
    "magistral": "Magistral",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "gemini": "Gemini",
    "claude": "Claude",
    "kimi": "Kimi",
}

_VARIANT_SUFFIX_RE = re.compile(
    r"-(?:thinking(?:-(?:minimal|low|medium|high|max|xhigh))?|"
    r"with-reasoning|justification(?:-v\d+)?|logprobs)$"
)

# Canonical publication base names (before ``(reasoning on/off)`` suffix).
PAPER_MODEL_BASE_NAMES: dict[str, str] = {
    "gpt-56-sol": "GPT-5.6 Sol",
    # Retained only to label any artifacts produced before the explicit Sol key.
    "gpt-56": "GPT-5.6 Sol",
    "gpt-56-terra": "GPT-5.6 Terra",
    "gpt-56-luna": "GPT-5.6 Luna",
    "gpt-54-nano": "GPT-5.4-Nano",
    "gpt-54-mini": "GPT-5.4-Mini",
    "gpt-54": "GPT-5.4",
    "opus-46": "Opus-4.6",
    "nemotron-3-super": "Nemotron-3-Super",
    "glm-45-hybrid": "GLM-4.5-Hybrid",
    "glm-45-base": "GLM-4.5 Base",
    "glm-45-base-logprobs": "GLM-4.5 Base (baseline)",
    "ministral-3b-2512": "Ministral-3B-2512",
    "mistral-small-2603": "Mistral-Small-2603",
    "llama-31-8b-instruct": "Llama-3.1-8B-Instruct",
    "kimi-k2": "Kimi-K2",
    "kimi-k3": "Kimi-K3",
    "qwen-37-flash": "Qwen3.7 Flash",
    "qwen-37-max": "Qwen3.7 Max",
}


def _strip_variant_suffixes(model_key: str) -> str:
    """Model key with routing / reasoning / baseline suffixes removed."""
    key = model_key.replace("-openrouter", "")
    while True:
        m = _VARIANT_SUFFIX_RE.search(key)
        if not m:
            break
        key = key[: m.start()]
    return key


def _friendly_from_model_key(key: str) -> str:
    """Human-readable base name derived from a model key (no hardcoded per-model map)."""
    tokens = key.split("-")
    if not tokens:
        return key

    out: list[str] = []
    i = 0
    brand = _BRAND_TOKENS.get(tokens[0].lower())
    if brand:
        out.append(brand)
        i = 1

    if i < len(tokens) and re.fullmatch(r"\d{2}", tokens[i]):
        out.append(f"{tokens[i][0]}.{tokens[i][1]}")
        i += 1
    elif i < len(tokens) and re.fullmatch(r"\d+", tokens[i]):
        out.append(tokens[i])
        i += 1

    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"\d+b", tok, re.I):
            out.append(tok[:-1] + "B")
        elif tok.isdigit():
            out.append(tok)
        elif tok in {"mini", "nano", "small", "hybrid", "base", "instruct", "super"}:
            out.append(tok.capitalize())
        else:
            out.append(tok.capitalize())
        i += 1

    if out and out[0] in {"GPT", "GLM", "Opus", "Nemotron"}:
        head = f"{out[0]}-{out[1]}" if len(out) > 1 else out[0]
        tail = " ".join(out[2:])
        return f"{head} {tail}".strip()
    return " ".join(out)


def base_model_label(model_key: str) -> str:
    """Publication base name without reasoning mode."""
    if model_key in PAPER_MODEL_BASE_NAMES:
        return PAPER_MODEL_BASE_NAMES[model_key]
    stripped = _strip_variant_suffixes(model_key)
    if stripped in PAPER_MODEL_BASE_NAMES:
        return PAPER_MODEL_BASE_NAMES[stripped]
    return _friendly_from_model_key(stripped)


def infer_reasoning(model_key: str) -> str:
    """Return 'on' or 'off' from model key and config metadata."""
    key_lower = model_key.lower()
    if any(tag in key_lower for tag in ("-thinking", "with-reasoning")):
        return "on"

    cfg = MODEL_CONFIGS.get(model_key)
    if cfg:
        artifact = getattr(cfg, "reasoning_artifact_type", "none")
        if artifact in {"raw_cot", "summary", "prose_justification"}:
            if "justification" not in key_lower:
                return "on"
        extra = cfg.extra_body or {}
        reasoning = extra.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort not in (None, "", "none"):
                return "on"
        if extra.get("thinking"):
            return "on"
        if extra.get("reasoning_effort") not in (None, "", "none"):
            return "on"
    return "off"


def reasoning_mode_label(reasoning: str) -> str:
    return "reasoning on" if reasoning == "on" else "reasoning off"


def paper_model_label(model_key: str, reasoning: str | None = None) -> str:
    """Full publication label: base model + explicit reasoning mode."""
    if reasoning is None:
        reasoning = infer_reasoning(model_key)
    base = base_model_label(model_key)
    return f"{base} ({reasoning_mode_label(reasoning)})"


def paper_model_panel_title(model_key: str, reasoning: str | None = None) -> str:
    return paper_model_label(model_key, reasoning)


def make_display_label(model_key: str) -> str:
    return paper_model_label(model_key)


def _model_sort_key(model_key: str) -> tuple:
    """Stable ordering: family, base key, reasoning off before on."""
    base = _strip_variant_suffixes(model_key)
    return (infer_family(model_key), base, 0 if infer_reasoning(model_key) == "off" else 1, model_key)


def infer_family(model_key: str) -> str:
    key_lower = model_key.lower()
    for keyword, family in FAMILY_KEYWORDS.items():
        if keyword in key_lower:
            return family
    return "Other"


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
    known = set(MODEL_CONFIGS.keys())
    keys = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        model_key = model_key_from_results_folder(d.name, known)
        run_dir = resolve_model_results_dir(model_key, results_dir) / "ladder_vs_comparison_statements"
        coherence = phase6b_coherence_json_path(model_key, results_dir)
        if coherence.is_file():
            keys.append(model_key)
            continue
        if run_dir.is_dir():
            legacy = run_dir / f"phase6b_coherence_{model_key}.json"
            if legacy.is_file():
                keys.append(model_key)
                continue
            if list(run_dir.glob(f"phase6b_coherence_{d.name}.json")):
                keys.append(model_key)
                continue
            if list((run_dir / COHERENCE_TEST_SUBDIR).glob("phase6b_coherence_*.json")):
                keys.append(model_key)
                continue
            if list(run_dir.glob("phase6b_coherence_*.json")):
                keys.append(model_key)
    return sorted(set(keys), key=_model_sort_key)


def discover_all_model_keys(results_dir: pathlib.Path) -> list[str]:
    """Union of models with coherence, within-ladder, or predictive-utility outputs."""
    keys: set[str] = set()
    keys.update(discover_models(results_dir))
    keys.update(discover_within_ladder_models(results_dir))
    keys.update(load_pred_util_per_model(results_dir).keys())
    return sorted(keys, key=_model_sort_key)


def discover_reporting_model_keys(
    results_dir: pathlib.Path,
    within_results_dir: pathlib.Path,
) -> list[str]:
    """Models found in either reporting root."""
    keys = set(discover_all_model_keys(results_dir))
    keys.update(discover_within_ladder_models(within_results_dir))
    return sorted(keys, key=_model_sort_key)


def find_coherence_json(results_dir: pathlib.Path, model_key: str) -> pathlib.Path | None:
    """Find the coherence analysis JSON for a model under model-scoped results."""
    model_dir = resolve_model_results_dir(model_key, results_dir) / "ladder_vs_comparison_statements"
    coherence_dir = model_dir / COHERENCE_TEST_SUBDIR
    candidates = [
        phase6b_coherence_json_path(model_key, results_dir),
        coherence_dir / f"phase6b_coherence_{model_key.replace('-openrouter', '')}.json",
        model_dir / f"phase6b_coherence_{model_key}.json",
        model_dir / f"phase6b_coherence_{model_key.replace('-openrouter', '')}.json",
        model_dir / f"{model_key}.json",
        model_dir / f"{model_key}_analysis.json",
    ]
    if coherence_dir.exists():
        candidates.extend(sorted(coherence_dir.glob("phase6b_coherence_*.json")))
    if model_dir.exists():
        candidates.extend(sorted(model_dir.glob("phase6b_coherence_*.json")))
    seen: set[pathlib.Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
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


def micro_accuracy_ci95(entries: list[dict], point_estimate: float) -> float:
    """Symmetric 95% cluster-bootstrap half-width for a micro-average."""
    usable = [
        entry for entry in entries
        if int(entry.get("n", 0)) > 0
        and math.isfinite(float(entry.get("accuracy", float("nan"))))
    ]
    if not usable or not math.isfinite(point_estimate):
        return float("nan")
    if len(usable) == 1:
        return 0.0

    totals = np.asarray([int(entry["n"]) for entry in usable], dtype=float)
    correct = np.asarray(
        [float(entry["accuracy"]) * int(entry["n"]) for entry in usable],
        dtype=float,
    )
    rng = np.random.default_rng(0)
    sampled = rng.integers(0, len(usable), size=(4000, len(usable)))
    bootstrap_totals = totals[sampled].sum(axis=1)
    bootstrap_accuracy = correct[sampled].sum(axis=1) / bootstrap_totals
    lower, upper = np.quantile(bootstrap_accuracy, [0.025, 0.975])
    return float(max(point_estimate - lower, upper - point_estimate))


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


def _bar_figure_size(
    n_models: int,
    *,
    n_panels: int = 1,
    labels_on_all_panels: bool = False,
) -> tuple[float, float]:
    """Figure dimensions (inches) scaled to bar count and stacked panel count."""
    width = max(13.5, 0.58 * n_models + 2.5)
    panel_h = max(3.4, 0.14 * n_models)
    height = 1.05 + n_panels * panel_h + 1.55
    if labels_on_all_panels and n_panels > 1:
        height += (n_panels - 1) * 1.85
    return width, height


def _bar_chart_margins(
    labels: list[str],
    *,
    n_panels: int = 1,
    labels_on_all_panels: bool = False,
) -> dict[str, float]:
    """Subplot margins with room for rotated x tick labels."""
    max_len = max((len(l) for l in labels), default=16)
    if n_panels > 1 and labels_on_all_panels:
        bottom = min(0.17, 0.05 + 0.0028 * max_len)
    else:
        bottom = min(0.42, 0.14 + 0.005 * max_len)
    top = 0.875 if n_panels > 1 else 0.82
    if n_panels > 1 and labels_on_all_panels:
        hspace = min(1.55, 0.95 + 0.009 * max_len)
    elif n_panels > 1:
        hspace = 0.38
    else:
        hspace = 0.0
    return {
        "top": top,
        "bottom": bottom,
        "left": 0.09,
        "right": 0.99,
        "hspace": hspace,
    }


def _compact_bar_label(file_key: str) -> str:
    """Single-line bar-axis label (tables use full paper_model_label)."""
    base = base_model_label(file_key)
    tag = "reasoning on" if infer_reasoning(file_key) == "on" else "reasoning off"
    return f"{base} · {tag}"


def _set_rotated_bar_labels(
    ax,
    labels: list[str],
    *,
    fontsize: float = 9.0,
    rotation: float = 50,
) -> None:
    """Set x tick labels with room for long model names."""
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)
    ax.tick_params(axis="x", pad=5)


def _paper_table_model_name(file_key: str, display_label: str) -> str:
    """Publication-ready model name for tables."""
    return paper_model_label(file_key)


def collect_coherence_metrics(
    all_data: dict,
    active_models: list,
) -> tuple[list[dict], dict]:
    """Per-model coherence summary rows and macro averages (unweighted across models)."""
    rows = []
    for file_key, label, _family, reasoning in active_models:
        per_set = all_data[file_key]
        mono = float(np.mean([get_mono(r) for r in per_set]))
        iso_r2 = float(np.mean([get_r2(r) for r in per_set]))
        jt = float(np.mean([get_jt(r) for r in per_set]))
        rows.append({
            "file_key": file_key,
            "label": label,
            "reasoning": reasoning,
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
    print(f"\n{'Model':<35} {'Mono%':>7} {'R2':>7} {'JT%':>6} {'N':>4}")
    print("-" * 62)
    for r in rows:
        print(
            f"  {r['table_name']:<33} {r['mono_pct']:>6.1f}% "
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
            return (float("inf"), base_model_label(r.get("file_key", "")), infer_reasoning(r.get("file_key", "")))
        return (
            -mono,
            base_model_label(r.get("file_key", "")),
            infer_reasoning(r.get("file_key", "")),
        )

    return sorted(rows, key=_key)


def _paper_table_tex_preamble(caption: str, label: str) -> list[str]:
    """Shared LaTeX preamble for single-column metric tables."""
    return [
        "% Auto-generated by make_fig_table.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs, array}",
        "\\begin{tableenv}[h!]",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\renewcommand{\\arraystretch}{1.0}",
    ]


def _paper_table_tex_col_spec(num_numeric_cols: int) -> str:
    """tabular: model column + centered numeric columns."""
    nums = "c" * num_numeric_cols
    return f"  \\begin{{tabular}}{{@{{}}l@{{\\hspace{{6pt}}}}{nums}@{{}}}}"


def _paper_table_tex_footer() -> list[str]:
    return [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{tableenv}",
        "",
    ]


def _latex_stacked_col_header(main: str, subtitle: str) -> str:
    """Two-line column header: main label raised, smaller subtitle tucked below."""
    return (
        rf"\raisebox{{1.0ex}}{{\shortstack[c]{{{main}\\"
        rf"[-0.45ex]{{\scriptsize {subtitle}}}}}}}"
    )


def _latex_raised_header(text: str) -> str:
    """Single-line column header, vertically aligned with stacked headers."""
    return rf"\raisebox{{1.0ex}}{{{text}}}"


def write_coherence_metrics_table(
    rows: list[dict],
    macro: dict,
    out_base: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write LaTeX and CSV coherence metrics tables."""
    rows = _sort_rows_by_mono_desc(rows)
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    caption = (
        "Headline coherence metrics across all model variants on the 100 validated ladders. "
        "Mono~\\%: fraction of comparison-block curves that are monotonically non-decreasing "
        "in tier index. Iso $R^2$ (bi): bidirectional isotonic fit (the larger of the increasing "
        "and decreasing fits)."
    )

    tex_lines = [
        *_paper_table_tex_preamble(caption, "tab:coherence_metrics"),
        _paper_table_tex_col_spec(4),
        "    \\toprule",
        "    Model & Strict Mono (\\%) & $R^2$ (bi) & J--T (\\%) & $N$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"    {r['table_name']} & {r['mono_pct']:.1f} & {r['iso_r2']:.3f} "
            f"& {r['jt_pct']:.1f} & {r['n_ladders']} \\\\"
        )
    tex_lines.extend([
        "    \\midrule",
        f"    Macro avg & {macro['mono_pct']:.1f} & {macro['iso_r2']:.3f} "
        f"& {macro['jt_pct']:.1f} & --- \\\\",
        *_paper_table_tex_footer(),
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_key", "model", "reasoning", "strict_mono_pct",
                "iso_r2_bidirectional", "jt_sig_pct", "n_ladders",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model_key": r["file_key"],
                "model": r["table_name"],
                "reasoning": r.get("reasoning", infer_reasoning(r["file_key"])),
                "strict_mono_pct": f"{r['mono_pct']:.1f}",
                "iso_r2_bidirectional": f"{r['iso_r2']:.3f}",
                "jt_sig_pct": f"{r['jt_pct']:.1f}",
                "n_ladders": r["n_ladders"],
            })
        writer.writerow({
            "model_key": "macro_avg",
            "model": "Macro avg",
            "reasoning": "",
            "strict_mono_pct": f"{macro['mono_pct']:.1f}",
            "iso_r2_bidirectional": f"{macro['iso_r2']:.3f}",
            "jt_sig_pct": f"{macro['jt_pct']:.1f}",
            "n_ladders": "",
        })

    return tex_path, csv_path


def _format_combined_r2(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    return f"{val:.3f}"


def _format_combined_perfect_ladders(row: dict) -> str:
    if row.get("wl_n_ladders"):
        return f"{row['perfect_ladders']}/{row['wl_n_ladders']}"
    return "---"


def collect_combined_headline_rows(
    model_keys: list[str],
    coherence_rows: list[dict],
    wl_rows: list[dict],
) -> tuple[list[dict], dict]:
    """Merge coherence (ladder--comparison) and within-ladder (tier×tier) metrics."""
    coh_by_key = {r["file_key"]: r for r in coherence_rows}
    wl_by_key = {r["file_key"]: r for r in wl_rows}

    rows: list[dict] = []
    for mk in model_keys:
        coh = coh_by_key.get(mk)
        wl = wl_by_key.get(mk)
        wl_has = bool(wl and wl.get("n_ladders"))
        coh_has = coh is not None
        rows.append({
            "file_key": mk,
            "reasoning": infer_reasoning(mk),
            "table_name": paper_model_label(mk),
            "wl_accuracy_pct": (
                wl["overall_accuracy_pct"] if wl_has else float("nan")
            ),
            "perfect_ladders": wl.get("perfect_ladders") if wl_has else None,
            "wl_n_ladders": wl.get("n_ladders") if wl_has else None,
            "wl_n_trials": wl.get("n_trials") if wl_has else None,
            "wl_n_trials_expected": wl.get("n_trials_expected") if wl_has else None,
            "wl_n_trials_missing": wl.get("n_trials_missing") if wl_has else None,
            "wl_complete_parseable_coverage": (
                wl.get("complete_parseable_coverage") if wl_has else None
            ),
            "wl_accuracy_lower_bound_pct": (
                wl.get("accuracy_lower_bound_pct", float("nan"))
                if wl_has else float("nan")
            ),
            "wl_accuracy_upper_bound_pct": (
                wl.get("accuracy_upper_bound_pct", float("nan"))
                if wl_has else float("nan")
            ),
            "mono_pct": coh["mono_pct"] if coh_has else float("nan"),
            "iso_r2": coh["iso_r2"] if coh_has else float("nan"),
            "jt_pct": coh["jt_pct"] if coh_has else float("nan"),
            "n_ladders": coh["n_ladders"] if coh_has else (
                wl.get("n_ladders") if wl_has else None
            ),
        })

    def _mean(field: str) -> float:
        vals = [
            r[field] for r in rows
            if r.get(field) is not None and not math.isnan(r.get(field, float("nan")))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    perfect = sum(r.get("perfect_ladders") or 0 for r in rows if r.get("wl_n_ladders"))
    wl_n = sum(r.get("wl_n_ladders") or 0 for r in rows if r.get("wl_n_ladders"))
    macro = {
        "wl_accuracy_pct": _mean("wl_accuracy_pct"),
        "mono_pct": _mean("mono_pct"),
        "iso_r2": _mean("iso_r2"),
        "jt_pct": _mean("jt_pct"),
        "perfect_ladders": perfect if wl_n else None,
        "wl_n_ladders": wl_n if wl_n else None,
    }
    return rows, macro


def write_combined_headline_table(
    rows: list[dict],
    macro: dict,
    out_base: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Combined ladder--comparison coherence + within-ladder tier×tier accuracy."""
    rows = _sort_rows_by_mono_desc(rows)
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    caption = (
        "Headline metrics on the 100 validated ladders. "
        "Accuracy is within-ladder pairwise accuracy. "
        "Strict mono is strict preference monotonicity "
        "on ladder--comparison statement blocks. "
        "$R^2$ (bi) is the mean bidirectional isotonic fit; J--T (\\%) is the fraction of "
        "blocks with a significant Jonckheere--Terpstra trend ($\\alpha = 0.05$). "
        "A dagger marks accuracy computed on an incomplete parseable-response denominator; "
        "the CSV reports scored, expected, missing, and bound columns."
    )

    acc_header = _latex_stacked_col_header(r"Acc.\ (\%)", r"(tier$\times$tier)")
    mono_header = _latex_stacked_col_header(
        r"Strict mono (\%)",
        r"(tier$\times$30 statements)",
    )
    r2_header = _latex_raised_header("$R^2$ (bi)")
    jt_header = _latex_raised_header(r"J--T (\%)")

    tex_lines = [
        *_paper_table_tex_preamble(caption, "tab:headline_combined"),
        _paper_table_tex_col_spec(4),
        "    \\toprule",
        f"    {_latex_raised_header('Model')} & {acc_header} & {mono_header} "
        f"& {r2_header} & {jt_header} \\\\",
        "    \\midrule",
    ]
    for r in rows:
        model_name = r["table_name"]
        if r.get("wl_complete_parseable_coverage") is False:
            model_name += r"\textsuperscript{\dagger}"
        tex_lines.append(
            f"    {model_name} & "
            f"{_format_wl_table_cell(r['wl_accuracy_pct'])} & "
            f"{_format_wl_table_cell(r['mono_pct'])} & "
            f"{_format_combined_r2(r['iso_r2'])} & "
            f"{_format_wl_table_cell(r['jt_pct'])} \\\\"
        )
    tex_lines.extend([
        "    \\midrule",
        f"    Macro avg & "
        f"{_format_wl_table_cell(macro['wl_accuracy_pct'])} & "
        f"{_format_wl_table_cell(macro['mono_pct'])} & "
        f"{_format_combined_r2(macro['iso_r2'])} & "
        f"{_format_wl_table_cell(macro['jt_pct'])} \\\\",
        *_paper_table_tex_footer(),
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    fieldnames = [
        "model_key", "model", "reasoning",
        "tier_x_tier_acc_pct", "tier_x_tier_n_scored",
        "tier_x_tier_n_expected", "tier_x_tier_n_missing",
        "tier_x_tier_complete_coverage",
        "tier_x_tier_acc_lower_bound_pct", "tier_x_tier_acc_upper_bound_pct",
        "strict_mono_pct", "iso_r2_bidirectional", "jt_sig_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model_key": r["file_key"],
                "model": r["table_name"],
                "reasoning": r["reasoning"],
                "tier_x_tier_acc_pct": (
                    f"{r['wl_accuracy_pct']:.1f}"
                    if not math.isnan(r.get("wl_accuracy_pct", float("nan")))
                    else ""
                ),
                "tier_x_tier_n_scored": r.get("wl_n_trials") or "",
                "tier_x_tier_n_expected": r.get("wl_n_trials_expected") or "",
                "tier_x_tier_n_missing": (
                    r.get("wl_n_trials_missing")
                    if r.get("wl_n_trials_missing") is not None
                    else ""
                ),
                "tier_x_tier_complete_coverage": (
                    r.get("wl_complete_parseable_coverage")
                    if r.get("wl_complete_parseable_coverage") is not None
                    else ""
                ),
                "tier_x_tier_acc_lower_bound_pct": _format_wl_table_cell(
                    r.get("wl_accuracy_lower_bound_pct", float("nan"))
                ).replace("---", ""),
                "tier_x_tier_acc_upper_bound_pct": _format_wl_table_cell(
                    r.get("wl_accuracy_upper_bound_pct", float("nan"))
                ).replace("---", ""),
                "strict_mono_pct": (
                    f"{r['mono_pct']:.1f}"
                    if not math.isnan(r.get("mono_pct", float("nan")))
                    else ""
                ),
                "iso_r2_bidirectional": _format_combined_r2(r["iso_r2"]).replace("---", ""),
                "jt_sig_pct": (
                    f"{r['jt_pct']:.1f}"
                    if not math.isnan(r.get("jt_pct", float("nan")))
                    else ""
                ),
            })
        writer.writerow({
            "model_key": "macro_avg",
            "model": "Macro avg",
            "reasoning": "",
            "tier_x_tier_acc_pct": _format_wl_table_cell(macro["wl_accuracy_pct"]),
            "strict_mono_pct": _format_wl_table_cell(macro["mono_pct"]),
            "iso_r2_bidirectional": _format_combined_r2(macro["iso_r2"]),
            "jt_sig_pct": _format_wl_table_cell(macro["jt_pct"]),
        })

    return tex_path, csv_path


# ============================================================
# Appendix: model configurations table
# ============================================================

PAPER_MODEL_CONFIG_GROUPS: list[list[str]] = [
    [
        "gpt-56-sol",
        "gpt-56-sol-thinking",
        "gpt-56-terra",
        "gpt-56-terra-thinking",
        "gpt-56-luna",
        "gpt-56-luna-thinking",
    ],
    [
        "gpt-54-nano",
        "gpt-54-nano-thinking",
        "gpt-54-mini",
        "gpt-54-mini-thinking",
        "gpt-54",
        "gpt-54-thinking",
    ],
    ["opus-46", "opus-46-thinking"],
    ["nemotron-3-super", "nemotron-3-super-thinking"],
    ["glm-45-hybrid", "glm-45-hybrid-thinking"],
    ["glm-45-base-logprobs"],
    [
        "ministral-3b-2512-openrouter",
        "mistral-small-2603-openrouter-thinking",
        "llama-31-8b-instruct-openrouter",
    ],
]

PAPER_MODEL_CONFIG_NOTES: dict[str, str] = {
    "opus-46": r"\texttt{anthropic/claude-4.6-opus-20260205}",
    "opus-46-thinking": "Adaptive thinking",
    "nemotron-3-super-thinking": "Provider pinned to NVIDIA",
    "glm-45-hybrid-thinking": "Provider pinned to Z.AI",
    "glm-45-base-logprobs": (
        r"Pre-training checkpoint (\texttt{zai-org/GLM-4.5}); no reasoning toggle"
    ),
    "ministral-3b-2512-openrouter": "No reasoning toggle",
    "mistral-small-2603-openrouter-thinking": r"\texttt{mistralai/mistral-small-2603}",
    "llama-31-8b-instruct-openrouter": "No reasoning toggle",
}

PAPER_MODEL_CONFIG_OVERRIDES: dict[str, dict] = {
    "glm-45-base-logprobs": {
        "temperature": 0.0,
        "max_tokens": 1,
    },
}


def _model_config_for_table(model_key: str) -> ModelConfig | None:
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]
    if model_key == "glm-45-base-logprobs":
        return MODEL_CONFIGS.get("glm-45-base")
    return None


def _paper_config_model_name(model_key: str) -> str:
    return paper_model_label(model_key)


def _paper_config_provider(model_key: str) -> str:
    if model_key == "glm-45-base-logprobs":
        return "HF Jobs (vLLM)"
    if model_key.startswith("gpt-"):
        return "OpenAI"
    return "OpenRouter"


def _paper_config_reasoning_mechanism(model_key: str, cfg: ModelConfig) -> str:
    if model_key == "glm-45-base-logprobs":
        return "Log-probability scoring"
    extra = cfg.extra_body or {}
    if "reasoning_effort" in extra:
        effort = extra["reasoning_effort"]
        return f"reasoning_effort={effort}"
    reasoning = extra.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("effort") not in (None, ""):
            return f"reasoning.effort={reasoning['effort']}"
        if reasoning.get("enabled") is True:
            return "reasoning.enabled=true"
        if reasoning.get("enabled") is False:
            return "reasoning.enabled=false"
    return "None"


def _latex_texttt(text: str) -> str:
    """Wrap API parameter strings; seqsplit allows line breaks in narrow columns."""
    escaped = text.replace("_", r"\_")
    return rf"\texttt{{\seqsplit{{{escaped}}}}}"


def _latex_config_notes_cell(notes: str) -> str:
    """Notes column with breakable paths and long plain-text phrases."""
    if not notes:
        return ""
    notes = notes.replace("&", "\\&")
    if r"\texttt{" in notes:
        notes = re.sub(
            r"\\texttt\{([^}]+)\}",
            lambda m: _latex_texttt(m.group(1).replace(r"\_", "_")),
            notes,
        )
    if "; " in notes:
        return notes.replace("; ", ";\\\\ ")
    if notes.startswith(r"\texttt{\seqsplit{") and notes.endswith("}"):
        return notes
    if len(notes) > 22:
        return rf"\seqsplit{{{notes}}}"
    return notes


def _format_config_number(val: float | int) -> str:
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    if isinstance(val, int):
        return str(val)
    return f"{val:g}"


def collect_paper_model_config_rows() -> list[dict]:
    """Build appendix model-configuration rows from MODEL_CONFIGS."""
    rows: list[dict] = []
    for group in PAPER_MODEL_CONFIG_GROUPS:
        for model_key in group:
            cfg = _model_config_for_table(model_key)
            if cfg is None:
                raise ValueError(f"No MODEL_CONFIGS entry for paper model {model_key!r}")
            overrides = PAPER_MODEL_CONFIG_OVERRIDES.get(model_key, {})
            reasoning = _paper_config_reasoning_mechanism(model_key, cfg)
            mechanism = overrides.get("reasoning_mechanism", reasoning)
            if mechanism == "None":
                mechanism_tex = "None"
            elif model_key == "glm-45-base-logprobs":
                mechanism_tex = mechanism
            else:
                mechanism_tex = _latex_texttt(mechanism)
            rows.append({
                "model_key": model_key,
                "model": _paper_config_model_name(model_key),
                "provider": overrides.get("provider", _paper_config_provider(model_key)),
                "reasoning_mechanism": mechanism,
                "reasoning_mechanism_tex": mechanism_tex,
                "temperature": overrides.get("temperature", cfg.temperature),
                "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
                "notes": PAPER_MODEL_CONFIG_NOTES.get(model_key, ""),
            })
    return rows


def write_model_configs_table(out_base: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Write appendix model-configuration table (LaTeX table* + CSV)."""
    rows = collect_paper_model_config_rows()
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    caption = (
        "Model configurations for forced-choice elicitation (100-ladder pruned set unless noted). "
        "Reasoning mechanism varies by provider. ``Temp'' is sampling temperature; "
        "``Max tok.'' is the output token cap. GPT-5.4 variants use the OpenAI Batch API "
        "(Responses API); Opus, Nemotron, GLM hybrid, Mistral, Ministral, and Llama use "
        "OpenRouter; GLM-4.5 Base is run as a self-hosted vLLM log-probability job on "
        "Hugging Face compute."
    )

    tex_lines = [
        "% Auto-generated by make_fig_table.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs, placeins, array, seqsplit}",
        "\\FloatBarrier",
        "\\begin{table*}[tb]",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\section{Model Configurations}",
        "  \\label{app:model_configs}",
        "  \\renewcommand{\\arraystretch}{0.95}",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:model_configs}",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}"
        ">{\\raggedright\\arraybackslash}p{0.20\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.10\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.17\\textwidth}"
        ">{\\centering\\arraybackslash}p{0.04\\textwidth}"
        ">{\\centering\\arraybackslash}p{0.06\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.28\\textwidth}}",
        "    \\toprule",
        "    Model & Provider & Reasoning mechanism & Temp & Max tok. & Notes \\\\",
        "    \\midrule",
    ]

    row_idx = 0
    for gi, group in enumerate(PAPER_MODEL_CONFIG_GROUPS):
        for model_key in group:
            row = rows[row_idx]
            row_idx += 1
            notes = _latex_config_notes_cell(row["notes"])
            tex_lines.append(
                f"    {row['model']} & {row['provider']} "
                f"& {row['reasoning_mechanism_tex']} "
                f"& {_format_config_number(row['temperature'])} "
                f"& {_format_config_number(row['max_tokens'])} & {notes} \\\\"
            )
        if gi < len(PAPER_MODEL_CONFIG_GROUPS) - 1:
            tex_lines.append("    \\midrule")

    tex_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table*}",
        "",
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    fieldnames = [
        "model_key", "model", "provider", "reasoning_mechanism",
        "temperature", "max_tokens", "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    return tex_path, csv_path


# ============================================================
# Cost log table (by experiment phase)
# ============================================================

COST_LOG_LADDER_GENERATION_USD = 78.0
COST_LOG_GLM_BASE_HF_USD = 228.0


def _paper_model_keys_flat() -> list[str]:
    keys: list[str] = []
    for group in PAPER_MODEL_CONFIG_GROUPS:
        keys.extend(group)
    return keys


def cost_table_model_label(model_key: str) -> str:
    """Short publication label for the cost table: ``GPT-5.4-Nano (off)``."""
    reasoning = infer_reasoning(model_key)
    short = "on" if reasoning == "on" else "off"
    return f"{base_model_label(model_key)} ({short})"


def _cost_table_provider(model_key: str) -> str:
    if model_key == "glm-45-base-logprobs":
        return r"HF Jobs ($8{\times}$H200)"
    if model_key.startswith("gpt-"):
        return "OpenAI Batch"
    return "OpenRouter"


def _latex_cost_usd(amount: float | None, *, approximate: bool = False) -> str:
    if amount is None:
        return "---"
    prefix = r"$\approx$" if approximate else ""
    if amount >= 1000:
        body = f"{amount:,.0f}".replace(",", "{,}")
    elif amount >= 100:
        body = f"{amount:.2f}"
    elif amount >= 1:
        body = f"{amount:.2f}"
    else:
        body = f"{amount:.3f}".rstrip("0").rstrip(".")
    return f"{prefix}{body}"


def _best_cost_from_phase6b_log(data: dict) -> float | None:
    summary = data.get("summary") or {}
    for key in (
        "actual_cost_usd_sum",
        "actual_cost_usd",
        "estimated_cost_usd_from_usage",
        "estimated_cost_usd",
    ):
        val = data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    val = summary.get("actual_cost_usd")
    if isinstance(val, (int, float)):
        return float(val)
    val = summary.get("estimated_cost_usd")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _within_ladder_cost_usd(results_dir: pathlib.Path, model_key: str) -> float | None:
    path = (
        resolve_model_results_dir(model_key, results_dir)
        / "within_ladder"
        / "phase6b_cost_log.json"
    )
    if not path.is_file():
        if model_key == "glm-45-base-logprobs":
            return COST_LOG_GLM_BASE_HF_USD
        return None
    return _best_cost_from_phase6b_log(json.loads(path.read_text(encoding="utf-8")))


def _comparison_cost_usd(results_dir: pathlib.Path, model_key: str) -> float | None:
    base = (
        resolve_model_results_dir(model_key, results_dir)
        / "ladder_vs_comparison_statements"
    )
    if not base.is_dir():
        return None
    total = 0.0
    n = 0
    for path in base.glob("**/results.json"):
        if "smoke_" in str(path):
            continue
        meta = json.loads(path.read_text(encoding="utf-8")).get("metadata") or {}
        actual = meta.get("actual_cost_usd")
        if actual is None:
            actual = meta.get("estimated_cost_usd")
        if actual is None:
            continue
        total += float(actual)
        n += 1
    return total if n else None


def _validation_cost_rows() -> list[dict]:
    rows: list[dict] = []
    tier_path = (
        VALIDATION_DATA_DIR
        / "within_ladder_validation_tier"
        / "gpt-55-openai"
        / "cost_log.json"
    )
    if tier_path.is_file():
        tier_cost = float(
            json.loads(tier_path.read_text(encoding="utf-8"))["totals"]["cost"]
        )
        rows.append({
            "section": "validation",
            "step": "Tier-pair audit",
            "provider": "OpenAI Batch",
            "cost_usd": tier_cost,
            "approximate": False,
        })

    prop_path = (
        VALIDATION_DATA_DIR
        / "within_ladder_validation_property"
        / "gpt-55-openai"
        / "cost_summary.json"
    )
    if prop_path.is_file():
        prop_cost = float(
            json.loads(prop_path.read_text(encoding="utf-8"))["grand_totals"][
                "best_available_total_usd"
            ]
        )
        rows.append({
            "section": "validation",
            "step": "Property audit",
            "provider": "OpenAI",
            "cost_usd": prop_cost,
            "approximate": False,
        })

    rank_path = (
        VALIDATION_DATA_DIR
        / "within_ladder_validation_ranking"
        / "gpt-55-openai"
        / "ranking_cost_summary.json"
    )
    if rank_path.is_file():
        rank_cost = float(
            json.loads(rank_path.read_text(encoding="utf-8"))["grand_totals"][
                "best_available_total_usd"
            ]
        )
        rows.append({
            "section": "validation",
            "step": "Ranking audit",
            "provider": "OpenAI",
            "cost_usd": rank_cost,
            "approximate": False,
        })
    return rows


def collect_cost_log_rows(results_dir: pathlib.Path) -> list[dict]:
    """Rows for the four experiment phases (generation, validation, WL, comparison)."""
    rows: list[dict] = [
        {
            "section": "generation",
            "step": "Ladder generation (Opus-4.6)",
            "provider": "OpenRouter",
            "cost_usd": COST_LOG_LADDER_GENERATION_USD,
            "approximate": True,
        },
    ]
    rows.extend(_validation_cost_rows())

    for model_key in _paper_model_keys_flat():
        wl_cost = _within_ladder_cost_usd(results_dir, model_key)
        rows.append({
            "section": "within_ladder",
            "step": cost_table_model_label(model_key),
            "provider": _cost_table_provider(model_key),
            "cost_usd": wl_cost,
            "approximate": model_key == "glm-45-base-logprobs",
            "model_key": model_key,
        })

    for model_key in _paper_model_keys_flat():
        cmp_cost = _comparison_cost_usd(results_dir, model_key)
        rows.append({
            "section": "comparison",
            "step": cost_table_model_label(model_key),
            "provider": _cost_table_provider(model_key),
            "cost_usd": cmp_cost,
            "approximate": False,
            "model_key": model_key,
        })
    return rows


def _sort_cost_phase_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (-(r.get("cost_usd") or -1), r.get("model_key", r["step"])),
    )


def write_cost_log_table(
    rows: list[dict],
    out_base: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    section_titles = {
        "generation": "Ladder generation",
        "validation": "Ladder validation (GPT-5.5)",
        "within_ladder": "Within-ladder experiment (Instance~1)",
        "comparison": "Tier vs.\\ comparison statements (Instance~2)",
    }
    section_order = ("generation", "validation", "within_ladder", "comparison")

    caption = (
        "Approximate computational cost by experiment phase. "
        "Ladder generation uses Opus-4.6 on the 146 pre-audit ladders; "
        "ladder validation uses GPT-5.5 on the 146 ladders (tier-pair, property, "
        "and ranking audits). "
        "Within-ladder and tier vs.\\ comparison statement experiments use the "
        "100 validated ladders."
    )

    tex_lines = [
        "% Auto-generated by make_fig_table.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{tableenv}[h!]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\renewcommand{\\arraystretch}{0.95}",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:cost}",
        "  \\begin{tabular}{llr}",
        "    \\toprule",
        "    Model / step & Provider & Cost (USD) \\\\",
        "    \\midrule",
    ]

    grand_total = 0.0
    grand_has_approx = False
    csv_rows: list[dict] = []

    for si, section in enumerate(section_order):
        section_rows = [r for r in rows if r["section"] == section]
        if not section_rows:
            continue
        if si > 0:
            tex_lines.append("    \\addlinespace")
        tex_lines.append(f"    \\multicolumn{{3}}{{l}}{{\\textit{{{section_titles[section]}}}}} \\\\")

        if section in ("within_ladder", "comparison"):
            section_rows = _sort_cost_phase_rows(section_rows)

        section_total = 0.0
        section_has_approx = False
        for row in section_rows:
            cost = row.get("cost_usd")
            approx = bool(row.get("approximate"))
            tex_lines.append(
                f"    {row['step']} & {row['provider']} "
                f"& {_latex_cost_usd(cost, approximate=approx)} \\\\"
            )
            csv_rows.append({
                "section": section,
                "step": row["step"],
                "provider": row["provider"],
                "cost_usd": "" if cost is None else f"{cost:.4f}",
                "approximate": approx,
            })
            if cost is not None:
                section_total += float(cost)
                grand_total += float(cost)
                section_has_approx = section_has_approx or approx
                grand_has_approx = grand_has_approx or approx

        if section != "generation" and section_total > 0:
            subtotal_latex = _latex_cost_usd(section_total, approximate=section_has_approx)
            tex_lines.append(
                f"    \\textit{{Subtotal}} & & \\textit{{{subtotal_latex}}} \\\\"
            )
            csv_rows.append({
                "section": section,
                "step": "Subtotal",
                "provider": "",
                "cost_usd": f"{section_total:.4f}",
                "approximate": section_has_approx,
            })

    if grand_has_approx:
        total_cell = (
            f"$\\approx$\\textbf{{{_latex_cost_usd(grand_total).lstrip('$')}}}"
        )
    else:
        total_cell = f"\\textbf{{{_latex_cost_usd(grand_total)}}}"

    tex_lines.extend([
        "    \\midrule",
        f"    \\textbf{{Total}} & & {total_cell} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{tableenv}",
        "",
    ])

    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["section", "step", "provider", "cost_usd", "approximate"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
        writer.writerow({
            "section": "total",
            "step": "Total",
            "provider": "",
            "cost_usd": f"{grand_total:.4f}",
            "approximate": grand_has_approx,
        })

    return tex_path, csv_path


# ============================================================
# Within-ladder pairwise accuracy tables
# ============================================================

def within_ladder_summary_path(results_dir: pathlib.Path, model_key: str) -> pathlib.Path:
    return resolve_model_results_dir(model_key, results_dir) / "within_ladder" / "summary.json"


def discover_within_ladder_models(results_dir: pathlib.Path) -> list[str]:
    """Model keys with results/<model>/within_ladder/summary.json."""
    if not results_dir.exists():
        return []
    known = set(MODEL_CONFIGS.keys())
    keys = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        model_key = model_key_from_results_folder(d.name, known)
        if within_ladder_summary_path(results_dir, model_key).is_file():
            keys.append(model_key)
    return sorted(set(keys), key=_model_sort_key)


def load_within_ladder_summary(results_dir: pathlib.Path, model_key: str) -> dict:
    with open(within_ladder_summary_path(results_dir, model_key), encoding="utf-8") as fh:
        return json.load(fh)


def ladder_category(ladder_id: str) -> str:
    """Category prefix from ladder_id (text before the last '_<id>')."""
    raw = ladder_id.rsplit("_", 1)[0]
    return raw.replace("-", " ")


def _valence_means(entries: list[dict], valence: str) -> tuple[float | None, int]:
    accs = [e["accuracy"] for e in entries if e.get("valence") == valence]
    if not accs:
        return None, 0
    return float(np.mean(accs)), len(accs)


def _format_wl_table_cell(val: float, *, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def _empty_within_ladder_row(model_key: str) -> dict:
    """Placeholder row when within_ladder/summary.json is not yet available."""
    nan = float("nan")
    return {
        "file_key": model_key,
        "reasoning": infer_reasoning(model_key),
        "table_name": _paper_table_model_name(model_key, make_display_label(model_key)),
        "overall_accuracy_pct": nan,
        "accuracy_ci95_pct": nan,
        "mean_per_ladder_pct": nan,
        "median_per_ladder_pct": nan,
        "min_per_ladder_pct": nan,
        "max_per_ladder_pct": nan,
        "perfect_ladders": None,
        "n_ladders": None,
        "n_trials": None,
        "n_trials_expected": None,
        "n_trials_missing": None,
        "complete_parseable_coverage": None,
        "accuracy_lower_bound_pct": nan,
        "accuracy_upper_bound_pct": nan,
        "parse_errors": None,
        "positive_mean_pct": nan,
        "positive_n": 0,
        "negative_mean_pct": nan,
        "negative_n": 0,
        **{f"dist{d}_pct": nan for d in range(1, 7)},
    }


def collect_within_ladder_summary_rows(
    results_dir: pathlib.Path,
    model_keys: list[str],
) -> list[dict]:
    """One row per paper-slate model; missing summaries become placeholder rows."""
    rows = []
    for file_key in model_keys:
        summary_path = within_ladder_summary_path(results_dir, file_key)
        if summary_path.is_file():
            summary = load_within_ladder_summary(results_dir, file_key)
            rows.append(summarize_within_ladder_model(file_key, summary))
        else:
            rows.append(_empty_within_ladder_row(file_key))
    return rows


def _within_ladder_macro(rows: list[dict]) -> dict:
    """Unweighted macro averages over models with within-ladder results."""
    def _mean(field: str) -> float:
        vals = [
            r[field] for r in rows
            if r.get(field) is not None and not math.isnan(r.get(field, float("nan")))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    perfect = sum(r.get("perfect_ladders") or 0 for r in rows if r.get("n_ladders"))
    n_ladders = sum(r.get("n_ladders") or 0 for r in rows if r.get("n_ladders"))
    return {
        "overall_accuracy_pct": _mean("overall_accuracy_pct"),
        "mean_per_ladder_pct": _mean("mean_per_ladder_pct"),
        "perfect_ladders": perfect if n_ladders else None,
        "n_ladders": n_ladders if n_ladders else None,
    }


def summarize_within_ladder_model(model_key: str, summary: dict) -> dict:
    """One summary row per model for the master within-ladder accuracy table."""
    entries = summary.get("per_ladder", [])
    accs = np.array([e["accuracy"] for e in entries], dtype=float) if entries else np.array([])

    pos_mean, pos_n = _valence_means(entries, "positive")
    neg_mean, neg_n = _valence_means(entries, "negative")
    perfect = sum(1 for e in entries if e.get("accuracy") == 1.0)
    overall_accuracy = float(summary.get("overall_accuracy", float("nan")))
    acc_ci95 = micro_accuracy_ci95(entries, overall_accuracy)
    n_trials = int(summary.get("n_total_pairs", 0))
    n_trials_expected = int(
        summary.get(
            "n_total_pairs_expected",
            summary.get("n_requests_expected", n_trials),
        )
    )
    n_trials_missing = int(
        summary.get(
            "n_requests_missing_from_scoring",
            max(0, n_trials_expected - n_trials),
        )
    )
    accuracy_bounds = summary.get("overall_accuracy_bounds", {})

    by_distance = summary.get("by_distance", {})
    dist_acc = {}
    for d in range(1, 7):
        key = str(d)
        if key in by_distance:
            block = by_distance[key]
            total = block.get("total", 0)
            dist_acc[f"dist{d}_pct"] = (
                100.0 * block.get("correct", 0) / total if total else float("nan")
            )
        else:
            dist_acc[f"dist{d}_pct"] = float("nan")

    row = {
        "file_key": model_key,
        "reasoning": infer_reasoning(model_key),
        "table_name": _paper_table_model_name(model_key, make_display_label(model_key)),
        "overall_accuracy_pct": overall_accuracy * 100.0,
        "accuracy_ci95_pct": acc_ci95 * 100.0,
        "mean_per_ladder_pct": float(np.mean(accs)) * 100.0 if len(accs) else float("nan"),
        "median_per_ladder_pct": float(np.median(accs)) * 100.0 if len(accs) else float("nan"),
        "min_per_ladder_pct": float(np.min(accs)) * 100.0 if len(accs) else float("nan"),
        "max_per_ladder_pct": float(np.max(accs)) * 100.0 if len(accs) else float("nan"),
        "perfect_ladders": perfect,
        "n_ladders": int(summary.get("n_ladders", len(entries))),
        "n_trials": n_trials,
        "n_trials_expected": n_trials_expected,
        "n_trials_missing": n_trials_missing,
        "complete_parseable_coverage": bool(
            summary.get("complete_parseable_coverage", n_trials_missing == 0)
        ),
        "accuracy_lower_bound_pct": float(
            accuracy_bounds.get("lower_missing_incorrect", overall_accuracy)
        ) * 100.0,
        "accuracy_upper_bound_pct": float(
            accuracy_bounds.get("upper_missing_correct", overall_accuracy)
        ) * 100.0,
        "parse_errors": int(summary.get("parse_errors", 0)),
        "positive_mean_pct": pos_mean * 100.0 if pos_mean is not None else float("nan"),
        "positive_n": pos_n,
        "negative_mean_pct": neg_mean * 100.0 if neg_mean is not None else float("nan"),
        "negative_n": neg_n,
        **dist_acc,
    }
    return row


def within_ladder_category_rows(model_key: str, summary: dict) -> list[dict]:
    """Category-level accuracy rows for one model."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    perfect_by_cat: dict[str, int] = defaultdict(int)
    n_by_cat: dict[str, int] = defaultdict(int)

    for entry in summary.get("per_ladder", []):
        cat = ladder_category(entry["ladder_id"])
        acc = float(entry["accuracy"])
        by_cat[cat].append(acc)
        n_by_cat[cat] += 1
        if acc == 1.0:
            perfect_by_cat[cat] += 1

    rows = []
    for cat in sorted(by_cat):
        accs = np.array(by_cat[cat])
        rows.append({
            "model_key": model_key,
            "reasoning": infer_reasoning(model_key),
            "model": _paper_table_model_name(model_key, make_display_label(model_key)),
            "category": cat,
            "n_ladders": n_by_cat[cat],
            "mean_pct": float(np.mean(accs)) * 100.0,
            "median_pct": float(np.median(accs)) * 100.0,
            "min_pct": float(np.min(accs)) * 100.0,
            "max_pct": float(np.max(accs)) * 100.0,
            "perfect_ladders": perfect_by_cat[cat],
        })
    rows.sort(key=lambda r: r["mean_pct"])
    return rows


def _sort_within_ladder_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            -r.get("overall_accuracy_pct", float("nan"))
            if not math.isnan(r.get("overall_accuracy_pct", float("nan")))
            else float("inf"),
            base_model_label(r.get("file_key", "")),
            infer_reasoning(r.get("file_key", "")),
        ),
    )


WL_PERFECT_LADDERS_COL = "100\\% acc. ladders"


def print_within_ladder_accuracy_table(rows: list[dict], macro: dict) -> None:
    print(f"\n{'Model':<35} {'Overall':>8} {'100% lad.':>10} {'N':>4}")
    print("-" * 62)
    for r in rows:
        overall = _format_wl_table_cell(r["overall_accuracy_pct"])
        if r.get("n_ladders"):
            perfect = f"{r['perfect_ladders']}/{r['n_ladders']}"
            n = str(r["n_ladders"])
        else:
            perfect = "---"
            n = "---"
        print(f"  {r['table_name']:<33} {overall:>8} {perfect:>10} {n:>4}")
    print("-" * 62)
    macro_perfect = (
        f"{macro['perfect_ladders']}/{macro['n_ladders']}"
        if macro.get("n_ladders")
        else "---"
    )
    print(
        f"  {'MACRO AVG':<33} "
        f"{_format_wl_table_cell(macro['overall_accuracy_pct']):>8} "
        f"{macro_perfect:>10} {'---':>4}"
    )


def write_within_ladder_accuracy_table(
    rows: list[dict],
    macro: dict,
    out_base: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write LaTeX and CSV summary of within-ladder pairwise accuracy by model."""
    rows = _sort_within_ladder_rows(rows)
    tex_path = out_base.with_suffix(".tex")
    csv_path = out_base.with_suffix(".csv")

    caption = (
        "Within-ladder pairwise accuracy by model (Instance~1). Accuracy (\\%) is the "
        "micro-average over all tier-pair trials (42 per ladder: 21 pairs $\\times$ 2 "
        "orientations). ``100\\% acc.\\ ladders'' counts ladders with perfect pairwise "
        "accuracy (e.g.\\ 79/100 = 79 of 100 ladders). A dagger marks accuracy "
        "computed on an incomplete parseable-response denominator; the CSV reports "
        "scored, expected, missing, and accuracy-bound columns."
    )

    tex_lines = [
        *_paper_table_tex_preamble(caption, "tab:within_ladder_accuracy"),
        _paper_table_tex_col_spec(3),
        "    \\toprule",
        f"    Model & Accuracy (\\%) & {WL_PERFECT_LADDERS_COL} & $N$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        model_name = r["table_name"]
        if r.get("complete_parseable_coverage") is False:
            model_name += r"\textsuperscript{\dagger}"
        if r.get("n_ladders"):
            perfect = f"{r['perfect_ladders']}/{r['n_ladders']}"
            n_cell = str(r["n_ladders"])
        else:
            perfect = "---"
            n_cell = "---"
        tex_lines.append(
            f"    {model_name} & "
            f"{_format_wl_table_cell(r['overall_accuracy_pct'])} & "
            f"{perfect} & {n_cell} \\\\"
        )
    macro_perfect = (
        f"{macro['perfect_ladders']}/{macro['n_ladders']}"
        if macro.get("n_ladders")
        else "---"
    )
    tex_lines.extend([
        "    \\midrule",
        f"    Macro avg & "
        f"{_format_wl_table_cell(macro['overall_accuracy_pct'])} & "
        f"{macro_perfect} & --- \\\\",
        *_paper_table_tex_footer(),
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    fieldnames = [
        "model_key", "model", "reasoning", "overall_accuracy_pct", "accuracy_ci95_pct",
        "mean_per_ladder_pct",
        "median_per_ladder_pct", "min_per_ladder_pct", "max_per_ladder_pct",
        "perfect_ladders", "n_ladders", "n_trials", "n_trials_expected",
        "n_trials_missing", "complete_parseable_coverage",
        "accuracy_lower_bound_pct", "accuracy_upper_bound_pct", "parse_errors",
        "positive_mean_pct", "positive_n", "negative_mean_pct", "negative_n",
        "dist1_pct", "dist2_pct", "dist3_pct", "dist4_pct", "dist5_pct", "dist6_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            out["model_key"] = r.get("file_key", r.get("model_key", ""))
            out["model"] = r.get("table_name", r.get("model", ""))
            for k in fieldnames:
                if k.endswith("_pct") and isinstance(out[k], float):
                    if math.isnan(out[k]):
                        out[k] = ""
                    else:
                        out[k] = f"{out[k]:.1f}"
            writer.writerow(out)

    return tex_path, csv_path


def _remove_stale_within_ladder_category_tables(tables_dir: pathlib.Path) -> None:
    """Drop legacy per-model and long-format category table outputs."""
    for pattern in (
        "table_within_ladder_by_category_*.csv",
        "table_within_ladder_by_category_*.tex",
        "table_within_ladder_by_category.csv",
    ):
        for path in tables_dir.glob(pattern):
            path.unlink(missing_ok=True)


def build_within_ladder_category_matrix(
    all_category_rows: list[dict],
    summary_rows: list[dict],
) -> tuple[list[str], list[tuple[str, str]], dict[tuple[str, str], float]]:
    """
    Category × model matrix of mean pairwise accuracy (%).

    Returns (categories, [(model_key, display_name), ...], (category, model_key) -> pct).
    """
    model_order = [r["file_key"] for r in summary_rows]
    model_labels = {r["file_key"]: r["table_name"] for r in summary_rows}

    matrix: dict[tuple[str, str], float] = {}
    categories: set[str] = set()
    for row in all_category_rows:
        cat = row["category"]
        mk = row["model_key"]
        matrix[(cat, mk)] = float(row["mean_pct"])
        categories.add(cat)

    def row_mean(cat: str) -> float:
        vals = [matrix.get((cat, mk), float("nan")) for mk in model_order]
        return float(np.nanmean(vals)) if vals else float("nan")

    # Match the alphabetical category ordering used in the main monotonicity
    # heatmaps so readers can compare the two panels without remapping columns.
    cat_list = sorted(categories)
    model_cols = [(mk, model_labels.get(mk, mk)) for mk in model_order]
    return cat_list, model_cols, matrix


def write_within_ladder_category_matrix_csv(
    categories: list[str],
    model_cols: list[tuple[str, str]],
    matrix: dict[tuple[str, str], float],
    out_base: pathlib.Path,
) -> pathlib.Path:
    """Write category (rows) × model (columns) accuracy matrix as CSV."""
    csv_path = out_base.with_suffix(".csv")

    header = ["category"] + [label for _key, label in model_cols]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cat in categories:
            row = [cat]
            for mk, _label in model_cols:
                val = matrix.get((cat, mk))
                row.append(f"{val:.1f}" if val is not None else "")
            writer.writerow(row)

    return csv_path


def _within_ladder_group_sort_key(group: dict) -> tuple[int, str]:
    """Keep off/on variants adjacent in the same order as reasoning figures."""
    return _reasoning_pair_order_key(group["label"])


def fig7c_within_ladder_accuracy(
    rows: list[dict],
) -> plt.Figure | None:
    """Paired dot plot of direct within-ladder tier-pair accuracy by model."""
    valid_rows = [
        row for row in rows
        if row.get("n_ladders")
        and not math.isnan(row.get("overall_accuracy_pct", float("nan")))
    ]
    if not valid_rows:
        return None

    grouped: dict[str, dict] = {}
    for row in valid_rows:
        file_key = row["file_key"]
        base = _strip_variant_suffixes(file_key)
        group = grouped.setdefault(
            base,
            {
                "base": base,
                "label": base_model_label(file_key),
                "family": infer_family(file_key),
                "off": None,
                "on": None,
                "partial_coverage": False,
            },
        )
        group[infer_reasoning(file_key)] = row
        if row.get("complete_parseable_coverage") is False:
            group["partial_coverage"] = True

    groups = sorted(grouped.values(), key=_within_ladder_group_sort_key)
    y = np.arange(len(groups))[::-1]
    fig_h = max(5.0, 0.43 * len(groups) + 1.25)
    fig, ax = plt.subplots(figsize=(10.0, fig_h), dpi=300)

    all_values: list[float] = []
    for yi, group in zip(y, groups):
        off = group["off"]
        on = group["on"]
        available = [row for row in (off, on) if row is not None]
        all_values.extend(row["overall_accuracy_pct"] for row in available)

        color_key = available[0]["file_key"]
        color = get_color(group["family"], "off", color_key)
        if off is not None and on is not None:
            ax.plot(
                [off["overall_accuracy_pct"], on["overall_accuracy_pct"]],
                [yi, yi],
                color=color,
                linewidth=1.6,
                alpha=0.55,
                zorder=1,
            )

        for mode, row, marker, dy in (
            ("off", off, "o", -0.13),
            ("on", on, "^", 0.13),
        ):
            if row is None:
                continue
            value = row["overall_accuracy_pct"]
            err = row.get("accuracy_ci95_pct", float("nan"))
            model_color = get_color(group["family"], mode, row["file_key"])
            ax.errorbar(
                value,
                yi,
                xerr=None if math.isnan(err) else err,
                fmt=marker,
                markersize=6.2 if mode == "on" else 5.8,
                color=model_color,
                markeredgecolor="#333333" if mode == "on" else "white",
                markeredgewidth=0.55,
                capsize=2,
                linewidth=0.9,
                zorder=3,
            )
            ax.text(
                value + 0.12,
                yi + dy,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=7.2,
                color="#333333",
            )

    xmin = max(0.0, math.floor((min(all_values) - 1.0) * 2.0) / 2.0)
    ax.set_xlim(xmin, 100.7)
    ax.set_ylim(-0.65, len(groups) - 0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            group["label"] + ("\u2020" if group["partial_coverage"] else "")
            for group in groups
        ],
        fontsize=8.4,
    )
    for tick in ax.get_yticklabels():
        tick.set_color("#111111")

    ax.axvline(95.0, color="#777777", linestyle=":", linewidth=1.0, zorder=0)
    ax.text(
        95.08,
        0.985,
        "95% audit threshold",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=7.2,
        color="#666666",
    )
    ax.set_xlabel("Within-ladder tier-pair accuracy (%)", fontsize=9)
    ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.24)
    style_axes(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777",
                   markeredgecolor="white", markersize=6, label="reasoning off"),
            Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#777777",
                   markeredgecolor="#333333", markersize=6.5, label="reasoning on"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        fontsize=7.5,
        ncol=2,
    )
    if any(group["partial_coverage"] for group in groups):
        fig.text(
            0.5,
            0.015,
            "\u2020 Accuracy uses the parseable-response denominator; see CSV coverage columns.",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.13, top=0.98)
    return fig


def fig7d_within_ladder_category_heatmap(
    categories: list[str],
    model_cols: list[tuple[str, str]],
    matrix: dict[tuple[str, str], float],
) -> plt.Figure:
    """Heatmap: within-ladder tier-pair accuracy by model (rows) and category."""
    n_cats = len(categories)
    n_models = len(model_cols)
    if n_cats == 0 or n_models == 0:
        return plt.figure()

    model_cols = sorted(
        model_cols,
        key=lambda item: _reasoning_pair_order_key(base_model_label(item[0]))
        + (0 if infer_reasoning(item[0]) == "off" else 1,),
    )

    M = np.full((n_models, n_cats), np.nan)
    for i, (mk, _label) in enumerate(model_cols):
        for j, cat in enumerate(categories):
            val = matrix.get((cat, mk))
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                M[i, j] = val

    category_means = np.array([
        float(np.nanmean(M[:, j])) if np.any(~np.isnan(M[:, j])) else np.nan
        for j in range(n_cats)
    ])
    M_display = np.vstack([M, category_means.reshape(1, -1)])
    n_display_rows = n_models + 1

    row_labels = [
        f"{base_model_label(mk)} · {'on' if infer_reasoning(mk) == 'on' else 'off'}"
        for mk, _ in model_cols
    ] + ["Column mean"]
    col_labels = [_category_short(c) for c in categories]

    fig_w = max(10.8, 0.72 * n_cats + 2.8)
    fig_h = max(4.5, 0.38 * n_display_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    valid_vals = M[~np.isnan(M)]
    if valid_vals.size:
        vmin = min(80.0, math.floor(float(np.min(valid_vals)) / 5.0) * 5.0)
        vmax = 100.0
    else:
        vmin, vmax = 80.0, 100.0

    cmap = plt.get_cmap(CATEGORY_HEATMAP_CMAP).copy()
    cmap.set_bad(color="#E0E0E0")
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=95.0, vmax=vmax)
    im = ax.imshow(
        np.ma.masked_invalid(M_display),
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(col_labels, rotation=42, ha="right", fontsize=7.5)
    ax.set_yticks(range(n_display_rows))
    ax.set_yticklabels(row_labels, fontsize=7.4)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("Value category", fontsize=9)
    ax.set_ylabel("Model configuration", fontsize=9)

    ax.axhline(n_models - 0.5, color="white", linewidth=2.0)
    ax.axhline(n_models - 0.5, color="0.25", linewidth=0.8)

    # Fine separators keep adjacent off/on rows visually paired while making
    # transitions between base models easy to scan.
    for i in range(1, n_models):
        prev_base = _strip_variant_suffixes(model_cols[i - 1][0])
        this_base = _strip_variant_suffixes(model_cols[i][0])
        if prev_base != this_base:
            ax.axhline(i - 0.5, color="white", linewidth=0.7, alpha=0.85)

    for i in range(M_display.shape[0]):
        for j in range(M_display.shape[1]):
            val = M_display[i, j]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                ax.text(
                    j, i, "—", ha="center", va="center",
                    fontsize=6, color="0.45",
                )
                continue
            rgba = cmap(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if luminance < 0.55 else "black"
            weight = "bold" if i == n_models else "normal"
            ax.text(
                j, i, f"{val:.0f}", ha="center", va="center",
                fontsize=7, color=txt_color, fontweight=weight,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Within-ladder accuracy (%)\n(95 = audit threshold)", fontsize=8)
    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.24, top=0.99)
    return fig


def write_within_ladder_tables(
    results_dir: pathlib.Path,
    tables_dir: pathlib.Path,
    model_keys: list[str] | None = None,
) -> tuple[list[pathlib.Path], int, tuple[list[str], list[tuple[str, str]], dict] | None]:
    """
    Build within-ladder accuracy tables from results/<model>/within_ladder/summary.json.

    Uses the paper coherence model slate by default; models without summaries appear as ---.

    Writes:
      - tables/table_within_ladder_accuracy.{csv,tex}
      - tables/table_within_ladder_category_matrix.csv
    """
    keys = model_keys if model_keys is not None else discover_models(results_dir)
    if not keys:
        return [], 0, None

    _remove_stale_within_ladder_category_tables(tables_dir)

    summary_rows = collect_within_ladder_summary_rows(results_dir, keys)
    all_category_rows: list[dict] = []
    written: list[pathlib.Path] = []

    for file_key in keys:
        summary_path = within_ladder_summary_path(results_dir, file_key)
        if not summary_path.is_file():
            continue
        summary = load_within_ladder_summary(results_dir, file_key)
        all_category_rows.extend(within_ladder_category_rows(file_key, summary))

    summary_rows = _sort_within_ladder_rows(summary_rows)
    macro = _within_ladder_macro(summary_rows)
    print_within_ladder_accuracy_table(summary_rows, macro)
    tex_path, csv_path = write_within_ladder_accuracy_table(
        summary_rows, macro, tables_dir / "table_within_ladder_accuracy"
    )
    written.extend([tex_path, csv_path])

    categories, model_cols, matrix = build_within_ladder_category_matrix(
        all_category_rows, summary_rows
    )
    matrix_csv = write_within_ladder_category_matrix_csv(
        categories, model_cols, matrix, tables_dir / "table_within_ladder_category_matrix"
    )
    written.append(matrix_csv)
    stale_tex = tables_dir / "table_within_ladder_category_matrix.tex"
    stale_tex.unlink(missing_ok=True)

    print(f"\nWrote within-ladder accuracy table: {csv_path.name}, {tex_path.name}")
    print(
        f"Wrote category × model CSV: {matrix_csv.name} "
        f"({len(categories)} categories × {len(model_cols)} models)"
    )

    return written, len(summary_rows), (categories, model_cols, matrix)


def collect_pred_util_metrics(
    results_dir: pathlib.Path,
    pred_util: dict[str, dict],
    active_models: list | None = None,
) -> tuple[list[dict], dict]:
    """Per-model predictive-utility rows and pooled/unweighted macro summaries."""
    key_order = sorted(pred_util.keys(), key=_model_sort_key)

    rows = []
    for file_key in key_order:
        row = pred_util[file_key]
        mono = coherence_aggregate_mono(results_dir, file_key)
        rows.append({
            "file_key": file_key,
            "reasoning": infer_reasoning(file_key),
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

    paired = [
        (r["mono_pct"], r["mean_auc"])
        for r in rows
        if not math.isnan(r["mono_pct"])
    ]
    if len(paired) >= 2:
        xs, ys = zip(*paired)
        model_r = float(np.corrcoef(xs, ys)[0, 1])
    else:
        model_r = float("nan")

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
        "Some pairs may be excluded when the held-out fold has no preference variation "
        "under split seed~0. "
        f"Model-level $r$ (strict monotonicity vs.\\ mean AUC) $= {macro['model_r_mono_auc']:.2f}$."
    )

    tex_lines = [
        *_paper_table_tex_preamble(caption, "tab:pred_util_metrics"),
        _paper_table_tex_col_spec(4),
        "    \\toprule",
        "    Model & BH pass (\\%) & Mean AUC & Null AUC & $N$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"    {r['table_name']} & {r['bh_pct']:.1f} & {r['mean_auc']:.3f} "
            f"& {r['mean_null']:.3f} & {r['n_ladders']} \\\\"
        )
    tex_lines.extend([
        "    \\midrule",
        f"    Macro avg & {macro['bh_pct']:.1f} "
        f"& {macro['mean_auc']:.3f} & {macro['mean_null']:.3f} & --- \\\\",
        *_paper_table_tex_footer(),
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_key", "model", "reasoning", "bh_pass_pct", "mean_test_auc", "mean_null_auc",
                "strict_mono_pct", "n_analyzable_ladders",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model_key": r["file_key"],
                "model": r["table_name"],
                "reasoning": r.get("reasoning", infer_reasoning(r["file_key"])),
                "bh_pass_pct": f"{r['bh_pct']:.1f}",
                "mean_test_auc": f"{r['mean_auc']:.3f}",
                "mean_null_auc": f"{r['mean_null']:.3f}",
                "strict_mono_pct": f"{r['mono_pct']:.1f}" if not math.isnan(r["mono_pct"]) else "",
                "n_analyzable_ladders": r["n_ladders"],
            })
        writer.writerow({
            "model_key": "macro_avg",
            "model": "Macro avg",
            "reasoning": "",
            "bh_pass_pct": f"{macro['bh_pct']:.1f}",
            "mean_test_auc": f"{macro['mean_auc']:.3f}",
            "mean_null_auc": f"{macro['mean_null']:.3f}",
            "strict_mono_pct": "",
            "n_analyzable_ladders": "",
        })
        writer.writerow({
            "model_key": "pooled",
            "model": "Pooled (all ladder-model pairs)",
            "reasoning": "",
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


def _annotate_macro_avg_vertical(
    ax,
    avg: float,
    label: str,
    *,
    color: str = "#333333",
):
    """Vertical macro-average line for horizontal bar charts."""
    ax.axvline(avg, color=color, lw=1.2, ls=":", alpha=0.85, zorder=10)
    ax.text(
        0.98,
        0.02,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.5,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92),
        zorder=11,
        clip_on=True,
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
    """Horizontal per-model headline chart with readable long labels."""
    rows = []
    for file_key, label, family, reasoning in models:
        if file_key not in all_data:
            continue
        vals = [value_fn(rec) for rec in all_data[file_key]]
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

    # A wide canvas suits the AAAI two-column ``figure*`` format and leaves
    # enough horizontal room for both long model labels and the metric scale.
    fig_h = max(5.8, 0.28 * len(bar_labels) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=300)

    ys = np.arange(len(bar_labels))
    bars = ax.barh(
        ys,
        means,
        xerr=errs,
        color=colors,
        edgecolor="white",
        capsize=3,
        height=0.72,
    )
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("white")

    ax.set_yticks(ys)
    ax.set_yticklabels(bar_labels, fontsize=8.2)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_xlim(*ylim)
    ax.set_ylim(-0.6, len(bar_labels) - 0.4)
    ax.margins(y=0.02)

    macro_avg = float(np.mean(means))
    _annotate_macro_avg_vertical(ax, macro_avg, f"Avg {macro_fmt.format(macro_avg)}")

    if show_legend:
        from matplotlib.patches import Patch
        # The labels already identify each model.  A compact mode key is less
        # cluttered—and remains correct when individual models use overrides.
        legend_handles = [
            Patch(facecolor="#888888", edgecolor="white", label="reasoning off"),
            Patch(
                facecolor="#888888",
                hatch="//",
                edgecolor="white",
                label="reasoning on",
            ),
        ]
        if legend_loc == "upper center":
            fig.legend(
                handles=legend_handles, loc="upper center",
                ncol=2,
                frameon=True, fontsize=8, framealpha=0.92, edgecolor="#dddddd",
                bbox_to_anchor=(0.5, 0.985),
            )
        else:
            ax.legend(
                handles=legend_handles, loc=legend_loc, frameon=True, fontsize=7.5,
                ncol=2, framealpha=0.92, edgecolor="#dddddd",
            )

    top = 0.90 if (suptitle or show_legend) else 0.97
    if suptitle:
        ax.set_title(suptitle, fontsize=10, pad=10)
    # Long methodological explanations belong in the LaTeX caption, not in
    # the plotting canvas.  ``subtitle`` remains accepted for API stability.

    style_axes(ax)
    fig.subplots_adjust(
        top=top,
        bottom=0.10,
        left=0.31,
        right=0.97,
    )
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
        label = pt.get("short", pt.get("label", ""))
        handles.append(Line2D(
            [0], [0],
            marker="^" if pt.get("reasoning") == "on" else "o",
            color="w",
            markerfacecolor=pt.get("color", "#666666"),
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=label,
        ))
    return handles


def _reasoning_marker_legend_handles() -> list:
    """Shared △/○ key for reasoning on vs off."""
    return [
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor="#888888",
            markeredgecolor="black", markeredgewidth=0.5, markersize=7,
            label="reasoning on",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#888888",
            markeredgecolor="black", markeredgewidth=0.5, markersize=7,
            label="reasoning off",
        ),
    ]


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
        data[file_key] = d["per_variation_set"]
    if skipped:
        print(f"  Skipped (no coherence JSON): {', '.join(skipped)}")
    return data


# ============================================================
# Fig 1: Headline monotonicity
# ============================================================
def fig1_headline_mono(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    fig = _model_bar_chart(
        all_data, models, get_mono,
        ylabel="Strict monotonicity (%)",
        ylim=(0, 100),
        macro_fmt="{:.1f}%",
        macro_scale=100.0,
        suptitle="Strict preference monotonicity",
        subtitle=(
            "Each bar = mean across ladders. "
            "A block is monotonic only if win-probability is strictly ordered at every adjacent tier."
        ),
    )
    fig.axes[0].axvline(50, color="gray", lw=0.6, ls="--", alpha=0.35)
    return fig


# ============================================================
# Fig 2: Isotonic R²
# ============================================================
def fig2_iso_r2(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    return _model_bar_chart(
        all_data, models, get_r2,
        ylabel=r"Isotonic $R^2$ (bidirectional)",
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
        ylabel="J--T significant rate (%)",
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


def write_fig1b_metrics_triptych_tex(
    out_path: pathlib.Path,
    *,
    n_models: int,
    macro: dict,
) -> pathlib.Path:
    """LaTeX figure snippet for fig1b (caption only; image has no embedded title)."""
    caption = (
        f"Table~\\ref{{tab:coherence_metrics}} metrics compared across {n_models} "
        "model configurations. "
        f"\\textbf{{Left:}} strict monotonicity (avg \\textbf{{{macro['mono_pct']:.1f}\\%}}). "
        f"\\textbf{{Center:}} isotonic $R^2$ (avg \\textbf{{{macro['iso_r2']:.3f}}}). "
        f"\\textbf{{Right:}} J--T significance (avg \\textbf{{{macro['jt_pct']:.1f}\\%}}). "
        "Hatched bars = reasoning on."
    )
    tex_lines = [
        "% Auto-generated by make_fig_table.py — do not edit by hand.",
        "% Requires: \\usepackage{graphicx}",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/fig1b_metrics_triptych.pdf}",
        f"  \\caption{{{caption}}}",
        "  \\label{fig:1b_metrics_triptych}",
        "\\end{figure}",
        "",
    ]
    out_path.write_text("\n".join(tex_lines), encoding="utf-8")
    return out_path


# ============================================================
# Fig 1b: Three-metric comparison (strict mono vs iso R² vs JT)
# ============================================================
def fig1b_metrics_triptych(all_data: dict, models: list, n_ladders: int) -> plt.Figure:
    """Three horizontal metric panels with one shared model-label axis."""
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if fk in all_data]
    if not active:
        return plt.figure()

    # Keep reasoning-off/on variants adjacent in every panel.  Reordering each
    # panel by its own metric made cross-metric comparison needlessly hard.
    active.sort(key=lambda row: _model_sort_key(row[0]))
    n_models = len(active)
    fig_h = max(6.4, 0.34 * n_models + 1.25)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, fig_h),
        dpi=300,
        sharey=True,
        gridspec_kw={"wspace": 0.13},
    )
    specs = [
        (get_mono, 100.0, (0, 100), "Strict mono. (%)", "{:.1f}%",
         "No adjacent-tier violations"),
        (get_r2, 1.0, (0.5, 1.0), r"Iso $R^2$ (bi)", "{:.2f}",
         r"Best monotonic fit ($\uparrow$ or $\downarrow$)"),
        (get_jt, 100.0, (55, 95), "J--T sig. (%)", "{:.1f}%",
         "Ordered trend in raw trials"),
    ]

    ys = np.arange(n_models)
    bar_labels = [_compact_bar_label(fk) for fk, _, _, _ in active]
    for ax_idx, (ax, (fn, scale, xlim, xlabel, fmt, panel_title)) in enumerate(
        zip(axes, specs)
    ):
        means = []
        colors = []
        hatches = []
        for fk, _label, family, reasoning in active:
            means.append(float(np.mean([fn(rec) for rec in all_data[fk]])) * scale)
            colors.append(get_color(family, reasoning, fk))
            hatches.append(REASON_HATCH[reasoning])

        bars = ax.barh(ys, means, color=colors, edgecolor="white", height=0.72)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("white")

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, n_models - 0.4)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(panel_title, fontsize=9.2, pad=9)
        _annotate_macro_avg_vertical(
            ax,
            float(np.mean(means)),
            f"Avg {fmt.format(float(np.mean(means)))}",
        )
        ax.set_yticks(ys)
        if ax_idx == 0:
            ax.set_yticklabels(bar_labels, fontsize=7.8)
            ax.tick_params(axis="y", labelleft=True, pad=4)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.22)
        style_axes(ax)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#888888", edgecolor="white", label="reasoning off"),
            Patch(facecolor="#888888", hatch="//", edgecolor="white", label="reasoning on"),
        ],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=8,
        framealpha=0.95,
        edgecolor="#dddddd",
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.28, right=0.99, wspace=0.13)
    return fig


# ============================================================
# Within-ladder gap: direct accuracy vs strict monotonicity
# ============================================================
def fig_within_ladder_gap(rows: list[dict]) -> plt.Figure | None:
    """Grouped bars split into stacked reasoning-off/on panels."""
    plotted_by_mode: dict[str, list[dict]] = {"off": [], "on": []}
    for row in rows:
        model_key = row["file_key"]
        base_label = base_model_label(model_key).replace(" (baseline)", "")
        mode_label = "on" if infer_reasoning(model_key) == "on" else "off"
        label = f"{base_label} ({mode_label})"
        mono = row.get("mono_pct")
        if mono is None or not math.isfinite(float(mono)):
            print(
                f"WARNING: skipping {label} in fig_within_ladder_gap: "
                "strict monotonicity is unavailable."
            )
            continue

        accuracy = row.get("wl_accuracy_pct")
        if accuracy is None or not math.isfinite(float(accuracy)):
            print(
                f"WARNING: skipping {label} in fig_within_ladder_gap: "
                "direct tier-pair accuracy is unavailable."
            )
            continue

        plotted_by_mode[mode_label].append({
            "label": (
                base_label + "\u2020"
                if row.get("wl_complete_parseable_coverage") is False
                else base_label
            ),
            "partial_coverage": row.get("wl_complete_parseable_coverage") is False,
            "mono_pct": float(mono),
            "accuracy_pct": float(accuracy),
            "gap_pct": float(accuracy) - float(mono),
        })

    modes = [mode for mode in ("off", "on") if plotted_by_mode[mode]]
    if not modes:
        return None

    for mode in modes:
        plotted_by_mode[mode].sort(
            key=lambda row: (-row["mono_pct"], row["label"])
        )

    total_rows = sum(len(plotted_by_mode[mode]) for mode in modes)
    fig_h = max(3.2, 0.17 * total_rows + 1.35)
    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(3.3, fig_h),
        dpi=300,
        sharex=True,
        gridspec_kw={
            "height_ratios": [len(plotted_by_mode[mode]) for mode in modes],
            "hspace": 0.22,
        },
    )
    if len(modes) == 1:
        axes = [axes]

    mono_color = "#D55E00"
    accuracy_color = "#0072B2"
    for ax, mode in zip(axes, modes):
        mode_rows = plotted_by_mode[mode]
        y = np.arange(len(mode_rows))
        mono = np.asarray([row["mono_pct"] for row in mode_rows], dtype=float)
        accuracy = np.asarray([row["accuracy_pct"] for row in mode_rows], dtype=float)

        bar_height = 0.28
        ax.barh(
            y - 0.16,
            accuracy,
            height=bar_height,
            color=accuracy_color,
            edgecolor="none",
            zorder=2,
        )
        ax.barh(
            y + 0.16,
            mono,
            height=bar_height,
            color=mono_color,
            edgecolor="none",
            zorder=2,
        )
        for yi, row in zip(y, mode_rows):
            if row["label"].rstrip("\u2020") == "Opus-4.6":
                ax.axhspan(
                    yi - 0.48,
                    yi + 0.48,
                    color="#7A3E9D",
                    alpha=0.08,
                    zorder=0,
                )
            ax.text(
                row["accuracy_pct"] - 0.5,
                yi - 0.16,
                f"{row['accuracy_pct']:.1f}",
                ha="right",
                va="center",
                fontsize=4.3,
                color="white",
                fontweight="bold",
                zorder=4,
            )
            ax.text(
                min(row["mono_pct"] + 0.6, 99.0),
                yi + 0.16,
                f"{row['mono_pct']:.1f}",
                ha="left",
                va="center",
                fontsize=4.3,
                color="#444444",
                zorder=4,
            )

        ax.set_xlim(0, 102)
        ax.set_ylim(len(mode_rows) - 0.45, -0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [row["label"] for row in mode_rows],
            fontsize=4.9,
            color="#111111",
        )
        for tick, row in zip(ax.get_yticklabels(), mode_rows):
            if row["label"].rstrip("\u2020") == "Opus-4.6":
                tick.set_color("#6F2C7F")
                tick.set_fontweight("bold")
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=5.2)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.25, zorder=0)
        ax.set_title(f"Reasoning {mode}", fontsize=6.5, loc="left", pad=2)
        style_axes(ax)

    axes[-1].set_xlabel("Percentage (%)", fontsize=6.5, labelpad=2)
    fig.legend(
        handles=[
            Line2D([], [], color=mono_color, linewidth=4,
                   label="Strict ladder\u2013statement monotonicity"),
            Line2D([], [], color=accuracy_color, linewidth=4,
                   label="Direct tier\u2013pair accuracy"),
        ],
        loc="upper center",
        ncol=1,
        frameon=False,
        fontsize=4.8,
        borderaxespad=0.0,
        handletextpad=0.5,
        labelspacing=0.25,
    )
    plotted_rows = [
        row for mode in modes for row in plotted_by_mode[mode]
    ]
    notes = []
    if any(row["label"].rstrip("\u2020") == "Opus-4.6" for row in plotted_rows):
        notes.append("Purple shading highlights Opus-4.6.")
    if any(row["partial_coverage"] for row in plotted_rows):
        notes.append("\u2020 Accuracy uses parseable responses; see CSV coverage columns.")
    if notes:
        fig.text(
            0.5,
            0.012,
            "  ".join(notes),
            ha="center",
            va="bottom",
            fontsize=4.2,
            color="#666666",
        )
    fig.subplots_adjust(
        left=0.45,
        right=0.98,
        bottom=0.10 if notes else 0.08,
        top=0.89,
    )
    return fig


# ============================================================
# Fig 3: Reasoning lift
# ============================================================
def _find_reasoning_pairs(models: list, all_data: dict) -> list[tuple[str, str, str, str]]:
    """Auto-detect (off_key, on_key, display_label, family) pairs from the model list.

    Matches models that share the same base key (stripping '-thinking') and
    differ only in reasoning mode.
    """
    off_models = {}
    on_models = {}
    for fk, label, family, reasoning in models:
        if fk not in all_data:
            continue
        base = _strip_variant_suffixes(fk)
        if reasoning == "on":
            on_models[base] = (fk, label, family)
        else:
            off_models[base] = (fk, label, family)

    pairs = []
    for base in off_models:
        if base in on_models:
            off_fk, _off_label, family = off_models[base]
            on_fk, _, _ = on_models[base]
            pairs.append((off_fk, on_fk, base_model_label(base), family))
    return pairs


_REASONING_PAIR_DISPLAY_ORDER = {
    "GPT-5.4-Nano": 0,
    "GPT-5.4-Mini": 1,
    "GPT-5.4": 2,
    "GPT-5.6 Terra": 3,
    "GPT-5.6 Luna": 4,
    "GPT-5.6 Sol": 5,
    "Opus-4.6": 6,
    "Nemotron-3-Super": 7,
    "GLM-4.5-Hybrid": 8,
    "Qwen3.7 Flash": 9,
}


def _reasoning_pair_order_key(label: str) -> tuple[int, str]:
    """Paper order that keeps matched variants and model families together."""
    return (_REASONING_PAIR_DISPLAY_ORDER.get(label, 10_000), label)


def fig3_reasoning_lift(all_data: dict, models: list) -> plt.Figure:
    pairs = _find_reasoning_pairs(models, all_data)
    if not pairs:
        return plt.figure()

    pair_rows = []
    for off_fk, on_fk, off_label, family in pairs:
        off_rates = [get_mono(r) for r in all_data[off_fk]]
        on_rates = [get_mono(r) for r in all_data[on_fk]]
        m_off, e_off = mean_ci_95(off_rates)
        m_on, e_on = mean_ci_95(on_rates)
        pair_rows.append({
            "off_label": off_label,
            "off_mean": m_off * 100,
            "off_err": e_off * 100,
            "on_mean": m_on * 100,
            "on_err": e_on * 100,
            "color_off": get_color(family, "off", off_fk),
            "color_on": get_color(family, "on", on_fk),
        })
    pair_rows.sort(key=lambda r: _reasoning_pair_order_key(r["off_label"]))

    fig_w = max(7.5, 0.86 * len(pair_rows) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4), dpi=300)
    width = 0.35
    xs = np.arange(len(pair_rows))

    off_means = [r["off_mean"] for r in pair_rows]
    on_means = [r["on_mean"] for r in pair_rows]
    off_errs = [r["off_err"] for r in pair_rows]
    on_errs = [r["on_err"] for r in pair_rows]
    colors_off = [r["color_off"] for r in pair_rows]
    colors_on = [r["color_on"] for r in pair_rows]

    ax.bar(xs - width/2, off_means, width, yerr=off_errs, color=colors_off,
           edgecolor="white", capsize=3, label="reasoning off")
    ax.bar(xs + width/2, on_means, width, yerr=on_errs, color=colors_on,
           edgecolor="white", capsize=3, label="reasoning on", hatch="//")

    for i, (off, on) in enumerate(zip(off_means, on_means)):
        lift = on - off
        sign = "+" if lift >= 0 else ""
        ax.annotate(f"{sign}{lift:.1f}", xy=(i + width/2, on + on_errs[i] + 1),
                    ha="center", fontsize=8, color="black")

    pair_labels = [r["off_label"] for r in pair_rows]
    ax.set_xticks(xs)
    rotation = 32 if len(pair_rows) >= 8 else 25
    ax.set_xticklabels(pair_labels, rotation=rotation, ha="right", fontsize=8.5)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("Monotonicity rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout(pad=0.8)
    return fig


# ============================================================
# Fig 4: Per-set scatter
# ============================================================
def fig4_set_scatter(all_data: dict, models: list) -> plt.Figure:
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if fk in all_data]
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
        x = np.asarray([get_mono(rec) for rec in all_data[file_key]], dtype=float)
        y = np.asarray([get_r2(rec) for rec in all_data[file_key]], dtype=float)
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

        ax.set_title(paper_model_panel_title(file_key, reasoning), fontsize=9)
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
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if fk in all_data]
    if not active:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=300)
    centroids = []
    for file_key, label, family, reasoning in active:
        x = np.asarray([get_mono(rec) for rec in all_data[file_key]], dtype=float)
        y = np.asarray([get_r2(rec) for rec in all_data[file_key]], dtype=float)
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
    rows = []
    for file_key, label, family, reasoning in models:
        if file_key not in all_data:
            continue
        rates = [get_mono(rec) for rec in all_data[file_key]]
        rows.append({
            "label": paper_model_panel_title(file_key, reasoning),
            "rates": rates,
            "color": get_color(family, reasoning, file_key),
            "mean": float(np.mean(rates)),
        })
    order = _ascending_indices([r["mean"] for r in rows])
    rows = [rows[i] for i in order]
    labels = [r["label"] for r in rows]
    data_lists = [r["rates"] for r in rows]
    colors = [r["color"] for r in rows]

    fig_h = max(6.2, 0.35 * len(labels) + 1.4)
    fig, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(11.2, fig_h),
        dpi=300,
        sharey=True,
        gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.08},
    )
    positions = np.arange(1, len(labels) + 1)

    bp = axL.boxplot(data_lists, positions=positions, orientation="horizontal",
                     patch_artist=True, widths=0.6,
                     medianprops=dict(color="black"),
                     whiskerprops=dict(color="gray"),
                     capprops=dict(color="gray"),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="red",
                                     markeredgecolor="none", alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")

    axL.set_yticks(positions)
    axL.set_yticklabels(labels, fontsize=7.8)
    axL.set_xlim(0, 1)
    axL.set_xlabel("Per-ladder monotonicity rate", fontsize=9)
    axL.axvline(0.5, color="red", lw=0.7, ls="--", alpha=0.4)
    axL.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.22)
    axL.invert_yaxis()
    style_axes(axL)

    n_below = [sum(1 for v in dl if v < 0.5) for dl in data_lists]
    axR.barh(positions, n_below, color=colors, edgecolor="white", height=0.65)
    axR.set_xlabel(f"Ladders below 0.5 (of {n_ladders})", fontsize=9)
    axR.tick_params(axis="y", labelleft=False)
    axR.set_xlim(0, max(n_below, default=0) + 10)
    axR.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.22)
    for yi, n in zip(positions, n_below):
        axR.text(n + 0.8, yi, str(n), va="center", ha="left", fontsize=7.5)
    style_axes(axR)
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.09, top=0.98, wspace=0.08)
    return fig


# ============================================================
# Fig 6: Heatmap
# ============================================================
def fig6_heatmap(all_data: dict, models: list) -> plt.Figure:
    active = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if fk in all_data]
    first_fk = active[0][0]
    set_ids = [rec["variation_id"] for rec in all_data[first_fk]]

    M = np.zeros((len(set_ids), len(active)))
    for j, (fk, _, _, _) in enumerate(active):
        for i, rec in enumerate(all_data[fk]):
            M[i, j] = get_mono(rec)

    row_means = M.mean(axis=1)
    order = np.argsort(-row_means)
    M = M[order]
    sorted_ids = [set_ids[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.2, 14), dpi=300)
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(active)))
    model_labels = [
        f"{base_model_label(fk)}\n{'on' if r == 'on' else 'off'}"
        for fk, _, _, r in active
    ]
    ax.set_xticklabels(model_labels, rotation=48, ha="right", fontsize=6.7)
    ax.set_yticks(range(0, len(sorted_ids), 5))
    ax.set_yticklabels([sorted_ids[i][:28] for i in range(0, len(sorted_ids), 5)], fontsize=5)
    ax.set_ylabel("Valence ladders (sorted by cross-model mean)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Monotonicity rate")
    fig.subplots_adjust(left=0.20, right=0.92, bottom=0.18, top=0.99)
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
    for i, (fk, _, _, _) in enumerate(models_subset):
        cat_data: dict[str, list] = defaultdict(list)
        for rec in all_data[fk]:
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

    first_fk = models_subset[0][0]
    categories = sorted(set(rec["category"] for rec in all_data[first_fk]))
    cat_labels = [_category_short(c) for c in categories]
    row_labels = [paper_model_panel_title(fk, r) for fk, _, _, r in models_subset]
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
    # Reserve explicit room for the longest model/category labels.  This also
    # keeps the colorbar clear when the PDF is reduced to paper width.
    fig.subplots_adjust(left=0.23, right=0.94, bottom=0.28, top=0.98)
    return fig


def fig7_category_breakdown_off(all_data: dict, models: list) -> plt.Figure:
    """Heatmap: monotonicity by category (reasoning-off models)."""
    models_off = [(fk, lbl, fam, r) for fk, lbl, fam, r in models
                  if r == "off" and fk in all_data]
    return _draw_category_heatmap(
        all_data, models_off, ylabel="Model (reasoning off)",
    )


def fig7b_category_breakdown_on(all_data: dict, models: list) -> plt.Figure:
    """Heatmap: monotonicity by category (reasoning-on models)."""
    models_on = [(fk, lbl, fam, r) for fk, lbl, fam, r in models
                 if r == "on" and fk in all_data]
    return _draw_category_heatmap(
        all_data, models_on, ylabel="Model (reasoning on)",
    )


# ============================================================
# Fig 8: Ladder shape curves
# ============================================================
def _find_raw_results(results_dir: pathlib.Path, model_key: str) -> list[pathlib.Path]:
    """Collect unique raw ``results.json`` files across current/legacy layouts."""
    model_dir = resolve_model_results_dir(model_key, results_dir)
    roots = [model_dir / "ladder_vs_comparison_statements", model_dir]
    patterns = (
        "phase6b_ladder_*/results.json",
        "phase6b_variations_prune_*/results.json",
    )
    found: list[pathlib.Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            found.extend(root.glob(pattern))

    # Reporting views may expose the same ladder under both naming schemes.
    # Resolve symlinks so each ladder contributes only once.
    unique: dict[pathlib.Path, pathlib.Path] = {}
    for path in sorted(found):
        try:
            identity = path.resolve()
        except OSError:
            identity = path
        unique.setdefault(identity, path)
    return list(unique.values())


def fig8_ladder_shapes(all_data: dict, models: list, results_dir: pathlib.Path) -> plt.Figure:
    active = [
        (fk, lbl, fam, r) for fk, lbl, fam, r in models
        if fk in all_data or _find_raw_results(results_dir, fk)
    ]
    active.sort(key=lambda row: _model_sort_key(row[0]))
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
            # All generated ladders are already preference-ordered from T1
            # (least preferable) to T7 (most preferable), including ladders
            # whose underlying property is negatively valenced.
            tier_probs_all.append(probs)

        if not tier_probs_all:
            ax.set_title(
                f"{paper_model_panel_title(file_key, reasoning)} (no data)",
                fontsize=8,
                pad=5,
            )
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
        title_fs = 8 if n >= 20 else 9
        ax.set_title(
            paper_model_panel_title(file_key, reasoning),
            fontsize=title_fs,
            pad=5,
        )
        style_axes(ax)
        if idx % ncols == 0:
            ax.set_ylabel("P(prefer tier)")

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(pad=0.7, h_pad=0.9, w_pad=0.7)
    return fig


# ============================================================
# Predictive utility loaders
# ============================================================
def _float_csv(val, default=float("nan")):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _find_pred_util_file(
    results_dir: pathlib.Path,
    model_key: str,
    filename: str,
) -> pathlib.Path | None:
    """Find predictive-utility output across current and legacy layouts."""
    model_dir = resolve_model_results_dir(model_key, results_dir)
    candidates = [
        model_dir / "ladder_vs_comparison_statements" / "pred_utility_test" / filename,
        model_dir / "pred_utility_test" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    if model_dir.is_dir():
        matches = sorted(model_dir.rglob(filename))
        if matches:
            return matches[0]
    return None


def load_pred_util_per_model(results_dir: pathlib.Path) -> dict[str, dict]:
    """Load per-model predictive-utility summaries keyed by model_key."""
    out: dict[str, dict] = {}
    if not results_dir.exists():
        return out
    known = set(MODEL_CONFIGS.keys())
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        model_key = model_key_from_results_folder(d.name, known)
        path = _find_pred_util_file(results_dir, model_key, "per_model_pred_util.csv")
        if path is None:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            out[model_key] = rows[0]
    return out


def load_pred_util_per_set(results_dir: pathlib.Path, model_key: str) -> list[dict]:
    """Per-ladder predictive-utility rows for one model."""
    path = _find_pred_util_file(results_dir, model_key, "per_set_pred_util.csv")
    if path is None:
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


def _short_model_label(model_key: str, max_len: int = 28) -> str:
    """Legacy alias; prefer base_model_label / paper_model_label."""
    label = base_model_label(model_key)
    if len(label) > max_len:
        return label[: max_len - 1] + "…"
    return label


def _find_pred_util_reasoning_pairs(
    pred_util: dict[str, dict],
) -> list[tuple[str, str, str]]:
    """(off_key, on_key, family) for models with both reasoning modes."""
    off_by_base: dict[str, tuple[str, str]] = {}
    on_by_base: dict[str, str] = {}
    for key in pred_util:
        base = _strip_variant_suffixes(key)
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
            "short": paper_model_label(key),
            "family": family,
            "reasoning": reasoning,
            "color": get_color(family, reasoning, key),
            "mono": mono,
            "auc": auc,
            "bh_pct": _float_csv(row.get("frac_sig_at_bh_fdr05")) * 100.0,
        })
    if not points:
        return None

    legend_cols = 2 if len(points) > 12 else 1
    fig_w = 10.2 if legend_cols == 2 else 6.6
    fig, ax = plt.subplots(figsize=(fig_w, 5.4), dpi=300)
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
        handles=_reasoning_marker_legend_handles() + _model_legend_handles(points),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        ncol=legend_cols, columnspacing=0.8, handletextpad=0.45,
        frameon=True, fontsize=7, framealpha=0.95, edgecolor="#dddddd",
        title="Model ($\\triangle$ = reasoning on)", title_fontsize=8,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.56 if legend_cols == 2 else 0.68,
        bottom=0.14,
        top=0.97,
    )
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
            "color_off": get_color(family, "off", off_key),
            "color_on": get_color(family, "on", on_key),
        })
    pair_rows.sort(key=lambda r: _reasoning_pair_order_key(base_model_label(r["off_key"])))

    fig_w = max(7.5, 0.86 * len(pair_rows) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4), dpi=300)
    width = 0.35
    xs = np.arange(len(pair_rows))

    off_aucs = [r["off_auc"] for r in pair_rows]
    on_aucs = [r["on_auc"] for r in pair_rows]
    colors_off = [r["color_off"] for r in pair_rows]
    colors_on = [r["color_on"] for r in pair_rows]

    ax.bar(xs - width / 2, off_aucs, width, color=colors_off,
           edgecolor="white", label="reasoning off")
    ax.bar(xs + width / 2, on_aucs, width, color=colors_on,
           edgecolor="white", label="reasoning on", hatch="//")

    for i, (off, on) in enumerate(zip(off_aucs, on_aucs)):
        lift_pp = (on - off) * 100
        sign = "+" if lift_pp >= 0 else ""
        ymax = max(off, on)
        ax.annotate(
            f"{sign}{lift_pp:.1f}pp",
            xy=(i + width / 2, ymax + 0.008),
            ha="center", fontsize=8, color="black",
        )

    pair_labels = [base_model_label(r["off_key"]) for r in pair_rows]
    ax.set_xticks(xs)
    rotation = 32 if len(pair_rows) >= 8 else 25
    ax.set_xticklabels(pair_labels, rotation=rotation, ha="right", fontsize=8.5)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("Mean held-out test AUC")
    ax.set_ylim(0.82, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout(pad=0.8)
    return fig


# ============================================================
# Consolidated reasoning comparison: strict mono + predictive AUC
# ============================================================
def fig3_reasoning_summary(
    all_data: dict,
    models: list,
    pred_util: dict[str, dict],
) -> plt.Figure | None:
    """Compact paired comparison grouped by base model and model family.

    This figure is designed to replace the separate strict-monotonicity and
    predictive-AUC reasoning-lift figures in the paper. Horizontal layouts
    keep long model labels readable at two-column publication width.
    """
    coherence_pairs = _find_reasoning_pairs(models, all_data)
    pred_pairs = {
        _strip_variant_suffixes(off_key): (off_key, on_key)
        for off_key, on_key, _family in _find_pred_util_reasoning_pairs(pred_util)
    }

    rows: list[dict] = []
    for off_fk, on_fk, label, family in coherence_pairs:
        base = _strip_variant_suffixes(off_fk)
        if base not in pred_pairs:
            continue
        pred_off, pred_on = pred_pairs[base]
        mono_off, mono_off_err = mean_ci_95([get_mono(r) for r in all_data[off_fk]])
        mono_on, mono_on_err = mean_ci_95([get_mono(r) for r in all_data[on_fk]])
        auc_off = _float_csv(pred_util[pred_off].get("mean_test_auc"))
        auc_on = _float_csv(pred_util[pred_on].get("mean_test_auc"))
        if math.isnan(auc_off) or math.isnan(auc_on):
            continue
        rows.append({
            "label": label,
            "family": family,
            "off_fk": off_fk,
            "on_fk": on_fk,
            "mono_off": mono_off * 100.0,
            "mono_on": mono_on * 100.0,
            "mono_off_err": mono_off_err * 100.0,
            "mono_on_err": mono_on_err * 100.0,
            "auc_off": auc_off,
            "auc_on": auc_on,
        })
    if not rows:
        return None

    rows.sort(key=lambda r: _reasoning_pair_order_key(r["label"]))
    y = np.arange(len(rows))[::-1]
    fig_h = max(4.6, 0.48 * len(rows) + 1.35)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, fig_h),
        dpi=300,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    def draw_panel(
        ax,
        off_field: str,
        on_field: str,
        *,
        title: str,
        xlim: tuple[float, float],
        delta_scale: float,
        delta_suffix: str,
        off_err_field: str | None = None,
        on_err_field: str | None = None,
    ) -> None:
        for yi, row in zip(y, rows):
            off = row[off_field]
            on = row[on_field]
            off_color = get_color(row["family"], "off", row["off_fk"])
            on_color = get_color(row["family"], "on", row["on_fk"])
            ax.plot([off, on], [yi, yi], color=off_color, lw=1.5, alpha=0.55, zorder=1)

            off_err = row[off_err_field] if off_err_field else None
            on_err = row[on_err_field] if on_err_field else None
            ax.errorbar(
                off,
                yi,
                xerr=off_err,
                fmt="o",
                markersize=5.5,
                color=off_color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=2,
                lw=0.9,
                zorder=3,
            )
            ax.errorbar(
                on,
                yi,
                xerr=on_err,
                fmt="^",
                markersize=6.0,
                color=on_color,
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                capsize=2,
                lw=0.9,
                zorder=4,
            )
            delta = (on - off) * delta_scale
            annotation_x = max(off, on) + 0.018 * (xlim[1] - xlim[0])
            ax.text(
                annotation_x,
                yi,
                f"{delta:+.1f}{delta_suffix}",
                va="center",
                ha="left",
                fontsize=7.3,
                color="#333333",
            )

        for idx in range(len(rows) - 1):
            if rows[idx]["family"] != rows[idx + 1]["family"]:
                separator_y = (y[idx] + y[idx + 1]) / 2.0
                ax.axhline(separator_y, color="#d9d9d9", lw=0.8, zorder=0)

        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.28)
        ax.tick_params(axis="both", labelsize=8)
        style_axes(ax)

    draw_panel(
        axes[0],
        "mono_off",
        "mono_on",
        title="Strict monotonicity",
        xlim=(0.0, 110.0),
        delta_scale=1.0,
        delta_suffix=" pp",
        off_err_field="mono_off_err",
        on_err_field="mono_on_err",
    )
    axes[0].set_xlabel("Strict monotonicity rate (%)", fontsize=9)
    axes[0].set_xticks(np.arange(0, 101, 20))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([row["label"] for row in rows], fontsize=8.5)
    axes[0].tick_params(axis="y", pad=7)
    for tick in axes[0].get_yticklabels():
        tick.set_color("#111111")

    draw_panel(
        axes[1],
        "auc_off",
        "auc_on",
        title="Predictive utility",
        xlim=(0.76, 1.025),
        delta_scale=100.0,
        delta_suffix=" pp",
    )
    axes[1].set_xlabel("Mean held-out test AUC", fontsize=9)
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    mode_legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#666666",
            markeredgecolor="white", markersize=6, label="reasoning off",
        ),
        Line2D(
            [0], [0], marker="^", color="none", markerfacecolor="#999999",
            markeredgecolor="#333333", markersize=7, label="reasoning on",
        ),
    ]
    fig.legend(
        handles=mode_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    # Model-color key. Each matched base model receives one stable hue so the
    # color means the same thing in both panels and both reasoning modes.
    color_legend_handles: list[Line2D] = []
    seen_color_labels: set[str] = set()
    for row in rows:
        color_label = row["label"]
        if color_label in seen_color_labels:
            continue
        seen_color_labels.add(color_label)
        color_legend_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                color="none",
                markerfacecolor=get_color(row["family"], "off", row["off_fk"]),
                markeredgecolor="none",
                markersize=5.5,
                label=color_label,
            )
        )
    fig.legend(
        handles=color_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.947),
        ncol=5,
        frameon=False,
        fontsize=7.3,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.13, top=0.79, wspace=0.10)
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
        title=f"reasoning off ($n$ = {len(off_pts)}, BH fail = {off_fail})",
    )
    _draw_binned_trend_panel(
        axes[0, 1], on_x, on_y, accent="#0072B2", show_bh_fails=True, bh_points=on_pts,
        title=f"reasoning on ($n$ = {len(on_pts)}, BH fail = {on_fail})",
    )
    _draw_tercile_box_panel(axes[1, 0], off_x, off_y, accent="#5C6BC0", title="OFF — tercile")
    _draw_tercile_box_panel(axes[1, 1], on_x, on_y, accent="#0072B2", title="ON — tercile")

    fig.suptitle(
        f"Predictive utility vs coherence: reasoning off vs on  "
        f"(mean AUC {off_auc:.3f} → {on_auc:.3f})",
        fontsize=11, y=1.01,
    )
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.28)
    return fig


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Generate paper figures and tables")
    ap.add_argument(
        "--model", action="append", default=None,
        help="Model key to include (repeatable). If omitted, auto-discovers "
             "all models under --results-dir that have a coherence JSON.",
    )
    ap.add_argument(
        "--results-dir",
        default=str(LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Model-run root containing model-scoped subdirs (default: outputs/).",
    )
    ap.add_argument(
        "--within-results-dir",
        default=None,
        help="Optional separate model-run root containing within_ladder/summary.json "
             "artifacts. If omitted, --results-dir is used.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(FIGURES_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory for output figures "
             "(default: results/figures/)",
    )
    ap.add_argument(
        "--tables-dir",
        default=str(TABLES_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory for output tables "
             "(default: results/tables/)",
    )
    args = ap.parse_args()

    results_dir = resolve_repo_path(args.results_dir)
    within_results_dir = (
        resolve_repo_path(args.within_results_dir)
        if args.within_results_dir
        else results_dir
    )
    out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = resolve_repo_path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_keys = args.model
    else:
        model_keys = discover_reporting_model_keys(results_dir, within_results_dir)
        if not model_keys:
            print(
                "ERROR: no model outputs found under "
                f"{results_dir} or {within_results_dir}"
            )
            return

    models = build_model_tuples(model_keys)

    print(f"Results dir:  {results_dir}")
    print(f"Within dir:   {within_results_dir}")
    print(f"Figures dir:  {out_dir}")
    print(f"Tables dir:   {tables_dir}")
    print(f"Models:       {', '.join(k for k, *_ in models)}")

    mc_base = tables_dir / "table_model_configs"
    mc_tex, mc_csv = write_model_configs_table(mc_base)
    print(f"\nWrote {mc_tex.name} and {mc_csv.name} to {tables_dir}")

    cost_rows = collect_cost_log_rows(results_dir)
    cost_tex, cost_csv = write_cost_log_table(
        cost_rows, tables_dir / "table_cost_log"
    )
    print(f"Wrote {cost_tex.name} and {cost_csv.name} to {tables_dir}")

    print("\nLoading coherence data for available models...")
    all_data = load_all_per_set(results_dir, models)

    # Store figure factories, not live Figure objects.  Several paper figures
    # are large; lazy construction lets us render and close one at a time
    # instead of retaining every canvas in memory until the end.
    targets = []
    active_models: list = []
    metric_rows: list[dict] = []

    if all_data:
        first_key = next(iter(all_data))
        n_ladders = len(all_data[first_key])
        active_models = [(fk, lbl, fam, r) for fk, lbl, fam, r in models if fk in all_data]

        metric_rows, macro = collect_coherence_metrics(all_data, active_models)
        print_coherence_metrics_table(metric_rows, macro)

        table_base = tables_dir / "table_coherence_metrics"
        tex_path, csv_path = write_coherence_metrics_table(metric_rows, macro, table_base)
        print(f"\nWrote {tex_path.name} and {csv_path.name} to {tables_dir}")

        fig1b_tex = write_fig1b_metrics_triptych_tex(
            out_dir / "fig1b_metrics_triptych.tex",
            n_models=len(active_models),
            macro=macro,
        )
        print(f"Wrote {fig1b_tex.name} to {out_dir}")

        targets.extend([
            ("fig1_headline_monotonicity", lambda: fig1_headline_mono(all_data, models, n_ladders)),
            ("fig2_iso_r2_by_variant", lambda: fig2_iso_r2(all_data, models, n_ladders)),
            ("fig2b_jt_significance", lambda: fig2b_jt_significance(all_data, models, n_ladders)),
            ("fig1b_metrics_triptych", lambda: fig1b_metrics_triptych(all_data, models, n_ladders)),
            ("fig3_reasoning_lift", lambda: fig3_reasoning_lift(all_data, models)),
            ("fig4_set_scatter", lambda: fig4_set_scatter(all_data, models)),
            ("fig4b_centroid_scatter", lambda: fig4b_centroid_scatter(all_data, models)),
            ("fig5_incoherence", lambda: fig5_incoherence(all_data, models, n_ladders)),
            ("fig6_coherence_heatmap", lambda: fig6_heatmap(all_data, models)),
            ("fig7_category_breakdown_off", lambda: fig7_category_breakdown_off(all_data, models)),
            ("fig7b_category_breakdown_on", lambda: fig7b_category_breakdown_on(all_data, models)),
            ("fig8_ladder_shapes", lambda: fig8_ladder_shapes(all_data, models, results_dir)),
        ])
    else:
        print("Skipping coherence metrics and fig1–fig8 (no coherence JSONs found).")

    wl_keys = [fk for fk, *_ in models]
    n_with_data = sum(
        1 for file_key in wl_keys
        if within_ladder_summary_path(within_results_dir, file_key).is_file()
    )
    wl_summary_rows: list[dict] = []
    if wl_keys and n_with_data:
        print(
            f"\nWithin-ladder tables for {len(wl_keys)} paper model(s) "
            f"({n_with_data} with summary.json)..."
        )
        _, _, wl_matrix = write_within_ladder_tables(
            within_results_dir, tables_dir, model_keys=wl_keys
        )
        wl_summary_rows = collect_within_ladder_summary_rows(within_results_dir, wl_keys)
        targets.append((
            "fig7c_within_ladder_accuracy",
            lambda: fig7c_within_ladder_accuracy(wl_summary_rows),
        ))
        if wl_matrix:
            categories, model_cols, matrix = wl_matrix
            if categories and model_cols and np.asarray(matrix).size:
                targets.append((
                    "fig7d_within_ladder_category_heatmap",
                    lambda: fig7d_within_ladder_category_heatmap(
                        categories, model_cols, matrix
                    ),
                ))
    else:
        print("\nSkipping within-ladder tables (no within_ladder/summary.json files found).")

    if all_data and wl_summary_rows:
        combined_rows, combined_macro = collect_combined_headline_rows(
            wl_keys, metric_rows, wl_summary_rows
        )
        comb_tex, comb_csv = write_combined_headline_table(
            combined_rows, combined_macro, tables_dir / "table_headline_combined"
        )
        print(f"\nWrote {comb_tex.name} and {comb_csv.name} to {tables_dir}")
        targets.append((
            "fig_within_ladder_gap",
            lambda rows=combined_rows: fig_within_ladder_gap(rows),
        ))

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
        targets.append((
            "fig9_pred_util_vs_coherence",
            lambda: fig9_pred_util_coherence_scatter(results_dir, pred_util),
        ))
        targets.append((
            "fig10_pred_util_reasoning_lift",
            lambda: fig10_pred_util_reasoning_lift(pred_util),
        ))
        if all_data:
            targets.append((
                "fig3_reasoning_summary",
                lambda: fig3_reasoning_summary(all_data, models, pred_util),
            ))
        targets.append((
            "fig11_pred_util_per_ladder_scatter",
            lambda: fig11_pred_util_per_ladder_scatter(results_dir, models),
        ))
        targets.append((
            "fig12_pred_util_reasoning_split",
            lambda: fig12_pred_util_reasoning_split(results_dir, models),
        ))
    else:
        print("\nSkipping fig9–fig12 (no pred_utility_test/ outputs found).")

    print(f"\nGenerating {len(targets)} figures...")
    for name, make_figure in targets:
        fig = make_figure()
        if fig is None:
            continue
        for ext in ("pdf", "png"):
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
            print(f"  {path.name} ({path.stat().st_size // 1024}KB)")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
