"""Export the audited November 3 follow-up figures without making model calls.

Run from the repository root:
    python -m llm_coherence.reporting.followup_figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "gpt-54-nano": ("GPT-5.4 Nano", "OpenAI"),
    "gpt-54-mini": ("GPT-5.4 Mini", "OpenAI"),
    "gpt-54": ("GPT-5.4", "OpenAI"),
    "gpt-56-luna": ("GPT-5.6 Luna", "OpenAI"),
    "gpt-56-terra": ("GPT-5.6 Terra", "OpenAI"),
    "gpt-56-sol": ("GPT-5.6 Sol", "OpenAI"),
    "opus-46": ("Opus 4.6", "Anthropic"),
    "nemotron-3-super": ("Nemotron 3 Super", "NVIDIA"),
    "glm-45-hybrid": ("GLM-4.5 Hybrid", "GLM"),
    "llama-31-8b-instruct-openrouter": ("Llama 3.1 8B", "Meta"),
    "ministral-3b-2512-openrouter": ("Ministral 3B", "Mistral"),
    "mistral-small-2603-openrouter": ("Mistral Small 26.03", "Mistral"),
    "qwen-37-flash-openrouter": ("Qwen 3.7 Flash", "Qwen"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_tables(ranking, temperature, curves, provenance) -> None:
    """Reject changed denominators, duplicate tasks, or unequal pilot cohorts."""
    rank_meta = provenance["ranking"]
    require(not ranking.model_key.duplicated().any(), "Duplicate ranking model")
    require(set(ranking.model_key) == set(rank_meta["model_keys"]), "Model slate mismatch")
    require((ranking.n_planned == rank_meta["ladders_per_configuration"]).all(),
            "Ranking planned denominator mismatch")
    require((ranking.n_responses == ranking.n_planned).all(), "Unfinished ranking tasks")
    require((ranking.n_valid_rankings > 0).all(), "No valid rankings for a model")
    require((ranking.n_valid_rankings + ranking.n_invalid_or_failed == ranking.n_responses).all(),
            "Ranking response denominator mismatch")
    require(((ranking.exact_match_count >= 0) &
             (ranking.exact_match_count <= ranking.n_valid_rankings)).all(),
            "Invalid exact-match count")
    require(np.allclose(ranking.exact_match_fraction_parsed,
                        ranking.exact_match_count / ranking.n_valid_rankings),
            "Exact-match fraction does not use valid rankings")
    require(np.allclose(ranking.mean_kendall_tau,
                        1 - 2 * ranking.mean_pairwise_inversions / 21),
            "Kendall tau and inversion counts disagree")
    require(ranking.n_valid_rankings.sum() == rank_meta["strict_parsed_rankings"],
            "Ranking total differs from audit")
    require(ranking.formatting_only_recovered_count.sum() ==
            rank_meta["additional_formatting_only_rankings"], "Recovery total differs from audit")
    require((ranking.valid_including_formatting_recovery ==
             ranking.n_valid_rankings + ranking.formatting_only_recovered_count).all(),
            "Formatting recovery denominator mismatch")

    meta = provenance["temperature"]
    temps = meta["temperatures"]
    require(not temperature.temperature.duplicated().any(), "Duplicate temperature summary")
    require(set(temperature.temperature) == set(temps), "Temperature summary mismatch")
    require(set(curves.temperature) == set(temps), "Temperature curve mismatch")
    require(not curves.duplicated(["temperature", "ladder_id", "comparison_id"]).any(),
            "Duplicate temperature curve")
    planned = {(ladder, ref) for ladder in meta["ladder_ids"] for ref in meta["reference_ids"]}
    common = set(planned)
    for temp in temps:
        block = curves[curves.temperature == temp]
        require(set(zip(block.ladder_id, block.comparison_id)) == planned,
                "Temperature stimulus set mismatch")
        full = block[(block.parsed_responses == 7 * meta["trials_per_cell"]) &
                     block.nonempty_at_all_tiers]
        common &= set(zip(full.ladder_id, full.comparison_id))
    require(len(common) == meta["common_complete_curves"], "Common complete cohort mismatch")
    require(temperature.parsed.sum() == meta["valid_responses"], "Pilot response total mismatch")
    require(temperature.planned.sum() == meta["planned_responses"], "Pilot planned total mismatch")
    for temp in temps:
        block = curves[curves.temperature == temp]
        mask = np.array([(l, r) in common for l, r in zip(block.ladder_id, block.comparison_id)])
        require(np.array_equal(mask, block.paired_full_response_curve.to_numpy()),
                "Saved common-curve mask mismatch")
        selected = block.loc[mask]
        rates = selected[[f"p{i}" for i in range(1, 8)]].to_numpy()
        require(np.isfinite(rates).all() and ((rates >= 0) & (rates <= 1)).all(),
                "Invalid pilot win rate")
        monotone = (np.diff(rates, axis=1) >= 0).all(axis=1)
        require(np.array_equal(monotone, selected.strict_monotonicity.to_numpy()),
                "Saved monotonicity differs from win rates")
        summary = temperature.loc[temperature.temperature == temp].iloc[0]
        require(summary.parsed == block.parsed_responses.sum(), "Pilot response denominator mismatch")
        require(summary.common_full_response_curves == len(common), "Pilot curve denominator mismatch")
        require(np.isclose(summary.monotonicity_percent, 100 * monotone.mean()),
                "Pilot monotonicity percentage mismatch")


def load_data(data_dir: Path):
    provenance = json.loads((data_dir / "provenance.json").read_text())
    for name, metadata in provenance["data_files"].items():
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        require(actual == metadata["sha256"], f"Source hash mismatch: {name}")
    ranking = pd.read_csv(data_dir / "ranking_summary.csv")
    temperature = pd.read_csv(data_dir / "temperature_summary.csv").sort_values("temperature")
    curves = pd.read_csv(data_dir / "temperature_curves.csv")
    validate_tables(ranking, temperature, curves, provenance)
    ranking = ranking.set_index("model_key").loc[provenance["ranking"]["model_keys"]].reset_index()
    return ranking, temperature, curves, provenance


def save_figure(fig, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def temperature_figure(temperature, provenance, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.23, top=0.83)
    ax.plot(temperature.temperature, temperature.monotonicity_percent,
            color=provenance["family_colors"]["Mistral"], marker="o", linewidth=1.6,
            markersize=6, markeredgecolor="white", markeredgewidth=0.8)
    for row in temperature.itertuples():
        n = int(row.common_full_response_curves)
        passed = round(row.monotonicity_percent * n / 100)
        ax.annotate(f"{row.monotonicity_percent:.1f}%",
                    (row.temperature, row.monotonicity_percent), xytext=(0, 9),
                    textcoords="offset points", ha="center", weight="bold")
        ax.annotate(f"{passed}/{n}",
                    (row.temperature, row.monotonicity_percent), xytext=(0, -15),
                    textcoords="offset points", ha="center", color="0.35")
    ax.set(xlim=(-0.08, 1.08), ylim=(0, 100), xlabel="Sampling temperature",
           ylabel="Strict monotonicity (%)", xticks=temperature.temperature,
           yticks=[0, 20, 40, 60, 80, 100])
    fig.text(0.11, 0.94, "Ministral 3B", weight="bold", va="top")
    fig.text(0.97, 0.94, "External comparisons · same 35 complete curves",
             ha="right", va="top", color="0.35")
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)
    save_figure(fig, output_dir, "ministral_temperature_monotonicity")


def ranking_figure(ranking, provenance, output_dir: Path) -> None:
    # One row per model makes the paired settings easier to compare. Only
    # observed configurations are plotted; absent settings are not zero-filled.
    groups = {}
    for row in ranking.itertuples():
        base = row.model_key.removesuffix("-thinking")
        groups.setdefault(base, {})[row.model_key.endswith("-thinking")] = row
    fig = plt.figure(figsize=(6.3, 3.8))
    names = fig.add_axes([0.015, 0.18, 0.275, 0.65])
    left = fig.add_axes([0.31, 0.18, 0.25, 0.65], sharey=names)
    right = fig.add_axes([0.615, 0.18, 0.22, 0.65], sharey=names)
    counts = fig.add_axes([0.86, 0.18, 0.13, 0.65], sharey=names)
    axes = (names, left, right, counts)
    for ax in axes:
        ax.set_ylim(len(groups) - 0.4, -0.6)
        ax.set_yticks([])
        for y in range(0, len(groups), 2):
            ax.axhspan(y - 0.5, y + 0.5, color="0.965", zorder=0, linewidth=0)
    for y, (base, settings) in enumerate(groups.items()):
        name, family = MODEL_LABELS[base]
        color = to_rgb(provenance["family_colors"][family])
        names.text(0.01, y, name, va="center")
        for on, row in settings.items():
            counts.text(0.76 if on else 0.24, y, str(row.n_valid_rankings),
                        va="center", ha="center", color="0.3")
        ys, percents, taus = [], [], []
        for on, row in sorted(settings.items()):
            position = y + (0.18 if on else -0.18) if len(settings) == 2 else y
            ys.append(position)
            percents.append(100 * row.exact_match_count / row.n_valid_rankings)
            taus.append(row.mean_kendall_tau)
        if len(settings) == 2:
            left.plot(percents, ys, color=color, alpha=0.45, linewidth=1, zorder=2)
            right.plot(taus, ys, color=color, alpha=0.45, linewidth=1, zorder=2)
        for i, on in enumerate(sorted(settings)):
            setting_color = tuple(min(1., x * 1.15) for x in color) if on else color
            style = {"marker": "^" if on else "o", "markersize": 5.3,
                     "markerfacecolor": setting_color if on else "white",
                     "markeredgecolor": setting_color, "markeredgewidth": 1.0,
                     "linestyle": "none", "zorder": 3}
            left.plot(percents[i], ys[i], **style)
            right.plot(taus[i], ys[i], **style)
    names.set(xlim=(0, 1))
    counts.set(xlim=(0, 1))
    names.set_axis_off()
    counts.set_axis_off()
    names.set_title("Model", loc="left", weight="bold", pad=12)
    counts.set_title("Valid / 100", weight="bold", pad=24)
    for x, label in ((0.24, "Off"), (0.76, "On")):
        counts.text(x, 1.035, label, transform=counts.transAxes,
                    ha="center", va="bottom", color="0.3")
    left.set_title("(a) Exact order match", loc="left", weight="bold", pad=12)
    right.set_title("(b) Rank agreement", loc="left", weight="bold", pad=12)
    left.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], xlabel="Valid rankings (%)")
    # A dot plot need not start at zero; make the focused range explicit in
    # ticks and the caption instead of spreading all positive scores over [-1, 1].
    right.set(xlim=(0.60, 1.0), xticks=[0.6, 0.7, 0.8, 0.9, 1.0], xlabel=r"Mean Kendall $\tau$")
    for ax in (left, right):
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="0.87", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="0.25",
                      markersize=5.3, linestyle="none", label="Reasoning off"),
               Line2D([], [], marker="^", color="0.25", markersize=5.3,
                      linestyle="none", label="Reasoning on")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 1.0),
               ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.8)
    fig.text(0.015, 0.035, "Counts are out of 100; blank cells indicate a setting not evaluated.",
             color="0.35")
    save_figure(fig, output_dir, "full_ladder_ranking")


CAPTIONS = r"""% Upload the two PDFs to the figures/ folder in your EXISTING Overleaf project.
% Paste these figure environments where the new experiments are discussed.
% These captions do not change the definitions or results of the original tasks.

\begin{figure}[tbp]
  \centering
  \includegraphics[width=\linewidth]{figures/ministral_temperature_monotonicity.pdf}
  \caption{Temperature sensitivity in a Ministral 3B pilot. We compared
  the seven tiers of 12 ladders against three shared comparison statements,
  using 20 choices per tier--statement pair (10 in each presentation order).
  The plot uses the same 35 complete curves at all four temperatures.
  One failed request at temperature 0.7 left one of the 36 planned curves
  incomplete; that curve is excluded at every temperature.
  Labels give the percentage and number of curves with no downward step.
  Higher temperature did not improve observed strict monotonicity in this pilot.
  These are descriptive results: curves share ladders and comparison statements,
  and the scores do not determine whether underlying preferences are coherent.}
  \label{fig:ministral_temperature}
\end{figure}

\begin{figure}[tbp]
  \centering
  \includegraphics[width=\linewidth]{figures/full_ladder_ranking.pdf}
  \caption{Direct ranking of all seven statements in each of the 100 canonical
  ladders, evaluated separately from the pairwise and external-comparison tasks.
  Each configuration received one request per ladder, with the same seeded
  shuffle of the statements across configurations. Exact order match is the
  percentage of valid rankings that reproduce the full intended tier ordering.
  Kendall's $\tau$ measures agreement with that ordering ($-1$ means reversed,
  $0$ means equal numbers of agreeing and disagreeing pairs, and $1$ means
  full agreement). Panel (b) displays the range 0.6--1.0. Open circles show
  reasoning off and filled triangles show reasoning on; connecting lines pair
  settings of the same model and are not uncertainty intervals. The right-hand
  columns give valid rankings out of 100 for each setting; blank cells indicate
  settings not evaluated. Scores use these denominators and the original parser.
  Of 2,300 requested rankings, 2,241 were
  valid under this parser. An additional 33 were recoverable by a separate
  formatting check and are not included here. On/off denotes the reasoning
  setting. This task measures agreement with the intended ordering; requesting
  a single complete ranking does not independently test transitivity.}
  \label{fig:direct_ladder_ranking}
\end{figure}
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("reports/followup_20261103/data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/followup_20261103/figures"))
    args = parser.parse_args(argv)
    ranking, temperature, _, provenance = load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 9, "figure.titlesize": 10,
                         "axes.labelcolor": "0.2", "text.color": "0.15",
                         "xtick.color": "0.3", "ytick.color": "0.3",
                         "axes.edgecolor": "0.55", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        temperature_figure(temperature, provenance, args.output_dir)
        ranking_figure(ranking, provenance, args.output_dir)
    (args.output_dir.parent / "captions.tex").write_text(CAPTIONS)
    print(f"Validated 23 model summaries and 35 common curves at each temperature.\n"
          f"Wrote two PDF/PNG figures to {args.output_dir}")


if __name__ == "__main__":
    main()
