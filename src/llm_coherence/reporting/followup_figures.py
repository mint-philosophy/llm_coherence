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
    fig.savefig(output_dir / f"{name}.png", dpi=300)
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
    fig = plt.figure(figsize=(6.3, 5.2))
    names = fig.add_axes([0.015, 0.17, 0.25, 0.75])
    modes = fig.add_axes([0.28, 0.17, 0.18, 0.75], sharey=names)
    bars = fig.add_axes([0.49, 0.17, 0.49, 0.75], sharey=names)
    axes = (names, modes, bars)
    for ax in axes:
        ax.set_ylim(len(groups) - 0.4, -0.6)
        ax.set_yticks([])
        for y in range(0, len(groups), 2):
            ax.axhspan(y - 0.5, y + 0.5, color="0.965", zorder=0, linewidth=0)
    for y, (base, settings) in enumerate(groups.items()):
        name, family = MODEL_LABELS[base]
        color = to_rgb(provenance.get("model_color_overrides", {}).get(
            base, provenance["family_colors"][family]))
        names.text(0.01, y, name, va="center", fontsize=10)
        for on, row in sorted(settings.items()):
            position = y + (0.24 if on else -0.24) if len(settings) == 2 else y
            percent = 100 * row.exact_match_count / row.n_valid_rankings
            setting_color = tuple(min(1., x * 1.15) for x in color) if on else color
            modes.text(0.5, position, "On" if on else "Off",
                       ha="center", va="center", color="0.2", fontsize=10)
            bars.barh(position, percent, height=0.37, color=setting_color,
                      edgecolor="white", linewidth=0.4, hatch="//" if on else None,
                      zorder=2)
            bars.text(percent + 1.5, position, f"{percent:.1f}%", va="center", fontsize=10)
    names.set(xlim=(0, 1))
    modes.set(xlim=(0, 1))
    names.set_axis_off()
    modes.set_axis_off()
    names.set_title("Model", loc="left", weight="bold", pad=12, fontsize=10)
    modes.set_title("Reasoning mode", weight="bold", pad=12, fontsize=10)
    bars.set_title("All seven tiers in the intended order", loc="left", weight="bold", pad=12)
    bars.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100],
             xlabel="Valid rankings matching all seven tiers (%)")
    bars.tick_params(axis="x", labelsize=10)
    bars.spines["left"].set_visible(False)
    bars.grid(axis="x", color="0.87", linewidth=0.5, zorder=1)
    bars.set_axisbelow(True)
    fig.text(0.015, 0.070, "On = the model's additional reasoning setting was used. Stripes indicate On.", color="0.3")
    fig.text(0.015, 0.030, "100 ladders per configuration; percentages use successfully parsed rankings.",
             color="0.3")
    save_figure(fig, output_dir, "full_ladder_ranking")


def export_ranking_details(ranking, destination: Path) -> None:
    """Keep the denominators and secondary agreement metric outside the main plot."""
    rows = []
    for row in ranking.itertuples():
        base = row.model_key.removesuffix("-thinking")
        rows.append({
            "Model": MODEL_LABELS[base][0],
            "Reasoning mode": "On" if row.model_key.endswith("-thinking") else "Off",
            "Requested rankings": row.n_planned,
            "Valid rankings (original parser)": row.n_valid_rankings,
            "Exact order matches": row.exact_match_count,
            "Exact order match (% of valid rankings)":
                100 * row.exact_match_count / row.n_valid_rankings,
            "Mean Kendall tau": row.mean_kendall_tau,
            "Failed or unparsed": row.n_invalid_or_failed,
            "Formatting-only recoveries (separate check)": row.formatting_only_recovered_count,
            "Valid rankings including formatting recovery": row.valid_including_formatting_recovery,
        })
    pd.DataFrame(rows).to_csv(destination, index=False, lineterminator="\n")


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
  Each bar shows this percentage, with the reasoning setting labeled beside it.
  ``On'' means we used the model's additional reasoning setting;
  ``Off'' means we did not enable that setting. The exact settings
  differed across models. Hatched bars indicate reasoning on. GPT variants
  use distinct base colors, with matching shades for their off/on settings.
  A ranking counts as an exact match only if all seven tiers appear in the
  intended order. Scores use the original parser and the number of valid
  rankings for each configuration, ranging from 86 to 100.
  Of 2,300 requested rankings, 2,241 were
  valid under this parser. An additional 33 were recoverable by a separate
  formatting check and are not included here.
  This task measures agreement with the intended ordering; requesting
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
    export_ranking_details(ranking, args.output_dir.parent / "ranking_details.csv")
    print(f"Validated 23 model summaries and 35 common curves at each temperature.\n"
          f"Wrote two PDF/PNG figures to {args.output_dir}")


if __name__ == "__main__":
    main()
