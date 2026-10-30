"""Standalone comparisons for the offline position-aware agent repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .noise_validation_plots import read_numeric
from .repair_noisy_agent import verified_run
from .rebuttal_analysis import write_json
from .rebuttal_inputs import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty plot directory")
    verified_run(args.results)
    files = [args.results / name for name in
             ("coverage.csv", "prediction_comparison.csv", "repeat_checks.csv", "monotonicity_sensitivity.csv")]
    coverage, prediction, repeats, simulation = [read_numeric(p) for p in files]
    models = [r["model_key"] for r in coverage if r["status"] == "complete"]
    if not models:
        raise ValueError("No completed models to plot")
    args.output.mkdir(parents=True, exist_ok=True)

    def axes(title):
        fig, ax = plt.subplots(figsize=(12, .38 * len(models) + 2.8))
        ax.set_yticks(range(len(models)), models, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        ax.set_title("PROVISIONAL — " + title, fontsize=12, pad=12)
        return fig, ax

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"{name}.{suffix}", dpi=160)
        plt.close(fig)

    heldout = {(r["model_key"], r["response_rule"]): r for r in prediction if r["prediction_source"] == "heldout"}
    fig, ax = axes("Predicting held-out responses in each presentation order")
    neutral = [heldout[m, "neutral"]["order_probability_rmse"] for m in models]
    position = [heldout[m, "position"]["order_probability_rmse"] for m in models]
    for i, (a, b) in enumerate(zip(neutral, position)):
        ax.plot([100 * a, 100 * b], [i, i], color=".65", linewidth=2)
    ax.scatter(np.array(neutral) * 100, range(len(models)), color="#8b6b59", s=30, label="Order-neutral rule")
    ax.scatter(np.array(position) * 100, range(len(models)), color="#177e89", s=30, marker="s", label="Position-adjusted rule")
    ax.set_xlabel("Per-order probability RMSE (percentage points; lower is better)\n"
                  "Same verified curves and three-fold holdouts for both rules; lines are not confidence intervals", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=9)
    save(fig, "heldout_order_prediction")

    independent = {r["model_key"]: r for r in repeats
                   if r["response_rule"] == "position" and r["repeat_correlation_scenario"] == 0}
    fig, ax = axes("Repeat behavior after accounting for presentation order")
    observed = [independent[m]["observed_unanimous_group_fraction"] for m in models]
    expected = [independent[m]["expected_unanimous_group_fraction"] for m in models]
    for i, (a, b) in enumerate(zip(observed, expected)):
        ax.plot([100 * a, 100 * b], [i, i], color=".75", linewidth=1)
    ax.scatter(np.array(expected) * 100, range(len(models)), color="#177e89", s=30,
               label="Expected from held-out probabilities + independent repeats")
    ax.scatter(np.array(observed) * 100, range(len(models)), color="black", s=35, marker="x", label="Observed")
    ax.set_xlim(0, 103)
    ax.set_xlabel("Ten-response blocks with every choice identical (%)\n"
                  "A remaining gap can reflect probability-model mismatch or dependence; it does not identify a correlation", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=8)
    save(fig, "remaining_repeat_mismatch")

    scenarios = {(r["model_key"], r["repeat_correlation_scenario"]): r for r in simulation if r["response_rule"] == "position"}
    values = np.array([[scenarios[m, 0.]["observed_monotonicity_rate"]] +
                       [scenarios[m, rho]["exact_expected_rate"] for rho in (0., .5, 1.)] for m in models]) * 100
    fig, ax = plt.subplots(figsize=(11.5, .4 * len(models) + 2.4))
    heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_xticks(range(4), ["Observed", "Synthetic: rho = 0", "Synthetic: rho = 0.5", "Synthetic: rho = 1"], fontsize=9)
    ax.set_title("PROVISIONAL — Monotonicity under repeat-correlation scenarios", fontsize=12, pad=15)
    for i in range(len(models)):
        for j in range(4):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > 57 else "black")
    ax.axvline(.5, color="white", linewidth=3)
    ax.set_xlabel("Nondecreasing seven-tier curves (%) on each model's matched cohort\n"
                  "Synthetic columns use full-cohort fitted utilities plus the position effect.\n"
                  "Rho values are assumptions, not estimates; these are conditional reference scores, not ceilings.", fontsize=9, labelpad=10)
    fig.colorbar(heatmap, ax=ax, fraction=.035, pad=.03, label="Monotonic curves (%)")
    save(fig, "monotonicity_sensitivity")
    write_json(args.output / "plot_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "provisional": True,
        "inputs": {str(p.resolve()): digest(p) for p in files + [args.results / "run_manifest.json"]},
        "code": {str(Path(__file__).resolve()): digest(Path(__file__))},
        "outputs": {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
