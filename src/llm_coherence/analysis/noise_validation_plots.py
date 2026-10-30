"""Standalone plots and an exact reference bound for the saved validation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rebuttal_inputs import digest
from .rebuttal_analysis import write_json


def opposite_unanimity_bound(trials_per_order):
    """Under equal p in both orders and independent choices, max_p 2[p(1-p)]^n."""
    if not isinstance(trials_per_order, (int, np.integer)) or trials_per_order < 1:
        raise ValueError("Expected a positive integer trial count")
    return 2.0 ** (1 - 2 * trials_per_order)


def read_numeric(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty plot directory")
    source_manifest = json.loads((args.results / "run_manifest.json").read_text())
    if source_manifest["status"] != "complete":
        raise ValueError("Numerical validation has not completed successfully")
    files = [args.results / name for name in ("response_process.csv", "response_scenario_expectations.csv", "fit_validation.csv")]
    for path in files:
        if digest(path) != source_manifest["output_files"][path.name]:
            raise ValueError(f"Changed numerical result: {path}")
    rows = [r for r in read_numeric(files[0]) if r["status"] == "complete"]
    expected = read_numeric(files[1])
    fits = read_numeric(files[2])
    args.output.mkdir(parents=True, exist_ok=True)

    def axes(title, labels):
        fig, ax = plt.subplots(figsize=(12, max(4, .36 * len(labels) + 2.3)))
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        ax.set_title("PROVISIONAL — " + title, fontsize=11)
        return fig, ax

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    labels = [f"{r['model_key']} ({100*r['coverage_fraction']:.1f}% coverage)" for r in rows]
    fig, ax = axes("Presentation-order difference", labels)
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.scatter([100 * r["mean_order_difference"] for r in rows], range(len(rows)), color="C0", s=23)
    ax.set_xlabel("Tier choice rate when shown as A minus when shown as B (percentage points)\n"
                  "Each row uses verified cells from that model's original simulation cohort", fontsize=9)
    save(fig, "presentation_order_difference")

    fig, ax = axes("Identical answers within ten repetitions", labels)
    for rho, color, offset in ((0., "C0", -.18), (.5, "C2", 0), (1., "C3", .18)):
        predicted = [next(r["expected_unanimous_group_fraction"] for r in expected
                          if r["model_key"] == actual["model_key"] and r["repeat_correlation"] == rho)
                     for actual in rows]
        ax.scatter(np.array(predicted) * 100, np.arange(len(rows)) + offset, s=18, color=color,
                   label=f"Fitted-utility scenario: repeat correlation {rho:g}")
    ax.scatter([100 * r["observed_unanimous_group_fraction"] for r in rows], range(len(rows)),
               marker="x", color="black", s=26, label="Observed")
    ax.set_xlim(0, 103)
    ax.set_xlabel("Percentage of order-specific groups with all repeated choices identical", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2, fontsize=7)
    save(fig, "repeat_unanimity")

    fig, ax = axes("Prediction error for held-out seven-tier curves", [r["model_key"] for r in fits])
    for i, row in enumerate(fits):
        ax.plot([100 * row["in_sample_rmse"], 100 * row["heldout_rmse"]], [i, i], color="grey", linewidth=1)
    ax.scatter([100 * r["in_sample_rmse"] for r in fits], range(len(fits)), s=21, label="Used in fitting")
    ax.scatter([100 * r["heldout_rmse"] for r in fits], range(len(fits)), s=21, label="Held out of fitting")
    ax.set_xlabel("Probability RMSE in percentage points; three folds; lower is better\n"
                  "Lines connect the two error estimates, not confidence intervals", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2, fontsize=8)
    save(fig, "heldout_prediction_error")

    bounds = [{"model_key": r["model_key"], "eligible_cells_with_verified_responses": int(r["cells"]),
               "coverage_fraction": r["coverage_fraction"],
               "observed_opposite_unanimous_fraction": r["opposite_unanimous_orders_fraction"],
               "maximum_expected_fraction_if_independent_and_order_neutral": opposite_unanimity_bound(int(r["trials_per_order"]))}
              for r in rows]
    write_json(args.output / "opposite_unanimity_reference.json", {
        "formula": "P(opposite unanimous order blocks) = 2[p(1-p)]^n <= 2^(1-2n)",
        "interpretation": "A bound on the expected frequency under independent, order-neutral choices. "
                          "It is not a deterministic bound on an observed sample, a p-value, or an estimate of repeat correlation.",
        "rows": bounds})
    write_json(args.output / "plot_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "provisional": True,
        "inputs": {str(p.resolve()): digest(p) for p in files},
        "code": {str(Path(__file__).resolve()): digest(Path(__file__))},
        "outputs": {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
