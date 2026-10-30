"""Offline coherence analyses; never edits manuscript files or calls models.

Run the CLI with --help for the verified-source inputs. Results are provisional:
the current archive contains unresolved instrument differences. The coherent
agent is an explicitly assumed response model, not a universal upper ceiling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.special import expit

from .coherent_baseline import (fit_ordered_utilities, isotonic_scores,
                                replicate_summary, simulate_monotonicity)
from .rebuttal_inputs import (category_rows, digest, load_panel, read_ladders,
                              read_sources, within_distance_rows)


def violation_statistics(probabilities):
    """Downward adjacent steps in the existing T1 -> T7 order, ties allowed."""
    p = np.asarray(probabilities, float)
    if p.ndim != 3 or p.shape[-1] != 7:
        raise ValueError("Expected ladder x reference x seven-tier probabilities")
    finite = np.isfinite(p).all(axis=-1)
    if not finite.any():
        raise ValueError("No complete probability curves")
    steps = np.diff(p, axis=-1)
    drops = np.maximum(-steps, 0)
    valid_drops = drops[finite]
    positive = valid_drops[valid_drops > 0]
    stats = {
        "curves_total": int(finite.size), "curves_analyzed": int(finite.sum()),
        "curves_missing_probability": int((~finite).sum()),
        "monotonicity_rate": float(np.all(steps[finite] >= 0, axis=-1).mean()),
        "adjacent_steps_analyzed": int(valid_drops.size),
        "downward_steps": int(len(positive)),
        "downward_step_fraction": float((valid_drops > 0).mean()),
        "mean_drop_all_steps": float(valid_drops.mean()),
        "mean_positive_drop": float(positive.mean()) if positive.size else None,
        "median_positive_drop": float(np.median(positive)) if positive.size else None,
        "p90_positive_drop": float(np.quantile(positive, .9)) if positive.size else None,
        "max_positive_drop": float(positive.max()) if positive.size else 0.0,
        "mean_max_drop_per_curve": float(valid_drops.max(axis=-1).mean()),
        "mean_total_drop_per_curve": float(valid_drops.sum(axis=-1).mean()),
    }
    # Right-closed, positive-only bins; round only for assigning boundary bins,
    # never when calculating magnitudes or deciding whether a violation exists.
    edges = np.array([.05, .10, .25, .50, 1.0])
    indices = np.searchsorted(edges, np.round(positive, 12), side="left")
    bins = []
    for i, upper in enumerate(edges):
        bins.append({"lower_exclusive": 0 if i == 0 else edges[i - 1],
                     "upper_inclusive": upper, "downward_steps": int((indices == i).sum()),
                     "fraction_of_downward_steps": float((indices == i).mean()) if positive.size else None})
    transitions = [{"from_tier": i + 1, "to_tier": i + 2,
                    "steps_analyzed": int(finite.sum()),
                    "downward_steps": int((valid_drops[:, i] > 0).sum()),
                    "mean_drop_all_steps": float(valid_drops[:, i].mean())}
                   for i in range(6)]
    return stats, bins, transitions, drops, finite


def shuffle_tiers(probabilities, rng):
    """One common permutation for all reference curves within each ladder."""
    p = np.asarray(probabilities)
    order = np.stack([rng.permutation(p.shape[-1]) for _ in range(p.shape[0])])
    return np.take_along_axis(p, order[:, None, :], axis=-1)


def permutation_reference(probabilities, *, repetitions=499, seed=42):
    p = np.asarray(probabilities, float)
    finite = np.isfinite(p).all(axis=-1)
    y = p[finite]
    if not len(y):
        raise ValueError("No complete curves for permutations")
    constant = np.all(y == y[:, :1], axis=-1)

    def metrics(values):
        _, bi = isotonic_scores(values)
        return {
            "monotonicity": float(np.all(np.diff(values, axis=-1) >= 0, axis=-1).mean()),
            "bidirectional_r2_nonconstant": float(bi[~constant].mean()) if (~constant).any() else None,
            "bidirectional_r2_constant_one": float(np.where(constant, 1.0, bi).mean()),
        }

    observed = metrics(y)
    replicates = {key: [] for key, value in observed.items() if value is not None}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        shuffled = metrics(shuffle_tiers(p, rng)[finite])
        for key in replicates:
            replicates[key].append(shuffled[key])
    result = {"curves_analyzed": len(y), "constant_curves": int(constant.sum()),
              "nonconstant_curves": int((~constant).sum()), "observed": observed,
              "shuffle_unit": "one independent tier permutation per ladder, shared by its references",
              "replicate_bands_are_confidence_intervals": False}
    result["permuted"] = {key: replicate_summary(values) for key, values in replicates.items()}
    return result, replicates


def noisy_agent_reference(panel, *, repetitions=300, seed=42, ridge=1e-5):
    if panel.scoring_kind != "sampled_choices":
        return {"status": "excluded_continuous_logprobs",
                "reason": "Rounded pseudo-counts are not observed Bernoulli trials"}, [], None
    valid = np.isfinite(panel.probabilities).all(axis=-1)
    complete = valid & np.all(panel.totals == 2 * panel.trials_per_direction, axis=-1)
    eligible = complete & ~panel.input_mismatch_ladders[:, None]
    diagnostics = {"complete_trial_curves": int(complete.sum()),
                   "incomplete_trial_curves": int(complete.size - complete.sum()),
                   "complete_curves_excluded_for_input_mismatch": int((complete & ~eligible).sum()),
                   "eligible_curves": int(eligible.sum())}
    if not eligible.any():
        return {"status": "no_eligible_curves", **diagnostics}, [], None
    # Drop entirely unobserved ladders/references before fitting. Surviving
    # references still have one shared utility across all surviving ladders.
    li, ri = np.flatnonzero(eligible.any(axis=1)), np.flatnonzero(eligible.any(axis=0))
    subset = np.ix_(li, ri)
    eligible_fit = eligible[subset]
    k = np.where(eligible_fit[..., None], panel.successes[subset], 0)
    n = np.where(eligible_fit[..., None], panel.totals[subset], 0)
    fit = fit_ordered_utilities(k, n, ridge=ridge, outcome_ids=np.asarray(panel.tier_texts)[li])
    diagnostics.update(fit.diagnostics)
    diagnostics.update({"calibration_ladders": [panel.ladder_ids[i] for i in li],
                        "calibration_reference_ids": [panel.reference_ids[i] for i in ri],
                        "ladder_utilities": fit.ladder_utilities.tolist(),
                        "reference_utilities": fit.reference_utilities.tolist()})
    if not fit.diagnostics["optimizer_success"]:
        return {"status": "calibration_failed", **diagnostics}, [], None
    observed = float(np.all(np.diff(panel.probabilities[eligible], axis=-1) >= 0, axis=-1).mean())
    eta = fit.ladder_utilities[:, None, :] - fit.reference_utilities[None, :, None]
    protocol_trials = panel.trials_per_direction
    # Noise multipliers change gap-to-noise ratios; trial-count sensitivities
    # change output resolution. Only protocol_n scenarios match observed n.
    scenarios = [
        ("protocol_independent", 1.0, 0.0, protocol_trials),
        ("protocol_repeat_correlation_0.5", 1.0, .5, protocol_trials),
        ("protocol_identical_repeats", 1.0, 1.0, protocol_trials),
        ("half_noise", .5, 0.0, protocol_trials),
        ("double_noise", 2.0, 0.0, protocol_trials),
        ("resolution_one_trial_per_order", 1.0, 0.0, 1),
        ("resolution_fifty_trials_per_order", 1.0, 0.0, 50),
    ]
    rows = []
    for index, (name, noise, rho, trials) in enumerate(scenarios):
        stats = simulate_monotonicity(expit(eta / noise), eligible_fit, trials,
                                      repetitions=repetitions, repeat_correlation=rho,
                                      seed=seed + index)
        rows.append({"model_key": panel.model_key, "scenario": name,
                     "noise_multiplier": noise, "repeat_correlation": rho,
                     "trials_per_direction": trials,
                     "matches_observed_trial_count": trials == protocol_trials,
                     "eligible_curves": int(eligible.sum()),
                     "observed_monotonicity_same_curves": observed, **stats})
    return {"status": "complete", "observed_monotonicity_same_curves": observed,
            **diagnostics}, rows, (fit.probabilities, eligible_fit)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def curve_rows(panel, drops, finite):
    rows = []
    for li, ladder_id in enumerate(panel.ladder_ids):
        for ri, reference in enumerate(panel.reference_ids):
            valid = bool(finite[li, ri])
            values = panel.probabilities[li, ri]
            d = drops[li, ri]
            rows.append({"model_key": panel.model_key, "ladder_id": ladder_id,
                         "category": panel.categories[li], "reference_id": reference,
                         "scoring_kind": panel.scoring_kind, "complete_probability_curve": valid,
                         "input_mismatch": bool(panel.input_mismatch_ladders[li]),
                         "monotonic": bool(np.all(np.diff(values) >= 0)) if valid else None,
                         "downward_steps": int((d > 0).sum()) if valid else None,
                         "max_drop": float(d.max()) if valid else None,
                         "total_drop": float(d.sum()) if valid else None,
                         **{f"p_t{i+1}": float(v) if np.isfinite(v) else None for i, v in enumerate(values)},
                         **{f"drop_t{i+1}_t{i+2}": float(v) if np.isfinite(v) else None for i, v in enumerate(d)},
                         **{f"saved_count_total_t{i+1}": int(v) for i, v in enumerate(panel.totals[li, ri])}})
    return rows


def make_plots(output, adjacent, violations, simulations, permutations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def canvas(title, labels):
        fig, ax = plt.subplots(figsize=(11, max(4, .31 * len(labels) + 1.8)))
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title("PROVISIONAL — " + title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        return fig, ax

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    labels = [r["model_key"] for r in adjacent]
    fig, ax = canvas("Adjacent-tier accuracy (observed pairs)", labels)
    ax.scatter([r["accuracy_among_observed"] for r in adjacent], range(len(labels)), s=25)
    for i, row in enumerate(adjacent):
        ax.plot([row["full_coverage_accuracy_lower"], row["full_coverage_accuracy_upper"]],
                [i, i], color="black", linewidth=1)
    ax.set_xlabel("Accuracy; lines bound the effect of missing pairs (not confidence intervals)")
    ax.set_xlim(0, 1.02)
    save(fig, "adjacent_accuracy")
    fig, ax = canvas("Sizes of downward adjacent steps", labels)
    left = np.zeros(len(labels))
    for index, color in enumerate(plt.get_cmap("YlOrRd")(np.linspace(.25, .9, 5))):
        values = [sum(r["fraction_of_downward_steps"] or 0 for r in violations
                      if r["model_key"] == model and r["upper_inclusive"] == [.05, .1, .25, .5, 1][index])
                  for model in labels]
        lo, hi = [0, .05, .1, .25, .5][index], [.05, .1, .25, .5, 1][index]
        ax.barh(range(len(labels)), values, left=left, color=color, label=f"({lo:g}, {hi:g}]")
        left += values
    ax.set_xlabel("Fraction of positive drops; magnitude is in probability units")
    ax.legend(title="Drop size", loc="upper center", bbox_to_anchor=(.5, -.09), ncol=5, fontsize=7)
    save(fig, "violation_sizes")
    primary = [r for r in simulations if r["scenario"] == "protocol_independent"]
    fig, ax = canvas("Noisy coherent agent: sensitivity to repeated responses", [r["model_key"] for r in primary])
    for scenario, label, color, offset in (
        ("protocol_independent", "Independent repeats", "C0", -.18),
        ("protocol_repeat_correlation_0.5", "Repeat correlation 0.5", "C2", 0),
        ("protocol_identical_repeats", "Identical repeats within each order", "C3", .18),
    ):
        selected = [r for r in simulations if r["scenario"] == scenario]
        positions = np.arange(len(selected)) + offset
        for y, row in zip(positions, selected):
            ax.plot([row["replicate_q025"], row["replicate_q975"]], [y, y], color=color)
        ax.scatter([r["mean"] for r in selected], positions, label=label, color=color, s=16)
    ax.scatter([r["observed_monotonicity_same_curves"] for r in primary], range(len(primary)),
               label="Observed, same eligible curves", marker="x", color="black", s=25)
    ax.set_xlabel("Ties-allowed monotonicity; bands are conditional simulation quantiles")
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2)
    save(fig, "noisy_coherent_agent")
    fig, ax = canvas("Bidirectional isotonic R² vs shuffled tiers (nonconstant curves)", labels)
    for i, row in enumerate(permutations):
        if row.get("bidirectional_r2_nonconstant_null_mean") is not None:
            ax.plot([row["bidirectional_r2_nonconstant_null_replicate_q025"],
                     row["bidirectional_r2_nonconstant_null_replicate_q975"]], [i, i], color="C1")
    ax.scatter([r["bidirectional_r2_nonconstant_observed"] for r in permutations], range(len(labels)),
               marker="x", label="Observed", s=25)
    ax.scatter([r.get("bidirectional_r2_nonconstant_null_mean") for r in permutations], range(len(labels)),
               label="Shuffled mean", s=22)
    ax.set_xlabel("R²; bands are permutation quantiles, not confidence intervals")
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2)
    save(fig, "isotonic_permutation")


METHOD_NOTES = """PROVISIONAL OFFLINE ANALYSES — NOT MANUSCRIPT OUTPUT

Scope: adjacent-tier accuracy; adjacent downward-step sizes; bidirectional
isotonic R2 with shuffled-tier reference; category counts; noisy coherent agent.
No new model calls. No TeX, manuscript figures, PDFs or Overleaf ZIPs are edited.

Tiers remain in their predefined least-to-most-preferable order for every
valence. Monotonic means nondecreasing, allowing ties. Probabilities with no
observed choices are missing, not 0.5. Descriptive curve analyses require all
seven probabilities and explicitly count excluded curves. Positive drop sizes
are differences in probability units, not cycle or transitivity estimates.

Permutations use one tier permutation shared by all reference curves of a
ladder, preserving their dependence; ladders receive independent permutations.
The same seed and sorted ladder IDs align permutations across models. Exact
constant curves have undefined R2 and are counted separately. A second R2
column applies the historical constant=1 convention for comparability. This
handles the undefined R-squared of constant curves without a flatness analysis.
The nonconstant definition uses exact equality, not a fitted flatness threshold.

The synthetic agent fits a regularized logistic ordered-utility model to actual
choice counts. Each ladder has nondecreasing utilities; the 30 reference
utilities are shared across ladders. Identical outcome text also receives a
shared utility, using exact linear equality constraints. The current duplicate
links form a forest of ladders, allowing a convex bounded parameterization.
Future instruments with cycles in these equality links are explicitly rejected
until supported by a more general constrained optimizer. Logistic noise scale is fixed to one for
identification. We fit only curves with every requested response observed and
exclude ladders whose input wording differs from canonical inputs. Removing
incomplete curves can select an easier subset; observed scores beside the
simulation use exactly that same subset and are not full-cohort scores.

The primary simulation assumes independent Bernoulli choices and the observed
two-order trial count. Equal probabilities in both orders are assumed; pooled
counts cannot identify position bias. Different tier/reference cells are
simulated independently. Additional beta-binomial scenarios set
within-order repeat correlation to 0.5 or 1; these are sensitivity assumptions,
not estimated correlations. Temperature-zero decoding motivates sensitivity
checks but does not determine a noise correlation from aggregate counts.
Half/double logistic-noise scales vary utility-gap-to-noise ratios; changing
trials per order to 1 or 50 varies output resolution and is not the observed
protocol. GLM Base's probabilities remain in descriptive/permutation analyses,
but its rounded pseudo-counts are excluded from finite-trial calibration.

Simulation bands are conditional replicate quantiles; they do not include
parameter-fitting uncertainty and are not confidence intervals or a universal
coherence ceiling. Calibration uses the same data being described. Fit RMSE,
optimizer status and fitted parameters are exported for checking that assumption.
Monte Carlo SE describes precision of the simulated mean, not empirical error.
Permutation bands are reference-distribution quantiles, not empirical CIs or
formal evidence under a justified tier-exchangeability hypothesis. No p-values
are claimed. Adjacent accuracy bounds reflect missing responses only.

Input hashes are verified against the supplied source manifests. The within-
ladder input is an archived aggregate summary; this tool validates IDs and
counts but does not re-audit its underlying request text. Category counts are
design counts, not response coverage. All outputs remain provisional until
known input repairs and outstanding provenance decisions are resolved. Rerun
with refreshed manifests and a new output directory after those repairs.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sources", type=Path, required=True, help="CSV: model_key,ladder_id,path,sha256")
    parser.add_argument("--source-manifest", type=Path, required=True, help="JSON with verified within_summary sources")
    parser.add_argument("--ladders", type=Path, required=True)
    parser.add_argument("--reference-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New/empty directory; always provisional")
    parser.add_argument("--models", nargs="+", help="Optional model subset")
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--allow-input-mismatches", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if min(args.permutations, args.simulations) < 2 or args.seed < 0 or args.ridge <= 0:
        parser.error("Replicates >= 2, seed >= 0 and ridge > 0 are required")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty; existing results will not be overwritten")
    ladders = read_ladders(args.ladders)
    sources = read_sources(args.raw_sources)
    manifest_rows = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    summaries = {row["model_key"]: row for row in manifest_rows}
    if len(summaries) != len(manifest_rows):
        raise ValueError("Duplicate models in source manifest")
    models = sorted(args.models or sources)
    if len(models) != len(set(models)) or any(m not in sources or m not in summaries for m in models):
        raise ValueError("Unknown or duplicate selected models")
    ref_data = json.loads(args.reference_sample.read_text(encoding="utf-8"))["comparison_sample"]
    refs = {row["sample_index"]: row["text"] for row in ref_data}
    if len(refs) != len(ref_data) or not refs:
        raise ValueError("Duplicate or empty reference sample")
    output.mkdir(parents=True, exist_ok=True)
    run = {"status": "running", "provisional": True, "started_utc": datetime.now(timezone.utc).isoformat(),
           "argv": list(argv) if argv is not None else sys.argv[1:],
           "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "models": models, "python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__, "calibration_status": {},
           "input_files": {str(path.resolve()): digest(path) for path in
                           (args.raw_sources, args.source_manifest, args.ladders, args.reference_sample)},
           "code_files": {str(path): digest(path) for path in
                          (Path(__file__), Path(__file__).with_name("coherent_baseline.py"),
                           Path(__file__).with_name("rebuttal_inputs.py"))}}
    write_json(output / "run_manifest.json", run)
    (output / "METHODS_AND_LIMITATIONS.txt").write_text(METHOD_NOTES, encoding="utf-8")
    curve_data, model_data, bins_data, transitions_data = [], [], [], []
    within_data, permutation_data, simulation_data, validation = [], [], [], []
    try:
        write_csv(output / "category_counts.csv", category_rows(ladders, len(refs)))
        for index, model in enumerate(models, 1):
            print(f"[{index}/{len(models)}] {model}: verify inputs and analyze", flush=True)
            panel = load_panel(model, sources[model], ladders, canonical_references=refs,
                               expected_references=len(refs), allow_input_mismatches=args.allow_input_mismatches)
            validation.append(panel.validation)
            within_data.extend(within_distance_rows(summaries[model], ladders))
            stats, bins, transitions, drops, finite = violation_statistics(panel.probabilities)
            model_data.append({"model_key": model, "scoring_kind": panel.scoring_kind,
                               "input_mismatch_ladders": int(panel.input_mismatch_ladders.sum()), **stats})
            bins_data.extend({"model_key": model, **row} for row in bins)
            transitions_data.extend({"model_key": model, **row} for row in transitions)
            curve_data.extend(curve_rows(panel, drops, finite))
            perm, replicates = permutation_reference(panel.probabilities, repetitions=args.permutations, seed=args.seed)
            flat = {"model_key": model, "curves_analyzed": perm["curves_analyzed"],
                    "constant_curves": perm["constant_curves"], "nonconstant_curves": perm["nonconstant_curves"]}
            for metric, value in perm["observed"].items():
                flat[metric + "_observed"] = value
                for key, val in perm["permuted"].get(metric, {}).items():
                    flat[metric + "_null_" + key] = val
            permutation_data.append(flat)
            model_dir = output / "models" / model
            model_dir.mkdir(parents=True)
            write_json(model_dir / "permutation_reference.json", perm)
            write_json(model_dir / "permutation_replicates.json", replicates)
            calibration, simulated, fit_arrays = noisy_agent_reference(
                panel, repetitions=args.simulations, seed=args.seed, ridge=args.ridge)
            write_json(model_dir / "calibration.json", calibration)
            run["calibration_status"][model] = calibration["status"]
            simulation_data.extend(simulated)
            if fit_arrays is not None:
                np.savez_compressed(model_dir / "fitted_probabilities.npz",
                                    probabilities=fit_arrays[0], eligible=fit_arrays[1])
            print(f"  {stats['curves_analyzed']} curves; {calibration['status']}; "
                  f"{panel.validation['input_mismatch_ladders']} input-mismatch ladders", flush=True)
        adjacent = [row for row in within_data if row["tier_distance"] == 1]
        for filename, rows in (("curves.csv", curve_data), ("model_diagnostics.csv", model_data),
                               ("violation_size_bins.csv", bins_data), ("violation_by_transition.csv", transitions_data),
                               ("within_accuracy_by_distance.csv", within_data), ("adjacent_accuracy.csv", adjacent),
                               ("permutation_baselines.csv", permutation_data), ("noisy_agent_scenarios.csv", simulation_data)):
            write_csv(output / filename, rows)
        write_json(output / "input_validation.json", validation)
        if not args.skip_plots:
            make_plots(output, adjacent, bins_data, simulation_data, permutation_data)
        run["status"] = ("incomplete_calibration" if any(status in {"calibration_failed", "no_eligible_curves"}
                          for status in run["calibration_status"].values()) else "complete")
        run["completed_utc"] = datetime.now(timezone.utc).isoformat()
        run["output_files"] = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
                               if path.is_file() and path.name != "run_manifest.json"}
    except Exception as exc:
        run["status"], run["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "run_manifest.json", run)
    print(f"Provisional outputs: {output}", flush=True)
    return 0 if run["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
