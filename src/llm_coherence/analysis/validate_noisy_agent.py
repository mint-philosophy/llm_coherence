"""Offline validation of the coherent agent's response assumptions.

This keeps the original simulation and manuscript frozen. It checks held-out
curve predictions and verified individual responses; it does not estimate a
repeat correlation from a single response block or launch inference jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import betaln
from scipy.stats import hypergeom

from .coherent_baseline import fit_ordered_utilities
from .noise_response_inputs import recover_trials
from .rebuttal_inputs import digest, read_ladders, read_sources, load_panel
from .rebuttal_analysis import write_csv, write_json


def saved_fit(panel, baseline):
    directory = Path(baseline) / "models" / panel.model_key
    metadata = json.loads((directory / "calibration.json").read_text())
    if metadata["status"] != "complete":
        return None, None, metadata
    li = [panel.ladder_ids.index(key) for key in metadata["calibration_ladders"]]
    ri = [panel.reference_ids.index(key) for key in metadata["calibration_reference_ids"]]
    p = np.full(panel.probabilities.shape, np.nan)
    eligible = np.zeros(panel.probabilities.shape[:-1], bool)
    with np.load(directory / "fitted_probabilities.npz") as data:
        p[np.ix_(li, ri)] = data["probabilities"]
        eligible[np.ix_(li, ri)] = data["eligible"]
    expected = (np.isfinite(panel.probabilities).all(axis=-1) &
                np.all(panel.totals == 2 * panel.trials_per_direction, axis=-1) &
                ~panel.input_mismatch_ladders[:, None])
    if not np.array_equal(expected, eligible):
        raise ValueError("Selected inputs no longer match the simulation cohort")
    return p, eligible, metadata


def prediction_error(predicted, successes, totals, eligible):
    mask = eligible & np.isfinite(predicted).all(axis=-1)
    if not mask.any():
        return {"curves": 0, "rmse": None, "mae": None}
    p, k, n = predicted[mask], successes[mask], totals[mask]
    observed = k / n
    return {"curves": int(mask.sum()), "rmse": float(np.sqrt(np.mean((observed - p) ** 2))),
            "mae": float(np.mean(abs(observed - p))),
            "mean_observed_probability": float(observed.mean()),
            "mean_predicted_probability": float(p.mean()),
            "independent_binomial_expected_mse": float(np.mean(p * (1 - p) / n))}


def cross_validate_curves(successes, totals, eligible, outcome_ids, *, folds=3, seed=42, ridge=1e-5):
    """Hold out whole seven-tier reference curves; never split their trials.

    This evaluates new comparisons within the existing ladder/reference design,
    not generalization to a new ladder or to a new population of prompts.
    """
    if folds < 2 or folds > eligible.shape[1]:
        raise ValueError("Fold count must be between two and the reference count")
    rng = np.random.default_rng(seed)
    assignment = np.empty(eligible.shape, int)
    for li in range(len(assignment)):
        assignment[li, rng.permutation(eligible.shape[1])] = np.arange(eligible.shape[1]) % folds
    predictions = np.full(successes.shape, np.nan)
    diagnostics = []
    for fold in range(folds):
        train = eligible & (assignment != fold)
        li, ri = np.flatnonzero(train.any(axis=1)), np.flatnonzero(train.any(axis=0))
        ix = np.ix_(li, ri)
        selected_train = train[ix]
        test = eligible[ix] & (assignment[ix] == fold)
        info = {"fold": fold, "training_curves": int(train.sum()), "test_curves": int(test.sum()),
                "unsupported_test_curves": int((eligible & (assignment == fold)).sum() - test.sum())}
        if not test.any():
            info["status"] = "no_supported_test_curves"
            diagnostics.append(info)
            continue
        fit = fit_ordered_utilities(np.where(selected_train[..., None], successes[ix], 0),
                                    np.where(selected_train[..., None], totals[ix], 0), ridge=ridge,
                                    outcome_ids=np.asarray(outcome_ids)[li])
        info.update(fit.diagnostics)
        info["status"] = "complete" if fit.diagnostics["optimizer_success"] else "calibration_failed"
        if info["status"] == "complete":
            local = predictions[ix]
            local[test] = fit.probabilities[test]
            predictions[ix] = local
        diagnostics.append(info)
    return predictions, assignment, diagnostics


def calibration_bins(predicted, successes, totals, eligible, label):
    mask = eligible & np.isfinite(predicted).all(axis=-1)
    p, observed = predicted[mask].ravel(), (successes[mask] / totals[mask]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for b in range(10):
        chosen = bins == b
        rows.append({"prediction_source": label, "bin_lower": b / 10, "bin_upper": (b + 1) / 10,
                     "cells": int(chosen.sum()),
                     "mean_predicted": float(p[chosen].mean()) if chosen.any() else None,
                     "mean_observed": float(observed[chosen].mean()) if chosen.any() else None})
    return rows


def unanimity_probabilities(p, n, rho):
    """Probability of zero or n successes in one order's n-choice block."""
    p = np.asarray(p, float)
    if not 0 <= rho <= 1 or n < 2:
        raise ValueError("Invalid repeat-correlation scenario or trial count")
    if rho == 0:
        return (1 - p) ** n, p ** n
    if rho == 1:
        return 1 - p, p
    concentration = 1 / rho - 1
    a, b = p * concentration, (1 - p) * concentration
    zero, full = np.zeros_like(p), np.zeros_like(p)
    interior = (p > 0) & (p < 1)
    zero[p == 0], full[p == 1] = 1, 1
    zero[interior] = np.exp(betaln(a[interior], b[interior] + n) - betaln(a[interior], b[interior]))
    full[interior] = np.exp(betaln(a[interior] + n, b[interior]) - betaln(a[interior], b[interior]))
    return zero, full


def trial_diagnostics(labels, predicted, eligible, source_codes):
    if labels.shape[:-2] != predicted.shape or labels.shape[-2] != 2:
        raise ValueError("Trial array does not match probability grid")
    # Only the exact original calibration cohort, with all 20 valid responses
    # recovered, supports the matching response-process comparison.
    mask = (eligible[..., None] & (source_codes > 0) &
            np.all(labels >= 0, axis=(-1, -2)) & np.isfinite(predicted))
    n = labels.shape[-1]
    if n < 2:
        raise ValueError("At least two repeated choices per order are needed")
    if not mask.any():
        return {"status": "no_verified_trials_on_calibration_cohort", "cells": 0}, [], []
    k = labels[mask].sum(axis=-1)
    q = k / n
    delta = q[:, 0] - q[:, 1]
    pooled = k.sum(axis=-1)
    unanimous = (k == 0) | (k == n)
    opposite = ((k[:, 0] == n) & (k[:, 1] == 0)) | ((k[:, 0] == 0) & (k[:, 1] == n))
    within_disagreement = 2 * k * (n - k) / (n * (n - 1))
    # Conditional random allocation of the same observed total successes to
    # the two orders. This avoids relying on fitted utility probabilities.
    expected_unanimity = hypergeom.pmf(0, 2 * n, pooled, n) + hypergeom.pmf(n, 2 * n, pooled, n)
    expected_order_mse = pooled * (2 * n - pooled) / (n ** 2 * (2 * n - 1))
    stats = {"status": "complete", "cells": int(mask.sum()), "response_groups": int(2 * mask.sum()),
             "responses": int(2 * n * mask.sum()), "trials_per_order": n,
             "calibration_cells": int(eligible.sum() * predicted.shape[-1]),
             "coverage_fraction": float(mask.sum() / (eligible.sum() * predicted.shape[-1])),
             "observed_unanimous_group_fraction": float(unanimous.mean()),
             "observed_mixed_group_fraction": float((~unanimous).mean()),
             "observed_within_group_pair_disagreement": float(within_disagreement.mean()),
             "mean_order_difference": float(delta.mean()),
             "mean_absolute_order_difference": float(abs(delta).mean()),
             "mean_squared_order_difference": float(np.mean(delta ** 2)),
             "opposite_unanimous_orders_fraction": float(opposite.mean()),
             "opposite_majority_orders_fraction": float(((q[:, 0] - .5) * (q[:, 1] - .5) < 0).mean()),
             "conditional_random_split_expected_unanimity": float(expected_unanimity.mean()),
             "conditional_random_split_expected_order_mse": float(expected_order_mse.mean()),
             "conditional_random_split_expected_signed_order_difference": 0.0}
    expected, distribution = [], []
    p = predicted[mask]
    for rho in (0.0, .5, 1.0):
        zero, full = unanimity_probabilities(p, n, rho)
        expected.append({"repeat_correlation": rho,
                         "expected_unanimous_group_fraction": float((zero + full).mean()),
                         "expected_within_group_pair_disagreement": float((2 * p * (1 - p) * (1 - rho)).mean()),
                         "expected_mean_order_difference": 0.0,
                         "expected_mean_squared_order_difference": float((2 * p * (1 - p) * (1 + (n - 1) * rho) / n).mean()),
                         "expected_opposite_unanimous_orders_fraction": float((2 * zero * full).mean())})
    for ka in range(n + 1):
        for kb in range(n + 1):
            distribution.append({"tier_choices_original": ka, "tier_choices_flipped": kb,
                                 "cells": int(((k[:, 0] == ka) & (k[:, 1] == kb)).sum())})
    return stats, expected, distribution


NOTES = """PROVISIONAL VALIDATION OF THE NOISY COHERENT AGENT

This is a local diagnostic run. No model calls, manuscript changes, or changes
to the original simulation are performed. Interpret the results jointly with
the original METHODS_AND_LIMITATIONS.txt.

Response recovery requires exact reconciliation with each selected result's
A/B counts. The original and flipped orders are mapped to the same outcome
before comparison. Embedded response arrays are preferred. Legacy response
sidecars require a byte-identical copy of the selected results. Modern batches
are linked by run ID, manifest, comparison text/order, and input-file hashes.
Legacy bulk batches require matching model IDs and displayed outcome text,
then aggregate reconciliation. Duplicate choices collapse; conflicting choices
are marked ambiguous. Complete legacy sequential logs are accepted only from
reviewed producer commits, with exactly ten unique trial indices per awaited
direction batch and exact count reconciliation for every comparison. Partial
logs, unknown producer versions, and mixed schemas are rejected in full. Other
unkeyed trace rows never supply quantitative trials. Legacy recovery uses the
recorded execution order; it does not infer a grouping from matching counts.
Missing responses are not imputed from aggregate counts. Every reported
response-process statistic uses complete recovered cells within the same
eligible-curve cohort used to calibrate the original simulation. Coverage may
be selective; partial coverage is explicitly reported and must not be treated
as representative of an entire model.

Positive signed order difference means the tier outcome is selected more often
when shown as Option A than when shown as Option B. Absolute/squared gaps can
also arise from finite sampling or correlated repeats. A single ten-response
block per order cannot by itself separate an order-specific underlying choice
probability from dependence between repeats. Unanimity is not an estimate of
repeat correlation: extreme choice probabilities can also produce unanimity.

The fixed-prediction reference expectations are exact moments of the three
existing noise scenarios (repeat correlation 0, 0.5, 1), holding the fitted
utilities fixed. All scenarios assume equal probabilities in both orders and
independent order blocks. The additional conditional random-split reference
randomly allocates the observed total successes between the two orders under
exchangeability; it does not rely on the fitted utility model. Disagreement
with that reference can reflect presentation effects, dependence, or both.
These are descriptive model checks, not p-values or population confidence
intervals. We do not infer one true correlation coefficient from these checks.

Three-fold validation holds out complete seven-tier reference curves, including
all their individual trials. Each fitted utility model sees only training
curves. Existing ladder and reference outcomes also occur in other training
comparisons, so this assesses prediction of held-out comparisons within the
existing design, not generalization to unseen ladders or new model runs.
Unsupported held-out curves and failed fits are reported. In-sample calibration
bins and held-out bins are labeled separately. Squared prediction error is
affected by both sampling variability and utility-model misspecification;
it is not a pure estimate of behavioral noise.

This validation does not repair the 116 known model/ladder input mismatches,
estimate general manuscript confidence intervals, or implement a
flatness/saturation analysis. The original simulation remains a
conditional sensitivity reference until its behavioral assumptions are justified.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("raw-sources", "ladders", "reference-sample", "baseline", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovered-trials", type=Path,
                        help="Optional cached choices/provenance; hashes of every source are verified")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    baseline_run = json.loads((args.baseline / "run_manifest.json").read_text())
    for name, expected in baseline_run["output_files"].items():
        if digest(args.baseline / name) != expected:
            raise ValueError(f"Baseline output hash mismatch: {name}")
    sources = read_sources(args.raw_sources)
    ladders = read_ladders(args.ladders)
    refs = {r["sample_index"]: r["text"] for r in json.loads(args.reference_sample.read_text())["comparison_sample"]}
    models = sorted(args.models or sources)
    args.output.mkdir(parents=True, exist_ok=True)
    run = {"status": "running", "provisional": True, "started_utc": datetime.now(timezone.utc).isoformat(),
           "argv": list(argv) if argv is not None else sys.argv[1:], "folds": args.folds, "seed": args.seed,
           "source_files": {str(p.resolve()): digest(p) for p in
                            (args.raw_sources, args.ladders, args.reference_sample, args.baseline / "run_manifest.json")},
           "code_files": {str(p.resolve()): digest(p) for p in
                          (Path(__file__), Path(__file__).with_name("noise_response_inputs.py"),
                           Path(__file__).with_name("coherent_baseline.py"))}}
    write_json(args.output / "run_manifest.json", run)
    (args.output / "METHODS_AND_LIMITATIONS.txt").write_text(NOTES, encoding="utf-8")
    coverage, fits, bins, repetitions, scenarios, distributions = [], [], [], [], [], []
    incomplete = False
    try:
        for index, model in enumerate(models, 1):
            print(f"[{index}/{len(models)}] {model}: recover choices and validate", flush=True)
            panel = load_panel(model, sources[model], ladders, canonical_references=refs,
                               expected_references=len(refs), allow_input_mismatches=True)
            p, eligible, calibration = saved_fit(panel, args.baseline)
            if p is None:
                coverage.append({"model_key": model, "status": calibration["status"], "reconciled_cells": 0})
                continue
            directory = args.output / "models" / model
            directory.mkdir(parents=True)
            cache = args.recovered_trials / model if args.recovered_trials else None
            if cache is not None and (cache / "choices.npz").exists():
                audit = json.loads((cache / "provenance.json").read_text())
                if (audit["model_key"] != model or audit["ladder_ids"] != panel.ladder_ids or
                        audit["reference_ids"] != panel.reference_ids or
                        audit["reader_code_sha256"] != digest(Path(__file__).with_name("noise_response_inputs.py"))):
                    raise ValueError("Cached response grid or reader version differs")
                for name, expected in audit["source_files"].items():
                    if digest(Path(name)) != expected:
                        raise ValueError(f"Recovered response source changed: {name}")
                with np.load(cache / "choices.npz") as data:
                    labels, codes = data["labels"], data["source_codes"]
                recovered = codes > 0
                if (labels.shape != panel.probabilities.shape + (2, panel.trials_per_direction) or
                        not np.array_equal((labels == 1).sum(axis=(-1, -2))[recovered], panel.successes[recovered]) or
                        not np.array_equal((labels >= 0).sum(axis=(-1, -2))[recovered], panel.totals[recovered])):
                    raise ValueError("Cached trials do not reconcile to selected results")
                run["source_files"][str(cache / "choices.npz")] = digest(cache / "choices.npz")
                run["source_files"][str(cache / "provenance.json")] = digest(cache / "provenance.json")
            else:
                labels, codes, audit = recover_trials(panel, sources[model], legacy_root=args.legacy_root)
            np.savez_compressed(directory / "choices.npz", labels=labels, source_codes=codes)
            write_json(directory / "response_provenance.json", audit)
            coverage.append({"model_key": model, "status": "audited", **audit["coverage"]})
            stats, expected, joint = trial_diagnostics(labels, p, eligible, codes)
            repetitions.append({"model_key": model, **stats})
            scenarios.extend({"model_key": model, **row} for row in expected)
            distributions.extend({"model_key": model, **row} for row in joint)
            print(f"  reconciled {audit['coverage']['reconciled_cells']}/{audit['coverage']['cells_total']} cells; cross-validating", flush=True)
            heldout, assignment, fold_info = cross_validate_curves(
                panel.successes, panel.totals, eligible, panel.tier_texts,
                folds=args.folds, seed=args.seed, ridge=calibration["ridge"])
            write_json(directory / "cross_validation_folds.json", fold_info)
            np.savez_compressed(directory / "heldout_predictions.npz", probabilities=heldout, folds=assignment, eligible=eligible)
            failed = any(row["status"] != "complete" for row in fold_info)
            incomplete |= failed
            insample = prediction_error(p, panel.successes, panel.totals, eligible)
            outsample = prediction_error(heldout, panel.successes, panel.totals, eligible)
            fits.append({"model_key": model, "status": "incomplete_cv" if failed else "complete",
                         **{"in_sample_" + k: v for k, v in insample.items()},
                         **{"heldout_" + k: v for k, v in outsample.items()}})
            bins.extend({"model_key": model, **row} for pred, label in ((p, "in_sample"), (heldout, "heldout"))
                        for row in calibration_bins(pred, panel.successes, panel.totals, eligible, label))
            print(f"  trial coverage {stats.get('coverage_fraction', 0):.1%}; "
                  f"fit RMSE {insample['rmse']:.3f}; held-out RMSE {outsample['rmse']:.3f}", flush=True)
        for name, rows in (("response_coverage.csv", coverage), ("fit_validation.csv", fits),
                           ("calibration_bins.csv", bins), ("response_process.csv", repetitions),
                           ("response_scenario_expectations.csv", scenarios), ("order_count_distribution.csv", distributions)):
            write_csv(args.output / name, rows)
        run["status"] = "incomplete_cross_validation" if incomplete else "complete"
        run["completed_utc"] = datetime.now(timezone.utc).isoformat()
        run["output_files"] = {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob("*"))
                               if p.is_file() and p.name != "run_manifest.json"}
    except Exception as exc:
        run["status"], run["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(args.output / "run_manifest.json", run)
    return int(incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
