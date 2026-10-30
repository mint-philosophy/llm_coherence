"""Offline, matched-cohort evaluation of a position-aware coherent agent."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from .ordered_response import fit_ordered_responses, order_block_pmf, simulate_order_monotonicity
from .rebuttal_analysis import write_csv, write_json
from .rebuttal_inputs import digest, load_panel, read_ladders, read_sources
from .validate_noisy_agent import saved_fit


def verified_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete upstream run: {directory}")
    for name, expected in manifest["output_files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Upstream output hash mismatch: {directory / name}")
    return manifest


def verify_validation_inputs(validation, source_paths, reader):
    """Bind reused inputs to verified bytes, independently of checkout location.

    Absolute paths remain provenance metadata, not machine-specific identities.
    Relocating unchanged manifests or code is safe; changing their bytes still
    requires a new upstream validation. A reader name with conflicting recorded
    hashes is ambiguous and is rejected.
    """
    source_hashes = set(validation["source_files"].values())
    for path in source_paths:
        if digest(path) not in source_hashes:
            raise ValueError(f"Validation used different source inputs: {path}")
    reader_hashes = {sha for name, sha in validation["code_files"].items()
                     if Path(name).name == Path(reader).name}
    if reader_hashes != {digest(reader)}:
        raise ValueError("Response reader differs from verified validation")


def matched_curve_counts(panel, labels, codes, original_eligible):
    """Admit a curve only if all seven cells have verified, complete trials."""
    n = panel.trials_per_direction
    if (labels.shape != panel.probabilities.shape + (2, n) or
            codes.shape != panel.probabilities.shape or
            original_eligible.shape != panel.probabilities.shape[:-1]):
        raise ValueError("Response cache grid differs from the result panel")
    if not np.isin(labels, [-3, -2, -1, 0, 1]).all():
        raise ValueError("Unknown recovered response label")
    recovered = codes > 0
    if (not np.array_equal((labels == 1).sum(axis=(-1, -2))[recovered], panel.successes[recovered]) or
            not np.array_equal((labels >= 0).sum(axis=(-1, -2))[recovered], panel.totals[recovered])):
        raise ValueError("Cached responses do not reconcile to saved counts")
    complete = recovered & np.all(labels >= 0, axis=(-1, -2))
    eligible = original_eligible & complete.all(axis=-1)
    successes = np.where(eligible[..., None, None], (labels == 1).sum(axis=-1), 0)
    totals = np.where(eligible[..., None, None], np.full(successes.shape, n), 0)
    return successes, totals, eligible


def fold_assignment(shape, folds, seed):
    if not 2 <= folds <= shape[1]:
        raise ValueError("Fold count must be between two and the reference count")
    rng = np.random.default_rng(seed)
    assignment = np.empty(shape, int)
    for li in range(shape[0]):
        assignment[li, rng.permutation(shape[1])] = np.arange(shape[1]) % folds
    return assignment


def fit_cohort(successes, totals, train, outcome_ids, *, position_effect, ridge):
    li, ri = np.flatnonzero(train.any(axis=1)), np.flatnonzero(train.any(axis=0))
    if not len(li) or not len(ri):
        raise ValueError("No supported training comparisons")
    ix = np.ix_(li, ri)
    mask = train[ix][..., None, None]
    fit = fit_ordered_responses(np.where(mask, successes[ix], 0), np.where(mask, totals[ix], 0),
                               position_effect=position_effect, ridge=ridge,
                               outcome_ids=np.asarray(outcome_ids)[li])
    return fit, li, ri


def cross_validate_orders(successes, totals, eligible, outcome_ids, *,
                           position_effect, assignment, ridge=1e-5):
    """Hold out a whole curve's seven tiers and both complete response blocks."""
    if assignment.shape != eligible.shape:
        raise ValueError("Fold assignment does not match the curve grid")
    predictions = np.full(successes.shape, np.nan)
    diagnostics = []
    for fold in sorted(np.unique(assignment)):
        train = eligible & (assignment != fold)
        requested = eligible & (assignment == fold)
        info = {"fold": int(fold), "training_curves": int(train.sum()),
                "requested_test_curves": int(requested.sum())}
        if not train.any():
            diagnostics.append({**info, "status": "no_training_curves",
                                "unsupported_test_curves": int(requested.sum())})
            continue
        fit, li, ri = fit_cohort(successes, totals, train, outcome_ids,
                                 position_effect=position_effect, ridge=ridge)
        ix = np.ix_(li, ri)
        test = requested[ix]
        info.update(fit.diagnostics)
        info.update({"test_curves": int(test.sum()),
                     "unsupported_test_curves": int(requested.sum() - test.sum()),
                     "status": "complete" if fit.diagnostics["optimizer_success"] else "calibration_failed"})
        if info["status"] == "complete":
            local = predictions[ix]
            local[test] = fit.probabilities[test]
            predictions[ix] = local
        diagnostics.append(info)
    return predictions, diagnostics


def prediction_metrics(predicted, successes, totals, eligible):
    mask = eligible & np.isfinite(predicted).all(axis=(-1, -2))
    if not mask.any():
        return {"curves": 0, "order_probability_rmse": None}
    p, k, n = predicted[mask], successes[mask], totals[mask]
    if np.any(n <= 0):
        raise ValueError("Validation requires complete positive trial counts")
    q = k / n
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    pooled_q, pooled_p = k.sum(axis=-1) / n.sum(axis=-1), (p * n).sum(axis=-1) / n.sum(axis=-1)
    return {
        "curves": int(mask.sum()), "responses": int(n.sum()),
        "order_probability_rmse": float(np.sqrt(np.mean((p - q) ** 2))),
        "pooled_probability_rmse": float(np.sqrt(np.mean((pooled_p - pooled_q) ** 2))),
        "per_choice_log_loss": float(-np.sum(k * np.log(clipped) + (n - k) * np.log1p(-clipped)) / n.sum()),
        "per_choice_brier_score": float(np.sum(k * (1 - p) ** 2 + (n - k) * p ** 2) / n.sum()),
        "observed_mean_order_gap": float((q[..., 0] - q[..., 1]).mean()),
        "predicted_mean_order_gap": float((p[..., 0] - p[..., 1]).mean()),
    }


def repeat_checks(predicted, successes, totals, eligible, correlations=(0., .5, 1.)):
    """Compare saved repeat behavior with conditional per-order predictions.

    Excess error can reflect probability misspecification, order heterogeneity,
    or dependent repeats. It is deliberately not converted into an estimated
    behavioral correlation. The three correlations are sensitivity scenarios.
    """
    mask = eligible & np.isfinite(predicted).all(axis=(-1, -2))
    if not mask.any():
        return []
    p, k, totals = predicted[mask], successes[mask], totals[mask]
    if not np.all(totals == totals.flat[0]) or totals.flat[0] < 2:
        raise ValueError("Repeat checks require equal block sizes of at least two")
    n = int(totals.flat[0])
    q = k / n
    opposite = ((k[..., 0] == 0) & (k[..., 1] == n)) | ((k[..., 0] == n) & (k[..., 1] == 0))
    observed = {
        "curves": int(mask.sum()), "cells": int(p.shape[0] * p.shape[1]), "trials_per_order": n,
        "observed_unanimous_group_fraction": float(((k == 0) | (k == n)).mean()),
        "observed_opposite_unanimous_orders_fraction": float(opposite.mean()),
        "observed_within_group_pair_disagreement": float((2 * k * (n - k) / (n * (n - 1))).mean()),
        "observed_mean_order_gap": float((q[..., 0] - q[..., 1]).mean()),
        "observed_mean_squared_order_gap": float(np.mean((q[..., 0] - q[..., 1]) ** 2)),
        "observed_group_prediction_mse": float(np.mean((q - p) ** 2)),
    }
    rows = []
    for rho in correlations:
        pmf = order_block_pmf(p, n, rho)
        zero, full = pmf[..., 0], pmf[..., -1]
        variance = p * (1 - p) * (1 + (n - 1) * rho) / n
        rows.append({**observed, "repeat_correlation_scenario": rho,
                     "expected_unanimous_group_fraction": float((zero + full).mean()),
                     "expected_opposite_unanimous_orders_fraction": float(
                         (zero[..., 0] * full[..., 1] + full[..., 0] * zero[..., 1]).mean()),
                     "expected_within_group_pair_disagreement": float((2 * p * (1 - p) * (1 - rho)).mean()),
                     "predicted_mean_order_gap": float((p[..., 0] - p[..., 1]).mean()),
                     "expected_mean_squared_order_gap": float(np.mean(
                         (p[..., 0] - p[..., 1]) ** 2 + variance.sum(axis=-1))),
                     "expected_group_prediction_mse": float(variance.mean())})
    return rows


NOTES = """POSITION-AWARE COHERENT-AGENT REPAIR — OFFLINE AND PROVISIONAL

The original simulation remains archived. This run compares two response rules
on exactly the same complete, verified seven-tier curves. Every admitted cell
has both ten-response order blocks and reconciles exactly to the selected raw
result counts. Curves with missing/unparseable/ambiguous responses are omitted;
known changed-input ladders remain excluded. Selection may still affect results.
GLM Base continuous probabilities are never treated as observed trial counts.
The recorded coverage threshold defines whether this run attempts calibration;
it does not establish representativeness for the remaining missing responses.

Both response models have shared reference utilities, seven nondecreasing tier
utilities per ladder, and equality constraints for identical tier texts. The
reference-0 utility is fixed to zero and the logistic scale to one. The neutral
rule fixes position offset delta=0. The adjusted rule jointly estimates one
delta per evaluated model: P(tier | tier displayed A) = sigmoid(u-v+delta), and
P(tier | tier displayed B) = sigmoid(u-v-delta). Thus positive delta favors A.
The weak quadratic penalty includes delta squared, and uses the same utility
penalty in both models. Underlying utility coherence is retained. A single
global delta cannot represent comparison-specific effects of opposite signs.

Calibration minimizes a binomial working loss; independence is not assumed
proven by optimization. Three folds hold out an entire seven-tier reference
curve, including all its trials and both orders. Identical fold assignments
and evaluation masks are used for the neutral and adjusted models. All fitted
parameters for a fold use its training curves only; unsupported test curves
are reported and omitted from both error scores. Prediction scores evaluate
comparisons within the existing design, not new ladders, new runs, or a new
population of statements. Log loss and Brier score are per choice. Probability
RMSE compares predicted probabilities with observed finite-block proportions.
There is no claim of statistical significance from a small numerical gain.

Repeat checks use held-out, order-specific predictions. Their residual error
mixes response variability with probability-model misspecification. One block
per order does not uniquely identify behavioral repeat dependence. Correlations
0, 0.5 and 1 remain sensitivity scenarios, not fitted or selected estimates.
Order blocks and cells are independent in these scenarios; within each block,
rho=0 is binomial, intermediate rho is beta-binomial, and rho=1 repeats one
Bernoulli outcome. These assumptions do not capture every form of dependence.

Simulation scenarios use the full-cohort fitted coherent utilities, separately
from held-out validation. The probability of a nondecreasing pooled seven-tier
count sequence is calculated exactly by convolving the two order-count PMFs
and using a cumulative-sum recursion across tiers. Replicate monotonicity rates
sample the resulting curve pass probabilities, giving the same score distribution
as full response simulation under independent cells. Ties count as monotonic.
Reported replicate intervals describe this conditional synthetic distribution;
they exclude utility-estimation uncertainty and are not empirical confidence
intervals or an established coherence ceiling. No scenario is chosen because
it reproduces the observed monotonicity score.

The response cache is checked against the completed validation manifest, grid,
parser hash, and raw-result counts. Original source recovery provenance is
retained by reference; this run does not reinterpret the original API messages.
No model/API calls or manuscript changes are made. Known GLM/input replacements
and other analysis workflows are outside this repair.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-sources", "ladders", "reference-sample", "baseline", "validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--minimum-curve-coverage", type=float, default=.9)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    if not 0 < args.minimum_curve_coverage <= 1 or args.repetitions < 2:
        parser.error("Invalid minimum coverage or simulation replicate count")
    verified_run(args.baseline)
    validation = verified_run(args.validation)
    reader = Path(__file__).with_name("noise_response_inputs.py")
    verify_validation_inputs(validation,
                             (args.raw_sources, args.ladders, args.reference_sample,
                              args.baseline / "run_manifest.json"), reader)
    sources, ladders = read_sources(args.raw_sources), read_ladders(args.ladders)
    refs = {r["sample_index"]: r["text"] for r in json.loads(args.reference_sample.read_text())["comparison_sample"]}
    models = sorted(args.models or sources)
    if set(models) - sources.keys():
        parser.error("Unknown model key")
    args.output.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__)] + [Path(__file__).with_name(name + ".py") for name in
                  ("ordered_response", "coherent_baseline", "rebuttal_inputs", "rebuttal_analysis", "validate_noisy_agent")]
    run = {"status": "running", "provisional": True,
           "started_utc": datetime.now(timezone.utc).isoformat(),
           "argv": list(argv) if argv is not None else sys.argv[1:],
           "folds": args.folds, "seed": args.seed, "repetitions": args.repetitions,
           "minimum_curve_coverage": args.minimum_curve_coverage,
           "python_version": platform.python_version(), "numpy_version": np.__version__, "scipy_version": scipy.__version__,
           "source_files": {str(p.resolve()): digest(p) for p in
                            (args.raw_sources, args.ladders, args.reference_sample,
                             args.baseline / "run_manifest.json", args.validation / "run_manifest.json")},
           "code_files": {str(p.resolve()): digest(p) for p in code_paths}}
    write_json(args.output / "run_manifest.json", run)
    (args.output / "METHODS_AND_LIMITATIONS.txt").write_text(NOTES, encoding="utf-8")
    coverage, comparison, repeats, simulations = [], [], [], []
    incomplete = False
    try:
        for index, model in enumerate(models, 1):
            print(f"[{index}/{len(models)}] {model}: matched-cohort order repair", flush=True)
            panel = load_panel(model, sources[model], ladders, canonical_references=refs,
                               expected_references=len(refs), allow_input_mismatches=True)
            _, original_eligible, calibration = saved_fit(panel, args.baseline)
            if original_eligible is None:
                coverage.append({"model_key": model, "status": calibration["status"], "verified_curves": 0})
                continue
            cache = args.validation / "models" / model
            audit = json.loads((cache / "response_provenance.json").read_text())
            if (audit["model_key"] != model or audit["ladder_ids"] != panel.ladder_ids or
                    audit["reference_ids"] != panel.reference_ids or audit["reader_code_sha256"] != digest(reader)):
                raise ValueError("Cached response provenance differs from the result grid/reader")
            with np.load(cache / "choices.npz", allow_pickle=False) as data:
                k, n, eligible = matched_curve_counts(panel, data["labels"], data["source_codes"], original_eligible)
            for name in ("response_provenance.json", "choices.npz"):
                run["source_files"][str((cache / name).resolve())] = digest(cache / name)
            fraction = float(eligible.sum() / original_eligible.sum()) if original_eligible.any() else 0.
            row = {"model_key": model, "original_eligible_curves": int(original_eligible.sum()),
                   "verified_curves": int(eligible.sum()), "verified_curve_fraction": fraction,
                   "verified_responses": int(n.sum()), "represented_ladders": int(eligible.any(axis=1).sum()),
                   "excluded_input_mismatch_ladders": int(panel.input_mismatch_ladders.sum())}
            if not eligible.any() or fraction < args.minimum_curve_coverage:
                row["status"] = "no_verified_complete_curves" if not eligible.any() else "insufficient_curve_coverage"
                coverage.append(row)
                print(f"  {row['status']}: {eligible.sum()}/{original_eligible.sum()} curves", flush=True)
                continue
            directory = args.output / "models" / model
            directory.mkdir(parents=True)
            assignment = fold_assignment(eligible.shape, args.folds, args.seed)
            arrays = {"eligible": eligible, "folds": assignment, "successes_by_order": k, "totals_by_order": n}
            metadata = {"model_key": model, "ladder_ids": panel.ladder_ids, "reference_ids": panel.reference_ids,
                        "response_provenance": str((cache / "response_provenance.json").resolve()), "fits": {}}
            heldout = {}
            model_complete = True
            for name, position in (("neutral", False), ("position", True)):
                fit, li, ri = fit_cohort(k, n, eligible, panel.tier_texts,
                                         position_effect=position, ridge=calibration["ridge"])
                full = np.full(k.shape, np.nan)
                full[np.ix_(li, ri)] = fit.probabilities
                heldout[name], folds = cross_validate_orders(k, n, eligible, panel.tier_texts,
                                                            position_effect=position, assignment=assignment,
                                                            ridge=calibration["ridge"])
                ok = fit.diagnostics["optimizer_success"] and all(d["status"] == "complete" for d in folds)
                model_complete &= ok
                arrays[name + "_fitted"] = full
                arrays[name + "_heldout"] = heldout[name]
                metadata["fits"][name] = {**fit.diagnostics,
                                          "fitted_ladder_indices": li.tolist(), "fitted_reference_indices": ri.tolist(),
                                          "ladder_utilities": fit.ladder_utilities.tolist(),
                                          "reference_utilities": fit.reference_utilities.tolist(), "folds": folds}
                if not fit.diagnostics["optimizer_success"]:
                    continue
                observed = float(np.all(np.diff(k.sum(axis=-1)[eligible], axis=-1) >= 0, axis=-1).mean())
                for rho in (0., .5, 1.):
                    summary = simulate_order_monotonicity(full, eligible, panel.trials_per_direction,
                                                          repeat_correlation=rho, repetitions=args.repetitions,
                                                          seed=args.seed)
                    simulations.append({"model_key": model, "response_rule": name,
                                        "calibration_source": "full_matched_cohort",
                                        "repeat_correlation_scenario": rho, "observed_monotonicity_rate": observed,
                                        **summary})
            paired = eligible & np.isfinite(heldout["neutral"]).all(axis=(-1, -2)) & np.isfinite(heldout["position"]).all(axis=(-1, -2))
            arrays["paired_heldout_eligible"] = paired
            row.update({"status": "complete" if model_complete else "incomplete_fit",
                        "paired_heldout_curves": int(paired.sum()),
                        "unsupported_or_failed_heldout_curves": int(eligible.sum() - paired.sum())})
            coverage.append(row)
            incomplete |= not model_complete
            for name in ("neutral", "position"):
                for label, pred, mask in (("in_sample", arrays[name + "_fitted"], eligible),
                                          ("heldout", heldout[name], paired)):
                    comparison.append({"model_key": model, "response_rule": name, "prediction_source": label,
                                       **prediction_metrics(pred, k, n, mask)})
                repeats.extend({"model_key": model, "response_rule": name, "prediction_source": "heldout", **r}
                               for r in repeat_checks(heldout[name], k, n, paired))
            np.savez_compressed(directory / "predictions_and_counts.npz", **arrays)
            write_json(directory / "calibration_and_folds.json", metadata)
            metrics = [prediction_metrics(heldout[name], k, n, paired) for name in ("neutral", "position")]
            print(f"  {eligible.sum()} curves; delta={metadata['fits']['position']['position_offset']:+.3f}; "
                  f"held-out per-order RMSE {metrics[0]['order_probability_rmse']:.4f} -> "
                  f"{metrics[1]['order_probability_rmse']:.4f}", flush=True)
        for filename, rows in (("coverage.csv", coverage), ("prediction_comparison.csv", comparison),
                               ("repeat_checks.csv", repeats), ("monotonicity_sensitivity.csv", simulations)):
            write_csv(args.output / filename, rows)
        run["status"] = "incomplete_fits" if incomplete else "complete"
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
