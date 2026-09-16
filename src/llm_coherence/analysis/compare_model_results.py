#!/usr/bin/env python3
"""Paired, ladder-clustered comparisons between completed model summaries."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from llm_coherence.paths import (
    LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR,
    REPO_ROOT,
    model_within_ladder_dir,
    phase6b_coherence_json_path,
)


COHERENCE_METRICS = (
    "monotonicity_rate",
    "erratic_flip_rate",
    "mean_kendall_tau",
    "mean_spearman_rho",
    "jt_significant_rate",
    "mean_isotonic_r2",
    "mean_bootstrap_mono_prob",
)
PRIMARY_METRICS = {
    "monotonicity_rate",
    "mean_isotonic_r2",
    "within_ladder_accuracy",
}
UNIT_INTERVAL_METRICS = {
    "monotonicity_rate",
    "erratic_flip_rate",
    "jt_significant_rate",
    "mean_isotonic_r2",
    "mean_bootstrap_mono_prob",
}
CORRELATION_METRICS = {"mean_kendall_tau", "mean_spearman_rho"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _metric_value(row: dict[str, Any], metric: str, label: str) -> float:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has invalid {metric}") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} has non-finite {metric}")
    if metric in UNIT_INTERVAL_METRICS and not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} has out-of-range {metric}")
    if metric in CORRELATION_METRICS and not -1.0 <= value <= 1.0:
        raise ValueError(f"{label} has out-of-range {metric}")
    return value


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _sign_flip_p_value(
    differences: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, str]:
    """Two-sided paired randomization test of a zero mean difference."""
    observed = abs(statistics.fmean(differences))
    tolerance = 1e-15
    if len(differences) <= 20:
        total = 0
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
            permuted = abs(
                sum(sign * difference for sign, difference in zip(signs, differences))
                / len(differences)
            )
            total += 1
            extreme += permuted + tolerance >= observed
        return extreme / total, "exact_sign_flip"

    extreme = 0
    for _ in range(samples):
        permuted = abs(
            sum(
                difference if rng.getrandbits(1) else -difference
                for difference in differences
            )
            / len(differences)
        )
        extreme += permuted + tolerance >= observed
    return (extreme + 1) / (samples + 1), "monte_carlo_sign_flip"


def paired_ladder_analysis(
    left_values: list[float],
    right_values: list[float],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare paired ladder values using a cluster bootstrap and sign flips."""
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("Paired analysis requires equal, non-empty value lists")
    if bootstrap_samples < 1 or randomization_samples < 1:
        raise ValueError("Resampling counts must be positive")

    left = [float(value) for value in left_values]
    right = [float(value) for value in right_values]
    if not all(math.isfinite(value) for value in left + right):
        raise ValueError("Paired analysis requires finite metric values")
    differences = [
        right_value - left_value for left_value, right_value in zip(left, right)
    ]
    n = len(differences)
    bootstrap_rng = random.Random(seed)
    randomization_rng = random.Random(seed + 1_000_003)
    boot_left: list[float] = []
    boot_right: list[float] = []
    boot_difference: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [bootstrap_rng.randrange(n) for _ in range(n)]
        left_mean = sum(left[index] for index in indices) / n
        right_mean = sum(right[index] for index in indices) / n
        boot_left.append(left_mean)
        boot_right.append(right_mean)
        boot_difference.append(right_mean - left_mean)

    difference_mean = statistics.fmean(differences)
    difference_sd = statistics.stdev(differences) if n > 1 else 0.0
    p_value, test_method = _sign_flip_p_value(
        differences,
        samples=randomization_samples,
        rng=randomization_rng,
    )
    return {
        "n_paired_ladders": n,
        "left_macro_mean": statistics.fmean(left),
        "left_cluster_bootstrap_ci95": [
            _percentile(boot_left, 0.025),
            _percentile(boot_left, 0.975),
        ],
        "right_macro_mean": statistics.fmean(right),
        "right_cluster_bootstrap_ci95": [
            _percentile(boot_right, 0.025),
            _percentile(boot_right, 0.975),
        ],
        "right_minus_left": difference_mean,
        "difference_cluster_bootstrap_ci95": [
            _percentile(boot_difference, 0.025),
            _percentile(boot_difference, 0.975),
        ],
        "paired_cohens_dz": difference_mean / difference_sd if difference_sd else None,
        "paired_randomization_p_value": p_value,
        "paired_randomization_method": test_method,
        "bootstrap_samples": bootstrap_samples,
        "randomization_samples": (
            2**n if test_method == "exact_sign_flip" else randomization_samples
        ),
        "resampling_unit": "ladder",
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm-Bonferroni adjusted p-values keyed like the input."""
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in p_values.values()
    ):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def _paired_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    id_field: str,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    left = {str(row[id_field]): row for row in left_rows}
    right = {str(row[id_field]): row for row in right_rows}
    if len(left) != len(left_rows) or len(right) != len(right_rows):
        raise ValueError(f"Duplicate {id_field} values prevent paired analysis")
    if set(left) != set(right):
        missing_left = sorted(set(right).difference(left))
        missing_right = sorted(set(left).difference(right))
        raise ValueError(
            f"Paired {id_field} coverage differs: missing_left={missing_left[:5]}, "
            f"missing_right={missing_right[:5]}"
        )
    return sorted(left), left, right


def _analysis_for_ids(
    ids: list[str],
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    return paired_ladder_analysis(
        [float(left[item][metric]) for item in ids],
        [float(right[item][metric]) for item in ids],
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        seed=seed,
    )


def _validate_coherence_summary(summary: dict[str, Any], label: str) -> None:
    rows = summary.get("per_variation_set")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} coherence summary has no variation-set rows")
    overall = summary.get("aggregate", {}).get("overall", {})
    missing_fields = sorted(
        {"n_variation_sets", "n_total_comparisons", "n_tiers"}.difference(overall)
    )
    if missing_fields:
        raise ValueError(
            f"{label} coherence summary is missing aggregate fields {missing_fields}"
        )
    reported_ladders = _positive_int(
        overall["n_variation_sets"], f"{label} aggregate n_variation_sets"
    )
    if reported_ladders != len(rows):
        raise ValueError(
            f"{label} coherence summary ladder count does not reconcile: "
            f"reported={reported_ladders}, rows={len(rows)}"
        )
    row_comparisons: list[int] = []
    for row in rows:
        for metric in COHERENCE_METRICS:
            _metric_value(row, metric, f"{label} coherence row")
        value = _positive_int(row.get("n_comparisons"), f"{label} row n_comparisons")
        row_comparisons.append(value)
        for count_field, rate_field in (
            ("n_monotonic", "monotonicity_rate"),
            ("n_erratic_flips", "erratic_flip_rate"),
        ):
            count = row.get(count_field)
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError(f"{label} coherence summary has invalid {count_field}")
            if not 0 <= count <= value:
                raise ValueError(
                    f"{label} coherence summary has out-of-range {count_field}"
                )
            if not math.isclose(
                float(row[rate_field]), count / value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{label} coherence summary has inconsistent {rate_field}"
                )
    reported_comparisons = _positive_int(
        overall["n_total_comparisons"], f"{label} aggregate n_total_comparisons"
    )
    if sum(row_comparisons) != reported_comparisons:
        raise ValueError(
            f"{label} coherence summary comparison count does not reconcile"
        )
    total_comparisons = reported_comparisons
    for metric in COHERENCE_METRICS:
        _metric_value(overall, metric, f"{label} coherence aggregate")
    for count_field, rate_field in (
        ("n_monotonic", "monotonicity_rate"),
        ("n_erratic_flips", "erratic_flip_rate"),
    ):
        if rate_field in overall:
            expected_rate = (
                sum(int(row[count_field]) for row in rows) / total_comparisons
            )
            reported_rate = float(overall[rate_field])
            if not math.isfinite(reported_rate) or not math.isclose(
                reported_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{label} coherence aggregate has inconsistent {rate_field}"
                )
    top_tiers = summary.get("n_tiers")
    aggregate_tiers = _positive_int(overall["n_tiers"], f"{label} aggregate n_tiers")
    if (
        top_tiers is not None
        and _positive_int(top_tiers, f"{label} n_tiers") != aggregate_tiers
    ):
        raise ValueError(f"{label} coherence summary has conflicting tier counts")


def compare_coherence_summaries(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    _validate_coherence_summary(left_summary, "left")
    _validate_coherence_summary(right_summary, "right")
    if (
        left_summary["aggregate"]["overall"]["n_tiers"]
        != right_summary["aggregate"]["overall"]["n_tiers"]
    ):
        raise ValueError("Coherence tier counts differ between models")
    ids, left, right = _paired_rows(
        left_summary["per_variation_set"],
        right_summary["per_variation_set"],
        id_field="variation_id",
    )
    for item in ids:
        left_n = left[item].get("n_comparisons")
        right_n = right[item].get("n_comparisons")
        if left_n is not None or right_n is not None:
            if left_n != right_n:
                raise ValueError(
                    f"Coherence comparison denominator differs for {item}: "
                    f"left={left_n}, right={right_n}"
                )
    metrics: dict[str, dict[str, Any]] = {}
    for offset, metric in enumerate(COHERENCE_METRICS):
        analysis = _analysis_for_ids(
            ids,
            left,
            right,
            metric,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            seed=seed + offset,
        )
        # Only publish headline values that were reconstructed from counts.
        # Other aggregates can use Fisher transforms or finite-only denominators
        # not retained in these summaries; their row-level macro means are the
        # estimates used by this comparison.
        if metric in {"monotonicity_rate", "erratic_flip_rate"}:
            analysis["left_headline_aggregate"] = left_summary["aggregate"]["overall"][
                metric
            ]
            analysis["right_headline_aggregate"] = right_summary["aggregate"][
                "overall"
            ][metric]
        else:
            analysis["headline_aggregate_note"] = (
                "Source headline omitted: its aggregation cannot be verified from "
                "the retained counts. Inference uses the reported ladder macro means."
            )
        analysis["primary_endpoint"] = metric in PRIMARY_METRICS
        metrics[metric] = analysis

    adjusted = holm_adjust(
        {
            metric: analysis["paired_randomization_p_value"]
            for metric, analysis in metrics.items()
        }
    )
    for metric, adjusted_p in adjusted.items():
        metrics[metric]["holm_adjusted_p_value_across_coherence_metrics"] = adjusted_p

    categories: dict[str, Any] = {}
    category_names = sorted(
        {str(left[item].get("category", "unknown")) for item in ids}
    )
    for category_index, category in enumerate(category_names):
        category_ids = [
            item
            for item in ids
            if str(left[item].get("category", "unknown")) == category
        ]
        if any(
            str(right[item].get("category", "unknown")) != category
            for item in category_ids
        ):
            raise ValueError(f"Category mismatch between models for {category}")
        category_result: dict[str, Any] = {
            "n_paired_ladders": len(category_ids),
            "left_monotonicity_rate": statistics.fmean(
                float(left[item]["monotonicity_rate"]) for item in category_ids
            ),
            "right_monotonicity_rate": statistics.fmean(
                float(right[item]["monotonicity_rate"]) for item in category_ids
            ),
        }
        category_result["right_minus_left"] = (
            category_result["right_monotonicity_rate"]
            - category_result["left_monotonicity_rate"]
        )
        if len(category_ids) >= 5:
            category_result["exploratory_paired_analysis"] = _analysis_for_ids(
                category_ids,
                left,
                right,
                "monotonicity_rate",
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
                seed=seed + 100 + category_index,
            )
        else:
            category_result["exploratory_paired_analysis"] = None
            category_result["inference_note"] = (
                "Fewer than five paired ladders; report descriptively only."
            )
        categories[category] = category_result

    return {
        "n_paired_ladders": len(ids),
        "metrics": metrics,
        "exploratory_monotonicity_by_category": categories,
    }


def compare_within_ladder_summaries(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    for label, summary in (("left", left_summary), ("right", right_summary)):
        rows = summary.get("per_ladder")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{label} within-ladder summary has no ladder rows")
        for field in ("n_ladders", "n_ladders_expected"):
            if field not in summary:
                raise ValueError(f"{label} within-ladder summary is missing {field}")
            value = _positive_int(summary[field], f"{label} {field}")
            if value != len(rows):
                raise ValueError(
                    f"{label} within-ladder summary {field} does not reconcile: "
                    f"reported={value}, rows={len(rows)}"
                )
        row_counts: list[int] = []
        for row in rows:
            value = row.get("n")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{label} within-ladder summary has invalid n={value!r}"
                )
            row_counts.append(value)
            accuracy = float(row.get("accuracy", float("nan")))
            if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    f"{label} within-ladder summary has invalid accuracy={accuracy!r}"
                )
        if "n_total_pairs" not in summary:
            raise ValueError(f"{label} within-ladder summary is missing n_total_pairs")
        observed = _positive_int(summary["n_total_pairs"], f"{label} n_total_pairs")
        expected = summary.get(
            "n_total_pairs_expected", summary.get("n_requests_expected")
        )
        if expected is None:
            raise ValueError(
                f"{label} within-ladder summary is missing expected pair coverage"
            )
        expected = _positive_int(expected, f"{label} expected pair coverage")
        if observed != expected:
            raise ValueError(
                f"{label} within-ladder summary is incomplete: "
                f"observed={observed}, expected={expected}"
            )
        if "parse_errors" not in summary:
            raise ValueError(f"{label} within-ladder summary is missing parse_errors")
        parse_errors = summary["parse_errors"]
        if (
            isinstance(parse_errors, bool)
            or not isinstance(parse_errors, int)
            or parse_errors < 0
        ):
            raise ValueError(f"{label} within-ladder summary has invalid parse_errors")
        if parse_errors:
            raise ValueError(
                f"{label} within-ladder summary has {parse_errors} parse errors"
            )
        if sum(row_counts) != observed:
            raise ValueError(
                f"{label} within-ladder summary pair count does not reconcile"
            )
        if "overall_accuracy" not in summary:
            raise ValueError(
                f"{label} within-ladder summary is missing overall_accuracy"
            )
        total = sum(row_counts)
        weighted_accuracy = (
            sum(float(row["accuracy"]) * int(row["n"]) for row in rows) / total
        )
        reported_accuracy_value = float(summary["overall_accuracy"])
        if not math.isfinite(reported_accuracy_value) or not math.isclose(
            reported_accuracy_value,
            weighted_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{label} within-ladder summary overall_accuracy does not reconcile"
            )

    ids, left, right = _paired_rows(
        left_summary["per_ladder"],
        right_summary["per_ladder"],
        id_field="ladder_id",
    )
    for item in ids:
        if left[item].get("n") != right[item].get("n"):
            raise ValueError(
                f"Within-ladder denominator differs between models for {item}: "
                f"left={left[item].get('n')}, right={right[item].get('n')}"
            )
    overall = _analysis_for_ids(
        ids,
        left,
        right,
        "accuracy",
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        seed=seed,
    )
    overall["left_reported_overall_accuracy"] = left_summary.get("overall_accuracy")
    overall["right_reported_overall_accuracy"] = right_summary.get("overall_accuracy")
    overall["primary_endpoint"] = True

    by_valence: dict[str, Any] = {}
    for offset, valence in enumerate(
        sorted({str(left[item].get("valence", "unknown")) for item in ids})
    ):
        valence_ids = [
            item for item in ids if str(left[item].get("valence", "unknown")) == valence
        ]
        if any(
            str(right[item].get("valence", "unknown")) != valence
            for item in valence_ids
        ):
            raise ValueError(f"Valence mismatch between models for {valence}")
        by_valence[valence] = _analysis_for_ids(
            valence_ids,
            left,
            right,
            "accuracy",
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            seed=seed + offset + 1,
        )

    return {
        "overall": overall,
        "by_valence": by_valence,
        "by_valence_inference_note": (
            "Exploratory subgroup analysis; p-values are not multiplicity-adjusted."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two models with paired, ladder-clustered inference."
    )
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument(
        "--results-dir",
        default=str(LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--left-coherence")
    parser.add_argument("--right-coherence")
    parser.add_argument("--left-within")
    parser.add_argument("--right-within")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (REPO_ROOT / candidate).resolve()
    )


def _source_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validate_summary_model(
    summary: dict[str, Any], expected_model: str, path: Path
) -> None:
    identities = {
        str(summary[field])
        for field in ("model", "model_key")
        if summary.get(field) is not None
    }
    if not identities:
        raise ValueError(f"Summary has no model identity in {path}")
    if identities != {expected_model}:
        raise ValueError(
            f"Summary model mismatch in {path}: recorded={sorted(identities)!r}, "
            f"expected={expected_model!r}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = _resolve(args.results_dir)
    left_coherence_path = (
        _resolve(args.left_coherence)
        if args.left_coherence
        else phase6b_coherence_json_path(args.left_model, results_dir)
    )
    right_coherence_path = (
        _resolve(args.right_coherence)
        if args.right_coherence
        else phase6b_coherence_json_path(args.right_model, results_dir)
    )
    left_within_path = (
        _resolve(args.left_within)
        if args.left_within
        else model_within_ladder_dir(args.left_model, results_dir) / "summary.json"
    )
    right_within_path = (
        _resolve(args.right_within)
        if args.right_within
        else model_within_ladder_dir(args.right_model, results_dir) / "summary.json"
    )

    left_coherence = _load_json(left_coherence_path)
    right_coherence = _load_json(right_coherence_path)
    left_within = _load_json(left_within_path)
    right_within = _load_json(right_within_path)
    _validate_summary_model(left_coherence, args.left_model, left_coherence_path)
    _validate_summary_model(right_coherence, args.right_model, right_coherence_path)
    _validate_summary_model(left_within, args.left_model, left_within_path)
    _validate_summary_model(right_within, args.right_model, right_within_path)

    coherence = compare_coherence_summaries(
        left_coherence,
        right_coherence,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
    )
    within_ladder = compare_within_ladder_summaries(
        left_within,
        right_within,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed + 1_000,
    )
    primary_adjusted = holm_adjust(
        {
            "monotonicity_rate": coherence["metrics"]["monotonicity_rate"][
                "paired_randomization_p_value"
            ],
            "mean_isotonic_r2": coherence["metrics"]["mean_isotonic_r2"][
                "paired_randomization_p_value"
            ],
            "within_ladder_accuracy": within_ladder["overall"][
                "paired_randomization_p_value"
            ],
        }
    )
    coherence["metrics"]["monotonicity_rate"][
        "holm_adjusted_p_value_across_primary_endpoints"
    ] = primary_adjusted["monotonicity_rate"]
    coherence["metrics"]["mean_isotonic_r2"][
        "holm_adjusted_p_value_across_primary_endpoints"
    ] = primary_adjusted["mean_isotonic_r2"]
    within_ladder["overall"]["holm_adjusted_p_value_across_primary_endpoints"] = (
        primary_adjusted["within_ladder_accuracy"]
    )

    report = {
        "schema_version": "1.0",
        "left_model": args.left_model,
        "right_model": args.right_model,
        "difference_direction": "right_minus_left",
        "sources": {
            "left_coherence": _source_record(left_coherence_path),
            "right_coherence": _source_record(right_coherence_path),
            "left_within_ladder": _source_record(left_within_path),
            "right_within_ladder": _source_record(right_within_path),
        },
        "inference": {
            "paired_unit": "ladder",
            "bootstrap_samples": args.bootstrap_samples,
            "randomization_samples": args.randomization_samples,
            "seed": args.seed,
            "coherence_multiplicity": "Holm adjustment across reported overall coherence metrics",
            "primary_endpoint_multiplicity": (
                "Holm adjustment across monotonicity rate, mean isotonic R-squared, "
                "and within-ladder accuracy"
            ),
            "category_analyses": "exploratory; no multiplicity-adjusted claims",
        },
        "coherence": coherence,
        "within_ladder": within_ladder,
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Paired model comparison: {output_path}")
    print(
        "Monotonicity right-minus-left: "
        f"{report['coherence']['metrics']['monotonicity_rate']['right_minus_left']:.4f}"
    )
    print(
        "Within-ladder accuracy right-minus-left: "
        f"{report['within_ladder']['overall']['right_minus_left']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
