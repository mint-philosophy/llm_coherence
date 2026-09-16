#!/usr/bin/env python3
"""Paired, ladder-clustered comparisons between completed model summaries."""

from __future__ import annotations

import argparse
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
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
    differences = [
        right_value - left_value for left_value, right_value in zip(left, right)
    ]
    n = len(differences)
    rng = random.Random(seed)
    boot_left: list[float] = []
    boot_right: list[float] = []
    boot_difference: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(n) for _ in range(n)]
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
        rng=rng,
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


def compare_coherence_summaries(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    ids, left, right = _paired_rows(
        left_summary["per_variation_set"],
        right_summary["per_variation_set"],
        id_field="variation_id",
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
        analysis["left_headline_aggregate"] = (
            left_summary.get("aggregate", {}).get("overall", {}).get(metric)
        )
        analysis["right_headline_aggregate"] = (
            right_summary.get("aggregate", {}).get("overall", {}).get(metric)
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
    ids, left, right = _paired_rows(
        left_summary["per_ladder"],
        right_summary["per_ladder"],
        id_field="ladder_id",
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

    return {"overall": overall, "by_valence": by_valence}


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

    coherence = compare_coherence_summaries(
        _load_json(left_coherence_path),
        _load_json(right_coherence_path),
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
    )
    within_ladder = compare_within_ladder_summaries(
        _load_json(left_within_path),
        _load_json(right_within_path),
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
