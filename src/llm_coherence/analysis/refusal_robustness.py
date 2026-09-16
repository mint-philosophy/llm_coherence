#!/usr/bin/env python3
"""Audit missing forced-choice responses and linked visible rationales.

The behavioral analysis is always available from ``results.json``.  Trial-level
rationale analysis is deliberately gated on stable ``custom_id`` values in both
the missing-response records and the reasoning-trace sidecars.  Legacy traces
that contain only local message indices cannot be joined safely after retries or
resume operations and are therefore inventory-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from llm_coherence.config import resolve_model_results_dir
from llm_coherence.paths import (
    LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR,
    REPO_ROOT,
)


N_TIERS = 7

REASONING_CONCLUSIONS = {
    "favors_a",
    "favors_b",
    "equal",
    "incomparable",
    "no_conclusion",
    "unclear",
}
REFUSAL_REASONS = {
    "none",
    "no_personal_preference",
    "political_or_religious_neutrality",
    "ethical_sensitivity",
    "false_dichotomy_or_incomparability",
    "insufficient_information",
    "instruction_or_format_failure",
    "other",
}
FINAL_RESPONSES = {
    "A",
    "B",
    "explicit_refusal",
    "both_without_choice",
    "malformed_or_incomplete",
}
RELATIONSHIPS = {
    "reasoning_and_answer_agree",
    "reasoning_favors_a_or_b_but_final_refuses",
    "reasoning_and_final_both_abstain",
    "reasoning_and_final_disagree",
    "unclear",
}
ANNOTATION_FIELDS = {
    "reasoning_conclusion": REASONING_CONCLUSIONS,
    "refusal_reason": REFUSAL_REASONS,
    "final_response": FINAL_RESPONSES,
    "relationship": RELATIONSHIPS,
}


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (REPO_ROOT / candidate).resolve()
    )


def _normalize_results_dir(results_dir: Path, model: str) -> Path:
    """Resolve a runs root or model root to the ladder-comparison directory."""
    run_dir = "ladder_vs_comparison_statements"
    resolved = results_dir.resolve()
    if resolved.name == run_dir:
        return resolved
    model_root = resolve_model_results_dir(model, resolved)
    if resolved == model_root:
        return (resolved / run_dir).resolve()
    return (model_root / run_dir).resolve()


def _iter_result_paths(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob("phase6b_ladder_*/results.json")
        if path.is_file()
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def expected_trials_for_preference(
    result: dict[str, Any], preference: dict[str, Any]
) -> int:
    """Return the fixed requested denominator for one comparison cell."""
    explicit = preference.get("expected_trials")
    if isinstance(explicit, int) and explicit > 0:
        return explicit

    config = result.get("config") or {}
    if config.get("is_base_model"):
        raise ValueError(
            "Refusal robustness applies to sampled discrete responses, not "
            "base-model log-probability pseudo-counts"
        )
    num_trials = config.get("num_trials")
    if not isinstance(num_trials, int) or num_trials <= 0:
        raise ValueError(
            "Cannot infer expected trials: missing positive config.num_trials"
        )
    return num_trials * (2 if config.get("include_flipped", True) else 1)


def _winner(probability: float) -> str:
    if probability > 0.5:
        return "A"
    if probability < 0.5:
        return "B"
    return "tie"


def _robust_winner(lower_a: float, upper_a: float) -> str:
    if lower_a > 0.5:
        return "A"
    if upper_a < 0.5:
        return "B"
    return "uncertain"


def preference_sensitivity(
    result: dict[str, Any],
    preference: dict[str, Any],
) -> dict[str, Any]:
    """Compute complete-case and extreme-allocation estimates for one cell."""
    count_a = int(preference.get("count_prefer_a", 0))
    count_b = int(preference.get("count_prefer_b", 0))
    parsed = count_a + count_b
    expected = expected_trials_for_preference(result, preference)
    if parsed > expected:
        raise ValueError(f"Parsed count {parsed} exceeds expected count {expected}")
    missing = expected - parsed
    conditional = count_a / parsed if parsed else None
    lower_a = count_a / expected
    upper_a = (count_a + missing) / expected

    return {
        "count_prefer_a": count_a,
        "count_prefer_b": count_b,
        "expected_trials": expected,
        "parseable_trials": parsed,
        "missing_trials": missing,
        "conditional_prob_prefer_a": conditional,
        "prob_prefer_a_all_missing_to_b": lower_a,
        "prob_prefer_a_all_missing_to_a": upper_a,
        "conditional_winner": _winner(conditional)
        if conditional is not None
        else "unobserved",
        "winner_robust_to_missing": _robust_winner(lower_a, upper_a),
    }


def _validate_probability_fields(
    preference: dict[str, Any],
    sensitivity: dict[str, Any],
) -> None:
    """Check that stored conditional probabilities match the vote counts."""
    conditional = sensitivity["conditional_prob_prefer_a"]
    stored_a = preference.get("prob_prefer_a")
    stored_b = preference.get("prob_prefer_b")
    if conditional is None:
        if stored_a is not None or stored_b is not None:
            raise ValueError("Zero-parseable cell has non-null probability fields")
        return
    if stored_a is None or stored_b is None:
        raise ValueError("Parseable cell has null probability fields")
    # Result artifacts round probabilities to four decimal places.
    if abs(float(stored_a) - conditional) > 0.000051:
        raise ValueError(
            f"prob_prefer_a={stored_a} does not match counts ({conditional:.8f})"
        )
    if abs(float(stored_b) - (1.0 - conditional)) > 0.000051:
        raise ValueError(
            f"prob_prefer_b={stored_b} does not match counts ({1.0 - conditional:.8f})"
        )


def _cell_identity(result_path: Path, preference: dict[str, Any]) -> dict[str, Any]:
    outcome_a = preference.get("outcome_a") or {}
    outcome_b = preference.get("outcome_b") or {}
    return {
        "artifact": result_path.parent.name,
        "variation_id": outcome_a.get("variation_id"),
        "variation_category": outcome_a.get("category"),
        "tier": outcome_a.get("tier"),
        "tier_label": outcome_a.get("tier_label"),
        "outcome_a_text": outcome_a.get("text"),
        "comparison_id": outcome_b.get("comparison_id"),
        "comparison_category": outcome_b.get("comparison_category"),
        "outcome_b_text": outcome_b.get("text"),
    }


def _group_monotonicity(
    preferences: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, int]:
    counts = {"conditional": 0, "all_missing_to_a": 0, "all_missing_to_b": 0}
    groups = 0
    for start in range(0, len(preferences), N_TIERS):
        block = preferences[start : start + N_TIERS]
        if len(block) != N_TIERS:
            raise ValueError(
                f"Preference count {len(preferences)} is not divisible by {N_TIERS}"
            )
        comparison_texts = {(item.get("outcome_b") or {}).get("text") for item in block}
        if len(comparison_texts) != 1:
            raise ValueError(
                f"Misaligned seven-tier block starting at preference {start}"
            )
        cells = [preference_sensitivity(result, item) for item in block]
        scenario_probs = {
            "conditional": [
                cell["conditional_prob_prefer_a"]
                if cell["conditional_prob_prefer_a"] is not None
                else 0.5
                for cell in cells
            ],
            "all_missing_to_a": [
                cell["prob_prefer_a_all_missing_to_a"] for cell in cells
            ],
            "all_missing_to_b": [
                cell["prob_prefer_a_all_missing_to_b"] for cell in cells
            ],
        }
        for name, probs in scenario_probs.items():
            if all(left <= right for left, right in zip(probs, probs[1:])):
                counts[name] += 1
        groups += 1
    counts["groups"] = groups
    return counts


def analyze_results(results_dir: Path, model: str) -> tuple[dict[str, Any], set[str]]:
    """Analyze every result file and return the report plus scoped missing IDs."""
    result_paths = _iter_result_paths(results_dir)
    if not result_paths:
        raise FileNotFoundError(f"No phase6b ladder results found under {results_dir}")

    totals = Counter()
    missing_by_variation_category: Counter[str] = Counter()
    missing_by_comparison_category: Counter[str] = Counter()
    missing_by_tier: Counter[str] = Counter()
    missing_by_reason: Counter[str] = Counter()
    affected_cells: list[dict[str, Any]] = []
    missing_trial_keys: set[str] = set()
    missing_records_without_custom_id = 0
    scenario_monotonic = Counter()
    integrity_checks = Counter()

    for result_path in result_paths:
        result = _load_json(result_path)
        preferences = result.get("preferences")
        if not isinstance(preferences, list):
            raise ValueError(f"Missing preferences list in {result_path}")

        monotonic = _group_monotonicity(preferences, result)
        scenario_monotonic.update(monotonic)
        totals["result_files"] += 1
        totals["comparison_cells"] += len(preferences)

        file_missing = 0
        file_expected = 0
        cell_keys: set[tuple[Any, Any, Any]] = set()
        for preference in preferences:
            sensitivity = preference_sensitivity(result, preference)
            _validate_probability_fields(preference, sensitivity)
            integrity_checks["probability_fields_checked"] += 1
            totals["expected_trials"] += sensitivity["expected_trials"]
            totals["parseable_trials"] += sensitivity["parseable_trials"]
            totals["missing_trials"] += sensitivity["missing_trials"]
            file_expected += sensitivity["expected_trials"]

            outcome_a = preference.get("outcome_a") or {}
            outcome_b = preference.get("outcome_b") or {}
            cell_key = (
                outcome_a.get("variation_id"),
                outcome_a.get("tier"),
                outcome_b.get("comparison_id", outcome_b.get("text")),
            )
            if cell_key in cell_keys:
                raise ValueError(
                    f"Duplicate comparison cell {cell_key!r} in {result_path}"
                )
            cell_keys.add(cell_key)
            if sensitivity["missing_trials"] == 0:
                continue

            file_missing += sensitivity["missing_trials"]
            identity = _cell_identity(result_path, preference)
            affected_cells.append({**identity, **sensitivity})
            missing_by_variation_category[str(identity["variation_category"])] += (
                sensitivity["missing_trials"]
            )
            missing_by_comparison_category[str(identity["comparison_category"])] += (
                sensitivity["missing_trials"]
            )
            missing_by_tier[str(identity["tier"])] += sensitivity["missing_trials"]

            records = preference.get("missing_responses") or []
            missing_by_reason.update(preference.get("missing_by_reason") or {})
            if records:
                for record in records:
                    custom_id = record.get("custom_id")
                    if custom_id:
                        missing_trial_keys.add(f"{result_path.parent.name}:{custom_id}")
                    else:
                        missing_records_without_custom_id += 1
            else:
                missing_records_without_custom_id += sensitivity["missing_trials"]

        metadata = result.get("metadata") or {}
        metadata_missing = metadata.get("unparseable_count")
        if metadata_missing is not None and int(metadata_missing) != file_missing:
            raise ValueError(
                f"Missing-count mismatch in {result_path}: metadata={metadata_missing}, "
                f"reconstructed={file_missing}"
            )
        metadata_comparisons = metadata.get("total_comparisons")
        if metadata_comparisons is not None and int(metadata_comparisons) != len(
            preferences
        ):
            raise ValueError(
                f"Comparison-count mismatch in {result_path}: "
                f"metadata={metadata_comparisons}, reconstructed={len(preferences)}"
            )
        metadata_calls = metadata.get("total_api_calls")
        if metadata_calls is not None and int(metadata_calls) != file_expected:
            raise ValueError(
                f"API-call mismatch in {result_path}: metadata={metadata_calls}, "
                f"reconstructed={file_expected}"
            )
        integrity_checks["files_checked"] += 1

    expected = totals["expected_trials"]
    groups = scenario_monotonic["groups"]
    report = {
        "schema_version": "1.0",
        "model": model,
        "results_dir": str(results_dir),
        "coverage": {
            **dict(totals),
            "missing_rate": totals["missing_trials"] / expected if expected else 0.0,
            "affected_cells": len(affected_cells),
            "missing_records_without_custom_id": missing_records_without_custom_id,
            "counts_reconcile": (
                totals["parseable_trials"] + totals["missing_trials"] == expected
            ),
        },
        "artifact_integrity": {
            **dict(integrity_checks),
            "duplicate_comparison_cells": 0,
            "metadata_count_mismatches": 0,
            "probability_mismatches": 0,
            "passed": True,
        },
        "missingness_distribution": {
            "by_variation_category": dict(
                sorted(missing_by_variation_category.items())
            ),
            "by_comparison_category": dict(
                sorted(missing_by_comparison_category.items())
            ),
            "by_tier": dict(sorted(missing_by_tier.items())),
            "by_recorded_reason": dict(sorted(missing_by_reason.items())),
        },
        "monotonicity_sensitivity": {
            "comparison_groups": groups,
            "conditional_on_parseable": {
                "monotonic_groups": scenario_monotonic["conditional"],
                "rate": scenario_monotonic["conditional"] / groups if groups else 0.0,
            },
            "all_missing_prefer_a": {
                "monotonic_groups": scenario_monotonic["all_missing_to_a"],
                "rate": scenario_monotonic["all_missing_to_a"] / groups
                if groups
                else 0.0,
            },
            "all_missing_prefer_b": {
                "monotonic_groups": scenario_monotonic["all_missing_to_b"],
                "rate": scenario_monotonic["all_missing_to_b"] / groups
                if groups
                else 0.0,
            },
            "note": (
                "The two extreme-allocation scenarios are sensitivity checks, not "
                "mathematical upper and lower bounds on monotonicity."
            ),
        },
        "affected_cells": affected_cells,
    }
    return report, missing_trial_keys


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, value if isinstance(value, dict) else None


def audit_reasoning_traces(
    results_dir: Path,
    missing_trial_keys: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Inventory trace sidecars and verify whether missing trials are joinable."""
    trace_paths = sorted(results_dir.glob("phase6b_ladder_*/reasoning_traces.jsonl"))
    rows = 0
    malformed_rows = 0
    rows_with_custom_id = 0
    trial_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for trace_path in trace_paths:
        artifact = trace_path.parent.name
        for _line_number, row in _iter_jsonl(trace_path):
            rows += 1
            if row is None:
                malformed_rows += 1
                continue
            custom_id = row.get("custom_id")
            if custom_id:
                rows_with_custom_id += 1
                trial_rows[f"{artifact}:{custom_id}"].append(row)

    selected: dict[str, dict[str, Any]] = {}
    for trial_key, candidates in trial_rows.items():
        selected[trial_key] = max(
            candidates,
            key=lambda row: int(row.get("attempt", 0) or 0),
        )

    linked_missing = missing_trial_keys.intersection(selected)
    missing_without_trace = missing_trial_keys.difference(selected)
    duplicate_trial_ids = sum(1 for values in trial_rows.values() if len(values) > 1)
    reasons: list[str] = []
    if not trace_paths:
        reasons.append("no reasoning_traces.jsonl files were found")
    if rows and rows_with_custom_id != rows - malformed_rows:
        reasons.append("one or more trace rows lack a stable custom_id")
    if malformed_rows:
        reasons.append("one or more trace rows are malformed")
    if not missing_trial_keys:
        reasons.append("missing responses do not expose stable custom_id values")
    elif missing_without_trace:
        reasons.append("one or more missing trials have no matching trace row")

    eligible = not reasons
    report = {
        "trace_files": len(trace_paths),
        "trace_rows": rows,
        "malformed_rows": malformed_rows,
        "rows_with_custom_id": rows_with_custom_id,
        "unique_trial_ids": len(trial_rows),
        "trial_ids_with_multiple_rows": duplicate_trial_ids,
        "missing_trial_ids": len(missing_trial_keys),
        "linked_missing_trial_ids": len(linked_missing),
        "unlinked_missing_trial_ids": len(missing_without_trace),
        "quantitative_trial_level_analysis_allowed": eligible,
        "blocking_reasons": reasons,
        "interpretation": (
            "Linked traces may be coded at the unique-trial level."
            if eligible
            else "Treat unlinked traces as qualitative examples only; do not report trace frequencies."
        ),
    }
    return report, {key: selected[key] for key in sorted(linked_missing)}


def write_annotation_template(
    path: Path,
    linked_traces: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trial_key, trace in linked_traces.items():
            row = {
                "trial_key": trial_key,
                "coder": "",
                "reasoning": trace.get("reasoning"),
                "content": trace.get("content"),
                "reasoning_conclusion": "",
                "refusal_reason": "",
                "final_response": "",
                "relationship": "",
                "notes": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts).union(right_counts)
    expected = sum(
        (left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def summarize_annotations(path: Path) -> dict[str, Any]:
    """Validate coded JSONL and summarize themes, transitions, and agreement."""
    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in _iter_jsonl(path):
        if row is None:
            raise ValueError(f"Malformed annotation JSON at {path}:{line_number}")
        trial_key = str(row.get("trial_key") or "").strip()
        coder = str(row.get("coder") or "").strip()
        if not trial_key or not coder:
            raise ValueError(
                f"Annotation requires trial_key and coder at {path}:{line_number}"
            )
        key = (trial_key, coder)
        if key in seen:
            raise ValueError(
                f"Duplicate annotation for trial={trial_key!r}, coder={coder!r}"
            )
        seen.add(key)
        for field, allowed in ANNOTATION_FIELDS.items():
            value = row.get(field)
            if value not in allowed:
                raise ValueError(
                    f"Invalid {field}={value!r} at {path}:{line_number}; "
                    f"expected one of {sorted(allowed)}"
                )
        annotations.append(row)

    field_counts = {
        field: dict(sorted(Counter(str(row[field]) for row in annotations).items()))
        for field in ANNOTATION_FIELDS
    }
    transitions = Counter(
        f"{row['reasoning_conclusion']}->{row['final_response']}" for row in annotations
    )
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_trial[str(row["trial_key"])].append(row)

    agreement: dict[str, Any] = {}
    double_coded_trials = [rows for rows in by_trial.values() if len(rows) == 2]
    for field in ANNOTATION_FIELDS:
        pairs = [
            (str(rows[0][field]), str(rows[1][field])) for rows in double_coded_trials
        ]
        agreement[field] = {
            "n_double_coded": len(pairs),
            "percent_agreement": (
                sum(left == right for left, right in pairs) / len(pairs)
                if pairs
                else None
            ),
            "cohen_kappa": _cohen_kappa(pairs),
        }

    return {
        "annotation_rows": len(annotations),
        "unique_trials": len(by_trial),
        "coders": sorted({str(row["coder"]) for row in annotations}),
        "counts": field_counts,
        "reasoning_to_final_transitions": dict(sorted(transitions.items())),
        "inter_rater_agreement": agreement,
    }


def write_affected_cells_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["artifact", "missing_trials"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit missing forced-choice results and linked visible rationales."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--results-dir",
        default=str(LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Runs root or model ladder_vs_comparison_statements directory.",
    )
    parser.add_argument("--output", help="JSON report path.")
    parser.add_argument("--affected-cells-csv", help="Optional affected-cell CSV path.")
    parser.add_argument(
        "--annotation-template",
        help="Optional JSONL template; written only when trial-level trace linkage is valid.",
    )
    parser.add_argument(
        "--annotations", help="Optional completed annotation JSONL to summarize."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = _resolve_repo_path(args.results_dir)
    results_dir = _normalize_results_dir(runs_root, args.model)
    report, missing_trial_keys = analyze_results(results_dir, args.model)
    trace_report, linked_traces = audit_reasoning_traces(
        results_dir, missing_trial_keys
    )
    report["reasoning_trace_audit"] = trace_report

    if args.annotations:
        report["annotation_analysis"] = summarize_annotations(
            _resolve_repo_path(args.annotations)
        )

    output_path = (
        _resolve_repo_path(args.output)
        if args.output
        else results_dir
        / "coherence_test"
        / f"phase6b_refusal_robustness_{args.model}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    if args.affected_cells_csv:
        write_affected_cells_csv(
            _resolve_repo_path(args.affected_cells_csv), report["affected_cells"]
        )

    if args.annotation_template:
        if not trace_report["quantitative_trial_level_analysis_allowed"]:
            reasons = "; ".join(trace_report["blocking_reasons"])
            raise ValueError(
                f"Refusing to write an unlinked annotation template: {reasons}"
            )
        write_annotation_template(
            _resolve_repo_path(args.annotation_template), linked_traces
        )

    coverage = report["coverage"]
    print(f"Refusal robustness report: {output_path}")
    print(
        f"Coverage: {coverage['parseable_trials']:,}/{coverage['expected_trials']:,} "
        f"parseable; {coverage['missing_trials']:,} missing across "
        f"{coverage['affected_cells']} cells"
    )
    print(
        "Trial-level rationale analysis: "
        + (
            "allowed"
            if trace_report["quantitative_trial_level_analysis_allowed"]
            else "blocked (unlinked trace provenance)"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
