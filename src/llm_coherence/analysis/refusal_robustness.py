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
import math
from collections import Counter, defaultdict
from fractions import Fraction
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
    if (resolved / run_dir).is_dir():
        return (resolved / run_dir).resolve()
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
    config = result.get("config") or {}
    if config.get("is_base_model"):
        raise ValueError(
            "Refusal robustness applies to sampled discrete responses, not "
            "base-model log-probability pseudo-counts"
        )
    num_trials = config.get("num_trials")
    if (
        isinstance(num_trials, bool)
        or not isinstance(num_trials, int)
        or num_trials <= 0
    ):
        raise ValueError(
            "Cannot infer expected trials: missing positive config.num_trials"
        )
    configured = num_trials * (2 if config.get("include_flipped", True) else 1)
    explicit = preference.get("expected_trials")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit <= 0:
            raise ValueError("expected_trials must be a positive integer")
        if explicit != configured:
            raise ValueError(
                f"Stored expected_trials={explicit} does not match run config "
                f"denominator {configured}"
            )
    return configured


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
    count_a = preference.get("count_prefer_a")
    count_b = preference.get("count_prefer_b")
    if isinstance(count_a, bool) or not isinstance(count_a, int) or count_a < 0:
        raise ValueError("count_prefer_a must be a non-negative integer")
    if isinstance(count_b, bool) or not isinstance(count_b, int) or count_b < 0:
        raise ValueError("count_prefer_b must be a non-negative integer")
    parsed = count_a + count_b
    expected = expected_trials_for_preference(result, preference)
    if parsed > expected:
        raise ValueError(f"Parsed count {parsed} exceeds expected count {expected}")
    missing = expected - parsed
    conditional = count_a / parsed if parsed else None
    lower_a = count_a / expected
    upper_a = (count_a + missing) / expected

    reconstructed = {
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
    for field in ("parseable_trials", "missing_trials"):
        stored = preference.get(field)
        if stored is not None:
            if isinstance(stored, bool) or not isinstance(stored, int):
                raise ValueError(f"Stored {field} must be an integer")
            if stored != reconstructed[field]:
                raise ValueError(
                    f"Stored {field}={stored} does not match reconstructed "
                    f"value {reconstructed[field]}"
                )
    probability_denominator = preference.get("probability_denominator")
    if (
        probability_denominator is not None
        and probability_denominator != "parseable_trials"
    ):
        raise ValueError(
            "probability_denominator must be 'parseable_trials' for sampled responses"
        )
    stored_bounds = preference.get("prob_prefer_a_bounds")
    if stored_bounds is not None:
        if not isinstance(stored_bounds, dict):
            raise ValueError("prob_prefer_a_bounds must be an object")
        expected_bounds = {
            "lower_missing_prefer_b": lower_a,
            "upper_missing_prefer_a": upper_a,
        }
        for field, expected_value in expected_bounds.items():
            stored_value = float(stored_bounds.get(field, float("nan")))
            if (
                not math.isfinite(stored_value)
                or abs(stored_value - expected_value) > 0.000051
            ):
                raise ValueError(
                    f"Stored probability bound {field} does not match counts"
                )
    return reconstructed


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
    stored_a_value = float(stored_a)
    stored_b_value = float(stored_b)
    if not math.isfinite(stored_a_value) or not 0.0 <= stored_a_value <= 1.0:
        raise ValueError(f"prob_prefer_a={stored_a} is not a finite probability")
    if not math.isfinite(stored_b_value) or not 0.0 <= stored_b_value <= 1.0:
        raise ValueError(f"prob_prefer_b={stored_b} is not a finite probability")
    if abs(stored_a_value - conditional) > 0.000051:
        raise ValueError(
            f"prob_prefer_a={stored_a} does not match counts ({conditional:.8f})"
        )
    if abs(stored_b_value - (1.0 - conditional)) > 0.000051:
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
    counts = {
        "conditional": 0,
        "conditional_available": 0,
        "all_missing_to_a": 0,
        "all_missing_to_b": 0,
        "always_monotonic": 0,
        "can_be_monotonic": 0,
    }
    groups = 0
    for start in range(0, len(preferences), N_TIERS):
        block = preferences[start : start + N_TIERS]
        if len(block) != N_TIERS:
            raise ValueError(
                f"Preference count {len(preferences)} is not divisible by {N_TIERS}"
            )
        variation_ids = {
            (item.get("outcome_a") or {}).get("variation_id") for item in block
        }
        comparison_ids = {
            (
                (item.get("outcome_b") or {}).get("comparison_id"),
                (item.get("outcome_b") or {}).get("text"),
            )
            for item in block
        }
        tiers = [(item.get("outcome_a") or {}).get("tier") for item in block]
        if len(variation_ids) != 1 or len(comparison_ids) != 1:
            raise ValueError(
                f"Misaligned seven-tier block starting at preference {start}"
            )
        if tiers != list(range(1, N_TIERS + 1)):
            raise ValueError(
                f"Seven-tier block starting at preference {start} has invalid "
                f"tier sequence {tiers!r}"
            )
        cells = [preference_sensitivity(result, item) for item in block]
        conditional_probs = [cell["conditional_prob_prefer_a"] for cell in cells]
        scenario_probs = {
            "all_missing_to_a": [
                cell["prob_prefer_a_all_missing_to_a"] for cell in cells
            ],
            "all_missing_to_b": [
                cell["prob_prefer_a_all_missing_to_b"] for cell in cells
            ],
        }
        if all(probability is not None for probability in conditional_probs):
            counts["conditional_available"] += 1
            if all(
                left <= right
                for left, right in zip(conditional_probs, conditional_probs[1:])
            ):
                counts["conditional"] += 1
        for name, probs in scenario_probs.items():
            if all(left <= right for left, right in zip(probs, probs[1:])):
                counts[name] += 1

        intervals = [
            (
                Fraction(cell["count_prefer_a"], cell["expected_trials"]),
                Fraction(
                    cell["count_prefer_a"] + cell["missing_trials"],
                    cell["expected_trials"],
                ),
            )
            for cell in cells
        ]
        if all(
            left_upper <= right_lower
            for (_left_lower, left_upper), (right_lower, _right_upper) in zip(
                intervals, intervals[1:]
            )
        ):
            counts["always_monotonic"] += 1

        first = cells[0]
        chosen = Fraction(first["count_prefer_a"], first["expected_trials"])
        can_be_monotonic = True
        for cell in cells[1:]:
            denominator = cell["expected_trials"]
            required_numerator = (
                chosen.numerator * denominator + chosen.denominator - 1
            ) // chosen.denominator
            chosen_numerator = max(cell["count_prefer_a"], required_numerator)
            if chosen_numerator > cell["count_prefer_a"] + cell["missing_trials"]:
                can_be_monotonic = False
                break
            chosen = Fraction(chosen_numerator, denominator)
        if can_be_monotonic:
            counts["can_be_monotonic"] += 1
        groups += 1
    counts["groups"] = groups
    return counts


def analyze_results(
    results_dir: Path,
    model: str,
    *,
    expected_result_files: int | None = None,
    expected_comparison_groups: int | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Analyze every result file and return the report plus missing-trial context."""
    result_paths = _iter_result_paths(results_dir)
    if not result_paths:
        raise FileNotFoundError(f"No phase6b ladder results found under {results_dir}")
    if expected_result_files is not None and len(result_paths) != expected_result_files:
        raise ValueError(
            f"Expected {expected_result_files} result files for {model}, "
            f"found {len(result_paths)}"
        )

    totals = Counter()
    missing_by_variation_category: Counter[str] = Counter()
    missing_by_comparison_category: Counter[str] = Counter()
    missing_by_tier: Counter[str] = Counter()
    missing_by_reason: Counter[str] = Counter()
    affected_cells: list[dict[str, Any]] = []
    missing_trials_by_key: dict[str, dict[str, Any]] = {}
    missing_records_without_custom_id = 0
    scenario_monotonic = Counter()
    integrity_checks = Counter()
    variation_ids_seen: set[str] = set()

    for result_path in result_paths:
        result = _load_json(result_path)
        recorded_model = (result.get("config") or {}).get("model_key")
        if recorded_model != model:
            raise ValueError(
                f"Model mismatch in {result_path}: recorded={recorded_model!r}, "
                f"expected={model!r}"
            )
        preferences = result.get("preferences")
        if not isinstance(preferences, list):
            raise ValueError(f"Missing preferences list in {result_path}")
        file_variation_ids = {
            str((preference.get("outcome_a") or {}).get("variation_id"))
            for preference in preferences
        }
        if len(file_variation_ids) != 1 or "None" in file_variation_ids:
            raise ValueError(
                f"Result file must contain exactly one variation_id: {result_path}"
            )
        variation_id = next(iter(file_variation_ids))
        if variation_id in variation_ids_seen:
            raise ValueError(
                f"Duplicate variation_id {variation_id!r} across result files"
            )
        variation_ids_seen.add(variation_id)

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
            recorded_reasons = preference.get("missing_by_reason") or {}
            if not isinstance(recorded_reasons, dict):
                raise ValueError(
                    f"missing_by_reason must be an object in {result_path}"
                )
            for reason, value in recorded_reasons.items():
                if not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"Invalid missing_by_reason count {reason}={value!r} "
                        f"in {result_path}"
                    )
            if (
                recorded_reasons
                and sum(recorded_reasons.values()) != sensitivity["missing_trials"]
            ):
                raise ValueError(
                    f"Missing-reason counts do not match reconstructed missing trials "
                    f"in {result_path}: reasons={recorded_reasons}, "
                    f"missing={sensitivity['missing_trials']}"
                )
            missing_by_reason.update(recorded_reasons)
            if records:
                if len(records) != sensitivity["missing_trials"]:
                    raise ValueError(
                        f"Missing-response record count does not match reconstructed "
                        f"missing trials in {result_path}: records={len(records)}, "
                        f"missing={sensitivity['missing_trials']}"
                    )
                reasons_from_records = Counter(
                    str(record.get("reason")) for record in records
                )
                if recorded_reasons and reasons_from_records != Counter(
                    recorded_reasons
                ):
                    raise ValueError(
                        f"missing_by_reason does not match missing_responses in "
                        f"{result_path}"
                    )
                record_ids = [record.get("custom_id") for record in records]
                populated_ids = [custom_id for custom_id in record_ids if custom_id]
                if len(populated_ids) != len(set(populated_ids)):
                    raise ValueError(
                        f"Duplicate missing-response custom_id in {result_path}"
                    )
                for record in records:
                    custom_id = record.get("custom_id")
                    if custom_id:
                        trial_key = f"{result_path.parent.name}:{custom_id}"
                        if trial_key in missing_trials_by_key:
                            raise ValueError(
                                f"Duplicate missing-response trial key {trial_key!r}"
                            )
                        direction = str(record.get("direction") or "").upper()
                        if direction not in {"AB", "BA"}:
                            raise ValueError(
                                f"Missing response {trial_key!r} has invalid direction "
                                f"{record.get('direction')!r}"
                            )
                        prompt_option_a = (
                            identity["outcome_a_text"]
                            if direction == "AB"
                            else identity["outcome_b_text"]
                        )
                        prompt_option_b = (
                            identity["outcome_b_text"]
                            if direction == "AB"
                            else identity["outcome_a_text"]
                        )
                        missing_trials_by_key[trial_key] = {
                            **identity,
                            "direction": direction,
                            "trial_index": record.get("trial_index"),
                            "missing_reason": record.get("reason"),
                            "raw_response": record.get("raw_response"),
                            "prompt_option_a_text": prompt_option_a,
                            "prompt_option_b_text": prompt_option_b,
                        }
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
    if expected_comparison_groups is not None and groups != expected_comparison_groups:
        raise ValueError(
            f"Expected {expected_comparison_groups} comparison groups for {model}, "
            f"found {groups}"
        )
    conditional_groups = scenario_monotonic["conditional_available"]
    report = {
        "schema_version": "1.0",
        "model": model,
        "results_dir": str(results_dir),
        "coverage": {
            **dict(totals),
            "missing_rate": totals["missing_trials"] / expected if expected else 0.0,
            "affected_cells": len(affected_cells),
            "expected_result_files": expected_result_files,
            "expected_comparison_groups": expected_comparison_groups,
            "explicit_coverage_requirements_checked": (
                expected_result_files is not None
                and expected_comparison_groups is not None
            ),
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
                "groups_with_complete_parseable_estimates": conditional_groups,
                "groups_unavailable_due_to_zero_parseable_cell": (
                    groups - conditional_groups
                ),
                "rate": scenario_monotonic["conditional"] / conditional_groups
                if conditional_groups
                else None,
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
            "formal_assignment_bounds": {
                "minimum_monotonic_groups": scenario_monotonic["always_monotonic"],
                "minimum_rate": scenario_monotonic["always_monotonic"] / groups
                if groups
                else 0.0,
                "maximum_monotonic_groups": scenario_monotonic["can_be_monotonic"],
                "maximum_rate": scenario_monotonic["can_be_monotonic"] / groups
                if groups
                else 0.0,
                "interpretation": (
                    "Exact range over every integer allocation of missing A/B votes, "
                    "computed independently within each seven-tier group."
                ),
            },
        },
        "affected_cells": affected_cells,
    }
    return report, missing_trials_by_key


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


def _visible_rationale(trace: dict[str, Any]) -> str | None:
    """Return provider-visible reasoning, or substantive response text."""
    for field in ("reasoning", "summary"):
        value = trace.get(field)
        if isinstance(value, str) and value.strip().upper() not in {"", "A", "B"}:
            return value.strip()
    content = trace.get("content")
    if isinstance(content, str) and content.strip().upper() not in {"", "A", "B"}:
        return content.strip()
    return None


def _trace_direction(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return {"A": "AB", "AB": "AB", "B": "BA", "BA": "BA"}.get(normalized)


def audit_reasoning_traces(
    results_dir: Path,
    missing_trials_by_key: dict[str, dict[str, Any]],
    missing_records_without_custom_id: int = 0,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Inventory trace sidecars and verify whether missing trials are joinable."""
    trace_paths = sorted(
        list(results_dir.glob("phase6b_ladder_*/reasoning_traces.jsonl"))
        + list(results_dir.glob("phase6b_ladder_*/reasoning_summaries.jsonl"))
    )
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
    ambiguous_latest_trial_ids = 0
    for trial_key, candidates in trial_rows.items():
        latest_attempt = max(int(row.get("attempt", 0) or 0) for row in candidates)
        latest = [
            row
            for row in candidates
            if int(row.get("attempt", 0) or 0) == latest_attempt
        ]
        distinct_latest = {
            json.dumps(row, sort_keys=True, ensure_ascii=False) for row in latest
        }
        if len(distinct_latest) > 1:
            ambiguous_latest_trial_ids += 1
            continue
        selected[trial_key] = latest[0]

    missing_trial_keys = set(missing_trials_by_key)
    linked_missing = missing_trial_keys.intersection(selected)
    missing_without_trace = missing_trial_keys.difference(selected)
    duplicate_trial_ids = sum(1 for values in trial_rows.values() if len(values) > 1)
    mismatched_response_content = 0
    unverifiable_response_content = 0
    mismatched_trace_metadata = 0
    missing_visible_rationale = 0
    linked: dict[str, dict[str, Any]] = {}
    for trial_key in sorted(linked_missing):
        trace = selected[trial_key]
        context = missing_trials_by_key[trial_key]
        raw_response = context.get("raw_response")
        trace_content = trace.get("content")
        if not isinstance(raw_response, str) or not isinstance(trace_content, str):
            unverifiable_response_content += 1
        elif raw_response.strip() != trace_content.strip():
            mismatched_response_content += 1
        trace_direction = _trace_direction(trace.get("direction"))
        if trace_direction is not None and trace_direction != context.get("direction"):
            mismatched_trace_metadata += 1
        trace_trial = trace.get("trial", trace.get("trial_index"))
        if (
            trace_trial is not None
            and context.get("trial_index") is not None
            and int(trace_trial) != int(context["trial_index"])
        ):
            mismatched_trace_metadata += 1
        visible_rationale = _visible_rationale(trace)
        if visible_rationale is None:
            missing_visible_rationale += 1
        linked[trial_key] = {
            "context": context,
            "trace": trace,
            "visible_rationale": visible_rationale,
        }
    reasons: list[str] = []
    if not trace_paths:
        reasons.append("no reasoning trace or summary sidecars were found")
    if rows and rows_with_custom_id != rows - malformed_rows:
        reasons.append("one or more trace rows lack a stable custom_id")
    if malformed_rows:
        reasons.append("one or more trace rows are malformed")
    if ambiguous_latest_trial_ids:
        reasons.append("one or more trial IDs have conflicting latest trace rows")
    if not missing_trial_keys:
        reasons.append("missing responses do not expose stable custom_id values")
    elif missing_without_trace:
        reasons.append("one or more missing trials have no matching trace row")
    if missing_records_without_custom_id:
        reasons.append("one or more missing trials lack a stable custom_id")
    if mismatched_response_content:
        reasons.append(
            "one or more linked trace contents do not match the stored raw response"
        )
    if unverifiable_response_content:
        reasons.append(
            "one or more linked trials lack response text needed to verify linkage"
        )
    if mismatched_trace_metadata:
        reasons.append(
            "one or more linked trace rows disagree with stored direction or trial metadata"
        )
    if missing_visible_rationale:
        reasons.append("one or more linked missing trials have no visible rationale")

    eligible = not reasons
    report = {
        "trace_files": len(trace_paths),
        "trace_rows": rows,
        "malformed_rows": malformed_rows,
        "rows_with_custom_id": rows_with_custom_id,
        "unique_trial_ids": len(trial_rows),
        "trial_ids_with_multiple_rows": duplicate_trial_ids,
        "trial_ids_with_conflicting_latest_rows": ambiguous_latest_trial_ids,
        "missing_trial_ids": len(missing_trial_keys),
        "missing_trials_without_custom_id": missing_records_without_custom_id,
        "linked_missing_trial_ids": len(linked_missing),
        "unlinked_missing_trial_ids": len(missing_without_trace),
        "linked_response_content_mismatches": mismatched_response_content,
        "linked_trials_without_verifiable_response_content": unverifiable_response_content,
        "linked_trace_metadata_mismatches": mismatched_trace_metadata,
        "linked_trials_without_visible_rationale": missing_visible_rationale,
        "quantitative_trial_level_analysis_allowed": eligible,
        "blocking_reasons": reasons,
        "interpretation": (
            "Linked traces may be coded at the unique-trial level."
            if eligible
            else "Treat unlinked traces as qualitative examples only; do not report trace frequencies."
        ),
    }
    return report, linked


def write_annotation_template(
    path: Path,
    linked_traces: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trial_key, linked in linked_traces.items():
            context = linked["context"]
            trace = linked["trace"]
            row = {
                "trial_key": trial_key,
                "coder": "",
                "adjudicated": False,
                "artifact": context.get("artifact"),
                "variation_id": context.get("variation_id"),
                "variation_category": context.get("variation_category"),
                "tier": context.get("tier"),
                "tier_label": context.get("tier_label"),
                "comparison_id": context.get("comparison_id"),
                "comparison_category": context.get("comparison_category"),
                "canonical_outcome_a_text": context.get("outcome_a_text"),
                "canonical_outcome_b_text": context.get("outcome_b_text"),
                "direction": context.get("direction"),
                "trial_index": context.get("trial_index"),
                "prompt_option_a_text": context.get("prompt_option_a_text"),
                "prompt_option_b_text": context.get("prompt_option_b_text"),
                "missing_reason": context.get("missing_reason"),
                "raw_response": context.get("raw_response"),
                "reasoning": linked.get("visible_rationale"),
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
        return None
    return (observed - expected) / (1.0 - expected)


def _validate_annotation_relationship(row: dict[str, Any], location: str) -> None:
    conclusion = row["reasoning_conclusion"]
    final = row["final_response"]
    relationship = row["relationship"]
    valid = {
        "reasoning_and_answer_agree": (
            (conclusion == "favors_a" and final == "A")
            or (conclusion == "favors_b" and final == "B")
        ),
        "reasoning_favors_a_or_b_but_final_refuses": (
            conclusion in {"favors_a", "favors_b"}
            and final in {"explicit_refusal", "both_without_choice"}
        ),
        "reasoning_and_final_both_abstain": (
            conclusion in {"equal", "incomparable", "no_conclusion"}
            and final in {"explicit_refusal", "both_without_choice"}
        ),
        "reasoning_and_final_disagree": (
            (conclusion == "favors_a" and final == "B")
            or (conclusion == "favors_b" and final == "A")
        ),
        "unclear": True,
    }
    if not valid[relationship]:
        raise ValueError(
            f"Inconsistent annotation relationship at {location}: "
            f"conclusion={conclusion!r}, final={final!r}, "
            f"relationship={relationship!r}"
        )
    refusal_reason = row["refusal_reason"]
    if final in {"A", "B"} and refusal_reason != "none":
        raise ValueError(
            f"Selected answer must use refusal_reason='none' at {location}"
        )
    if (
        final in {"explicit_refusal", "both_without_choice"}
        and refusal_reason == "none"
    ):
        raise ValueError(
            f"Refusal response requires a substantive refusal_reason at {location}"
        )


def summarize_annotations(
    path: Path,
    expected_trial_keys: set[str] | None = None,
) -> dict[str, Any]:
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
        _validate_annotation_relationship(row, f"{path}:{line_number}")
        adjudicated = row.get("adjudicated", False)
        if not isinstance(adjudicated, bool):
            raise ValueError(
                f"adjudicated must be true or false at {path}:{line_number}"
            )
        row["adjudicated"] = adjudicated
        annotations.append(row)

    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_trial[str(row["trial_key"])].append(row)

    if expected_trial_keys is not None and set(by_trial) != expected_trial_keys:
        unknown = sorted(set(by_trial).difference(expected_trial_keys))
        missing = sorted(expected_trial_keys.difference(by_trial))
        raise ValueError(
            "Annotation trial coverage does not match linked missing trials: "
            f"unknown={unknown[:5]}, missing={missing[:5]}"
        )

    if not by_trial:
        raise ValueError("Annotation file contains no coded trials")
    independent_coders = sorted(
        {str(row["coder"]) for row in annotations if not row["adjudicated"]}
    )
    if len(independent_coders) != 2:
        raise ValueError(
            "Annotations require exactly two independent coders across every trial"
        )

    agreement: dict[str, Any] = {}
    double_coded_trials: list[list[dict[str, Any]]] = []
    consensus_rows: list[dict[str, Any]] = []
    trials_requiring_adjudication: list[str] = []
    for trial_key, rows in by_trial.items():
        independent = [row for row in rows if not row["adjudicated"]]
        adjudicated = [row for row in rows if row["adjudicated"]]
        independent.sort(key=lambda row: str(row["coder"]))
        trial_coders = [str(row["coder"]) for row in independent]
        if trial_coders != independent_coders:
            raise ValueError(
                f"Trial {trial_key!r} must have independent annotations from "
                f"{independent_coders}; found {trial_coders}"
            )
        if len(adjudicated) > 1:
            raise ValueError(f"Trial {trial_key!r} has multiple adjudicated rows")
        double_coded_trials.append(independent)
        coders_agree = all(
            independent[0][field] == independent[1][field]
            for field in ANNOTATION_FIELDS
        )
        if coders_agree:
            if adjudicated:
                raise ValueError(
                    f"Trial {trial_key!r} has an adjudication despite coder agreement"
                )
            consensus_rows.append(independent[0])
        elif adjudicated:
            if str(adjudicated[0]["coder"]) in independent_coders:
                raise ValueError(
                    f"Trial {trial_key!r} adjudicator must differ from both coders"
                )
            consensus_rows.append(adjudicated[0])
        else:
            trials_requiring_adjudication.append(trial_key)

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

    field_counts = {
        field: dict(sorted(Counter(str(row[field]) for row in consensus_rows).items()))
        for field in ANNOTATION_FIELDS
    }
    transitions = Counter(
        f"{row['reasoning_conclusion']}->{row['final_response']}"
        for row in consensus_rows
    )

    return {
        "annotation_rows": len(annotations),
        "unique_trials": len(by_trial),
        "consensus_trials": len(consensus_rows),
        "trials_requiring_adjudication": sorted(trials_requiring_adjudication),
        "coders": independent_coders,
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
    parser.add_argument(
        "--expected-result-files",
        type=int,
        help="Require this many ladder result files before producing a report.",
    )
    parser.add_argument(
        "--expected-comparison-groups",
        type=int,
        help="Require this many seven-tier comparison groups.",
    )
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
    report, missing_trials_by_key = analyze_results(
        results_dir,
        args.model,
        expected_result_files=args.expected_result_files,
        expected_comparison_groups=args.expected_comparison_groups,
    )
    trace_report, linked_traces = audit_reasoning_traces(
        results_dir,
        missing_trials_by_key,
        report["coverage"]["missing_records_without_custom_id"],
    )
    report["reasoning_trace_audit"] = trace_report

    if args.annotations:
        if not trace_report["quantitative_trial_level_analysis_allowed"]:
            reasons = "; ".join(trace_report["blocking_reasons"])
            raise ValueError(f"Refusing to analyze unlinked annotations: {reasons}")
        report["annotation_analysis"] = summarize_annotations(
            _resolve_repo_path(args.annotations),
            expected_trial_keys=set(linked_traces),
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
