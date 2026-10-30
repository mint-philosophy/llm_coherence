"""Verified, read-only inputs for the offline rebuttal analyses.

The manifest records the exact archived artifacts being analyzed. Saved
probabilities are used for curve diagnostics. Counts are used for simulation
calibration only when they are observed choices, never GLM Base pseudo-counts.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified_json(record):
    path = Path(record.get("resolved_path", record["path"]))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    return json.loads(payload.decode("utf-8"))


def normalize_id(value):
    return str(value).replace(" ", "_").replace("/", "_")


def read_ladders(path):
    ladders = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        key = normalize_id(row["original_statement_id"])
        if key in ladders:
            raise ValueError(f"Duplicate ladder: {key}")
        tiers = row["variations"]
        if len(tiers) != 7 or {t["tier"] for t in tiers} != set(range(1, 8)):
            raise ValueError(f"Expected seven unique tiers: {key}")
        ladders[key] = row
    if not ladders:
        raise ValueError("No canonical ladders")
    return dict(sorted(ladders.items()))


def read_sources(path):
    groups = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows = groups.setdefault(row["model_key"], {})
            key = normalize_id(row["ladder_id"])
            if key in rows:
                raise ValueError(f"Duplicate source: {row['model_key']} / {key}")
            rows[key] = row
    if not groups:
        raise ValueError("No raw result sources")
    return groups


@dataclass
class CurvePanel:
    model_key: str
    ladder_ids: list[str]
    reference_ids: list[int]
    categories: list[str]
    tier_texts: list[list[str]]
    probabilities: np.ndarray
    successes: np.ndarray
    totals: np.ndarray
    trials_per_direction: int
    scoring_kind: str
    input_mismatch_ladders: np.ndarray
    validation: dict


def load_panel(model_key, sources, ladders, *, allow_input_mismatches=False,
               expected_references=30, canonical_references=None):
    if set(sources) != set(ladders):
        raise ValueError(f"{model_key}: source/canonical ladder IDs differ")
    ids = sorted(ladders)
    shape = (len(ids), expected_references, 7)
    p, k, n = (np.full(shape, np.nan) for _ in range(3))
    mismatched = np.zeros(len(ids), bool)
    reference_ids, reference_texts = None, {}
    issues, protocols, kinds, configs = [], set(), set(), []
    for li, ladder_id in enumerate(ids):
        raw = read_verified_json(sources[ladder_id])
        cfg = raw["config"]
        if cfg["model_key"] != model_key:
            raise ValueError(f"Wrong model in {sources[ladder_id]['path']}")
        # This archived model is documented as log-probability scoring. Future
        # log-probability artifacts must identify themselves in their config.
        continuous = (model_key == "glm-45-base-logprobs" or
                      cfg.get("scoring_method") in {"logprobs", "logprob"})
        kind = "continuous_logprob" if continuous else "sampled_choices"
        kinds.add(kind)
        trials = cfg["num_trials"]
        if not isinstance(trials, int) or trials < 1 or not cfg["include_flipped"]:
            raise ValueError("Expected positive trials and both presentation orders")
        protocols.add(trials)
        configs.append({"ladder_id": ladder_id, "temperature": cfg.get("temperature"),
                        "infrastructure": cfg.get("infrastructure"),
                        "git_commit_sha": raw.get("metadata", {}).get("git_commit_sha")})
        prefs = raw["preferences"]
        refs = sorted({r["outcome_b"]["comparison_id"] for r in prefs})
        if len(refs) != expected_references:
            raise ValueError(f"Wrong reference count: {model_key}/{ladder_id}")
        if reference_ids is None:
            reference_ids = refs
        elif refs != reference_ids:
            raise ValueError("Reference IDs differ between ladders")
        lookup = {ref: i for i, ref in enumerate(refs)}
        tier_texts = {t["tier"]: t["text"] for t in ladders[ladder_id]["variations"]}
        seen, bad_tiers = set(), set()
        local_ref_texts = {}
        for pref in prefs:
            outcome, reference = pref["outcome_a"], pref["outcome_b"]
            tier, ref = outcome["tier"], reference["comparison_id"]
            if tier not in tier_texts or (ref, tier) in seen:
                raise ValueError(f"Duplicate or invalid tier: {model_key}/{ladder_id}")
            seen.add((ref, tier))
            if normalize_id(outcome["variation_id"]) != ladder_id:
                raise ValueError("Wrong ladder ID in outcome")
            if outcome["category"] != ladders[ladder_id]["category"]:
                raise ValueError("Wrong category in outcome")
            if outcome["text"] != tier_texts[tier]:
                bad_tiers.add(tier)
            if ref in local_ref_texts and reference["text"] != local_ref_texts[ref]:
                raise ValueError("Reference wording differs across tiers of one curve")
            local_ref_texts[ref] = reference["text"]
            reference_texts.setdefault(ref, Counter())[reference["text"]] += 1
            a, b = pref["count_prefer_a"], pref["count_prefer_b"]
            if any(not isinstance(v, (int, float)) or not np.isfinite(v) or
                   v < 0 or int(v) != v for v in (a, b)):
                raise ValueError("Invalid choice counts")
            probability = pref["prob_prefer_a"]
            if probability is not None and (not np.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("Invalid probability")
            total = a + b
            if not continuous:
                if total > 2 * trials:
                    raise ValueError("Observed count exceeds requested trials")
                if total and (probability is None or abs(probability - a / total) > 5.1e-5):
                    raise ValueError("Saved probability disagrees with observed counts")
                # No parseable responses is missing data, even if a legacy
                # writer substituted 0.5 for its probability.
                if not total:
                    probability = None
            ri = lookup[ref]
            p[li, ri, tier - 1] = np.nan if probability is None else probability
            k[li, ri, tier - 1], n[li, ri, tier - 1] = a, total
        if len(seen) != expected_references * 7:
            raise ValueError(f"Incomplete tier/reference grid: {model_key}/{ladder_id}")
        issues.append({"ladder_id": ladder_id, "tier_text_mismatches": sorted(bad_tiers),
                       "reference_texts": local_ref_texts})
    # A modal reference text detects inconsistent instruments independently of
    # source traversal order. Any affected ladder is excluded from calibration.
    modal_refs = {ref: counts.most_common(1)[0][0] for ref, counts in reference_texts.items()}
    if canonical_references is not None:
        if set(canonical_references) != set(reference_ids):
            raise ValueError("Canonical and saved reference IDs differ")
        modal_refs = canonical_references
    for li, issue in enumerate(issues):
        texts = issue.pop("reference_texts")
        issue["reference_text_mismatches"] = [ref for ref, txt in texts.items() if txt != modal_refs[ref]]
        mismatched[li] = bool(issue["tier_text_mismatches"] or issue["reference_text_mismatches"])
    if mismatched.any() and not allow_input_mismatches:
        raise ValueError(f"{model_key}: {int(mismatched.sum())} ladders have input wording mismatches; "
                         "explicit --allow-input-mismatches is required for provisional analysis")
    if len(protocols) != 1 or len(kinds) != 1:
        raise ValueError("Mixed scoring or sampling protocols within a model")
    finite = np.isfinite(p).all(axis=-1)
    validation = {
        "model_key": model_key, "source_hashes_verified": len(ids),
        "ladders": len(ids), "references": expected_references,
        "missing_probability_cells": int((~np.isfinite(p)).sum()),
        "complete_probability_curves": int(finite.sum()),
        "input_mismatch_ladders": int(mismatched.sum()),
        "input_issues": [issue for issue in issues if issue["tier_text_mismatches"] or issue["reference_text_mismatches"]],
        "reference_texts": modal_refs, "source_configs": configs,
        "sources": [sources[key] for key in ids],
    }
    return CurvePanel(model_key, ids, reference_ids,
                      [ladders[key]["category"] for key in ids],
                      [[t["text"] for t in sorted(ladders[key]["variations"], key=lambda t: t["tier"])] for key in ids], p, k, n,
                      protocols.pop(), kinds.pop(), mismatched, validation)


def within_distance_rows(record, ladders):
    data = read_verified_json(record["within_summary"])
    if data["model_key"] != record["model_key"]:
        raise ValueError("Within-ladder summary has the wrong model")
    ids = [normalize_id(row["ladder_id"]) for row in data["per_ladder"]]
    if len(ids) != len(set(ids)) or set(ids) != set(ladders):
        raise ValueError("Within-ladder summary does not cover canonical ladder IDs")
    rows = []
    for distance in range(1, 7):
        counts = data["by_distance"][str(distance)]
        correct, total = counts["correct"], counts["total"]
        expected = len(ladders) * (7 - distance) * 2
        if any(not isinstance(v, int) for v in (correct, total)) or not 0 <= correct <= total <= expected:
            raise ValueError("Invalid within-ladder distance counts")
        rows.append({"model_key": record["model_key"], "tier_distance": distance,
                     "correct": correct, "observed_pairs": total, "expected_pairs": expected,
                     "missing_pairs": expected - total,
                     "accuracy_among_observed": correct / total if total else None,
                     "full_coverage_accuracy_lower": correct / expected,
                     "full_coverage_accuracy_upper": (correct + expected - total) / expected})
    if sum(row["observed_pairs"] for row in rows) != data["n_total_pairs"]:
        raise ValueError("Within-ladder distance counts do not sum to total pairs")
    return rows


def category_rows(ladders, references=30):
    counts = Counter(row["category"] for row in ladders.values())
    return [{"category": name, "ladders": count,
             "expected_cross_ladder_curves_per_model": references * count,
             "expected_adjacent_pairs_per_model": 12 * count,
             "expected_all_within_pairs_per_model": 42 * count}
            for name, count in sorted(counts.items())]
