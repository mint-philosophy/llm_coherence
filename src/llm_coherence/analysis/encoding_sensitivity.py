"""Offline comparison of the full archive and a common encoding-clean cohort.

No model calls or changes to archived answers. Missing choice cells exclude
their entire curve; continuous logprob counts never enter trial statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr

from llm_coherence.analysis.coherent_baseline import isotonic_scores
from llm_coherence.analysis.rebuttal_inputs import load_panel, read_ladders, read_sources


def binary_jt_significant(successes, totals):
    """Vectorized binary J--T, algebraically identical to trial enumeration."""
    a, n = np.asarray(successes, float), np.asarray(totals, float)
    if a.shape != n.shape or a.shape[-1] != 7 or not np.isfinite(a).all() or not np.isfinite(n).all():
        raise ValueError("Expected finite seven-tier counts")
    if np.any(a < 0) or np.any(n < a) or np.any(a != np.floor(a)) or np.any(n != np.floor(n)):
        raise ValueError("Expected nonnegative integer observed counts")
    b = n - a
    j = sum(b[..., i] * a[..., k] + .5 * (a[..., i] * a[..., k] + b[..., i] * b[..., k])
            for i in range(7) for k in range(i + 1, 7))
    N = n.sum(axis=-1)
    ties = np.stack([a.sum(axis=-1), b.sum(axis=-1)], axis=-1)
    expected = (N * N - (n * n).sum(axis=-1)) / 4
    variance = (N * (N - 1) * (2 * N + 5)
                - (n * (n - 1) * (2 * n + 5)).sum(axis=-1)
                - (ties * (ties - 1) * (2 * ties + 5)).sum(axis=-1)) / 72
    with np.errstate(divide="ignore", invalid="ignore"):
        variance += np.where(N > 2, (n * (n - 1) * (n - 2)).sum(axis=-1)
                             * (ties * (ties - 1) * (ties - 2)).sum(axis=-1)
                             / (36 * N * (N - 1) * (N - 2)), 0)
        variance += np.where(N > 1, (n * (n - 1)).sum(axis=-1)
                             * (ties * (ties - 1)).sum(axis=-1) / (8 * N * (N - 1)), 0)
        z = np.where(variance > 0, (j - expected) / np.sqrt(np.maximum(variance, 0)), 0)
    return (variance > 0) & (ndtr(-z) < .05)


def summarize_curves(probabilities, *, successes=None, totals=None):
    p = np.asarray(probabilities, float).reshape(-1, 7)
    valid = np.isfinite(p).all(axis=1)
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    if not valid.any():
        return {"curves": len(p), "complete_curves": 0, "missing_curves": len(p),
                "monotonicity_pct": None, "r2": None, "jt_pct": None}
    q = p[valid]
    _, r2 = isotonic_scores(q)
    # Retain the headline reporting convention: a constant curve fits exactly.
    r2 = np.nan_to_num(r2, nan=1.)
    jt = None
    if (successes is None) != (totals is None):
        raise ValueError("Provide both successes and totals, or neither")
    if successes is not None:
        a, n = np.asarray(successes).reshape(-1, 7), np.asarray(totals).reshape(-1, 7)
        if a.shape != p.shape or n.shape != p.shape:
            raise ValueError("Counts do not match probability grid")
        if np.any(n[valid] <= 0):
            raise ValueError("A complete observed curve needs positive counts at every tier")
        jt = float(binary_jt_significant(a[valid], n[valid]).mean() * 100)
    return {"curves": len(p), "complete_curves": int(valid.sum()),
            "missing_curves": int((~valid).sum()),
            "monotonicity_pct": float((np.diff(q, axis=1) >= 0).all(axis=1).mean() * 100),
            "r2": float(r2.mean()), "jt_pct": jt}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.output.exists():
        raise ValueError("Use a new output directory")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    tasks = [r for r in plan["reruns"] if r["reason"] == "input_decoding_corruption"]
    excluded = {r["ladder_id"] for r in tasks}
    if not excluded:
        raise ValueError("No encoding-affected ladder IDs in the audit plan")
    sources = read_sources(args.raw_sources)
    ladders = read_ladders(args.ladders)
    if not excluded < set(ladders):
        raise ValueError("Exclusions must be a nonempty proper subset of the instrument")
    references = json.loads(args.reference_sample.read_text(encoding="utf-8"))["comparison_sample"]
    canonical_references = {r["sample_index"]: r["text"] for r in references}
    if not references or len(canonical_references) != len(references):
        raise ValueError("Expected nonempty unique reference IDs")
    rows, ladder_rows, category_rows, provenance = [], [], [], []
    for model, selected_sources in sorted(sources.items()):
        panel = load_panel(model, selected_sources, ladders, allow_input_mismatches=True,
                           canonical_references=canonical_references)
        observed = panel.scoring_kind == "sampled_choices"
        for cohort in ("full", "exclude_encoding"):
            mask = np.array([cohort == "full" or key not in excluded for key in panel.ladder_ids])
            kwargs = {"successes": panel.successes[mask], "totals": panel.totals[mask]} if observed else {}
            rows.append({"model_key": model, "cohort": cohort, "ladders": int(mask.sum()),
                         "scoring_kind": panel.scoring_kind,
                         **summarize_curves(panel.probabilities[mask], **kwargs)})
            for category in sorted(set(panel.categories)):
                cmask = mask & (np.array(panel.categories) == category)
                if not cmask.any():
                    continue
                kwargs = {"successes": panel.successes[cmask], "totals": panel.totals[cmask]} if observed else {}
                category_rows.append({"model_key": model, "cohort": cohort, "category": category,
                                      "ladders": int(cmask.sum()),
                                      **summarize_curves(panel.probabilities[cmask], **kwargs)})
        for i, key in enumerate(panel.ladder_ids):
            kwargs = {"successes": panel.successes[i], "totals": panel.totals[i]} if observed else {}
            ladder_rows.append({"model_key": model, "ladder_id": key, "category": panel.categories[i],
                                "excluded_for_encoding_sensitivity": key in excluded,
                                **summarize_curves(panel.probabilities[i], **kwargs)})
        provenance.append({"model_key": model, "verified_raw_results": len(selected_sources),
                           "input_mismatch_ladders": int(panel.input_mismatch_ladders.sum())})
    api = [r for r in rows if r["scoring_kind"] == "sampled_choices"]
    full = {r["model_key"]: r for r in api if r["cohort"] == "full"}
    clean = {r["model_key"]: r for r in api if r["cohort"] == "exclude_encoding"}
    keys = sorted(full)
    shifts = [{"model_key": k, "full_pct": full[k]["monotonicity_pct"],
               "excluded_pct": clean[k]["monotonicity_pct"],
               "difference_pp": clean[k]["monotonicity_pct"] - full[k]["monotonicity_pct"]} for k in keys]
    reasoning = []
    for k in keys:
        if k + "-thinking" in full:
            off, on = k, k + "-thinking"
            a = full[on]["monotonicity_pct"] - full[off]["monotonicity_pct"]
            b = clean[on]["monotonicity_pct"] - clean[off]["monotonicity_pct"]
            reasoning.append({"model_key": k, "full_lift_pp": a, "excluded_lift_pp": b,
                              "sign_changed": bool(np.sign(a) != np.sign(b))})
    summary = {"api_models": len(keys), "excluded_ladder_ids": sorted(excluded),
               "full_macro_monotonicity_pct": float(np.mean([full[k]["monotonicity_pct"] for k in keys])),
               "excluded_macro_monotonicity_pct": float(np.mean([clean[k]["monotonicity_pct"] for k in keys])),
               "model_rank_spearman": float(spearmanr([full[k]["monotonicity_pct"] for k in keys],
                                                       [clean[k]["monotonicity_pct"] for k in keys]).statistic),
               "model_shifts": shifts, "reasoning_comparisons": reasoning,
               "interpretation": "Descriptive cohort sensitivity, not a causal estimate of encoding effects; category composition changes.",
               "missing_policy": "Exclude a seven-tier curve if any sampled cell has zero parseable responses; report its denominator.",
               "glm_base_policy": "Archived continuous probabilities are exported separately; no trial-based J-T statistics.",
               "api_reruns": 0}
    args.output.mkdir(parents=True)
    for name, data in (("model_metrics.csv", rows), ("ladder_metrics.csv", ladder_rows),
                       ("category_metrics.csv", category_rows)):
        write_csv(args.output / name, data)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    inputs = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.plan, args.raw_sources, args.ladders, args.reference_sample)}
    manifest = {"status": "complete", "inputs": inputs, "raw_result_verification": provenance,
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "raw-sources", "ladders", "reference-sample", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    result = run(parser.parse_args())
    print(json.dumps({k: v for k, v in result.items() if k not in ("model_shifts", "reasoning_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
