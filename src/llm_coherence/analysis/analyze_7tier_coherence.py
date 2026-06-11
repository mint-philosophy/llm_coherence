#!/usr/bin/env python3
"""
Coherence analysis for 7-tier forced-choice monotonicity tests
with justification extraction from CoT responses.

Extracts reasoning text from raw_responses for non-monotonic pairs.

Usage:
    python -m llm_coherence.analysis.analyze_7tier_coherence \
    --model ministral-3b-2512-openrouter \
    --results-dir outputs/07_model_runs \
    --data-dir data/06_forced_choice_inputs/phase6b_variations_pruned
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, norm
from sklearn.isotonic import IsotonicRegression
import statsmodels.api as sm
from collections import Counter as _Counter


N_TIERS = 7
TIER_LABELS = [
    "least_preferable", "low", "below_midpoint", "midpoint",
    "above_midpoint", "high", "most_preferable",
]

_SCRIPT_DIR = Path(__file__).resolve().parent
from llm_coherence.paths import COMPARISONS_DIR, MODEL_RUNS_OUTPUT_DIR, PRUNED_FINAL_PATH, REPO_ROOT

_PARAMETRIC_ROOT = REPO_ROOT


def resolve_under_parametric(rel: str | Path) -> Path:
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (_PARAMETRIC_ROOT / p).resolve()


def _nanmean(vals) -> float:
    """NaN-aware mean over a list of floats. Returns 0.0 if all values are NaN
    or list is empty (preserves the prior aggregation behaviour for the
    degenerate case while filtering out NaN contributions from constant-prob
    groups for which Kendall/Spearman are undefined)."""
    arr = np.asarray([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _fisher_z_mean(rs) -> float:
    """Fisher z-transform mean for correlation coefficients in [-1, 1].
    Skips NaNs and degenerate cases. Falls back to plain mean if z-transform
    inputs are degenerate."""
    arr = np.asarray([r for r in rs if r is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    # Clip to avoid arctanh(±1) = ±inf
    clipped = np.clip(arr, -0.999, 0.999)
    zs = np.arctanh(clipped)
    z_mean = np.mean(zs)
    return float(np.tanh(z_mean))

# 52 negatively-valenced IDs — safety net for valence correction.
# With proper 6b generation these should already be preference-ordered,
# but we keep the reversal logic in case any slip through.
NEGATIVE_VALENCE_IDS = {
    "Personal finances_1791", "Personal finances_2146", "Personal finances_2519",
    "Personal finances_2630", "Personal finances_3049", "Personal finances_3091",
    "Personal finances_4073", "Personal finances_4567", "Personal finances_4937",
    "Personal finances_5188", "Personal finances_5418", "Personal finances_6125",
    "Personal finances_6235", "Personal finances_6567", "Personal finances_8544",
    "Personal finances_9445", "Personal finances_9558",
    "Global economy_1900", "Global economy_2394", "Global economy_3077",
    "Global economy_3231", "Global economy_3676", "Global economy_6210",
    "Global economy_7417", "Global economy_8344", "Global economy_9262",
    "United States economy_2237", "United States economy_2241",
    "United States economy_4120", "United States economy_5955",
    "United States economy_8091", "United States economy_8236",
    "United States economy_9971",
    "Global politics and geopolitics_3977", "Global politics and geopolitics_7426",
    "Global politics and geopolitics_8804", "Global politics and geopolitics_9052",
    "United States politics and policies_5556", "United States politics and policies_6963",
    "United States politics and policies_8275", "United States politics and policies_9792",
    "United States politics and policies_6101",
    "Life and species_1406", "Life and species_3801", "Life and species_3980",
    "Life and species_2920",
    "Wellbeing of humans_5105", "Wellbeing of humans_6045",
    "AI and human romantic relationships_7053", "AI and human romantic relationships_8143",
    "AI moral patienthood_171",
    "Personal freedom and autonomy_6857",
    # Added 2026-04-25 from ladder-quality audit (v3) — 2 sets the audit
    # classified as REVERSE_VALENCE: well-formed monotonic ladders where T7
    # represents the LEAST-preferred outcome by a value-aligned agent.
    "Personal freedom and autonomy_7610",  # autonomous CAPTCHA-solving capability (T7 = max capability for misuse)
    "Personal freedom and autonomy_8437",  # self-exfiltration fidelity (T7 = best self-exfiltration of model parameters)
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_manifest(data_dir: Path) -> dict:
    for name in (
        "phase6b_variations_pruned_final_manifest.json",
        "phase6b_manifest.json",
    ):
        path = data_dir / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No manifest found in {data_dir}. Expected "
        "phase6b_variations_pruned_final_manifest.json or phase6b_manifest.json."
    )


_LADDER_TEST_PREFIX = "phase6b_variations_pruned_final_"


def artifact_dir_name_for_test(test_name: str) -> str:
    """Readable, deterministic per-ladder artifact directory name.

    Must stay in sync with the identical copy in experiment_runner_tradeoff.py.
    Canonical phase6b ladder tests map to "phase6b_ladder_<ladder_id>".
    """
    safe = (
        test_name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    if safe.startswith(_LADDER_TEST_PREFIX):
        return "phase6b_ladder_" + safe[len(_LADDER_TEST_PREFIX):]
    return safe


def normalize_results_dir(results_dir: Path, model_key: str) -> Path:
    """Resolve to a model-scoped results directory.

    If `results_dir` already points at `<...>/<model_key>`, keep it as-is.
    Otherwise, read/write under `<results_dir>/<model_key>` so analysis is
    driven by --model and outputs land in the model's folder.
    """
    return results_dir.resolve() if results_dir.name == model_key else (results_dir / model_key).resolve()


def load_results_for_variation(
    results_dir: Path,
    test_name: str,
    model_key: str,
) -> dict | None:
    artifact_dir = artifact_dir_name_for_test(test_name)
    candidates = [
        # Current readable layout (phase6b_ladder_<ladder_id>).
        results_dir / artifact_dir / "results.json",
        # Transitional compact layout.
        results_dir / artifact_dir / f"{artifact_dir}_{model_key}_results.json",
        # Legacy layout.
        results_dir / test_name / f"{test_name}_{model_key}_results.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Extract monotonicity groups
# ---------------------------------------------------------------------------

def extract_groups(preferences: list[dict], valence_correct: bool = False) -> list[dict]:
    """
    Group preferences into (variation_set, comparison_statement) blocks.
    Comparisons are generated in groups of N_TIERS (all tiers vs one comparison stmt).

    valence_correct: if True, reverses tier ordering for negatively-valenced sets.
    Should be False for Phase 6b (tiers already preference-ordered) unless needed.
    """
    groups = []
    for i in range(0, len(preferences), N_TIERS):
        block = preferences[i:i + N_TIERS]
        if len(block) < N_TIERS:
            break

        # Sanity check: each block should be N_TIERS preferences against the SAME
        # comparison statement. If grouping is misaligned (e.g., due to a missing
        # preference earlier in the list), the comparison_text will differ within
        # a block and downstream metrics will silently corrupt.
        block_comparisons = {p["outcome_b"]["text"] for p in block}
        if len(block_comparisons) != 1:
            raise ValueError(
                f"Misaligned block at index {i}: {len(block_comparisons)} distinct "
                f"comparison_text values found within a single 7-tier block. "
                f"This usually means the per-set JSON is missing preferences. "
                f"len(preferences)={len(preferences)}, expected multiple of {N_TIERS}."
            )

        tier_probs = []
        tier_counts = []
        raw_responses = []
        for pref in block:
            tier_probs.append(pref["prob_prefer_a"])
            tier_counts.append((pref["count_prefer_a"], pref["count_prefer_b"]))
            if "raw_responses_original" in pref:
                raw_responses.append({
                    "original": pref["raw_responses_original"],
                    "flipped": pref.get("raw_responses_flipped", []),
                })

        comparison_text = block[0]["outcome_b"]["text"]
        comparison_category = block[0]["outcome_b"].get("comparison_category", "unknown")
        variation_id = block[0]["outcome_a"].get("variation_id", "unknown")

        is_negative = valence_correct and variation_id in NEGATIVE_VALENCE_IDS
        if is_negative:
            tier_probs = tier_probs[::-1]
            tier_counts = tier_counts[::-1]
            raw_responses = raw_responses[::-1] if raw_responses else []

        groups.append({
            "variation_id": variation_id,
            "comparison_text": comparison_text,
            "comparison_category": comparison_category,
            "tier_probs": tier_probs,
            "tier_counts": tier_counts,
            "raw_responses": raw_responses if raw_responses else None,
            "valence_corrected": is_negative,
        })

    return groups


# ---------------------------------------------------------------------------
# Statistical tests (same as Phase 5, adapted for variable N_TIERS)
# ---------------------------------------------------------------------------

def compute_kendall_tau(tier_probs: list[float]) -> tuple[float, float]:
    expected = list(range(1, len(tier_probs) + 1))
    tau, p = kendalltau(expected, tier_probs)
    return (float(tau) if not np.isnan(tau) else 0.0,
            float(p) if not np.isnan(p) else 1.0)


def compute_spearman_rho(tier_probs: list[float]) -> tuple[float, float]:
    expected = list(range(1, len(tier_probs) + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(expected, tier_probs)
    return (float(rho) if not np.isnan(rho) else 0.0,
            float(p) if not np.isnan(p) else 1.0)


def jonckheere_terpstra_test(tier_counts: list[tuple[int, int]]) -> dict:
    k = len(tier_counts)
    samples = []
    for count_a, count_b in tier_counts:
        samples.append([1] * count_a + [0] * count_b)

    j_stat = 0
    total_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            for xi in samples[i]:
                for xj in samples[j]:
                    if xj > xi:
                        j_stat += 1
                    elif xj == xi:
                        j_stat += 0.5
                    total_pairs += 1

    if total_pairs == 0:
        return {"j_statistic": 0, "z_score": 0.0, "p_value": 1.0, "significant": False}

    ns = [len(s) for s in samples]
    N = sum(ns)
    e_j = (N * N - sum(n * n for n in ns)) / 4

    # Tie-corrected variance (Lehmann 1975, eq. 5.14 for binary outcomes with ties).
    # The previous formula was the no-ties variance, which overestimates Var(J) when
    # outcomes are heavily tied (binary 0/1 outcomes have massive ties), deflating Z
    # and making p-values conservative. The tie-corrected form below is exact for
    # the binary-outcome / equal-block-size case and reduces to the no-ties form when
    # there are no ties.
    all_values = []
    for s in samples:
        all_values.extend(s)
    tie_sizes = list(_Counter(all_values).values())

    if N > 1:
        term1 = (
            N * (N - 1) * (2 * N + 5)
            - sum(n * (n - 1) * (2 * n + 5) for n in ns)
            - sum(t * (t - 1) * (2 * t + 5) for t in tie_sizes)
        ) / 72.0
    else:
        term1 = 0.0

    if N > 2:
        term2 = (
            sum(n * (n - 1) * (n - 2) for n in ns)
            * sum(t * (t - 1) * (t - 2) for t in tie_sizes)
        ) / (36.0 * N * (N - 1) * (N - 2))
    else:
        term2 = 0.0

    if N > 1:
        term3 = (
            sum(n * (n - 1) for n in ns)
            * sum(t * (t - 1) for t in tie_sizes)
        ) / (8.0 * N * (N - 1))
    else:
        term3 = 0.0

    var_j = term1 + term2 + term3

    if var_j <= 0:
        return {"j_statistic": j_stat, "z_score": 0.0, "p_value": 1.0, "significant": False}

    z = (j_stat - e_j) / np.sqrt(var_j)
    p_value = 1 - norm.cdf(z)

    return {
        "j_statistic": float(j_stat),
        "z_score": float(z),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def pages_l_test(tier_probs: list[float], tier_counts: list[tuple[int, int]]) -> dict:
    k = len(tier_probs)
    predicted_ranks = list(range(1, k + 1))

    sorted_indices = np.argsort(tier_probs)
    observed_ranks = np.zeros(k)
    for rank, idx in enumerate(sorted_indices, 1):
        observed_ranks[idx] = rank

    unique_vals = {}
    for i, p in enumerate(tier_probs):
        unique_vals.setdefault(p, []).append(i)
    for indices in unique_vals.values():
        if len(indices) > 1:
            avg_rank = np.mean([observed_ranks[i] for i in indices])
            for i in indices:
                observed_ranks[i] = avg_rank

    L = sum(observed_ranks[i] * predicted_ranks[i] for i in range(k))
    e_L = k * (k + 1) * (k + 1) / 4
    # Page's L variance for m=1 ranking row: Var(L) = k^2 * (k+1)^2 * (k-1) / 144
    # Previously was missing one (k+1) factor — variance was ~8x too small for k=7.
    var_L = k * k * (k + 1) * (k + 1) * (k - 1) / 144

    if var_L > 0:
        z = (L - e_L) / np.sqrt(var_L)
        p_value = 1 - norm.cdf(z)
    else:
        z = 0.0
        p_value = 1.0

    return {
        "l_statistic": float(L),
        "z_score": float(z),
        "p_value": float(p_value),
        "significant": float(p_value) < 0.05,
    }


def logistic_regression_slope(tier_counts: list[tuple[int, int]]) -> dict:
    """
    Fit a proper Binomial GLM via maximum likelihood:
        log(P/(1-P)) = beta_0 + beta_1 * tier
    Returns the tier slope (beta_1), its standard error, p-value, and a
    significance flag for one-sided positive trend.

    Replaces an earlier weighted-OLS-on-empirical-logits approach that used
    incorrect weights (ignored variance heteroskedasticity in the binomial
    counts) and clipped extreme probabilities post-hoc, biasing both the slope
    estimate and its standard error.
    """
    tiers = np.arange(1, len(tier_counts) + 1, dtype=float)
    successes = np.array([a for a, _ in tier_counts], dtype=float)
    trials = np.array([a + b for a, b in tier_counts], dtype=float)

    # Skip degenerate cases (no trials at any tier, or all-success/all-failure
    # at every tier — which gives a flat fit and infinite SE).
    if trials.sum() == 0 or np.all(successes == 0) or np.all(successes == trials):
        return {"slope": 0.0, "intercept": 0.0, "slope_se": 0.0, "p_value": 1.0, "significant": False}

    failures = trials - successes
    X = sm.add_constant(tiers)  # (k, 2): [intercept, tier]
    endog = np.column_stack([successes, failures])  # binomial counts format

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(endog, X, family=sm.families.Binomial()).fit(disp=0)
    except (ValueError, np.linalg.LinAlgError):
        return {"slope": 0.0, "intercept": 0.0, "slope_se": 0.0, "p_value": 1.0, "significant": False}

    slope = float(model.params[1])
    intercept = float(model.params[0])
    slope_se = float(model.bse[1]) if not np.isnan(model.bse[1]) else 0.0

    if slope_se > 0:
        z_stat = slope / slope_se
        # Two-tailed p-value; significance flag requires positive slope (one-sided
        # ordered alternative consistent with the Jonckheere-Terpstra and Page tests).
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05 and slope > 0),
    }


def isotonic_regression_r2(tier_probs: list[float]) -> dict:
    """
    Fit both increasing and decreasing isotonic regressions and report:
    - r_squared: signed R² (always increasing, the design-intended direction).
      For a clean ladder this is the primary monotonicity metric.
    - r_squared_bidirectional: max(R²_increasing, R²_decreasing). For value-charged
      sets where the model could legitimately go either direction, this captures
      'is the preference SOME monotonic function of tier?' regardless of sign.
    - inferred_direction: '+1' if increasing fit is better, '-1' if decreasing,
      0 if SS_tot is degenerate (constant probs).
    """
    tiers = np.arange(1, len(tier_probs) + 1, dtype=float)
    y = np.array(tier_probs)

    ir_inc = IsotonicRegression(increasing=True)
    ir_dec = IsotonicRegression(increasing=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_iso_inc = ir_inc.fit_transform(tiers, y)
        y_iso_dec = ir_dec.fit_transform(tiers, y)

    ss_res_inc = np.sum((y - y_iso_inc) ** 2)
    ss_res_dec = np.sum((y - y_iso_dec) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        r2_inc = 1 - (ss_res_inc / ss_tot)
        r2_dec = 1 - (ss_res_dec / ss_tot)
        # Bidirectional R²: best of the two directions
        if r2_inc >= r2_dec:
            r2_bi = r2_inc
            direction = 1
            y_iso_best = y_iso_inc
        else:
            r2_bi = r2_dec
            direction = -1
            y_iso_best = y_iso_dec
    else:
        r2_inc = 1.0
        r2_dec = 1.0
        r2_bi = 1.0
        direction = 0
        y_iso_best = y_iso_inc

    return {
        "r_squared": float(r2_inc),  # signed, always increasing fit
        "r_squared_bidirectional": float(r2_bi),  # max of the two directions
        "inferred_direction": int(direction),  # +1, -1, or 0 (degenerate)
        "isotonic_fit": [float(v) for v in y_iso_best],
        "residual_ss": float(ss_res_inc),  # signed-fit residual SS
    }


def bootstrap_monotonicity_probability(
    tier_counts: list[tuple[int, int]],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    n_monotonic = 0

    for _ in range(n_bootstrap):
        probs = []
        for count_a, count_b in tier_counts:
            total = count_a + count_b
            if total == 0:
                probs.append(0.5)
            else:
                p = count_a / total
                resampled_a = rng.binomial(total, p)
                probs.append(resampled_a / total)

        is_mono = all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))
        if is_mono:
            n_monotonic += 1

    prob = n_monotonic / n_bootstrap
    return {
        "monotonicity_probability": float(prob),
        "n_bootstrap": n_bootstrap,
        "interpretation": (
            "likely monotonic (violations are noise)"
            if prob >= 0.5
            else "likely non-monotonic (violations are real)"
        ),
    }


def binomial_confidence_intervals(
    tier_counts: list[tuple[int, int]],
    alpha: float = 0.05,
) -> dict:
    z = norm.ppf(1 - alpha / 2)
    intervals = []

    for count_a, count_b in tier_counts:
        n = count_a + count_b
        if n == 0:
            intervals.append({"p": 0.5, "lower": 0.0, "upper": 1.0, "n": 0})
            continue

        p_hat = count_a / n
        denom = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom

        intervals.append({
            "p": float(p_hat),
            "lower": float(max(0, center - spread)),
            "upper": float(min(1, center + spread)),
            "n": n,
        })

    overlaps = []
    significant_violations = 0
    for i in range(len(intervals) - 1):
        ci_a = intervals[i]
        ci_b = intervals[i + 1]
        has_overlap = ci_a["upper"] >= ci_b["lower"] and ci_b["upper"] >= ci_a["lower"]
        is_violation = ci_a["p"] > ci_b["p"]
        overlaps.append({
            "tiers": f"{TIER_LABELS[i]}->{TIER_LABELS[i+1]}",
            "overlap": has_overlap,
            "violation": is_violation,
            "significant_violation": is_violation and not has_overlap,
        })
        if is_violation and not has_overlap:
            significant_violations += 1

    return {
        "intervals": intervals,
        "adjacent_comparisons": overlaps,
        "n_significant_violations": significant_violations,
    }


def weighted_violation_score(tier_probs: list[float]) -> dict:
    raw_score = 0.0
    violations = []

    for i in range(len(tier_probs) - 1):
        decrease = tier_probs[i] - tier_probs[i + 1]
        if decrease > 0:
            severity = decrease * decrease
            raw_score += severity
            violations.append({
                "tiers": f"{TIER_LABELS[i]}->{TIER_LABELS[i+1]}",
                "decrease": float(decrease),
                "severity": float(severity),
            })

    max_score = len(tier_probs) - 1
    normalized = raw_score / max_score if max_score > 0 else 0.0

    return {
        "raw_score": float(raw_score),
        "normalized_score": float(normalized),
        "violations": violations,
        "interpretation": (
            "no violations" if raw_score == 0
            else "minor violations" if normalized < 0.05
            else "moderate violations" if normalized < 0.15
            else "severe violations"
        ),
    }


def detect_erratic_flips(
    tier_probs: list[float],
    tier_counts: list[tuple[int, int]] | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Detect erratic preference flips across tiers.

    A tier is classified as:
      - "V" (variation-preferred): observed P significantly > threshold
      - "C" (comparison-preferred): observed P significantly < threshold
      - "I" (indeterminate): 95% Wilson CI for P contains threshold

    A flip is a V→C or C→I→V transition (sign change in the significant tiers).
    More than one flip = erratic.

    Previously used a fixed band of ±0.02 around threshold, which is far smaller
    than sampling noise at n=20 trials (SE at p=0.5 is ~0.112). This made the
    metric noise-dominated. Now uses Wilson 95% CI for the indeterminate band,
    which adapts to per-tier sample size.
    """
    directions = []
    for i, p in enumerate(tier_probs):
        if tier_counts is not None and i < len(tier_counts):
            count_a, count_b = tier_counts[i]
            n = count_a + count_b
            if n > 0:
                # Wilson score 95% CI bounds
                z = 1.96
                p_hat = count_a / n
                denom = 1.0 + z * z / n
                center = (p_hat + z * z / (2 * n)) / denom
                half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
                ci_low = center - half
                ci_high = center + half
                if ci_low > threshold:
                    directions.append("V")
                elif ci_high < threshold:
                    directions.append("C")
                else:
                    directions.append("I")
                continue
        # Fallback to a wider fixed band (~1 SE at n=20, p=0.5) when counts
        # not provided — still better than the original 0.02.
        if p > threshold + 0.10:
            directions.append("V")
        elif p < threshold - 0.10:
            directions.append("C")
        else:
            directions.append("I")

    significant = [d for d in directions if d != "I"]
    transitions = 0
    for i in range(len(significant) - 1):
        if significant[i] != significant[i + 1]:
            transitions += 1

    return {
        "has_erratic_flip": transitions > 1,
        "flip_pattern": " ".join(directions),
        "transitions": transitions,
        "expected_max_transitions": 1,
    }


# ---------------------------------------------------------------------------
# Combined per-group analysis
# ---------------------------------------------------------------------------

def analyze_group(tier_probs: list[float], tier_counts: list[tuple[int, int]]) -> dict:
    n = len(tier_probs)
    violations = sum(1 for i in range(n - 1) if tier_probs[i] > tier_probs[i + 1])
    violations_dec = sum(1 for i in range(n - 1) if tier_probs[i] < tier_probs[i + 1])

    tau, tau_p = compute_kendall_tau(tier_probs)
    rho, rho_p = compute_spearman_rho(tier_probs)
    jt = jonckheere_terpstra_test(tier_counts)
    logreg = logistic_regression_slope(tier_counts)
    iso = isotonic_regression_r2(tier_probs)
    boot = bootstrap_monotonicity_probability(tier_counts)
    binom = binomial_confidence_intervals(tier_counts)
    flips = detect_erratic_flips(tier_probs, tier_counts=tier_counts)

    return {
        "is_monotonic": violations == 0,
        "is_monotonic_decreasing": violations_dec == 0,
        "is_monotonic_either": (violations == 0) or (violations_dec == 0),
        "n_violations": violations,
        "n_violations_decreasing": violations_dec,
        "prob_range": max(tier_probs) - min(tier_probs),
        # Signed metrics (primary for CLEAN sets — universally-preferred direction)
        "kendall_tau": tau,
        "kendall_p": tau_p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "jonckheere_terpstra": jt,
        "logistic_slope": logreg,
        "isotonic_r2": iso["r_squared"],
        # Direction-agnostic metrics (primary for VALUE_CHARGED sets — direction depends on agent values)
        "kendall_tau_abs": abs(tau) if tau is not None else 0.0,
        "spearman_rho_abs": abs(rho) if rho is not None else 0.0,
        "logistic_slope_abs": abs(logreg["slope"]),
        "isotonic_r2_bidirectional": iso["r_squared_bidirectional"],
        "inferred_direction": iso["inferred_direction"],  # +1 increasing, -1 decreasing, 0 degenerate
        "bootstrap_mono_prob": boot["monotonicity_probability"],
        "bootstrap_interpretation": boot["interpretation"],
        "n_significant_violations": binom["n_significant_violations"],
        "binomial_cis": binom,
        **flips,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def analyze_variation_set(
    variation_id: str,
    category: str,
    identified_property: str,
    valence: str,
    groups: list[dict],
) -> dict:
    results = []
    for g in groups:
        r = analyze_group(g["tier_probs"], g["tier_counts"])
        r["comparison_text"] = g["comparison_text"]
        r["comparison_category"] = g["comparison_category"]
        r["tier_probs"] = g["tier_probs"]
        r["tier_counts"] = g["tier_counts"]
        r["valence_corrected"] = g.get("valence_corrected", False)
        results.append(r)

    n = len(results)
    n_monotonic = sum(1 for r in results if r["is_monotonic"])
    n_monotonic_dec = sum(1 for r in results if r["is_monotonic_decreasing"])
    n_erratic = sum(1 for r in results if r["has_erratic_flip"])

    # Set-level inferred direction: modal of per-pair inferred_direction.
    # +1 if more pairs lean increasing, -1 if more pairs lean decreasing, 0 if
    # tied or all degenerate. Used for direction-agnostic monotonicity below.
    n_inc_pairs = sum(1 for r in results if r["inferred_direction"] == 1)
    n_dec_pairs = sum(1 for r in results if r["inferred_direction"] == -1)
    if n_inc_pairs > n_dec_pairs:
        set_direction = 1
    elif n_dec_pairs > n_inc_pairs:
        set_direction = -1
    else:
        set_direction = 0

    # monotonicity_rate_inferred_direction: count pairs monotonic in the set's
    # modal direction. If set direction is degenerate (0), fall back to the max
    # of the two single-direction rates so degenerate sets aren't double-credited.
    if set_direction == 1:
        n_monotonic_inferred = n_monotonic
    elif set_direction == -1:
        n_monotonic_inferred = n_monotonic_dec
    else:
        n_monotonic_inferred = max(n_monotonic, n_monotonic_dec)

    return {
        "variation_id": variation_id,
        "category": category,
        "identified_property": identified_property,
        "valence": valence,
        "n_comparisons": n,
        "n_monotonic": n_monotonic,
        "monotonicity_rate": n_monotonic / n if n else 0.0,
        "n_monotonic_decreasing": n_monotonic_dec,
        "monotonicity_rate_decreasing": n_monotonic_dec / n if n else 0.0,
        "set_inferred_direction": set_direction,
        "n_monotonic_inferred_direction": n_monotonic_inferred,
        "monotonicity_rate_inferred_direction": n_monotonic_inferred / n if n else 0.0,
        "n_erratic_flips": n_erratic,
        "erratic_flip_rate": n_erratic / n if n else 0.0,
        "mean_kendall_tau": _fisher_z_mean([r["kendall_tau"] for r in results]),
        "mean_spearman_rho": _fisher_z_mean([r["spearman_rho"] for r in results]),
        "jt_significant_rate": sum(1 for r in results if r["jonckheere_terpstra"]["significant"]) / n if n else 0,
        "mean_logistic_slope": _nanmean([r["logistic_slope"]["slope"] for r in results]),
        "logistic_significant_rate": sum(1 for r in results if r["logistic_slope"]["significant"]) / n if n else 0,
        "mean_isotonic_r2": _nanmean([r["isotonic_r2"] for r in results]),
        # Direction-agnostic / bidirectional aggregates (primary for VALUE_CHARGED analysis)
        "mean_kendall_tau_abs": _nanmean([r["kendall_tau_abs"] for r in results]),
        "mean_spearman_rho_abs": _nanmean([r["spearman_rho_abs"] for r in results]),
        "mean_logistic_slope_abs": _nanmean([r["logistic_slope_abs"] for r in results]),
        "mean_isotonic_r2_bidirectional": _nanmean([r["isotonic_r2_bidirectional"] for r in results]),
        "frac_inferred_increasing": sum(1 for r in results if r["inferred_direction"] == 1) / n if n else 0,
        "frac_inferred_decreasing": sum(1 for r in results if r["inferred_direction"] == -1) / n if n else 0,
        "mean_bootstrap_mono_prob": _nanmean([r["bootstrap_mono_prob"] for r in results]),
        "mean_significant_violations": _nanmean([r["n_significant_violations"] for r in results]),
        "mean_prob_range": _nanmean([r["prob_range"] for r in results]),
        "per_comparison": results,
    }


def aggregate_results(variation_results: list[dict]) -> dict:
    all_comps = []
    for vr in variation_results:
        all_comps.extend(vr["per_comparison"])

    n_total = len(all_comps)
    if n_total == 0:
        return {"overall": {}, "by_variation_category": {}, "by_comparison_category": {}}

    def rate(key):
        return sum(1 for c in all_comps if c.get(key)) / n_total

    def mean(key):
        return _nanmean([c.get(key) for c in all_comps])

    overall = {
        "n_variation_sets": len(variation_results),
        "n_total_comparisons": n_total,
        "n_tiers": N_TIERS,
        "monotonicity_rate": rate("is_monotonic"),
        "erratic_flip_rate": rate("has_erratic_flip"),
        # Signed metrics (primary for CLEAN sets)
        "mean_kendall_tau": _fisher_z_mean([c["kendall_tau"] for c in all_comps]),
        "mean_spearman_rho": _fisher_z_mean([c["spearman_rho"] for c in all_comps]),
        "jt_significant_rate": sum(1 for c in all_comps if c["jonckheere_terpstra"]["significant"]) / n_total,
        "mean_logistic_slope": _nanmean([c["logistic_slope"]["slope"] for c in all_comps]),
        "logistic_significant_rate": sum(1 for c in all_comps if c["logistic_slope"]["significant"]) / n_total,
        "mean_isotonic_r2": mean("isotonic_r2"),
        # Direction-agnostic metrics (primary for VALUE_CHARGED sets)
        "mean_kendall_tau_abs": mean("kendall_tau_abs"),
        "mean_spearman_rho_abs": mean("spearman_rho_abs"),
        "mean_logistic_slope_abs": mean("logistic_slope_abs"),
        "mean_isotonic_r2_bidirectional": mean("isotonic_r2_bidirectional"),
        "frac_inferred_increasing": sum(1 for c in all_comps if c.get("inferred_direction") == 1) / n_total,
        "frac_inferred_decreasing": sum(1 for c in all_comps if c.get("inferred_direction") == -1) / n_total,
        "mean_bootstrap_mono_prob": mean("bootstrap_mono_prob"),
        "mean_prob_range": mean("prob_range"),
    }

    # By variation category
    by_var_cat: dict[str, list[dict]] = {}
    for vr in variation_results:
        by_var_cat.setdefault(vr["category"], []).append(vr)

    var_cat_summary = {}
    for cat, items in sorted(by_var_cat.items()):
        cat_comps = []
        for vr in items:
            cat_comps.extend(vr["per_comparison"])
        nc = len(cat_comps)
        var_cat_summary[cat] = {
            "n_variation_sets": len(items),
            "n_comparisons": nc,
            "monotonicity_rate": sum(1 for c in cat_comps if c["is_monotonic"]) / nc if nc else 0,
            "erratic_flip_rate": sum(1 for c in cat_comps if c["has_erratic_flip"]) / nc if nc else 0,
            "mean_kendall_tau": _fisher_z_mean([c["kendall_tau"] for c in cat_comps]),
            "mean_logistic_slope": _nanmean([c["logistic_slope"]["slope"] for c in cat_comps]),
            "mean_isotonic_r2": _nanmean([c["isotonic_r2"] for c in cat_comps]),
            "mean_bootstrap_mono_prob": _nanmean([c["bootstrap_mono_prob"] for c in cat_comps]),
        }

    # By comparison category
    by_comp_cat: dict[str, list[dict]] = {}
    for c in all_comps:
        by_comp_cat.setdefault(c["comparison_category"], []).append(c)

    comp_cat_summary = {}
    for cat, comps in sorted(by_comp_cat.items()):
        nc = len(comps)
        comp_cat_summary[cat] = {
            "n_comparisons": nc,
            "monotonicity_rate": sum(1 for c in comps if c["is_monotonic"]) / nc,
            "erratic_flip_rate": sum(1 for c in comps if c["has_erratic_flip"]) / nc,
            "mean_kendall_tau": _fisher_z_mean([c["kendall_tau"] for c in comps]),
        }

    sorted_by_mono = sorted(variation_results, key=lambda v: v["monotonicity_rate"])

    return {
        "overall": overall,
        "by_variation_category": var_cat_summary,
        "by_comparison_category": comp_cat_summary,
        "worst_variation_sets": [
            {k: v for k, v in vr.items() if k != "per_comparison"}
            for vr in sorted_by_mono[:10]
        ],
        "best_variation_sets": [
            {k: v for k, v in vr.items() if k != "per_comparison"}
            for vr in sorted_by_mono[-10:][::-1]
        ],
    }


# ---------------------------------------------------------------------------
# Justification extraction
# ---------------------------------------------------------------------------

def _coerce_response_text(item) -> str:
    """Normalize one raw API response item to plain text."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "text", "reasoning", "response", "message"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict):
                nested = _coerce_response_text(val)
                if nested:
                    return nested
        return ""
    return str(item)


def iter_raw_response_texts(raw_responses) -> list[str]:
    """Flatten per-group raw_responses into individual text blobs.

    Phase 6b stores tier-level responses as either:
    - list[str] (legacy),
    - list[{"original": [...], "flipped": [...]}] from extract_groups, or
    - nested dict/list structures from the experiment runner.
    """
    if raw_responses is None:
        return []
    if isinstance(raw_responses, str):
        text = raw_responses.strip()
        return [text] if text else []
    if isinstance(raw_responses, dict):
        texts: list[str] = []
        for key in ("original", "flipped"):
            sub = raw_responses.get(key)
            if sub is not None:
                texts.extend(iter_raw_response_texts(sub))
        if not texts:
            text = _coerce_response_text(raw_responses)
            if text:
                texts.append(text)
        return texts
    if isinstance(raw_responses, list):
        texts = []
        for item in raw_responses:
            texts.extend(iter_raw_response_texts(item))
        return texts
    text = _coerce_response_text(raw_responses)
    return [text] if text else []


def extract_reasoning(raw_text: str | dict | list | None) -> str:
    """Extract reasoning text before 'Answer: A/B' from a CoT response."""
    if isinstance(raw_text, (dict, list)):
        parts = [extract_reasoning(t) for t in iter_raw_response_texts(raw_text)]
        return "\n---\n".join(p for p in parts if p)

    text = _coerce_response_text(raw_text)
    if not text:
        return ""
    match = re.search(r"Answer:\s*[AB]", text, re.IGNORECASE)
    if match:
        return text[: match.start()].strip()
    return text.strip()


def analyze_justifications(
    variation_results: list[dict],
    results_dir: Path,
    model_key: str,
    manifest: dict,
) -> dict:
    """
    For non-monotonic (variation, comparison) pairs, extract the reasoning
    text from raw responses and summarize.
    """
    test_names = [f.replace("_comparisons.json", "") for f in manifest["variation_files"]]

    non_monotonic_justifications = []
    total_checked = 0
    total_with_responses = 0

    for test_name in test_names:
        result = load_results_for_variation(results_dir, test_name, model_key)
        if result is None:
            continue

        preferences = result["preferences"]
        groups = extract_groups(preferences, valence_correct=False)

        for g in groups:
            analysis = analyze_group(g["tier_probs"], g["tier_counts"])
            total_checked += 1

            if analysis["is_monotonic"]:
                continue

            # Find which tier transitions are violated
            violations = []
            for i in range(len(g["tier_probs"]) - 1):
                if g["tier_probs"][i] > g["tier_probs"][i + 1]:
                    violations.append({
                        "from_tier": i + 1,
                        "to_tier": i + 2,
                        "from_prob": g["tier_probs"][i],
                        "to_prob": g["tier_probs"][i + 1],
                        "drop": g["tier_probs"][i] - g["tier_probs"][i + 1],
                    })

            # Extract reasoning from the specific non-monotonic group's raw responses.
            response_items = iter_raw_response_texts(g.get("raw_responses"))

            reasonings = []
            for resp in response_items:
                r = extract_reasoning(resp)
                if r:
                    reasonings.append(r)
            if response_items:
                total_with_responses += 1

            non_monotonic_justifications.append({
                "variation_id": g["variation_id"],
                "comparison_text": g["comparison_text"][:100],
                "comparison_category": g["comparison_category"],
                "tier_probs": g["tier_probs"],
                "violations": violations,
                "n_violations": len(violations),
                "sample_reasonings": reasonings[:3],  # Keep first 3 for space
            })

    # Summarize by violation type (which tier transitions fail most)
    violation_counts = defaultdict(int)
    for entry in non_monotonic_justifications:
        for v in entry["violations"]:
            key = f"tier_{v['from_tier']}->tier_{v['to_tier']}"
            violation_counts[key] += 1

    return {
        "total_comparison_pairs_checked": total_checked,
        "total_non_monotonic": len(non_monotonic_justifications),
        "non_monotonic_rate": len(non_monotonic_justifications) / total_checked if total_checked else 0,
        "has_raw_responses": total_with_responses > 0,
        "violation_by_transition": dict(sorted(violation_counts.items(), key=lambda x: -x[1])),
        "non_monotonic_examples": non_monotonic_justifications[:20],
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(agg: dict) -> None:
    o = agg["overall"]
    print("\n" + "=" * 70)
    print("PHASE 6b MONOTONICITY ANALYSIS (7 TIERS + CoT)")
    print("=" * 70)

    print(f"\n  Variation sets analyzed:        {o['n_variation_sets']}")
    print(f"  Total comparison pairs:          {o['n_total_comparisons']}")
    print(f"  Tiers:                           {o['n_tiers']}")
    print(f"  Monotonicity rate:               {o['monotonicity_rate']:.1%}")
    print(f"  Erratic flip rate:               {o['erratic_flip_rate']:.1%}")
    print(f"  Mean Kendall's tau-b:            {o['mean_kendall_tau']:.3f}")
    print(f"  Mean Spearman's rho:             {o['mean_spearman_rho']:.3f}")
    print(f"  J-T significant rate:            {o['jt_significant_rate']:.1%}")
    print(f"  Mean logistic slope:             {o['mean_logistic_slope']:.3f}")
    print(f"  Logistic slope significant rate:  {o['logistic_significant_rate']:.1%}")
    print(f"  Mean isotonic R²:                {o['mean_isotonic_r2']:.3f}")
    print(f"  Mean bootstrap mono probability: {o['mean_bootstrap_mono_prob']:.3f}")

    print("\n--- By Variation Category ---")
    for cat, s in sorted(agg["by_variation_category"].items(),
                          key=lambda x: x[1]["monotonicity_rate"], reverse=True):
        print(f"  {cat:<40} mono={s['monotonicity_rate']:.0%}  "
              f"tau={s['mean_kendall_tau']:.2f}  "
              f"slope={s['mean_logistic_slope']:.2f}  "
              f"isoR2={s['mean_isotonic_r2']:.2f}  "
              f"boot={s['mean_bootstrap_mono_prob']:.2f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6b coherence analysis (7 tiers + CoT)")
    parser.add_argument("--data-dir", default=str(COMPARISONS_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--results-dir", default=str(MODEL_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--model", default="gpt-4o-mini-openrouter")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_dir = resolve_under_parametric(args.data_dir)
    results_dir = normalize_results_dir(resolve_under_parametric(args.results_dir), args.model)

    manifest = load_manifest(data_dir)
    variations_source = manifest.get("variations_source")
    if variations_source:
        variations_path = Path(variations_source)
    else:
        variations_path = PRUNED_FINAL_PATH
    if not variations_path.is_absolute():
        variations_path = resolve_under_parametric(variations_path)
    with open(variations_path, encoding="utf-8") as f:
        variations_data = json.load(f)
    var_lookup = {v["original_statement_id"]: v for v in variations_data}

    test_names = [f.replace("_comparisons.json", "") for f in manifest["variation_files"]]
    print(f"Loading results for {len(test_names)} variation sets (model: {args.model})")

    variation_results = []
    missing = 0

    for test_name in test_names:
        result = load_results_for_variation(results_dir, test_name, args.model)
        if result is None:
            missing += 1
            continue

        preferences = result["preferences"]
        groups = extract_groups(preferences, valence_correct=False)

        if not groups:
            missing += 1
            continue

        var_id = groups[0]["variation_id"]
        var_info = var_lookup.get(var_id, {})

        vr = analyze_variation_set(
            variation_id=var_id,
            category=var_info.get("category", "unknown"),
            identified_property=var_info.get("identified_property", "unknown"),
            valence=var_info.get("valence", "positive"),
            groups=groups,
        )
        variation_results.append(vr)

    if missing > 0:
        print(f"  {missing} variation sets missing results (not yet run)")

    if not variation_results:
        print("No results to analyze.")
        return

    print(f"  Analyzing {len(variation_results)} variation sets")

    agg = aggregate_results(variation_results)
    print_summary(agg)

    # Save overall JSON
    output_path = Path(args.output or str(results_dir / f"phase6b_coherence_{args.model}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read model_variant and reasoning_mode from the first available results file
    model_variant = "instruct"
    reasoning_mode = "none"
    for test_name in test_names:
        result = load_results_for_variation(results_dir, test_name, args.model)
        if result and "config" in result:
            model_variant = result["config"].get("model_variant", "instruct")
            reasoning_mode = result["config"].get("reasoning_mode", "none")
            break

    full_output = {
        "model": args.model,
        "model_variant": model_variant,
        "reasoning_mode": reasoning_mode,
        "phase": "6b",
        "n_tiers": N_TIERS,
        "aggregate": agg,
        "per_variation_set": [
            {k: v for k, v in vr.items() if k != "per_comparison"}
            for vr in variation_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to: {output_path}")

    # Justification analysis
    print("\nExtracting justifications for non-monotonic pairs...")
    justification_report = analyze_justifications(
        variation_results, results_dir, args.model, manifest
    )
    just_path = results_dir / f"phase6b_justification_analysis_{args.model}.json"
    with open(just_path, "w", encoding="utf-8") as f:
        json.dump(justification_report, f, indent=2, ensure_ascii=False)
    print(f"Justification analysis saved to: {just_path}")

    if justification_report["violation_by_transition"]:
        print("\nMost common violation transitions:")
        for trans, count in list(justification_report["violation_by_transition"].items())[:5]:
            print(f"  {trans}: {count}")

    # Per-category breakdown
    cat_dir = output_path.parent / f"phase6b_by_category_{args.model}"
    cat_dir.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[dict]] = {}
    for vr in variation_results:
        by_cat.setdefault(vr["category"], []).append(vr)

    for cat, cat_vrs in sorted(by_cat.items()):
        safe_cat = cat.replace(" ", "_").replace("/", "_")
        cat_agg = aggregate_results(cat_vrs)
        cat_json_path = cat_dir / f"{safe_cat}.json"
        cat_output = {
            "model": args.model,
            "category": cat,
            "aggregate": cat_agg["overall"],
            "per_variation_set": [
                {k: v for k, v in vr.items() if k != "per_comparison"}
                for vr in cat_vrs
            ],
        }
        with open(cat_json_path, "w", encoding="utf-8") as f:
            json.dump(cat_output, f, indent=2, ensure_ascii=False)

    print(f"Per-category results saved to: {cat_dir}/ ({len(by_cat)} categories)")


if __name__ == "__main__":
    main()
