"""Synthetic ordered-utility references, not empirical coherence ceilings.

The calibration is a constrained Bradley--Terry model. Each ladder has seven
nondecreasing outcome utilities; the reference utilities are shared across
ladders. The logistic noise scale is fixed to one for identification. A weak
quadratic penalty keeps saturated estimates finite. Simulations are conditional
on these fitted utilities; their intervals do not include calibration uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.special import expit, logit


def increasing_projection(values: np.ndarray) -> np.ndarray:
    """Equal-weight isotonic projection along the final axis, in batches.

    Uses the min-max characterization of isotonic least squares. This avoids
    constructing a separate estimator for each seven-point permutation.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim < 1 or y.shape[-1] < 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite curves with at least two points")
    t = y.shape[-1]
    prefix = np.concatenate([np.zeros(y.shape[:-1] + (1,)), np.cumsum(y, axis=-1)], axis=-1)
    lengths = np.arange(t)[None, :] - np.arange(t)[:, None] + 1
    segment_means = np.zeros(y.shape[:-1] + (t, t))
    np.divide(prefix[..., None, 1:] - prefix[..., :-1, None], lengths,
              out=segment_means, where=lengths > 0)
    return np.stack([
        segment_means[..., :i + 1, i:].min(axis=-1).max(axis=-1)
        for i in range(t)
    ], axis=-1)


def isotonic_scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Increasing and bidirectional R2; exactly constant curves return NaN."""
    y = np.asarray(values, dtype=float)
    inc = increasing_projection(y)
    dec = -increasing_projection(-y)
    total = np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    constant = np.all(y == y[..., :1], axis=-1)
    inc_r2 = np.full(total.shape, np.nan)
    dec_r2 = np.full(total.shape, np.nan)
    np.divide(np.sum((y - inc) ** 2, axis=-1), total, out=inc_r2, where=~constant)
    np.divide(np.sum((y - dec) ** 2, axis=-1), total, out=dec_r2, where=~constant)
    inc_r2 = np.clip(1 - inc_r2, 0, 1)
    dec_r2 = np.clip(1 - dec_r2, 0, 1)
    return inc_r2, np.maximum(inc_r2, dec_r2)


@dataclass
class OrderedUtilityFit:
    ladder_utilities: np.ndarray
    reference_utilities: np.ndarray
    probabilities: np.ndarray
    diagnostics: dict


def _utilities(parameters: np.ndarray, ladders: int, references: int, tiers: int):
    pieces = parameters[:ladders * tiers].reshape(ladders, tiers)
    utilities = pieces[:, :1] + np.concatenate(
        [np.zeros((ladders, 1)), np.cumsum(pieces[:, 1:], axis=1)], axis=1)
    refs = np.concatenate([[0.0], parameters[ladders * tiers:]])
    if len(refs) != references:
        raise ValueError("Incorrect reference parameter count")
    return utilities, refs


def ordered_objective(parameters, successes, totals, ridge):
    """Mean binomial negative log-likelihood and its analytic gradient."""
    k, n = np.asarray(successes, float), np.asarray(totals, float)
    ladders, references, tiers = k.shape
    u, v = _utilities(parameters, ladders, references, tiers)
    eta = u[:, None, :] - v[None, :, None]
    normalization = n.sum()
    if normalization <= 0:
        raise ValueError("Calibration requires observed choices")
    objective = np.sum(n * np.logaddexp(0, eta) - k * eta) / normalization
    objective += ridge * (np.mean(u ** 2) + np.mean(v ** 2))
    residual = (n * expit(eta) - k) / normalization
    grad_u = residual.sum(axis=1) + 2 * ridge * u / u.size
    grad_v = -residual.sum(axis=(0, 2)) + 2 * ridge * v / v.size
    grad_parts = np.empty((ladders, tiers))
    grad_parts[:, 0] = grad_u.sum(axis=1)
    grad_parts[:, 1:] = np.flip(np.cumsum(np.flip(grad_u[:, 1:], axis=1), axis=1), axis=1)
    return float(objective), np.concatenate([grad_parts.ravel(), grad_v[1:]])


def _tied_parameterization(outcome_ids, initial_parts, references):
    """Eliminate intercepts to share utilities for identical outcome text.

    The current instrument's duplicate links form a forest of ladders. Along
    each link, solving for the child ladder's intercept preserves all six
    nonnegative tier increments. This is an exact linear reparameterization,
    so the bounded likelihood problem remains convex. Future instruments with
    nonredundant cycles need a general linearly constrained optimizer; reject
    them explicitly instead of silently assigning different utilities.
    """
    ladders, tiers = initial_parts.shape
    ids = np.asarray(outcome_ids)
    if ids.shape != (ladders, tiers):
        raise ValueError("Outcome identities must match the ladder/tier grid")
    groups = {}
    for li in range(ladders):
        for ti in range(tiers):
            groups.setdefault(str(ids[li, ti]), []).append((li, ti))
    duplicates = [positions for positions in groups.values() if len(positions) > 1]
    parents = list(range(ladders))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    neighbors = [[] for _ in range(ladders)]
    zero_increments = set()
    for positions in duplicates:
        li, ti = positions[0]
        for lj, tj in positions[1:]:
            if li == lj:
                zero_increments.update((li, t) for t in range(min(ti, tj) + 1, max(ti, tj) + 1))
                continue
            if root(li) == root(lj):
                raise ValueError("Duplicate-outcome equality cycle requires a general constrained fit")
            parents[root(lj)] = root(li)
            neighbors[li].append((lj, ti, tj))
            neighbors[lj].append((li, tj, ti))
    roots, visited = [], set()
    for li in range(ladders):
        if li not in visited:
            roots.append(li)
            stack = [li]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    stack.extend(j for j, _, _ in neighbors[current])
    components = len(roots)
    dimension = components + ladders * (tiers - 1) + references - 1
    transform = lil_matrix((ladders * tiers + references - 1, dimension))

    def delta_column(li, ti):
        return components + li * (tiers - 1) + ti - 1

    intercept_rows = {}
    for component, first in enumerate(roots):
        intercept_rows[first] = {component: 1.0}
        stack = [first]
        while stack:
            current = stack.pop()
            for child, tier_current, tier_child in neighbors[current]:
                if child in intercept_rows:
                    continue
                row = intercept_rows[current].copy()
                for ladder, tier, sign in ((current, tier_current, 1), (child, tier_child, -1)):
                    for t in range(1, tier + 1):
                        col = delta_column(ladder, t)
                        row[col] = row.get(col, 0) + sign
                intercept_rows[child] = row
                stack.append(child)
    for li in range(ladders):
        for col, value in intercept_rows[li].items():
            transform[li * tiers, col] = value
        for ti in range(1, tiers):
            transform[li * tiers + ti, delta_column(li, ti)] = 1
    for ri in range(references - 1):
        transform[ladders * tiers + ri, components + ladders * (tiers - 1) + ri] = 1
    initial = np.concatenate([initial_parts[roots, 0], initial_parts[:, 1:].ravel(), np.zeros(references - 1)])
    bounds = [(None, None)] * components
    for li in range(ladders):
        for ti in range(1, tiers):
            fixed = (li, ti) in zero_increments
            bounds.append((0, 0 if fixed else None))
            if fixed:
                initial[delta_column(li, ti)] = 0
    bounds.extend([(None, None)] * (references - 1))
    return transform.tocsr(), initial, bounds, duplicates


def fit_ordered_utilities(successes, totals, *, ridge=1e-5, max_iterations=2000, outcome_ids=None):
    k, n = np.asarray(successes, float), np.asarray(totals, float)
    if k.shape != n.shape or k.ndim != 3 or k.shape[-1] != 7:
        raise ValueError("Expected matching ladder x reference x seven-tier arrays")
    if not np.isfinite(k).all() or not np.isfinite(n).all() or np.any(k < 0) or np.any(n < k):
        raise ValueError("Invalid observed choice counts")
    if np.any(n.sum(axis=(1, 2)) == 0) or np.any(n.sum(axis=(0, 2)) == 0):
        raise ValueError("Every fitted ladder and reference needs observations")
    if ridge <= 0 or max_iterations < 1:
        raise ValueError("A positive regularization and iteration limit are required")
    ladders, references, tiers = k.shape
    marginal = (k.sum(axis=1) + 0.5) / (n.sum(axis=1) + 1)
    initial_u = increasing_projection(logit(marginal))
    parts = np.concatenate([initial_u[:, :1], np.maximum(np.diff(initial_u, axis=1), 0)], axis=1)
    initial = np.concatenate([parts.ravel(), np.zeros(references - 1)])
    bounds = [(None, None) if i % tiers == 0 else (0, None)
              for i in range(ladders * tiers)] + [(None, None)] * (references - 1)
    transform, duplicates = None, []
    if outcome_ids is not None:
        transform, initial, bounds, duplicates = _tied_parameterization(outcome_ids, parts, references)

    def objective(parameters):
        original = parameters if transform is None else transform @ parameters
        value, gradient = ordered_objective(original, k, n, ridge)
        return value, gradient if transform is None else np.asarray(transform.T @ gradient)

    result = minimize(objective, initial, jac=True,
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-7})
    parameters = result.x if transform is None else transform @ result.x
    u, v = _utilities(parameters, ladders, references, tiers)
    probabilities = expit(u[:, None, :] - v[None, :, None])
    observed = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    selected = n > 0
    bins = np.linspace(0, 1, 11)
    diagnostics = {
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "penalized_mean_negative_log_likelihood": float(result.fun),
        "ridge": ridge, "link": "logistic", "reference_0_utility": 0,
        "weighted_probability_rmse": float(np.sqrt(np.sum(n * (observed - probabilities) ** 2) / n.sum())),
        "observed_mean_probability": float(np.mean(observed[selected])),
        "fitted_mean_probability": float(np.mean(probabilities[selected])),
        "histogram_edges": bins.tolist(),
        "observed_probability_histogram": np.histogram(observed[selected], bins)[0].tolist(),
        "fitted_probability_histogram": np.histogram(probabilities[selected], bins)[0].tolist(),
        "maximum_probability_order_violation": float(np.maximum(-np.diff(probabilities, axis=-1), 0).max()),
        "shared_outcome_groups": len(duplicates),
        "maximum_shared_outcome_utility_difference": max(
            (float(np.ptp([u[li, ti] for li, ti in group])) for group in duplicates), default=0.0),
    }
    return OrderedUtilityFit(u, v, probabilities, diagnostics)


def draw_protocol_counts(probabilities, trials_per_direction, *, repeat_correlation, rng):
    """Two counterbalanced groups; equal true preference probability in each.

    Correlation 0 gives independent trials. Correlation 1 makes repeats within
    each presentation order identical, yielding fractions 0, 0.5 or 1. Other
    correlations use a beta-binomial mixture per presentation direction.
    This is a sensitivity model; the correlation is not inferred from aggregates.
    """
    p = np.asarray(probabilities, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    if not isinstance(trials_per_direction, (int, np.integer)) or trials_per_direction < 1:
        raise ValueError("A positive integer trial count is required")
    rho = float(repeat_correlation)
    if not 0 <= rho <= 1:
        raise ValueError("Repeat correlation must be in [0, 1]")
    answer = np.zeros(p.shape, dtype=int)
    for _ in range(2):
        if rho == 0:
            answer += rng.binomial(trials_per_direction, p)
        elif rho == 1:
            answer += trials_per_direction * rng.binomial(1, p)
        else:
            concentration = 1 / rho - 1
            latent = p.copy()
            interior = (p > 0) & (p < 1)
            latent[interior] = rng.beta(p[interior] * concentration,
                                        (1 - p[interior]) * concentration)
            answer += rng.binomial(trials_per_direction, latent)
    return answer


def replicate_summary(values):
    a = np.asarray(values, float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("At least two finite replicate statistics are required")
    return {"mean": float(a.mean()), "replicate_q025": float(np.quantile(a, .025)),
            "replicate_q975": float(np.quantile(a, .975)),
            "mean_monte_carlo_se": float(a.std(ddof=1) / np.sqrt(len(a))),
            "replicates": len(a)}


def simulate_monotonicity(probabilities, eligible, trials_per_direction, *,
                          repetitions=500, repeat_correlation=0, seed=42):
    p, eligible = np.asarray(probabilities, float), np.asarray(eligible, bool)
    if p.shape[:-1] != eligible.shape or not eligible.any():
        raise ValueError("Simulation requires an eligible curve mask")
    rng = np.random.default_rng(seed)
    selected = p[eligible]
    rates = []
    for _ in range(repetitions):
        counts = draw_protocol_counts(selected, trials_per_direction,
                                      repeat_correlation=repeat_correlation, rng=rng)
        rates.append(float(np.all(np.diff(counts, axis=-1) >= 0, axis=-1).mean()))
    return replicate_summary(rates)
