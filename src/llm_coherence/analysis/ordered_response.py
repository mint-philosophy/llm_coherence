"""Coherent utilities with an explicit, model-wide presentation-position effect.

Probabilities always refer to the tier outcome, even when it is displayed as
Option B. The two logits are u_tier - u_reference + delta and the same utility
difference minus delta. A positive delta favors displayed Option A. This is a
working response model, not a claim that order or repeat effects are identified
without model assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import betabinom, binom

from .coherent_baseline import (
    _tied_parameterization, _utilities, increasing_projection, replicate_summary,
)


@dataclass
class OrderedResponseFit:
    ladder_utilities: np.ndarray
    reference_utilities: np.ndarray
    position_offset: float
    probabilities: np.ndarray
    diagnostics: dict


def order_objective(parameters, successes, totals, ridge):
    """Normalized binomial working loss, with an exact analytic gradient."""
    k, n = np.asarray(successes, float), np.asarray(totals, float)
    ladders, references, tiers, _ = k.shape
    u, v = _utilities(parameters[:-1], ladders, references, tiers)
    delta = parameters[-1]
    eta = (u[:, None, :, None] - v[None, :, None, None] +
           delta * np.array([1., -1.]))
    normalization = n.sum()
    if normalization <= 0:
        raise ValueError("Calibration requires observed choices")
    value = np.sum(n * np.logaddexp(0, eta) - k * eta) / normalization
    value += ridge * (np.mean(u ** 2) + np.mean(v ** 2) + delta ** 2)
    residual = (n * expit(eta) - k) / normalization
    grad_u = residual.sum(axis=(1, 3)) + 2 * ridge * u / u.size
    grad_v = -residual.sum(axis=(0, 2, 3)) + 2 * ridge * v / v.size
    grad_parts = np.empty((ladders, tiers))
    grad_parts[:, 0] = grad_u.sum(axis=1)
    grad_parts[:, 1:] = np.flip(np.cumsum(np.flip(grad_u[:, 1:], axis=1), axis=1), axis=1)
    grad_delta = residual[..., 0].sum() - residual[..., 1].sum() + 2 * ridge * delta
    return float(value), np.concatenate([grad_parts.ravel(), grad_v[1:], [grad_delta]])


def fit_ordered_responses(successes, totals, *, position_effect=True, outcome_ids=None,
                          ridge=1e-5, max_iterations=2000):
    """Jointly fit ordered shared utilities and, optionally, one position offset.

Setting position_effect=False fixes delta to zero. The utility penalty and
training data are identical in both fits. The first reference anchors location;
the logistic scale is fixed to one. Identical tier texts share exactly one
utility through the same equality constraints as the original simulation.
"""
    k, n = np.asarray(successes, float), np.asarray(totals, float)
    if k.shape != n.shape or k.ndim != 4 or k.shape[-2:] != (7, 2):
        raise ValueError("Expected matching ladder x reference x seven-tier x two-order counts")
    if not np.isfinite(k).all() or not np.isfinite(n).all() or np.any(k < 0) or np.any(n < k):
        raise ValueError("Invalid observed choice counts")
    if np.any(n.sum(axis=(1, 2, 3)) == 0) or np.any(n.sum(axis=(0, 2, 3)) == 0):
        raise ValueError("Every fitted ladder and reference needs observations")
    if np.any(n.sum(axis=(0, 1, 2)) == 0):
        raise ValueError("Both presentation orders need observations")
    if not np.isfinite(ridge) or ridge <= 0 or max_iterations < 1:
        raise ValueError("A positive regularization and iteration limit are required")
    ladders, references, tiers, _ = k.shape
    marginal = (k.sum(axis=(1, 3)) + .5) / (n.sum(axis=(1, 3)) + 1)
    initial_u = increasing_projection(logit(marginal))
    parts = np.concatenate([initial_u[:, :1], np.maximum(np.diff(initial_u, axis=1), 0)], axis=1)
    transform, duplicates = None, []
    initial = np.concatenate([parts.ravel(), np.zeros(references - 1)])
    bounds = [(None, None) if i % tiers == 0 else (0, None)
              for i in range(ladders * tiers)] + [(None, None)] * (references - 1)
    if outcome_ids is not None:
        transform, initial, bounds, duplicates = _tied_parameterization(outcome_ids, parts, references)
    initial = np.concatenate([initial, [0.]])
    bounds = bounds + ([(None, None)] if position_effect else [(0., 0.)])

    def objective(parameters):
        original = parameters[:-1] if transform is None else transform @ parameters[:-1]
        value, gradient = order_objective(np.concatenate([original, parameters[-1:]]), k, n, ridge)
        if transform is not None:
            gradient = np.concatenate([np.asarray(transform.T @ gradient[:-1]), gradient[-1:]])
        return value, gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-7})
    original = result.x[:-1] if transform is None else transform @ result.x[:-1]
    u, v = _utilities(original, ladders, references, tiers)
    delta = float(result.x[-1])
    p = expit(u[:, None, :, None] - v[None, :, None, None] + delta * np.array([1., -1.]))
    diagnostics = {
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "iterations": int(result.nit), "penalized_mean_negative_log_likelihood": float(result.fun),
        "ridge": ridge, "link": "logistic", "reference_0_utility": 0.,
        "position_effect": bool(position_effect), "position_offset": delta,
        "maximum_probability_order_violation": float(np.maximum(-np.diff(p, axis=2), 0).max()),
        "shared_outcome_groups": len(duplicates),
        "maximum_shared_outcome_utility_difference": max(
            (float(np.ptp([u[li, ti] for li, ti in group])) for group in duplicates), default=0.),
    }
    return OrderedResponseFit(u, v, delta, p, diagnostics)


def order_block_pmf(probabilities, trials_per_order, repeat_correlation):
    """Count distribution per order; beta-binomial correlation is a scenario."""
    p = np.asarray(probabilities, float)
    n, rho = trials_per_order, float(repeat_correlation)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 <= rho <= 1:
        raise ValueError("Invalid trial count or repeat-correlation scenario")
    if rho == 0:
        return binom.pmf(np.arange(n + 1), n, p[..., None])
    answer = np.zeros(p.shape + (n + 1,))
    if rho == 1:
        answer[..., 0], answer[..., -1] = 1 - p, p
    else:
        interior = (p > 0) & (p < 1)
        concentration = 1 / rho - 1
        answer[interior] = betabinom.pmf(
            np.arange(n + 1), n, p[interior, None] * concentration,
            (1 - p[interior, None]) * concentration)
        answer[..., 0][p == 0] = 1
        answer[..., -1][p == 1] = 1
    return answer


def draw_order_counts(probabilities, trials_per_order, *, repeat_correlation, rng):
    """Sample each presentation-order block independently; keep their counts."""
    p = np.asarray(probabilities, float)
    n, rho = trials_per_order, float(repeat_correlation)
    if p.ndim < 1 or p.shape[-1] != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite probabilities for both presentation orders")
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 <= rho <= 1:
        raise ValueError("Invalid trial count or repeat-correlation scenario")
    if rho == 0:
        return rng.binomial(n, p)
    if rho == 1:
        return n * rng.binomial(1, p)
    latent = p.copy()
    interior = (p > 0) & (p < 1)
    concentration = 1 / rho - 1
    latent[interior] = rng.beta(p[interior] * concentration, (1 - p[interior]) * concentration)
    return rng.binomial(n, latent)


def monotonicity_probabilities(probabilities, trials_per_order, repeat_correlation):
    """Exact probability that seven pooled counts are nondecreasing (ties pass).

Order blocks and different cells are independent under this simulation. Each
order block can still have correlated repeats. Convolution gives the pooled
count PMF; a cumulative-sum recursion sums all nondecreasing count sequences.
"""
    p = np.asarray(probabilities, float)
    if p.ndim < 2 or p.shape[-2:] != (7, 2):
        raise ValueError("Expected seven tiers and two presentation orders")
    pmf = order_block_pmf(p, trials_per_order, repeat_correlation)
    n = trials_per_order
    pooled = np.zeros(p.shape[:-1] + (2 * n + 1,))
    for k in range(n + 1):
        pooled[..., k:k + n + 1] += pmf[..., 0, k, None] * pmf[..., 1, :]
    cumulative = pooled[..., 0, :]
    for tier in range(1, 7):
        cumulative = pooled[..., tier, :] * np.cumsum(cumulative, axis=-1)
    return np.clip(cumulative.sum(axis=-1), 0, 1)


def simulate_order_monotonicity(probabilities, eligible, trials_per_order, *,
                                repeat_correlation=0., repetitions=500, seed=42):
    p, eligible = np.asarray(probabilities, float), np.asarray(eligible, bool)
    if p.shape[:-2] != eligible.shape or not eligible.any() or repetitions < 2:
        raise ValueError("Simulation requires an eligible curve mask and at least two replicates")
    curve_pass = monotonicity_probabilities(p[eligible], trials_per_order, repeat_correlation)
    # The score uses only each curve's pass/fail outcome. Sampling these exact
    # Bernoulli probabilities has the same score distribution as full trials.
    rng = np.random.default_rng(seed)
    rates = rng.binomial(1, curve_pass, size=(repetitions, len(curve_pass))).mean(axis=1)
    return {**replicate_summary(rates), "exact_expected_rate": float(curve_pass.mean()),
            "curves": len(curve_pass), "score_sampling": "exact_curve_probability_then_bernoulli"}
