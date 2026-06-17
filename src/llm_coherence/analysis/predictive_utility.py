"""
Predictive-utility test on phase6b result files.

For each (variation_set, model), estimate how well a simple tier/comparison
logistic model predicts held-out preferences.

Procedure (per set, per model):
    1. Load pair-level rows (tier, comparison_block) with binomial counts
       (n_success, n_trial). There are 7 tiers x 30 comparisons = 210 rows.
    2. Split train/test at the 210-row pair level (80/20, controlled by
       `--split-seed`, default 0) to prevent pair spill between train and test.
    3. Expand each split to Bernoulli trials, then fit logistic regression:
       P = invlogit(alpha + beta*tier_centered + one_hot(comparison_block)).
    4. Score held-out data with AUC and log-loss.
    5. Build permutation null by shuffling tier labels at pair level, refitting,
       and rescoring (`--n-perm`, default 200). The SAME train/test split as
       the observed statistic is reused for every permutation so the null
       isolates the contribution of the tier-outcome association.
    6. Compute per-set p-values vs null, then apply Benjamini-Hochberg FDR
       correction within each model (alpha=0.05).

Outputs:
    - Per-set CSV with observed metrics, null metrics, raw p-values, BH-adjusted
      p-values, and explicit reject/fail decisions.
    - Per-model summary CSV with mean metrics and both raw and BH significance rates.

Default output location:
    - If `--model` is provided and `--out-dir` is omitted:
        results/<model>/pred_utility_test/

Usage:
    PYTHONPATH=src python -m llm_coherence.analysis.predictive_utility --model ministral-3b-2512-openrouter
"""

import argparse
import glob
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from llm_coherence.config import resolve_model_results_dir
from llm_coherence.paths import LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR, REPO_ROOT

BASE = Path(__file__).resolve().parent
PARAMETRIC_ROOT = REPO_ROOT
N_TIERS = 7
N_COMPS_PER_SET = 30
TEST_FRAC = 0.20
N_PERMUTATIONS = 200
BH_ALPHA = 0.05

PER_SET_COLUMNS = [
    "model_key",
    "label",
    "variation_id",
    "train_auc",
    "test_auc",
    "test_logloss",
    "n_train",
    "n_test",
    "n_train_pairs",
    "n_test_pairs",
    "null_n",
    "null_auc_mean",
    "null_auc_sd",
    "null_auc_p95",
    "null_logloss_mean",
    "p_value_vs_null",
    "alpha_raw_p05",
    "alpha_bh_fdr",
    "sig_raw_p05",
    "decision_raw_p05",
    "p_value_bh",
    "sig_bh_fdr05",
    "decision_bh_fdr05",
]

PER_MODEL_COLUMNS = [
    "model_key",
    "label",
    "n_sets",
    "mean_test_auc",
    "median_test_auc",
    "mean_null_auc",
    "mean_test_logloss",
    "mean_p_value",
    "mean_p_value_bh",
    "frac_sig_at_p05",
    "frac_sig_at_bh_fdr05",
]


def canonical_variation_id(test_name: str) -> str:
    """Normalize test_name to variation_id token used by clean-set filters."""
    for prefix in ("phase6b_var_", "phase6b_variations_pruned_final_"):
        if test_name.startswith(prefix):
            return test_name[len(prefix):]
    return test_name


def resolve_under_parametric(rel: str | Path) -> Path:
    """Resolve a path under parametric_variations/ unless absolute."""
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (PARAMETRIC_ROOT / p).resolve()


def collect_result_files(results_dir: Path, model_key: str) -> list[Path]:
    """Per-set results under outputs/<model>/ladder_vs_comparison_statements/."""
    files: list[Path] = []

    run_dir = resolve_model_results_dir(model_key, results_dir) / "ladder_vs_comparison_statements"
    if run_dir.is_dir():
        files.extend(sorted(run_dir.glob("phase6b_ladder_*/results.json")))
        files.extend(sorted(run_dir.glob("phase6b_variations_prune_*/results.json")))

    # Legacy flat layout: <root>/<model>/<artifact_dir>/results.json
    model_root = results_dir / model_key
    if model_root.is_dir():
        files.extend(sorted(model_root.glob("phase6b_ladder_*/results.json")))
        files.extend(sorted(model_root.glob("phase6b_variations_prune_*/results.json")))

    # Legacy layout: results/phase6b_var_*/phase6b_var_*_<model>_results.json
    legacy = glob.glob(
        str(results_dir / "phase6b_var_*" / f"phase6b_var_*_{model_key}_results.json")
    )
    files.extend(Path(p) for p in sorted(legacy))

    seen = set()
    uniq: list[Path] = []
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq



def load_set_records(json_path: Path) -> pd.DataFrame:
    """One row per (tier, comparison_block) for a single (set, model). Columns:
    tier (int 1..7), comp_idx (int 0..29), n_success, n_trial."""
    with open(json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    prefs = d["preferences"]
    if len(prefs) % N_TIERS != 0:
        raise ValueError(f"{json_path}: prefs length {len(prefs)} not multiple of {N_TIERS}")
    rows = []
    n_blocks = len(prefs) // N_TIERS
    for block_idx in range(n_blocks):
        block = prefs[block_idx * N_TIERS:(block_idx + 1) * N_TIERS]
        for tier_idx, p in enumerate(block, 1):
            n_success = p["count_prefer_a"]
            n_trial = p["count_prefer_a"] + p["count_prefer_b"]
            if n_trial == 0:
                continue
            rows.append({
                "tier": tier_idx,
                "comp_idx": block_idx,
                "n_success": n_success,
                "n_trial": n_trial,
            })
    return pd.DataFrame(rows)


def expand_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Expand binomial counts to one row per trial. Returns columns:
    tier, comp_idx, y (0/1)."""
    rows = []
    for _, r in df.iterrows():
        for _ in range(r["n_success"]):
            rows.append({"tier": r["tier"], "comp_idx": r["comp_idx"], "y": 1})
        for _ in range(r["n_trial"] - r["n_success"]):
            rows.append({"tier": r["tier"], "comp_idx": r["comp_idx"], "y": 0})
    return pd.DataFrame(rows)


def make_features(df: pd.DataFrame, comp_encoder=None):
    """Build feature matrix: tier (numeric, centered) + one-hot comp_idx."""
    tier_c = (df["tier"].to_numpy() - 4).reshape(-1, 1).astype(float)
    if comp_encoder is None:
        comp_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        comp_encoder.fit(df[["comp_idx"]])
    comp_feat = comp_encoder.transform(df[["comp_idx"]])
    X = np.hstack([tier_c, comp_feat])
    return X, comp_encoder


def cv_score(df_pairs: pd.DataFrame, seed: int = 0) -> dict | None:
    """Single 80/20 split + logistic fit with pair-level split first.

    IMPORTANT: train/test partitioning is done on the original pair rows
    (tier, comparison_block), then each split is expanded to binary trials.
    This prevents train/test spill where the same pair appears in both splits.
    """
    rng = np.random.default_rng(seed)
    n = len(df_pairs)
    if n < 30:
        return None
    test_idx = rng.choice(n, size=max(20, int(n * TEST_FRAC)), replace=False)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True

    # Split at pair level first to avoid pair leakage across train/test.
    train_pairs = df_pairs.loc[~test_mask].copy()
    test_pairs = df_pairs.loc[test_mask].copy()

    # Expand each split to Bernoulli trials for logistic fitting.
    train = expand_to_binary(train_pairs)
    test = expand_to_binary(test_pairs)

    if train["y"].nunique() < 2 or test["y"].nunique() < 2:
        return None
    X_train, enc = make_features(train)
    X_test, _ = make_features(test, comp_encoder=enc)
    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf = LogisticRegression(max_iter=500, C=1.0, solver="liblinear")
            clf.fit(X_train, y_train)
        except (ValueError, RuntimeError):
            return None
    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    try:
        return {
            "train_auc": float(roc_auc_score(y_train, p_train)),
            "test_auc": float(roc_auc_score(y_test, p_test)),
            "test_logloss": float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6))),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_train_pairs": int(len(train_pairs)),
            "n_test_pairs": int(len(test_pairs)),
        }
    except ValueError:
        return None


def permutation_null(df_pairs: pd.DataFrame, n_perm: int = 200,
                    base_seed: int = 0, split_seed: int = 0) -> dict:
    """Build null distribution by shuffling tier labels.

    Uses a FIXED train/test split (controlled by `split_seed`) across all
    permutations so the only source of variation between observed and null
    statistics is the tier-label shuffle. This preserves the exchangeability
    that makes the permutation p-value valid.
    """
    rng = np.random.default_rng(base_seed)
    null_aucs = []
    null_lls = []
    for _ in range(n_perm):
        df_perm = df_pairs.copy()
        # Shuffle tier labels within the set (breaks tier-outcome relationship)
        df_perm["tier"] = rng.permutation(df_perm["tier"].to_numpy())
        # IMPORTANT: use the SAME split seed as the observed statistic.
        score = cv_score(df_perm, seed=split_seed)
        if score is None:
            continue
        null_aucs.append(score["test_auc"])
        null_lls.append(score["test_logloss"])
    if not null_aucs:
        return {"null_n": 0, "null_auc_mean": float("nan"),
                "null_auc_p95": float("nan"), "null_logloss_mean": float("nan")}
    return {
        "null_n": len(null_aucs),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_sd": float(np.std(null_aucs)),
        "null_auc_p95": float(np.quantile(null_aucs, 0.95)),
        "null_logloss_mean": float(np.mean(null_lls)),
        "null_aucs": null_aucs,
    }


def run_one_set_model(set_path: Path, n_perm: int = 200,
                       split_seed: int = 0) -> dict | None:
    """Per (set, model) predictive-utility test.

    Both the observed AUC and every permutation AUC are computed on the SAME
    train/test split (controlled by `split_seed`), so the permutation p-value
    isolates the contribution of the tier-outcome association.
    """
    df = load_set_records(set_path)
    if len(df) < 30:
        return None
    obs = cv_score(df, seed=split_seed)
    if obs is None:
        return None
    null = permutation_null(df, n_perm=n_perm, split_seed=split_seed)
    null_aucs = null.pop("null_aucs", [])
    p_value = (1 + sum(1 for a in null_aucs if a >= obs["test_auc"])) / (1 + len(null_aucs)) if null_aucs else float("nan")
    out = {**obs, **null, "p_value_vs_null": float(p_value)}
    return out


def apply_bh_by_model(df: pd.DataFrame, alpha: float = BH_ALPHA) -> pd.DataFrame:
    """Add BH-adjusted p-values and significance flags per model_key."""
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=PER_SET_COLUMNS)

    out["alpha_raw_p05"] = 0.05
    out["alpha_bh_fdr"] = float(alpha)
    out["sig_raw_p05"] = out["p_value_vs_null"].astype(float) < 0.05
    out["decision_raw_p05"] = np.where(
        out["sig_raw_p05"], "reject_h0", "fail_to_reject_h0"
    )
    out["p_value_bh"] = np.nan
    out["sig_bh_fdr05"] = False
    out["decision_bh_fdr05"] = "fail_to_reject_h0"

    for model_key, idx in out.groupby("model_key").groups.items():
        p = out.loc[idx, "p_value_vs_null"].astype(float)
        valid_mask = p.notna()
        if valid_mask.sum() == 0:
            continue

        valid_idx = p[valid_mask].index
        pvals = p.loc[valid_idx].to_numpy()
        m = len(pvals)

        order = np.argsort(pvals)
        sorted_p = pvals[order]
        ranks = np.arange(1, m + 1, dtype=float)

        # BH adjusted p-values (q-values), monotone from high->low rank.
        q_sorted = (sorted_p * m) / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0.0, 1.0)

        # BH decisions at chosen FDR alpha.
        thresh = (ranks / m) * alpha
        passed = sorted_p <= thresh
        sig_sorted = np.zeros(m, dtype=bool)
        if np.any(passed):
            k = np.max(np.where(passed)[0])
            sig_sorted[: k + 1] = True

        # Map sorted arrays back to original order.
        inv = np.empty(m, dtype=int)
        inv[order] = np.arange(m)
        q_unsorted = q_sorted[inv]
        sig_unsorted = sig_sorted[inv]

        out.loc[valid_idx, "p_value_bh"] = q_unsorted
        out.loc[valid_idx, "sig_bh_fdr05"] = sig_unsorted
        out.loc[valid_idx, "decision_bh_fdr05"] = np.where(
            sig_unsorted, "reject_h0", "fail_to_reject_h0"
        )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default=str(LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If omitted and --model is set, defaults to "
             "outputs/<model>/ladder_vs_comparison_statements/pred_utility_test",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Model key to analyze. Reads inputs from results/<model>/.",
    )
    ap.add_argument("--n-perm", type=int, default=N_PERMUTATIONS)
    ap.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for the 80/20 pair-level train/test split. The SAME split "
             "is used for both the observed AUC and every permutation, so the "
             "permutation null isolates the contribution of tier labels.",
    )
    args = ap.parse_args()

    results_dir = resolve_under_parametric(args.results_dir)
    if args.out_dir:
        out_dir = resolve_under_parametric(args.out_dir)
    else:
        out_dir = (
            results_dir
            / args.model
            / "ladder_vs_comparison_statements"
            / "pred_utility_test"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    model_variants = [(args.model, args.model)]
    print(f"Output directory: {out_dir}")
    for model_key, label in model_variants:
        files = collect_result_files(results_dir, model_key)
        print(f"\n{label} ({model_key}): {len(files)} sets")
        for i, f in enumerate(files):
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
                test_name = (d.get("config") or {}).get("test_name", Path(f).parent.name)
            except Exception:
                test_name = Path(f).parent.name
            variation_id = canonical_variation_id(test_name)
            res = run_one_set_model(Path(f), n_perm=args.n_perm,
                                     split_seed=args.split_seed)
            if res is None:
                continue
            rows.append({
                "model_key": model_key,
                "label": label,
                "variation_id": variation_id,
                **res,
            })
            if (i + 1) % 50 == 0:
                print(f"  ...processed {i + 1}/{len(files)}")

    df = pd.DataFrame(rows)
    df = apply_bh_by_model(df, alpha=BH_ALPHA)
    per_set_path = out_dir / "per_set_pred_util.csv"
    df.to_csv(per_set_path, index=False)
    print(f"\nPer-set predictive-utility results saved: {per_set_path}")

    # Aggregate per model variant
    if df.empty:
        summary = pd.DataFrame(columns=PER_MODEL_COLUMNS)
        print(
            "No predictive-utility rows were generated. This usually means the "
            "input is a tiny smoke slice with fewer than 30 tier/comparison "
            "pairs or no valid train/test split."
        )
    else:
        summary = (
            df.groupby(["model_key", "label"], as_index=False)
            .agg(
                n_sets=("test_auc", "size"),
                mean_test_auc=("test_auc", "mean"),
                median_test_auc=("test_auc", "median"),
                mean_null_auc=("null_auc_mean", "mean"),
                mean_test_logloss=("test_logloss", "mean"),
                mean_p_value=("p_value_vs_null", "mean"),
                mean_p_value_bh=("p_value_bh", "mean"),
                frac_sig_at_p05=("p_value_vs_null", lambda s: float((s < 0.05).mean())),
                frac_sig_at_bh_fdr05=("sig_bh_fdr05", lambda s: float(pd.Series(s).astype(bool).mean())),
            )
        )
    per_model_path = out_dir / "per_model_pred_util.csv"
    summary.to_csv(per_model_path, index=False)
    print(f"Per-model summary saved: {per_model_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
