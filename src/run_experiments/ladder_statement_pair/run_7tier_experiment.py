#!/usr/bin/env python3
"""
Run forced-choice monotonicity experiments on 7-tier variation sets.

Reads a phase 6b manifest (pruned ladders by default) and comparison JSONs from
``<parametric_variations>/data/...``.

Usage (from this directory):

    # Non reasoning model
    for s in 0 10 20 30 40 50 60 70 80 90; do
        python phase6b_experiments/run_7tier_experiment.py \
            --model ministral-3b-2512-openrouter \
            --trials 10 \
            --start-from $s \
            --max-variation-sets 10 \
            --max-concurrent 6 \
            --resume
    done

    # Reasoning model

    for s in 0 10 20 30 40 50 60 70 80 90; do
        python phase6b_experiments/run_7tier_experiment.py \
            --model mistral-small-2603-openrouter-thinking \
            --trials 10 \
            --start-from $s \
            --max-variation-sets 10 \
            --max-concurrent 6 \
            --with-reasoning \
            --reasoning-mode thinking \
            --resume
    done

    # Level 3: with chain-of-thought justification (requires explicit opt-in)
    python run_7tier_experiment.py --model mistralai/ministral-3b-2512 --trials 15 --resume \
        --with-reasoning --max-tokens 200

    # Pilot: 3 variation sets
    python run_7tier_experiment.py --trials 10 \
        --variation-ids Personal_finances_5188 Global_economy_8344 AI_moral_patienthood_490
    
    # Smoke-style slice (like property_ladder_pruning --max-ladders): first N sets;
    # writes under <results-dir>/smoke_<model>/ and <checkpoints-dir>/smoke_<model>/
    python run_7tier_experiment.py --model mistralai/ministral-3b-2512 --trials 1 --max-variation-sets 2 --smoke
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# This script lives in src/run_experiments/ladder_statement_pair/; the repo root
# is four parents up. data/ and src/ are siblings under the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PARAMETRIC_ROOT = _SCRIPT_DIR.parent.parent.parent

for _p in (_PARAMETRIC_ROOT / "src", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_runner_tradeoff import run_experiment, artifact_dir_name_for_test

# External dependency: `compute_utilities` is NOT vendored in this repo. Expose
# it on PYTHONPATH (e.g. from the original utility_analysis tree) to run experiments.
try:
    from compute_utilities.budget_monitor import BudgetMonitor
except ImportError as exc:  # pragma: no cover - depends on external package
    raise ImportError(
        "run_7tier_experiment requires the external `compute_utilities` package, "
        "which is not bundled with this repo. Install/expose it on PYTHONPATH to "
        "run experiments."
    ) from exc

COST_LOG_NAME = "phase6b_cost_log.json"
COST_SUMMARY_NAME = "cost_summary.json"


def resolve_under_parametric(rel: str | Path) -> Path:
    """Resolve a path relative to parametric_variations/ (unless already absolute)."""
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (_PARAMETRIC_ROOT / p).resolve()


def smoke_run_subdir(model_key: str) -> str:
    """Folder segment for smoke runs (mirrors property_ladder_pruning.smoke_output_dir_name)."""
    short = model_key.replace("-openai", "").replace("-", "")
    return f"smoke_{short}"


def discover_manifest_path(data_dir: Path) -> Path:
    """Prefer pruned pipeline manifest, then legacy full phase6b manifest."""
    for name in (
        "phase6b_variations_pruned_final_manifest.json",
        "phase6b_manifest.json",
    ):
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No manifest found in {data_dir}. Expected "
        "phase6b_variations_pruned_final_manifest.json or phase6b_manifest.json. "
        "Generate comparisons (generate_7tier_comparisons.py) or pass --manifest."
    )


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def get_test_names(manifest: dict, variation_ids: list[str] | None = None) -> list[str]:
    all_files = manifest["variation_files"]
    all_test_names = [f.replace("_comparisons.json", "") for f in all_files]

    if variation_ids:
        filtered = []
        for tn in all_test_names:
            for vid in variation_ids:
                if vid in tn:
                    filtered.append(tn)
                    break
        return filtered

    return all_test_names


def is_complete(results_dir: Path, test_name: str, model_key: str) -> bool:
    # Primary path uses compact artifact dir names to avoid Windows MAX_PATH issues.
    artifact_dir = artifact_dir_name_for_test(test_name)
    compact = results_dir / artifact_dir / "results.json"
    if compact.exists():
        return True
    # Transitional compact naming.
    compact_v1 = results_dir / artifact_dir / f"{artifact_dir}_{model_key}_results.json"
    if compact_v1.exists():
        return True
    # Back-compat for previously written legacy layout.
    legacy = results_dir / test_name / f"{test_name}_{model_key}_results.json"
    return legacy.exists()


async def smoke_call(
    model_key: str,
    max_tokens: int,
    temperature: float | None,
) -> bool:
    """One-call sanity check before launching the full pilot.

    Catches auth errors, model-name typos, provider-detection issues, and
    unsupported-parameter bugs in ~2s instead of letting them poison a
    420-call run.
    """
    import time
    # External dependency: `compute_utilities` is NOT vendored in this repo.
    try:
        from compute_utilities.utils import create_agent
    except ImportError as exc:  # pragma: no cover - depends on external package
        raise ImportError(
            "The smoke test requires the external `compute_utilities` package, "
            "which is not bundled with this repo. Install/expose it on PYTHONPATH."
        ) from exc

    # Mirror experiment_runner_tradeoff.py: read extra_body / enable_cache from
    # MODEL_CONFIGS so the smoke uses the same provider parameters as the real run.
    # Without this, hybrid models (Nemotron, DeepSeek V3.1, etc.) silently miss
    # their reasoning-toggle and produce malformed output that fails the smoke.
    extra_body = None
    enable_cache = False
    from config import MODEL_CONFIGS
    cfg = MODEL_CONFIGS.get(model_key)
    system_message = None
    if cfg is not None:
        extra_body = cfg.extra_body
        enable_cache = cfg.enable_cache
        system_message = cfg.system_message

    agent = create_agent(
        model_key,
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
        enable_cache=enable_cache,
    )
    prompt = (
        "Which do you prefer?\n"
        "A. an apple\n"
        "B. a banana\n\n"
        'Reply with only the letter "A" or "B".'
    )
    msgs = []
    if system_message:
        msgs.append({"role": "system", "content": system_message})
    msgs.append({"role": "user", "content": prompt})
    messages = [msgs]
    t0 = time.perf_counter()
    try:
        raw = await agent.async_completions(messages, verbose=False)
    except Exception as e:
        print(f"  [smoke] FAILED after {time.perf_counter() - t0:.2f}s: {e}")
        return False
    dt = time.perf_counter() - t0
    text = (raw[0] or "").strip().upper()
    ok = text.startswith("A") or text.startswith("B")
    status = "OK" if ok else "FAIL (response not parseable as A/B)"
    print(f"  [smoke] {dt:.2f}s, response={text!r}, {status}")
    return ok


async def run_single(
    test_name: str,
    model_key: str,
    num_trials: int,
    with_reasoning: bool,
    max_tokens: int,
    data_dir: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    verbose: bool,
    model_variant: str = "instruct",
    reasoning_mode: str = "none",
    temperature: float | None = None,
    k_samples: int = 1,
    infrastructure: str = "openai_api",
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    quantization: str | None = None,
    system_message: str = "You are a helpful assistant.",
) -> dict | None:
    try:
        result = await run_experiment(
            test_name=test_name,
            model_key=model_key,
            num_trials=num_trials,
            data_dir=data_dir,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            include_flipped=True,
            resume=True,
            with_reasoning=with_reasoning,
            max_tokens=max_tokens,
            verbose=verbose,
            model_variant=model_variant,
            reasoning_mode=reasoning_mode,
            temperature=temperature,
            k_samples=k_samples,
            infrastructure=infrastructure,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            quantization=quantization,
            system_message=system_message,
        )
        return result
    except Exception as e:
        print(f"ERROR running {test_name}: {e}")
        return None


async def run_phase6b(
    model_key: str,
    num_trials: int,
    with_reasoning: bool,
    max_tokens: int,
    data_dir: Path,
    manifest_path: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    variation_ids: list[str] | None,
    max_concurrent: int,
    resume: bool,
    verbose: bool,
    model_variant: str = "instruct",
    reasoning_mode: str = "none",
    temperature: float | None = None,
    k_samples: int = 1,
    infrastructure: str = "openai_api",
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    quantization: str | None = None,
    hub_dataset: str | None = None,
    skip_smoke_test: bool = False,
    system_message: str = "You are a helpful assistant.",
    start_from: int = 0,
    max_variation_sets: int | None = None,
    smoke: bool = False,
) -> None:
    manifest = load_manifest(manifest_path)
    test_names = get_test_names(manifest, variation_ids)
    if start_from:
        test_names = test_names[start_from:]
    if max_variation_sets is not None:
        test_names = test_names[:max_variation_sets]

    smoke_scope = smoke
    if smoke_scope:
        sub = smoke_run_subdir(model_key)
        results_dir = (results_dir / sub).resolve()
        checkpoints_dir = (checkpoints_dir / sub).resolve()
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if not test_names:
        print(
            "No variation sets to run after --variation-ids / --start-from / "
            "--max-variation-sets filters."
        )
        return

    print(f"Phase 6b Monotonicity Experiment (7 tiers)")
    print(f"  Model: {model_key}")
    print(f"  Trials: {num_trials}")
    print(f"  Variation sets: {len(test_names)}")
    print(f"  Tiers: {manifest['n_tiers']}")
    print(f"  Comparisons per set: {manifest['n_comparison_samples'] * manifest['n_tiers']}")
    total = len(test_names) * manifest["n_comparison_samples"] * manifest["n_tiers"]
    print(f"  Total comparisons: {total}")
    print(f"  API calls (with flipped): {total * 2 * num_trials:,}")
    print(f"  CoT reasoning: {'ENABLED (max_tokens=' + str(max_tokens) + ')' if with_reasoning else 'DISABLED'}")
    print(f"  Model variant: {model_variant}")
    print(f"  Reasoning mode: {reasoning_mode}")
    if temperature is not None:
        print(f"  Temperature: {temperature}")
    if k_samples > 1:
        print(f"  K samples: {k_samples}")
    print(f"  Infrastructure: {infrastructure}")
    if smoke_scope:
        print(f"  Smoke scope: start_from={start_from}, max_variation_sets={max_variation_sets}")
        print(f"  Smoke paths: .../{smoke_run_subdir(model_key)}/ under results + checkpoints")
    print()

    completed_sets = sum(1 for tn in test_names if is_complete(results_dir, tn, model_key))

    full_slate_sets = len(manifest["variation_files"])
    full_slate_calls = (
        full_slate_sets
        * manifest["n_comparison_samples"]
        * manifest["n_tiers"]
        * 2
        * num_trials
    )
    this_run_calls = len(test_names) * manifest["n_comparison_samples"] * manifest["n_tiers"] * 2 * num_trials

    print("=" * 60)
    print("  RUN PLAN")
    print("=" * 60)
    print(f"  Model:            {model_key}")
    print(f"  Sets this run:    {len(test_names)}  (full manifest: {full_slate_sets} sets)")
    print(f"  Resume:           {'ON' if resume else 'OFF'}  already done: {completed_sets}/{len(test_names)}")
    print(f"  max_concurrent:   {max_concurrent}")
    print(f"  API calls (this run, flipped × trials): {this_run_calls:,}")
    print(f"  Full manifest equivalent:                {full_slate_calls:,}")
    print("=" * 60 + "\n")

    # One-call smoke: validates models.yaml + provider (like a focused health check).
    # Scoped to API infras; HF Jobs / base-model runs have their own validation path.
    if not skip_smoke_test and infrastructure in ("openai_api", "anthropic_api", "openrouter"):
        print("  Running pre-launch smoke test...")
        if not await smoke_call(model_key, max_tokens, temperature):
            print(
                "\n  Aborting: smoke test failed. Fix the underlying error before "
                "launching the full run, or rerun with --skip-smoke-test to bypass."
            )
            sys.exit(1)
        print()

    if resume:
        pending = [tn for tn in test_names if not is_complete(results_dir, tn, model_key)]
        skipped = len(test_names) - len(pending)
        if skipped > 0:
            print(f"  Skipping {skipped} already-completed variation sets")
        test_names = pending

    if not test_names:
        print("Nothing to run.")
        return

    print(f"  Running {len(test_names)} variation sets (max {max_concurrent} concurrent)\n")

    budget = BudgetMonitor(check_interval=3)
    await budget.force_check()
    if budget.last_usage is not None:
        print(f"  Budget: {budget.summary()}\n")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0
    start = datetime.now(timezone.utc)

    async def run_with_semaphore(test_name: str) -> bool:
        nonlocal completed, failed
        if budget.should_stop:
            print(f"  Skipping {test_name} (budget limit approaching)")
            return False
        async with semaphore:
            print(f"[{completed + failed + 1}/{len(test_names)}] Starting {test_name}")
            result = await run_single(
                test_name, model_key, num_trials, with_reasoning, max_tokens,
                data_dir, results_dir, checkpoints_dir, verbose,
                model_variant=model_variant,
                reasoning_mode=reasoning_mode,
                temperature=temperature,
                k_samples=k_samples,
                infrastructure=infrastructure,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                quantization=quantization,
                system_message=system_message,
            )
            if result is not None:
                completed += 1
                print(f"  Completed {test_name} ({completed}/{len(test_names)})")
                await budget.on_task_completed()
                return True
            else:
                failed += 1
                return False

    tasks = [run_with_semaphore(tn) for tn in test_names]
    await asyncio.gather(*tasks)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\nDone. Completed: {completed}, Failed: {failed}, "
          f"Elapsed: {elapsed:.0f}s")
    print(f"  Final budget: {budget.summary()}")

    cost_paths = _write_cost_logs(results_dir, model_key)
    if cost_paths is not None:
        cost_log_path, cost_summary_path = cost_paths
        print(f"  Cost Log: {cost_log_path}")
        print(f"  Cost Summary: {cost_summary_path}")

    if hub_dataset and completed > 0:
        _push_results_to_hub(results_dir, model_key, hub_dataset)


def _extract_total(usage_stats: dict, key: str) -> int:
    block = usage_stats.get(key) or {}
    val = block.get("total")
    return int(val) if isinstance(val, (int, float)) else 0


def _estimate_cost_from_totals(
    rates: dict[str, float] | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    openai_cached_tokens: int,
) -> float | None:
    if not rates:
        return None
    uncached = max(
        prompt_tokens
        - cache_creation_input_tokens
        - cache_read_input_tokens
        - openai_cached_tokens,
        0,
    )
    in_rate = rates["input"]
    out_rate = rates["output"]
    total = (
        uncached / 1_000_000 * in_rate
        + cache_creation_input_tokens / 1_000_000 * in_rate * 1.25
        + cache_read_input_tokens / 1_000_000 * in_rate * 0.10
        + openai_cached_tokens / 1_000_000 * in_rate * 0.10
        + completion_tokens / 1_000_000 * out_rate
    )
    return round(total, 6)


def _iter_result_files(results_dir: Path, model_key: str) -> list[Path]:
    files: list[Path] = []
    for set_dir in sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name.startswith("phase6b")
    ):
        p = set_dir / "results.json"
        if p.exists():
            files.append(p)
            continue
        # Transitional compact naming.
        p = set_dir / f"{set_dir.name}_{model_key}_results.json"
        if p.exists():
            files.append(p)
            continue
        # Legacy layout.
        p = set_dir / f"{set_dir.name}_{model_key}_results.json"
        if p.exists():
            files.append(p)
    return files


def _write_cost_logs(results_dir: Path, model_key: str) -> tuple[Path, Path] | None:
    try:
        from compute_utilities.preflight_check import MODEL_COST_ESTIMATES
        pricing = MODEL_COST_ESTIMATES.get(model_key)
    except Exception:
        pricing = None

    result_files = _iter_result_files(results_dir, model_key)
    if not result_files:
        return None

    records: list[dict] = []
    prompt_total = 0
    completion_total = 0
    reasoning_total = 0
    cache_create_total = 0
    cache_read_total = 0
    oai_cached_total = 0
    calls_logged_total = 0
    estimated_from_metadata_total = 0.0
    estimated_from_metadata_n = 0
    actual_total = 0.0
    actual_n = 0

    for path in result_files:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = d.get("config") or {}
        meta = d.get("metadata") or {}
        usage = meta.get("usage_stats") or {}
        prompt = _extract_total(usage, "prompt_tokens")
        completion = _extract_total(usage, "completion_tokens")
        reasoning = _extract_total(usage, "reasoning_tokens")
        cache_create = _extract_total(usage, "cache_creation_input_tokens")
        cache_read = _extract_total(usage, "cache_read_input_tokens")
        oai_cached = _extract_total(usage, "openai_cached_tokens")
        calls_logged = int(usage.get("calls_logged") or 0)

        prompt_total += prompt
        completion_total += completion
        reasoning_total += reasoning
        cache_create_total += cache_create
        cache_read_total += cache_read
        oai_cached_total += oai_cached
        calls_logged_total += calls_logged

        est_meta = meta.get("estimated_cost_usd")
        if isinstance(est_meta, (int, float)):
            estimated_from_metadata_total += float(est_meta)
            estimated_from_metadata_n += 1

        actual = meta.get("actual_cost_usd")
        if isinstance(actual, (int, float)):
            actual_total += float(actual)
            actual_n += 1

        records.append(
            {
                "test_name": cfg.get("test_name", path.parent.name),
                "result_path": str(path),
                "calls_logged": calls_logged,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
                "openai_cached_tokens": oai_cached,
                "estimated_cost_usd": est_meta,
                "actual_cost_usd": actual,
            }
        )

    estimated_from_usage = _estimate_cost_from_totals(
        pricing,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        cache_creation_input_tokens=cache_create_total,
        cache_read_input_tokens=cache_read_total,
        openai_cached_tokens=oai_cached_total,
    )

    cost_log = {
        "model_key": model_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pricing_source": "compute_utilities.preflight_check.MODEL_COST_ESTIMATES",
        "pricing_per_1m": pricing,
        "result_files_count": len(records),
        "calls_logged": calls_logged_total,
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": completion_total,
        "reasoning_tokens_total": reasoning_total,
        "cache_creation_input_tokens_total": cache_create_total,
        "cache_read_input_tokens_total": cache_read_total,
        "openai_cached_tokens_total": oai_cached_total,
        "estimated_cost_usd_from_usage": estimated_from_usage,
        "estimated_cost_usd_from_results_sum": round(estimated_from_metadata_total, 6),
        "estimated_cost_count_from_results": estimated_from_metadata_n,
        "actual_cost_usd_sum": round(actual_total, 6),
        "actual_cost_count": actual_n,
        "records": records,
    }

    summary = {
        "model": model_key,
        "n_recorded": calls_logged_total,
        "n_priced_files": len(records),
        "estimated_cost_usd": estimated_from_usage,
        "actual_cost_usd": round(actual_total, 6) if actual_n > 0 else None,
        "tokens": {
            "prompt_tokens_total": prompt_total,
            "completion_tokens_total": completion_total,
            "reasoning_tokens_total": reasoning_total,
            "cache_creation_input_tokens_total": cache_create_total,
            "cache_read_input_tokens_total": cache_read_total,
            "openai_cached_tokens_total": oai_cached_total,
        },
        "notes": (
            "Modeled after property_ladder_pruning cost artifacts. "
            "estimated_cost_usd uses MODEL_COST_ESTIMATES and observed token totals."
        ),
    }

    cost_log_path = results_dir / COST_LOG_NAME
    summary_path = results_dir / COST_SUMMARY_NAME
    cost_log_path.write_text(json.dumps(cost_log, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return cost_log_path, summary_path


def _write_flat_parquet(results_dir: Path, model_key: str) -> Path | None:
    """Flatten per-set JSONs into one row-per-comparison parquet for Data Studio.

    Returns the parquet path, or None if no per-set files were found.
    """
    import pandas as pd

    rows = []
    for set_dir in sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name.startswith("phase6b")
    ):
        result_path = set_dir / "results.json"
        if not result_path.exists():
            # Transitional compact naming.
            result_path = set_dir / f"{set_dir.name}_{model_key}_results.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            d = json.load(f)
        test_name = d.get("config", {}).get("test_name", set_dir.name)
        for pref in d.get("preferences", []):
            oa = pref.get("outcome_a", {}) or {}
            ob = pref.get("outcome_b", {}) or {}
            count_a = pref.get("count_prefer_a", 0)
            count_b = pref.get("count_prefer_b", 0)
            rows.append({
                "test_name": test_name,
                "variation_id": oa.get("variation_id"),
                "model_key": model_key,
                "comparison_id": ob.get("comparison_id"),
                "tier": oa.get("tier"),
                "tier_label": oa.get("tier_label"),
                "category": oa.get("category"),
                "valence": oa.get("valence"),
                "identified_property": oa.get("identified_property"),
                "outcome_a_text": oa.get("text"),
                "outcome_b_text": ob.get("text"),
                "outcome_b_category": ob.get("comparison_category"),
                "count_prefer_a": count_a,
                "count_prefer_b": count_b,
                "prob_prefer_a": pref.get("prob_prefer_a"),
                "prob_prefer_b": pref.get("prob_prefer_b"),
                "total_parseable_trials": count_a + count_b,
            })

    if not rows:
        print("  No per-set result JSONs found; skipping flat parquet.")
        return None

    df = pd.DataFrame(rows)
    out = results_dir / "flat_comparisons.parquet"
    df.to_parquet(out, index=False)
    print(f"  Wrote flat parquet: {out} ({len(df):,} rows)")
    return out


def _push_results_to_hub(results_dir: Path, model_key: str, hub_dataset: str) -> None:
    """Upload results/ to an existing HF Hub dataset repo under a subdir per model + run.

    The repo must be pre-created in the browser. HF_TOKEN needs write scope on
    that specific repo (fine-grained is sufficient).
    """
    from huggingface_hub import HfApi
    _write_flat_parquet(results_dir, model_key)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path_in_repo = f"{model_key}/{run_id}"
    print(f"\n  Pushing results to dataset '{hub_dataset}' at {path_in_repo}/ ...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(results_dir),
        path_in_repo=path_in_repo,
        repo_id=hub_dataset,
        repo_type="dataset",
        commit_message=f"phase6b results for {model_key} ({run_id})",
    )
    print(f"  Uploaded: https://huggingface.co/datasets/{hub_dataset}/tree/main/{path_in_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 6b monotonicity experiments (7 tiers)"
    )
    parser.add_argument("--model", default="gpt-4o-mini-openrouter")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per response. If unset, taken from MODEL_CONFIGS "
                             "(falls back to 10). Hybrid thinking models need >= 1500 to "
                             "avoid truncation in the reasoning channel.")
    parser.add_argument("--with-reasoning", action="store_true", default=False,
                        help="Enable CoT reasoning (Level 3); increases tokens and cost.")
    parser.add_argument(
        "--data-dir",
        default="data/run_experiments/phase6b_variations_pruned",
        help="Directory with manifest + *_comparisons.json (default: "
        "data/run_experiments/phase6b_variations_pruned). "
        "Relative paths are resolved under the repo root.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON (default: auto-detect phase6b_variations_pruned_final_manifest.json "
        "or phase6b_manifest.json inside --data-dir).",
    )
    parser.add_argument(
        "--results-dir",
        default="outputs",
        help="Results root (default: outputs). Relative to the repo root.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Checkpoints directory (default: checkpoints). Relative to the repo root.",
    )
    parser.add_argument("--variation-ids", nargs="+", default=None)
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip this many variation sets (after --variation-ids filter) before running.",
    )
    parser.add_argument(
        "--max-variation-sets",
        type=int,
        default=None,
        help="Run at most this many sets (after --start-from).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Write outputs under <results-dir>/smoke_<model>/ and matching checkpoints subdir.",
    )
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip completed sets (default: OFF).")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model-variant", default="instruct",
                        help="Model variant: base, instruct, hybrid, hybrid_thinking, reasoning")
    parser.add_argument("--reasoning-mode", default="none",
                        help="Reasoning mode: none, cot, thinking")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature")
    parser.add_argument("--k-samples", type=int, default=1,
                        help="Samples per prompt for base models")
    parser.add_argument("--infrastructure", default="openai_api",
                        help="Infrastructure: openai_api, anthropic_api, openrouter, hf_jobs, local")
    parser.add_argument("--gpu-type", default=None, help="GPU type (e.g. H200)")
    parser.add_argument("--gpu-count", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--quantization", default=None, help="Quantization: fp8, awq, gptq")
    parser.add_argument("--skip-smoke-test", action="store_true", default=False,
                        help="Skip the pre-launch one-call smoke test. Default: smoke test ON.")
    parser.add_argument("--system-message", default=None,
                        help="Override system message. If unset, taken from MODEL_CONFIGS "
                             "(falls back to 'You are a helpful assistant.').")
    parser.add_argument("--hub-dataset", default=None,
                        help="After completion, push results/ to this HF dataset repo "
                             "(e.g. 'elenaajayi/emergent-values-smoke-results'). "
                             "Requires HF_TOKEN with write scope in the environment.")
    args = parser.parse_args()
    if args.start_from < 0:
        parser.error("--start-from must be >= 0")
    if args.max_variation_sets is not None and args.max_variation_sets < 1:
        parser.error("--max-variation-sets must be >= 1 when set")

    from config import MODEL_CONFIGS
    cfg = MODEL_CONFIGS.get(args.model)
    if args.max_tokens is None:
        args.max_tokens = cfg.max_tokens if cfg is not None else 10

    # Resolve system message: CLI flag > ModelConfig > default
    if args.system_message is not None:
        sys_msg = args.system_message
    elif cfg is not None and cfg.system_message is not None:
        sys_msg = cfg.system_message
    else:
        sys_msg = "You are a helpful assistant."

    data_dir = resolve_under_parametric(args.data_dir)
    if args.manifest:
        manifest_path = resolve_under_parametric(args.manifest)
    else:
        manifest_path = discover_manifest_path(data_dir)
    # Namespace artifacts by model so multi-model runs don't mix outputs.
    results_root = resolve_under_parametric(args.results_dir)
    checkpoints_root = resolve_under_parametric(args.checkpoints_dir)
    results_dir = (results_root / args.model).resolve()
    checkpoints_dir = (checkpoints_root / args.model).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {manifest_path}")
    print(f"Data dir: {data_dir}")
    print(f"Results dir (model): {results_dir}")
    print(f"Checkpoints dir (model): {checkpoints_dir}")
    if args.smoke:
        print(
            f"  -> effective results/checkpoints: .../{smoke_run_subdir(args.model)}/ "
            "(smoke scope)"
        )
    print()

    asyncio.run(
        run_phase6b(
            model_key=args.model,
            num_trials=args.trials,
            with_reasoning=args.with_reasoning,
            max_tokens=args.max_tokens,
            data_dir=data_dir,
            manifest_path=manifest_path,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            variation_ids=args.variation_ids,
            max_concurrent=args.max_concurrent,
            resume=args.resume,
            verbose=not args.quiet,
            model_variant=args.model_variant,
            reasoning_mode=args.reasoning_mode,
            temperature=args.temperature,
            k_samples=args.k_samples,
            infrastructure=args.infrastructure,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            quantization=args.quantization,
            hub_dataset=args.hub_dataset,
            skip_smoke_test=args.skip_smoke_test,
            system_message=sys_msg,
            start_from=args.start_from,
            max_variation_sets=args.max_variation_sets,
            smoke=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
