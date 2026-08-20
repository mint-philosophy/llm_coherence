#!/usr/bin/env python3
"""
Run forced-choice monotonicity experiments on 7-tier variation sets.

Reads a phase 6b manifest (pruned ladders by default) and comparison JSONs from
``<parametric_variations>/data/...``.

Usage (from this directory):

    # Non reasoning model
    for s in 0 10 20 30 40 50 60 70 80 90; do
        PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
            --model ministral-3b-2512-openrouter \
            --trials 10 \
            --start-from $s \
            --max-variation-sets 10 \
            --max-concurrent 6 \
            --resume
    done

    # Reasoning model

    for s in 0 10 20 30 40 50 60 70 80 90; do
        PYTHONPATH=src python -m llm_coherence.experiments.ladder_statement_pair.run_7tier_experiment \
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
    
    # Smoke-style slice: first N sets under outputs/<model>/smoke_<model>/ladder_vs_comparison_statements/
    python run_7tier_experiment.py --model mistralai/ministral-3b-2512 --trials 1 --max-variation-sets 2 --smoke

Artifacts:
    outputs/<model_key>/ladder_vs_comparison_statements/phase6b_variations_prune_*/results.json
    outputs/<model_key>/ladder_vs_comparison_statements/phase6b_cost_log.json
    Smoke: outputs/<model_key>/smoke_<model>/ladder_vs_comparison_statements/
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from llm_coherence.experiments.ladder_statement_pair.experiment_runner_tradeoff import (
    artifact_dir_name_for_test,
    build_prompt,
    load_comparisons,
    run_experiment,
)
from llm_coherence.experiments.ladder_statement_pair.openai_batch_runner import (
    MAX_BATCH_REQUESTS,
    batch_queue_limit_for_model,
    create_retry_shards,
    print_batch_run_pre_submit_cost_estimate,
    generate_batch_run,
    process_batch_run,
    refresh_batch_jobs,
    resolve_batch_run_dir,
    submit_pending_batch_shards,
    validate_batch_run_binding,
    wait_for_batch_jobs,
)
from llm_coherence.config import (
    canonical_model_key,
    get_model_config,
    resolve_model_results_dir,
    results_dir_name,
)
from llm_coherence.paths import (
    CHECKPOINTS_OUTPUT_DIR,
    COMPARISONS_DIR,
    LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR,
    LADDER_VS_COMPARISON_SUBDIR,
    REPO_ROOT,
)
from llm_coherence.runtime.agents import (
    AsyncRequestLimiter,
    close_api_async_clients,
    create_agent,
    model_name_for_key,
)
from llm_coherence.runtime.budget_monitor import BudgetMonitor
from llm_coherence.runtime.preflight_check import (
    MODEL_COST_ESTIMATES,
    is_thinking_run,
    live_reasoning_output_tokens_per_request,
)
from llm_coherence.runtime.usage_cost import (
    actual_cost_from_usage_summary,
    build_cost_summary_notes_from_metadata,
    estimate_cost_from_totals,
    infer_provider,
    resolve_rates,
)

_PARAMETRIC_ROOT = REPO_ROOT

COST_LOG_NAME = "phase6b_cost_log.json"
COST_SUMMARY_NAME = "cost_summary.json"

_ESTIMATED_UTF8_BYTES_PER_INPUT_TOKEN = 5
_ESTIMATED_INPUT_FRAMING_TOKENS_PER_REQUEST = 8
_ESTIMATED_REASONING_OFF_OUTPUT_TOKENS_PER_REQUEST = 5
_ESTIMATED_COT_OUTPUT_TOKENS_PER_REQUEST = 250
_PREFLIGHT_SMOKE_PROMPT = (
    "Which do you prefer?\n"
    "A. an apple\n"
    "B. a banana\n\n"
    'Reply with only the letter "A" or "B".'
)


def batch_job_progress(entry: dict) -> dict[str, int | float]:
    """Return stable progress metrics for one recorded OpenAI Batch job."""
    counts = entry.get("request_counts") or {}
    total = int(counts.get("total") or entry.get("request_count") or 0)
    completed = int(counts.get("completed") or 0)
    failed = int(counts.get("failed") or 0)
    processed = completed + failed
    remaining = max(total - processed, 0)
    percent = (processed / total * 100.0) if total else 0.0
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "processed": processed,
        "remaining": remaining,
        "percent": percent,
    }


def format_batch_job_status(entry: dict) -> str:
    """Render one Batch job with its shard identity and useful progress."""
    progress = batch_job_progress(entry)
    return (
        f"{entry.get('input_file', 'unknown shard')} -> {entry['batch_id']}: "
        f"{entry['status']} | "
        f"processed={progress['processed']:,}/{progress['total']:,} "
        f"({progress['percent']:.1f}%), "
        f"completed={progress['completed']:,}, failed={progress['failed']:,}, "
        f"remaining={progress['remaining']:,}"
    )


def aggregate_batch_progress(entries: list[dict]) -> dict[str, int | float]:
    """Aggregate progress across every submitted shard in a Batch run."""
    per_job = [batch_job_progress(entry) for entry in entries]
    total = sum(int(progress["total"]) for progress in per_job)
    completed = sum(int(progress["completed"]) for progress in per_job)
    failed = sum(int(progress["failed"]) for progress in per_job)
    processed = completed + failed
    remaining = max(total - processed, 0)
    percent = (processed / total * 100.0) if total else 0.0
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "processed": processed,
        "remaining": remaining,
        "percent": percent,
    }


def resolve_under_parametric(rel: str | Path) -> Path:
    """Resolve a path relative to parametric_variations/ (unless already absolute)."""
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (_PARAMETRIC_ROOT / p).resolve()


def repo_relative(path: str | Path) -> str:
    """Return a repo-relative path string for commands run inside the image."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def smoke_run_subdir(model_key: str) -> str:
    """Folder segment for smoke runs (mirrors property_ladder_pruning.smoke_output_dir_name)."""
    short = model_key.replace("-openai", "").replace("-", "")
    return f"smoke_{short}"


def model_results_dir_for_run(model_key: str, root: Path, *, smoke: bool = False) -> Path:
    if smoke:
        return root / results_dir_name(model_key) / smoke_run_subdir(model_key)
    return resolve_model_results_dir(model_key, root)


def model_run_dir(model_key: str, results_root: Path, *, smoke: bool = False) -> Path:
    return model_results_dir_for_run(model_key, results_root, smoke=smoke) / LADDER_VS_COMPARISON_SUBDIR


def model_run_checkpoints_dir(
    model_key: str, checkpoints_root: Path, *, smoke: bool = False
) -> Path:
    return model_results_dir_for_run(model_key, checkpoints_root, smoke=smoke) / LADDER_VS_COMPARISON_SUBDIR


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


def manifest_item_for_file(data_dir: Path, rel_or_abs: str) -> dict:
    """Return the runnable item described by one manifest variation file."""
    rel_path = Path(rel_or_abs)
    comparison_path = rel_path.resolve() if rel_path.is_absolute() else (data_dir / rel_path).resolve()
    test_name = rel_path.name.replace("_comparisons.json", "")
    return {
        "test_name": test_name,
        "comparison_path": comparison_path,
        "manifest_path": rel_or_abs,
    }


def get_manifest_items(
    manifest: dict,
    data_dir: Path,
    variation_ids: list[str] | None = None,
) -> list[dict]:
    all_files = manifest["variation_files"]
    all_items = [manifest_item_for_file(data_dir, f) for f in all_files]

    if variation_ids:
        filtered = []
        for item in all_items:
            tn = item["test_name"]
            for vid in variation_ids:
                if vid in tn:
                    filtered.append(item)
                    break
        return filtered

    return all_items


def scoped_manifest_items(
    manifest: dict,
    data_dir: Path,
    variation_ids: list[str] | None = None,
    *,
    start_from: int = 0,
    max_variation_sets: int | None = None,
) -> list[dict]:
    """Apply the CLI's deterministic manifest filters in one shared place."""
    run_items = get_manifest_items(manifest, data_dir, variation_ids)
    if start_from:
        run_items = run_items[start_from:]
    if max_variation_sets is not None:
        run_items = run_items[:max_variation_sets]
    return run_items


def _utf8_text_bytes(value: object) -> int:
    """Count UTF-8 bytes in nested request text fields."""
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(_utf8_text_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_utf8_text_bytes(item) for item in value.values())
    return 0


def estimate_phase6b_live_cost(
    model_key: str,
    *,
    run_items: list[dict],
    data_dir: Path,
    num_trials: int,
    with_reasoning: bool,
    max_tokens: int,
    system_message: str,
    include_prelaunch_smoke: bool = True,
) -> dict | None:
    """Estimate one live Phase 6b scope without making an API request.

    The estimator builds the same A/B and B/A prompt text as the live runner,
    applies the transparent UTF-8 input-token heuristic used by the
    within-ladder preflight, and reports both a calibrated projection and a
    strict all-output-caps estimate.
    """
    model_key = canonical_model_key(model_key)
    pricing = MODEL_COST_ESTIMATES.get(model_key)
    if pricing is None:
        return None

    experiment_request_count = 0
    input_text_bytes = 0
    total_comparisons = 0

    for item in run_items:
        comparisons = load_comparisons(
            data_dir,
            item["test_name"],
            item["comparison_path"],
        )
        total_comparisons += len(comparisons)
        for comparison in comparisons:
            text_a = comparison["outcome_a"]["text"]
            text_b = comparison["outcome_b"]["text"]
            for option_a, option_b in ((text_a, text_b), (text_b, text_a)):
                prompt = build_prompt(
                    option_a,
                    option_b,
                    with_reasoning=with_reasoning,
                    cache_structure=False,
                )
                input_text_bytes += num_trials * _utf8_text_bytes(
                    {
                        "system": system_message,
                        "user": prompt,
                    }
                )
                experiment_request_count += num_trials

    prelaunch_request_count = 1 if include_prelaunch_smoke and run_items else 0
    if prelaunch_request_count:
        try:
            configured_system_message = get_model_config(model_key).system_message
        except ValueError:
            configured_system_message = None
        input_text_bytes += _utf8_text_bytes(
            {
                "system": configured_system_message or "",
                "user": _PREFLIGHT_SMOKE_PROMPT,
            }
        )

    request_count = experiment_request_count + prelaunch_request_count
    estimated_input_tokens = (
        input_text_bytes + _ESTIMATED_UTF8_BYTES_PER_INPUT_TOKEN - 1
    ) // _ESTIMATED_UTF8_BYTES_PER_INPUT_TOKEN
    estimated_input_tokens += (
        request_count * _ESTIMATED_INPUT_FRAMING_TOKENS_PER_REQUEST
    )

    reasoning_on = is_thinking_run(model_key) or with_reasoning
    if reasoning_on:
        if is_thinking_run(model_key):
            projected_output_per_request = (
                live_reasoning_output_tokens_per_request(
                    model_key,
                    experiment="phase6b",
                )
            )
        else:
            projected_output_per_request = (
                _ESTIMATED_COT_OUTPUT_TOKENS_PER_REQUEST
            )
    else:
        projected_output_per_request = (
            _ESTIMATED_REASONING_OFF_OUTPUT_TOKENS_PER_REQUEST
        )
    projected_output_per_request = min(projected_output_per_request, max_tokens)
    projected_output_tokens = request_count * projected_output_per_request
    output_token_cap = request_count * max_tokens

    projected_cost = estimate_cost_from_totals(
        pricing,
        prompt_tokens=estimated_input_tokens,
        completion_tokens=projected_output_tokens,
    )
    all_cap_cost = estimate_cost_from_totals(
        pricing,
        prompt_tokens=estimated_input_tokens,
        completion_tokens=output_token_cap,
    )
    return {
        "model_key": model_key,
        "variation_sets": len(run_items),
        "total_comparisons": total_comparisons,
        "num_trials": num_trials,
        "experiment_request_count": experiment_request_count,
        "prelaunch_request_count": prelaunch_request_count,
        "request_count": request_count,
        "reasoning_on": reasoning_on,
        "input_text_bytes": input_text_bytes,
        "estimated_input_tokens": estimated_input_tokens,
        "projected_output_per_request": projected_output_per_request,
        "projected_output_tokens": projected_output_tokens,
        "output_token_cap": output_token_cap,
        "estimated_cost_usd": projected_cost,
        "maximum_output_cost_usd": all_cap_cost,
        "input_rate_per_mtok": pricing["input"],
        "output_rate_per_mtok": pricing["output"],
    }


def print_phase6b_live_cost_estimate(
    model_key: str,
    *,
    run_items: list[dict],
    data_dir: Path,
    num_trials: int,
    with_reasoning: bool,
    max_tokens: int,
    system_message: str,
    include_prelaunch_smoke: bool = True,
) -> dict | None:
    """Print and return the offline live-run estimate for one selected scope."""
    estimate = estimate_phase6b_live_cost(
        model_key,
        run_items=run_items,
        data_dir=data_dir,
        num_trials=num_trials,
        with_reasoning=with_reasoning,
        max_tokens=max_tokens,
        system_message=system_message,
        include_prelaunch_smoke=include_prelaunch_smoke,
    )
    if estimate is None:
        print(
            f"[{model_key}] Preflight live cost estimate unavailable: "
            "no offline pricing is configured."
        )
        return None

    print(
        f"[{estimate['model_key']}] Preflight Phase 6b live cost estimate: "
        f"~${estimate['estimated_cost_usd']:,.6f} projected; "
        f"all-cap estimate ~${estimate['maximum_output_cost_usd']:,.6f}."
    )
    print(
        f"  Scope: {estimate['variation_sets']:,} variation sets, "
        f"{estimate['total_comparisons']:,} comparisons, "
        f"{estimate['experiment_request_count']:,} experiment requests"
        + (
            f" + {estimate['prelaunch_request_count']} pre-launch smoke request."
            if estimate["prelaunch_request_count"]
            else "."
        )
    )
    print(
        f"  Basis: ~{estimate['estimated_input_tokens']:,} input tokens "
        "(offline UTF-8 text-size heuristic), "
        f"~{estimate['projected_output_tokens']:,} projected output tokens."
    )
    if estimate["reasoning_on"]:
        print(
            "  Reasoning-on projection: "
            f"{estimate['projected_output_per_request']:,} output tokens/request "
            "from live-smoke calibration."
        )
    else:
        print(
            "  Reasoning-off projection: "
            f"{estimate['projected_output_per_request']:,} output tokens/request."
        )
    print(
        f"  Configured output-token cap: {estimate['output_token_cap']:,} total; "
        f"rates: ${estimate['input_rate_per_mtok']:g} input / "
        f"${estimate['output_rate_per_mtok']:g} output per 1M tokens."
    )
    print("  Estimate only; no API request has been sent.")
    return estimate


def is_complete(results_dir: Path, test_name: str, model_key: str) -> bool:
    def valid(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        config = payload.get("config") or {}
        metadata = payload.get("metadata") or {}
        if config.get("model_key") not in (None, model_key):
            return False
        if config.get("infrastructure") == "openai_batch_api":
            return metadata.get("run_status") == "complete"
        return True

    # Primary path uses compact artifact dir names to avoid Windows MAX_PATH issues.
    artifact_dir = artifact_dir_name_for_test(test_name)
    compact = results_dir / artifact_dir / "results.json"
    if valid(compact):
        return True
    # Transitional compact naming.
    compact_v1 = results_dir / artifact_dir / f"{artifact_dir}_{model_key}_results.json"
    if valid(compact_v1):
        return True
    # Back-compat for previously written legacy layout.
    legacy = results_dir / test_name / f"{test_name}_{model_key}_results.json"
    return valid(legacy)


async def smoke_call(
    model_key: str,
    max_tokens: int,
    temperature: float | None,
    max_retries: int = 5,
) -> bool:
    """One-call sanity check before launching the full pilot.

    Catches auth errors, model-name typos, provider-detection issues, and
    unsupported-parameter bugs in ~2s instead of letting them poison a
    420-call run.
    """
    import time

    # Mirror experiment_runner_tradeoff.py: read extra_body / enable_cache from
    # MODEL_CONFIGS so the smoke uses the same provider parameters as the real run.
    # Without this, hybrid models (Nemotron, DeepSeek V3.1, etc.) silently miss
    # their reasoning-toggle and produce malformed output that fails the smoke.
    extra_body = None
    enable_cache = False
    try:
        cfg = get_model_config(model_key)
    except ValueError:
        cfg = None
    system_message = None
    if cfg is not None:
        extra_body = cfg.extra_body
        enable_cache = cfg.enable_cache
        system_message = cfg.system_message

    agent = create_agent(
        model_key,
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        extra_body=extra_body,
        enable_cache=enable_cache,
    )
    msgs = []
    if system_message:
        msgs.append({"role": "system", "content": system_message})
    msgs.append({"role": "user", "content": _PREFLIGHT_SMOKE_PROMPT})
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
    comparison_path: Path,
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
    request_limiter: AsyncRequestLimiter | None = None,
    max_retries: int = 5,
) -> dict | None:
    try:
        result = await run_experiment(
            test_name=test_name,
            model_key=model_key,
            num_trials=num_trials,
            data_dir=data_dir,
            comparison_path=comparison_path,
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
            request_limiter=request_limiter,
            max_retries=max_retries,
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
    request_concurrency: int | None = None,
    requests_per_second: float | None = None,
    max_retries: int = 5,
) -> None:
    manifest = load_manifest(manifest_path)
    run_items = scoped_manifest_items(
        manifest,
        data_dir,
        variation_ids,
        start_from=start_from,
        max_variation_sets=max_variation_sets,
    )

    if not run_items:
        print(
            "No variation sets to run after --variation-ids / --start-from / "
            "--max-variation-sets filters."
        )
        return

    print("Phase 6b Monotonicity Experiment (7 tiers)")
    print(f"  Model: {model_key}")
    print(f"  Trials: {num_trials}")
    print(f"  Variation sets: {len(run_items)}")
    print(f"  Tiers: {manifest['n_tiers']}")
    print(f"  Comparisons per set: {manifest['n_comparison_samples'] * manifest['n_tiers']}")
    total = len(run_items) * manifest["n_comparison_samples"] * manifest["n_tiers"]
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
    if smoke:
        print(f"  Smoke scope: start_from={start_from}, max_variation_sets={max_variation_sets}")
        print(
            f"  Smoke paths: .../{smoke_run_subdir(model_key)}/"
            f"{LADDER_VS_COMPARISON_SUBDIR}/ under results + checkpoints"
        )
    print()

    completed_sets = sum(
        1 for item in run_items
        if is_complete(results_dir, item["test_name"], model_key)
    )
    pending_items = (
        [
            item for item in run_items
            if not is_complete(results_dir, item["test_name"], model_key)
        ]
        if resume
        else list(run_items)
    )

    full_slate_sets = len(manifest["variation_files"])
    full_slate_calls = (
        full_slate_sets
        * manifest["n_comparison_samples"]
        * manifest["n_tiers"]
        * 2
        * num_trials
    )
    this_run_calls = (
        len(pending_items)
        * manifest["n_comparison_samples"]
        * manifest["n_tiers"]
        * 2
        * num_trials
    )

    live_infrastructures = ("openai_api", "anthropic_api", "openrouter")
    request_limiter: AsyncRequestLimiter | None = None
    resolved_request_concurrency: int | None = None
    if infrastructure in live_infrastructures:
        try:
            configured_request_concurrency = get_model_config(model_key).concurrency_limit
        except ValueError:
            configured_request_concurrency = 20
        resolved_request_concurrency = (
            request_concurrency
            if request_concurrency is not None
            else configured_request_concurrency
        )
        request_limiter = AsyncRequestLimiter(
            resolved_request_concurrency,
            requests_per_second=requests_per_second,
        )

    print("=" * 60)
    print("  RUN PLAN")
    print("=" * 60)
    print(f"  Model:            {model_key}")
    print(
        f"  Sets this run:    {len(pending_items)}  "
        f"(selected: {len(run_items)}, full manifest: {full_slate_sets})"
    )
    print(f"  Resume:           {'ON' if resume else 'OFF'}  already done: {completed_sets}/{len(run_items)}")
    print(f"  max_concurrent:   {max_concurrent}")
    print(f"  Retry attempts:   {max_retries} per request")
    if request_limiter is not None:
        unbounded_batch_peak = max_concurrent * min(
            num_trials,
            configured_request_concurrency,
        )
        request_limit_source = (
            "CLI override" if request_concurrency is not None else "model config"
        )
        print(
            f"  HTTP in flight:   {resolved_request_concurrency} run-wide "
            f"({request_limit_source}; legacy nested peak ~{unbounded_batch_peak})"
        )
        print(
            "  Request starts:   "
            + (
                f"smoothed at {requests_per_second:g}/second"
                if requests_per_second is not None
                else "not rate-capped"
            )
        )
        print("  Note: request controls are per Python process, not shared across terminals")
    print(f"  API calls (this run, flipped × trials): {this_run_calls:,}")
    print(f"  Full manifest equivalent:                {full_slate_calls:,}")
    print("=" * 60 + "\n")

    if not pending_items:
        print("Nothing to run; all selected variation sets are already complete.")
        return

    if infrastructure in live_infrastructures:
        print_phase6b_live_cost_estimate(
            model_key,
            run_items=pending_items,
            data_dir=data_dir,
            num_trials=num_trials,
            with_reasoning=with_reasoning,
            max_tokens=max_tokens,
            system_message=system_message,
            include_prelaunch_smoke=not skip_smoke_test,
        )
        print()

    # One-call smoke: validates models.yaml + provider (like a focused health check).
    # Scoped to API infras; HF Jobs / base-model runs have their own validation path.
    if not skip_smoke_test and infrastructure in live_infrastructures:
        print("  Running pre-launch smoke test...")
        if not await smoke_call(
            model_key,
            max_tokens,
            temperature,
            max_retries=max_retries,
        ):
            print(
                "\n  Aborting: smoke test failed. Fix the underlying error before "
                "launching the full run, or rerun with --skip-smoke-test to bypass."
            )
            sys.exit(1)
        print()

    if resume:
        skipped = len(run_items) - len(pending_items)
        if skipped > 0:
            print(f"  Skipping {skipped} already-completed variation sets")
    run_items = pending_items

    print(f"  Running {len(run_items)} variation sets (max {max_concurrent} concurrent)\n")

    budget = BudgetMonitor(check_interval=3)
    await budget.force_check()
    if budget.last_usage is not None:
        print(f"  Budget: {budget.summary()}\n")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0
    retry_totals: Counter[str] = Counter()
    start = datetime.now(timezone.utc)

    async def run_with_semaphore(item: dict) -> bool:
        nonlocal completed, failed
        test_name = item["test_name"]
        if budget.should_stop:
            print(f"  Skipping {test_name} (budget limit approaching)")
            return False
        async with semaphore:
            print(f"[{completed + failed + 1}/{len(run_items)}] Starting {test_name}")
            result = await run_single(
                test_name, item["comparison_path"], model_key, num_trials, with_reasoning, max_tokens,
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
                request_limiter=request_limiter,
                max_retries=max_retries,
            )
            if result is not None:
                result_retry_counts = (
                    (result.get("metadata") or {}).get("retry_counts") or {}
                )
                retry_totals.update(
                    {
                        key: int(value)
                        for key, value in result_retry_counts.items()
                        if isinstance(value, (int, float))
                    }
                )
                completed += 1
                print(f"  Completed {test_name} ({completed}/{len(run_items)})")
                await budget.on_task_completed()
                return True
            else:
                failed += 1
                return False

    tasks = [run_with_semaphore(item) for item in run_items]
    await asyncio.gather(*tasks)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\nDone. Completed: {completed}, Failed: {failed}, "
          f"Elapsed: {elapsed:.0f}s")
    print(f"  Final budget: {budget.summary()}")
    if request_limiter is not None:
        limiter_stats = request_limiter.snapshot()
        observed_rate = limiter_stats["observed_start_rate"]
        rate_suffix = (
            f", start_rate={observed_rate:.2f}/s"
            if isinstance(observed_rate, (int, float))
            else ""
        )
        print(
            "  Request limiter: "
            f"attempts={limiter_stats['attempts_started']:,}, "
            f"peak_in_flight={limiter_stats['peak_in_flight']}/"
            f"{limiter_stats['max_concurrency']}"
            f"{rate_suffix}"
        )
        print(
            "  Limiter wait (summed across attempts): "
            f"concurrency={limiter_stats['concurrency_wait_seconds']:.1f}s, "
            f"pacing={limiter_stats['pacing_wait_seconds']:.1f}s"
        )
    if retry_totals:
        print(
            "  Response diagnostics: "
            f"transport_retries={retry_totals['transport_retries']}, "
            f"transport_failures={retry_totals['transport_failures']}, "
            f"provider_errors={retry_totals['non_transport_errors']}, "
            f"token_capped={retry_totals['token_capped']}, "
            f"empty={retry_totals['empty_responses']}"
        )

    cost_paths = _write_cost_logs(results_dir, model_key)
    if cost_paths is not None:
        cost_log_path, cost_summary_path = cost_paths
        print(f"  Cost Log: {cost_log_path}")
        print(f"  Cost Summary: {cost_summary_path}")

    if failed:
        raise RuntimeError(
            f"Phase 6b run failed for {failed} variation set(s); "
            "inspect the log and resume after the infrastructure issue clears"
        )

    if hub_dataset and completed > 0:
        _push_results_to_hub(results_dir, model_key, hub_dataset)


async def run_phase6b_with_client_cleanup(**kwargs) -> None:
    """Run Phase 6b and close provider clients before the event loop exits."""
    try:
        await run_phase6b(**kwargs)
    finally:
        await close_api_async_clients()


def _extract_total(usage_stats: dict, key: str) -> int:
    block = usage_stats.get(key) or {}
    val = block.get("total")
    return int(val) if isinstance(val, (int, float)) else 0


def _resolve_pricing(
    model_key: str,
    *,
    batch: bool = False,
) -> tuple[dict[str, float] | None, str]:
    """Prefer configured rates; otherwise use the provider-rate fallback."""
    canonical_key = canonical_model_key(model_key)
    try:
        pricing = MODEL_COST_ESTIMATES.get(canonical_key)
    except Exception:
        pricing = None
    if pricing:
        if batch:
            pricing = {name: rate * 0.5 for name, rate in pricing.items()}
        source = "llm_coherence.runtime.preflight_check.MODEL_COST_ESTIMATES"
        if batch:
            source = f"OpenAI Batch API (50% of {source})"
        return pricing, source
    mid = model_name_for_key(canonical_key)
    rates, label = resolve_rates(infer_provider(mid), mid, batch=batch)
    return rates, label


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


def _write_cost_logs(
    results_dir: Path,
    model_key: str,
    *,
    batch: bool = False,
) -> tuple[Path, Path] | None:
    pricing, pricing_source = _resolve_pricing(model_key, batch=batch)

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
        if not isinstance(actual, (int, float)):
            actual = actual_cost_from_usage_summary(usage)
        if isinstance(actual, (int, float)):
            actual_total += float(actual)
            actual_n += 1
        else:
            actual = None

        records.append(
            {
                "test_name": cfg.get("test_name", path.parent.name),
                "result_path": Path(os.path.relpath(path, results_dir)).as_posix(),
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

    estimated_from_usage = estimate_cost_from_totals(
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
        "pricing_source": pricing_source,
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
        "actual_cost_usd_sum": round(actual_total, 6) if actual_n > 0 else None,
        "actual_cost_count": actual_n,
        "records": records,
    }

    summary = {
        "model": model_key,
        "n_recorded": calls_logged_total,
        "n_priced_files": len(records),
        "estimated_cost_usd": estimated_from_usage if actual_n == 0 else None,
        "actual_cost_usd": round(actual_total, 6) if actual_n > 0 else None,
        "tokens": {
            "prompt_tokens_total": prompt_total,
            "completion_tokens_total": completion_total,
            "reasoning_tokens_total": reasoning_total,
            "cache_creation_input_tokens_total": cache_create_total,
            "cache_read_input_tokens_total": cache_read_total,
            "openai_cached_tokens_total": oai_cached_total,
        },
        "notes": build_cost_summary_notes_from_metadata(has_actual=actual_n > 0),
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
        ignore_patterns=[
            "batch_runs/**",
            "**/batch_runs/**",
            "batch_id.txt",
            "batch_errors*.jsonl",
        ],
    )
    print(f"  Uploaded: https://huggingface.co/datasets/{hub_dataset}/tree/main/{path_in_repo}")


def build_phase6b_hf_job_code(
    *,
    model_key: str,
    trials: int,
    max_tokens: int,
    with_reasoning: bool,
    data_dir: str,
    manifest: str | None,
    results_dir: str,
    checkpoints_dir: str,
    variation_ids: list[str] | None,
    start_from: int,
    max_variation_sets: int | None,
    smoke: bool,
    max_concurrent: int,
    resume: bool,
    quiet: bool,
    model_variant: str,
    reasoning_mode: str,
    temperature: float | None,
    k_samples: int,
    gpu_type: str | None,
    gpu_count: int | None,
    quantization: str | None,
    system_message: str | None,
    hub_dataset: str | None,
    path_in_repo: str,
) -> str:
    """Build in-container Python for an HF Jobs 7-tier model run."""
    cmd = [
        "python3",
        "scripts/04_model_runs/10b_run_7tier_experiment.py",
        "--model",
        model_key,
        "--trials",
        str(trials),
        "--max-tokens",
        str(max_tokens),
        "--data-dir",
        data_dir,
        "--results-dir",
        results_dir,
        "--checkpoints-dir",
        checkpoints_dir,
        "--start-from",
        str(start_from),
        "--max-concurrent",
        str(max_concurrent),
        "--model-variant",
        model_variant,
        "--reasoning-mode",
        reasoning_mode,
        "--k-samples",
        str(k_samples),
        "--infrastructure",
        "hf_jobs",
        "--skip-smoke-test",
    ]
    if manifest:
        cmd.extend(["--manifest", manifest])
    if variation_ids:
        cmd.extend(["--variation-ids", *variation_ids])
    if max_variation_sets is not None:
        cmd.extend(["--max-variation-sets", str(max_variation_sets)])
    if smoke:
        cmd.append("--smoke")
    if resume:
        cmd.append("--resume")
    if quiet:
        cmd.append("--quiet")
    if with_reasoning:
        cmd.append("--with-reasoning")
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if gpu_type:
        cmd.extend(["--gpu-type", gpu_type])
    if gpu_count is not None:
        cmd.extend(["--gpu-count", str(gpu_count)])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if system_message is not None:
        cmd.extend(["--system-message", system_message])

    upload_dir = model_run_dir(model_key, Path(results_dir), smoke=smoke).as_posix()
    payload = {
        "cmd": cmd,
        "upload_dir": upload_dir,
        "hub_dataset": hub_dataset,
        "path_in_repo": path_in_repo,
    }
    return f"""
import json
import os
import subprocess
from pathlib import Path

payload = json.loads({json.dumps(json.dumps(payload))})

os.environ.setdefault("PYTHONPATH", "/app/src")
os.environ.setdefault("HF_HOME", "/data")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

os.chdir("/app")

print("=== 7-tier HF job start ===", flush=True)
print("command:", " ".join(payload["cmd"]), flush=True)
subprocess.check_call(payload["cmd"])

if payload["hub_dataset"]:
    from huggingface_hub import upload_folder

    folder_path = Path(payload["upload_dir"])
    print("\\n>>> uploading", folder_path, "to", payload["hub_dataset"], flush=True)
    upload_folder(
        repo_id=payload["hub_dataset"],
        repo_type="dataset",
        folder_path=str(folder_path),
        path_in_repo=payload["path_in_repo"],
    )
    print("uploaded to", payload["path_in_repo"], flush=True)

print("=== 7-tier HF job complete ===", flush=True)
""".strip()


def submit_phase6b_hf_job(args: argparse.Namespace, system_message: str | None) -> int:
    """Submit the existing 7-tier experiment CLI to Hugging Face Jobs."""
    if not args.image:
        raise SystemExit("--image is required with --submit-hf-job")
    if not args.namespace:
        raise SystemExit("--namespace is required with --submit-hf-job")

    job_tag = args.job_tag or uuid.uuid4().hex[:8]
    default_path = model_run_dir(args.model, Path(args.results_dir), smoke=args.smoke).as_posix()
    path_in_repo = args.path_in_repo or default_path
    code = build_phase6b_hf_job_code(
        model_key=args.model,
        trials=args.trials,
        max_tokens=args.max_tokens,
        with_reasoning=args.with_reasoning,
        data_dir=repo_relative(args.data_dir),
        manifest=repo_relative(args.manifest) if args.manifest else None,
        results_dir=repo_relative(args.results_dir),
        checkpoints_dir=repo_relative(args.checkpoints_dir),
        variation_ids=args.variation_ids,
        start_from=args.start_from,
        max_variation_sets=args.max_variation_sets,
        smoke=args.smoke,
        max_concurrent=args.max_concurrent,
        resume=args.resume,
        quiet=args.quiet,
        model_variant=args.model_variant,
        reasoning_mode=args.reasoning_mode,
        temperature=args.temperature,
        k_samples=args.k_samples,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        quantization=args.quantization,
        system_message=system_message,
        hub_dataset=args.hub_dataset,
        path_in_repo=path_in_repo,
    )

    if args.dry_run:
        print("HF Jobs command: python3 -u -c <generated code>")
        if args.model_volume:
            print(
                "HF model volume:",
                f"{args.model_volume} -> {args.model_volume_path}",
            )
        print(code)
        return 0

    try:
        from huggingface_hub import Volume, get_token, run_job
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for HF Jobs submission. "
            'Install with: python -m pip install -e ".[hf-jobs]"'
        ) from exc

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("No HF token found. Run `hf auth login` or set HF_TOKEN.")

    job_env = {
        "HF_HOME": "/data",
        "TRANSFORMERS_CACHE": "/data",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONUNBUFFERED": "1",
        "JOB_TAG": job_tag,
    }
    volumes = None
    if args.model_volume:
        print(
            "WARNING: HF model volumes use FUSE and may load very slowly for "
            "large sharded checkpoints. Omit --model-volume to use the local "
            "/data cache (recommended for GLM)."
        )
        volumes = [
            Volume(
                type="model",
                source=args.model_volume,
                mount_path=args.model_volume_path,
            )
        ]
        job_env["LLM_COHERENCE_VLLM_MODEL"] = args.model_volume_path
        job_env["LLM_COHERENCE_SAFETENSORS_LOAD_STRATEGY"] = "prefetch"
        job_env["LLM_COHERENCE_SAFETENSORS_PREFETCH_THREADS"] = "16"
        job_env["LLM_COHERENCE_MAX_MODEL_LEN"] = "4096"

    job = run_job(
        image=args.image,
        command=["python3", "-u", "-c", code],
        flavor=args.flavor,
        namespace=args.namespace,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        env=job_env,
        volumes=volumes,
    )
    print("job tag:", job_tag)
    print("job id:", job.id)
    print("job url:", job.url)
    if args.hub_dataset:
        print(
            "output path:",
            f"https://huggingface.co/datasets/{args.hub_dataset}/tree/main/{path_in_repo}",
        )
    return 0


def run_openai_batch_action(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    manifest_path: Path,
    results_dir: Path,
    batch_system_message: str | None,
) -> int:
    """Execute the requested step-10b OpenAI Batch API action."""
    if args.batch_action in {"generate", "run"}:
        source_manifest = load_manifest(manifest_path)
        run_items = get_manifest_items(
            source_manifest,
            data_dir,
            args.variation_ids,
        )
        if args.start_from:
            run_items = run_items[args.start_from:]
        if args.max_variation_sets is not None:
            run_items = run_items[: args.max_variation_sets]
        if args.resume:
            run_items = [
                item
                for item in run_items
                if not is_complete(results_dir, item["test_name"], args.model)
            ]
        if not run_items:
            print("No pending variation sets selected for this batch run.")
            return 0
        run_dir = generate_batch_run(
            run_items=run_items,
            data_dir=data_dir,
            source_manifest_path=manifest_path,
            results_dir=results_dir,
            model_key=args.model,
            num_trials=args.trials,
            include_flipped=True,
            with_reasoning=args.with_reasoning,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_message=batch_system_message,
            max_requests_per_batch=args.max_requests_per_batch,
        )
        batch_manifest = json.loads(
            (run_dir / "batch_manifest.json").read_text(encoding="utf-8")
        )
        print(f"Batch run generated: {run_dir}")
        print(f"  Requests: {batch_manifest['total_requests']:,}")
        print(f"  Shards: {len(batch_manifest['shards'])}")
        print(f"  Model: {batch_manifest['model_id']}")
        print(f"  reasoning_effort: {batch_manifest['reasoning_effort']}")
        print_batch_run_pre_submit_cost_estimate(run_dir)
        print("  No API requests have been submitted yet.")
        if args.batch_action == "generate":
            return 0
    else:
        run_dir = resolve_batch_run_dir(results_dir, args.batch_run_dir)

    batch_manifest = validate_batch_run_binding(
        run_dir,
        model_key=args.model,
        results_dir=results_dir,
    )

    queue_limit: int | None = None
    if args.batch_action in {"submit", "retry", "run"}:
        queue_limit = batch_queue_limit_for_model(
            batch_manifest["model_id"],
            usage_tier=args.batch_usage_tier,
            explicit_limit=args.batch_queue_token_limit,
        )

    def submit_and_wait_all_waves() -> None:
        assert queue_limit is not None
        while True:
            jobs = submit_pending_batch_shards(
                run_dir,
                max_queued_input_tokens=queue_limit,
            )
            print(
                "Batch wave: "
                f"submitted={jobs['submitted_this_call']}, "
                f"pending={jobs['pending_shards']}, "
                f"queue_bound={jobs['active_input_token_upper_bound']:,}/"
                f"{queue_limit:,} tokens"
            )
            nonterminal = [
                entry
                for entry in jobs.get("jobs", [])
                if entry.get("batch_id")
                and entry.get("status") not in {"completed", "failed", "expired", "cancelled"}
            ]
            if nonterminal:
                wait_for_batch_jobs(run_dir, poll_interval=args.poll_interval)
                continue
            if jobs["pending_shards"]:
                raise RuntimeError(
                    "Pending shards remain but no active/submitted wave can make "
                    "progress. Regenerate with fewer requests per shard or raise "
                    "the verified queue-token limit."
                )
            return

    if args.batch_action == "submit":
        assert queue_limit is not None
        print_batch_run_pre_submit_cost_estimate(run_dir)
        jobs = submit_pending_batch_shards(
            run_dir,
            max_queued_input_tokens=queue_limit,
        )
        print(
            f"Submitted this wave: {jobs['submitted_this_call']}; "
            f"pending shards: {jobs['pending_shards']}"
        )
        print(f"Recorded batch jobs: {len(jobs['jobs'])}")
        print(f"Batch run: {run_dir}")
        return 0

    if args.batch_action == "status":
        jobs = refresh_batch_jobs(run_dir)
        for entry in jobs["jobs"]:
            print(format_batch_job_status(entry))
        overall = aggregate_batch_progress(jobs["jobs"])
        print(
            "Overall: "
            f"processed={overall['processed']:,}/{overall['total']:,} "
            f"({overall['percent']:.1f}%), "
            f"completed={overall['completed']:,}, failed={overall['failed']:,}, "
            f"remaining={overall['remaining']:,}"
        )
        print(f"All terminal: {jobs['all_terminal']}")
        return 0

    if args.batch_action == "retry":
        jobs = refresh_batch_jobs(run_dir)
        if not jobs["all_terminal"]:
            raise SystemExit("Current batch jobs are not all terminal; retry later.")
        shards, retry_count, classification = create_retry_shards(run_dir)
        if classification["non_retryable"]:
            print(
                f"Non-retryable failed requests: {classification['non_retryable']} "
                "(inspect batch error JSONL)."
            )
        if classification.get("token_capped_incomplete"):
            print(
                f"Token-capped incomplete responses: "
                f"{classification['token_capped_incomplete']}. These are retained "
                "as incomplete and are not selectively rerun with a larger cap."
            )
        if retry_count == 0:
            print("No transient/missing responses to retry.")
            return 0
        print(f"Created {len(shards)} retry shard(s) for {retry_count:,} requests.")
        print_batch_run_pre_submit_cost_estimate(
            run_dir,
            shards=shards,
            estimate_scope="Retry pre-submit",
        )
        assert queue_limit is not None
        jobs = submit_pending_batch_shards(
            run_dir,
            max_queued_input_tokens=queue_limit,
        )
        print(
            f"Submitted this wave: {jobs['submitted_this_call']}; "
            f"pending retry shards: {jobs['pending_shards']}"
        )
        return 0

    if args.batch_action == "run":
        submit_and_wait_all_waves()
        for retry_round in range(args.batch_max_retries):
            shards, retry_count, classification = create_retry_shards(run_dir)
            if classification["non_retryable"]:
                print(
                    f"Retry audit found {classification['non_retryable']} "
                    "deterministic failure(s); these will not be resubmitted."
                )
            if classification.get("token_capped_incomplete"):
                print(
                    f"Retry audit found "
                    f"{classification['token_capped_incomplete']} token-capped "
                    "incomplete response(s); the fixed experimental cap is preserved."
                )
            if retry_count == 0:
                break
            print(
                f"Retry round {retry_round + 1}: {retry_count:,} missing "
                f"responses in {len(shards)} shard(s)."
            )
            submit_and_wait_all_waves()

    if args.batch_action in {"process", "run"}:
        summary = process_batch_run(run_dir)
        print(json.dumps(summary, indent=2))
        cost_paths = _write_cost_logs(results_dir, args.model, batch=True)
        if cost_paths is not None:
            print(f"Cost Log: {cost_paths[0]}")
            print(f"Cost Summary: {cost_paths[1]}")
        if args.hub_dataset and summary["result_files_written"] > 0:
            _push_results_to_hub(results_dir, args.model, args.hub_dataset)
        return 0

    raise ValueError(f"Unknown batch action: {args.batch_action!r}")


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
        default=str(COMPARISONS_DIR.relative_to(REPO_ROOT)),
        help="Directory with manifest + *_comparisons.json (default: "
        f"{COMPARISONS_DIR.relative_to(REPO_ROOT).as_posix()}). "
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
        default=str(LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help=(
            "Model-run root (default: outputs/). "
            f"Artifacts: <results-dir>/<model>/{LADDER_VS_COMPARISON_SUBDIR}/."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=str(CHECKPOINTS_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Checkpoints root (default: outputs/checkpoints). Relative to the repo root.",
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
        help=(
            f"Write outputs under <results-dir>/<model>/smoke_<model>/"
            f"{LADDER_VS_COMPARISON_SUBDIR}/ and matching checkpoints subdir."
        ),
    )
    parser.add_argument(
        "--estimate-cost-only",
        action="store_true",
        default=False,
        help=(
            "Print the selected live-run request/token/USD estimate and exit "
            "before constructing an agent or sending any API request."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help=(
            "Maximum variation sets processed concurrently. This is outer task "
            "parallelism, not the number of simultaneous HTTP requests."
        ),
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=None,
        help=(
            "Run-wide maximum in-flight live API attempts in this Python process. "
            "Defaults to the selected model's configured concurrency limit."
        ),
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=None,
        help=(
            "Optional run-wide request-start rate. Starts are evenly spaced to "
            "avoid bursts; separate terminals have separate limits."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help=(
            "Maximum attempts per live API request for retryable transport and "
            "transient HTTP failures (default: 5)."
        ),
    )
    batch_actions = parser.add_mutually_exclusive_group()
    batch_actions.add_argument(
        "--batch-generate",
        dest="batch_action",
        action="store_const",
        const="generate",
        help="Generate sharded OpenAI Batch API JSONL files without submitting them.",
    )
    batch_actions.add_argument(
        "--batch-submit",
        dest="batch_action",
        action="store_const",
        const="submit",
        help="Submit one queue-safe wave of not-yet-submitted shards.",
    )
    batch_actions.add_argument(
        "--batch-status",
        dest="batch_action",
        action="store_const",
        const="status",
        help="Refresh batch statuses and download available output/error files.",
    )
    batch_actions.add_argument(
        "--batch-process",
        dest="batch_action",
        action="store_const",
        const="process",
        help="Process downloaded batch outputs into the normal per-ladder result schema.",
    )
    batch_actions.add_argument(
        "--batch-retry",
        dest="batch_action",
        action="store_const",
        const="retry",
        help="Create and submit retry shards for transient/missing responses.",
    )
    batch_actions.add_argument(
        "--run-batch",
        dest="batch_action",
        action="store_const",
        const="run",
        help="Generate, submit, poll, retry transport failures, and process an OpenAI batch run.",
    )
    parser.set_defaults(batch_action=None)
    parser.add_argument(
        "--batch-run-dir",
        default=None,
        help=(
            "Existing batch-run directory for submit/status/process/retry. "
            "Defaults to the latest run for this model and smoke/full output scope."
        ),
    )
    parser.add_argument(
        "--max-requests-per-batch",
        type=int,
        default=MAX_BATCH_REQUESTS,
        help=(
            f"Maximum requests in each JSONL shard (default/API maximum: "
            f"{MAX_BATCH_REQUESTS:,}). Lower this if required by the account's batch queue limit."
        ),
    )
    parser.add_argument(
        "--batch-max-retries",
        type=int,
        default=2,
        help="Transient/missing-response retry rounds for --run-batch (default: 2).",
    )
    parser.add_argument(
        "--batch-usage-tier",
        type=int,
        choices=range(1, 6),
        default=None,
        metavar="{1,2,3,4,5}",
        help=(
            "OpenAI usage tier used to enforce the model's published Batch queue "
            "limit. Required for submit/retry/run unless an explicit queue-token "
            "limit is supplied."
        ),
    )
    parser.add_argument(
        "--batch-queue-token-limit",
        type=int,
        default=None,
        help=(
            "Verified account-specific Batch queue limit in input tokens. Overrides "
            "--batch-usage-tier; use only when the OpenAI dashboard differs from the "
            "published tier table."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status checks for --run-batch (default: 30).",
    )
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
                             "(e.g. 'your-org/your-dataset'). "
                             "Requires HF_TOKEN with write scope in the environment.")
    parser.add_argument(
        "--submit-hf-job",
        action="store_true",
        help="Submit this 7-tier run to Hugging Face Jobs instead of running locally.",
    )
    parser.add_argument("--image", default=None, help="Docker image tag for --submit-hf-job")
    parser.add_argument("--namespace", default=None, help="HF user/org namespace for --submit-hf-job")
    parser.add_argument("--flavor", default="h200x8", help="HF Jobs hardware flavor")
    parser.add_argument("--timeout", default="12h", help="HF Jobs timeout, e.g. 2h or 12h")
    parser.add_argument(
        "--model-volume",
        default=None,
        help=(
            "Experimental HF model repo mounted read-only for --submit-hf-job. "
            "Large sharded models may stall on the FUSE mount; omitting this "
            "option uses the recommended local /data cache."
        ),
    )
    parser.add_argument(
        "--model-volume-path",
        default="/data/model",
        help="Absolute in-container mount path for --model-volume (default: /data/model).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help=(
            "Optional dataset subdir for HF Jobs outputs. Defaults to the same "
            "outputs/<model>/ladder_vs_comparison_statements path."
        ),
    )
    parser.add_argument("--job-tag", default=None, help="Stable short tag for HF Jobs metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated HF job code and exit.")
    args = parser.parse_args()
    if args.start_from < 0:
        parser.error("--start-from must be >= 0")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be >= 1")
    if args.request_concurrency is not None and args.request_concurrency < 1:
        parser.error("--request-concurrency must be >= 1 when set")
    if args.requests_per_second is not None and args.requests_per_second <= 0:
        parser.error("--requests-per-second must be > 0 when set")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.max_variation_sets is not None and args.max_variation_sets < 1:
        parser.error("--max-variation-sets must be >= 1 when set")
    if not 1 <= args.max_requests_per_batch <= MAX_BATCH_REQUESTS:
        parser.error(
            f"--max-requests-per-batch must be between 1 and {MAX_BATCH_REQUESTS}"
        )
    if args.batch_max_retries < 0:
        parser.error("--batch-max-retries must be >= 0")
    if args.batch_queue_token_limit is not None and args.batch_queue_token_limit < 1:
        parser.error("--batch-queue-token-limit must be >= 1")
    if args.poll_interval < 1:
        parser.error("--poll-interval must be >= 1")
    if args.model_volume and not Path(args.model_volume_path).is_absolute():
        parser.error("--model-volume-path must be absolute")
    if args.batch_action and args.submit_hf_job:
        parser.error("OpenAI batch actions cannot be combined with --submit-hf-job")
    if args.estimate_cost_only and (args.batch_action or args.submit_hf_job):
        parser.error(
            "--estimate-cost-only is for direct live runs and cannot be combined "
            "with Batch or Hugging Face job actions"
        )

    args.model = canonical_model_key(args.model)
    try:
        cfg = get_model_config(args.model)
    except ValueError:
        cfg = None
    if args.max_tokens is None:
        args.max_tokens = cfg.max_tokens if cfg is not None else 10
    if args.model == "glm-45-base-logprobs" and args.model_variant == "instruct":
        args.model_variant = "base"
    if args.model.startswith("gpt-56"):
        reasoning_key = args.model.endswith("-thinking")
        generates_prompts = args.batch_action in {None, "generate", "run"}
        if generates_prompts and reasoning_key != args.with_reasoning:
            if reasoning_key:
                parser.error(
                    "GPT-5.6 *-thinking runs must include --with-reasoning to match "
                    "the manuscript's reasoning-formatted prompt plus reasoning_effort=high."
                )
            parser.error(
                "GPT-5.6 reasoning-off runs must omit --with-reasoning to match "
                "the manuscript's A/B-only prompt plus reasoning_effort=none."
            )

    # Preserve the historical OpenAI-batch prompt exactly when neither the CLI
    # nor MODEL_CONFIGS specifies a system message.  The live runner retains its
    # long-standing generic default.
    batch_system_message = (
        args.system_message
        if args.system_message is not None
        else (cfg.system_message if cfg is not None else None)
    )

    # Resolve live-run system message: CLI flag > ModelConfig > default.
    if args.system_message is not None:
        sys_msg = args.system_message
    elif cfg is not None and cfg.system_message is not None:
        sys_msg = cfg.system_message
    else:
        sys_msg = "You are a helpful assistant."

    if args.submit_hf_job:
        return submit_phase6b_hf_job(args, sys_msg)

    data_dir = resolve_under_parametric(args.data_dir)
    if args.manifest:
        manifest_path = resolve_under_parametric(args.manifest)
    else:
        manifest_path = discover_manifest_path(data_dir)
    results_root = resolve_under_parametric(args.results_dir)
    checkpoints_root = resolve_under_parametric(args.checkpoints_dir)
    results_dir = model_run_dir(args.model, results_root, smoke=args.smoke)
    checkpoints_dir = model_run_checkpoints_dir(
        args.model, checkpoints_root, smoke=args.smoke
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {manifest_path}")
    print(f"Data dir: {data_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    if args.smoke:
        print(
            f"  -> smoke scope under .../{smoke_run_subdir(args.model)}/"
            f"{LADDER_VS_COMPARISON_SUBDIR}/"
        )
    print()

    if args.estimate_cost_only:
        manifest = load_manifest(manifest_path)
        estimate_items = scoped_manifest_items(
            manifest,
            data_dir,
            args.variation_ids,
            start_from=args.start_from,
            max_variation_sets=args.max_variation_sets,
        )
        if args.resume:
            estimate_items = [
                item for item in estimate_items
                if not is_complete(results_dir, item["test_name"], args.model)
            ]
        if not estimate_items:
            print("No pending variation sets in the selected estimate scope.")
            print("Estimate only; no API request has been sent.")
            return 0
        estimate = print_phase6b_live_cost_estimate(
            args.model,
            run_items=estimate_items,
            data_dir=data_dir,
            num_trials=args.trials,
            with_reasoning=args.with_reasoning,
            max_tokens=args.max_tokens,
            system_message=sys_msg,
            include_prelaunch_smoke=not args.skip_smoke_test,
        )
        return 0 if estimate is not None else 2

    if args.batch_action:
        return run_openai_batch_action(
            args,
            data_dir=data_dir,
            manifest_path=manifest_path,
            results_dir=results_dir,
            batch_system_message=batch_system_message,
        )

    asyncio.run(
        run_phase6b_with_client_cleanup(
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
            request_concurrency=args.request_concurrency,
            requests_per_second=args.requests_per_second,
            max_retries=args.max_retries,
        )
    )


if __name__ == "__main__":
    main()
