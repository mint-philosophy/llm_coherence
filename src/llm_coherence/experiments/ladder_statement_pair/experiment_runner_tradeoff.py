"""
experiment_runner_tradeoff.py

Run forced-choice preference elicitation for trade-off consistency tests.
Loads comparison JSONs from the category-organized forced-choice input
directory, runs (A,B) and (B,A) trials per comparison, and saves results in the
format consumed by llm_coherence.analysis.analyze_7tier_coherence.
Supports checkpointing for resume after interruption.
"""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

from llm_coherence.paths import LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR, REPO_ROOT
from llm_coherence.runtime.agents import (
    AsyncRequestLimiter,
    create_agent,
    model_name_for_key,
)
from llm_coherence.runtime.preflight_check import MODEL_COST_ESTIMATES, estimate_cost
from llm_coherence.runtime.templates import (
    comparison_prompt_template_default,
    comparison_prompt_template_reasoning_default,
)
from llm_coherence.runtime.usage_cost import (
    actual_cost_from_usage_summary,
    estimate_cost_from_totals,
    infer_provider,
    resolve_rates,
    summarize_usage_log,
)
from llm_coherence.runtime.utils import generate_responses, parse_responses_forced_choice

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_PARAMETRIC_ROOT = REPO_ROOT

# Import model config for extra_body / enable_cache lookup
from llm_coherence.config import MODEL_CONFIGS


RESULTS_SCHEMA_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = "2.0"
_TOKEN_CAP_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "max_completion_tokens"}
)


def _git_sha() -> str | None:
    """Current repo commit SHA, or None if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _lookup_model_name_full(model_key: str) -> str | None:
    """Return the provider model name for a local model key."""
    return model_name_for_key(model_key)


def _estimate_cost(model_key: str, total_api_calls: int, with_reasoning: bool) -> float | None:
    """Best-effort cost estimate via the preflight table."""
    try:
        return estimate_cost(model_key, total_api_calls, with_reasoning)
    except Exception:
        return None


def _actual_cost(model_key: str, usage_stats: dict) -> float | None:
    """Best available USD for one ladder run (same preference order as 10a).

    1. Sum per-request ``cost_usd`` when the agent logged provider/live costs.
    2. Else tokens × ``MODEL_COST_ESTIMATES``.
    3. Else tokens × live OpenRouter published rates.
    """
    try:
        if not usage_stats:
            return None
        reported = actual_cost_from_usage_summary(usage_stats)
        if reported is not None:
            return reported

        prices = MODEL_COST_ESTIMATES.get(model_key)
        if not prices:
            mid = model_name_for_key(model_key)
            prices, _ = resolve_rates(infer_provider(mid), mid)
        if not prices:
            return None

        prompt = (usage_stats.get("prompt_tokens") or {}).get("total") or 0
        completion = (usage_stats.get("completion_tokens") or {}).get("total") or 0
        cache_create = (usage_stats.get("cache_creation_input_tokens") or {}).get("total") or 0
        cache_read = (usage_stats.get("cache_read_input_tokens") or {}).get("total") or 0
        oai_cached = (usage_stats.get("openai_cached_tokens") or {}).get("total") or 0
        if int(prompt) + int(completion) == 0:
            return None
        return estimate_cost_from_totals(
            prices,
            prompt_tokens=int(prompt),
            completion_tokens=int(completion),
            cache_creation_input_tokens=int(cache_create),
            cache_read_input_tokens=int(cache_read),
            openai_cached_tokens=int(oai_cached),
        )
    except Exception:
        return None


def _package_versions() -> dict:
    """Pin exact versions of libs that affect reproduction (per-request format, auth, retries)."""
    out = {"python": sys.version.split()[0]}
    for pkg in ("litellm", "openai", "anthropic"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", None)
        except Exception:
            out[pkg] = None
    return out


def _file_sha256(path: Path) -> str | None:
    """Hex sha256 of a file's bytes, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _host_info() -> dict:
    """Hostname + user for multi-machine runs."""
    try:
        return {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        }
    except Exception:
        return {}


def _summarize_usage(entries: list) -> dict:
    """Aggregate per-call usage (tokens + cost) for results metadata."""
    return summarize_usage_log(entries)


# Prompt building

def build_prompt(
    option_a_text: str,
    option_b_text: str,
    with_reasoning: bool = False,
    cache_structure: bool = False,
):
    """Build a single forced-choice prompt (Option A / Option B).

    If cache_structure is False (default) returns a single formatted string,
    matching the historical interface.

    If cache_structure is True, returns a list of content blocks with an
    ephemeral 1-hour cache_control marker on the stable prefix+Option-A block.
    Only the final Option-B block (which varies across the 30 comparisons
    within a set) is left uncached. This is Anthropic-specific structure and
    should only be passed for Anthropic/Claude models.
    """
    template = (
        comparison_prompt_template_reasoning_default
        if with_reasoning
        else comparison_prompt_template_default
    )
    if not cache_structure:
        return template.format(option_A=option_a_text, option_B=option_b_text)

    # Split the template so varying content (Option B) is in its own block.
    # Within a set the forward-direction user message has a stable prefix
    # (template preamble + Option A tier statement) that repeats across all
    # 30 comparisons for a given tier, so caching it pays off.
    prefix, rest = template.split("{option_A}", 1)
    middle, suffix = rest.split("{option_B}", 1)
    return [
        {
            "type": "text",
            "text": prefix + option_a_text + middle,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        {
            "type": "text",
            "text": option_b_text + suffix,
        },
    ]


_LADDER_TEST_PREFIX = "phase6b_variations_pruned_final_"


def category_for_test_name(test_name: str) -> str | None:
    """Return category slug for canonical phase6b test names."""
    if not test_name.startswith(_LADDER_TEST_PREFIX):
        return None
    short = test_name[len(_LADDER_TEST_PREFIX):]
    if "_" not in short:
        return None
    category, _ladder_id = short.rsplit("_", 1)
    return category


def comparison_file_path(
    data_dir: Path,
    test_name: str,
    comparison_path: Optional[Path] = None,
) -> Path:
    """Resolve comparison JSON path for category-organized or legacy flat data."""
    if comparison_path is not None:
        return comparison_path
    filename = f"{test_name}_comparisons.json"
    flat = data_dir / filename
    if flat.exists():
        return flat
    category = category_for_test_name(test_name)
    if category:
        return data_dir / category / filename
    return flat


# Loading and saving data

def load_comparisons(
    data_dir: Path,
    test_name: str,
    comparison_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load comparison list from category-organized or legacy flat data."""
    path = comparison_file_path(data_dir, test_name, comparison_path)
    if not path.exists():
        raise FileNotFoundError(f"Comparisons file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data["comparisons"]


def artifact_dir_name_for_test(test_name: str) -> str:
    """Readable, deterministic per-ladder artifact directory name.

    Canonical phase6b ladder tests (test_name
    "phase6b_variations_pruned_final_<ladder_id>") map to
    "phase6b_ladder_<ladder_id>" so the folder is self-describing when browsing
    the repo. Other test names fall back to their sanitized form. Must stay in
    sync with the identical copy in analyze_7tier_coherence.py.
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


def save_results(output_path: Path, payload: Dict[str, Any]) -> None:
    """Write results JSON atomically (tmp + rename)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    # Windows-safe atomic replace: Path.rename fails if destination exists.
    os.replace(tmp, output_path)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load a checkpoint payload; callers validate its exact run binding."""
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r") as f:
        return json.load(f)


def _run_fingerprint(run_config: Dict[str, Any]) -> str:
    """Stable identity for every scientific input and response-affecting setting."""
    encoded = json.dumps(
        run_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agent_telemetry(agent: Any) -> Dict[str, Any]:
    """Serializable telemetry that must survive a process restart."""
    return {
        "usage_log": list(getattr(agent, "usage_log", []) or []),
        "reasoning_log": list(getattr(agent, "reasoning_log", []) or []),
        "retry_counts": dict(getattr(agent, "retry_counts", {}) or {}),
    }


def _restore_agent_telemetry(agent: Any, telemetry: Dict[str, Any]) -> None:
    """Restore checkpointed billable usage, reasoning, and retry accounting."""
    if hasattr(agent, "usage_log"):
        agent.usage_log = list(telemetry.get("usage_log") or [])
    if hasattr(agent, "reasoning_log"):
        agent.reasoning_log = list(telemetry.get("reasoning_log") or [])
    if hasattr(agent, "retry_counts"):
        restored = dict(getattr(agent, "retry_counts", {}) or {})
        restored.update(telemetry.get("retry_counts") or {})
        agent.retry_counts = restored


def save_checkpoint(
    checkpoint_path: Path,
    run_config: Dict[str, Any],
    run_fingerprint: str,
    comparisons_done: List[int],
    preferences: List[Dict[str, Any]],
    start_time: str,
    *,
    partial_comparison: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save completed and per-trial in-flight progress."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_config": run_config,
        "run_fingerprint": run_fingerprint,
        "comparisons_done": comparisons_done,
        "preferences": preferences,
        "start_time": start_time,
        "partial_comparison": partial_comparison,
        "telemetry": telemetry or {},
    }
    tmp = checkpoint_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    # Windows-safe atomic replace: Path.rename fails if destination exists.
    os.replace(tmp, checkpoint_path)


# Preference elicitation

def counts_from_responses(
    original_parsed: List[str],
    flipped_parsed: List[str],
) -> tuple:
    """
    Compute count_prefer_a and count_prefer_b from (A,B) and (B,A) response lists.
    outcome_a = first option, outcome_b = second.
    Original order: A=outcome_a, B=outcome_b. Flipped order: A=outcome_b, B=outcome_a.
    """
    count_a_orig = sum(1 for r in original_parsed if r == "A")
    count_b_orig = sum(1 for r in original_parsed if r == "B")
    count_a_flip = sum(1 for r in flipped_parsed if r == "B")  # flipped: B means prefer first (outcome_a)
    count_b_flip = sum(1 for r in flipped_parsed if r == "A")  # flipped: A means prefer second (outcome_b)
    count_prefer_a = count_a_orig + count_a_flip
    count_prefer_b = count_b_orig + count_b_flip
    return count_prefer_a, count_prefer_b


def _live_call_timeout(agent: Any) -> float:
    """Use the model-configured timeout instead of the historical 10s cap."""
    try:
        return float(getattr(agent, "base_timeout"))
    except (AttributeError, TypeError, ValueError):
        return 10.0


def _missing_trial_records(
    direction: str,
    parsed: List[str],
    raw: List[str | None],
    outcomes: List[Dict[str, Any]],
    *,
    comparison_index: int = 0,
) -> List[Dict[str, Any]]:
    """Retain every non-parseable trial with its provider outcome metadata."""
    missing: List[Dict[str, Any]] = []
    for trial_index, parsed_value in enumerate(parsed):
        if parsed_value in ("A", "B"):
            continue
        outcome = outcomes[trial_index] if trial_index < len(outcomes) else {}
        status = outcome.get("status")
        finish_reason = outcome.get("finish_reason")
        if status == "token_capped" or finish_reason in _TOKEN_CAP_REASONS:
            reason = "token_capped"
        elif status in (None, "unknown", "completed"):
            reason = "unparseable"
        else:
            reason = str(status)
        raw_response = outcome.get("raw_response")
        if raw_response is None and trial_index < len(raw):
            raw_response = raw[trial_index]
        custom_id = (
            f"c{comparison_index:04d}-d{direction.lower()}-t{trial_index:03d}"
        )
        record = missing_result_record(
            direction=direction,
            trial_index=trial_index,
            custom_id=custom_id,
            reason=reason,
            finish_reason=finish_reason,
            response_status=str(status or "unknown"),
            raw_response=raw_response,
            error=outcome.get("error"),
            attempts=outcome.get("attempts", 1),
            transport_retries=outcome.get("transport_retries", 0),
        )
        missing.append(record)
    return missing


def missing_result_record(
    *,
    direction: str,
    trial_index: int,
    custom_id: str,
    reason: str,
    finish_reason: Any,
    response_status: str,
    raw_response: Any,
    error: Any,
    attempts: int,
    transport_retries: int,
) -> Dict[str, Any]:
    """Build the shared current-schema missing-result record for live and Batch."""
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "direction": direction,
        "trial_index": trial_index,
        "custom_id": custom_id,
        "reason": reason,
        "finish_reason": finish_reason,
        "response_status": response_status,
        "raw_response": raw_response,
        "error": error,
        "attempts": int(attempts),
        "transport_retries": int(transport_retries),
    }


async def _run_live_direction(
    *,
    agent: Any,
    prompt: Any,
    direction: str,
    num_trials: int,
    comparison_index: int,
    system_message: str,
    verbose: bool,
    trial_state: Dict[str, Any],
    checkpoint_trial_state: Optional[Callable[[Dict[str, Any]], None]],
) -> tuple[List[str | None], List[Dict[str, Any]]]:
    """Run only unresolved transport trials and durably merge their outcomes."""
    records = trial_state.setdefault("directions", {}).setdefault(
        direction,
        [None] * num_trials,
    )
    if len(records) != num_trials:
        raise ValueError(
            f"Checkpoint trial count mismatch for {direction}: "
            f"{len(records)} != {num_trials}"
        )

    pending_indices = [
        trial_index
        for trial_index, record in enumerate(records)
        if record is None
        or (record.get("outcome") or {}).get("status") == "transport_failure"
    ]
    if pending_indices:
        reasoning_before = len(getattr(agent, "reasoning_log", []) or [])
        generated = await generate_responses(
            agent,
            [prompt] * len(pending_indices),
            system_message=system_message,
            K=1,
            timeout=_live_call_timeout(agent),
            verbose=verbose,
        )
        new_outcomes = list(getattr(agent, "last_completion_outcomes", []) or [])
        new_responses = [
            (generated.get(index) or [None])[0]
            for index in range(len(pending_indices))
        ]

        reasoning_log = getattr(agent, "reasoning_log", []) or []
        for entry in reasoning_log[reasoning_before:]:
            message_index = entry.get("message_idx")
            if isinstance(message_index, int) and 0 <= message_index < len(pending_indices):
                trial_index = pending_indices[message_index]
                entry["custom_id"] = (
                    f"c{comparison_index:04d}-d{direction.lower()}-"
                    f"t{trial_index:03d}"
                )

        for local_index, trial_index in enumerate(pending_indices):
            prior = records[trial_index] or {}
            prior_outcome = prior.get("outcome") or {}
            outcome = (
                dict(new_outcomes[local_index])
                if local_index < len(new_outcomes)
                else {
                    "status": "transport_failure",
                    "error": "completion outcome was not returned",
                    "attempts": 1,
                    "transport_retries": 0,
                }
            )
            outcome["attempts"] = int(prior_outcome.get("attempts", 0)) + int(
                outcome.get("attempts", 1)
            )
            outcome["transport_retries"] = int(
                prior_outcome.get("transport_retries", 0)
            ) + int(outcome.get("transport_retries", 0))
            custom_id = (
                f"c{comparison_index:04d}-d{direction.lower()}-t{trial_index:03d}"
            )
            records[trial_index] = {
                "custom_id": custom_id,
                "response": new_responses[local_index],
                "outcome": outcome,
            }

        if checkpoint_trial_state is not None:
            checkpoint_trial_state(trial_state)

    responses = [
        (record or {}).get("response")
        for record in records
    ]
    outcomes = [
        dict((record or {}).get("outcome") or {})
        for record in records
    ]
    return responses, outcomes


async def run_single_comparison(
    agent,
    comparison: Dict[str, Any],
    num_trials: int,
    include_flipped: bool,
    system_message: str,
    with_reasoning: bool,
    verbose: bool,
    *,
    comparison_index: int = 0,
    trial_state: Optional[Dict[str, Any]] = None,
    checkpoint_trial_state: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run forced-choice elicitation for one comparison: num_trials (A,B) and
    num_trials (B,A) if include_flipped, then aggregate counts and probs.
    """
    outcome_a = comparison["outcome_a"]
    outcome_b = comparison["outcome_b"]
    text_a = outcome_a["text"]
    text_b = outcome_b["text"]

    # Use Anthropic content-block prompt caching when the agent is Claude and
    # caching is enabled. Non-Anthropic agents get a plain formatted string
    # (historical behavior, unchanged).
    agent_model = getattr(agent, "model", "").lower()
    use_cache_blocks = (
        getattr(agent, "enable_cache", False)
        and ("claude" in agent_model or "anthropic" in agent_model)
    )

    # Logprob path: a single forward pass per direction yields the full P(A)/P(B)
    # distribution from the answer-position softmax. K-sampling is unnecessary;
    # we fall back to the schema fields downstream code expects.
    if getattr(agent, "uses_logits", False):
        prompts_original = [
            build_prompt(text_a, text_b, with_reasoning, cache_structure=False)
        ]
        raw_original = await generate_responses(
            agent, prompts_original,
            system_message=system_message,
            K=1,
            timeout=_live_call_timeout(agent),
            verbose=verbose,
        )
        dist_orig = (raw_original.get(0) or [None])[0]
        if not isinstance(dist_orig, dict):
            raise RuntimeError(
                f"Logprob agent returned non-dict response: {type(dist_orig)}"
            )
        p_orig_a = float(dist_orig.get("A", 0.5))

        if include_flipped:
            prompts_flipped = [
                build_prompt(text_b, text_a, with_reasoning, cache_structure=False)
            ]
            raw_flipped = await generate_responses(
                agent, prompts_flipped,
                system_message=system_message,
                K=1,
                timeout=_live_call_timeout(agent),
                verbose=verbose,
            )
            dist_flip = (raw_flipped.get(0) or [None])[0]
            if not isinstance(dist_flip, dict):
                raise RuntimeError(
                    f"Logprob agent returned non-dict response: {type(dist_flip)}"
                )
            # In the flipped prompt, "B" corresponds to outcome_a.
            p_flip_a = float(dist_flip.get("B", 0.5))
            prob_a = (p_orig_a + p_flip_a) / 2.0
        else:
            prob_a = p_orig_a
        prob_b = 1.0 - prob_a

        # Schema-compat counts: round to integer votes against num_trials so
        # downstream aggregation/checkpointing keeps the same shape.
        count_a = int(round(prob_a * num_trials))
        count_b = num_trials - count_a

        return {
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            "count_prefer_a": count_a,
            "count_prefer_b": count_b,
            "prob_prefer_a": round(prob_a, 4),
            "prob_prefer_b": round(prob_b, 4),
        }

    active_trial_state = trial_state if trial_state is not None else {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "directions": {},
    }
    if active_trial_state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Partial-comparison checkpoint schema mismatch; refusing to mix "
            "unvalidated trial records."
        )
    list_original, outcomes_original = await _run_live_direction(
        agent=agent,
        prompt=build_prompt(
            text_a,
            text_b,
            with_reasoning,
            cache_structure=use_cache_blocks,
        ),
        direction="AB",
        num_trials=num_trials,
        comparison_index=comparison_index,
        system_message=system_message,
        verbose=verbose,
        trial_state=active_trial_state,
        checkpoint_trial_state=checkpoint_trial_state,
    )
    parsed_original = parse_responses_forced_choice(
        {0: list_original},
        with_reasoning=with_reasoning,
        verbose=verbose,
    )[0]

    if include_flipped:
        list_flipped, outcomes_flipped = await _run_live_direction(
            agent=agent,
            prompt=build_prompt(
                text_b,
                text_a,
                with_reasoning,
                cache_structure=use_cache_blocks,
            ),
            direction="BA",
            num_trials=num_trials,
            comparison_index=comparison_index,
            system_message=system_message,
            verbose=verbose,
            trial_state=active_trial_state,
            checkpoint_trial_state=checkpoint_trial_state,
        )
        parsed_flipped = parse_responses_forced_choice(
            {0: list_flipped},
            with_reasoning=with_reasoning,
            verbose=verbose,
        )[0]
    else:
        parsed_flipped = []
        outcomes_flipped = []

    count_prefer_a, count_prefer_b = counts_from_responses(parsed_original, parsed_flipped)
    total = count_prefer_a + count_prefer_b
    expected = num_trials * (2 if include_flipped else 1)
    missing_records = _missing_trial_records(
        "AB",
        parsed_original,
        list_original,
        outcomes_original,
        comparison_index=comparison_index,
    ) + _missing_trial_records(
        "BA",
        parsed_flipped,
        list_flipped if include_flipped else [],
        outcomes_flipped,
        comparison_index=comparison_index,
    )
    missing_count = expected - total
    missing_by_reason = Counter(item["reason"] for item in missing_records)
    prob_a = (count_prefer_a / total) if total else None
    prob_b = (count_prefer_b / total) if total else None

    result = {
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
        "count_prefer_a": count_prefer_a,
        "count_prefer_b": count_prefer_b,
        "prob_prefer_a": round(prob_a, 4) if prob_a is not None else None,
        "prob_prefer_b": round(prob_b, 4) if prob_b is not None else None,
        "expected_trials": expected,
        "parseable_trials": total,
        "missing_trials": missing_count,
        "missing_by_reason": dict(sorted(missing_by_reason.items())),
        "missing_responses": missing_records,
        "probability_denominator": "parseable_trials",
        "prob_prefer_a_bounds": {
            "lower_missing_prefer_b": round(count_prefer_a / expected, 4),
            "upper_missing_prefer_a": round((count_prefer_a + missing_count) / expected, 4),
        },
    }
    if with_reasoning:
        result["raw_responses_original"] = list_original
        result["raw_responses_flipped"] = list_flipped if include_flipped else []
    return result


# Main experiment loop

async def run_experiment(
    test_name: str,
    model_key: str,
    num_trials: int = 30,
    data_dir: Optional[Path] = None,
    comparison_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    include_flipped: bool = True,
    with_reasoning: bool = False,
    max_tokens: Optional[int] = None,
    system_message: str = "You are a helpful assistant.",
    checkpoint_interval: int = 1,
    resume: bool = True,
    verbose: bool = True,
    # Extended metadata for multi-model experiment
    model_variant: str = "instruct",
    reasoning_mode: str = "none",
    temperature: Optional[float] = None,
    k_samples: int = 1,
    infrastructure: str = "openai_api",
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    quantization: Optional[str] = None,
    request_limiter: AsyncRequestLimiter | None = None,
    max_retries: int = 5,
) -> Dict[str, Any]:
    base = _PARAMETRIC_ROOT
    data_dir = data_dir or base / "data"
    results_dir = results_dir or LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR
    checkpoints_dir = checkpoints_dir or base / "checkpoints"

    comparisons = load_comparisons(data_dir, test_name, comparison_path)
    total_comparisons = len(comparisons)
    comparison_path = comparison_file_path(data_dir, test_name, comparison_path)

    artifact_dir = artifact_dir_name_for_test(test_name)
    artifact_run = hashlib.sha1(f"{test_name}|{model_key}".encode("utf-8")).hexdigest()[:12]
    results_path = results_dir / artifact_dir / "results.json"
    checkpoint_path = checkpoints_dir / f"ckpt_{artifact_run}.json"

    start_time = datetime.utcnow().isoformat()
    preferences: List[Dict[str, Any]] = []
    comparisons_done: List[int] = []
    partial_comparison: Optional[Dict[str, Any]] = None
    checkpoint_telemetry: Dict[str, Any] = {}

    if max_tokens is None:
        max_tokens = 10

    # Look up extra_body, enable_cache, and system_message from MODEL_CONFIGS
    extra_body = None
    enable_cache = False
    model_cfg = MODEL_CONFIGS.get(model_key)
    if model_cfg is not None:
        extra_body = model_cfg.extra_body
        enable_cache = model_cfg.enable_cache
        if model_cfg.system_message is not None:
            system_message = model_cfg.system_message

    effective_temperature = temperature if temperature is not None else 0.0
    agent_concurrency_limit = model_cfg.concurrency_limit if model_cfg else 50
    prompt_template_used = (
        "comparison_prompt_template_reasoning_default"
        if with_reasoning
        else "comparison_prompt_template_default"
    )
    run_config = {
        "test_name": test_name,
        "model_key": model_key,
        "model_name_full": _lookup_model_name_full(model_key),
        "model_variant": model_variant,
        "reasoning_mode": reasoning_mode,
        "num_trials": num_trials,
        "include_flipped": include_flipped,
        "with_reasoning": with_reasoning,
        "max_tokens": max_tokens,
        "temperature": effective_temperature,
        "k_samples": k_samples,
        "infrastructure": infrastructure,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "quantization": quantization,
        "max_retries": max_retries,
        "system_message": system_message,
        "extra_body": extra_body,
        "enable_cache": enable_cache,
        "base_timeout": model_cfg.base_timeout if model_cfg else None,
        "agent_concurrency_limit": agent_concurrency_limit,
        "retry_transport_only": True,
        "prompt_template_used": prompt_template_used,
        "comparison_file_sha256": _file_sha256(comparison_path),
        "comparison_count": total_comparisons,
        "request_concurrency_limit": (
            request_limiter.max_concurrency if request_limiter else None
        ),
        "request_starts_per_second": (
            request_limiter.requests_per_second if request_limiter else None
        ),
    }
    run_fingerprint = _run_fingerprint(run_config)

    if resume and checkpoint_path.exists():
        ck = load_checkpoint(checkpoint_path)
        if ck:
            checkpoint_fingerprint = ck.get("run_fingerprint")
            checkpoint_config = ck.get("run_config")
            checkpoint_config_fingerprint = (
                _run_fingerprint(checkpoint_config)
                if isinstance(checkpoint_config, dict)
                else None
            )
            if (
                checkpoint_fingerprint != run_fingerprint
                or checkpoint_config != run_config
                or checkpoint_config_fingerprint != checkpoint_fingerprint
            ):
                raise ValueError(
                    "Checkpoint run configuration/input fingerprint mismatch; "
                    "refusing to mix experimental conditions. Start with "
                    "--no-resume or use a matching checkpoint."
                )
            if ck.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    "Checkpoint schema mismatch; refusing to resume an "
                    "unvalidated legacy checkpoint."
                )
            preferences = list(ck.get("preferences") or [])
            comparisons_done = list(ck.get("comparisons_done") or [])
            start_time = ck.get("start_time", start_time)
            partial_comparison = ck.get("partial_comparison")
            checkpoint_telemetry = dict(ck.get("telemetry") or {})
            if verbose:
                partial_note = " with an in-flight comparison" if partial_comparison else ""
                print(
                    f"Resuming: {len(preferences)} comparisons already done"
                    f"{partial_note}."
                )

    agent = create_agent(
        model_key,
        temperature=effective_temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        enable_cache=enable_cache,
        concurrency_limit=agent_concurrency_limit,
        max_retries=max_retries,
        request_limiter=request_limiter,
        k_samples=k_samples,
        quantization=quantization,
        # Experiment policy: retry network transport failures only. Provider
        # errors, empty/capped outputs, and parse failures remain observed
        # outcomes and are never selectively regenerated.
        retry_transport_only=True,
    )
    _restore_agent_telemetry(agent, checkpoint_telemetry)

    for idx in range(total_comparisons):
        if idx in comparisons_done:
            continue
        if partial_comparison is not None:
            partial_index = partial_comparison.get("comparison_index")
            if partial_index != idx:
                raise ValueError(
                    "Checkpoint partial-comparison index is inconsistent with "
                    f"completed progress: expected {idx}, found {partial_index}."
                )
            trial_state = dict(partial_comparison.get("trial_state") or {})
        else:
            trial_state = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "directions": {},
            }

        def checkpoint_trial_state(state: Dict[str, Any]) -> None:
            nonlocal partial_comparison
            partial_comparison = {
                "comparison_index": idx,
                "trial_state": state,
            }
            save_checkpoint(
                checkpoint_path,
                run_config,
                run_fingerprint,
                comparisons_done,
                preferences,
                start_time,
                partial_comparison=partial_comparison,
                telemetry=_agent_telemetry(agent),
            )

        comp = comparisons[idx]
        if verbose:
            print(f"Comparison {idx + 1}/{total_comparisons} ...")
        pref = await run_single_comparison(
            agent,
            comp,
            num_trials=num_trials,
            include_flipped=include_flipped,
            system_message=system_message,
            with_reasoning=with_reasoning,
            verbose=verbose,
            comparison_index=idx,
            trial_state=trial_state,
            checkpoint_trial_state=checkpoint_trial_state,
        )
        retryable_missing = int(
            (pref.get("missing_by_reason") or {}).get("transport_failure", 0)
        )
        if retryable_missing:
            checkpoint_trial_state(trial_state)
            raise RuntimeError(
                f"{retryable_missing} retryable infrastructure response(s) "
                f"remained missing for comparison {idx + 1}/{total_comparisons}; "
                "successful trials and telemetry were checkpointed, and resume "
                "will retry only failed transport IDs"
            )
        preferences.append(pref)
        comparisons_done.append(idx)
        partial_comparison = None
        # Every completed comparison is durable. checkpoint_interval controls
        # progress reporting only; a larger value must never expose paid trials
        # to duplicate billing after restart.
        save_checkpoint(
            checkpoint_path,
            run_config,
            run_fingerprint,
            comparisons_done,
            preferences,
            start_time,
            partial_comparison=None,
            telemetry=_agent_telemetry(agent),
        )
        if verbose and (idx + 1) % checkpoint_interval == 0:
            print(f"  Checkpoint saved ({len(preferences)} comparisons).")

    end_time = datetime.utcnow().isoformat()

    # Compute unparseable stats
    expected_per_comp = num_trials * (2 if include_flipped else 1)
    total_api_calls = total_comparisons * expected_per_comp
    unparseable_count = 0
    missing_by_reason: Counter[str] = Counter()
    for pref in preferences:
        actual = pref["count_prefer_a"] + pref["count_prefer_b"]
        unparseable_count += expected_per_comp - actual
        missing_by_reason.update(pref.get("missing_by_reason") or {})

    elapsed_seconds = (
        datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
    ).total_seconds()

    agent_extra_body = getattr(agent, "extra_body", None) or None
    agent_retry_counts = getattr(agent, "retry_counts", None)
    usage_summary = _summarize_usage(getattr(agent, "usage_log", []) or [])
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "config": {
            "test_name": test_name,
            "model_key": model_key,
            "model_variant": model_variant,
            "is_base_model": model_variant == "base",
            "reasoning_mode": reasoning_mode,
            "num_trials": num_trials,
            "include_flipped": include_flipped,
            "with_reasoning": with_reasoning,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
            "k_samples": k_samples,
            "infrastructure": infrastructure,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "quantization": quantization,
            "request_concurrency_limit": (
                request_limiter.max_concurrency if request_limiter else None
            ),
            "request_starts_per_second": (
                request_limiter.requests_per_second if request_limiter else None
            ),
            "max_retries": max_retries,
        },
        "metadata": {
            "start_time": start_time,
            "end_time": end_time,
            "total_comparisons": total_comparisons,
            "total_api_calls": total_api_calls,
            "unparseable_count": unparseable_count,
            "unparseable_rate": unparseable_count / total_api_calls if total_api_calls else 0.0,
            "missing_response_count": unparseable_count,
            "missing_response_counts": dict(sorted(missing_by_reason.items())),
            "run_status": "complete" if unparseable_count == 0 else "complete_with_missing",
            "response_policy": {
                "token_caps": "retain_as_missing_never_retry",
                "unparseable_outputs": "retain_as_missing_never_retry",
                "empty_outputs": "retain_as_missing_never_retry",
                "retries": "transport_and_transient_http_failures_only",
                "max_attempts_per_request": max_retries,
                "fixed_max_tokens": max_tokens,
            },
            "elapsed_seconds": round(elapsed_seconds, 1),
            "usage_stats": usage_summary,
            "model_name_full": _lookup_model_name_full(model_key),
            "extra_body": agent_extra_body,
            "estimated_cost_usd": _estimate_cost(model_key, total_api_calls, with_reasoning),
            "actual_cost_usd": _actual_cost(model_key, usage_summary),
            "git_commit_sha": _git_sha(),
            "package_versions": _package_versions(),
            "prompt_template_used": prompt_template_used,
            "system_message": system_message,
            "comparison_file_sha256": _file_sha256(comparison_path),
            "retry_counts": agent_retry_counts,
            **_host_info(),
        },
        "preferences": preferences,
    }
    save_results(results_path, payload)

    # Dump reasoning traces (where the provider exposes them) to a sidecar
    # JSONL file alongside the results. Only fires when the agent collected
    # any reasoning content. After dumping, clear the agent's reasoning_log
    # so the next set starts fresh and traces don't pollute across sets.
    rlog = getattr(agent, "reasoning_log", None)
    traces_path = results_path.parent / "reasoning_traces.jsonl"
    if rlog:
        # Don't swallow disk errors silently — losing traces silently is what
        # caused 139 of 146 nemotron-thinking ladders to ship without text data.
        with open(traces_path, "w") as f:
            for entry in rlog:
                f.write(json.dumps(entry) + "\n")
        if verbose:
            print(f"  Reasoning traces saved to {traces_path} ({len(rlog)} entries)")
        rlog.clear()
    elif not with_reasoning and traces_path.exists():
        # A previous interrupted or misconfigured run may have left a generic
        # telemetry sidecar with null reasoning values. Reasoning-off outputs
        # must not advertise a reasoning artifact.
        traces_path.unlink()

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if verbose:
        print(f"Results saved to {results_path}")
    return payload


# Command-line interface

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run forced-choice preference elicitation and save results for "
            "llm_coherence.analysis.analyze_7tier_coherence"
        ),
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Test name (e.g. agent_tradeoff_10to1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key (e.g. gpt-4o-mini-openrouter)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of trials per (A,B) and per (B,A) (default: 30)",
    )
    parser.add_argument(
        "--no-flipped",
        action="store_true",
        help="Disable flipped (B,A) trials",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Use reasoning prompt format",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N comparisons (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="instruct",
        help="Model variant: base, instruct, hybrid, hybrid_thinking, reasoning (default: instruct)",
    )
    parser.add_argument(
        "--reasoning-mode",
        type=str,
        default="none",
        help="Reasoning mode: none, cot, thinking (default: none)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature (default: use model config)",
    )
    parser.add_argument(
        "--k-samples",
        type=int,
        default=1,
        help="Samples per prompt for base models (default: 1)",
    )
    parser.add_argument(
        "--infrastructure",
        type=str,
        default="openai_api",
        help="Infrastructure: openai_api, anthropic_api, openrouter, hf_jobs, local (default: openai_api)",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=None,
        help="GPU type for self-hosted runs (e.g. H200, H100)",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs for self-hosted runs",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method: fp8, awq, gptq, or omit for BF16",
    )
    args = parser.parse_args()

    asyncio.run(
        run_experiment(
            test_name=args.test,
            model_key=args.model,
            num_trials=args.trials,
            include_flipped=not args.no_flipped,
            with_reasoning=args.reasoning,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.no_resume,
            verbose=not args.quiet,
            model_variant=args.model_variant,
            reasoning_mode=args.reasoning_mode,
            temperature=args.temperature,
            k_samples=args.k_samples,
            infrastructure=args.infrastructure,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            quantization=args.quantization,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
