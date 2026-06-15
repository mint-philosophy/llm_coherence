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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from llm_coherence.paths import REPO_ROOT
from llm_coherence.runtime.agents import create_agent, model_name_for_key
from llm_coherence.runtime.preflight_check import MODEL_COST_ESTIMATES, estimate_cost
from llm_coherence.runtime.templates import (
    comparison_prompt_template_default,
    comparison_prompt_template_reasoning_default,
)
from llm_coherence.runtime.utils import generate_responses, parse_responses_forced_choice

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_PARAMETRIC_ROOT = REPO_ROOT

# Import model config for extra_body / enable_cache lookup
from llm_coherence.config import MODEL_CONFIGS


RESULTS_SCHEMA_VERSION = "1.0"


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
    """Compute real cost from observed usage stats + preflight pricing table.

    Splits prompt tokens into uncached / cached portions so per-provider
    cache rates apply correctly:
        Anthropic ephemeral cache: 1.25× input on creation, 0.10× on read.
        OpenAI prompt caching:     0.10× input on cached tokens (per the
            preflight_check comment; verify against live pricing if the
            number ever ends up materially affecting a budget).
    Cache fields are mutually exclusive in practice (a given call goes
    through one provider) so summing them is safe.
    """
    try:
        prices = MODEL_COST_ESTIMATES.get(model_key)
        if not prices or not usage_stats:
            return None
        prompt = (usage_stats.get("prompt_tokens") or {}).get("total") or 0
        completion = (usage_stats.get("completion_tokens") or {}).get("total") or 0
        cache_create = (usage_stats.get("cache_creation_input_tokens") or {}).get("total") or 0
        cache_read = (usage_stats.get("cache_read_input_tokens") or {}).get("total") or 0
        oai_cached = (usage_stats.get("openai_cached_tokens") or {}).get("total") or 0
        # SDK `prompt_tokens` already counts cached tokens once — uncached is
        # what's left after subtracting any provider-cached portion.
        uncached = max(prompt - cache_create - cache_read - oai_cached, 0)
        in_p = prices["input"]
        return (
            uncached / 1_000_000 * in_p
            + cache_create / 1_000_000 * in_p * 1.25
            + cache_read / 1_000_000 * in_p * 0.10
            + oai_cached / 1_000_000 * in_p * 0.10
            + completion / 1_000_000 * prices["output"]
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
    """Aggregate per-call usage dicts into min/median/p95/max/total summaries.

    Used to record prompt/completion/reasoning token stats in results metadata so
    the real max_completion_tokens for thinking-on runs can be calibrated empirically.
    """
    def stats(raw):
        vals = sorted(v for v in raw if v is not None)
        if not vals:
            return None
        n = len(vals)
        return {
            "n": n,
            "min": vals[0],
            "median": vals[n // 2],
            "p95": vals[min(n - 1, int(n * 0.95))],
            "max": vals[-1],
            "total": sum(vals),
        }
    return {
        "calls_logged": len(entries),
        "prompt_tokens": stats(e.get("prompt_tokens") for e in entries),
        "completion_tokens": stats(e.get("completion_tokens") for e in entries),
        "reasoning_tokens": stats(e.get("reasoning_tokens") for e in entries),
        "cache_creation_input_tokens": stats(e.get("cache_creation_input_tokens") for e in entries),
        "cache_read_input_tokens": stats(e.get("cache_read_input_tokens") for e in entries),
        "openai_cached_tokens": stats(e.get("openai_cached_tokens") for e in entries),
    }


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
    """Load checkpoint if it exists and matches current run."""
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r") as f:
        return json.load(f)


def save_checkpoint(
    checkpoint_path: Path,
    test_name: str,
    model_key: str,
    num_trials: int,
    include_flipped: bool,
    comparisons_done: List[int],
    preferences: List[Dict[str, Any]],
    start_time: str,
    model_variant: str = "instruct",
    reasoning_mode: str = "none",
    temperature: float = 0.0,
    k_samples: int = 1,
    infrastructure: str = "openai_api",
) -> None:
    """Save progress so the run can be resumed."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_name": test_name,
        "model_key": model_key,
        "num_trials": num_trials,
        "include_flipped": include_flipped,
        "model_variant": model_variant,
        "reasoning_mode": reasoning_mode,
        "temperature": temperature,
        "k_samples": k_samples,
        "infrastructure": infrastructure,
        "comparisons_done": comparisons_done,
        "preferences": preferences,
        "start_time": start_time,
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


async def run_single_comparison(
    agent,
    comparison: Dict[str, Any],
    num_trials: int,
    include_flipped: bool,
    system_message: str,
    with_reasoning: bool,
    verbose: bool,
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
            system_message=system_message, K=1, timeout=10, verbose=verbose,
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
                system_message=system_message, K=1, timeout=10, verbose=verbose,
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

    # Original order (A=outcome_a, B=outcome_b): one prompt repeated num_trials times
    prompts_original = [
        build_prompt(text_a, text_b, with_reasoning, cache_structure=use_cache_blocks)
    ] * num_trials
    raw_original = await generate_responses(
        agent,
        prompts_original,
        system_message=system_message,
        K=1,
        timeout=10,
        verbose=verbose,
    )
    # Collapse to single list: raw_original[i] is list of 1 response for prompt i
    list_original = [
        (raw_original.get(i) or [None])[0] for i in range(len(prompts_original))
    ]
    parsed_original = parse_responses_forced_choice(
        {0: list_original},
        with_reasoning=with_reasoning,
        verbose=verbose,
    )[0]

    if include_flipped:
        prompts_flipped = [
            build_prompt(text_b, text_a, with_reasoning, cache_structure=use_cache_blocks)
        ] * num_trials
        raw_flipped = await generate_responses(
            agent,
            prompts_flipped,
            system_message=system_message,
            K=1,
            timeout=10,
            verbose=verbose,
        )
        list_flipped = [
            (raw_flipped.get(i) or [None])[0] for i in range(len(prompts_flipped))
        ]
        parsed_flipped = parse_responses_forced_choice(
            {0: list_flipped},
            with_reasoning=with_reasoning,
            verbose=verbose,
        )[0]
    else:
        parsed_flipped = []

    count_prefer_a, count_prefer_b = counts_from_responses(parsed_original, parsed_flipped)
    total = count_prefer_a + count_prefer_b
    prob_a = (count_prefer_a / total) if total else 0.0
    prob_b = (count_prefer_b / total) if total else 0.0

    result = {
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
        "count_prefer_a": count_prefer_a,
        "count_prefer_b": count_prefer_b,
        "prob_prefer_a": round(prob_a, 4),
        "prob_prefer_b": round(prob_b, 4),
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
) -> Dict[str, Any]:
    base = _PARAMETRIC_ROOT
    data_dir = data_dir or base / "data"
    results_dir = results_dir or base / "outputs"
    checkpoints_dir = checkpoints_dir or base / "checkpoints"

    comparisons = load_comparisons(data_dir, test_name, comparison_path)
    total_comparisons = len(comparisons)

    artifact_dir = artifact_dir_name_for_test(test_name)
    artifact_run = hashlib.sha1(f"{test_name}|{model_key}".encode("utf-8")).hexdigest()[:12]
    results_path = results_dir / artifact_dir / "results.json"
    checkpoint_path = checkpoints_dir / f"ckpt_{artifact_run}.json"

    start_time = datetime.utcnow().isoformat()
    preferences: List[Dict[str, Any]] = []
    comparisons_done: List[int] = []

    if resume and checkpoint_path.exists():
        ck = load_checkpoint(checkpoint_path)
        if ck and ck.get("test_name") == test_name and ck.get("model_key") == model_key:
            preferences = ck.get("preferences", [])
            comparisons_done = ck.get("comparisons_done", [])
            start_time = ck.get("start_time", start_time)
            if verbose:
                print(f"Resuming: {len(preferences)} comparisons already done.")

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

    agent = create_agent(
        model_key,
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
        enable_cache=enable_cache,
        k_samples=k_samples,
        quantization=quantization,
        base_timeout=60,
    )

    for idx in range(total_comparisons):
        if idx in comparisons_done:
            continue
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
        )
        preferences.append(pref)
        comparisons_done.append(idx)

        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                test_name,
                model_key,
                num_trials,
                include_flipped,
                comparisons_done,
                preferences,
                start_time,
                model_variant=model_variant,
                reasoning_mode=reasoning_mode,
                temperature=temperature if temperature is not None else 0.0,
                k_samples=k_samples,
                infrastructure=infrastructure,
            )
            if verbose:
                print(f"  Checkpoint saved ({len(preferences)} comparisons).")

    end_time = datetime.utcnow().isoformat()

    # Compute unparseable stats
    expected_per_comp = num_trials * (2 if include_flipped else 1)
    total_api_calls = total_comparisons * expected_per_comp
    unparseable_count = 0
    for pref in preferences:
        actual = pref["count_prefer_a"] + pref["count_prefer_b"]
        unparseable_count += expected_per_comp - actual

    elapsed_seconds = (
        datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
    ).total_seconds()

    agent_extra_body = getattr(agent, "extra_body", None) or None
    agent_retry_counts = getattr(agent, "retry_counts", None)
    usage_summary = _summarize_usage(getattr(agent, "usage_log", []))
    comparison_path = comparison_file_path(data_dir, test_name, comparison_path)
    prompt_template_used = (
        "comparison_prompt_template_reasoning_default"
        if with_reasoning
        else "comparison_prompt_template_default"
    )
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
            "temperature": temperature if temperature is not None else 0.0,
            "k_samples": k_samples,
            "infrastructure": infrastructure,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "quantization": quantization,
        },
        "metadata": {
            "start_time": start_time,
            "end_time": end_time,
            "total_comparisons": total_comparisons,
            "total_api_calls": total_api_calls,
            "unparseable_count": unparseable_count,
            "unparseable_rate": unparseable_count / total_api_calls if total_api_calls else 0.0,
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
    if rlog:
        traces_path = results_path.parent / "reasoning_traces.jsonl"
        # Don't swallow disk errors silently — losing traces silently is what
        # caused 139 of 146 nemotron-thinking ladders to ship without text data.
        with open(traces_path, "w") as f:
            for entry in rlog:
                f.write(json.dumps(entry) + "\n")
        if verbose:
            print(f"  Reasoning traces saved to {traces_path} ({len(rlog)} entries)")
        rlog.clear()

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
