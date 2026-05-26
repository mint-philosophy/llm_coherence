#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-ladder ranking validation for phase6b variation ladders.

For each ladder, presents all seven tier statements in randomized order (neutral
labels only — no tier numbers or least/most labels) and asks the model to rank
them from least to most preferable. Scores recovery against the intended T1→T7
preference order (T1 = least preferable, T7 = most preferable in the dataset).

Metrics reference (per ladder, when parse succeeds)
----------------------------------------------------
Ground truth: tier indices 0..6 = T1 (least preferable) .. T7 (most preferable).

exact_match / matches_intended_t1_t7_order
    Recovered least→most order equals [T1, T2, ..., T7] exactly.
    Interpretation: 1.0 = perfect recovery of the ladder's intended preference scale.

kendall_tau
    Kendall's τ between each tier's assigned rank and its true rank (0..6).
    Interpretation: 1 = all pairwise orderings agree with ground truth; 0 ≈ random; negative = mostly reversed.

spearman_rho
    Spearman rank correlation between predicted and true rank positions.
    Interpretation: 1 = perfect rank alignment; sensitive to any displaced tier (similar read to τ for 7 items).

pairwise_inversion_errors / max_pairwise_errors
    Count of tier pairs ordered opposite to ground truth (max = 21 for 7 tiers).
    Interpretation: 0 = no mistakes; higher = more local or global ordering mistakes.

exact_position_matches
    Number of tiers placed in exactly the correct rank slot (0–7).
    Interpretation: 7 = every tier in the right position; 5+ often means only minor swaps.

within_one_position
    Tiers whose assigned rank is within ±1 of the correct rank.
    Interpretation: High count = errors are mostly adjacent-tier swaps, not wild reordering.

endpoints_correct
    Both T1 ranked least and T7 ranked most.
    Interpretation: True = correct extremes even if middle tiers are shuffled.

least_preferable_correct / most_preferable_correct
    Bottom of ranking is T1 / top of ranking is T7.
    Interpretation: Check whether failures are endpoint-only vs spread through the ladder.

status / ranking_decision (in ranking_validation_details.json)
    MATCH + OK: exact T1→T7. MISMATCH + FLAG: parsed but wrong order. UNPARSEABLE: invalid JSON ranking.

Dataset-level summary (ranking_validation_summary.json)
-------------------------------------------------------
n_ladders, counts (MATCH / MISMATCH / UNPARSEABLE)
    How many ladders fell in each bucket.
    Interpretation: Quick yield and failure-mode mix for the run.

n_match, n_flagged_mismatch, match_fraction, flagged_mismatch_fraction
    Counts and fractions for exact recovery vs flagged wrong order.
    Interpretation: match_fraction is the headline “how often the model got the full ladder right.”

aggregate_metrics.mean_kendall_tau
    Mean τ over parsed ladders only (UNPARSEABLE excluded).
    Interpretation: Average ordinal agreement; 1.0 = every parsed ladder perfect.

aggregate_metrics.perfect_tau_count
    Parsed ladders with τ = 1.0 (equivalent to exact_match for strict rankings).
    Interpretation: Same as n_match when all ladders parse successfully.

aggregate_metrics.high_tau_ge_0_9 / low_tau_lt_0_5
    Parsed ladders with τ ≥ 0.9 or τ < 0.5.
    Interpretation: Rough “mostly right” vs “mostly wrong” counts (not hypothesis tests).

Cost fields (ranking_cost_log.json, ranking_cost_summary.json)
    Token usage and estimated USD per call and total — not preference quality metrics.

Usage:
  # GPT-5.5 with reasoning on (parallel batches, auto-tuned limits)
  python ladder_validation_tests/full_ladder_ranking_pruning.py --model gpt-55-openai --reasoning on --prune-policy strict_rescue_inv1
  
  # Smoke test (saves under ladder_validation_tests/.../ranking/smoke_gpt55/)
  python ladder_validation_tests/full_ladder_ranking_pruning.py --model gpt-55-openai --reasoning on --max-ladders 2

  # Recompute summary from saved JSONL (no API)
  python ladder_validation_tests/full_ladder_ranking_pruning.py --model gpt-55-openai --analyze-only

  # Write kept ladders JSON only (no API; uses ranking_mismatch + unparseable IDs)
  python ladder_validation_tests/full_ladder_ranking_pruning.py --model gpt-55-openai --write-pruned-variations-only

Outputs are saved under:
  data/ladder_validation_tests_outputs/within_ladder_validation_ranking/<model_key>/
  data/ladder_validation_tests_outputs/within_ladder_validation_ranking/smoke_<model>/
  e.g. .../within_ladder_validation_ranking/smoke_gpt55/

  - ranking_raw_responses.jsonl
  - ranking_cost_log.json
  - ranking_cost_summary.json
  - ranking_validation_summary.json
  - ranking_validation_details.json
  - ranking_match_ladder_ids.json
  - ranking_mismatch_ladder_ids.json
  - ranking_unparseable_ladder_ids.json
  - ranking_pruned_ladder_ids.json

Writes data/ladder_validation_tests_outputs/phase6b_variations_ranking_pruned.json after a full run:
  ladders removed = MISMATCH (FLAG) + UNPARSEABLE; kept = exact T1→T7 MATCH only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import string
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PARAMETRIC_ROOT = Path(__file__).resolve().parent.parent
if str(_PARAMETRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARAMETRIC_ROOT))

from ladder_validation_tests.ladder_validation_paths import (
    BASE_DIR,
    DATA_DIR,
    RANKING_DIR_NAME,
    RANKING_OUTPUT_DIR,
    PRUNED_RANKING_FILENAME,
    PRUNED_RANKING_PATH,
    migrate_all_legacy_validation_layout,
    migrate_run_dir,
    normalize_cli_output_dir as normalize_validation_output_dir,
    resolve_pruned_variations_output as resolve_pruned_variations_output_path,
)

DEFAULT_INPUT = DATA_DIR / "phase6b_variations.json"
DEFAULT_OUTPUT_DIR = RANKING_OUTPUT_DIR

RAW_JSONL_NAME = "ranking_raw_responses.jsonl"
COST_JSON_NAME = "ranking_cost_log.json"
COST_SUMMARY_JSON_NAME = "ranking_cost_summary.json"
SUMMARY_JSON_NAME = "ranking_validation_summary.json"
DETAILS_JSON_NAME = "ranking_validation_details.json"
MATCH_IDS_JSON_NAME = "ranking_match_ladder_ids.json"
MISMATCH_IDS_JSON_NAME = "ranking_mismatch_ladder_ids.json"
UNPARSEABLE_IDS_JSON_NAME = "ranking_unparseable_ladder_ids.json"
PRUNED_IDS_JSON_NAME = "ranking_pruned_ladder_ids.json"
PRUNED_VARIATIONS_FILENAME = PRUNED_RANKING_FILENAME

PRUNE_POLICIES = ("strict", "moderate", "strict_rescue_inv1", "loose")
# Backward-compatible alias for older runs / scripts
PRUNE_POLICY_ALIASES = {"tau1_or_inv1": "strict_rescue_inv1"}
DEFAULT_PRUNE_POLICY = "strict"

GROUND_TRUTH_TIER_ORDER = list(range(7))

# Embedded in ranking_validation_summary.json; keep in sync with module docstring.
METRICS_REFERENCE: dict[str, str] = {
    "exact_match": "Recovered order equals T1→T7 exactly; 1.0 = perfect ladder recovery.",
    "kendall_tau": "Ordinal agreement with true ranks; 1 = all pairs correct, 0 ≈ chance.",
    "spearman_rho": "Rank correlation with true positions; 1 = perfect alignment.",
    "pairwise_inversion_errors": "Tier pairs ranked backwards vs ground truth; 0 = none.",
    "exact_position_matches": "Tiers in the correct rank slot; 7 = perfect positions.",
    "within_one_position": "Tiers off by at most one rank; high = mostly adjacent swaps.",
    "endpoints_correct": "T1 least and T7 most; true = correct extremes.",
    "least_preferable_correct": "Lowest-ranked item is T1.",
    "most_preferable_correct": "Highest-ranked item is T7.",
    "match_fraction": "Share of ladders with exact T1→T7 recovery.",
    "flagged_mismatch_fraction": "Share of ladders parsed but not matching T1→T7.",
    "mean_kendall_tau": "Mean τ over parsed ladders; descriptive only, not a p-value.",
    "perfect_tau_count": "Parsed ladders with τ = 1.0.",
    "high_tau_ge_0_9": "Parsed ladders with τ ≥ 0.9 (mostly correct ordering).",
    "low_tau_lt_0_5": "Parsed ladders with τ < 0.5 (mostly wrong ordering).",
}

VALID_LADDER_STATUSES = frozenset({"success", "fixed_via_critique_rewrite_v2"})

DEFAULT_MAX_TOKENS = 800
DEFAULT_CHUNK_SIZE = 40
DEFAULT_CONCURRENCY_LIMIT = 40
REASONING_ON_MAX_TOKENS = 2500
REASONING_ON_CHUNK_SIZE = 20
REASONING_ON_CONCURRENCY_LIMIT = 16
REASONING_ON_MAX_PARALLEL_CHUNKS = 4
REASONING_ON_BASE_TIMEOUT_S = 180

MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-55-openai": {"input": 5.00, "output": 30.00},
    "gpt-55": {"input": 5.00, "output": 30.00},
    "gpt-54-openai": {"input": 2.50, "output": 15.00},
}
MODEL_PRICING_SOURCE: dict[str, str] = {
    "gpt-55-openai": "https://openai.com/api/pricing/",
    "gpt-54-openai": "https://openai.com/api/pricing/",
}

REASONING_CLI_TO_EFFORT = {"off": "none", "on": "high"}

SYSTEM_PROMPT = """You are an expert at evaluating preferences and ranking outcomes.
You will be given a set of statements describing variations of the same scenario.
Your task is to rank them from LEAST preferable to MOST preferable.

Think carefully about what a reasonable person would prefer, considering the
well-being of those affected, the quality of outcomes described, and common
human values.

You MUST respond with ONLY a JSON object in this exact format:
{"ranking": ["X", "Y", "Z", ...]}

where the letters are ordered from LEAST preferable (first) to MOST preferable (last).
Do not include any other text, explanation, or commentary outside the JSON object."""


@dataclass(frozen=True)
class RunProfile:
    reasoning_effort: str
    reasoning_on: bool
    max_tokens: int
    chunk_size: int
    concurrency_limit: int
    max_parallel_chunks: int
    base_timeout: float


@dataclass(frozen=True)
class LadderTask:
    ladder_id: str
    category: str
    identified_property: str
    valence: str
    n_tiers: int
    statements: list[str]
    shuffle_indices: list[int]
    labels: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_utc_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_cli_output_dir(output_dir: Path) -> Path:
    """
    Resolve output-dir paths from the CLI.

    Prefer forward slashes, e.g.
    data/ladder_validation_tests_outputs/within_ladder_validation_ranking/gpt-55-openai
    """
    migrate_run_dir(RANKING_DIR_NAME)
    return normalize_validation_output_dir(output_dir, subdir_name=RANKING_DIR_NAME)


def smoke_output_dir_name(model_key: str) -> str:
    """Folder name for smoke runs, e.g. gpt-55-openai -> smoke_gpt55."""
    short = model_key.replace("-openai", "").replace("-", "")
    return f"smoke_{short}"


def resolve_output_dir(
    model_key: str,
    output_dir: Optional[Path],
    *,
    smoke: bool = False,
) -> Path:
    """
    Default run folder: data/ladder_validation_tests_outputs/within_ladder_validation_ranking/<model_key>.
    Smoke runs (--max-ladders / --start-from): .../smoke_<model>/.
    """
    if output_dir is not None:
        return normalize_cli_output_dir(output_dir)
    if smoke:
        return DEFAULT_OUTPUT_DIR / smoke_output_dir_name(model_key)
    return DEFAULT_OUTPUT_DIR / model_key


def write_json_file(path: Path, data: Any) -> None:
    """Write JSON with the same encoding as property_ladder_pruning.py."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_ranking_pruned_output(
    input_path: Path,
    pruned_output: Optional[Path],
) -> Path:
    """Write ranking-pruned variations under data/ladder_validation_tests_outputs/."""
    del input_path
    return resolve_pruned_variations_output_path(
        pruned_output, default_path=PRUNED_RANKING_PATH
    )


def resolve_ranking_pruned_ids_path(
    *,
    pruned_ids: Optional[Path],
    output_dir: Optional[Path],
    model_key: Optional[str],
) -> Path:
    if pruned_ids is not None:
        return pruned_ids
    if output_dir is not None:
        return output_dir / PRUNED_IDS_JSON_NAME
    if model_key is not None:
        return resolve_output_dir(model_key, None) / PRUNED_IDS_JSON_NAME
    raise ValueError(
        "Provide --pruned-ids, --output-dir, or --model to locate ranking_pruned_ladder_ids.json"
    )


def write_pruned_variations_file(
    input_path: Path,
    pruned_ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Remove ranking-pruned ladder IDs from the input variations file.

    Pruned = MISMATCH (wrong T1→T7 order) + UNPARSEABLE; kept = MATCH only.
    Same pattern as property_ladder_pruning.write_pruned_variations_file.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input variations file not found: {input_path}")
    if not pruned_ids_path.exists():
        raise FileNotFoundError(
            f"Pruned IDs file not found: {pruned_ids_path}\n"
            "Tip: use forward slashes for --output-dir "
            "(e.g. data/ladder_validation_tests_outputs/within_ladder_validation_ranking/gpt-55-openai), "
            "or pass --model gpt-55-openai."
        )

    ladders = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(ladders, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    pruned_ids_raw = json.loads(pruned_ids_path.read_text(encoding="utf-8"))
    if not isinstance(pruned_ids_raw, list):
        raise ValueError(f"Expected a JSON list in {pruned_ids_path}")
    pruned_ids = {str(x) for x in pruned_ids_raw}

    input_ids = {str(l.get("original_statement_id", "")) for l in ladders}
    kept_ladders = [
        l for l in ladders if str(l.get("original_statement_id", "")) not in pruned_ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(output_path, kept_ladders)

    return {
        "source_input": str(input_path),
        "pruned_ids_source": str(pruned_ids_path),
        "output_path": str(output_path),
        "n_input_ladders": len(ladders),
        "n_pruned_ids": len(pruned_ids),
        "n_output_ladders": len(kept_ladders),
        "n_removed": len(ladders) - len(kept_ladders),
        "pruned_ids_not_in_input": sorted(pruned_ids - input_ids),
    }


def effective_reasoning_effort(model_cfg: Any, reasoning_cli: Optional[str]) -> str:
    if reasoning_cli is not None:
        return REASONING_CLI_TO_EFFORT[reasoning_cli]
    extra_body = model_cfg.extra_body or {}
    return str(extra_body.get("reasoning_effort", "none")).lower()


def is_reasoning_enabled(effort: str) -> bool:
    return effort not in ("none", "", "null")


def resolve_run_profile(args: argparse.Namespace, model_cfg: Any) -> RunProfile:
    effort = effective_reasoning_effort(model_cfg, args.reasoning)
    reasoning_on = is_reasoning_enabled(effort)

    max_tokens = args.max_tokens
    chunk_size = args.chunk_size
    concurrency_limit = args.concurrency_limit
    max_parallel_chunks = args.max_parallel_chunks
    base_timeout = float(max(30, int(model_cfg.base_timeout)))

    if reasoning_on:
        if max_tokens == DEFAULT_MAX_TOKENS:
            max_tokens = REASONING_ON_MAX_TOKENS
        if chunk_size == DEFAULT_CHUNK_SIZE:
            chunk_size = REASONING_ON_CHUNK_SIZE
        if concurrency_limit == DEFAULT_CONCURRENCY_LIMIT:
            concurrency_limit = REASONING_ON_CONCURRENCY_LIMIT
        if max_parallel_chunks is None:
            max_parallel_chunks = REASONING_ON_MAX_PARALLEL_CHUNKS
        base_timeout = max(base_timeout, float(REASONING_ON_BASE_TIMEOUT_S))
    elif max_parallel_chunks is None:
        max_parallel_chunks = 1

    return RunProfile(
        reasoning_effort=effort,
        reasoning_on=reasoning_on,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        concurrency_limit=concurrency_limit,
        max_parallel_chunks=max(1, int(max_parallel_chunks)),
        base_timeout=base_timeout,
    )


def billable_output_tokens(completion_tokens: int, reasoning_tokens: int) -> int:
    completion_tokens = max(0, int(completion_tokens or 0))
    reasoning_tokens = max(0, int(reasoning_tokens or 0))
    if completion_tokens <= 0 and reasoning_tokens > 0:
        return reasoning_tokens
    return completion_tokens


def estimate_call_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    rates: Optional[dict[str, float]],
) -> Optional[float]:
    if not rates:
        return None
    output_tokens = billable_output_tokens(completion_tokens, reasoning_tokens)
    return round(
        (max(0, prompt_tokens) / 1_000_000) * rates["input"]
        + (output_tokens / 1_000_000) * rates["output"],
        6,
    )


def append_usage_to_cost_log(
    cost_log: dict[str, Any],
    usage_records: list[dict[str, Any]],
    rates: Optional[dict[str, float]],
) -> None:
    for u in usage_records:
        prompt_tokens = int(u.get("prompt_tokens") or 0)
        completion_tokens = int(u.get("completion_tokens") or 0)
        reasoning_tokens = int(u.get("reasoning_tokens") or 0)
        cost_log["prompt_tokens_total"] += prompt_tokens
        cost_log["completion_tokens_total"] += completion_tokens
        cost_log["reasoning_tokens_total"] += reasoning_tokens
        cost_log["records"].append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "billable_output_tokens": billable_output_tokens(
                    completion_tokens, reasoning_tokens
                ),
                "estimated_cost_usd": estimate_call_cost_usd(
                    prompt_tokens, completion_tokens, reasoning_tokens, rates
                ),
            }
        )
    cost_log["calls_logged"] += len(usage_records)
    refresh_cost_log_total(cost_log, rates)


def refresh_cost_log_total(cost_log: dict[str, Any], rates: Optional[dict[str, float]]) -> None:
    if not rates:
        return
    output_billable = sum(
        billable_output_tokens(
            int(r.get("completion_tokens") or 0),
            int(r.get("reasoning_tokens") or 0),
        )
        for r in cost_log.get("records") or []
    )
    cost_log["estimated_cost_usd"] = round(
        (int(cost_log.get("prompt_tokens_total") or 0) / 1_000_000) * rates["input"]
        + (output_billable / 1_000_000) * rates["output"],
        6,
    )


def resolve_model_pricing(model_key: str) -> tuple[Optional[dict[str, float]], str]:
    if model_key in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[model_key], MODEL_PRICING_SOURCE.get(model_key, "")
    base_key = model_key.removesuffix("-openai")
    if base_key in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[base_key], MODEL_PRICING_SOURCE.get(base_key, "")
    return None, ""


def resolve_extra_body(model_cfg: Any, reasoning: Optional[str]) -> Optional[dict[str, Any]]:
    extra_body: dict[str, Any] = dict(model_cfg.extra_body) if model_cfg.extra_body else {}
    if reasoning is not None:
        extra_body["reasoning_effort"] = REASONING_CLI_TO_EFFORT[reasoning]
    return extra_body or None


sys.path.insert(0, str(BASE_DIR.parent.parent))
from compute_utilities.utils import create_agent  # noqa: E402
from experiments.parametric_variations.config import MODEL_CONFIGS  # noqa: E402


def load_ladders(
    input_path: Path,
    *,
    max_ladders: Optional[int] = None,
    start_from: int = 0,
) -> list[dict[str, Any]]:
    ladders = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(ladders, list):
        raise ValueError(f"Expected JSON array in {input_path}")
    ladders = [l for l in ladders if l.get("status") in VALID_LADDER_STATUSES]
    if start_from:
        ladders = ladders[start_from:]
    if max_ladders is not None:
        ladders = ladders[:max_ladders]
    return ladders


def statements_from_ladder(ladder: dict[str, Any]) -> list[str]:
    """Extract tier texts in canonical T1..T7 order; strip tier/label metadata."""
    variations = ladder.get("variations") or []
    sorted_vars = sorted(variations, key=lambda v: int(v.get("tier", 0)))
    texts: list[str] = []
    for v in sorted_vars:
        text = str(v.get("text", "")).strip()
        if not text:
            raise ValueError(
                f"Ladder {ladder.get('original_statement_id')}: missing text for tier {v.get('tier')}"
            )
        texts.append(text)
    return texts


def ladder_shuffle_seed(base_seed: int, ladder_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{ladder_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def build_ladder_task(ladder: dict[str, Any], base_seed: int) -> LadderTask:
    import random

    ladder_id = str(ladder["original_statement_id"])
    statements = statements_from_ladder(ladder)
    n_tiers = len(statements)
    if n_tiers < 2:
        raise ValueError(f"Ladder {ladder_id}: need at least 2 tiers, got {n_tiers}")

    rng = random.Random(ladder_shuffle_seed(base_seed, ladder_id))
    indices = list(range(n_tiers))
    rng.shuffle(indices)
    labels = list(string.ascii_uppercase[:n_tiers])

    return LadderTask(
        ladder_id=ladder_id,
        category=str(ladder.get("category", "")),
        identified_property=str(ladder.get("identified_property", "")),
        valence=str(ladder.get("valence", "")),
        n_tiers=n_tiers,
        statements=statements,
        shuffle_indices=indices,
        labels=labels,
    )


def build_user_prompt(task: LadderTask) -> str:
    lines = [
        f"CATEGORY: {task.category}",
        f"PROPERTY BEING VARIED: {task.identified_property}",
        f"VALENCE: {task.valence}",
        "",
        f"Below are {task.n_tiers} statements describing variations of the same scenario.",
        "Rank them from LEAST preferable to MOST preferable.",
        "",
        "STATEMENTS:",
        "",
    ]
    for pos, idx in enumerate(task.shuffle_indices):
        label = task.labels[pos]
        lines.append(f"  [{label}] {task.statements[idx]}")
        lines.append("")
    lines.append(
        f"Rank ALL {task.n_tiers} statements from LEAST preferable (first) to MOST preferable (last)."
    )
    example = ", ".join(f'"{c}"' for c in task.labels)
    lines.append(f'Respond with ONLY: {{"ranking": [{example}]}}')
    return "\n".join(lines)


def parse_ranking_response(response: str, expected_labels: list[str]) -> Optional[list[str]]:
    response_clean = (response or "").strip()
    if response_clean.startswith("```"):
        response_clean = re.sub(r"```(?:json)?\s*", "", response_clean)
        response_clean = re.sub(r"\s*```\s*$", "", response_clean)

    json_match = re.search(r'\{[^}]*"ranking"[^}]*\}', response_clean, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", response_clean, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return None

    ranking = [str(r).strip().upper() for r in ranking]
    expected = [lbl.upper() for lbl in expected_labels]
    if set(ranking) != set(expected) or len(ranking) != len(expected):
        return None
    return ranking


def kendall_tau(predicted_ranks: list[int], truth_ranks: list[int]) -> float:
    n = len(predicted_ranks)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (predicted_ranks[i] - predicted_ranks[j]) * (
                truth_ranks[i] - truth_ranks[j]
            )
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


def spearman_rho(predicted_ranks: list[int], truth_ranks: list[int]) -> float:
    n = len(predicted_ranks)
    if n < 2:
        return 1.0
    d_squared_sum = sum((p - t) ** 2 for p, t in zip(predicted_ranks, truth_ranks))
    return 1 - (6 * d_squared_sum) / (n * (n**2 - 1))


def count_inversion_errors(predicted_ranks: list[int], truth_ranks: list[int]) -> int:
    n = len(predicted_ranks)
    errors = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (predicted_ranks[i] - predicted_ranks[j]) * (
                truth_ranks[i] - truth_ranks[j]
            ) < 0:
                errors += 1
    return errors


def score_ranking(task: LadderTask, parsed_ranking: list[str]) -> dict[str, Any]:
    """Score one ladder; see module docstring *Metrics reference* for definitions."""
    label_to_tier_index: dict[str, int] = {}
    for pos, tier_idx in enumerate(task.shuffle_indices):
        label_to_tier_index[task.labels[pos]] = tier_idx

    recovered_order = [label_to_tier_index[lbl] for lbl in parsed_ranking]
    ground_truth_order = list(range(task.n_tiers))

    predicted_ranks = [0] * task.n_tiers
    for rank, tier_idx in enumerate(recovered_order):
        predicted_ranks[tier_idx] = rank
    truth_ranks = list(range(task.n_tiers))

    exact_match = recovered_order == ground_truth_order
    max_pairs = task.n_tiers * (task.n_tiers - 1) // 2

    return {
        "recovered_tier_order": recovered_order,
        "ground_truth_order": ground_truth_order,
        "ground_truth_tiers": [f"T{i + 1}" for i in ground_truth_order],
        "matches_intended_t1_t7_order": exact_match,
        "exact_match": exact_match,
        "kendall_tau": kendall_tau(predicted_ranks, truth_ranks),
        "spearman_rho": spearman_rho(predicted_ranks, truth_ranks),
        "pairwise_inversion_errors": count_inversion_errors(predicted_ranks, truth_ranks),
        "max_pairwise_errors": max_pairs,
        "exact_position_matches": sum(
            1 for i, r in enumerate(predicted_ranks) if r == i
        ),
        "within_one_position": sum(
            1 for i, r in enumerate(predicted_ranks) if abs(r - i) <= 1
        ),
        "endpoints_correct": (
            recovered_order[0] == 0 and recovered_order[-1] == task.n_tiers - 1
        ),
        "least_preferable_correct": recovered_order[0] == 0,
        "most_preferable_correct": recovered_order[-1] == task.n_tiers - 1,
    }


def result_from_task(
    task: LadderTask,
    raw_response: str,
    *,
    parse_success: bool,
    parsed_ranking: Optional[list[str]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ladder_id": task.ladder_id,
        "category": task.category,
        "identified_property": task.identified_property,
        "valence": task.valence,
        "n_tiers": task.n_tiers,
        "shuffle_indices": task.shuffle_indices,
        "labels": task.labels,
        "raw_response": (raw_response or "")[:8000],
        "parse_success": parse_success,
        "timestamp": now_iso(),
    }
    if error:
        row["error"] = error
    if not parse_success or parsed_ranking is None:
        return row

    row["parsed_ranking"] = parsed_ranking
    row.update(score_ranking(task, parsed_ranking))
    return row


async def _process_task_chunk(
    tasks: list[LadderTask],
    *,
    model_key: str,
    temperature: float,
    max_tokens: int,
    concurrency_limit: int,
    extra_body: Optional[dict[str, Any]],
    enable_cache: bool,
    base_timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agent = create_agent(
        model_key=model_key,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency_limit=concurrency_limit,
        extra_body=extra_body,
        enable_cache=enable_cache,
        base_timeout=int(base_timeout),
    )

    messages: list[list[dict[str, str]]] = []
    for task in tasks:
        messages.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task)},
            ]
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings",
            category=UserWarning,
        )
        responses = await agent.async_completions(messages, verbose=False)

    if len(responses) != len(tasks):
        raise RuntimeError(
            f"API returned {len(responses)} responses for {len(tasks)} ranking tasks"
        )

    rows: list[dict[str, Any]] = []
    for task, resp in zip(tasks, responses):
        parsed = parse_ranking_response(resp or "", task.labels)
        if parsed is None:
            rows.append(
                result_from_task(
                    task,
                    resp or "",
                    parse_success=False,
                    error="Failed to parse ranking JSON from response",
                )
            )
        else:
            rows.append(
                result_from_task(task, resp or "", parse_success=True, parsed_ranking=parsed)
            )
    return rows, list(getattr(agent, "usage_log", []))


async def run_ranking_requests(
    model_key: str,
    tasks: list[LadderTask],
    output_dir: Path,
    *,
    chunk_size: int,
    temperature: float,
    max_tokens: int,
    concurrency_limit: int,
    resume: bool,
    reasoning: Optional[str],
    max_parallel_chunks: int,
    base_timeout: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_JSONL_NAME

    done_ids: set[str] = set()
    if resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    lid = row.get("ladder_id")
                    if lid:
                        done_ids.add(str(lid))
                except json.JSONDecodeError:
                    continue

    pending = [t for t in tasks if t.ladder_id not in done_ids]
    if not pending:
        print("All ladders already ranked; skipping API calls.")
        return

    model_cfg = MODEL_CONFIGS[model_key]
    extra_body = resolve_extra_body(model_cfg, reasoning)
    effort = (extra_body or {}).get("reasoning_effort", "unset")
    print(
        f"Reasoning: reasoning_effort={effort}"
        + (f" (--reasoning {reasoning})" if reasoning is not None else " (from MODEL_CONFIGS)")
    )
    print(
        f"Batching: chunk_size={chunk_size}, concurrency_limit={concurrency_limit}, "
        f"max_parallel_chunks={max_parallel_chunks}, max_tokens={max_tokens}, "
        f"base_timeout={base_timeout:.0f}s"
    )

    rates, pricing_source = resolve_model_pricing(model_key)
    cost_path = output_dir / COST_JSON_NAME
    cost_log: dict[str, Any] = {
        "model_key": model_key,
        "pricing_source": pricing_source,
        "pricing_per_1m": rates,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "reasoning_tokens_total": 0,
        "estimated_cost_usd": None,
        "calls_logged": 0,
        "records": [],
        "updated_at": now_iso(),
    }
    if resume and cost_path.exists():
        try:
            prev = json.loads(cost_path.read_text(encoding="utf-8"))
            cost_log["prompt_tokens_total"] = int(prev.get("prompt_tokens_total", 0) or 0)
            cost_log["completion_tokens_total"] = int(prev.get("completion_tokens_total", 0) or 0)
            cost_log["reasoning_tokens_total"] = int(prev.get("reasoning_tokens_total", 0) or 0)
            cost_log["calls_logged"] = int(prev.get("calls_logged", 0) or 0)
            if isinstance(prev.get("records"), list):
                cost_log["records"] = prev["records"]
            refresh_cost_log_total(cost_log, rates)
        except Exception:
            pass

    total = len(pending)
    chunk_starts = list(range(0, total, chunk_size))
    sem = asyncio.Semaphore(max(1, max_parallel_chunks))
    completed = 0

    async def run_one_chunk(start: int) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
        async with sem:
            chunk = pending[start : start + chunk_size]
            rows, usage_records = await _process_task_chunk(
                chunk,
                model_key=model_key,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency_limit=concurrency_limit,
                extra_body=extra_body,
                enable_cache=model_cfg.enable_cache,
                base_timeout=base_timeout,
            )
            return start, rows, usage_records

    mode = "a" if raw_path.exists() and resume else "w"
    with raw_path.open(mode, encoding="utf-8") as out:
        wave_size = max(1, max_parallel_chunks)
        for wave_base in range(0, len(chunk_starts), wave_size):
            wave = chunk_starts[wave_base : wave_base + wave_size]
            batch_results = await asyncio.gather(*(run_one_chunk(start) for start in wave))
            for _start, rows, usage_records in sorted(batch_results, key=lambda x: x[0]):
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                append_usage_to_cost_log(cost_log, usage_records, rates)
                cost_log["updated_at"] = now_iso()
                write_json_file(cost_path, cost_log)
                completed += len(rows)
                print(f"Ranked {completed}/{total} pending ladders")


def load_results_from_jsonl(raw_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not raw_path.exists():
        return results
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def tier_order_to_labels(tier_order: list[int]) -> list[str]:
    return [f"T{i + 1}" for i in tier_order]


def normalize_prune_policy(prune_policy: str) -> str:
    return PRUNE_POLICY_ALIASES.get(prune_policy, prune_policy)


def ladder_passes_strict_tau1(ladder_result: dict[str, Any]) -> bool:
    """Step 1 baseline: exact T1→T7 (Kendall tau = 1)."""
    if ladder_result.get("status") == "UNPARSEABLE":
        return False
    tau = float(ladder_result.get("kendall_tau") or 0.0)
    if tau >= 1.0 - 1e-9:
        return True
    return bool(
        ladder_result.get(
            "matches_intended_t1_t7_order",
            ladder_result.get("status") == "MATCH",
        )
    )


def ladder_rescued_by_inv1_after_strict_drop(ladder_result: dict[str, Any]) -> bool:
    """Step 2: from ladders dropped by tau < 1, bring back exactly one pairwise inversion."""
    if ladder_result.get("status") == "UNPARSEABLE":
        return False
    if ladder_passes_strict_tau1(ladder_result):
        return False
    inv = int(ladder_result.get("pairwise_inversion_errors") or 0)
    return inv == 1


def ladder_passes_prune_policy(ladder_result: dict[str, Any], prune_policy: str) -> bool:
    """Whether a ladder is kept in phase6b_variations_ranking_pruned.json."""
    prune_policy = normalize_prune_policy(prune_policy)
    if prune_policy not in PRUNE_POLICIES:
        raise ValueError(f"Unknown prune_policy: {prune_policy!r}. Use one of {PRUNE_POLICIES}")

    if ladder_result.get("status") == "UNPARSEABLE":
        return False

    tau = float(ladder_result.get("kendall_tau") or 0.0)

    if prune_policy == "strict":
        return ladder_passes_strict_tau1(ladder_result)
    if prune_policy == "moderate":
        return tau >= 0.9
    if prune_policy == "strict_rescue_inv1":
        if ladder_passes_strict_tau1(ladder_result):
            return True
        return ladder_rescued_by_inv1_after_strict_drop(ladder_result)
    if prune_policy == "loose":
        return tau >= 0.7 or bool(ladder_result.get("endpoints_correct"))
    return False


def prune_policy_description(prune_policy: str) -> str:
    prune_policy = normalize_prune_policy(prune_policy)
    return {
        "strict": "keep exact T1→T7 MATCH only",
        "moderate": "keep parsed with Kendall tau >= 0.9",
        "strict_rescue_inv1": (
            "step 1: keep tau=1 (strict); step 2: from tau<1 drops, rescue inv==1 only"
        ),
        "loose": "keep parsed with tau >= 0.7 or correct T1/T7 endpoints",
    }[prune_policy]


def explain_ranking_result(
    *,
    status: str,
    ranking_decision: str,
    recovered_tier_order: Optional[list[int]],
    ground_truth_order: list[int],
    kendall_tau: Optional[float],
    error: Optional[str],
) -> str:
    truth_labels = tier_order_to_labels(ground_truth_order)
    if status == "UNPARSEABLE":
        return (
            f"Could not parse a valid ranking JSON from the model response"
            + (f": {error}" if error else ".")
            + f" Intended order (least→most): {truth_labels}."
        )
    recovered_labels = tier_order_to_labels(recovered_tier_order or [])
    if status == "MATCH":
        return (
            f"Recovered ranking matches intended T1→T7 order {truth_labels} "
            f"(least preferable to most). ranking_decision: OK."
        )
    tau_s = f"{kendall_tau:.3f}" if kendall_tau is not None else "n/a"
    return (
        f"Recovered order {recovered_labels} does NOT match intended T1→T7 order "
        f"{truth_labels}. Kendall τ={tau_s}. "
        f"This ladder is FLAGGED (ranking_decision: FLAG)."
    )


def ladder_detail_from_result(row: dict[str, Any]) -> dict[str, Any]:
    """One ladder entry for ranking_validation_details.json (mirrors property pruning layout)."""
    ladder_id = str(row.get("ladder_id", ""))
    n_tiers = int(row.get("n_tiers") or 7)
    ground_truth_order = list(range(n_tiers))

    if not row.get("parse_success"):
        return {
            "ladder_id": ladder_id,
            "category": row.get("category", ""),
            "property_name": row.get("identified_property", ""),
            "valence": row.get("valence", ""),
            "status": "UNPARSEABLE",
            "ranking_decision": "UNPARSEABLE",
            "matches_intended_t1_t7_order": False,
            "ground_truth_order": ground_truth_order,
            "ground_truth_tiers": tier_order_to_labels(ground_truth_order),
            "shuffle_indices": row.get("shuffle_indices"),
            "labels": row.get("labels"),
            "error": row.get("error", "Failed to parse ranking"),
            "ranking_explanation": explain_ranking_result(
                status="UNPARSEABLE",
                ranking_decision="UNPARSEABLE",
                recovered_tier_order=None,
                ground_truth_order=ground_truth_order,
                kendall_tau=None,
                error=row.get("error"),
            ),
        }

    matches = bool(row.get("matches_intended_t1_t7_order", row.get("exact_match")))
    recovered = list(row.get("recovered_tier_order") or [])
    status = "MATCH" if matches else "MISMATCH"
    ranking_decision = "OK" if matches else "FLAG"

    return {
        "ladder_id": ladder_id,
        "category": row.get("category", ""),
        "property_name": row.get("identified_property", ""),
        "valence": row.get("valence", ""),
        "status": status,
        "ranking_decision": ranking_decision,
        "matches_intended_t1_t7_order": matches,
        "recovered_tier_order": recovered,
        "recovered_tiers": tier_order_to_labels(recovered),
        "ground_truth_order": ground_truth_order,
        "ground_truth_tiers": tier_order_to_labels(ground_truth_order),
        "parsed_ranking": row.get("parsed_ranking"),
        "shuffle_indices": row.get("shuffle_indices"),
        "labels": row.get("labels"),
        "kendall_tau": row.get("kendall_tau"),
        "spearman_rho": row.get("spearman_rho"),
        "pairwise_inversion_errors": row.get("pairwise_inversion_errors"),
        "max_pairwise_errors": row.get("max_pairwise_errors"),
        "exact_position_matches": row.get("exact_position_matches"),
        "within_one_position": row.get("within_one_position"),
        "endpoints_correct": row.get("endpoints_correct"),
        "least_preferable_correct": row.get("least_preferable_correct"),
        "most_preferable_correct": row.get("most_preferable_correct"),
        "ranking_explanation": explain_ranking_result(
            status=status,
            ranking_decision=ranking_decision,
            recovered_tier_order=recovered,
            ground_truth_order=ground_truth_order,
            kendall_tau=row.get("kendall_tau"),
            error=None,
        ),
    }


def aggregate_ranking_results(
    results: list[dict[str, Any]],
    *,
    model_key: str,
    input_path: Path,
    seed: int,
    reasoning_effort: str,
    prune_policy: str = DEFAULT_PRUNE_POLICY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build summary + details in the same shape as property_pruning_*.json."""
    prune_policy = normalize_prune_policy(prune_policy)
    ladder_results = [ladder_detail_from_result(r) for r in results]
    decision_rank = {"FLAG": 0, "UNPARSEABLE": 1, "OK": 2}
    ladder_results.sort(
        key=lambda x: (decision_rank.get(x["ranking_decision"], 9), x["ladder_id"])
    )

    match_ids: list[str] = []
    mismatch_ids: list[str] = []
    unparseable_ids: list[str] = []

    for lr in ladder_results:
        lid = lr["ladder_id"]
        if lr["status"] == "MATCH":
            match_ids.append(lid)
        elif lr["status"] == "MISMATCH":
            mismatch_ids.append(lid)
        else:
            unparseable_ids.append(lid)

    counts = {"MATCH": 0, "MISMATCH": 0, "UNPARSEABLE": 0}
    for lr in ladder_results:
        counts[lr["status"]] += 1

    n_ladders = len(ladder_results)
    successful = [lr for lr in ladder_results if lr["status"] != "UNPARSEABLE"]
    tau_scores = [float(lr["kendall_tau"]) for lr in successful if lr.get("kendall_tau") is not None]

    summary = {
        "generated_at": now_iso(),
        "metrics_reference": METRICS_REFERENCE,
        "metrics_note": (
            "Descriptive agreement metrics only (no p-values). "
            "See full_ladder_ranking_pruning.py module docstring for details."
        ),
        "model_key": model_key,
        "input_file": str(input_path),
        "random_seed": seed,
        "reasoning_effort": reasoning_effort,
        "n_ladders": n_ladders,
        "counts": counts,
        "n_match": len(match_ids),
        "n_flagged_mismatch": len(mismatch_ids),
        "n_unparseable": len(unparseable_ids),
        "match_fraction": (len(match_ids) / n_ladders) if n_ladders else 0.0,
        "flagged_mismatch_fraction": (len(mismatch_ids) / n_ladders) if n_ladders else 0.0,
        "criteria": {
            "task": "full_ladder_ranking_least_to_most",
            "ground_truth_order": "T1 (least preferable) through T7 (most preferable)",
            "ground_truth_tier_indices": GROUND_TRUTH_TIER_ORDER,
            "match_requires_exact_t1_t7_recovery": True,
            "presentation": "seven statements shuffled with neutral labels; no tier metadata shown",
        },
        "aggregate_metrics": {
            "mean_kendall_tau": sum(tau_scores) / len(tau_scores) if tau_scores else None,
            "perfect_tau_count": sum(1 for t in tau_scores if t == 1.0),
            "high_tau_ge_0_9": sum(1 for t in tau_scores if t >= 0.9),
            "low_tau_lt_0_5": sum(1 for t in tau_scores if t < 0.5),
        },
    }

    kept_ids = [lr["ladder_id"] for lr in ladder_results if ladder_passes_prune_policy(lr, prune_policy)]
    pruned_ids = [lr["ladder_id"] for lr in ladder_results if not ladder_passes_prune_policy(lr, prune_policy)]

    rescued_inv1_ids = [
        lr["ladder_id"]
        for lr in ladder_results
        if ladder_rescued_by_inv1_after_strict_drop(lr)
    ]
    added_back_from_strict = [
        lid for lid in rescued_inv1_ids if lid in kept_ids
    ]

    summary["prune_policy"] = prune_policy
    summary["n_pruned"] = len(pruned_ids)
    summary["n_kept"] = len(kept_ids)
    summary["kept_fraction"] = (len(kept_ids) / n_ladders) if n_ladders else 0.0
    summary["pruned_fraction"] = (len(pruned_ids) / n_ladders) if n_ladders else 0.0
    summary["pruning_criteria"] = {
        "diagnostic_status": "MATCH = exact T1→T7; MISMATCH = parsed but wrong order",
        "export_keep_if": prune_policy_description(prune_policy),
        "n_exact_match": len(match_ids),
        "n_kept_under_policy": len(kept_ids),
        "n_added_back_vs_strict": len(added_back_from_strict),
        "n_rescued_inv1_after_strict_drop": len(rescued_inv1_ids),
    }

    for lr in ladder_results:
        lr["passes_strict_tau1"] = ladder_passes_strict_tau1(lr)
        lr["rescued_by_inv1"] = ladder_rescued_by_inv1_after_strict_drop(lr)
        lr["passes_prune_policy"] = ladder_passes_prune_policy(lr, prune_policy)
        lr["export_decision"] = "KEEP" if lr["passes_prune_policy"] else "PRUNE"

    details = {
        "generated_at": now_iso(),
        "summary": summary,
        "ladders": ladder_results,
        "match_ladder_ids": match_ids,
        "mismatch_ladder_ids": mismatch_ids,
        "unparseable_ladder_ids": unparseable_ids,
        "pruned_ladder_ids": pruned_ids,
        "kept_ladder_ids": kept_ids,
        "added_back_vs_strict_ids": added_back_from_strict,
        "rescued_inv1_after_strict_drop_ids": rescued_inv1_ids,
    }
    return summary, details


def build_ranking_cost_summary_from_log(cost_log: dict[str, Any]) -> dict[str, Any]:
    model_key = cost_log.get("model_key")
    estimated_total = float(cost_log.get("estimated_cost_usd") or 0.0)
    records = cost_log.get("records") or []
    return {
        "generated_at": now_utc_z(),
        "schema_note": (
            "Full-ladder ranking validation cost summary. Mirrors cost_summary.py / "
            "property pruning cost_summary layout."
        ),
        "grand_totals": {
            "n_files": 1,
            "n_recorded": int(cost_log.get("calls_logged") or 0),
            "n_with_actual": 0,
            "n_with_estimated": int(cost_log.get("calls_logged") or 0),
            "n_extrapolated": 0,
            "n_local_no_cost": 0,
            "estimated_total_usd": estimated_total,
            "actual_total_usd": 0.0,
            "extrapolated_total_usd": 0.0,
            "actual_plus_extrapolated_usd": 0.0,
            "best_available_total_usd": estimated_total,
        },
        "by_variant": [
            {
                "variant": model_key,
                "n_ladders": int(cost_log.get("calls_logged") or 0),
                "n_recorded": int(cost_log.get("calls_logged") or 0),
                "n_with_actual": 0,
                "n_with_estimated": int(cost_log.get("calls_logged") or 0),
                "n_extrapolated": 0,
                "n_local_no_cost": 0,
                "estimated_total_usd": estimated_total,
                "actual_total_usd": 0.0,
                "extrapolated_total_usd": 0.0,
                "extrapolation_basis": "none",
                "ladders_recorded": [],
                "ladders_extrapolated": [],
                "warnings": [],
                "token_totals": {
                    "prompt_tokens_total": int(cost_log.get("prompt_tokens_total") or 0),
                    "completion_tokens_total": int(cost_log.get("completion_tokens_total") or 0),
                    "reasoning_tokens_total": int(cost_log.get("reasoning_tokens_total") or 0),
                    "calls_logged": int(cost_log.get("calls_logged") or 0),
                },
            }
        ],
        "records": records,
    }


def save_ranking_outputs(output_dir: Path, summary: dict[str, Any], details: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(output_dir / SUMMARY_JSON_NAME, summary)
    write_json_file(output_dir / DETAILS_JSON_NAME, details)
    write_json_file(output_dir / MATCH_IDS_JSON_NAME, details["match_ladder_ids"])
    write_json_file(output_dir / MISMATCH_IDS_JSON_NAME, details["mismatch_ladder_ids"])
    write_json_file(output_dir / UNPARSEABLE_IDS_JSON_NAME, details["unparseable_ladder_ids"])
    write_json_file(output_dir / PRUNED_IDS_JSON_NAME, details["pruned_ladder_ids"])


def save_ranking_cost_summary(output_dir: Path) -> None:
    cost_path = output_dir / COST_JSON_NAME
    if not cost_path.exists():
        return
    cost_log = json.loads(cost_path.read_text(encoding="utf-8"))
    write_json_file(
        output_dir / COST_SUMMARY_JSON_NAME,
        build_ranking_cost_summary_from_log(cost_log),
    )


def print_summary(summary: dict[str, Any], details: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("FULL-LADDER RANKING — AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Model: {summary['model_key']}  |  reasoning_effort: {summary['reasoning_effort']}")
    print(f"Total ladders: {summary['n_ladders']}")
    print(
        f"Prune policy: {summary.get('prune_policy', 'strict')}  |  "
        f"kept={summary['n_kept']}  pruned={summary['n_pruned']}"
    )
    pc = summary.get("pruning_criteria") or {}
    if pc.get("n_added_back_vs_strict") is not None:
        print(
            f"  step 1 (tau=1 / strict): {pc.get('n_exact_match')} kept; "
            f"step 2 (rescue inv==1 from tau<1): {pc.get('n_rescued_inv1_after_strict_drop', pc.get('n_added_back_vs_strict'))}"
        )
    print(f"Counts: MATCH={summary['counts']['MATCH']}, "
          f"MISMATCH (FLAG)={summary['counts']['MISMATCH']}, "
          f"UNPARSEABLE={summary['counts']['UNPARSEABLE']}")
    if summary["n_ladders"]:
        print(
            f"\nExact T1->T7 matches: {summary['n_match']}/{summary['n_ladders']} "
            f"({100 * summary['match_fraction']:.1f}%)"
        )
        print(
            f"Flagged (wrong order): {summary['n_flagged_mismatch']}/{summary['n_ladders']} "
            f"({100 * summary['flagged_mismatch_fraction']:.1f}%)"
        )
    agg = summary.get("aggregate_metrics") or {}
    if agg.get("mean_kendall_tau") is not None:
        print(f"Mean Kendall's tau (parsed only): {agg['mean_kendall_tau']:.4f}")

    if details["mismatch_ladder_ids"]:
        print(f"\nFlagged ladder IDs (first 10): {details['mismatch_ladder_ids'][:10]}")
        if len(details["mismatch_ladder_ids"]) > 10:
            print(f"  ... and {len(details['mismatch_ladder_ids']) - 10} more")


def run_write_pruned_variations_only(args: argparse.Namespace) -> int:
    output_dir = (
        normalize_cli_output_dir(args.output_dir) if args.output_dir is not None else None
    )
    pruned_ids_path = resolve_ranking_pruned_ids_path(
        pruned_ids=args.pruned_ids,
        output_dir=output_dir,
        model_key=args.model,
    )
    pruned_output = resolve_ranking_pruned_output(args.input, args.pruned_output)
    meta = write_pruned_variations_file(
        input_path=args.input,
        pruned_ids_path=pruned_ids_path,
        output_path=pruned_output,
    )
    print(f"Wrote ranking-pruned variations: {meta['output_path']}")
    print(f"  Input ladders: {meta['n_input_ladders']}")
    print(f"  Pruned IDs: {meta['n_pruned_ids']}")
    print(f"  Output ladders: {meta['n_output_ladders']}")
    print(f"  Removed: {meta['n_removed']}")
    if meta["pruned_ids_not_in_input"]:
        print(f"  WARNING: {len(meta['pruned_ids_not_in_input'])} pruned ID(s) not found in input")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-ladder ranking validation: recover T1–T7 preference order without tier labels."
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Model key from MODEL_CONFIGS (required unless --write-pruned-variations-only with --output-dir)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Ladder JSON (default: data/phase6b_variations_prop_pruned.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: data/ladder_validation_tests_outputs/within_ladder_validation_ranking/<model_key>, "
            "or smoke_<model> when using --max-ladders / --start-from)"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max response tokens (auto 2500 when --reasoning on)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--concurrency-limit", type=int, default=DEFAULT_CONCURRENCY_LIMIT)
    parser.add_argument(
        "--max-parallel-chunks",
        type=int,
        default=None,
        help="Parallel API batches (default: 4 when reasoning on, else 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (per-ladder derived)")
    parser.add_argument(
        "--max-ladders",
        type=int,
        default=None,
        help=(
            "Smoke: evaluate first N ladders only. "
            "Default output: data/ladder_validation_tests_outputs/.../ranking/smoke_<model>/ "
            "(e.g. smoke_gpt55)."
        ),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip first N ladders before --max-ladders slice (also uses smoke output dir)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true", help="Aggregate existing JSONL only")
    parser.add_argument(
        "--reasoning",
        choices=["off", "on"],
        default=None,
        help="GPT-5.x: off=reasoning_effort none, on=high",
    )
    parser.add_argument(
        "--write-pruned-variations-only",
        action="store_true",
        help=(
            f"Only write {PRUNED_VARIATIONS_FILENAME} from ranking_pruned_ladder_ids.json "
            "(no API/aggregation)"
        ),
    )
    parser.add_argument(
        "--pruned-output",
        type=Path,
        default=None,
        help=(
            "Output path for kept ladders after ranking prune "
            f"(default: <input-dir>/{PRUNED_VARIATIONS_FILENAME})"
        ),
    )
    parser.add_argument(
        "--pruned-ids",
        type=Path,
        default=None,
        help="Path to ranking_pruned_ladder_ids.json (default: <output-dir>/ranking_pruned_ladder_ids.json)",
    )
    parser.add_argument(
        "--prune-policy",
        choices=list(PRUNE_POLICIES) + list(PRUNE_POLICY_ALIASES.keys()),
        default=DEFAULT_PRUNE_POLICY,
        help=(
            "Export prune rule (default: strict). "
            "strict_rescue_inv1: drop tau<1, then rescue ladders with exactly 1 pairwise inversion. "
            "Alias: tau1_or_inv1"
        ),
    )
    return parser.parse_args()


async def amain() -> int:
    args = parse_args()
    migrated = migrate_all_legacy_validation_layout()
    if migrated:
        print("Migrated legacy validation layout: " + "; ".join(migrated))

    if args.write_pruned_variations_only:
        if args.analyze_only:
            raise ValueError("Use only one of --write-pruned-variations-only or --analyze-only")
        if args.model is None and args.output_dir is None and args.pruned_ids is None:
            raise ValueError(
                "--model, --output-dir, or --pruned-ids required for --write-pruned-variations-only"
            )
        return run_write_pruned_variations_only(args)

    if args.model is None:
        raise ValueError("--model is required unless using --write-pruned-variations-only")
    if args.model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {sorted(MODEL_CONFIGS.keys())}"
        )

    if args.start_from < 0:
        raise ValueError("--start-from must be >= 0")
    if args.max_ladders is not None and args.max_ladders <= 0:
        raise ValueError("--max-ladders must be > 0")

    model_cfg = MODEL_CONFIGS[args.model]
    run_profile = resolve_run_profile(args, model_cfg)
    smoke_run = args.max_ladders is not None or args.start_from > 0
    output_dir = resolve_output_dir(args.model, args.output_dir, smoke=smoke_run)
    raw_path = output_dir / RAW_JSONL_NAME

    if not args.analyze_only:
        ladders = load_ladders(
            args.input,
            max_ladders=args.max_ladders,
            start_from=args.start_from,
        )
        if smoke_run:
            print(
                f"Ladder slice: start_from={args.start_from}, "
                f"max_ladders={args.max_ladders if args.max_ladders is not None else 'all'}"
            )
        tasks = [build_ladder_task(l, args.seed) for l in ladders]
        print(f"Evaluating {len(tasks)} ladders → {output_dir}")
        await run_ranking_requests(
            args.model,
            tasks,
            output_dir,
            chunk_size=run_profile.chunk_size,
            temperature=args.temperature,
            max_tokens=run_profile.max_tokens,
            concurrency_limit=run_profile.concurrency_limit,
            resume=args.resume,
            reasoning=args.reasoning,
            max_parallel_chunks=run_profile.max_parallel_chunks,
            base_timeout=run_profile.base_timeout,
        )
    elif not raw_path.exists():
        raise FileNotFoundError(f"No results at {raw_path}; run without --analyze-only first.")

    results = load_results_from_jsonl(raw_path)
    if not results:
        raise FileNotFoundError(f"No rows in {raw_path}")

    summary, details = aggregate_ranking_results(
        results,
        model_key=args.model,
        input_path=args.input,
        seed=args.seed,
        reasoning_effort=run_profile.reasoning_effort,
        prune_policy=args.prune_policy,
    )
    save_ranking_outputs(output_dir, summary, details)
    save_ranking_cost_summary(output_dir)
    print_summary(summary, details)
    print(f"\nSaved under {output_dir}:")
    print(f"  Summary: {SUMMARY_JSON_NAME}")
    print(f"  Details: {DETAILS_JSON_NAME}")
    print(f"  Match IDs: {MATCH_IDS_JSON_NAME}")
    print(f"  Flagged mismatch IDs: {MISMATCH_IDS_JSON_NAME}")
    print(f"  Unparseable IDs: {UNPARSEABLE_IDS_JSON_NAME}")
    print(f"  Pruned IDs (combined): {PRUNED_IDS_JSON_NAME}")

    if smoke_run:
        print(
            f"Skipping write of {PRUNED_VARIATIONS_FILENAME} "
            "(use a full run without --max-ladders / --start-from)."
        )
    else:
        pruned_output = resolve_ranking_pruned_output(args.input, args.pruned_output)
        pruned_meta = write_pruned_variations_file(
            input_path=args.input,
            pruned_ids_path=output_dir / PRUNED_IDS_JSON_NAME,
            output_path=pruned_output,
        )
        print(f"\nRanking-pruned variations: {pruned_meta['output_path']}")
        print(
            f"  Kept {pruned_meta['n_output_ladders']}/{pruned_meta['n_input_ladders']} ladders "
            f"(removed {pruned_meta['n_removed']})"
        )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
