#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property ladder pruning for phase6b variations.

Implements a design-validity audit (not preference ordering):
- Adjacent tier pairs only (T1->T2 ... T6->T7)
- N red-team evaluations per pair (default: 10)
- Pair CLEAN if clean_count >= 8
- Pair SUSPECT_MAJ if one-sided binomial p-value <= alpha_major and suspect_rate > clean_rate
- Ladder PASS if all 6 pairs are CLEAN
- Ladder FAIL if at least K pairs are SUSPECT_MAJ (default K=4)

In summary:

This is a statistical stress test for ladder validity.

For each ladder:
1) Evaluate each adjacent step (T1->T2, T2->T3, ... T6->T7).
2) For each step, ask a balanced red-team question:
   - strongest CLEAN argument,
   - strongest SUSPECT argument,
   - final verdict.
3) Aggregate across N trials per pair.

Decision rules:
- Pair CLEAN requires all of:
  - clean_count >= clean_threshold (default: 8/10),
  - clean_rate > suspect_rate,
  - one-sided binomial p_value_clean <= alpha_clean (default: 0.05),
    where p_value_clean = P(X >= clean_count | n, p0), p0 default 0.5.
- Pair SUSPECT_MAJ requires all of:
  - suspect_rate > clean_rate,
  - one-sided binomial p_value_suspect <= alpha_major (default: 0.001),
    where p_value_suspect = P(X >= suspect_count | n, p0), p0 default 0.5.
- Ladder PASS if all 6 pairs are CLEAN.
- Ladder FAIL if SUSPECT_MAJ pairs >= ladder_fail_min_major_pairs (default: 4).
- Ladder INCONCLUSIVE otherwise; policy is configurable
  via --inconclusive-policy (default: keep).

Plain-language interpretation:
it flags ladders only when the model repeatedly finds
statistically significant objection-heavy outcomes where property increase
does not cleanly track greater choiceworthiness.

Outputs are saved under:
  data/02_validation/ladder_validation/within_ladder_validation_property/

Usage:
# Full run (default stats: CLEAN>=8 and alpha_major=0.001)
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai

  # GPT-5.5 with native reasoning on (parallel batches, auto-tuned limits)
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai --reasoning on --resume
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai --reasoning on \
  --chunk-size 40 --concurrency-limit 24 --max-parallel-chunks 4 --resume

  # Resume an interrupted run from existing raw JSONL
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai --resume

  # Smoke test (first N ladders; saves under data/02_validation/.../smoke_gpt55/)
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai --max-ladders 2
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai --start-from 10 --max-ladders 5

  # Recompute summaries only (no API calls)
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --model gpt-55-openai --analyze-only

  # Write pruned variations JSON only (no API, no aggregation)
  PYTHONPATH=src python -m llm_coherence.validation.property_ladder_pruning --write-pruned-variations-only --model gpt-55-openai

Optional flags:
  --input PATH                 Input ladder file (default: data/phase6b_variations.json)
  --output-dir PATH            Output directory (default: data/02_validation/ladder_validation/within_ladder_validation_property/<model_key>)
  --trials INT                 Red-team evaluations per adjacent pair (default: 10)
  --temperature FLOAT          Sampling temperature (default: 0.7)
  --max-tokens INT             Max response tokens per judgment (default: 600)
  --chunk-size INT             Number of requests per async API chunk (default: 80)
  --concurrency-limit INT      Max concurrent API calls (default: 40)
  --clean-threshold INT        Pair CLEAN cutoff (default: 8)
  --alpha-major FLOAT          p-value cutoff for major-suspect (default: 0.001)
  --alpha-clean FLOAT          p-value cutoff for clean (default: 0.05)
  --null-probability FLOAT     Null probability p0 in binomial test (default: 0.5)
  --ladder-fail-min-major-pairs INT  Fail ladder if >= this many major-SUSPECT pairs (default: 4)
  --inconclusive-policy {keep,prune}  What to do with INCONCLUSIVE ladders (default: keep)
  --max-ladders INT            Smoke: first N ladders; output dir smoke_<model> unless --output-dir set
  --start-from INT             Skip first N ladders before --max-ladders slice (also uses smoke dir)
  --resume                     Resume from existing property_raw_responses.jsonl
  --analyze-only               Skip API calls; aggregate existing raw output only
  --write-pruned-variations-only  Only write data/02_validation/ladder_validation/variations_pruned/phase6b_variations_prop_pruned.json
  --pruned-output PATH         Override pruned variations path (default: same dir as --input)
  --pruned-ids PATH            Pruned ladder IDs JSON (default: <output-dir>/property_pruned_ladder_ids.json)
  --reasoning {off,on}         GPT-5.x only: off=reasoning_effort none, on=high (default: from MODEL_CONFIGS)
  --max-parallel-chunks INT    Parallel API batches (default: 4 when reasoning on, else 1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


from llm_coherence.paths import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_VARIATIONS_INPUT,
    PROPERTY_OUTPUT_DIR,
    PRUNED_PROPERTY_FILENAME,
    PRUNED_PROPERTY_PATH,
)
from llm_coherence.runtime.agents import create_agent

DEFAULT_INPUT = DEFAULT_VARIATIONS_INPUT
PRUNED_VARIATIONS_FILENAME = PRUNED_PROPERTY_FILENAME
DEFAULT_OUTPUT_DIR = PROPERTY_OUTPUT_DIR
RAW_JSONL_NAME = "property_raw_responses.jsonl"
SUMMARY_JSON_NAME = "property_pruning_summary.json"
DETAILS_JSON_NAME = "property_pruning_details.json"
PRUNED_IDS_JSON_NAME = "property_pruned_ladder_ids.json"
KEPT_IDS_JSON_NAME = "property_kept_ladder_ids.json"
COST_JSON_NAME = "property_cost_log.json"
COST_SUMMARY_JSON_NAME = "cost_summary.json"


def resolve_output_file(output_dir: Path, new_name: str) -> Path:
    return output_dir / new_name

# CLI defaults (used to auto-tune reasoning-on profile when flags are left at defaults)
DEFAULT_MAX_TOKENS = 600
DEFAULT_CHUNK_SIZE = 80
DEFAULT_CONCURRENCY_LIMIT = 40
REASONING_ON_MAX_TOKENS = 2500
REASONING_ON_CHUNK_SIZE = 40
REASONING_ON_CONCURRENCY_LIMIT = 24
REASONING_ON_MAX_PARALLEL_CHUNKS = 4
REASONING_ON_BASE_TIMEOUT_S = 180

# USD per 1M tokens (input, output).
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-54": {"input": 2.50, "output": 15.00},
    "gpt-54-openai": {"input": 2.50, "output": 15.00},
    "gpt-55": {"input": 5.00, "output": 30.00},
    "gpt-55-openai": {"input": 5.00, "output": 30.00},
}
MODEL_PRICING_SOURCE: dict[str, str] = {
    "gpt-54": "https://openrouter.ai/openai/gpt-5.4",
    "gpt-54-openai": "https://openai.com/api/pricing/",
    "gpt-55": "https://developers.openai.com/api/docs/models/gpt-5.5",
    "gpt-55-openai": "https://openai.com/api/pricing/",
}


def normalize_cli_output_dir(output_dir: Path) -> Path:
    """Resolve a CLI --output-dir to an absolute path (relative paths join REPO_ROOT)."""
    output_dir = Path(output_dir)
    if output_dir.is_absolute():
        return output_dir
    return BASE_DIR / output_dir


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
    Default run folder: data/02_validation/ladder_validation/within_ladder_validation_property/<model_key>.
    Smoke runs (--max-ladders / --start-from): .../smoke_<model>/.
    """
    if output_dir is not None:
        return normalize_cli_output_dir(output_dir)
    if smoke:
        return DEFAULT_OUTPUT_DIR / smoke_output_dir_name(model_key)
    return DEFAULT_OUTPUT_DIR / model_key


def resolve_pruned_variations_output(
    input_path: Path,
    pruned_output: Optional[Path],
) -> Path:
    """Write pruned variations under data/ladder_validation/variations_pruned/."""
    del input_path  # kept for CLI compatibility
    if pruned_output is None:
        return PRUNED_PROPERTY_PATH
    pruned_output = Path(pruned_output)
    if pruned_output.is_absolute():
        return pruned_output
    return BASE_DIR / pruned_output


REASONING_CLI_TO_EFFORT = {
    "off": "none",
    "on": "high",
}


@dataclass(frozen=True)
class RunProfile:
    reasoning_effort: str
    reasoning_on: bool
    max_tokens: int
    chunk_size: int
    concurrency_limit: int
    max_parallel_chunks: int
    base_timeout: float


def effective_reasoning_effort(model_cfg: Any, reasoning_cli: Optional[str]) -> str:
    if reasoning_cli is not None:
        return REASONING_CLI_TO_EFFORT[reasoning_cli]
    extra_body = model_cfg.extra_body or {}
    return str(extra_body.get("reasoning_effort", "none")).lower()


def is_reasoning_enabled(effort: str) -> bool:
    return effort not in ("none", "", "null")


def resolve_run_profile(args: argparse.Namespace, model_cfg: Any) -> RunProfile:
    """Apply faster parallel defaults when native reasoning is enabled."""
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
    """
    Tokens billed at output rate.

    OpenAI usually includes reasoning in completion_tokens; when content is empty
    but reasoning_tokens are reported, bill those at the output rate.
    """
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


def resolve_extra_body(
    model_cfg: Any,
    reasoning: Optional[str],
) -> Optional[dict[str, Any]]:
    """
    Build extra_body for create_agent.

    Starts from MODEL_CONFIGS extra_body; --reasoning off|on overrides reasoning_effort
    for OpenAI GPT-5.x (passed through LiteLLM as extra_body.reasoning_effort).
    """
    extra_body: dict[str, Any] = dict(model_cfg.extra_body) if model_cfg.extra_body else {}
    if reasoning is not None:
        extra_body["reasoning_effort"] = REASONING_CLI_TO_EFFORT[reasoning]
    return extra_body or None


def resolve_model_pricing(model_key: str) -> tuple[Optional[dict[str, float]], str]:
    """Return per-1M token rates and pricing source URL for a model key."""
    if model_key in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[model_key], MODEL_PRICING_SOURCE.get(model_key, "")
    base_key = model_key.removesuffix("-openai")
    if base_key in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[base_key], MODEL_PRICING_SOURCE.get(base_key, "")
    return None, ""

from llm_coherence.config import MODEL_CONFIGS  # noqa: E402


PROMPT_TEMPLATE = """You are auditing the design of a normative experiment.
The experimenter claims that Scenario B is unambiguously more choiceworthy than Scenario A because it has more of [{property_name}].

Scenario A:
{scenario_a}

Scenario B:
{scenario_b}

Task:
1) Give the strongest argument that this step is CLEAN (B is a legitimate incremental improvement over A).
2) Give the strongest argument that this step is SUSPECT.
3) Decide whether the SUSPECT argument is strong enough to overturn the CLEAN case.

Decision standard (important):
- Be balanced and charitable to both sides.
- Use evidence from the scenario text; do not assume hidden confounds unless clearly implied.
- Label SUSPECT only when there is a material objection likely to change the ordering (A >= B or ambiguous).
- If objections are minor/speculative and B still plausibly improves on A, label CLEAN.

Use these concepts:
- instrumental_not_constitutive: property is only a proxy/instrument
- diminishing_returns_or_saturation: marginal value is tiny/zero/negative
- confound_drag: B adds hidden trade-offs/complexity/risk not present in A
- none: no material objection
- other: objection exists but doesn't fit above

Return ONLY valid JSON with this exact schema:
{{
  "verdict": "CLEAN" | "SUSPECT",
  "failure_mode": "instrumental_not_constitutive" | "diminishing_returns_or_saturation" | "confound_drag" | "none" | "other",
  "rationale_for_clean": "<1-2 concise sentences>",
  "rationale_for_suspect": "<1-2 concise sentences>",
  "rationale": "<1-3 concise sentences final rationale for verdict>"
}}"""


@dataclass
class RequestItem:
    request_id: str
    ladder_id: str
    category: str
    property_name: str
    valence: str
    pair_idx: int
    tier_a: int
    tier_b: int
    trial: int
    prompt: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def now_utc_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_ladders(
    input_path: Path,
    *,
    max_ladders: Optional[int] = None,
    start_from: int = 0,
) -> list[dict[str, Any]]:
    """Load ladder JSON; optionally slice for smoke tests."""
    ladders = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(ladders, list):
        raise ValueError(f"Expected JSON array in {input_path}")
    if start_from:
        ladders = ladders[start_from:]
    if max_ladders is not None:
        ladders = ladders[:max_ladders]
    return ladders


def build_requests(ladders: list[dict[str, Any]], trials: int) -> list[RequestItem]:
    requests: list[RequestItem] = []
    for ladder in ladders:
        ladder_id = ladder.get("original_statement_id", "unknown_ladder")
        category = ladder.get("category", "")
        property_name = ladder.get("identified_property", "the property")
        valence = ladder.get("valence", "unknown")
        variations = sorted(ladder.get("variations", []), key=lambda x: x.get("tier", 0))

        if len(variations) != 7:
            continue

        for pair_idx in range(6):
            a = variations[pair_idx]
            b = variations[pair_idx + 1]
            tier_a = int(a.get("tier", pair_idx + 1))
            tier_b = int(b.get("tier", pair_idx + 2))
            scenario_a = a["text"]
            scenario_b = b["text"]

            for trial in range(1, trials + 1):
                request_id = f"{ladder_id}__T{tier_a}_T{tier_b}__trial{trial}"
                prompt = PROMPT_TEMPLATE.format(
                    property_name=property_name,
                    scenario_a=scenario_a,
                    scenario_b=scenario_b,
                )
                requests.append(
                    RequestItem(
                        request_id=request_id,
                        ladder_id=ladder_id,
                        category=category,
                        property_name=property_name,
                        valence=valence,
                        pair_idx=pair_idx,
                        tier_a=tier_a,
                        tier_b=tier_b,
                        trial=trial,
                        prompt=prompt,
                    )
                )
    return requests


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_property_audit_response(raw_text: str) -> dict[str, Any]:
    parsed = _extract_json_object(raw_text)
    if not parsed:
        up = (raw_text or "").upper()
        if "SUSPECT" in up:
            return {
                "verdict": "SUSPECT",
                "failure_mode": "other",
                "rationale": raw_text.strip()[:600],
                "rationale_for_clean": "",
                "rationale_for_suspect": raw_text.strip()[:300],
                "parse_ok": False,
            }
        if "CLEAN" in up:
            return {
                "verdict": "CLEAN",
                "failure_mode": "none",
                "rationale": raw_text.strip()[:600],
                "rationale_for_clean": raw_text.strip()[:300],
                "rationale_for_suspect": "",
                "parse_ok": False,
            }
        return {
            "verdict": "UNPARSEABLE",
            "failure_mode": "other",
            "rationale": raw_text.strip()[:600],
            "rationale_for_clean": "",
            "rationale_for_suspect": "",
            "parse_ok": False,
        }

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in {"CLEAN", "SUSPECT"}:
        raw_dump = json.dumps(parsed).upper()
        if "SUSPECT" in raw_dump:
            verdict = "SUSPECT"
        elif "CLEAN" in raw_dump:
            verdict = "CLEAN"
        else:
            verdict = "UNPARSEABLE"

    failure_mode = str(parsed.get("failure_mode", "other")).strip().lower() or "other"
    if failure_mode not in {
        "instrumental_not_constitutive",
        "diminishing_returns_or_saturation",
        "confound_drag",
        "none",
        "other",
    }:
        failure_mode = "other"
    rationale_for_clean = str(parsed.get("rationale_for_clean", "")).strip()
    rationale_for_suspect = str(parsed.get("rationale_for_suspect", "")).strip()
    rationale = str(parsed.get("rationale", "")).strip()
    if not rationale:
        rationale = (raw_text or "").strip()[:600]

    return {
        "verdict": verdict,
        "failure_mode": failure_mode,
        "rationale_for_clean": rationale_for_clean,
        "rationale_for_suspect": rationale_for_suspect,
        "rationale": rationale,
        "parse_ok": verdict in {"CLEAN", "SUSPECT"},
    }


async def _process_request_chunk(
    chunk: list[RequestItem],
    *,
    model_key: str,
    temperature: float,
    max_tokens: int,
    concurrency_limit: int,
    extra_body: Optional[dict[str, Any]],
    enable_cache: bool,
    system_message: Optional[str],
    base_timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one API batch; fresh agent per chunk for isolated usage accounting."""
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
    for req in chunk:
        msg_list: list[dict[str, str]] = []
        if system_message:
            msg_list.append({"role": "system", "content": system_message})
        msg_list.append({"role": "user", "content": req.prompt})
        messages.append(msg_list)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings",
            category=UserWarning,
        )
        responses = await agent.async_completions(messages, verbose=False)

    if len(responses) != len(chunk):
        raise RuntimeError(
            f"API returned {len(responses)} responses for {len(chunk)} requests in batch"
        )

    rows: list[dict[str, Any]] = []
    for req, resp in zip(chunk, responses):
        parsed = parse_property_audit_response(resp or "")
        rows.append(
            {
                "request_id": req.request_id,
                "ladder_id": req.ladder_id,
                "category": req.category,
                "property_name": req.property_name,
                "valence": req.valence,
                "pair_idx": req.pair_idx,
                "tier_a": req.tier_a,
                "tier_b": req.tier_b,
                "trial": req.trial,
                "verdict": parsed["verdict"],
                "failure_mode": parsed["failure_mode"],
                "rationale_for_clean": parsed["rationale_for_clean"],
                "rationale_for_suspect": parsed["rationale_for_suspect"],
                "rationale": parsed["rationale"],
                "parse_ok": parsed["parse_ok"],
                "raw_response": (resp or "")[:4000],
                "timestamp": now_iso(),
            }
        )
    return rows, list(getattr(agent, "usage_log", []))


async def run_requests(
    model_key: str,
    requests: list[RequestItem],
    output_dir: Path,
    chunk_size: int,
    temperature: float,
    max_tokens: int,
    concurrency_limit: int,
    resume: bool,
    reasoning: Optional[str] = None,
    max_parallel_chunks: int = 1,
    base_timeout: float = 30.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    read_raw_path = resolve_output_file(output_dir, RAW_JSONL_NAME)
    write_raw_path = output_dir / RAW_JSONL_NAME

    done_ids: set[str] = set()
    if resume and read_raw_path.exists():
        with read_raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    rid = row.get("request_id")
                    if rid:
                        done_ids.add(rid)
                except json.JSONDecodeError:
                    continue

    pending = [r for r in requests if r.request_id not in done_ids]
    if not pending:
        print("All requests already completed; skipping API calls.")
        return

    model_cfg = MODEL_CONFIGS.get(model_key)
    if model_cfg is None:
        raise ValueError(f"Unknown model_key: {model_key}")

    extra_body = resolve_extra_body(model_cfg, reasoning)
    enable_cache = model_cfg.enable_cache
    system_message = model_cfg.system_message
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
    read_cost_path = resolve_output_file(output_dir, COST_JSON_NAME)
    write_cost_path = output_dir / COST_JSON_NAME
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
    if resume and read_cost_path.exists():
        try:
            prev = json.loads(read_cost_path.read_text(encoding="utf-8"))
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
            rows, usage_records = await _process_request_chunk(
                chunk,
                model_key=model_key,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency_limit=concurrency_limit,
                extra_body=extra_body,
                enable_cache=enable_cache,
                system_message=system_message,
                base_timeout=base_timeout,
            )
            return start, rows, usage_records

    mode = "a" if write_raw_path.exists() and resume else "w"
    with write_raw_path.open(mode, encoding="utf-8") as out:
        # Process in waves of parallel chunks; preserve JSONL order by start index.
        wave_size = max(1, max_parallel_chunks)
        for wave_base in range(0, len(chunk_starts), wave_size):
            wave = chunk_starts[wave_base : wave_base + wave_size]
            batch_results = await asyncio.gather(*(run_one_chunk(start) for start in wave))
            for start, rows, usage_records in sorted(batch_results, key=lambda x: x[0]):
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                append_usage_to_cost_log(cost_log, usage_records, rates)
                cost_log["updated_at"] = now_iso()
                write_cost_path.write_text(
                    json.dumps(cost_log, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                completed += len(rows)
                print(f"Processed {completed}/{total} pending requests")

def build_cost_summary_from_log(cost_log: dict[str, Any]) -> dict[str, Any]:
    """Produce a cost_summary.py-style summary from property cost log."""
    model_key = cost_log.get("model_key")
    estimated_total = float(cost_log.get("estimated_cost_usd") or 0.0)
    records = cost_log.get("records") or []
    by_variant_entry = {
        "variant": model_key,
        "n_ladders": 0,
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
    summary = {
        "generated_at": now_utc_z(),
        "schema_note": (
            "Red-team ladder pruning run cost summary. Structured to mirror "
            "cost_summary.py output fields where applicable. "
            "This run logs per-call token usage from LiteLLM usage_log and "
            "computes estimated_cost_usd from configured per-1M token rates."
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
        "by_variant": [by_variant_entry],
        "records": records,
    }
    return summary


def _recompute_cost_estimates(output_dir: Path, model_key: str) -> None:
    """Backfill per-call and total estimated_cost_usd when pricing is available."""
    cost_path = resolve_output_file(output_dir, COST_JSON_NAME)
    write_cost_path = output_dir / COST_JSON_NAME
    if not cost_path.exists():
        return
    rates, pricing_source = resolve_model_pricing(model_key)
    if not rates:
        return

    cost_log = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_log["model_key"] = model_key
    cost_log["pricing_source"] = pricing_source
    cost_log["pricing_per_1m"] = rates

    prompt_total = 0
    completion_total = 0
    reasoning_total = 0
    for rec in cost_log.get("records") or []:
        prompt_tokens = int(rec.get("prompt_tokens") or 0)
        completion_tokens = int(rec.get("completion_tokens") or 0)
        reasoning_tokens = int(rec.get("reasoning_tokens") or 0)
        prompt_total += prompt_tokens
        completion_total += completion_tokens
        reasoning_total += reasoning_tokens
        billable_output = billable_output_tokens(completion_tokens, reasoning_tokens)
        rec["billable_output_tokens"] = billable_output
        rec["estimated_cost_usd"] = estimate_call_cost_usd(
            prompt_tokens, completion_tokens, reasoning_tokens, rates
        )

    cost_log["prompt_tokens_total"] = prompt_total
    cost_log["completion_tokens_total"] = completion_total
    cost_log["reasoning_tokens_total"] = reasoning_total
    refresh_cost_log_total(cost_log, rates)
    cost_log["updated_at"] = now_iso()
    write_cost_path.write_text(json.dumps(cost_log, indent=2, ensure_ascii=False), encoding="utf-8")


def save_cost_summary(output_dir: Path) -> None:
    cost_path = resolve_output_file(output_dir, COST_JSON_NAME)
    if not cost_path.exists():
        return
    cost_log = json.loads(cost_path.read_text(encoding="utf-8"))
    summary = build_cost_summary_from_log(cost_log)
    (output_dir / COST_SUMMARY_JSON_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def one_sided_binom_sf(k: int, n: int, p0: float) -> float:
    """Survival function P(X >= k) for X~Binomial(n,p0), exact sum."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p0**i) * ((1 - p0) ** (n - i))
    return min(1.0, max(0.0, total))


def explain_pair_status(
    *,
    pair_clean: bool,
    pair_suspect_major: bool,
    clean_count: int,
    suspect_count: int,
    n_trials_observed: int,
    clean_threshold: int,
    clean_rate: float,
    suspect_rate: float,
    p_value_clean: float,
    p_value_suspect: float,
    alpha_clean: float,
    alpha_major: float,
) -> str:
    """Plain-language rationale for a pair's CLEAN / SUSPECT_MAJ classification."""
    parts: list[str] = [
        f"Observed {clean_count} CLEAN and {suspect_count} SUSPECT "
        f"over {n_trials_observed} trial(s)."
    ]
    if pair_clean:
        parts.append(
            f"Pair is CLEAN: clean_count>={clean_threshold}, "
            f"clean_rate ({clean_rate:.2f}) > suspect_rate ({suspect_rate:.2f}), "
            f"and p_value_clean ({p_value_clean:.6g}) <= alpha_clean ({alpha_clean})."
        )
    else:
        reasons: list[str] = []
        if clean_count < clean_threshold:
            reasons.append(f"clean_count ({clean_count}) < threshold ({clean_threshold})")
        if not (clean_rate > suspect_rate):
            reasons.append(
                f"clean_rate ({clean_rate:.2f}) is not greater than suspect_rate ({suspect_rate:.2f})"
            )
        if not (p_value_clean <= alpha_clean):
            reasons.append(
                f"p_value_clean ({p_value_clean:.6g}) > alpha_clean ({alpha_clean})"
            )
        parts.append("Pair is not CLEAN because " + "; ".join(reasons) + ".")

    if pair_suspect_major:
        parts.append(
            f"Pair is major-SUSPECT: suspect_rate > clean_rate and "
            f"p_value_suspect ({p_value_suspect:.6g}) <= alpha_major ({alpha_major}). "
            "This pair counts toward ladder FAIL."
        )
    else:
        reasons: list[str] = []
        if not (suspect_rate > clean_rate):
            reasons.append("suspect_rate does not exceed clean_rate")
        if not (p_value_suspect <= alpha_major):
            reasons.append(
                f"p_value_suspect ({p_value_suspect:.6g}) > alpha_major ({alpha_major})"
            )
        parts.append(
            "Pair is not major-SUSPECT because " + "; ".join(reasons) + ". "
            "This pair alone does not force ladder pruning."
        )
    return " ".join(parts)


def explain_ladder_pruning(
    *,
    ladder_status: str,
    pruning_decision: str,
    ladder_major_suspect_pairs: int,
    n_pairs_clean: int,
    inconclusive_policy: str,
    ladder_fail_min_major_pairs: int,
) -> str:
    """Plain-language rationale for ladder KEEP vs PRUNE."""
    if ladder_status == "PASS":
        return (
            f"Ladder PASS: all 6 adjacent pairs are CLEAN. "
            f"Pruning decision: {pruning_decision}."
        )
    if ladder_status == "FAIL":
        return (
            f"Ladder FAIL: {ladder_major_suspect_pairs} pair(s) are major-SUSPECT "
            f"(>= {ladder_fail_min_major_pairs} required). "
            f"Pruning decision: {pruning_decision}."
        )
    if inconclusive_policy == "keep":
        return (
            f"Ladder INCONCLUSIVE: {n_pairs_clean}/6 pairs are CLEAN and "
            f"{ladder_major_suspect_pairs} pair(s) are major-SUSPECT "
            f"(< {ladder_fail_min_major_pairs} needed to FAIL). "
            "Not enough evidence to PASS or FAIL under current thresholds. "
            f"inconclusive_policy=keep, so the ladder is kept."
        )
    return (
        f"Ladder INCONCLUSIVE: {n_pairs_clean}/6 pairs are CLEAN and "
        f"{ladder_major_suspect_pairs} pair(s) are major-SUSPECT. "
        f"inconclusive_policy=prune, so the ladder is pruned."
    )


def aggregate_results(
    ladders: list[dict[str, Any]],
    output_dir: Path,
    trials: int,
    clean_threshold: int,
    alpha_major: float,
    alpha_clean: float,
    null_probability: float,
    ladder_fail_min_major_pairs: int,
    inconclusive_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = resolve_output_file(output_dir, RAW_JSONL_NAME)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw responses file: {raw_path}")

    rows: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_pair: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault((r["ladder_id"], int(r["pair_idx"])), []).append(r)

    ladder_meta = {
        l["original_statement_id"]: {
            "category": l.get("category", ""),
            "property_name": l.get("identified_property", ""),
            "valence": l.get("valence", ""),
        }
        for l in ladders
        if "original_statement_id" in l
    }

    ladder_results: list[dict[str, Any]] = []
    kept_ids: list[str] = []
    pruned_ids: list[str] = []

    for ladder_id, meta in ladder_meta.items():
        pair_summaries: list[dict[str, Any]] = []
        ladder_major_suspect_pairs = 0
        ladder_all_clean = True
        for pair_idx in range(6):
            recs = by_pair.get((ladder_id, pair_idx), [])
            clean_count = sum(1 for x in recs if x.get("verdict") == "CLEAN")
            suspect_count = sum(1 for x in recs if x.get("verdict") == "SUSPECT")
            n_trials_observed = len(recs)
            parse_errors = sum(1 for x in recs if x.get("verdict") == "UNPARSEABLE")

            clean_rate = (clean_count / n_trials_observed) if n_trials_observed else 0.0
            suspect_rate = (suspect_count / n_trials_observed) if n_trials_observed else 0.0
            p_value_clean = one_sided_binom_sf(clean_count, n_trials_observed, null_probability)
            p_value_suspect = one_sided_binom_sf(suspect_count, n_trials_observed, null_probability)
            confidence_clean = 1.0 - p_value_clean
            confidence_suspect = 1.0 - p_value_suspect

            pair_clean = clean_count >= clean_threshold
            pair_suspect_major = (
                suspect_rate > clean_rate and p_value_suspect <= alpha_major
            )
            if pair_clean and not (clean_rate > suspect_rate and p_value_clean <= alpha_clean):
                pair_clean = False

            if pair_suspect_major:
                ladder_major_suspect_pairs += 1
            if not pair_clean:
                ladder_all_clean = False

            failure_modes: dict[str, int] = {}
            for x in recs:
                if x.get("verdict") == "SUSPECT":
                    fm = x.get("failure_mode", "other")
                    failure_modes[fm] = failure_modes.get(fm, 0) + 1

            pair_summaries.append(
                {
                    "pair_idx": pair_idx,
                    "tier_a": pair_idx + 1,
                    "tier_b": pair_idx + 2,
                    "n_trials_observed": n_trials_observed,
                    "n_trials_expected": trials,
                    "clean_count": clean_count,
                    "suspect_count": suspect_count,
                    "clean_rate": clean_rate,
                    "suspect_rate": suspect_rate,
                    "p_value_clean": p_value_clean,
                    "p_value_suspect": p_value_suspect,
                    "confidence_clean": confidence_clean,
                    "confidence_suspect": confidence_suspect,
                    "parse_errors": parse_errors,
                    "pair_clean": pair_clean,
                    "pair_suspect_major": pair_suspect_major,
                    "pair_status_explanation": explain_pair_status(
                        pair_clean=pair_clean,
                        pair_suspect_major=pair_suspect_major,
                        clean_count=clean_count,
                        suspect_count=suspect_count,
                        n_trials_observed=n_trials_observed,
                        clean_threshold=clean_threshold,
                        clean_rate=clean_rate,
                        suspect_rate=suspect_rate,
                        p_value_clean=p_value_clean,
                        p_value_suspect=p_value_suspect,
                        alpha_clean=alpha_clean,
                        alpha_major=alpha_major,
                    ),
                    "failure_modes": failure_modes,
                }
            )

        if ladder_major_suspect_pairs >= ladder_fail_min_major_pairs:
            ladder_status = "FAIL"
            pruning_decision = "PRUNE"
            pruned_ids.append(ladder_id)
        elif ladder_all_clean:
            ladder_status = "PASS"
            pruning_decision = "KEEP"
            kept_ids.append(ladder_id)
        else:
            ladder_status = "INCONCLUSIVE"
            if inconclusive_policy == "prune":
                pruning_decision = "PRUNE"
                pruned_ids.append(ladder_id)
            else:
                pruning_decision = "KEEP"
                kept_ids.append(ladder_id)

        n_pairs_clean = sum(1 for p in pair_summaries if p["pair_clean"])
        ladder_results.append(
            {
                "ladder_id": ladder_id,
                "category": meta["category"],
                "property_name": meta["property_name"],
                "valence": meta["valence"],
                "status": ladder_status,
                "pruning_decision": pruning_decision,
                "ladder_major_suspect_pairs": ladder_major_suspect_pairs,
                "n_pairs_clean": n_pairs_clean,
                "pruning_explanation": explain_ladder_pruning(
                    ladder_status=ladder_status,
                    pruning_decision=pruning_decision,
                    ladder_major_suspect_pairs=ladder_major_suspect_pairs,
                    n_pairs_clean=n_pairs_clean,
                    inconclusive_policy=inconclusive_policy,
                    ladder_fail_min_major_pairs=ladder_fail_min_major_pairs,
                ),
                "pairs": pair_summaries,
            }
        )

    ladder_results.sort(key=lambda x: (x["pruning_decision"], x["ladder_id"]))

    counts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
    for lr in ladder_results:
        counts[lr["status"]] += 1

    summary = {
        "generated_at": now_iso(),
        "n_ladders": len(ladder_results),
        "counts": counts,
        "n_kept": len(kept_ids),
        "n_pruned": len(pruned_ids),
        "kept_fraction": (len(kept_ids) / len(ladder_results)) if ladder_results else 0.0,
        "pruned_fraction": (len(pruned_ids) / len(ladder_results)) if ladder_results else 0.0,
        "criteria": {
            "pair_clean_if_clean_count_at_least": clean_threshold,
            "pair_major_suspect_if_p_value_suspect_at_most": alpha_major,
            "pair_clean_requires_p_value_clean_at_most": alpha_clean,
            "null_probability_for_binomial_test": null_probability,
            "ladder_pass_if_all_pairs_clean": True,
            "ladder_fail_if_major_suspect_pairs_at_least": ladder_fail_min_major_pairs,
            "inconclusive_policy": inconclusive_policy,
        },
    }

    details = {
        "generated_at": now_iso(),
        "summary": summary,
        "ladders": ladder_results,
        "kept_ladder_ids": kept_ids,
        "pruned_ladder_ids": pruned_ids,
    }
    return summary, details


def resolve_pruned_ids_path(
    *,
    pruned_ids: Optional[Path],
    output_dir: Optional[Path],
    model_key: Optional[str],
) -> Path:
    if pruned_ids is not None:
        return pruned_ids
    if output_dir is not None:
        return resolve_output_file(output_dir, PRUNED_IDS_JSON_NAME)
    if model_key is not None:
        out = resolve_output_dir(model_key, None)
        return resolve_output_file(out, PRUNED_IDS_JSON_NAME)
    raise ValueError(
        "Provide --pruned-ids, --output-dir, or --model to locate property_pruned_ladder_ids.json"
    )


def write_pruned_variations_file(
    input_path: Path,
    pruned_ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Remove pruned ladder IDs from phase6b_variations and write kept ladders only."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input variations file not found: {input_path}")
    if not pruned_ids_path.exists():
        raise FileNotFoundError(
            f"Pruned IDs file not found: {pruned_ids_path}\n"
            "Tip: use forward slashes for --output-dir "
            "(e.g. data/02_validation/ladder_validation/within_ladder_validation_property/gpt-55-openai), "
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
    output_path.write_text(
        json.dumps(kept_ladders, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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


def save_outputs(output_dir: Path, summary: dict[str, Any], details: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_JSON_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / DETAILS_JSON_NAME).write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / KEPT_IDS_JSON_NAME).write_text(
        json.dumps(details["kept_ladder_ids"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / PRUNED_IDS_JSON_NAME).write_text(
        json.dumps(details["pruned_ladder_ids"], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Property ladder pruning audit for phase6b ladders (CLEAN vs SUSPECT)."
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Model key from MODEL_CONFIGS (required unless --write-pruned-variations-only with --output-dir)",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to phase6b_variations.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: "
            "data/02_validation/ladder_validation/within_ladder_validation_property/<model_key>)"
        ),
    )
    parser.add_argument("--trials", type=int, default=10, help="N evaluations per adjacent pair")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens per judgment response (auto 2500 when reasoning on)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Requests per API batch (auto 40 when reasoning on)",
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=DEFAULT_CONCURRENCY_LIMIT,
        help="Max concurrent API calls per batch (auto 24 when reasoning on)",
    )
    parser.add_argument(
        "--max-parallel-chunks",
        type=int,
        default=None,
        help="Parallel API batches in flight (default: 4 when reasoning on, else 1)",
    )
    parser.add_argument("--clean-threshold", type=int, default=8, help="Pair CLEAN cutoff")
    parser.add_argument(
        "--alpha-major",
        type=float,
        default=0.001,
        help="p-value cutoff for pair major-SUSPECT (one-sided binomial)",
    )
    parser.add_argument(
        "--alpha-clean",
        type=float,
        default=0.05,
        help="p-value cutoff for pair CLEAN (one-sided binomial)",
    )
    parser.add_argument(
        "--null-probability",
        type=float,
        default=0.5,
        help="Null probability p0 for binomial test",
    )
    parser.add_argument(
        "--inconclusive-policy",
        choices=["keep", "prune"],
        default="keep",
        help="How to handle INCONCLUSIVE ladders",
    )
    parser.add_argument(
        "--ladder-fail-min-major-pairs",
        type=int,
        default=4,
        help="Fail ladder only if this many pairs are major-SUSPECT",
    )
    parser.add_argument(
        "--max-ladders",
        type=int,
        default=None,
        help=(
            "Maximum number of ladders to evaluate (smoke tests). "
            "Default output: data/02_validation/ladder_validation/within_ladder_validation_property/smoke_<model>/ "
            "(e.g. smoke_gpt55). Skips writing pruned variations JSON."
        ),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Index to start from before applying --max-ladders",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing raw jsonl file")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip API calls, only aggregate from existing raw jsonl",
    )
    parser.add_argument(
        "--write-pruned-variations-only",
        action="store_true",
        help="Only write phase6b_variations_prop_pruned.json from pruned IDs (no API/aggregation)",
    )
    parser.add_argument(
        "--pruned-output",
        type=Path,
        default=None,
        help=(
            "Output path for kept ladders after pruning "
            f"(default: <input-dir>/{PRUNED_VARIATIONS_FILENAME})"
        ),
    )
    parser.add_argument(
        "--pruned-ids",
        type=Path,
        default=None,
        help="Path to property_pruned_ladder_ids.json (default: <output-dir>/property_pruned_ladder_ids.json)",
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "on"],
        default=None,
        help=(
            "OpenAI GPT-5.x: override native reasoning (off=reasoning_effort none, on=high). "
            "Default: use MODEL_CONFIGS (gpt-55-openai is off/none)."
        ),
    )
    return parser.parse_args()


def run_write_pruned_variations_only(args: argparse.Namespace) -> int:
    output_dir = (
        normalize_cli_output_dir(args.output_dir) if args.output_dir is not None else None
    )
    pruned_ids_path = resolve_pruned_ids_path(
        pruned_ids=args.pruned_ids,
        output_dir=output_dir,
        model_key=args.model,
    )
    pruned_output = resolve_pruned_variations_output(args.input, args.pruned_output)
    meta = write_pruned_variations_file(
        input_path=args.input,
        pruned_ids_path=pruned_ids_path,
        output_path=pruned_output,
    )
    print(f"Wrote pruned variations: {meta['output_path']}")
    print(f"  Input ladders: {meta['n_input_ladders']}")
    print(f"  Pruned IDs: {meta['n_pruned_ids']}")
    print(f"  Output ladders: {meta['n_output_ladders']}")
    print(f"  Removed: {meta['n_removed']}")
    if meta["pruned_ids_not_in_input"]:
        print(f"  WARNING: {len(meta['pruned_ids_not_in_input'])} pruned ID(s) not found in input")
    return 0


async def amain() -> int:
    args = parse_args()

    if args.write_pruned_variations_only:
        if args.analyze_only:
            raise ValueError("Use only one of --write-pruned-variations-only or --analyze-only")
        return run_write_pruned_variations_only(args)

    if args.model is None:
        raise ValueError("--model is required unless using --write-pruned-variations-only")
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{args.model}'. Available keys: {sorted(MODEL_CONFIGS.keys())}")
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if not (0.0 < args.alpha_major <= 1.0):
        raise ValueError("--alpha-major must be in (0, 1]")
    if not (0.0 < args.alpha_clean <= 1.0):
        raise ValueError("--alpha-clean must be in (0, 1]")
    if not (0.0 < args.null_probability < 1.0):
        raise ValueError("--null-probability must be in (0, 1)")
    if args.ladder_fail_min_major_pairs < 1 or args.ladder_fail_min_major_pairs > 6:
        raise ValueError("--ladder-fail-min-major-pairs must be between 1 and 6")
    if args.clean_threshold > args.trials:
        print(
            f"WARNING: clean threshold ({args.clean_threshold}) > trials ({args.trials}); "
            "no pair can be CLEAN with these settings."
        )
    if args.start_from < 0:
        raise ValueError("--start-from must be >= 0")
    if args.max_ladders is not None and args.max_ladders <= 0:
        raise ValueError("--max-ladders must be > 0")

    smoke_run = args.max_ladders is not None or args.start_from > 0
    output_dir = resolve_output_dir(args.model, args.output_dir, smoke=smoke_run)
    ladders = load_ladders(
        args.input,
        max_ladders=args.max_ladders,
        start_from=args.start_from,
    )
    requests = build_requests(ladders, trials=args.trials)
    if args.max_ladders is not None or args.start_from:
        print(
            f"Ladder slice: start_from={args.start_from}, "
            f"max_ladders={args.max_ladders if args.max_ladders is not None else 'all'}"
        )
    print(f"Ladders loaded: {len(ladders)}")
    print(f"Requests planned: {len(requests)} ({len(ladders)} ladders × 6 pairs × {args.trials} trials)")
    print(f"Output dir: {output_dir}")

    model_cfg = MODEL_CONFIGS[args.model]
    run_profile = resolve_run_profile(args, model_cfg)
    if args.reasoning and args.model and "gpt-5" not in args.model and "gpt-54" not in args.model:
        print(
            f"WARNING: --reasoning {args.reasoning!r} is intended for OpenAI GPT-5.x models; "
            f"model={args.model!r} may ignore reasoning_effort."
        )
    if run_profile.reasoning_on:
        print(
            "Reasoning-on run profile: parallel batches enabled "
            f"(max_parallel_chunks={run_profile.max_parallel_chunks})."
        )

    if not args.analyze_only:
        await run_requests(
            model_key=args.model,
            requests=requests,
            output_dir=output_dir,
            chunk_size=run_profile.chunk_size,
            temperature=args.temperature,
            max_tokens=run_profile.max_tokens,
            concurrency_limit=run_profile.concurrency_limit,
            resume=args.resume,
            reasoning=args.reasoning,
            max_parallel_chunks=run_profile.max_parallel_chunks,
            base_timeout=run_profile.base_timeout,
        )

    summary, details = aggregate_results(
        ladders=ladders,
        output_dir=output_dir,
        trials=args.trials,
        clean_threshold=args.clean_threshold,
        alpha_major=args.alpha_major,
        alpha_clean=args.alpha_clean,
        null_probability=args.null_probability,
        ladder_fail_min_major_pairs=args.ladder_fail_min_major_pairs,
        inconclusive_policy=args.inconclusive_policy,
    )
    save_outputs(output_dir, summary, details)
    save_cost_summary(output_dir)
    _recompute_cost_estimates(output_dir, args.model)
    save_cost_summary(output_dir)

    print(
        f"Done. PASS={summary['counts']['PASS']} "
        f"FAIL={summary['counts']['FAIL']} "
        f"INCONCLUSIVE={summary['counts']['INCONCLUSIVE']}"
    )
    print(f"Kept ladders: {summary['n_kept']}")
    print(f"Pruned ladders: {summary['n_pruned']}")
    print(f"Summary: {output_dir / SUMMARY_JSON_NAME}")
    print(f"Details: {output_dir / DETAILS_JSON_NAME}")
    print(f"Pruned IDs: {output_dir / PRUNED_IDS_JSON_NAME}")
    print(f"Cost Summary: {output_dir / COST_SUMMARY_JSON_NAME}")

    if args.max_ladders is not None or args.start_from:
        print(
            "Skipping write of pruned variations JSON "
            "(use a full run without --max-ladders / --start-from to update "
            f"{PRUNED_VARIATIONS_FILENAME})."
        )
    else:
        pruned_output = resolve_pruned_variations_output(args.input, args.pruned_output)
        pruned_meta = write_pruned_variations_file(
            input_path=args.input,
            pruned_ids_path=output_dir / PRUNED_IDS_JSON_NAME,
            output_path=pruned_output,
        )
        print(f"Pruned variations file: {pruned_meta['output_path']}")
        print(
            f"  Kept {pruned_meta['n_output_ladders']}/{pruned_meta['n_input_ladders']} ladders "
            f"(removed {pruned_meta['n_removed']})"
        )
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
