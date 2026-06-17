"""API cost tracking: per-request pricing, roll-ups, and experiment cost artifacts.

Per-request priority:
  1. OpenRouter ``usage.cost`` when present (billed USD).
  2. Token-based cost from live OpenRouter ``/models`` rates.
  3. Token-based cost from OpenAI / Anthropic published rates (batch discount optional).

Roll-up writers produce:
  - ``cost_log.json`` — per-request detail (within-ladder batch/live)
  - ``phase6b_cost_log.json`` — aggregated records plus embedded ``summary``
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Standard (live API) USD per 1M tokens — keep in sync with preflight_check.py.
OPENAI_STANDARD_PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-5.4-2026-03-05": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano-2026-03-17": {"input": 0.20, "output": 1.25},
}

OPENAI_BATCH_PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    model: {"input": r["input"] * 0.5, "output": r["output"] * 0.5}
    for model, r in OPENAI_STANDARD_PRICE_PER_MTOK.items()
}

ANTHROPIC_STANDARD_PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
}

ANTHROPIC_BATCH_PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    model: {"input": r["input"] * 0.5, "output": r["output"] * 0.5}
    for model, r in ANTHROPIC_STANDARD_PRICE_PER_MTOK.items()
}

OPENAI_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.4": "gpt-5.4-2026-03-05",
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano": "gpt-5.4-nano-2026-03-17",
}

PRICING_SOURCE_URL = {
    "openai": "https://developers.openai.com/api/docs/pricing",
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "openrouter": "https://openrouter.ai/api/v1/models",
}

_OPENROUTER_CACHE: dict[str, Any] = {"fetched_at": 0.0, "models": {}}
_OPENROUTER_CACHE_TTL_S = 3600


def _resolve_openai_model_id(model_id: str, table: dict[str, dict[str, float]]) -> str | None:
    if model_id in table:
        return model_id
    aliased = OPENAI_MODEL_ALIASES.get(model_id)
    if aliased and aliased in table:
        return aliased
    for key in table:
        if model_id.startswith(key) or key.startswith(model_id):
            return key
    return None


def _strip_openrouter_prefix(model_id: str) -> str:
    prefix = "openrouter/"
    return model_id[len(prefix):] if model_id.startswith(prefix) else model_id


def infer_provider(model_id: str | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if not model_id:
        return "openai"
    mid = model_id.lower()
    if mid.startswith("openrouter/") or "/" in mid and not mid.startswith("gpt-"):
        return "openrouter"
    if "claude" in mid or "anthropic" in mid:
        return "anthropic"
    if mid.startswith("gpt-") or mid.startswith("o1") or mid.startswith("o3"):
        return "openai"
    return "openrouter"


def resolve_rates(
    provider: str,
    model_id: str | None,
    *,
    batch: bool = False,
) -> tuple[dict[str, float] | None, str]:
    """Return ``{input, output}`` USD/Mtok and a human-readable source label."""
    if not model_id:
        return None, "unknown_model"

    if provider == "openai":
        table = OPENAI_BATCH_PRICE_PER_MTOK if batch else OPENAI_STANDARD_PRICE_PER_MTOK
        resolved = _resolve_openai_model_id(model_id, table) if model_id else None
        rates = table.get(resolved) if resolved else None
        label = f"openai_{'batch' if batch else 'standard'}:{PRICING_SOURCE_URL['openai']}"
        return rates, label

    if provider == "anthropic":
        table = ANTHROPIC_BATCH_PRICE_PER_MTOK if batch else ANTHROPIC_STANDARD_PRICE_PER_MTOK
        rates = table.get(model_id)
        label = f"anthropic_{'batch' if batch else 'standard'}:{PRICING_SOURCE_URL['anthropic']}"
        return rates, label

    if provider == "openrouter":
        slug = _strip_openrouter_prefix(model_id)
        live = fetch_openrouter_rates(slug)
        if live:
            return live, f"openrouter_api/models:{slug}"
        return None, f"openrouter_api/models:{slug} (not found)"

    return None, provider


def compute_cost_from_tokens(
    rates: dict[str, float],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    openai_cached_tokens: int = 0,
) -> float:
    """USD for one request from token counts and $/Mtok rates."""
    uncached = max(
        prompt_tokens - cache_creation_input_tokens - cache_read_input_tokens - openai_cached_tokens,
        0,
    )
    in_rate = rates["input"]
    out_rate = rates["output"]
    return round(
        uncached / 1_000_000 * in_rate
        + cache_creation_input_tokens / 1_000_000 * in_rate * 1.25
        + cache_read_input_tokens / 1_000_000 * in_rate * 0.10
        + openai_cached_tokens / 1_000_000 * in_rate * 0.10
        + completion_tokens / 1_000_000 * out_rate,
        8,
    )


def compute_provider_cost_usd(
    provider: str,
    model_id: str | None,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    openai_cached_tokens: int = 0,
    batch: bool = False,
) -> tuple[float | None, str | None]:
    rates, source = resolve_rates(provider, model_id, batch=batch)
    if not rates:
        return None, None
    return (
        compute_cost_from_tokens(
            rates,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            openai_cached_tokens=openai_cached_tokens,
        ),
        source,
    )


def fetch_openrouter_rates(model_slug: str) -> dict[str, float] | None:
    """Live $/Mtok from OpenRouter ``GET /api/v1/models`` (cached 1h)."""
    slug = _strip_openrouter_prefix(model_slug)
    now = time.time()
    if now - _OPENROUTER_CACHE["fetched_at"] > _OPENROUTER_CACHE_TTL_S:
        _refresh_openrouter_cache()
    entry = _OPENROUTER_CACHE["models"].get(slug)
    if not entry:
        return None
    return {"input": entry["input"], "output": entry["output"]}


def _refresh_openrouter_cache() -> None:
    from llm_coherence.runtime.api_keys import load_api_key

    models: dict[str, dict[str, float]] = {}
    try:
        import urllib.request

        headers = {}
        key = load_api_key("openrouter")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for m in payload.get("data") or []:
            mid = m.get("id")
            pricing = m.get("pricing") or {}
            if not mid or not pricing:
                continue
            try:
                prompt = float(pricing.get("prompt", 0)) * 1_000_000
                completion = float(pricing.get("completion", 0)) * 1_000_000
            except (TypeError, ValueError):
                continue
            models[mid] = {"input": prompt, "output": completion}
        _OPENROUTER_CACHE["fetched_at"] = time.time()
        _OPENROUTER_CACHE["models"] = models
    except Exception:
        if not _OPENROUTER_CACHE["models"]:
            from llm_coherence.paths import REPO_ROOT

            cache_file = REPO_ROOT / "openrouter_models.json"
            if cache_file.is_file():
                try:
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
                    for m in payload.get("data") or []:
                        mid = m.get("id")
                        pricing = m.get("pricing") or {}
                        if not mid or not pricing:
                            continue
                        models[mid] = {
                            "input": float(pricing.get("prompt", 0)) * 1_000_000,
                            "output": float(pricing.get("completion", 0)) * 1_000_000,
                        }
                    _OPENROUTER_CACHE["models"] = models
                    _OPENROUTER_CACHE["fetched_at"] = time.time()
                except Exception:
                    pass


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Normalize an SDK usage object or JSON dict."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        d = dict(usage)
    else:
        d = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "input_tokens",
            "output_tokens",
            "cost",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            val = getattr(usage, key, None)
            if val is not None:
                d[key] = val
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            if isinstance(details, dict):
                d["completion_tokens_details"] = details
            else:
                rt = getattr(details, "reasoning_tokens", None)
                if rt is not None:
                    d.setdefault("completion_tokens_details", {})["reasoning_tokens"] = rt
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            if isinstance(prompt_details, dict):
                d["prompt_tokens_details"] = prompt_details
            else:
                cached = getattr(prompt_details, "cached_tokens", None)
                if cached is not None:
                    d.setdefault("prompt_tokens_details", {})["cached_tokens"] = cached
        cost_details = getattr(usage, "cost_details", None)
        if cost_details is not None:
            d["cost_details"] = (
                cost_details
                if isinstance(cost_details, dict)
                else {
                    k: getattr(cost_details, k, None)
                    for k in (
                        "upstream_inference_cost",
                        "upstream_inference_prompt_cost",
                        "upstream_inference_completions_cost",
                    )
                }
            )
    if "input_tokens" in d and "prompt_tokens" not in d:
        d["prompt_tokens"] = d["input_tokens"]
    if "output_tokens" in d and "completion_tokens" not in d:
        d["completion_tokens"] = d["output_tokens"]
    if "total_tokens" not in d and d.get("prompt_tokens") is not None:
        d["total_tokens"] = int(d.get("prompt_tokens") or 0) + int(d.get("completion_tokens") or 0)
    return d


def reported_cost_usd_from_usage(usage: Any) -> float | None:
    """USD explicitly returned by the provider (OpenRouter ``usage.cost``)."""
    d = usage_to_dict(usage)
    cost = d.get("cost")
    if isinstance(cost, (int, float)):
        return float(cost)

    details = d.get("cost_details") or {}
    if isinstance(details, dict):
        upstream = details.get("upstream_inference_cost")
        if isinstance(upstream, (int, float)):
            return float(upstream)
        prompt_c = details.get("upstream_inference_prompt_cost")
        compl_c = details.get("upstream_inference_completions_cost")
        if isinstance(prompt_c, (int, float)) and isinstance(compl_c, (int, float)):
            return float(prompt_c) + float(compl_c)
    return None


def actual_cost_usd_from_usage(
    usage: Any,
    *,
    provider: str | None = None,
    model_id: str | None = None,
    batch: bool = False,
) -> float | None:
    """Best available USD for one request (reported, else computed from usage)."""
    fields = usage_cost_breakdown(
        usage,
        provider=provider,
        model_id=model_id,
        batch=batch,
    )
    return fields.get("cost")


def usage_cost_breakdown(
    usage: Any,
    *,
    provider: str | None = None,
    model_id: str | None = None,
    batch: bool = False,
) -> dict[str, Any]:
    """Token counts + per-request USD and provenance for one API call."""
    d = usage_to_dict(usage)
    details = d.get("completion_tokens_details") or {}
    prompt_details = d.get("prompt_tokens_details") or {}
    reasoning_tokens = int(
        details.get("reasoning_tokens")
        or d.get("reasoning_tokens")
        or 0
    )
    prompt_tokens = int(d.get("prompt_tokens") or 0)
    completion_tokens = int(d.get("completion_tokens") or 0)
    cache_create = int(d.get("cache_creation_input_tokens") or 0)
    cache_read = int(d.get("cache_read_input_tokens") or 0)
    oai_cached = int(prompt_details.get("cached_tokens") or 0)

    reported = reported_cost_usd_from_usage(d)
    cost_source = None
    pricing_source = None
    cost = reported

    if reported is not None:
        cost_source = "provider_reported"
    else:
        prov = infer_provider(model_id, provider)
        computed, pricing_source = compute_provider_cost_usd(
            prov,
            model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            openai_cached_tokens=oai_cached,
            batch=batch,
        )
        if computed is not None:
            cost = computed
            cost_source = "computed_from_usage"

    cost_details = d.get("cost_details") if isinstance(d.get("cost_details"), dict) else {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": int(d.get("total_tokens") or (prompt_tokens + completion_tokens)),
        "cost": cost,
        "cost_usd": cost,
        "cost_source": cost_source,
        "pricing_source": pricing_source,
        "cache_creation_input_tokens": cache_create,
        "cache_read_input_tokens": cache_read,
        "openai_cached_tokens": oai_cached,
        "upstream_inference_cost": (
            float(cost_details["upstream_inference_cost"])
            if isinstance(cost_details.get("upstream_inference_cost"), (int, float))
            else None
        ),
    }


def calculate_cost_from_usage(
    usage_obj: Any,
    *,
    provider: str | None = None,
    model_id: str | None = None,
    batch: bool = False,
) -> dict[str, Any]:
    """Legacy dict shape used by ``utils.log_cost_to_file`` and ``run_experiments``."""
    fields = usage_cost_breakdown(
        usage_obj,
        provider=provider,
        model_id=model_id,
        batch=batch,
    )
    cost_details = usage_to_dict(usage_obj).get("cost_details") or {}
    if not isinstance(cost_details, dict):
        cost_details = {}
    total = fields["cost"]
    return {
        "total_cost": float(total) if total is not None else 0.0,
        "prompt_cost": float(cost_details.get("upstream_inference_prompt_cost") or 0.0),
        "completion_cost": float(cost_details.get("upstream_inference_completions_cost") or 0.0),
        "total_tokens": fields["total_tokens"],
        "prompt_tokens": fields["prompt_tokens"],
        "completion_tokens": fields["completion_tokens"],
        "cost_reported": fields.get("cost_source") == "provider_reported",
        "cost_source": fields.get("cost_source"),
        "pricing_source": fields.get("pricing_source"),
    }


# ---------------------------------------------------------------------------
# Cost roll-ups and artifact writers (phase6b experiments)
# ---------------------------------------------------------------------------

PHASE6B_COST_LOG_NAME = "phase6b_cost_log.json"
PER_REQUEST_COST_LOG_NAME = "cost_log.json"
# Legacy standalone file — summary now lives under phase6b_cost_log.json["summary"].
LEGACY_COST_SUMMARY_NAME = "cost_summary.json"


def get_model_pricing(model_key: str) -> dict[str, float] | None:
    try:
        from llm_coherence.runtime.preflight_check import MODEL_COST_ESTIMATES

        return MODEL_COST_ESTIMATES.get(model_key)
    except Exception:
        return None


def estimate_cost_from_totals(
    rates: dict[str, float] | None,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    openai_cached_tokens: int = 0,
) -> float | None:
    if not rates:
        return None
    return round(
        compute_cost_from_tokens(
            rates,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            openai_cached_tokens=openai_cached_tokens,
        ),
        6,
    )


def summarize_per_request_entries(cost_entries: list[dict]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost": 0.0,
        "n_requests": len(cost_entries),
        "n_priced_requests": 0,
    }
    for entry in cost_entries:
        totals["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        totals["reasoning_tokens"] += int(entry.get("reasoning_tokens") or 0)
        cost = entry.get("cost")
        if cost is not None and isinstance(cost, (int, float)):
            totals["cost"] += float(cost)
            totals["n_priced_requests"] += 1
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    if totals["n_priced_requests"] == 0:
        totals["cost"] = None
    return totals


def cost_counts_from_entries(entries: list[dict]) -> dict[str, Any]:
    actual_total = sum(
        float(e["cost"])
        for e in entries
        if e.get("cost") is not None and isinstance(e.get("cost"), (int, float))
    )
    actual_n = sum(
        1
        for e in entries
        if e.get("cost") is not None and isinstance(e.get("cost"), (int, float))
    )
    provider_reported_n = sum(
        1
        for e in entries
        if e.get("cost_source") == "provider_reported" and e.get("cost") is not None
    )
    computed_n = sum(
        1
        for e in entries
        if e.get("cost_source") == "computed_from_usage" and e.get("cost") is not None
    )
    pricing_sources = sorted(
        {str(e["pricing_source"]) for e in entries if e.get("pricing_source")}
    )
    return {
        "actual_total": actual_total,
        "actual_n": actual_n,
        "provider_reported_n": provider_reported_n,
        "computed_n": computed_n,
        "pricing_sources": pricing_sources,
    }


def resolve_pricing_source(
    *,
    provider_reported_n: int,
    computed_n: int,
    pricing_sources: list[str],
) -> str | dict[str, Any] | list[str]:
    if provider_reported_n > 0 and computed_n == 0:
        return "provider_api_usage.cost"
    if computed_n > 0 and provider_reported_n == 0:
        return pricing_sources[0] if len(pricing_sources) == 1 else pricing_sources
    if provider_reported_n > 0 and computed_n > 0:
        return {
            "provider_reported": "provider_api_usage.cost",
            "computed_from_usage": pricing_sources,
        }
    return "compute_utilities.preflight_check.MODEL_COST_ESTIMATES"


def resolve_cost_source_label(provider_reported_n: int, computed_n: int) -> str | None:
    if provider_reported_n and not computed_n:
        return "provider_reported"
    if computed_n and not provider_reported_n:
        return "computed_from_usage"
    if provider_reported_n and computed_n:
        return "mixed"
    return None


def build_cost_summary_notes(
    *,
    actual_n: int,
    calls_logged_total: int,
    provider_reported_n: int,
    computed_n: int,
) -> tuple[str | None, str]:
    """Return ``(summary_estimated, notes)`` for cost_summary.json."""
    if actual_n > 0:
        parts = []
        if provider_reported_n:
            parts.append(f"{provider_reported_n} provider-reported (usage.cost)")
        if computed_n:
            parts.append(f"{computed_n} computed from token usage + published rates")
        notes = (
            f"actual_cost_usd is the sum of per-request costs over "
            f"{actual_n}/{calls_logged_total} requests ({', '.join(parts)})."
        )
        return None, notes
    notes = (
        "estimated_cost_usd uses MODEL_COST_ESTIMATES ($/1M) and observed token totals. "
        "actual_cost_usd is null when per-request cost could not be resolved."
    )
    return None, notes


def build_cost_summary_notes_from_metadata(*, has_actual: bool) -> str:
    if has_actual:
        return (
            "actual_cost_usd is the sum of per-set metadata actual_cost_usd from live runs."
        )
    return (
        "Modeled after property_ladder_pruning cost artifacts. "
        "estimated_cost_usd uses MODEL_COST_ESTIMATES and observed token totals."
    )


def summarize_usage_log(entries: list[dict]) -> dict[str, Any]:
    """Aggregate per-call usage dicts into min/median/p95/max/total summaries."""

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

    provider_reported = sum(
        1
        for e in entries
        if e.get("cost_source") == "provider_reported" and e.get("cost_usd") is not None
    )
    computed = sum(
        1
        for e in entries
        if e.get("cost_source") == "computed_from_usage" and e.get("cost_usd") is not None
    )
    return {
        "calls_logged": len(entries),
        "prompt_tokens": stats(e.get("prompt_tokens") for e in entries),
        "completion_tokens": stats(e.get("completion_tokens") for e in entries),
        "reasoning_tokens": stats(e.get("reasoning_tokens") for e in entries),
        "cache_creation_input_tokens": stats(
            e.get("cache_creation_input_tokens") for e in entries
        ),
        "cache_read_input_tokens": stats(e.get("cache_read_input_tokens") for e in entries),
        "openai_cached_tokens": stats(e.get("openai_cached_tokens") for e in entries),
        "cost_usd": stats(e.get("cost_usd") for e in entries),
        "provider_reported_cost_count": provider_reported,
        "computed_from_usage_cost_count": computed,
    }


def actual_cost_from_usage_summary(usage_stats: dict | None) -> float | None:
    cost_stats = usage_stats.get("cost_usd") if usage_stats else None
    if cost_stats and cost_stats.get("n"):
        total = cost_stats.get("total")
        if total is not None:
            return round(float(total), 6)
    return None


def usage_stats_field_total(usage_stats: dict, key: str) -> int:
    block = usage_stats.get(key) or {}
    val = block.get("total")
    return int(val) if isinstance(val, (int, float)) else 0


def records_from_per_request_entries(
    cost_entries: list[dict],
    *,
    group_key: Callable[[dict], str],
    result_path: str,
    pricing: dict[str, float] | None,
) -> tuple[list[dict], dict[str, int], float, int]:
    """Group per-request entries into phase6b cost-log records."""
    by_group: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls_logged": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "actual_cost_usd": 0.0,
            "actual_cost_count": 0,
        }
    )
    for entry in cost_entries:
        key = group_key(entry)
        rec = by_group[key]
        rec["calls_logged"] += 1
        rec["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        rec["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        rec["reasoning_tokens"] += int(entry.get("reasoning_tokens") or 0)
        cost = entry.get("cost")
        if cost is not None and isinstance(cost, (int, float)):
            rec["actual_cost_usd"] += float(cost)
            rec["actual_cost_count"] += 1

    records: list[dict] = []
    token_totals = {
        "prompt": 0,
        "completion": 0,
        "reasoning": 0,
        "calls": 0,
    }
    estimated_from_records_total = 0.0
    estimated_from_records_n = 0

    for group_id, agg in sorted(by_group.items()):
        prompt = agg["prompt_tokens"]
        completion = agg["completion_tokens"]
        reasoning = agg["reasoning_tokens"]
        calls = agg["calls_logged"]
        token_totals["prompt"] += prompt
        token_totals["completion"] += completion
        token_totals["reasoning"] += reasoning
        token_totals["calls"] += calls

        est = estimate_cost_from_totals(
            pricing, prompt_tokens=prompt, completion_tokens=completion
        )
        if est is not None:
            estimated_from_records_total += est
            estimated_from_records_n += 1

        actual = round(agg["actual_cost_usd"], 6) if agg["actual_cost_count"] > 0 else None
        records.append(
            {
                "test_name": group_id,
                "result_path": result_path,
                "calls_logged": calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "openai_cached_tokens": 0,
                "estimated_cost_usd": est,
                "actual_cost_usd": actual,
            }
        )

    return records, token_totals, estimated_from_records_total, estimated_from_records_n


def build_phase6b_cost_log(
    model_key: str,
    records: list[dict],
    *,
    pricing: dict[str, float] | None,
    prompt_total: int,
    completion_total: int,
    reasoning_total: int,
    calls_logged_total: int,
    estimated_from_usage: float | None,
    estimated_from_records_total: float,
    estimated_from_records_n: int,
    actual_total: float,
    actual_n: int,
    provider_reported_n: int = 0,
    computed_n: int = 0,
    pricing_sources: list[str] | None = None,
    cache_creation_input_tokens_total: int = 0,
    cache_read_input_tokens_total: int = 0,
    openai_cached_tokens_total: int = 0,
    pricing_source_override: str | dict[str, Any] | list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = created_at or datetime.now(timezone.utc).isoformat()
    pricing_sources = pricing_sources or []
    pricing_source = pricing_source_override or resolve_pricing_source(
        provider_reported_n=provider_reported_n,
        computed_n=computed_n,
        pricing_sources=pricing_sources,
    )
    return {
        "model_key": model_key,
        "created_at": now,
        "updated_at": now,
        "pricing_source": pricing_source,
        "pricing_per_1m": pricing if actual_n == 0 else None,
        "result_files_count": len(records),
        "calls_logged": calls_logged_total,
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": completion_total,
        "reasoning_tokens_total": reasoning_total,
        "cache_creation_input_tokens_total": cache_creation_input_tokens_total,
        "cache_read_input_tokens_total": cache_read_input_tokens_total,
        "openai_cached_tokens_total": openai_cached_tokens_total,
        "estimated_cost_usd_from_usage": estimated_from_usage,
        "estimated_cost_usd_from_results_sum": round(estimated_from_records_total, 6),
        "estimated_cost_count_from_results": estimated_from_records_n,
        "actual_cost_usd_sum": round(actual_total, 6) if actual_n > 0 else None,
        "actual_cost_count": actual_n,
        "provider_reported_cost_count": provider_reported_n,
        "computed_from_usage_cost_count": computed_n,
        "pricing_sources": pricing_sources,
        "records": records,
    }


def build_phase6b_cost_summary(
    model_key: str,
    *,
    calls_logged_total: int,
    n_records: int,
    estimated_from_usage: float | None,
    actual_total: float,
    actual_n: int,
    prompt_total: int,
    completion_total: int,
    reasoning_total: int,
    provider_reported_n: int = 0,
    computed_n: int = 0,
    cache_creation_input_tokens_total: int = 0,
    cache_read_input_tokens_total: int = 0,
    openai_cached_tokens_total: int = 0,
    notes: str | None = None,
    summary_estimated: float | None = None,
) -> dict[str, Any]:
    if notes is None:
        _, notes = build_cost_summary_notes(
            actual_n=actual_n,
            calls_logged_total=calls_logged_total,
            provider_reported_n=provider_reported_n,
            computed_n=computed_n,
        )
    if actual_n == 0 and summary_estimated is None:
        summary_estimated = estimated_from_usage

    return {
        "model": model_key,
        "n_recorded": calls_logged_total,
        "n_priced_files": n_records,
        "estimated_cost_usd": summary_estimated if actual_n == 0 else None,
        "actual_cost_usd": round(actual_total, 6) if actual_n > 0 else None,
        "actual_cost_count": actual_n,
        "provider_reported_cost_count": provider_reported_n,
        "computed_from_usage_cost_count": computed_n,
        "cost_source": resolve_cost_source_label(provider_reported_n, computed_n),
        "tokens": {
            "prompt_tokens_total": prompt_total,
            "completion_tokens_total": completion_total,
            "reasoning_tokens_total": reasoning_total,
            "cache_creation_input_tokens_total": cache_creation_input_tokens_total,
            "cache_read_input_tokens_total": cache_read_input_tokens_total,
            "openai_cached_tokens_total": openai_cached_tokens_total,
        },
        "notes": notes,
    }


def write_phase6b_cost_artifacts(
    out_dir: Path,
    cost_log: dict[str, Any],
    summary: dict[str, Any],
    *,
    cost_log_name: str = PHASE6B_COST_LOG_NAME,
) -> Path:
    """Write one aggregated cost log with embedded summary; remove legacy duplicate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**cost_log, "summary": summary}
    cost_log_path = out_dir / cost_log_name
    cost_log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    legacy_summary = out_dir / LEGACY_COST_SUMMARY_NAME
    if legacy_summary.is_file():
        legacy_summary.unlink()
    return cost_log_path


def summary_from_phase6b_cost_log(cost_log: dict[str, Any]) -> dict[str, Any]:
    """Return embedded summary, or build a minimal view from legacy flat logs."""
    if isinstance(cost_log.get("summary"), dict):
        return cost_log["summary"]
    return {
        "model": cost_log.get("model_key"),
        "n_recorded": cost_log.get("calls_logged"),
        "estimated_cost_usd": cost_log.get("estimated_cost_usd_from_usage"),
        "actual_cost_usd": cost_log.get("actual_cost_usd_sum"),
        "actual_cost_count": cost_log.get("actual_cost_count"),
        "tokens": {
            "prompt_tokens_total": cost_log.get("prompt_tokens_total"),
            "completion_tokens_total": cost_log.get("completion_tokens_total"),
            "reasoning_tokens_total": cost_log.get("reasoning_tokens_total"),
            "cache_creation_input_tokens_total": cost_log.get(
                "cache_creation_input_tokens_total"
            ),
            "cache_read_input_tokens_total": cost_log.get("cache_read_input_tokens_total"),
            "openai_cached_tokens_total": cost_log.get("openai_cached_tokens_total"),
        },
    }


def write_per_request_cost_log_file(
    path: Path,
    model_key: str,
    cost_entries: list[dict],
) -> dict[str, Any]:
    totals = summarize_per_request_entries(cost_entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model": model_key, "totals": totals, "per_request": cost_entries}, indent=2),
        encoding="utf-8",
    )
    return totals


def enrich_per_request_entries(
    cost_entries: list[dict],
    *,
    provider: str | None,
    model_id: str | None,
    batch: bool = False,
) -> tuple[list[dict], bool]:
    enriched: list[dict] = []
    changed = False
    for entry in cost_entries:
        if entry.get("cost") is not None:
            enriched.append(entry)
            continue
        resolved_model = entry.get("model") or model_id
        if not resolved_model:
            enriched.append(entry)
            continue
        fields = usage_cost_breakdown(
            {
                "prompt_tokens": entry.get("prompt_tokens"),
                "completion_tokens": entry.get("completion_tokens"),
                "reasoning_tokens": entry.get("reasoning_tokens"),
            },
            provider=provider or infer_provider(resolved_model),
            model_id=resolved_model,
            batch=batch,
        )
        if fields.get("cost") is None:
            enriched.append(entry)
            continue
        enriched.append(
            {
                **entry,
                "cost": fields["cost"],
                "cost_usd": fields["cost"],
                "cost_source": fields.get("cost_source"),
                "pricing_source": fields.get("pricing_source"),
            }
        )
        changed = True
    return enriched, changed


def format_cost_artifact_note(summary: dict[str, Any], totals: dict[str, Any]) -> str:
    actual = summary.get("actual_cost_usd")
    est = summary.get("estimated_cost_usd")
    if actual is not None:
        return f", actual ${actual:.6f} ({summary.get('actual_cost_count', 0)} reqs)"
    if est is not None:
        return f", est ${est:.6f}"
    return ""
