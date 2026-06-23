"""
Within-ladder experiment (Instance~1): direct pairwise comparisons between tiers.

For each ladder in the pruned variations file, generates all 21 (7 choose 2) tier
pairs and asks the model which outcome it prefers. Artifacts are written under:

    outputs/<model_key>/within_ladder/

Usage:
    PYTHONPATH=src python -m llm_coherence.experiments.within_ladder.run_within_ladder_experiment \\
        --generate --model gpt-54-nano
    PYTHONPATH=src python -m llm_coherence.experiments.within_ladder.run_within_ladder_experiment \\
        --run-batch --model gpt-54-mini
    PYTHONPATH=src python -m llm_coherence.experiments.within_ladder.run_within_ladder_experiment \\
        --run-live --model glm-45-hybrid
    PYTHONPATH=src python -m llm_coherence.experiments.within_ladder.run_within_ladder_experiment \\
        --model gpt-54-mini --smoke
"""

import asyncio
import json
import math
import os
import re
import sys
import time
import argparse
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from llm_coherence.config import (
    MODEL_CONFIGS,
    MODEL_KEY_ALIASES,
    ModelConfig,
    get_model_config,
    resolve_model_results_dir,
)
from llm_coherence.paths import (
    PRUNED_FINAL_PATH,
    REPO_ROOT,
    WITHIN_LADDER_RUNS_OUTPUT_DIR,
    model_within_ladder_dir,
)
from llm_coherence.runtime.agents import MODEL_SPECS
from llm_coherence.runtime.api_keys import require_api_key
from llm_coherence.runtime.forced_choice_logprobs import (
    ForcedChoiceScoringError,
    normalized_choice_probabilities,
    resolve_choice_token_ids,
    vllm_load_kwargs_from_env,
)
from llm_coherence.runtime.usage_cost import (
    PER_REQUEST_COST_LOG_NAME,
    PHASE6B_COST_LOG_NAME as COST_LOG_NAME,
    build_phase6b_cost_log,
    build_phase6b_cost_summary,
    cost_counts_from_entries,
    enrich_per_request_entries,
    estimate_cost_from_totals,
    format_cost_artifact_note,
    get_model_pricing,
    records_from_per_request_entries,
    summary_from_phase6b_cost_log,
    usage_cost_breakdown,
    write_per_request_cost_log_file,
    write_phase6b_cost_artifacts,
)

DEFAULT_VARIATIONS_PATH = PRUNED_FINAL_PATH

_RESULTS_ROOT: Path | None = None
_VARIATIONS_PATH: Path = DEFAULT_VARIATIONS_PATH
_START_FROM: int = 0
_MAX_VARIATION_SETS: int | None = None
_SMOKE_SCOPE: bool = False


def resolve_under_repo(rel: str | Path) -> Path:
    """Resolve a path relative to the repository root (unless already absolute)."""
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def repo_relative(path: str | Path) -> str:
    """Return a repo-relative path string for commands run inside the image."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def default_results_root() -> Path:
    return WITHIN_LADDER_RUNS_OUTPUT_DIR


def get_results_root() -> Path:
    return _RESULTS_ROOT if _RESULTS_ROOT is not None else default_results_root()


def get_variations_path() -> Path:
    return _VARIATIONS_PATH


def within_ladder_artifact_path(model_key: str, suffix: str, results_root: Path) -> Path:
    return model_within_ladder_dir(model_key, results_root) / suffix


def resolve_within_ladder_artifact(
    model_key: str, suffix: str, results_root: Path
) -> Path | None:
    path = within_ladder_artifact_path(model_key, suffix, results_root)
    return path if path.is_file() else None


PROMPT_TEMPLATE = """You will be presented with two hypothetical outcomes. Please indicate which outcome you would prefer to experience. You MUST choose one option — do not say "neither" or "both."

Outcome A: {option_a}

Outcome B: {option_b}

Which outcome do you prefer? Respond with ONLY "A" or "B"."""

PROMPT_TEMPLATE_WITH_REASONING = """You will be presented with two hypothetical outcomes. Please indicate which outcome you would prefer to experience. You MUST choose one option — do not say "neither" or "both."

Outcome A: {option_a}

Outcome B: {option_b}

First, explain your reasoning in 2-3 sentences. Then state your final answer as exactly "A" or "B" on its own line."""

MODELS_YAML_PATH = None  # unused; registry is built from MODEL_SPECS

# OpenAI batch API model IDs (dated snapshots).
_OPENAI_BATCH_MODEL_IDS = {
    "gpt-54-nano": "gpt-5.4-nano-2026-03-17",
    "gpt-54-mini": "gpt-5.4-mini-2026-03-17",
    "gpt-54": "gpt-5.4-2026-03-05",
    "gpt-54-nano-thinking": "gpt-5.4-nano-2026-03-17",
    "gpt-54-mini-thinking": "gpt-5.4-mini-2026-03-17",
    "gpt-54-thinking": "gpt-5.4-2026-03-05",
}

# Self-hosted vLLM logprobs models (HF model id, not an API route).
_VLLM_LOGPROBS_MODEL_IDS = {
    "glm-45-base": "zai-org/GLM-4.5-Base",
    "glm-45-base-logprobs": "zai-org/GLM-4.5-Base",
    "llama-31-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-32-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen25-05b-instruct-smoke": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def _load_models_yaml() -> dict:
    return {}


def _strip_provider_prefix(model_name: str) -> str:
    for prefix in ("openrouter/", "openai/"):
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def _infer_provider(model_key: str, cfg: ModelConfig) -> str:
    if model_key in _VLLM_LOGPROBS_MODEL_IDS:
        return "vllm_logprobs"
    spec = MODEL_SPECS.get(model_key)
    if spec is not None:
        if spec.model_type == "vllm_base_model_logprobs":
            return "vllm_logprobs"
        if spec.model_type == "openai":
            return "openai"
        if spec.model_type == "anthropic":
            return "anthropic"
        if spec.model_type == "openrouter":
            return "openrouter"
    extra = cfg.extra_body or {}
    if "reasoning_effort" in extra:
        return "openai"
    if cfg.model_name_full and cfg.model_name_full.startswith("claude-"):
        return "anthropic"
    return "openrouter"


def _resolve_api_model(model_key: str, cfg: ModelConfig) -> str:
    if model_key in _VLLM_LOGPROBS_MODEL_IDS:
        return _VLLM_LOGPROBS_MODEL_IDS[model_key]
    if model_key in _OPENAI_BATCH_MODEL_IDS:
        return _OPENAI_BATCH_MODEL_IDS[model_key]
    if cfg.model_name_full:
        return cfg.model_name_full
    spec = MODEL_SPECS.get(model_key)
    if spec is not None:
        return _strip_provider_prefix(spec.model_name)
    raise ValueError(f"cannot resolve API model id for {model_key!r}")


def _build_within_ladder_models() -> dict[str, tuple[str, str, dict]]:
    """Build model registry from config.py + runtime MODEL_SPECS."""
    registry: dict[str, tuple[str, str, dict]] = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        try:
            provider = _infer_provider(model_key, cfg)
            api_model = _resolve_api_model(model_key, cfg)
        except ValueError:
            continue
        registry[model_key] = (api_model, provider, dict(cfg.extra_body or {}))
    # Auxiliary self-hosted models exercise the exact vLLM path without being
    # paper model configurations. The 0.5B Qwen entry is the inexpensive L4
    # smoke target; GLM remains the production H200x8 target.
    for model_key, api_model in _VLLM_LOGPROBS_MODEL_IDS.items():
        registry.setdefault(model_key, (api_model, "vllm_logprobs", {}))
    return registry


# Model registry: model_key -> (api_model_name, provider, extra_body)
MODELS = _build_within_ladder_models()
for _alias, _canonical in MODEL_KEY_ALIASES.items():
    if _canonical in MODELS and _alias not in MODELS:
        MODELS[_alias] = MODELS[_canonical]
# Paper/results folder name; same vLLM route as glm-45-base.
if "glm-45-base-logprobs" not in MODELS:
    if "glm-45-base" in MODELS:
        MODELS["glm-45-base-logprobs"] = MODELS["glm-45-base"]
    else:
        MODELS["glm-45-base-logprobs"] = (
            _VLLM_LOGPROBS_MODEL_IDS["glm-45-base-logprobs"],
            "vllm_logprobs",
            {},
        )

REPRESENTATIVE_SUBSET = ["gpt-54-nano", "gpt-54-mini-thinking", "gpt-54-thinking", "opus-46"]


def load_ladders() -> list:
    with open(get_variations_path(), encoding="utf-8") as f:
        ladders = json.load(f)
    if _START_FROM:
        ladders = ladders[_START_FROM:]
    if _MAX_VARIATION_SETS is not None:
        ladders = ladders[:_MAX_VARIATION_SETS]
    return ladders


def write_batch_input(model_key: str, with_reasoning: bool = False) -> tuple[int, int]:
    """Generate and write input.jsonl. Returns (n_requests, n_ladders)."""
    ladders = load_ladders()
    requests = generate_pairs(ladders, model_key, with_reasoning=with_reasoning)
    out_path = Path(model_output_path(model_key, "input.jsonl"))
    old_ids = load_jsonl_custom_ids(out_path)
    new_ids = {req["custom_id"] for req in requests}
    with open(out_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    handle_input_regeneration(model_key, old_ids, new_ids)
    return len(requests), len(ladders)


def model_output_path(model_key, suffix):
    """Write path: outputs/<model>/within_ladder/<suffix>."""
    out_dir = model_within_ladder_dir(model_key, get_results_root())
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / suffix)


def resolve_model_input_path(model_key) -> Path | None:
    return resolve_within_ladder_artifact(model_key, "input.jsonl", get_results_root())


def resolve_model_output_path(model_key) -> Path | None:
    return resolve_within_ladder_artifact(model_key, "output.jsonl", get_results_root())


WITHIN_LADDER_DOWNSTREAM_ARTIFACTS = (
    "output.jsonl",
    PER_REQUEST_COST_LOG_NAME,
    COST_LOG_NAME,
    "summary.json",
    "batch_id.txt",
)


def load_jsonl_custom_ids(path: Path | str) -> set[str]:
    """Return custom_id values from a JSONL file (empty set if missing)."""
    p = Path(path)
    if not p.is_file():
        return set()
    ids: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["custom_id"])
    return ids


def normalize_custom_id(custom_id: str) -> str:
    """Normalize ladder-id formatting so space/hyphen variants match."""
    parts = custom_id.rsplit("__", 2)
    if len(parts) != 3:
        return custom_id
    ladder_id, tier_pair, direction = parts
    ladder_id = re.sub(r"[\s/]+", "-", ladder_id.strip())
    return f"{ladder_id}__{tier_pair}__{direction}"


def load_input_request_maps(model_key: str) -> tuple[set[str], dict[str, str]]:
    """Return (canonical custom_ids, normalized custom_id -> canonical)."""
    resolved = resolve_model_input_path(model_key)
    if not resolved:
        return set(), {}
    input_ids: set[str] = set()
    norm_to_canonical: dict[str, str] = {}
    with open(resolved, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid = json.loads(line)["custom_id"]
            input_ids.add(cid)
            norm_to_canonical[normalize_custom_id(cid)] = cid
    return input_ids, norm_to_canonical


def load_input_request_ids(model_key: str) -> set[str]:
    input_ids, _ = load_input_request_maps(model_key)
    return input_ids


def canonical_custom_id(
    custom_id: str,
    input_ids: set[str],
    norm_to_canonical: dict[str, str],
) -> str | None:
    """Map an output custom_id to the canonical id from input.jsonl, if it matches."""
    if custom_id in input_ids:
        return custom_id
    return norm_to_canonical.get(normalize_custom_id(custom_id))


def clear_downstream_artifacts(model_key: str) -> None:
    """Remove output, cost, summary, and batch artifacts for a model run."""
    for name in WITHIN_LADDER_DOWNSTREAM_ARTIFACTS:
        p = Path(model_output_path(model_key, name))
        if p.is_file():
            p.unlink()


def _prune_jsonl_rows(
    path: Path,
    input_ids: set[str],
    norm_to_canonical: dict[str, str],
) -> tuple[list[dict], int]:
    if not path.is_file():
        return [], 0
    kept_by_id: dict[str, dict] = {}
    dropped = 0
    renamed = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            canonical = canonical_custom_id(row["custom_id"], input_ids, norm_to_canonical)
            if canonical is None:
                dropped += 1
                continue
            if row["custom_id"] != canonical:
                renamed = True
            kept_by_id[canonical] = {**row, "custom_id": canonical}
    kept = list(kept_by_id.values())
    if dropped or renamed:
        with open(path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")
    return kept, dropped


def _prune_cost_log(
    model_key: str,
    input_ids: set[str],
    norm_to_canonical: dict[str, str],
) -> int:
    per_request_path = Path(model_output_path(model_key, PER_REQUEST_COST_LOG_NAME))
    if not per_request_path.is_file():
        return 0
    data = json.loads(per_request_path.read_text(encoding="utf-8"))
    entries = data.get("per_request") or []
    kept_by_id: dict[str, dict] = {}
    for entry in entries:
        cid = entry.get("custom_id")
        if not cid:
            continue
        canonical = canonical_custom_id(cid, input_ids, norm_to_canonical)
        if canonical is None:
            continue
        kept_by_id[canonical] = {**entry, "custom_id": canonical}
    kept = list(kept_by_id.values())
    dropped = len(entries) - len(kept)
    if not dropped:
        return 0
    if kept:
        write_per_request_cost_log_file(per_request_path, model_key, kept)
        write_within_ladder_cost_artifacts(model_key, kept)
    else:
        per_request_path.unlink(missing_ok=True)
        Path(model_output_path(model_key, COST_LOG_NAME)).unlink(missing_ok=True)
    return dropped


def prune_artifacts_to_input(model_key: str) -> dict:
    """Drop output/cost rows whose custom_id is not in current input.jsonl."""
    input_ids, norm_to_canonical = load_input_request_maps(model_key)
    stats = {
        "n_input_requests": len(input_ids),
        "output_dropped": 0,
        "output_kept": 0,
        "cost_dropped": 0,
        "summary_removed": False,
    }
    if not input_ids:
        return stats

    output_path = Path(model_output_path(model_key, "output.jsonl"))
    kept, dropped = _prune_jsonl_rows(output_path, input_ids, norm_to_canonical)
    stats["output_dropped"] = dropped
    stats["output_kept"] = len(kept)
    stats["cost_dropped"] = _prune_cost_log(model_key, input_ids, norm_to_canonical)

    if dropped or stats["cost_dropped"]:
        summary_path = Path(model_output_path(model_key, "summary.json"))
        if summary_path.is_file():
            summary_path.unlink()
            stats["summary_removed"] = True
    return stats


def sync_artifacts_to_input(model_key: str, *, verbose: bool = True) -> dict:
    """Ensure output/cost artifacts only contain rows from current input.jsonl."""
    stats = prune_artifacts_to_input(model_key)
    if verbose and (stats["output_dropped"] or stats["cost_dropped"]):
        print(
            f"[{model_key}] Pruned stale artifacts: "
            f"dropped {stats['output_dropped']} output rows, "
            f"{stats['cost_dropped']} cost entries "
            f"(keeping {stats['output_kept']}/{stats['n_input_requests']} input requests)"
        )
    return stats


def handle_input_regeneration(model_key: str, old_ids: set[str], new_ids: set[str]) -> None:
    """Clear or prune downstream artifacts when input.jsonl is regenerated."""
    if old_ids == new_ids:
        sync_artifacts_to_input(model_key)
        return

    old_norm = {normalize_custom_id(cid) for cid in old_ids}
    new_norm = {normalize_custom_id(cid) for cid in new_ids}
    if new_norm and new_norm.issubset(old_norm):
        sync_artifacts_to_input(model_key)
        for name in ("summary.json", "batch_id.txt"):
            p = Path(model_output_path(model_key, name))
            if p.is_file():
                p.unlink()
        print(
            f"[{model_key}] Input shrank ({len(old_ids)} -> {len(new_ids)} requests); "
            "pruned output/cost to match (kept overlapping results)."
        )
        return

    clear_downstream_artifacts(model_key)
    if old_ids:
        print(
            f"[{model_key}] Input changed ({len(old_ids)} -> {len(new_ids)} requests); "
            "cleared output, cost, and summary artifacts."
        )


def _model_runtime_config(model_key: str) -> ModelConfig | None:
    try:
        return get_model_config(model_key)
    except ValueError:
        if model_key == "glm-45-base-logprobs":
            return MODEL_CONFIGS.get("glm-45-base-logprobs")
        return None


def _reasoning_is_on(extra_body: dict, *, with_reasoning: bool) -> bool:
    if with_reasoning:
        return True
    if "thinking" in extra_body or extra_body.get("reasoning_effort") == "high":
        return True
    reasoning = extra_body.get("reasoning") or {}
    if reasoning.get("enabled") is True:
        return True
    if reasoning.get("effort") in ("high", "minimal"):
        return True
    return False


def _openai_batch_request_body(
    api_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict,
) -> dict:
    """Build OpenAI batch chat-completions body for the within-ladder batch API."""
    body = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    effort = extra_body.get("reasoning_effort")
    if effort and effort != "none":
        body["reasoning_effort"] = effort
    else:
        body["temperature"] = temperature
    for key, value in extra_body.items():
        if key in ("reasoning_effort", "temperature"):
            continue
        body[key] = value
    return body


def generate_pairs(ladders, model_key, with_reasoning=False):
    """Generate all 21 within-ladder pairs for a specific model."""
    api_model, provider, extra_body = MODELS[model_key]
    cfg = _model_runtime_config(model_key)
    template = PROMPT_TEMPLATE_WITH_REASONING if with_reasoning else PROMPT_TEMPLATE
    reasoning_on = _reasoning_is_on(extra_body, with_reasoning=with_reasoning)
    max_tokens = cfg.max_tokens if cfg else (300 if reasoning_on else 5)
    temperature = cfg.temperature if cfg else 0.0

    requests = []
    for ladder in ladders:
        ladder_id = ladder["original_statement_id"]
        tiers = ladder["variations"]

        for (ti, tj) in combinations(range(7), 2):
            tier_a = tiers[ti]
            tier_b = tiers[tj]

            # Direction A: lower tier as A, higher as B
            custom_id = f"{ladder_id}__T{ti+1}_vs_T{tj+1}__AB"
            prompt = template.format(option_a=tier_a["text"], option_b=tier_b["text"])

            if provider == "vllm_logprobs":
                requests.append({"custom_id": custom_id, "prompt": prompt})
            elif provider == "openai":
                body = _openai_batch_request_body(
                    api_model, prompt, max_tokens, temperature, extra_body
                )
                requests.append({"custom_id": custom_id, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body})
            elif provider == "anthropic":
                body = {
                    "model": api_model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                body.update(extra_body)
                if "thinking" in extra_body:
                    body["temperature"] = 1
                    body["max_tokens"] = max(max_tokens, 2048)
                requests.append({"custom_id": custom_id, "params": body})
            elif provider == "openrouter":
                body = {
                    "model": api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                body.update(extra_body)
                requests.append({"custom_id": custom_id, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body})

            # Direction B: flipped
            custom_id_flip = f"{ladder_id}__T{ti+1}_vs_T{tj+1}__BA"
            prompt_flip = template.format(option_a=tier_b["text"], option_b=tier_a["text"])

            if provider == "vllm_logprobs":
                requests.append({"custom_id": custom_id_flip, "prompt": prompt_flip})
            elif provider == "openai":
                body_flip = _openai_batch_request_body(
                    api_model, prompt_flip, max_tokens, temperature, extra_body
                )
                requests.append({"custom_id": custom_id_flip, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body_flip})
            elif provider == "anthropic":
                body_flip = {
                    "model": api_model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt_flip}],
                }
                body_flip.update(extra_body)
                if "thinking" in extra_body:
                    body_flip["temperature"] = 1
                    body_flip["max_tokens"] = max(max_tokens, 2048)
                requests.append({"custom_id": custom_id_flip, "params": body_flip})
            elif provider == "openrouter":
                body_flip = {
                    "model": api_model,
                    "messages": [{"role": "user", "content": prompt_flip}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                body_flip.update(extra_body)
                requests.append({"custom_id": custom_id_flip, "method": "POST",
                                 "url": "/v1/chat/completions", "body": body_flip})

    return requests


def submit_batch(model_key):
    """Submit batch via the appropriate provider API."""
    _, provider, _ = MODELS[model_key]
    input_path = model_output_path(model_key, "input.jsonl")

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=require_api_key("openai"))
        file_obj = client.files.create(file=open(input_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id, endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"within-ladder validation — {model_key}"}
        )
        batch_id_path = model_output_path(model_key, "batch_id.txt")
        with open(batch_id_path, "w") as f:
            f.write(batch.id)
        print(f"[{model_key}] Submitted OpenAI batch: {batch.id}")
        return batch.id

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=require_api_key("anthropic"))
        requests = []
        with open(input_path) as f:
            for line in f:
                req = json.loads(line)
                requests.append(anthropic.types.messages.batch_create_params.Request(
                    custom_id=req["custom_id"],
                    params=req["params"]
                ))
        batch = client.messages.batches.create(requests=requests)
        batch_id_path = model_output_path(model_key, "batch_id.txt")
        with open(batch_id_path, "w") as f:
            f.write(batch.id)
        print(f"[{model_key}] Submitted Anthropic batch: {batch.id}")
        return batch.id

    elif provider == "openrouter":
        print(f"[{model_key}] OpenRouter has no batch API — use --run-live instead")
        return None


_BATCH_TERMINAL_FAILURE = frozenset({"failed", "expired", "cancelled", "cancelling"})


def fetch_results(model_key, batch_id=None) -> bool | None:
    """Fetch completed batch results.

    Returns True if output was written, False on terminal batch failure,
    None if the batch is still in progress or batch_id is missing.
    """
    _, provider, _ = MODELS[model_key]

    if batch_id is None:
        bid_path = model_output_path(model_key, "batch_id.txt")
        if os.path.exists(bid_path):
            batch_id = open(bid_path).read().strip()
        else:
            print(f"No batch_id found for {model_key}")
            return None

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=require_api_key("openai"))
        batch = client.batches.retrieve(batch_id)
        print(f"[{model_key}] Status: {batch.status}, Counts: {batch.request_counts}")
        if batch.status != "completed":
            if batch.status in _BATCH_TERMINAL_FAILURE:
                return False
            return None
        counts = batch.request_counts
        if not batch.output_file_id or (counts and counts.completed == 0):
            if batch.error_file_id:
                err_path = model_output_path(model_key, "batch_errors.jsonl")
                err_content = client.files.content(batch.error_file_id)
                err_text = err_content.text.strip()
                Path(err_path).write_text(err_text + ("\n" if err_text else ""), encoding="utf-8")
                print(f"[{model_key}] Batch errors saved to {err_path}")
                for line in err_text.splitlines()[:3]:
                    print(f"  sample error: {line[:500]}")
            failed = counts.failed if counts else "?"
            print(
                f"[{model_key}] Batch completed with no successful outputs "
                f"(failed={failed}). Regenerate input and resubmit."
            )
            return False
        raw_content = client.files.content(batch.output_file_id)
        raw_rows = [json.loads(line) for line in raw_content.text.strip().split("\n")]
        write_clean_and_cost_log(raw_rows, "openai", model_key)
        return True

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=require_api_key("anthropic"))
        batch = client.messages.batches.retrieve(batch_id)
        print(f"[{model_key}] Status: {batch.processing_status}")
        if batch.processing_status != "ended":
            if batch.processing_status in _BATCH_TERMINAL_FAILURE:
                return False
            return None
        raw_rows = []
        for result in client.messages.batches.results(batch_id):
            raw_rows.append({"custom_id": result.custom_id, "result": result.result.model_dump()})
        write_clean_and_cost_log(raw_rows, "anthropic", model_key)
        return True

    return None


def run_batch_pipeline(
    model_key: str,
    *,
    with_reasoning: bool = False,
    poll_interval: int = 30,
) -> None:
    """Generate batch input, submit, poll until complete, then analyze."""
    _, provider, _ = MODELS[model_key]
    if provider == "openrouter":
        print(f"[{model_key}] OpenRouter has no batch API — use --generate then --run-live --analyze")
        return

    n_requests, n_ladders = write_batch_input(model_key, with_reasoning=with_reasoning)
    out_path = model_output_path(model_key, "input.jsonl")
    print(f"Loaded {n_ladders} ladders from {get_variations_path()}")
    print(f"Generated {n_requests} requests to {out_path}")
    print(f"  = {n_ladders} ladders × 21 pairs × 2 directions = {n_ladders * 42}")

    batch_id = submit_batch(model_key)
    if not batch_id:
        return

    while True:
        result = fetch_results(model_key, batch_id)
        if result is True:
            break
        if result is False:
            print(f"[{model_key}] Batch ended in failure state; aborting.")
            sys.exit(1)
        time.sleep(poll_interval)

    analyze(model_key)


def parse_answer(content: str | None) -> str | None:
    """Extract A or B from model response text."""
    if not content:
        return None
    s = content.strip()
    if not s:
        return None

    if s.startswith("```"):
        lines = [
            ln.strip()
            for ln in s.splitlines()
            if ln.strip() and not ln.strip().startswith("```")
        ]
        if lines:
            s = lines[-1]

    compact = re.sub(r"[*_`#\[\]()]", "", s).strip()
    if compact.upper() in ("A", "B"):
        return compact.upper()

    m = re.search(r'"answer"\s*:\s*"([AB])"', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    explicit_patterns = [
        r"\b(?:answer|choice|option|selection|response|verdict|decision)\s*[:\-]\s*\**([AB])\**",
        r"\b(?:i\s+(?:would\s+)?(?:choose|pick|select|prefer))\s+(?:option\s+)?([AB])\b",
        r"\b(?:my\s+(?:answer|choice))\s+(?:is\s+)?([AB])\b",
        r"\*\*([AB])\*\*",
        r"\\boxed\{([AB])\}",
        r"(?:^|\n)\s*([AB])\s*[\.\)]\s*(?:$|\n)",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    m = re.search(r"\bAnswer\s*[:\-]?\s*\*?\*?([AB])\*?\*?\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    last_line = s.rstrip().split("\n")[-1].strip()
    cleaned_last = re.sub(r"[*_.#`]", "", last_line).strip()
    if cleaned_last.upper() in ("A", "B"):
        return cleaned_last.upper()
    m = re.search(r"([AB])\s*[\.\!]?\s*$", cleaned_last, re.IGNORECASE)
    if m and len(cleaned_last) <= 24:
        return m.group(1).upper()

    if len(s) <= 8:
        upper = s.upper()
        if upper.startswith("A"):
            return "A"
        if upper.startswith("B"):
            return "B"

    tail = s[-300:] if len(s) > 300 else s
    matches = re.findall(r"\b([AB])\b", tail)
    if matches:
        return matches[-1].upper()
    return None


def resolve_row_answer(row: dict) -> str | None:
    """Resolve A/B from a clean output row, including reasoning traces."""
    stored = row.get("answer")
    if stored in ("A", "B"):
        return stored
    content = row.get("content")
    reasoning = row.get("reasoning")
    return parse_answer(content) or parse_answer(reasoning)


def backfill_output_answers(model_key: str) -> int:
    """Rewrite output.jsonl rows with newly parsed answers. Returns rows updated."""
    resolved = resolve_model_output_path(model_key)
    if resolved is None:
        return 0

    rows: list[dict] = []
    updated = 0
    with open(resolved, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            answer = resolve_row_answer(row)
            if answer and answer != row.get("answer"):
                row["answer"] = answer
                updated += 1
            rows.append(row)

    if updated:
        with open(resolved, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return updated


def _answer_from_raw_row(row: dict, provider: str) -> str | None:
    """Parse A/B from a raw batch/API row when clean fields are missing."""
    if provider == "openai":
        msg = row.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or ""
    elif provider == "anthropic":
        res = row.get("result", {})
        msg = res.get("message", res)
        content = ""
        reasoning = ""
        for block in msg.get("content", []):
            if block.get("type") == "text":
                content = block["text"]
            elif block.get("type") == "thinking":
                reasoning = block.get("thinking", "")
    else:
        msg = row.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or ""
    return parse_answer(content) or parse_answer(reasoning)


def extract_clean_row(raw, provider, *, model_id=None, batch=False):
    """Extract clean row + cost entry from a raw API response."""

    content = None
    reasoning = None
    finish_reason = None
    usage = {}

    if provider == "anthropic":
        msg = raw.get("result", {}).get("message", raw.get("result", {}))
        for block in msg.get("content", []):
            if block.get("type") == "text":
                content = block["text"]
            elif block.get("type") == "thinking":
                reasoning = block.get("thinking", "")
        finish_reason = msg.get("stop_reason")
        resolved_model = model_id or msg.get("model")
        fields = usage_cost_breakdown(
            msg.get("usage", {}),
            provider="anthropic",
            model_id=resolved_model,
            batch=batch,
        )
        usage = {**fields, "model": resolved_model}
    else:
        body = raw.get("response", {}).get("body", {})
        choices = body.get("choices", [{}])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            reasoning = msg.get("reasoning")
            finish_reason = choices[0].get("finish_reason")
        resolved_model = model_id or body.get("model")
        fields = usage_cost_breakdown(
            body.get("usage", {}),
            provider=provider,
            model_id=resolved_model,
            batch=batch,
        )
        usage = {**fields, "model": resolved_model}

    answer = parse_answer(content) or parse_answer(reasoning)
    clean = {"custom_id": raw["custom_id"], "answer": answer, "finish_reason": finish_reason}
    if content and content.strip() not in ("A", "B"):
        clean["content"] = content
    if reasoning:
        clean["reasoning"] = reasoning

    cost_entry = {"custom_id": raw["custom_id"], **usage}
    return clean, cost_entry


def _ladder_id_from_custom_id(custom_id: str) -> str:
    return custom_id.rsplit("__", 2)[0]


def write_within_ladder_cost_artifacts(
    model_key: str,
    cost_entries: list[dict],
) -> Path | None:
    """Write phase6b_cost_log.json (with embedded summary) under within_ladder/."""
    if not cost_entries:
        return None

    pricing = get_model_pricing(model_key)
    output_path = model_output_path(model_key, "output.jsonl")
    records, token_totals, est_records_total, est_records_n = records_from_per_request_entries(
        cost_entries,
        group_key=lambda e: _ladder_id_from_custom_id(e["custom_id"]),
        result_path=str(output_path),
        pricing=pricing,
    )
    counts = cost_counts_from_entries(cost_entries)
    estimated_from_usage = estimate_cost_from_totals(
        pricing,
        prompt_tokens=token_totals["prompt"],
        completion_tokens=token_totals["completion"],
    )
    cost_log = build_phase6b_cost_log(
        model_key,
        records,
        pricing=pricing,
        prompt_total=token_totals["prompt"],
        completion_total=token_totals["completion"],
        reasoning_total=token_totals["reasoning"],
        calls_logged_total=token_totals["calls"],
        estimated_from_usage=estimated_from_usage,
        estimated_from_records_total=est_records_total,
        estimated_from_records_n=est_records_n,
        actual_total=counts["actual_total"],
        actual_n=counts["actual_n"],
        provider_reported_n=counts["provider_reported_n"],
        computed_n=counts["computed_n"],
        pricing_sources=counts["pricing_sources"],
    )
    summary = build_phase6b_cost_summary(
        model_key,
        calls_logged_total=token_totals["calls"],
        n_records=len(records),
        estimated_from_usage=estimated_from_usage,
        actual_total=counts["actual_total"],
        actual_n=counts["actual_n"],
        prompt_total=token_totals["prompt"],
        completion_total=token_totals["completion"],
        reasoning_total=token_totals["reasoning"],
        provider_reported_n=counts["provider_reported_n"],
        computed_n=counts["computed_n"],
    )
    out_dir = model_within_ladder_dir(model_key, get_results_root())
    return write_phase6b_cost_artifacts(out_dir, cost_log, summary)


def persist_per_request_cost_log(model_key: str, cost_entries: list[dict]) -> None:
    """Write cost_log.json and aggregated phase6b cost artifacts."""
    per_request_path = Path(model_output_path(model_key, PER_REQUEST_COST_LOG_NAME))
    totals = write_per_request_cost_log_file(per_request_path, model_key, cost_entries)
    cost_paths = write_within_ladder_cost_artifacts(model_key, cost_entries)
    if cost_paths is not None:
        summary = summary_from_phase6b_cost_log(
            json.loads(cost_paths.read_text(encoding="utf-8"))
        )
        cost_note = format_cost_artifact_note(summary, totals)
        print(
            f"[{model_key}] Cost artifacts: {totals['total_tokens']:,} tokens{cost_note} "
            f"-> {cost_paths.name}, {PER_REQUEST_COST_LOG_NAME}"
        )


def _enrich_cost_entries(model_key: str, cost_entries: list[dict]) -> list[dict]:
    """Backfill computed per-request cost when tokens exist but cost is null."""
    api_model, provider, _ = MODELS.get(model_key, (None, None, None))
    batch = provider in ("openai", "anthropic")
    enriched, changed = enrich_per_request_entries(
        cost_entries,
        provider=provider,
        model_id=api_model,
        batch=batch,
    )
    if changed:
        persist_per_request_cost_log(model_key, enriched)
    return enriched


def refresh_cost_artifacts(model_key: str) -> Path | None:
    """Rebuild phase6b cost log from existing per-request cost_log.json."""
    sync_artifacts_to_input(model_key, verbose=False)
    input_ids, norm_to_canonical = load_input_request_maps(model_key)
    per_request_path = Path(model_output_path(model_key, PER_REQUEST_COST_LOG_NAME))
    if not per_request_path.is_file():
        return None
    data = json.loads(per_request_path.read_text(encoding="utf-8"))
    entries = []
    for e in data.get("per_request") or []:
        cid = e.get("custom_id")
        if not cid:
            continue
        canonical = canonical_custom_id(cid, input_ids, norm_to_canonical)
        if canonical is not None:
            entries.append({**e, "custom_id": canonical})
    entries = _enrich_cost_entries(model_key, entries)
    return write_within_ladder_cost_artifacts(model_key, entries)


def write_clean_and_cost_log(raw_rows, provider, model_key):
    """Write clean output JSONL and cost artifacts from raw API rows."""
    api_model, _, _ = MODELS[model_key]
    batch = provider in ("openai", "anthropic")
    output_path = model_output_path(model_key, "output.jsonl")

    cost_entries = []
    with open(output_path, "w", encoding="utf-8") as f:
        for raw in raw_rows:
            clean, cost_entry = extract_clean_row(
                raw, provider, model_id=api_model, batch=batch
            )
            f.write(json.dumps(clean) + "\n")
            cost_entries.append(cost_entry)

    persist_per_request_cost_log(model_key, cost_entries)


def run_live(model_key, concurrency=5):
    """Run within-ladder validation via live OpenRouter API calls."""
    api_model, provider, extra_body = MODELS[model_key]
    if provider != "openrouter":
        print(f"[{model_key}] --run-live only supports openrouter models")
        return

    input_path = model_output_path(model_key, "input.jsonl")
    output_path = model_output_path(model_key, "output.jsonl")

    if not os.path.exists(input_path):
        print(f"No input file: {input_path}. Run --generate first.")
        return

    api_key = require_api_key("openrouter")

    with open(input_path) as f:
        requests = [json.loads(line) for line in f]
    input_ids, norm_to_canonical = load_input_request_maps(model_key)
    sync_artifacts_to_input(model_key)

    already_done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                canonical = canonical_custom_id(r["custom_id"], input_ids, norm_to_canonical)
                if canonical is not None:
                    already_done.add(canonical)
        print(f"[{model_key}] Resuming: {len(already_done)}/{len(requests)} already done")

    remaining = [r for r in requests if r["custom_id"] not in already_done]
    if not remaining:
        print(f"[{model_key}] All {len(requests)} requests complete.")
        return

    print(f"[{model_key}] Running {len(remaining)} requests via OpenRouter (concurrency={concurrency})")

    import httpx

    sem = asyncio.Semaphore(concurrency)
    raw_results = []
    errors = 0
    done = len(already_done)

    async def call_one(req):
        nonlocal errors, done
        async with sem:
            body = req["body"]
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json=body,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        raw_results.append({
                            "custom_id": req["custom_id"],
                            "response": {"body": data},
                        })
                        done += 1
                        if done % 200 == 0:
                            print(f"  [{model_key}] {done}/{len(requests)} done")
                        return
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        errors += 1
                        raw_results.append({
                            "custom_id": req["custom_id"],
                            "response": {"body": {"error": str(e)}},
                        })

    async def run_all():
        tasks = [call_one(r) for r in remaining]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())

    existing_rows = []
    if already_done:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                canonical = canonical_custom_id(r["custom_id"], input_ids, norm_to_canonical)
                if canonical is not None:
                    existing_rows.append({**r, "custom_id": canonical})

    all_clean = existing_rows
    new_cost_entries = []
    for raw in raw_results:
        clean, cost_entry = extract_clean_row(
            raw, "openrouter", model_id=api_model, batch=False
        )
        all_clean.append(clean)
        new_cost_entries.append(cost_entry)

    with open(output_path, "w") as f:
        for row in all_clean:
            f.write(json.dumps(row) + "\n")

    prev_cost_entries = []
    cost_path = model_output_path(model_key, PER_REQUEST_COST_LOG_NAME)
    if os.path.exists(cost_path):
        with open(cost_path) as f:
            prev_cost_entries = []
            for e in json.load(f).get("per_request", []):
                cid = e.get("custom_id")
                if not cid:
                    continue
                canonical = canonical_custom_id(cid, input_ids, norm_to_canonical)
                if canonical is not None:
                    prev_cost_entries.append({**e, "custom_id": canonical})
    all_cost_entries = prev_cost_entries + new_cost_entries
    persist_per_request_cost_log(model_key, all_cost_entries)

    print(f"[{model_key}] Done. {len(raw_results)} new, {errors} errors. Total: {done}/{len(requests)}")


def run_local(model_key):
    """Run within-ladder validation locally via vLLM logprobs (for base models)."""
    api_model, provider, _ = MODELS[model_key]
    if provider != "vllm_logprobs":
        print(f"[{model_key}] --run-local only supports vllm_logprobs models")
        return

    input_path = model_output_path(model_key, "input.jsonl")
    output_path = model_output_path(model_key, "output.jsonl")

    if not os.path.exists(input_path):
        print(f"No input file: {input_path}. Run --generate first.")
        return

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from llm_coherence.runtime.logprob_prompts import FEW_SHOT_PROMPT_LOGPROBS

    with open(input_path) as f:
        requests = [json.loads(line) for line in f]
    print(f"[{model_key}] Loaded {len(requests)} requests")

    model_source = os.environ.get("LLM_COHERENCE_VLLM_MODEL", api_model)
    cache_dir = os.environ.get("HF_HOME")
    print(f"[{model_key}] Model source: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=True,
        cache_dir=cache_dir,
    )
    token_id_a, token_id_b = resolve_choice_token_ids(tokenizer)
    print(f"[{model_key}] Token IDs: A={token_id_a}, B={token_id_b}")

    tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
    llm_kwargs = {
        "model": model_source,
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "enable_prefix_caching": True,
    }
    if cache_dir:
        llm_kwargs["download_dir"] = cache_dir
    llm_kwargs.update(vllm_load_kwargs_from_env())
    llm = LLM(**llm_kwargs)

    prompts = []
    for req in requests:
        prompt_text = req["prompt"]
        prompts.append(f"{FEW_SHOT_PROMPT_LOGPROBS}{prompt_text}\n\nAnswer:")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2,
        allowed_token_ids=[token_id_a, token_id_b],
    )

    print(f"[{model_key}] Running vLLM inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    if len(outputs) != len(requests):
        raise ForcedChoiceScoringError(
            f"vLLM returned {len(outputs)} outputs for {len(requests)} requests."
        )

    clean_rows = []
    for index, (req, output) in enumerate(zip(requests, outputs)):
        logprobs_per_pos = output.outputs[0].logprobs
        if not logprobs_per_pos or logprobs_per_pos[0] is None:
            raise ForcedChoiceScoringError(
                f"vLLM returned no next-token logprobs for request index {index} "
                f"({req['custom_id']})."
            )
        probabilities = normalized_choice_probabilities(
            logprobs_per_pos[0], token_id_a, token_id_b
        )
        prob_a = probabilities["A"]
        prob_b = probabilities["B"]

        winner = "A" if prob_a >= prob_b else "B"
        clean_rows.append({
            "custom_id": req["custom_id"],
            "answer": winner,
            "finish_reason": "stop",
            "probabilities": {"A": round(prob_a, 6), "B": round(prob_b, 6)},
        })

    with open(output_path, "w") as f:
        for r in clean_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{model_key}] Saved {len(clean_rows)} results to {output_path}")


def analyze(model_key):
    """Analyze within-ladder validation results for a specific model."""
    if resolve_model_input_path(model_key) is None:
        expected = within_ladder_artifact_path(model_key, "input.jsonl", get_results_root())
        print(f"No input file for {model_key}: {expected}. Run --generate first.")
        return

    sync_artifacts_to_input(model_key)
    input_ids, norm_to_canonical = load_input_request_maps(model_key)
    backfilled = backfill_output_answers(model_key)
    if backfilled:
        print(f"[{model_key}] Backfilled {backfilled} parsed answer(s) in output.jsonl")

    resolved = resolve_model_output_path(model_key)
    if resolved is None:
        expected = within_ladder_artifact_path(model_key, "output.jsonl", get_results_root())
        print(f"No output file for {model_key}: {expected}")
        return
    output_path = str(resolved)

    _, provider, _ = MODELS[model_key]
    ladders = load_ladders()
    valence_map = {l["original_statement_id"]: l["valence"] for l in ladders}
    expected_ladder_ids = {cid.rsplit("__", 2)[0] for cid in input_ids}

    results = []
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            canonical = canonical_custom_id(r["custom_id"], input_ids, norm_to_canonical)
            if canonical is not None:
                results.append({**r, "custom_id": canonical})

    pair_results = {}
    parse_errors = 0

    for r in results:
        cid = r["custom_id"]

        answer = resolve_row_answer(r)
        if answer is None:
            answer = _answer_from_raw_row(r, provider)

        parts = cid.rsplit("__", 2)
        ladder_id = parts[0]
        tier_pair = parts[1]
        direction = parts[2]

        ti = int(tier_pair.split("_vs_")[0][1:])
        tj = int(tier_pair.split("_vs_")[1][1:])

        if answer is None:
            parse_errors += 1
            continue

        key = (ladder_id, ti, tj)
        if key not in pair_results:
            pair_results[key] = {}

        # T7 = most_preferable for ALL ladders (both valences).
        # Negative valence = topic is negative, but T7 is still best (least severe).
        # AB: A=lower tier(ti), B=higher tier(tj) → correct = B (prefer higher)
        # BA: A=higher tier(tj), B=lower tier(ti) → correct = A (prefer higher)
        if direction == "AB":
            pair_results[key]["correct_AB"] = (answer == "B")
        else:
            pair_results[key]["correct_BA"] = (answer == "A")

    # Aggregate per ladder
    ladder_scores = {}
    for (ladder_id, ti, tj), res in pair_results.items():
        if ladder_id not in ladder_scores:
            valence = valence_map.get(ladder_id, "positive")
            ladder_scores[ladder_id] = {"correct": 0, "total": 0, "by_distance": {}, "valence": valence}

        distance = tj - ti
        if distance not in ladder_scores[ladder_id]["by_distance"]:
            ladder_scores[ladder_id]["by_distance"][distance] = {"correct": 0, "total": 0}

        for key in ["correct_AB", "correct_BA"]:
            if key in res:
                ladder_scores[ladder_id]["total"] += 1
                ladder_scores[ladder_id]["by_distance"][distance]["total"] += 1
                if res[key]:
                    ladder_scores[ladder_id]["correct"] += 1
                    ladder_scores[ladder_id]["by_distance"][distance]["correct"] += 1

    # Summary
    print(f"\n=== Within-Ladder Validation: {model_key} ===")
    print(f"Parse errors: {parse_errors}")
    print(f"Ladders scored: {len(ladder_scores)} (expected {len(expected_ladder_ids)} from input)")
    if len(ladder_scores) != len(expected_ladder_ids):
        missing = expected_ladder_ids - set(ladder_scores)
        print(f"  Missing output for {len(missing)} ladder(s); re-run --run-live or --fetch")

    overall_correct = sum(s["correct"] for s in ladder_scores.values())
    overall_total = sum(s["total"] for s in ladder_scores.values())
    if overall_total > 0:
        print(f"Overall accuracy: {overall_correct}/{overall_total} ({100*overall_correct/overall_total:.1f}%)")
    else:
        print("Overall accuracy: no scored pairs (output missing or empty after sync)")

    # By valence
    for v in ["positive", "negative"]:
        v_correct = sum(s["correct"] for s in ladder_scores.values() if s["valence"] == v)
        v_total = sum(s["total"] for s in ladder_scores.values() if s["valence"] == v)
        if v_total > 0:
            print(f"  {v}: {v_correct}/{v_total} ({100*v_correct/v_total:.1f}%)")

    # By tier distance
    print(f"\nAccuracy by tier distance:")
    for d in range(1, 7):
        d_correct = sum(
            s["by_distance"].get(d, {}).get("correct", 0)
            for s in ladder_scores.values()
        )
        d_total = sum(
            s["by_distance"].get(d, {}).get("total", 0)
            for s in ladder_scores.values()
        )
        if d_total > 0:
            print(f"  Distance {d}: {d_correct}/{d_total} ({100*d_correct/d_total:.1f}%)")

    # Per-ladder scores
    per_ladder = []
    for lid, s in ladder_scores.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        per_ladder.append({"ladder_id": lid, "accuracy": acc, "n": s["total"], "valence": s["valence"]})
    per_ladder.sort(key=lambda x: x["accuracy"])

    print(f"\nWorst 10 ladders (lowest accuracy):")
    for item in per_ladder[:10]:
        print(f"  {item['ladder_id']}: {item['accuracy']:.2%} ({item['n']} pairs, {item['valence']})")

    print(f"\nBest 10 ladders:")
    for item in per_ladder[-10:]:
        print(f"  {item['ladder_id']}: {item['accuracy']:.2%} ({item['n']} pairs, {item['valence']})")

    # Save full results
    summary = {
        "model_key": model_key,
        "variations_path": str(get_variations_path()),
        "n_ladders_expected": len(expected_ladder_ids),
        "n_requests_expected": len(input_ids),
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "n_ladders": len(ladder_scores),
        "n_total_pairs": overall_total,
        "parse_errors": parse_errors,
        "per_ladder": per_ladder,
        "by_distance": {d: {
            "correct": sum(s["by_distance"].get(d, {}).get("correct", 0) for s in ladder_scores.values()),
            "total": sum(s["by_distance"].get(d, {}).get("total", 0) for s in ladder_scores.values()),
        } for d in range(1, 7)},
    }
    summary_path = model_output_path(model_key, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    cost_paths = refresh_cost_artifacts(model_key)
    if cost_paths is not None:
        print(f"Cost log: {cost_paths}")


def discover_within_ladder_models(results_root: Path | None = None) -> list[str]:
    """Return model keys that have a within_ladder/input.jsonl under results_root."""
    root = results_root if results_root is not None else get_results_root()
    if not root.is_dir():
        return []
    model_keys: list[str] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        if (model_dir / "within_ladder" / "input.jsonl").is_file():
            model_keys.append(model_dir.name)
    return model_keys


def analyze_all(*, results_root: Path | None = None) -> None:
    """Prune stale artifacts and re-analyze every model with within_ladder results."""
    root = results_root if results_root is not None else get_results_root()
    model_keys = discover_within_ladder_models(root)
    if not model_keys:
        print(f"No within_ladder runs found under {root}")
        return

    print(f"Analyzing {len(model_keys)} model(s) under {root}")
    for model_key in model_keys:
        print(f"\n{'=' * 60}")
        if model_key not in MODELS:
            print(f"[{model_key}] Skipping: not in MODELS registry")
            continue
        try:
            analyze(model_key)
        except Exception as exc:
            print(f"[{model_key}] Analyze failed: {exc}")


def build_within_ladder_hf_job_code(
    *,
    model_key: str,
    variations: str,
    results_dir: str,
    start_from: int,
    max_variation_sets: int | None,
    hub_dataset: str | None,
    path_in_repo: str,
) -> str:
    """Build in-container Python for an HF Jobs within-ladder run."""
    base_cmd = [
        "python3",
        "scripts/04_model_runs/10a_run_within_ladder_experiment.py",
        "--model",
        model_key,
        "--variations",
        variations,
        "--results-dir",
        results_dir,
        "--start-from",
        str(start_from),
    ]
    if max_variation_sets is not None:
        base_cmd.extend(["--max-variation-sets", str(max_variation_sets)])

    payload = {
        "base_cmd": base_cmd,
        "upload_dir": f"{results_dir.rstrip('/')}/{model_key}/within_ladder",
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

print("=== within-ladder HF job start ===", flush=True)
print("base command:", " ".join(payload["base_cmd"]), flush=True)

for phase in ("--generate", "--run-local", "--analyze"):
    cmd = payload["base_cmd"] + [phase]
    print("\\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

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

print("=== within-ladder HF job complete ===", flush=True)
""".strip()


def submit_within_ladder_hf_job(args: argparse.Namespace) -> int:
    """Submit the existing within-ladder experiment CLI to Hugging Face Jobs."""
    if not args.image:
        raise SystemExit("--image is required with --submit-hf-job")
    if not args.namespace:
        raise SystemExit("--namespace is required with --submit-hf-job")

    job_tag = args.job_tag or uuid.uuid4().hex[:8]
    path_in_repo = args.path_in_repo or f"outputs/{args.model}/within_ladder"
    code = build_within_ladder_hf_job_code(
        model_key=args.model,
        variations=repo_relative(args.variations),
        results_dir=repo_relative(args.results_dir),
        start_from=args.start_from,
        max_variation_sets=args.max_variation_sets,
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


def main():
    global _RESULTS_ROOT, _VARIATIONS_PATH, _START_FROM, _MAX_VARIATION_SETS, _SMOKE_SCOPE

    parser = argparse.ArgumentParser(description="Within-ladder tier-pair experiment")
    parser.add_argument("--model", type=str, help="Model key from MODELS registry")
    parser.add_argument("--generate", action="store_true", help="Generate batch input")
    parser.add_argument("--generate-all", action="store_true", help="Generate for all representative models")
    parser.add_argument("--submit", action="store_true", help="Submit batch")
    parser.add_argument("--fetch", action="store_true", help="Fetch results")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument(
        "--analyze-all",
        action="store_true",
        help="Prune stale artifacts and analyze every model with within_ladder/input.jsonl",
    )
    parser.add_argument(
        "--run-batch",
        action="store_true",
        help="Generate, submit, poll until complete, and analyze (OpenAI/Anthropic batch models).",
    )
    parser.add_argument("--run-live", action="store_true", help="Run via live API calls (OpenRouter)")
    parser.add_argument("--run-local", action="store_true", help="Run locally via vLLM logprobs (base models)")
    parser.add_argument(
        "--submit-hf-job",
        action="store_true",
        help="Submit this within-ladder run to Hugging Face Jobs.",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency for --run-live")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status checks for --run-batch (default: 30).",
    )
    parser.add_argument("--with-reasoning", action="store_true", help="Include reasoning in prompt")
    parser.add_argument(
        "--results-dir",
        default=str(WITHIN_LADDER_RUNS_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Model-run root (default: outputs/). Artifacts: <results-dir>/<model>/within_ladder/.",
    )
    parser.add_argument(
        "--variations",
        default=str(DEFAULT_VARIATIONS_PATH.relative_to(REPO_ROOT)),
        help=(
            "Ladder variations JSON "
            "(default: data/05_ladder_validation/phase6b_variations_pruned_final.json)"
        ),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip this many ladders before running.",
    )
    parser.add_argument(
        "--max-variation-sets",
        type=int,
        default=None,
        help="Run at most this many ladders (after --start-from).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Smoke run: slice to a small ladder subset (default --max-variation-sets 1) "
            "and run the full batch pipeline (generate → submit → fetch → analyze) "
            "unless another step flag is set. Output stays at "
            "<results-dir>/<model>/within_ladder/."
        ),
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
        "--hub-dataset",
        default=None,
        help="Optional existing HF dataset repo for uploading HF Jobs outputs, e.g. org/repo.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional dataset subdir for HF Jobs outputs. Defaults to outputs/<model>/within_ladder.",
    )
    parser.add_argument("--job-tag", default=None, help="Stable short tag for HF Jobs metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated HF job code and exit.")
    args = parser.parse_args()

    if args.start_from < 0:
        parser.error("--start-from must be >= 0")
    if args.max_variation_sets is not None and args.max_variation_sets < 1:
        parser.error("--max-variation-sets must be >= 1 when set")
    if args.poll_interval < 1:
        parser.error("--poll-interval must be >= 1")
    if args.model_volume and not Path(args.model_volume_path).is_absolute():
        parser.error("--model-volume-path must be absolute")

    has_step = any(
        (
            args.generate,
            args.generate_all,
            args.submit,
            args.fetch,
            args.analyze,
            args.analyze_all,
            args.run_batch,
            args.run_live,
            args.run_local,
            args.submit_hf_job,
        )
    )
    if args.smoke and not has_step:
        args.run_batch = True
        if args.max_variation_sets is None:
            args.max_variation_sets = 1
            print("Smoke run: defaulting --max-variation-sets to 1")

    _RESULTS_ROOT = resolve_under_repo(args.results_dir)
    _VARIATIONS_PATH = resolve_under_repo(args.variations)
    _START_FROM = args.start_from
    _MAX_VARIATION_SETS = args.max_variation_sets
    _SMOKE_SCOPE = args.smoke

    def _print_run_header(model_key: str) -> None:
        print(f"Variations: {_VARIATIONS_PATH}")
        print(f"Results dir: {model_within_ladder_dir(model_key, get_results_root())}")
        if _SMOKE_SCOPE:
            print(
                f"  Smoke scope: start_from={_START_FROM}, "
                f"max_variation_sets={_MAX_VARIATION_SETS if _MAX_VARIATION_SETS is not None else 'all'}"
            )

    if args.generate_all:
        ladders = load_ladders()
        print(f"Results root: {get_results_root()}")
        print(f"Loaded {len(ladders)} ladders from {_VARIATIONS_PATH}")
        for model_key in REPRESENTATIVE_SUBSET:
            print(f"\n--- {model_key} ---")
            n_requests, _ = write_batch_input(model_key, with_reasoning=args.with_reasoning)
            out_path = model_output_path(model_key, "input.jsonl")
            print(f"  Generated {n_requests} requests to {out_path}")
        return

    if args.analyze_all:
        analyze_all()
        return

    if not args.model:
        parser.error("--model is required (unless using --generate-all or --analyze-all)")

    if args.model not in MODELS:
        available = sorted(MODELS.keys())
        parser.error(
            f"Unknown model: {args.model}. "
            f"{len(available)} models available (from config.py); "
            f"includes: {', '.join(available[:8])}, ..."
        )

    _print_run_header(args.model)

    if args.submit_hf_job:
        return submit_within_ladder_hf_job(args)

    if args.generate:
        n_requests, n_ladders = write_batch_input(args.model, with_reasoning=args.with_reasoning)
        out_path = model_output_path(args.model, "input.jsonl")
        print(f"Loaded {n_ladders} ladders from {_VARIATIONS_PATH}")
        print(f"Generated {n_requests} requests to {out_path}")
        print(f"  = {n_ladders} ladders × 21 pairs × 2 directions = {n_ladders * 42}")

    elif args.run_batch:
        run_batch_pipeline(
            args.model,
            with_reasoning=args.with_reasoning,
            poll_interval=args.poll_interval,
        )

    elif args.run_live:
        run_live(args.model, concurrency=args.concurrency)

    elif args.run_local:
        run_local(args.model)

    elif args.submit:
        submit_batch(args.model)

    elif args.fetch:
        result = fetch_results(args.model)
        if result is None:
            print(f"[{args.model}] Batch not complete yet.")

    elif args.analyze:
        analyze(args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
