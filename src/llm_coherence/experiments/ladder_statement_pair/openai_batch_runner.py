"""OpenAI Batch API support for the ladder-vs-comparison experiment.

The historical GPT-5.4 subject-model runs used the OpenAI Batch API, while the
current step-10b runner otherwise issues live requests through LiteLLM.  This
module restores a reproducible batch path that:

* writes the official per-request JSONL format;
* shards at the API's 50,000-request/file limit;
* records enough run metadata to rebuild the existing per-ladder result schema;
* downloads successful and error files without assuming output order; and
* retries transient transport/server failures while preserving deterministic
  errors and fixed experimental response-token ceilings;
* preserves available opted-in OpenAI reasoning summaries in per-ladder JSONL
  sidecars and records coverage when the provider omits a summary.

Native reasoning is controlled by ``MODEL_CONFIGS.extra_body``.  In particular,
GPT-5.6's off condition must include ``reasoning_effort="none"`` explicitly;
omitting the field would use the model's non-off default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from llm_coherence.config import (
    canonical_model_key,
    get_model_config,
    validate_openai_responses_max_output_tokens,
)
from llm_coherence.experiments.ladder_statement_pair.experiment_runner_tradeoff import (
    RESULTS_SCHEMA_VERSION,
    _file_sha256,
    _git_sha,
    _package_versions,
    _summarize_usage,
    artifact_dir_name_for_test,
    build_prompt,
    counts_from_responses,
    load_comparisons,
    save_results,
)
from llm_coherence.runtime.agents import MODEL_SPECS, model_name_for_key
from llm_coherence.runtime.api_keys import require_api_key
from llm_coherence.runtime.usage_cost import (
    estimate_cost_from_totals,
    resolve_rates,
    usage_cost_breakdown,
)
from llm_coherence.runtime.utils import parse_responses_forced_choice


MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200_000_000
OPENAI_BATCH_ENDPOINT = "/v1/responses"
BATCH_RUNS_DIRNAME = "batch_runs"
LATEST_BATCH_RUN_NAME = "latest_batch_run.txt"
BATCH_MANIFEST_NAME = "batch_manifest.json"
BATCH_JOBS_NAME = "batch_jobs.json"
BATCH_PROCESSING_SUMMARY_NAME = "batch_processing_summary.json"
REASONING_SUMMARIES_NAME = "reasoning_summaries.jsonl"

# Pre-submit estimates must remain offline. These values match the transparent
# heuristic used by the within-ladder Batch runner and are calibrated against
# completed GPT-5.6 forced-choice runs.
_ESTIMATED_UTF8_BYTES_PER_INPUT_TOKEN = 5
_ESTIMATED_INPUT_FRAMING_TOKENS_PER_REQUEST = 8
_ESTIMATED_REASONING_OFF_OUTPUT_TOKENS_PER_REQUEST = 5

_CUSTOM_ID_RE = re.compile(
    r"^s(?P<set>\d+)-c(?P<comparison>\d+)-d(?P<direction>ab|ba)-t(?P<trial>\d+)$"
)
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429})
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "batch_expired",
        "connection_error",
        "rate_limit_exceeded",
        "server_error",
        "timeout",
    }
)

# Current published per-model Batch queue limits, in enqueued input tokens.
# Source checked 2026-08-10:
# https://developers.openai.com/api/docs/models/gpt-5.6-sol
# https://developers.openai.com/api/docs/models/gpt-5.6-terra
# https://developers.openai.com/api/docs/models/gpt-5.6-luna
OPENAI_BATCH_QUEUE_LIMITS: dict[str, dict[int, int]] = {
    "gpt-5.6-sol": {
        1: 1_500_000,
        2: 3_000_000,
        3: 100_000_000,
        4: 200_000_000,
        5: 15_000_000_000,
    },
    "gpt-5.6-terra": {
        1: 1_500_000,
        2: 3_000_000,
        3: 100_000_000,
        4: 200_000_000,
        5: 15_000_000_000,
    },
    "gpt-5.6-luna": {
        1: 5_000_000,
        2: 20_000_000,
        3: 40_000_000,
        4: 1_000_000_000,
        5: 15_000_000_000,
    },
}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_id(model_key: str) -> str:
    resolved_key = canonical_model_key(model_key)
    spec = MODEL_SPECS.get(resolved_key)
    if spec is None or spec.model_type != "openai":
        raise ValueError(
            f"OpenAI Batch API requires an OpenAI model key; got {model_key!r}."
        )
    model_id = model_name_for_key(model_key)
    if not model_id:
        raise ValueError(f"No provider model ID configured for {model_key!r}.")
    return model_id.removeprefix("openai/")


def _reasoning_effort(model_key: str) -> str | None:
    cfg = get_model_config(model_key)
    effort = (cfg.extra_body or {}).get("reasoning_effort")
    return str(effort) if effort is not None else None


def build_openai_responses_batch_body(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict[str, Any] | None,
    system_message: str | None,
) -> dict[str, Any]:
    """Build one Responses API request body for an OpenAI batch line."""
    validate_openai_responses_max_output_tokens(model_id, max_tokens)
    extra = dict(extra_body or {})
    body: dict[str, Any] = {
        "model": model_id,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if system_message is not None:
        body["instructions"] = system_message
    effort = extra.get("reasoning_effort")
    # Do not collapse explicit "none" into omission: GPT-5.6 otherwise uses a
    # reasoning-enabled default.  Temperature is valid for the off condition;
    # reasoning-on requests intentionally omit it, matching the GPT-5.4 path.
    if effort is not None:
        body["reasoning"] = {"effort": effort}
    if effort in (None, "none"):
        body["temperature"] = temperature
    for key, value in extra.items():
        if key in {"reasoning_effort", "temperature"}:
            continue
        if key == "reasoning" and isinstance(value, dict):
            body["reasoning"] = {**body.get("reasoning", {}), **value}
            continue
        body[key] = value
    return body


def batch_queue_limit_for_model(
    model_id: str,
    *,
    usage_tier: int | None,
    explicit_limit: int | None = None,
) -> int:
    """Resolve a safe queue limit from an explicit override or usage tier."""
    if explicit_limit is not None:
        if explicit_limit < 1:
            raise ValueError("batch queue token limit must be at least 1.")
        return explicit_limit
    if usage_tier is None:
        raise ValueError(
            "Submitting OpenAI batches requires --batch-usage-tier or "
            "--batch-queue-token-limit so the runner cannot overfill the account queue."
        )
    limits = OPENAI_BATCH_QUEUE_LIMITS.get(model_id)
    if limits is None or usage_tier not in limits:
        raise ValueError(
            f"No published Batch queue limit is recorded for {model_id!r} at "
            f"usage tier {usage_tier}; pass --batch-queue-token-limit explicitly."
        )
    return limits[usage_tier]


def encode_custom_id(
    set_index: int,
    comparison_index: int,
    direction: str,
    trial_index: int,
) -> str:
    if direction not in {"ab", "ba"}:
        raise ValueError(f"Unknown direction: {direction!r}")
    return (
        f"s{set_index:04d}-c{comparison_index:04d}-"
        f"d{direction}-t{trial_index:03d}"
    )


def decode_custom_id(custom_id: str) -> tuple[int, int, str, int]:
    match = _CUSTOM_ID_RE.fullmatch(custom_id)
    if match is None:
        raise ValueError(f"Invalid ladder-vs-comparison batch custom_id: {custom_id!r}")
    return (
        int(match.group("set")),
        int(match.group("comparison")),
        match.group("direction"),
        int(match.group("trial")),
    )


def _iter_batch_requests(
    *,
    run_items: list[dict[str, Any]],
    data_dir: Path,
    model_key: str,
    num_trials: int,
    include_flipped: bool,
    with_reasoning: bool,
    max_tokens: int,
    temperature: float,
    system_message: str | None,
) -> Iterator[dict[str, Any]]:
    cfg = get_model_config(model_key)
    model_id = _model_id(model_key)
    for set_index, item in enumerate(run_items):
        comparisons = load_comparisons(
            data_dir,
            item["test_name"],
            Path(item["comparison_path"]),
        )
        for comparison_index, comparison in enumerate(comparisons):
            text_a = comparison["outcome_a"]["text"]
            text_b = comparison["outcome_b"]["text"]
            directions = [("ab", text_a, text_b)]
            if include_flipped:
                directions.append(("ba", text_b, text_a))
            for direction, option_a, option_b in directions:
                prompt = build_prompt(
                    option_a,
                    option_b,
                    with_reasoning=with_reasoning,
                    cache_structure=False,
                )
                body = build_openai_responses_batch_body(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=cfg.extra_body,
                    system_message=system_message,
                )
                for trial_index in range(num_trials):
                    yield {
                        "custom_id": encode_custom_id(
                            set_index,
                            comparison_index,
                            direction,
                            trial_index,
                        ),
                        "method": "POST",
                        "url": OPENAI_BATCH_ENDPOINT,
                        "body": body,
                    }


def _request_input_token_upper_bound(request: dict[str, Any]) -> int:
    """Conservative byte-level upper bound for enqueued input tokens.

    OpenAI does not expose an offline Batch queue-token counter. A byte-level
    bound is deliberately conservative: byte-pair tokenization cannot require
    more tokens than the UTF-8 bytes representing the request text. The small
    fixed allowance covers request/message framing.
    """

    def text_bytes(value: Any) -> int:
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, list):
            return sum(text_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(text_bytes(item) for item in value.values())
        return 0

    body = request.get("body") or {}
    text_fields = {
        key: body[key]
        for key in ("input", "instructions", "messages")
        if key in body
    }
    return max(1, text_bytes(text_fields) + 64)


def _write_sharded_requests(
    requests: Iterable[dict[str, Any]],
    *,
    run_dir: Path,
    prefix: str,
    kind: str,
    attempt: int,
    max_requests_per_batch: int,
    max_bytes_per_batch: int = MAX_BATCH_FILE_BYTES,
) -> tuple[list[dict[str, Any]], int]:
    if not 1 <= max_requests_per_batch <= MAX_BATCH_REQUESTS:
        raise ValueError(
            f"max_requests_per_batch must be between 1 and {MAX_BATCH_REQUESTS:,}."
        )
    if not 1 <= max_bytes_per_batch <= MAX_BATCH_FILE_BYTES:
        raise ValueError(
            f"max_bytes_per_batch must be between 1 and {MAX_BATCH_FILE_BYTES:,}."
        )

    shards: list[dict[str, Any]] = []
    handle = None
    shard_path: Path | None = None
    shard_count = 0
    shard_bytes = 0
    shard_input_token_upper_bound = 0
    total = 0
    shard_index = 0

    def close_shard() -> None:
        nonlocal handle, shard_path, shard_count, shard_bytes
        nonlocal shard_input_token_upper_bound
        if handle is None or shard_path is None:
            return
        handle.close()
        shards.append(
            {
                "input_file": shard_path.name,
                "request_count": shard_count,
                "byte_size": shard_bytes,
                "input_token_upper_bound": shard_input_token_upper_bound,
                "sha256": _file_sha256(shard_path),
                "kind": kind,
                "attempt": attempt,
            }
        )
        handle = None
        shard_path = None
        shard_count = 0
        shard_bytes = 0
        shard_input_token_upper_bound = 0

    try:
        for request in requests:
            line = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
            if len(line) > max_bytes_per_batch:
                raise ValueError(
                    f"One Batch request is {len(line):,} bytes, exceeding the "
                    f"{max_bytes_per_batch:,}-byte file limit."
                )
            if handle is not None and (
                shard_count >= max_requests_per_batch
                or shard_bytes + len(line) > max_bytes_per_batch
            ):
                close_shard()
            if handle is None:
                shard_path = run_dir / f"{prefix}_{shard_index:03d}.jsonl"
                handle = shard_path.open("wb")
                shard_index += 1
            handle.write(line)
            shard_count += 1
            shard_bytes += len(line)
            shard_input_token_upper_bound += _request_input_token_upper_bound(request)
            total += 1
    finally:
        close_shard()
    return shards, total


def generate_batch_run(
    *,
    run_items: list[dict[str, Any]],
    data_dir: Path,
    source_manifest_path: Path,
    results_dir: Path,
    model_key: str,
    num_trials: int,
    include_flipped: bool = True,
    with_reasoning: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    system_message: str | None = None,
    max_requests_per_batch: int = MAX_BATCH_REQUESTS,
) -> Path:
    """Generate sharded JSONL inputs and a reconstruction manifest."""
    if not run_items:
        raise ValueError("At least one variation set is required for a batch run.")
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1.")
    model_key = canonical_model_key(model_key)
    cfg = get_model_config(model_key)
    resolved_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
    resolved_temperature = temperature if temperature is not None else cfg.temperature
    model_id = _model_id(model_key)
    validate_openai_responses_max_output_tokens(model_id, resolved_max_tokens)

    batch_root = results_dir / BATCH_RUNS_DIRNAME
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = batch_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    def relative_from_run(path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), run_dir.resolve())).as_posix()

    set_records: list[dict[str, Any]] = []
    for item in run_items:
        comparison_path = Path(item["comparison_path"]).resolve()
        comparisons = load_comparisons(data_dir, item["test_name"], comparison_path)
        set_records.append(
            {
                "test_name": item["test_name"],
                "comparison_path": relative_from_run(comparison_path),
                "comparison_file_sha256": _file_sha256(comparison_path),
                "comparison_count": len(comparisons),
            }
        )

    requests = _iter_batch_requests(
        run_items=run_items,
        data_dir=data_dir,
        model_key=model_key,
        num_trials=num_trials,
        include_flipped=include_flipped,
        with_reasoning=with_reasoning,
        max_tokens=resolved_max_tokens,
        temperature=resolved_temperature,
        system_message=system_message,
    )
    shards, total_requests = _write_sharded_requests(
        requests,
        run_dir=run_dir,
        prefix="batch_input",
        kind="initial",
        attempt=0,
        max_requests_per_batch=max_requests_per_batch,
    )
    expected_total = sum(
        item["comparison_count"]
        * num_trials
        * (2 if include_flipped else 1)
        for item in set_records
    )
    if total_requests != expected_total:
        raise RuntimeError(
            f"Generated {total_requests:,} requests; expected {expected_total:,}."
        )

    manifest = {
        "schema_version": "2.0",
        "created_at": _utc_now(),
        "model_key": model_key,
        "model_id": model_id,
        "reasoning_effort": _reasoning_effort(model_key),
        "reasoning_artifact_type": cfg.reasoning_artifact_type,
        "extra_body": dict(cfg.extra_body or {}),
        "num_trials": num_trials,
        "include_flipped": include_flipped,
        "with_reasoning": with_reasoning,
        "prompt_template_used": (
            "comparison_prompt_template_reasoning_default"
            if with_reasoning
            else "comparison_prompt_template_default"
        ),
        "max_tokens": resolved_max_tokens,
        "temperature": (
            resolved_temperature
            if _reasoning_effort(model_key) in (None, "none")
            else None
        ),
        "configured_temperature": resolved_temperature,
        "system_message": system_message,
        "infrastructure": "openai_batch_api",
        "api_endpoint": OPENAI_BATCH_ENDPOINT,
        "source_manifest_path": relative_from_run(source_manifest_path),
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "data_dir": relative_from_run(data_dir),
        "results_dir": relative_from_run(results_dir),
        "total_requests": total_requests,
        "total_input_token_upper_bound": sum(
            int(shard.get("input_token_upper_bound", 0)) for shard in shards
        ),
        "max_requests_per_batch": max_requests_per_batch,
        "max_bytes_per_batch": MAX_BATCH_FILE_BYTES,
        "sets": set_records,
        "shards": shards,
        "retry_history": [],
    }
    _atomic_write_json(run_dir / BATCH_MANIFEST_NAME, manifest)
    _atomic_write_text(batch_root / LATEST_BATCH_RUN_NAME, run_dir.name + "\n")
    return run_dir


def resolve_batch_run_dir(results_dir: Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path.resolve() if path.is_absolute() else (results_dir / path).resolve()
    pointer = results_dir / BATCH_RUNS_DIRNAME / LATEST_BATCH_RUN_NAME
    if not pointer.is_file():
        raise FileNotFoundError(
            f"No latest batch-run pointer at {pointer}. Generate a batch first or "
            "pass --batch-run-dir."
        )
    run_name = pointer.read_text(encoding="utf-8").strip()
    return (pointer.parent / run_name).resolve()


def load_batch_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / BATCH_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_relative_path(run_dir: Path, stored: str) -> Path:
    """Resolve schema-v2 relative paths while remaining compatible with v1."""
    path = Path(stored)
    return path.resolve() if path.is_absolute() else (run_dir / path).resolve()


def _manifest_data_dir(run_dir: Path, manifest: dict[str, Any]) -> Path:
    return _resolve_run_relative_path(run_dir, manifest["data_dir"])


def _manifest_results_dir(run_dir: Path, manifest: dict[str, Any]) -> Path:
    return _resolve_run_relative_path(run_dir, manifest["results_dir"])


def validate_batch_run_binding(
    run_dir: Path,
    *,
    model_key: str,
    results_dir: Path,
) -> dict[str, Any]:
    """Prevent an explicit run directory from being processed under another model/scope."""
    manifest = load_batch_manifest(run_dir)
    manifest_model_key = str(manifest.get("model_key", ""))
    if canonical_model_key(manifest_model_key) != canonical_model_key(model_key):
        raise ValueError(
            f"Batch run {run_dir.name!r} belongs to {manifest.get('model_key')!r}, "
            f"not CLI model {model_key!r}."
        )
    recorded_results = _manifest_results_dir(run_dir, manifest)
    if recorded_results != results_dir.resolve():
        raise ValueError(
            f"Batch run {run_dir.name!r} writes to {recorded_results}, but the CLI "
            f"resolved {results_dir.resolve()}. Check --smoke, --results-dir, and "
            "--batch-run-dir."
        )
    return manifest


def validate_batch_inputs(run_dir: Path) -> dict[str, Any]:
    """Fail if any scientific input or generated JSONL shard drifted after generation."""
    manifest = load_batch_manifest(run_dir)
    problems: list[str] = []

    source_path = _resolve_run_relative_path(run_dir, manifest["source_manifest_path"])
    if not source_path.is_file():
        problems.append(f"source manifest is missing: {source_path}")
    elif _file_sha256(source_path) != manifest.get("source_manifest_sha256"):
        problems.append(f"source manifest hash changed: {source_path}")

    data_dir = _manifest_data_dir(run_dir, manifest)
    for record in manifest.get("sets", []):
        path = _resolve_run_relative_path(run_dir, record["comparison_path"])
        if not path.is_file():
            problems.append(f"comparison file is missing: {path}")
            continue
        if _file_sha256(path) != record.get("comparison_file_sha256"):
            problems.append(f"comparison file hash changed: {path}")
            continue
        comparisons = load_comparisons(data_dir, record["test_name"], path)
        if len(comparisons) != int(record.get("comparison_count", -1)):
            problems.append(
                f"comparison count changed for {record['test_name']}: "
                f"{len(comparisons)} != {record.get('comparison_count')}"
            )

    for shard in manifest.get("shards", []):
        path = run_dir / shard["input_file"]
        if not path.is_file():
            problems.append(f"batch input shard is missing: {path}")
            continue
        expected_hash = shard.get("sha256")
        if expected_hash and _file_sha256(path) != expected_hash:
            problems.append(f"batch input shard hash changed: {path}")
        expected_bytes = shard.get("byte_size")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            problems.append(
                f"batch input shard size changed: {path} "
                f"({path.stat().st_size} != {expected_bytes})"
            )
        if path.stat().st_size > MAX_BATCH_FILE_BYTES:
            problems.append(
                f"batch input shard exceeds {MAX_BATCH_FILE_BYTES:,} bytes: {path}"
            )
        row_count = 0
        token_bound = 0
        expected_endpoint = manifest.get("api_endpoint")
        for row in _iter_jsonl(path):
            row_count += 1
            token_bound += _request_input_token_upper_bound(row)
            if expected_endpoint and row.get("url") != expected_endpoint:
                problems.append(
                    f"batch input endpoint mismatch in {path}: "
                    f"{row.get('url')!r} != {expected_endpoint!r}"
                )
                break
        if row_count != int(shard.get("request_count", -1)):
            problems.append(
                f"batch input request count changed: {path} "
                f"({row_count} != {shard.get('request_count')})"
            )
        recorded_bound = shard.get("input_token_upper_bound")
        if recorded_bound is not None and token_bound != int(recorded_bound):
            problems.append(
                f"batch input token bound changed: {path} "
                f"({token_bound} != {recorded_bound})"
            )

    if problems:
        preview = "\n  - ".join(problems[:10])
        raise ValueError(
            "Batch inputs no longer match the generation manifest; refusing to "
            f"submit or reconstruct results:\n  - {preview}"
        )
    return manifest


def _request_input_text_bytes(request: dict[str, Any]) -> int:
    """Count UTF-8 bytes in the text-bearing fields of one Batch request."""

    def text_bytes(value: Any) -> int:
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, list):
            return sum(text_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(text_bytes(item) for item in value.values())
        return 0

    body = request.get("body") or {}
    text_fields = {
        key: body[key]
        for key in ("input", "instructions", "messages")
        if key in body
    }
    return text_bytes(text_fields)


def _batch_request_uses_reasoning(body: dict[str, Any]) -> bool:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    else:
        # Compatibility with older generated Chat Completions Batch files.
        effort = body.get("reasoning_effort")
    return effort not in (None, "none")


def estimate_batch_run_pre_submit_cost(
    run_dir: Path,
    *,
    shards: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Estimate a generated Step-10b OpenAI Batch run without API calls.

    Input tokens use the same UTF-8 text-size heuristic as the within-ladder
    runner. Reasoning-off requests use the observed forced-choice norm of five
    output tokens per request; reasoning-on requests use their configured caps
    so the planning estimate remains conservative. ``shards`` restricts the
    estimate to a newly created retry wave.
    """
    manifest = validate_batch_inputs(run_dir)
    model_id = str(manifest["model_id"])
    pricing, pricing_source = resolve_rates("openai", model_id, batch=True)
    if pricing is None:
        return None

    request_count = 0
    reasoning_on_requests = 0
    input_text_bytes = 0
    projected_output_tokens = 0
    output_token_cap = 0

    selected_shards = (
        list(shards) if shards is not None else list(manifest.get("shards", []))
    )
    for shard in selected_shards:
        path = run_dir / shard["input_file"]
        for line_number, request in enumerate(_iter_jsonl(path), start=1):
            body = request.get("body")
            if not isinstance(body, dict):
                raise ValueError(f"Missing request body in {path} at line {line_number}")
            max_output_tokens = body.get("max_output_tokens")
            if (
                isinstance(max_output_tokens, bool)
                or not isinstance(max_output_tokens, int)
                or max_output_tokens < 1
            ):
                raise ValueError(
                    f"Invalid max_output_tokens in {path} at line {line_number}"
                )

            request_count += 1
            input_text_bytes += _request_input_text_bytes(request)
            output_token_cap += max_output_tokens
            if _batch_request_uses_reasoning(body):
                reasoning_on_requests += 1
                projected_output_tokens += max_output_tokens
            else:
                projected_output_tokens += min(
                    _ESTIMATED_REASONING_OFF_OUTPUT_TOKENS_PER_REQUEST,
                    max_output_tokens,
                )

    expected_requests = (
        sum(int(shard.get("request_count", 0)) for shard in selected_shards)
        if shards is not None
        else int(manifest.get("total_requests", 0))
    )
    if request_count == 0:
        raise ValueError(f"Batch input is empty: {run_dir}")
    if request_count != expected_requests:
        raise ValueError(
            f"Generated Batch request count changed: {request_count:,} != "
            f"{expected_requests:,}"
        )

    estimated_input_tokens = (
        input_text_bytes + _ESTIMATED_UTF8_BYTES_PER_INPUT_TOKEN - 1
    ) // _ESTIMATED_UTF8_BYTES_PER_INPUT_TOKEN
    estimated_input_tokens += (
        request_count * _ESTIMATED_INPUT_FRAMING_TOKENS_PER_REQUEST
    )
    projected_cost = estimate_cost_from_totals(
        pricing,
        prompt_tokens=estimated_input_tokens,
        completion_tokens=projected_output_tokens,
    )
    maximum_output_cost = estimate_cost_from_totals(
        pricing,
        prompt_tokens=estimated_input_tokens,
        completion_tokens=output_token_cap,
    )

    return {
        "model_key": manifest["model_key"],
        "model_id": model_id,
        "request_count": request_count,
        "reasoning_on_requests": reasoning_on_requests,
        "reasoning_off_requests": request_count - reasoning_on_requests,
        "input_text_bytes": input_text_bytes,
        "estimated_input_tokens": estimated_input_tokens,
        "projected_output_tokens": projected_output_tokens,
        "output_token_cap": output_token_cap,
        "estimated_cost_usd": projected_cost,
        "maximum_output_cost_usd": maximum_output_cost,
        "input_rate_per_mtok": pricing["input"],
        "output_rate_per_mtok": pricing["output"],
        "pricing_source": pricing_source,
    }


def print_batch_run_pre_submit_cost_estimate(
    run_dir: Path,
    *,
    shards: Iterable[dict[str, Any]] | None = None,
    estimate_scope: str = "Pre-submit",
) -> dict[str, Any] | None:
    """Print and return the offline planning estimate for a Step-10b run."""
    selected_shards = list(shards) if shards is not None else None
    estimate = estimate_batch_run_pre_submit_cost(
        run_dir,
        shards=selected_shards,
    )
    if estimate is None:
        return None

    reasoning_on = estimate["reasoning_on_requests"]
    descriptor = "conservative planning estimate" if reasoning_on else "projected"
    print(
        f"[{estimate['model_key']}] {estimate_scope} OpenAI Batch cost estimate: "
        f"~${estimate['estimated_cost_usd']:,.6f} ({descriptor})."
    )
    print(
        f"  Basis: {estimate['request_count']:,} requests, "
        f"~{estimate['estimated_input_tokens']:,} input tokens "
        "(offline UTF-8 text-size heuristic), "
        f"~{estimate['projected_output_tokens']:,} output tokens."
    )
    if reasoning_on:
        print(
            f"  Reasoning-on assumption: {reasoning_on:,} requests reach their "
            "configured output-token caps."
        )
    else:
        print(
            "  Reasoning-off assumption: 5 output tokens/request; "
            f"all-cap scenario ~${estimate['maximum_output_cost_usd']:,.6f}."
        )
    print(
        f"  Batch rates: ${estimate['input_rate_per_mtok']:g} input / "
        f"${estimate['output_rate_per_mtok']:g} output per 1M tokens. "
        "Estimate only; after --batch-process, phase6b_cost_log.json and "
        "cost_summary.json use observed API token usage."
    )
    return estimate


def _load_jobs(run_dir: Path) -> dict[str, Any]:
    path = run_dir / BATCH_JOBS_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": "2.0", "created_at": _utc_now(), "jobs": []}


def _openai_client(client: Any | None = None) -> Any:
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI(api_key=require_api_key("openai"))


def _object_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dumped = getattr(value, "model_dump", None)
    if callable(dumped):
        result = dumped()
        return result if isinstance(result, dict) else {}
    return {}


def _find_remote_batch(api: Any, submission_key: str) -> Any | None:
    """Recover a just-created batch after a local crash before jobs.json persisted."""
    try:
        page = api.batches.list(limit=100)
    except (AttributeError, NotImplementedError):
        return None
    entries = _object_field(page, "data", page)
    try:
        iterator = iter(entries)
    except TypeError:
        return None
    for batch in iterator:
        metadata = _metadata_dict(_object_field(batch, "metadata", {}))
        if metadata.get("submission_key") == submission_key:
            return batch
    return None


def _record_remote_batch(job: dict[str, Any], batch: Any) -> None:
    job["batch_id"] = _object_field(batch, "id")
    job["status"] = _object_field(batch, "status", "validating")
    job["output_file_id"] = _object_field(batch, "output_file_id")
    job["error_file_id"] = _object_field(batch, "error_file_id")
    job["submitted_at"] = job.get("submitted_at") or _utc_now()


def submit_pending_batch_shards(
    run_dir: Path,
    *,
    max_queued_input_tokens: int,
    client: Any | None = None,
) -> dict[str, Any]:
    """Submit one queue-safe wave of pending shards.

    A durable placeholder is written before upload. A deterministic metadata key
    then lets a rerun recover a remotely created batch instead of charging for a
    duplicate submission if the process died before recording the batch ID.
    """
    if max_queued_input_tokens < 1:
        raise ValueError("max_queued_input_tokens must be at least 1.")
    manifest = validate_batch_inputs(run_dir)
    jobs = _load_jobs(run_dir)
    api = _openai_client(client)
    job_by_file = {entry["input_file"]: entry for entry in jobs.get("jobs", [])}

    active_tokens = sum(
        int(entry.get("input_token_upper_bound", 0))
        for entry in jobs.get("jobs", [])
        if entry.get("batch_id")
        and entry.get("status") not in _TERMINAL_BATCH_STATUSES
    )
    submitted_this_call = 0

    for shard in manifest.get("shards", []):
        input_name = shard["input_file"]
        token_bound = int(shard.get("input_token_upper_bound", 0))
        if token_bound < 1:
            raise ValueError(
                f"Shard {input_name} has no queue-token bound. Regenerate it with "
                "the current runner before submission."
            )
        if token_bound > max_queued_input_tokens:
            raise ValueError(
                f"Shard {input_name} has a conservative input-token bound of "
                f"{token_bound:,}, above the configured queue limit of "
                f"{max_queued_input_tokens:,}. Regenerate with a smaller "
                "--max-requests-per-batch."
            )

        job = job_by_file.get(input_name)
        if job and job.get("batch_id"):
            continue
        if active_tokens + token_bound > max_queued_input_tokens:
            continue

        submission_key = hashlib.sha256(
            f"{run_dir.name}:{input_name}".encode("utf-8")
        ).hexdigest()[:32]
        if job is None:
            job = {
                "input_file": input_name,
                "kind": shard.get("kind", "initial"),
                "attempt": int(shard.get("attempt", 0)),
                "request_count": int(shard["request_count"]),
                "byte_size": int(shard.get("byte_size", 0)),
                "input_token_upper_bound": token_bound,
                "submission_key": submission_key,
                "uploaded_file_id": None,
                "batch_id": None,
                "status": "pending_upload",
                "output_file_id": None,
                "error_file_id": None,
                "created_at": _utc_now(),
            }
            jobs.setdefault("jobs", []).append(job)
            job_by_file[input_name] = job
            jobs["updated_at"] = _utc_now()
            _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)

        recovered = _find_remote_batch(api, submission_key)
        if recovered is not None:
            _record_remote_batch(job, recovered)
            jobs["updated_at"] = _utc_now()
            _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)
            if job.get("status") not in _TERMINAL_BATCH_STATUSES:
                active_tokens += token_bound
            submitted_this_call += 1
            continue

        if not job.get("uploaded_file_id"):
            input_path = run_dir / input_name
            with input_path.open("rb") as handle:
                uploaded = api.files.create(file=handle, purpose="batch")
            job["uploaded_file_id"] = _object_field(uploaded, "id")
            job["status"] = "uploaded"
            jobs["updated_at"] = _utc_now()
            _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)

        batch = api.batches.create(
            input_file_id=job["uploaded_file_id"],
            endpoint=manifest.get("api_endpoint", OPENAI_BATCH_ENDPOINT),
            completion_window="24h",
            metadata={
                "description": f"llm_coherence step10b {manifest['model_key']}",
                "run_id": run_dir.name,
                "input_file": input_name,
                "submission_key": submission_key,
            },
        )
        _record_remote_batch(job, batch)
        jobs["updated_at"] = _utc_now()
        _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)
        active_tokens += token_bound
        submitted_this_call += 1

    submitted_files = {
        entry["input_file"]
        for entry in jobs.get("jobs", [])
        if entry.get("batch_id")
    }
    pending = [
        shard["input_file"]
        for shard in manifest.get("shards", [])
        if shard["input_file"] not in submitted_files
    ]
    jobs["submitted_this_call"] = submitted_this_call
    jobs["pending_shards"] = len(pending)
    jobs["pending_input_files"] = pending
    jobs["active_input_token_upper_bound"] = active_tokens
    jobs["max_queued_input_tokens"] = max_queued_input_tokens
    return jobs


def _content_text(content: Any) -> str:
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    raw = getattr(content, "content", None)
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, str):
        return raw
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    read = getattr(content, "read", None)
    if callable(read):
        value = read()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    raise TypeError(f"Unsupported OpenAI file-content response: {type(content)!r}")


def refresh_batch_jobs(run_dir: Path, *, client: Any | None = None) -> dict[str, Any]:
    """Refresh statuses and download any newly available output/error files."""
    jobs = _load_jobs(run_dir)
    if not jobs.get("jobs"):
        raise ValueError(f"No submitted batch jobs recorded in {run_dir}.")
    api = _openai_client(client)

    for index, entry in enumerate(jobs["jobs"]):
        if not entry.get("batch_id"):
            raise ValueError(
                f"Submission for {entry.get('input_file')} stopped at "
                f"{entry.get('status')!r}. Rerun --batch-submit with the same "
                "run directory to recover it before requesting status."
            )
        batch = api.batches.retrieve(entry["batch_id"])
        entry["status"] = _object_field(batch, "status")
        entry["output_file_id"] = _object_field(batch, "output_file_id")
        entry["error_file_id"] = _object_field(batch, "error_file_id")
        counts = _object_field(batch, "request_counts")
        entry["request_counts"] = (
            {
                "total": _object_field(counts, "total", 0),
                "completed": _object_field(counts, "completed", 0),
                "failed": _object_field(counts, "failed", 0),
            }
            if counts is not None
            else None
        )
        batch_errors = _object_field(batch, "errors")
        entry["batch_errors"] = _metadata_dict(batch_errors) if batch_errors else None
        if entry["output_file_id"]:
            output_name = entry.get("output_file") or f"batch_output_{index:03d}.jsonl"
            output_path = run_dir / output_name
            if not output_path.is_file():
                payload = _content_text(api.files.content(entry["output_file_id"]))
                _atomic_write_text(output_path, payload.rstrip("\n") + "\n")
            entry["output_file"] = output_name
        if entry["error_file_id"]:
            error_name = entry.get("error_file") or f"batch_errors_{index:03d}.jsonl"
            error_path = run_dir / error_name
            if not error_path.is_file():
                payload = _content_text(api.files.content(entry["error_file_id"]))
                _atomic_write_text(error_path, payload.rstrip("\n") + "\n")
            entry["error_file"] = error_name

    statuses = [entry.get("status") for entry in jobs["jobs"]]
    jobs["all_terminal"] = all(status in _TERMINAL_BATCH_STATUSES for status in statuses)
    jobs["all_completed"] = all(status == "completed" for status in statuses)
    jobs["updated_at"] = _utc_now()
    _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)
    return jobs


def wait_for_batch_jobs(
    run_dir: Path,
    *,
    poll_interval: int = 30,
    client: Any | None = None,
) -> dict[str, Any]:
    if poll_interval < 1:
        raise ValueError("poll_interval must be at least 1 second.")
    api = _openai_client(client)
    while True:
        jobs = refresh_batch_jobs(run_dir, client=api)
        status_counts: dict[str, int] = defaultdict(int)
        for entry in jobs["jobs"]:
            status_counts[str(entry.get("status"))] += 1
        print("Batch statuses:", ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
        if jobs["all_terminal"]:
            return jobs
        time.sleep(poll_interval)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def _is_retryable_failure(status_code: Any, error: Any) -> bool:
    try:
        status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status = None
    if status is not None:
        if status in _RETRYABLE_HTTP_STATUSES or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    details = error if isinstance(error, dict) else {}
    code = str(details.get("code") or "").lower()
    if code in _RETRYABLE_ERROR_CODES:
        return True
    if code.startswith("invalid_") or code in {
        "bad_request",
        "invalid_request_error",
        "model_not_found",
        "unsupported_parameter",
    }:
        return False
    # A missing row or an unclassified provider/transport failure is safe to
    # retry; deterministic HTTP 4xx failures above are explicitly excluded.
    return True


def _response_inventory(
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    jobs = _load_jobs(run_dir)
    responses: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    stats = {
        "rows": 0,
        "successful": 0,
        "http_errors": 0,
        "malformed": 0,
        "duplicates": 0,
        "retryable_failures": 0,
        "non_retryable_failures": 0,
    }
    for entry in jobs.get("jobs", []):
        names = [entry.get("output_file"), entry.get("error_file")]
        for name in dict.fromkeys(item for item in names if item):
            for row in _iter_jsonl(run_dir / name):
                stats["rows"] += 1
                custom_id = row.get("custom_id")
                response = row.get("response") or {}
                status_code = response.get("status_code")
                body = response.get("body")
                if not custom_id:
                    stats["malformed"] += 1
                    continue
                try:
                    decode_custom_id(custom_id)
                except ValueError:
                    stats["malformed"] += 1
                    continue
                if status_code == 200 and isinstance(body, dict):
                    if custom_id in responses:
                        stats["duplicates"] += 1
                    responses[custom_id] = row
                    failures.pop(custom_id, None)
                    stats["successful"] += 1
                    continue
                stats["http_errors"] += 1
                error = row.get("error") or (
                    body.get("error") if isinstance(body, dict) else None
                )
                retryable = _is_retryable_failure(status_code, error)
                failures[custom_id] = {
                    "status_code": status_code,
                    "error": error,
                    "retryable": retryable,
                    "batch_id": entry.get("batch_id"),
                }
    for custom_id in responses:
        failures.pop(custom_id, None)
    stats["unique_successful"] = len(responses)
    stats["retryable_failures"] = sum(
        1 for item in failures.values() if item["retryable"]
    )
    stats["non_retryable_failures"] = sum(
        1 for item in failures.values() if not item["retryable"]
    )
    return responses, failures, stats


def _successful_response_map(
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    responses, _, stats = _response_inventory(run_dir)
    return responses, stats


def _initial_request_ids(run_dir: Path, manifest: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for shard in manifest.get("shards", []):
        if shard.get("kind", "initial") != "initial":
            continue
        for row in _iter_jsonl(run_dir / shard["input_file"]):
            custom_id = row.get("custom_id")
            if custom_id:
                if custom_id in ids:
                    raise ValueError(f"Duplicate initial Batch custom_id: {custom_id}")
                ids.add(custom_id)
    return ids


def create_retry_shards(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Retry missing/transient requests without changing model conditions.

    OpenAI can return ``status=incomplete`` with HTTP 200 when a response
    reaches ``max_output_tokens``. Those rows remain incomplete analytical
    observations. They are not selectively rerun at a larger token ceiling,
    because doing so would give only the hardest items a different condition.
    """
    manifest = validate_batch_inputs(run_dir)
    jobs = _load_jobs(run_dir)
    if any(
        entry.get("status") not in _TERMINAL_BATCH_STATUSES
        for entry in jobs.get("jobs", [])
    ):
        raise ValueError("Cannot create retry shards while a submitted job is nonterminal.")
    successful, failures, _ = _response_inventory(run_dir)
    expected = _initial_request_ids(run_dir, manifest)
    missing = expected.difference(successful)
    retryable_missing = {
        custom_id
        for custom_id in missing
        if custom_id not in failures or failures[custom_id]["retryable"]
    }
    token_capped_incomplete = {
        custom_id
        for custom_id, row in successful.items()
        if (row.get("response") or {}).get("body", {}).get("status") == "incomplete"
        and (
            ((row.get("response") or {}).get("body", {}).get("incomplete_details") or {})
            .get("reason")
            == "max_output_tokens"
        )
    }
    retryable = retryable_missing
    non_retryable = missing.difference(retryable_missing)
    classification = {
        "missing_total": len(missing),
        "incomplete_total": sum(
            1
            for row in successful.values()
            if (row.get("response") or {}).get("body", {}).get("status")
            == "incomplete"
        ),
        "retryable_missing": len(retryable_missing),
        "token_capped_incomplete": len(token_capped_incomplete),
        "retryable_incomplete": 0,
        "retryable": len(retryable),
        "non_retryable": len(non_retryable),
        "non_retryable_examples": [
            {"custom_id": custom_id, **failures[custom_id]}
            for custom_id in sorted(non_retryable)[:10]
        ],
        "incomplete_retry_token_caps": [],
    }
    if not missing:
        return [], 0, classification
    if not retryable:
        return [], 0, classification

    attempt = 1 + max((int(s.get("attempt", 0)) for s in manifest["shards"]), default=0)

    def retry_requests() -> Iterator[dict[str, Any]]:
        latest_requests: dict[str, dict[str, Any]] = {}
        ordered_shards = sorted(
            enumerate(manifest["shards"]),
            key=lambda item: (int(item[1].get("attempt", 0)), item[0]),
        )
        for _, shard in ordered_shards:
            for row in _iter_jsonl(run_dir / shard["input_file"]):
                custom_id = row.get("custom_id")
                if custom_id in retryable:
                    latest_requests[custom_id] = row

        unavailable = retryable.difference(latest_requests)
        if unavailable:
            raise RuntimeError(
                f"Could not recover {len(unavailable)} retry requests from "
                "generated Batch shards."
            )

        for custom_id in sorted(retryable):
            yield latest_requests[custom_id]

    shards, retry_count = _write_sharded_requests(
        retry_requests(),
        run_dir=run_dir,
        prefix=f"batch_retry_a{attempt}",
        kind="retry",
        attempt=attempt,
        max_requests_per_batch=int(manifest["max_requests_per_batch"]),
    )
    manifest["shards"].extend(shards)
    manifest.setdefault("retry_history", []).append(
        {
            "attempt": attempt,
            "created_at": _utc_now(),
            "request_count": retry_count,
            "classification": classification,
        }
    )
    _atomic_write_json(run_dir / BATCH_MANIFEST_NAME, manifest)
    return shards, retry_count, classification


def _message_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        ]
        joined = "".join(texts)
        return joined or None
    return None


def _reasoning_summaries(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract native OpenAI reasoning-summary blocks without joining them.

    The Responses API returns summaries in a separate ``type=reasoning``
    output item.  Keeping the item ID and block index makes each stored trace
    auditable against the unmodified Batch output JSONL.
    """
    summaries: list[dict[str, Any]] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        reasoning_item_id = item.get("id")
        for summary_index, block in enumerate(item.get("summary") or []):
            if (
                not isinstance(block, dict)
                or block.get("type") != "summary_text"
                or not isinstance(block.get("text"), str)
                or not block["text"].strip()
            ):
                continue
            summaries.append(
                {
                    "reasoning_item_id": reasoning_item_id,
                    "summary_index": summary_index,
                    "type": "summary_text",
                    "text": block["text"],
                }
            )
    return summaries


def _response_text_and_diagnostics(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize Responses API output, with Chat Completions v1 compatibility."""
    texts: list[str] = []
    refusals: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                refusals.append(block["refusal"])

    response_status = body.get("status")
    incomplete = body.get("incomplete_details") or {}
    incomplete_reason = (
        incomplete.get("reason") if isinstance(incomplete, dict) else None
    )
    finish_reason = incomplete_reason
    if response_status == "completed" and finish_reason is None:
        finish_reason = "stop"

    # Schema-v1 runs used Chat Completions. Retaining the parser makes old,
    # already-paid runs processable while all newly generated runs use Responses.
    if not texts and body.get("choices"):
        choice = body["choices"][0] or {}
        message = choice.get("message") or {}
        text = _message_text(message)
        if text:
            texts.append(text)
        finish_reason = choice.get("finish_reason")
        response_status = "completed" if finish_reason == "stop" else response_status

    return {
        "text": "".join(texts) or None,
        "reasoning_summaries": _reasoning_summaries(body),
        "refusals": refusals,
        "response_status": response_status or "unknown",
        "finish_reason": finish_reason or "unknown",
        "incomplete_reason": incomplete_reason,
    }


def process_batch_run(run_dir: Path) -> dict[str, Any]:
    """Rebuild per-ladder ``results.json`` files from downloaded batch output."""
    manifest = validate_batch_inputs(run_dir)
    jobs = _load_jobs(run_dir)
    if not jobs.get("jobs"):
        raise ValueError(
            f"No submitted batch jobs recorded in {run_dir}; there is nothing to process."
        )
    manifest_shards = {item["input_file"] for item in manifest.get("shards", [])}
    submitted_shards = {
        item["input_file"] for item in jobs.get("jobs", []) if item.get("batch_id")
    }
    unsubmitted_shards = sorted(manifest_shards.difference(submitted_shards))
    if unsubmitted_shards:
        raise ValueError(
            f"Cannot process: {len(unsubmitted_shards)} manifest shard(s) were never "
            f"submitted, including {unsubmitted_shards[:3]}."
        )
    nonterminal = [
        entry
        for entry in jobs["jobs"]
        if entry.get("status") not in _TERMINAL_BATCH_STATUSES
    ]
    if nonterminal:
        statuses = ", ".join(
            f"{entry.get('batch_id', '<unknown>')}={entry.get('status', '<unknown>')}"
            for entry in nonterminal
        )
        raise ValueError(
            "Cannot process a batch run before every submitted job is terminal: "
            f"{statuses}. Run --batch-status first."
        )
    responses, failures, response_stats = _response_inventory(run_dir)
    expected_ids = _initial_request_ids(run_dir, manifest)
    if len(expected_ids) != int(manifest.get("total_requests", -1)):
        raise ValueError(
            f"Manifest expected {manifest.get('total_requests')} initial requests, "
            f"but its JSONL files contain {len(expected_ids)} unique IDs."
        )
    missing_ids = expected_ids.difference(responses)
    unexpected_ids = set(responses).difference(expected_ids)
    if missing_ids or unexpected_ids:
        deterministic = [
            custom_id
            for custom_id in sorted(missing_ids)
            if custom_id in failures and not failures[custom_id]["retryable"]
        ]
        raise ValueError(
            "Batch response coverage is incomplete; no result files were written. "
            f"missing={len(missing_ids)}, unexpected={len(unexpected_ids)}, "
            f"non_retryable_missing={len(deterministic)}. Run --batch-retry for "
            "transient failures and inspect batch_errors_*.jsonl for deterministic "
            f"failures. Examples: {sorted(missing_ids)[:3]}"
        )
    model_id = manifest["model_id"]
    num_trials = int(manifest["num_trials"])
    include_flipped = bool(manifest["include_flipped"])
    with_reasoning = bool(manifest["with_reasoning"])
    results_dir = _manifest_results_dir(run_dir, manifest)
    configured_reasoning = (manifest.get("extra_body") or {}).get("reasoning")
    configured_reasoning = (
        configured_reasoning if isinstance(configured_reasoning, dict) else {}
    )
    reasoning_summary_mode = configured_reasoning.get("summary")
    reasoning_summary_requested = reasoning_summary_mode is not None

    slots: dict[tuple[int, int, str], list[str | None]] = {}
    reasoning_traces_by_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    reasoning_summary_block_count_by_set: dict[int, int] = defaultdict(int)
    missing_reasoning_summary_ids: list[str] = []
    missing_reasoning_summary_ids_by_set: dict[int, list[str]] = defaultdict(list)
    usage_by_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cost_by_set: dict[int, float] = defaultdict(float)
    response_count_by_set: dict[int, int] = defaultdict(int)
    finish_reason_by_set: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    response_status_by_set: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    refusals_by_set: dict[int, int] = defaultdict(int)
    incomplete_by_set: dict[int, int] = defaultdict(int)
    for custom_id, row in responses.items():
        set_index, comparison_index, direction, trial_index = decode_custom_id(custom_id)
        key = (set_index, comparison_index, direction)
        values = slots.setdefault(key, [None] * num_trials)
        if not 0 <= trial_index < num_trials:
            raise ValueError(f"Trial index out of range in {custom_id!r}.")
        body = row["response"]["body"]
        diagnostics = _response_text_and_diagnostics(body)
        values[trial_index] = (
            diagnostics["text"]
            if diagnostics["response_status"] == "completed"
            and not diagnostics["incomplete_reason"]
            else None
        )
        if reasoning_summary_requested:
            summaries = diagnostics["reasoning_summaries"]
            if not summaries:
                missing_reasoning_summary_ids.append(custom_id)
                missing_reasoning_summary_ids_by_set[set_index].append(custom_id)
            else:
                reasoning_summary_block_count_by_set[set_index] += len(summaries)
                reasoning_traces_by_set[set_index].append(
                    {
                        "custom_id": custom_id,
                        "pair_idx": comparison_index,
                        "direction": "A" if direction == "ab" else "B",
                        "trial": trial_index,
                        "content": diagnostics["text"],
                        "summary": "\n\n".join(
                            summary["text"] for summary in summaries
                        ),
                    }
                )
        response_count_by_set[set_index] += 1
        finish_reason_by_set[set_index][diagnostics["finish_reason"]] += 1
        response_status_by_set[set_index][diagnostics["response_status"]] += 1
        refusals_by_set[set_index] += len(diagnostics["refusals"])
        if diagnostics["incomplete_reason"]:
            incomplete_by_set[set_index] += 1
        usage = usage_cost_breakdown(
            body.get("usage") or {},
            provider="openai",
            model_id=body.get("model") or model_id,
            batch=True,
        )
        usage_by_set[set_index].append(usage)
        if isinstance(usage.get("cost"), (int, float)):
            cost_by_set[set_index] += float(usage["cost"])

    total_unparseable = 0
    result_payloads: list[
        tuple[Path, dict[str, Any], Path | None, list[dict[str, Any]]]
    ] = []
    zero_parseable_pairs: list[str] = []
    for set_index, set_record in enumerate(manifest["sets"]):
        comparison_path = _resolve_run_relative_path(
            run_dir, set_record["comparison_path"]
        )
        comparisons = load_comparisons(
            _manifest_data_dir(run_dir, manifest),
            set_record["test_name"],
            comparison_path,
        )
        preferences: list[dict[str, Any]] = []
        set_unparseable = 0
        set_zero_parseable_comparison_indices: list[int] = []
        for comparison_index, comparison in enumerate(comparisons):
            original = slots.get(
                (set_index, comparison_index, "ab"),
                [None] * num_trials,
            )
            flipped = (
                slots.get(
                    (set_index, comparison_index, "ba"),
                    [None] * num_trials,
                )
                if include_flipped
                else []
            )
            parsed_original = parse_responses_forced_choice(
                {0: original},
                with_reasoning=with_reasoning,
                verbose=False,
            )[0]
            parsed_flipped = parse_responses_forced_choice(
                {0: flipped},
                with_reasoning=with_reasoning,
                verbose=False,
            )[0]
            count_a, count_b = counts_from_responses(parsed_original, parsed_flipped)
            expected = num_trials * (2 if include_flipped else 1)
            parsed_total = count_a + count_b
            set_unparseable += expected - parsed_total
            if parsed_total == 0:
                set_zero_parseable_comparison_indices.append(comparison_index)
                zero_parseable_pairs.append(
                    f"{set_record['test_name']} comparison {comparison_index}"
                )
            pref = {
                "outcome_a": comparison["outcome_a"],
                "outcome_b": comparison["outcome_b"],
                "count_prefer_a": count_a,
                "count_prefer_b": count_b,
                "prob_prefer_a": round(count_a / parsed_total, 4) if parsed_total else None,
                "prob_prefer_b": round(count_b / parsed_total, 4) if parsed_total else None,
                "parseable_trials": parsed_total,
                "expected_trials": expected,
            }
            if with_reasoning:
                pref["raw_responses_original"] = original
                pref["raw_responses_flipped"] = flipped
            preferences.append(pref)

        total_unparseable += set_unparseable
        expected_calls = len(comparisons) * num_trials * (2 if include_flipped else 1)
        usage_summary = _summarize_usage(usage_by_set.get(set_index, []))
        computed_cost = round(cost_by_set.get(set_index, 0.0), 6)
        reasoning_traces = sorted(
            reasoning_traces_by_set.get(set_index, []),
            key=lambda record: str(record["custom_id"]),
        )
        reasoning_summary_blocks = reasoning_summary_block_count_by_set.get(
            set_index, 0
        )
        missing_set_summary_ids = sorted(
            missing_reasoning_summary_ids_by_set.get(set_index, [])
        )
        set_response_count = response_count_by_set.get(set_index, 0)
        summary_coverage_rate = (
            len(reasoning_traces) / set_response_count
            if set_response_count
            else None
        )
        end_time = _utc_now()
        payload = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "config": {
                "test_name": set_record["test_name"],
                "model_key": manifest["model_key"],
                "model_variant": "instruct",
                "is_base_model": False,
                "reasoning_mode": manifest.get("reasoning_effort") or "none",
                "num_trials": num_trials,
                "include_flipped": include_flipped,
                "with_reasoning": with_reasoning,
                "max_tokens": int(manifest["max_tokens"]),
                "temperature": manifest.get("temperature"),
                "k_samples": 1,
                "infrastructure": "openai_batch_api",
                "gpu_type": None,
                "gpu_count": None,
                "quantization": None,
            },
            "metadata": {
                "start_time": manifest["created_at"],
                "end_time": end_time,
                "total_comparisons": len(comparisons),
                "total_api_calls": expected_calls,
                "successful_api_responses": response_count_by_set.get(set_index, 0),
                "unparseable_count": set_unparseable,
                "unparseable_rate": set_unparseable / expected_calls if expected_calls else 0.0,
                "elapsed_seconds": None,
                "usage_stats": usage_summary,
                "model_name_full": model_id,
                "extra_body": manifest.get("extra_body") or None,
                "reasoning_artifacts": {
                    "artifact_type": manifest.get("reasoning_artifact_type")
                    or ("summary" if reasoning_summary_requested else "none"),
                    "summary_requested": reasoning_summary_requested,
                    "summary_mode_requested": reasoning_summary_mode,
                    "responses_with_summary": len(reasoning_traces),
                    "responses_without_summary": len(missing_set_summary_ids),
                    "summary_coverage_rate": summary_coverage_rate,
                    "missing_summary_examples": missing_set_summary_ids[:10],
                    "summary_block_count": reasoning_summary_blocks,
                    "sidecar": (
                        REASONING_SUMMARIES_NAME
                        if reasoning_summary_requested
                        else None
                    ),
                    "raw_batch_output_retained": True,
                },
                "batch_job_count": len(jobs.get("jobs", [])),
                "batch_run_id": run_dir.name,
                "run_status": "complete" if set_unparseable == 0 else "incomplete",
                "response_diagnostics": {
                    "finish_reason_counts": dict(finish_reason_by_set.get(set_index, {})),
                    "response_status_counts": dict(response_status_by_set.get(set_index, {})),
                    "refusal_count": refusals_by_set.get(set_index, 0),
                    "incomplete_count": incomplete_by_set.get(set_index, 0),
                    "zero_parseable_comparison_count": len(
                        set_zero_parseable_comparison_indices
                    ),
                    "zero_parseable_comparison_indices": (
                        set_zero_parseable_comparison_indices
                    ),
                },
                "estimated_cost_usd": computed_cost,
                "actual_cost_usd": None,
                "cost_source": "computed_from_usage_and_published_batch_rates",
                "git_commit_sha": _git_sha(),
                "package_versions": _package_versions(),
                "prompt_template_used": manifest["prompt_template_used"],
                "system_message": manifest.get("system_message"),
                "comparison_file_sha256": set_record.get("comparison_file_sha256"),
                "retry_counts": {
                    "batch_transport_retries": sum(
                        int(item.get("request_count", 0))
                        for item in manifest.get("retry_history", [])
                    )
                },
            },
            "preferences": preferences,
        }
        result_path = (
            results_dir
            / artifact_dir_name_for_test(set_record["test_name"])
            / "results.json"
        )
        traces_path = (
            result_path.parent / REASONING_SUMMARIES_NAME
            if reasoning_summary_requested
            else None
        )
        result_payloads.append((result_path, payload, traces_path, reasoning_traces))

    written_files: list[str] = []
    written_trace_files: list[str] = []
    for result_path, payload, traces_path, reasoning_traces in result_payloads:
        if traces_path is not None:
            trace_text = "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in reasoning_traces
            )
            _atomic_write_text(traces_path, trace_text)
            written_trace_files.append(
                Path(os.path.relpath(traces_path, results_dir)).as_posix()
            )
        save_results(result_path, payload)
        written_files.append(
            Path(os.path.relpath(result_path, results_dir)).as_posix()
        )

    job_status_counts: dict[str, int] = defaultdict(int)
    for entry in jobs.get("jobs", []):
        job_status_counts[str(entry.get("status"))] += 1
    summary = {
        "schema_version": "2.0",
        "processed_at": _utc_now(),
        "model_key": manifest["model_key"],
        "model_id": model_id,
        "batch_run_id": run_dir.name,
        "expected_requests": len(expected_ids),
        "successful_responses": len(responses),
        "missing_responses": len(missing_ids),
        "unparseable_or_missing_responses": total_unparseable,
        "processing_status": "complete" if total_unparseable == 0 else "incomplete",
        "zero_parseable_comparison_count": len(zero_parseable_pairs),
        "zero_parseable_comparison_examples": zero_parseable_pairs[:10],
        "response_file_stats": response_stats,
        "batch_status_counts": dict(job_status_counts),
        "reasoning_summary_requested": reasoning_summary_requested,
        "responses_with_reasoning_summary": sum(
            len(records) for records in reasoning_traces_by_set.values()
        ),
        "responses_without_reasoning_summary": len(missing_reasoning_summary_ids),
        "reasoning_summary_coverage_rate": (
            sum(len(records) for records in reasoning_traces_by_set.values())
            / len(responses)
            if reasoning_summary_requested and responses
            else None
        ),
        "missing_reasoning_summary_examples": sorted(
            missing_reasoning_summary_ids
        )[:10],
        "reasoning_summary_files_written": len(written_trace_files),
        "reasoning_summary_files": written_trace_files,
        "result_files_written": len(written_files),
        "result_files": written_files,
    }
    _atomic_write_json(run_dir / BATCH_PROCESSING_SUMMARY_NAME, summary)
    return summary
