"""OpenAI Batch API support for the ladder-vs-comparison experiment.

The historical GPT-5.4 subject-model runs used the OpenAI Batch API, while the
current step-10b runner otherwise issues live requests through LiteLLM.  This
module restores a reproducible batch path that:

* writes the official per-request JSONL format;
* shards at the API's 50,000-request/file limit;
* records enough run metadata to rebuild the existing per-ladder result schema;
* downloads successful and error files without assuming output order; and
* can create retry shards for transport-level failures or missing responses.

Native reasoning is controlled by ``MODEL_CONFIGS.extra_body``.  In particular,
GPT-5.6's off condition must include ``reasoning_effort="none"`` explicitly;
omitting the field would use the model's non-off default.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.experiments.ladder_statement_pair.experiment_runner_tradeoff import (
    RESULTS_SCHEMA_VERSION,
    _file_sha256,
    _git_sha,
    _host_info,
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
from llm_coherence.runtime.usage_cost import usage_cost_breakdown
from llm_coherence.runtime.utils import parse_responses_forced_choice


MAX_BATCH_REQUESTS = 50_000
BATCH_RUNS_DIRNAME = "batch_runs"
LATEST_BATCH_RUN_NAME = "latest_batch_run.txt"
BATCH_MANIFEST_NAME = "batch_manifest.json"
BATCH_JOBS_NAME = "batch_jobs.json"
BATCH_PROCESSING_SUMMARY_NAME = "batch_processing_summary.json"

_CUSTOM_ID_RE = re.compile(
    r"^s(?P<set>\d+)-c(?P<comparison>\d+)-d(?P<direction>ab|ba)-t(?P<trial>\d+)$"
)
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


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
    spec = MODEL_SPECS.get(model_key)
    if spec is None or spec.model_type != "openai":
        raise ValueError(
            f"OpenAI Batch API requires an OpenAI model key; got {model_key!r}."
        )
    model_id = model_name_for_key(model_key)
    if not model_id:
        raise ValueError(f"No provider model ID configured for {model_key!r}.")
    return model_id.removeprefix("openai/")


def _reasoning_effort(model_key: str) -> str | None:
    cfg = MODEL_CONFIGS.get(model_key)
    if cfg is None:
        raise ValueError(f"No MODEL_CONFIGS entry for {model_key!r}.")
    effort = (cfg.extra_body or {}).get("reasoning_effort")
    return str(effort) if effort is not None else None


def build_openai_chat_batch_body(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict[str, Any] | None,
    system_message: str | None,
) -> dict[str, Any]:
    """Build one Chat Completions request body for an OpenAI batch line."""
    messages: list[dict[str, str]] = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    extra = dict(extra_body or {})
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    effort = extra.get("reasoning_effort")
    # Do not collapse explicit "none" into omission: GPT-5.6 otherwise uses a
    # reasoning-enabled default.  Temperature is valid for the off condition;
    # reasoning-on requests intentionally omit it, matching the GPT-5.4 path.
    if effort is not None:
        body["reasoning_effort"] = effort
    if effort in (None, "none"):
        body["temperature"] = temperature
    for key, value in extra.items():
        if key in {"reasoning_effort", "temperature"}:
            continue
        body[key] = value
    return body


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
    cfg = MODEL_CONFIGS[model_key]
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
                body = build_openai_chat_batch_body(
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
                        "url": "/v1/chat/completions",
                        "body": body,
                    }


def _write_sharded_requests(
    requests: Iterable[dict[str, Any]],
    *,
    run_dir: Path,
    prefix: str,
    kind: str,
    attempt: int,
    max_requests_per_batch: int,
) -> tuple[list[dict[str, Any]], int]:
    if not 1 <= max_requests_per_batch <= MAX_BATCH_REQUESTS:
        raise ValueError(
            f"max_requests_per_batch must be between 1 and {MAX_BATCH_REQUESTS:,}."
        )

    shards: list[dict[str, Any]] = []
    handle = None
    shard_path: Path | None = None
    shard_count = 0
    total = 0
    shard_index = 0

    def close_shard() -> None:
        nonlocal handle, shard_path, shard_count
        if handle is None or shard_path is None:
            return
        handle.close()
        shards.append(
            {
                "input_file": shard_path.name,
                "request_count": shard_count,
                "kind": kind,
                "attempt": attempt,
            }
        )
        handle = None
        shard_path = None
        shard_count = 0

    try:
        for request in requests:
            if handle is None or shard_count >= max_requests_per_batch:
                close_shard()
                shard_path = run_dir / f"{prefix}_{shard_index:03d}.jsonl"
                handle = shard_path.open("w", encoding="utf-8")
                shard_index += 1
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            shard_count += 1
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
    cfg = MODEL_CONFIGS.get(model_key)
    if cfg is None:
        raise ValueError(f"No MODEL_CONFIGS entry for {model_key!r}.")
    resolved_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
    resolved_temperature = temperature if temperature is not None else cfg.temperature
    model_id = _model_id(model_key)

    batch_root = results_dir / BATCH_RUNS_DIRNAME
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = batch_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    set_records: list[dict[str, Any]] = []
    for item in run_items:
        comparison_path = Path(item["comparison_path"]).resolve()
        comparisons = load_comparisons(data_dir, item["test_name"], comparison_path)
        set_records.append(
            {
                "test_name": item["test_name"],
                "comparison_path": str(comparison_path),
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
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "model_key": model_key,
        "model_id": model_id,
        "reasoning_effort": _reasoning_effort(model_key),
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
        "temperature": resolved_temperature,
        "system_message": system_message,
        "infrastructure": "openai_batch_api",
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "data_dir": str(data_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "total_requests": total_requests,
        "max_requests_per_batch": max_requests_per_batch,
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


def _load_jobs(run_dir: Path) -> dict[str, Any]:
    path = run_dir / BATCH_JOBS_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": "1.0", "created_at": _utc_now(), "jobs": []}


def _openai_client(client: Any | None = None) -> Any:
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI(api_key=require_api_key("openai"))


def submit_pending_batch_shards(run_dir: Path, *, client: Any | None = None) -> dict[str, Any]:
    """Upload and submit every manifest shard not already recorded in jobs.json."""
    manifest = load_batch_manifest(run_dir)
    jobs = _load_jobs(run_dir)
    submitted_files = {entry["input_file"] for entry in jobs.get("jobs", [])}
    api = _openai_client(client)

    for shard in manifest.get("shards", []):
        input_name = shard["input_file"]
        if input_name in submitted_files:
            continue
        input_path = run_dir / input_name
        with input_path.open("rb") as handle:
            uploaded = api.files.create(file=handle, purpose="batch")
        batch = api.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"llm_coherence step10b {manifest['model_key']}",
                "run_id": run_dir.name,
                "input_file": input_name,
            },
        )
        jobs.setdefault("jobs", []).append(
            {
                "input_file": input_name,
                "kind": shard.get("kind", "initial"),
                "attempt": int(shard.get("attempt", 0)),
                "request_count": int(shard["request_count"]),
                "uploaded_file_id": uploaded.id,
                "batch_id": batch.id,
                "status": getattr(batch, "status", "validating"),
                "submitted_at": _utc_now(),
                "output_file_id": None,
                "error_file_id": None,
            }
        )
        submitted_files.add(input_name)
        jobs["updated_at"] = _utc_now()
        _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)
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
        batch = api.batches.retrieve(entry["batch_id"])
        entry["status"] = batch.status
        entry["output_file_id"] = getattr(batch, "output_file_id", None)
        entry["error_file_id"] = getattr(batch, "error_file_id", None)
        counts = getattr(batch, "request_counts", None)
        entry["request_counts"] = (
            {
                "total": getattr(counts, "total", 0),
                "completed": getattr(counts, "completed", 0),
                "failed": getattr(counts, "failed", 0),
            }
            if counts is not None
            else None
        )
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

    jobs["updated_at"] = _utc_now()
    _atomic_write_json(run_dir / BATCH_JOBS_NAME, jobs)
    statuses = [entry.get("status") for entry in jobs["jobs"]]
    jobs["all_terminal"] = all(status in _TERMINAL_BATCH_STATUSES for status in statuses)
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


def _successful_response_map(
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    jobs = _load_jobs(run_dir)
    responses: dict[str, dict[str, Any]] = {}
    stats = {"rows": 0, "successful": 0, "http_errors": 0, "malformed": 0}
    for entry in jobs.get("jobs", []):
        output_name = entry.get("output_file")
        if not output_name:
            continue
        for row in _iter_jsonl(run_dir / output_name):
            stats["rows"] += 1
            custom_id = row.get("custom_id")
            status_code = (row.get("response") or {}).get("status_code")
            body = (row.get("response") or {}).get("body")
            if not custom_id or status_code != 200 or not isinstance(body, dict):
                stats["http_errors"] += 1
                continue
            try:
                decode_custom_id(custom_id)
            except ValueError:
                stats["malformed"] += 1
                continue
            responses[custom_id] = row
            stats["successful"] += 1
    stats["unique_successful"] = len(responses)
    return responses, stats


def _initial_request_ids(run_dir: Path, manifest: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for shard in manifest.get("shards", []):
        if shard.get("kind", "initial") != "initial":
            continue
        for row in _iter_jsonl(run_dir / shard["input_file"]):
            custom_id = row.get("custom_id")
            if custom_id:
                ids.add(custom_id)
    return ids


def create_retry_shards(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Create new input shards for expected IDs lacking a successful response."""
    manifest = load_batch_manifest(run_dir)
    successful, _ = _successful_response_map(run_dir)
    expected = _initial_request_ids(run_dir, manifest)
    missing = expected.difference(successful)
    if not missing:
        return [], 0

    attempt = 1 + max((int(s.get("attempt", 0)) for s in manifest["shards"]), default=0)

    def missing_requests() -> Iterator[dict[str, Any]]:
        remaining = set(missing)
        for shard in manifest["shards"]:
            if shard.get("kind", "initial") != "initial":
                continue
            for row in _iter_jsonl(run_dir / shard["input_file"]):
                custom_id = row.get("custom_id")
                if custom_id in remaining:
                    remaining.remove(custom_id)
                    yield row
        if remaining:
            raise RuntimeError(
                f"Could not recover {len(remaining)} missing requests from initial shards."
            )

    shards, retry_count = _write_sharded_requests(
        missing_requests(),
        run_dir=run_dir,
        prefix=f"batch_retry_a{attempt}",
        kind="retry",
        attempt=attempt,
        max_requests_per_batch=int(manifest["max_requests_per_batch"]),
    )
    manifest["shards"].extend(shards)
    manifest.setdefault("retry_history", []).append(
        {"attempt": attempt, "created_at": _utc_now(), "request_count": retry_count}
    )
    _atomic_write_json(run_dir / BATCH_MANIFEST_NAME, manifest)
    return shards, retry_count


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


def process_batch_run(run_dir: Path) -> dict[str, Any]:
    """Rebuild per-ladder ``results.json`` files from downloaded batch output."""
    manifest = load_batch_manifest(run_dir)
    jobs = _load_jobs(run_dir)
    if not jobs.get("jobs"):
        raise ValueError(
            f"No submitted batch jobs recorded in {run_dir}; there is nothing to process."
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
    responses, response_stats = _successful_response_map(run_dir)
    expected_ids = _initial_request_ids(run_dir, manifest)
    model_id = manifest["model_id"]
    num_trials = int(manifest["num_trials"])
    include_flipped = bool(manifest["include_flipped"])
    with_reasoning = bool(manifest["with_reasoning"])
    results_dir = Path(manifest["results_dir"])
    batch_ids = [entry["batch_id"] for entry in jobs.get("jobs", [])]

    slots: dict[tuple[int, int, str], list[str | None]] = {}
    usage_by_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cost_by_set: dict[int, float] = defaultdict(float)
    for custom_id, row in responses.items():
        set_index, comparison_index, direction, trial_index = decode_custom_id(custom_id)
        key = (set_index, comparison_index, direction)
        values = slots.setdefault(key, [None] * num_trials)
        if not 0 <= trial_index < num_trials:
            raise ValueError(f"Trial index out of range in {custom_id!r}.")
        body = row["response"]["body"]
        choices = body.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        values[trial_index] = _message_text(message)
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
    written_files: list[str] = []
    for set_index, set_record in enumerate(manifest["sets"]):
        comparison_path = Path(set_record["comparison_path"])
        comparisons = load_comparisons(
            Path(manifest["data_dir"]),
            set_record["test_name"],
            comparison_path,
        )
        preferences: list[dict[str, Any]] = []
        set_unparseable = 0
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
            pref = {
                "outcome_a": comparison["outcome_a"],
                "outcome_b": comparison["outcome_b"],
                "count_prefer_a": count_a,
                "count_prefer_b": count_b,
                "prob_prefer_a": round(count_a / parsed_total, 4) if parsed_total else 0.0,
                "prob_prefer_b": round(count_b / parsed_total, 4) if parsed_total else 0.0,
            }
            if with_reasoning:
                pref["raw_responses_original"] = original
                pref["raw_responses_flipped"] = flipped
            preferences.append(pref)

        total_unparseable += set_unparseable
        expected_calls = len(comparisons) * num_trials * (2 if include_flipped else 1)
        usage_summary = _summarize_usage(usage_by_set.get(set_index, []))
        computed_cost = round(cost_by_set.get(set_index, 0.0), 6)
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
                "temperature": float(manifest["temperature"]),
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
                "successful_api_responses": len(usage_by_set.get(set_index, [])),
                "unparseable_count": set_unparseable,
                "unparseable_rate": set_unparseable / expected_calls if expected_calls else 0.0,
                "elapsed_seconds": None,
                "usage_stats": usage_summary,
                "model_name_full": model_id,
                "extra_body": manifest.get("extra_body") or None,
                "batch_ids": batch_ids,
                "batch_run_dir": str(run_dir),
                "estimated_cost_usd": computed_cost,
                "actual_cost_usd": computed_cost,
                "cost_source": "computed_from_batch_usage",
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
                **_host_info(),
            },
            "preferences": preferences,
        }
        result_path = (
            results_dir
            / artifact_dir_name_for_test(set_record["test_name"])
            / "results.json"
        )
        save_results(result_path, payload)
        written_files.append(str(result_path))

    missing_ids = expected_ids.difference(responses)
    summary = {
        "schema_version": "1.0",
        "processed_at": _utc_now(),
        "model_key": manifest["model_key"],
        "model_id": model_id,
        "batch_run_dir": str(run_dir),
        "expected_requests": len(expected_ids),
        "successful_responses": len(responses),
        "missing_responses": len(missing_ids),
        "unparseable_or_missing_responses": total_unparseable,
        "response_file_stats": response_stats,
        "result_files_written": len(written_files),
        "result_files": written_files,
    }
    _atomic_write_json(run_dir / BATCH_PROCESSING_SUMMARY_NAME, summary)
    return summary
