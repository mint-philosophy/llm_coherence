"""Recover unrequested rankings and explicit API rejections without replaying answers.

The original artifacts are read-only. Native OpenAI Responses and Anthropic
Messages bodies fix the initial adapter errors. Model refusals, malformed
rankings, token caps, timeouts, and unresolved requests are never replayed.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import httpx

from llm_coherence.experiments import full_ladder_ranking as original


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_retry(row):
    if row is None or row.get("raw_response") or row.get("receipts"):
        return False
    outcome = row["outcome"]
    error = outcome.get("error") or ""
    return (
        outcome["status"] == "provider_error" and (
            "Unknown parameter: 'reasoning'" in error
            or "extra_body: Extra inputs are not permitted" in error
        )
    ) or (outcome["status"] == "transport_failure" and "RateLimitError" in error and "429" in error)


def native_request(profile, messages):
    provider = profile["endpoint"]["model_type"]
    model = profile["endpoint"]["model_name"]
    extra = json.loads(json.dumps(profile["extra_body"]))
    if provider == "openai":
        effort = extra.pop("reasoning_effort", None)
        reasoning = extra.pop("reasoning", {})
        if effort is not None:
            reasoning["effort"] = effort
        body = dict(model=model.removeprefix("openai/"), input=messages,
                    max_output_tokens=profile["max_tokens"], reasoning=reasoning,
                    service_tier="default", store=False, **extra)
        if effort in (None, "none"):
            body["temperature"] = profile["temperature"]
        return "https://api.openai.com/v1/responses", body
    if provider == "anthropic":
        system = [{"type": "text", "text": m["content"]} for m in messages if m["role"] == "system"]
        if profile["enable_cache"] and system:
            system[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        return "https://api.anthropic.com/v1/messages", dict(
            model=model, system=system, messages=[m for m in messages if m["role"] != "system"],
            max_tokens=profile["max_tokens"], temperature=profile["temperature"], **extra,
        )
    if provider == "openrouter":
        extra["usage"] = {"include": True}
        return "https://openrouter.ai/api/v1/chat/completions", dict(
            model=model.removeprefix("openrouter/"), messages=messages,
            max_tokens=profile["max_tokens"], temperature=profile["temperature"], **extra,
        )
    raise ValueError(provider)


def response_fields(provider, body):
    usage = body.get("usage") or {}
    if provider == "openai":
        raw = "".join(block.get("text", block.get("refusal", ""))
                      for item in body.get("output", []) if item.get("type") == "message"
                      for block in item.get("content", [])
                      if block.get("type") in ("output_text", "refusal"))
        reason = (body.get("incomplete_details") or {}).get("reason") or body.get("status")
        successful = body.get("status") == "completed"
        tokens = (usage.get("input_tokens"), usage.get("output_tokens"))
    elif provider == "anthropic":
        raw = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        reason = body.get("stop_reason")
        successful = reason in ("end_turn", "stop_sequence", "refusal")
        inp = usage.get("input_tokens")
        if isinstance(inp, int):
            inp += (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
        tokens = (inp, usage.get("output_tokens"))
    else:
        choice = (body.get("choices") or [{}])[0]
        raw = (choice.get("message") or {}).get("content") or ""
        reason = choice.get("finish_reason")
        successful = reason == "stop"
        tokens = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
    status = "completed" if successful else "provider_error"
    if reason in ("length", "max_tokens", "max_output_tokens"):
        status = "token_capped"
    return raw, status, reason, tokens


def prepare(source, output):
    if not (source / "summary.json").is_file():
        raise ValueError("Wait for the original runner to finish before preparing recovery.")
    saved = json.loads((source / "manifest.json").read_text())
    tasks, requests, regenerated = original.prepare(original.DEFAULT_INPUT, list(saved["models"]), saved["seed"])
    if regenerated != saved:
        raise ValueError("Original protocol or source files changed.")
    if sha(original.DEFAULT_INPUT) != saved["input_sha256"]:
        raise ValueError("Canonical instrument changed.")
    allowed = {t.ladder_id for t in tasks}
    request_by_id = {r["ladder_id"]: r for r in requests}
    pending, hashes = [], {}
    extra_reservation = 0.0
    for model, profile in saved["models"].items():
        path = source / model / "responses.jsonl"
        started, results = original.read_events(path, allowed)
        hashes[str(path)] = sha(path)
        for task in tasks:
            old = results.get(task.ladder_id)
            retry = eligible_retry(old)
            if task.ladder_id in started and not retry:
                continue
            messages = request_by_id[task.ladder_id]["messages"]
            url, body = native_request(profile, messages)
            reservation = original.reserved_cost(messages, profile)
            if retry:
                # Retain the original reservation even for rejected calls.
                extra_reservation += reservation
            pending.append({"request_id": original.digest([model, task.ladder_id]),
                            "model_key": model, "ladder_id": task.ladder_id,
                            "reason": "explicit_api_rejection" if retry else "not_previously_requested",
                            "url": url, "body": body, "reserved_cost_usd": reservation,
                            "input_token_allowance": original.input_allowance(messages)})
    manifest = {
        "schema": "full_ladder_ranking_recovery_v1", "source": str(source),
        "source_manifest_sha256": sha(source / "manifest.json"), "source_logs_sha256": hashes,
        "runner_sha256": sha(Path(__file__)), "requests_sha256": original.digest(pending),
        "n_requests": len(pending), "requests_per_model": dict(Counter(p["model_key"] for p in pending)),
        "original_reservation_usd": saved["reserved_cost_usd"],
        "additional_retry_reservation_usd": extra_reservation,
        "combined_reservation_usd": saved["reserved_cost_usd"] + extra_reservation,
        "budget_scope": "Original ranking run plus this recovery; excludes the separate temperature pilot.",
        "rule": "Never replay a model answer, token-capped response, timeout, or unresolved request.",
    }
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty recovery output directory.")
    original.write_json(output / "requests.json", pending)
    original.write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def read_log(path, ids):
    started, results = set(), {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line); key = r["request_id"]
            if key not in ids:
                raise ValueError("Unknown recovery request ID.")
            if r["event"] == "started" and key not in started:
                started.add(key)
            elif r["event"] == "result" and key in started and key not in results:
                results[key] = r
            else:
                raise ValueError("Invalid or duplicate recovery event.")
    return started, results


def load(output):
    manifest = json.loads((output / "manifest.json").read_text())
    requests = json.loads((output / "requests.json").read_text())
    if sha(Path(__file__)) != manifest["runner_sha256"] or original.digest(requests) != manifest["requests_sha256"]:
        raise ValueError("Recovery source or requests changed.")
    source = Path(manifest["source"])
    if sha(source / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Original manifest changed.")
    for path, digest in manifest["source_logs_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Original response log changed.")
    saved = json.loads((source / "manifest.json").read_text())
    tasks, _, regenerated = original.prepare(original.DEFAULT_INPUT, list(saved["models"]), saved["seed"])
    if regenerated != saved:
        raise ValueError("Original source or instrument changed.")
    return manifest, requests, saved, tasks


async def run(output, key_dir, max_cost):
    manifest, requests, saved, tasks = load(output)
    if not math.isfinite(max_cost) or max_cost < manifest["combined_reservation_usd"]:
        raise ValueError("Budget must cover the original run plus recovery reservations.")
    original.load_key_directory(key_dir, saved["models"])
    started, results = read_log(output / "responses.jsonl", {r["request_id"] for r in requests})
    task_by_id = {t.ladder_id: t for t in tasks}
    all_stop = asyncio.Event()
    model_slots = asyncio.Semaphore(12)
    model_stops = {m: asyncio.Event() for m in manifest["requests_per_model"]}
    pilot_done = {m: asyncio.Event() for m in manifest["requests_per_model"]}
    failures = Counter()
    async with httpx.AsyncClient(timeout=180.0, transport=httpx.AsyncHTTPTransport(retries=0)) as client:
        with (output / "responses.jsonl").open("a", encoding="utf-8") as handle:
            async def model_run(model, worker_index, workers):
                if worker_index:
                    await pilot_done[model].wait()
                async with model_slots:
                    profile = saved["models"][model]
                    provider = profile["endpoint"]["model_type"]
                    key = os.environ[original.API_KEY_ENV_BY_TYPE[provider]]
                    headers = ({"x-api-key": key, "anthropic-version": "2023-06-01"}
                               if provider == "anthropic" else {"Authorization": "Bearer " + key})
                    model_requests = [r for r in requests if r["model_key"] == model]
                    if not model_requests or all(r["request_id"] in started for r in model_requests):
                        pilot_done[model].set()
                    for index, request in enumerate(model_requests):
                        if index % workers != worker_index or request["request_id"] in started:
                            if worker_index == 0 and index == 0:
                                pilot_done[model].set()
                            continue
                        if all_stop.is_set() or model_stops[model].is_set():
                            pilot_done[model].set()
                            break
                        ident = {k: request[k] for k in ("request_id", "model_key", "ladder_id")}
                        original.append_event(handle, dict(event="started", **ident, timestamp=original.ranking.now_iso()))
                        started.add(request["request_id"])
                        row = dict(event="result", **ident, timestamp=original.ranking.now_iso(),
                                   parse_success=False, raw_response="", outcome={"status": "unknown"},
                                   reserved_cost_usd=request["reserved_cost_usd"])
                        stop_model = False
                        try:
                            response = await client.post(request["url"], headers=headers, json=request["body"])
                            row["http_status"] = response.status_code
                            body = response.json(); row["response"] = body
                            if response.status_code != 200:
                                row["outcome"] = {"status": "provider_error", "http_status": response.status_code}
                                failures[model] += 1
                                stop_model = response.status_code in (400, 401, 402, 403, 404) or failures[model] >= 3
                                if response.status_code == 429:
                                    await asyncio.sleep(10)
                            else:
                                raw, status, reason, (inp, out) = response_fields(provider, body)
                                row.update(raw_response=raw, outcome={"status": status, "finish_reason": reason})
                                parsed = original.ranking.parse_ranking_response(raw, task_by_id[request["ladder_id"]].labels) if status == "completed" else None
                                if parsed is not None:
                                    row.update(parse_success=True, parsed_ranking=parsed)
                                    row.update(original.ranking.score_ranking(task_by_id[request["ladder_id"]], parsed))
                                cost = (body.get("usage") or {}).get("cost")
                                if (not isinstance(inp, int) or not isinstance(out, int)
                                        or inp > request["input_token_allowance"] or out > profile["max_tokens"]
                                        or (cost is not None and float(cost) > request["reserved_cost_usd"] + 1e-8)):
                                    row["budget_check_error"] = "Missing usage or exceeded reservation."
                                    all_stop.set()
                        except Exception as exc:
                            row["outcome"] = {"status": "transport_failure", "error_type": type(exc).__name__}
                            failures[model] += 1; stop_model = failures[model] >= 3
                        original.append_event(handle, row)
                        results[request["request_id"]] = row
                        if stop_model:
                            model_stops[model].set()
                        if worker_index == 0:
                            pilot_done[model].set()
                        count = sum(r["model_key"] == model for r in results.values())
                        if count % 10 == 0 or count == 1:
                            print(f"{model}: {count} recovery responses saved", flush=True)
                        if stop_model:
                            print(f"{model}: stopped after provider/transport errors.", flush=True)
                            break
                        # OpenRouter remains single-file to avoid shared-pool bursts.
                        await asyncio.sleep(0.5)
            jobs = []
            for model in manifest["requests_per_model"]:
                workers = 1 if saved["models"][model]["endpoint"]["model_type"] == "openrouter" else 3
                jobs.extend(model_run(model, i, workers) for i in range(workers))
            await asyncio.gather(*jobs)
    return analyze(output)


def analyze(output):
    manifest, requests, saved, tasks = load(output)
    started_new, results_new = read_log(output / "responses.jsonl", {r["request_id"] for r in requests})
    rows = []
    for model in saved["models"]:
        started, results = original.read_events(Path(manifest["source"]) / model / "responses.jsonl", {t.ladder_id for t in tasks})
        for request in requests:
            if request["model_key"] != model or request["request_id"] not in started_new:
                continue
            ladder = request["ladder_id"]
            if ladder in results and not eligible_retry(results[ladder]):
                raise ValueError("Attempt to replace a model outcome.")
            started.add(ladder)
            results.pop(ladder, None)
            if request["request_id"] in results_new:
                results[ladder] = results_new[request["request_id"]]
        rows.append(original.summarize(model, tasks, started, results))
    original.write_json(output / "combined_summary.json", rows)
    with (output / "combined_summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "analyze"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key-dir", type=Path)
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args(); output = args.output.resolve()
    if args.mode == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.source.resolve(), output); return
    if args.mode == "analyze":
        analyze(output); return
    if args.api_key_dir is None or args.max_cost_usd is None:
        parser.error("run requires --api-key-dir and --max-cost-usd")
    source = Path(json.loads((output / "manifest.json").read_text())["source"])
    with (output / ".run.lock").open("a") as lock, (source / ".run.lock").open("r") as source_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        rows = asyncio.run(run(output, args.api_key_dir.resolve(), args.max_cost_usd))
    if any(r["n_not_requested"] or r["n_unresolved_requests"] for r in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
