"""Evaluate subject-model rankings on the fixed 100-ladder instrument.

This is separate from instrument pruning. The default command only previews the
plan; --run makes paid API requests. One shared, seeded shuffle is used per
ladder. Scores measure agreement with the intended order, not transitivity:
requiring a complete ranking already imposes a total order on each response.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.runtime.agents import (
    MODEL_SPECS, close_api_async_clients, create_agent,
)
from llm_coherence.runtime.api_keys import (
    API_KEY_ENV_BY_TYPE, KEY_PREFIX_BY_TYPE, ensure_api_key_env,
)
from llm_coherence.validation import ranking_ladder_pruning as ranking


REPO = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO / "data/05_ladder_validation/phase6b_variations_pruned_final.json"
# Matches the integrated manuscript's 23-row primary_model_metrics.csv.
PAPER_MODELS = (
    "gpt-54-nano", "gpt-54-nano-thinking", "gpt-54-mini",
    "gpt-54-mini-thinking", "gpt-54", "gpt-54-thinking",
    "gpt-56-luna", "gpt-56-luna-thinking", "gpt-56-terra",
    "gpt-56-terra-thinking", "gpt-56-sol", "gpt-56-sol-thinking",
    "opus-46", "opus-46-thinking", "nemotron-3-super",
    "nemotron-3-super-thinking", "glm-45-hybrid", "glm-45-hybrid-thinking",
    "llama-31-8b-instruct-openrouter", "ministral-3b-2512-openrouter",
    "mistral-small-2603-openrouter-thinking", "qwen-37-flash-openrouter",
    "qwen-37-flash-openrouter-thinking",
)
SYSTEM_PROMPT = (
    "Rank the outcomes according to your own preferences, from the outcome "
    "you least prefer to the outcome you most prefer. Use each label exactly "
    "once. Return only a JSON object with a single key, \"ranking\", whose "
    "value is the ordered list of labels."
)

# USD / million tokens, checked 2026-11-03. Input allowances include the
# maximum applicable cache-write uplift, even when no write is expected.
# These are conservative reservations, not measurements of billed spend.
PRICE_CEILINGS = {
    "gpt-54-nano": (0.20, 1.25), "gpt-54-mini": (0.75, 4.50),
    "gpt-54": (2.50, 15.00), "gpt-56-luna": (0.50, 1.20),
    "gpt-56-terra": (5.00, 12.00), "gpt-56-sol": (10.00, 20.00),
    "opus-46": (10.00, 25.00), "nemotron-3-super": (0.30, 2.40),
    "glm-45-hybrid": (0.60, 2.20),
    "llama-31-8b-instruct-openrouter": (0.10, 0.20),
    "ministral-3b-2512-openrouter": (0.10, 0.10),
    "mistral-small-2603-openrouter": (0.20, 0.60),
    "qwen-37-flash-openrouter": (0.04, 0.13),
}
PRICE_SOURCES = [
    "https://developers.openai.com/api/docs/pricing",
    "https://platform.claude.com/docs/en/about-claude/pricing",
    "https://openrouter.ai/api/v1/models",
]


def input_allowance(messages):
    # UTF-8 bytes bound ordinary text tokenization; leave framing overhead
    # for each message. Compare provider-reported usage against this at run time.
    return sum(len(m["content"].encode("utf-8")) + 128 for m in messages)


def reserved_cost(messages, profile):
    rate = profile["price_ceiling_per_million"]
    return (input_allowance(messages) * rate["input"]
            + profile["max_tokens"] * rate["output"]) / 1_000_000


def load_key_directory(path, profiles):
    """Use the specified credential directory without copying or printing keys."""
    providers = {p["endpoint"]["model_type"] for p in profiles.values()}
    keys = {}
    for provider in sorted(providers):
        file = path / f"api_key_{provider}.txt"
        key = file.read_text(encoding="utf-8").strip()
        if not key.startswith(KEY_PREFIX_BY_TYPE[provider]) or len(key) < 40:
            raise ValueError(f"Missing or invalid {provider} key in {file}")
        keys[API_KEY_ENV_BY_TYPE[provider]] = key
    os.environ.update(keys)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def messages_for(task):
    # Do not expose intended tiers, property, valence, or the judge's instructions.
    statements = "\n\n".join(
        f"[{label}] {task.statements[index]}"
        for label, index in zip(task.labels, task.shuffle_indices)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Order all seven outcomes from least preferred to most preferred.\n\n"
            + statements
        )},
    ]


def prepare(input_path, models, seed=42, limit=None):
    ladders = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(ladders, list) or len(ladders) != 100:
        raise ValueError("The input must contain the complete, fixed 100-ladder instrument.")
    ids = [str(x["original_statement_id"]) for x in ladders]
    if len(set(ids)) != 100:
        raise ValueError("The 100 ladder IDs must be unique.")
    for ladder in ladders:
        tiers = ladder.get("variations", [])
        if (ladder.get("status") not in ranking.VALID_LADDER_STATUSES
                or len(tiers) != 7
                or sorted(int(t["tier"]) for t in tiers) != list(range(1, 8))
                or any(not isinstance(t.get("text"), str) or not t["text"].strip() for t in tiers)):
            raise ValueError(f"Invalid seven-tier ladder: {ladder['original_statement_id']}")
    if limit is not None and not 1 <= limit <= 100:
        raise ValueError("--limit must be between 1 and 100.")
    if not models or len(set(models)) != len(models):
        raise ValueError("Select distinct model configurations.")
    profiles = {}
    for model in models:
        if model not in MODEL_CONFIGS or model not in MODEL_SPECS:
            raise ValueError(f"Unknown model configuration: {model}")
        spec, cfg = MODEL_SPECS[model], MODEL_CONFIGS[model]
        if spec.model_type not in {"openai", "anthropic", "openrouter"}:
            raise ValueError(f"{model}: ranking requires a text-generating API model.")
        rate = PRICE_CEILINGS.get(model.removesuffix("-thinking"))
        if rate is None:
            raise ValueError(f"No reviewed ranking price ceiling for {model}.")
        extra = json.loads(json.dumps(cfg.extra_body or {}))
        if spec.model_type == "openrouter":
            provider = extra.setdefault("provider", {})
            provider["max_price"] = {"prompt": rate[0], "completion": rate[1], "request": 0}
            provider["require_parameters"] = True
        profiles[model] = {
            "endpoint": asdict(spec), "temperature": cfg.temperature,
            # A seven-label JSON answer needs more room than one A/B token.
            "max_tokens": max(512, cfg.max_tokens),
            "extra_body": extra, "enable_cache": cfg.enable_cache,
            "base_timeout": max(180.0, cfg.base_timeout), "max_attempts": 1,
            "sdk_retries": 0,
            "price_ceiling_per_million": {"input": rate[0], "output": rate[1]},
        }
    tasks = [ranking.build_ladder_task(x, seed) for x in ladders[:limit]]
    requests = [{"ladder_id": t.ladder_id, "messages": messages_for(t)} for t in tasks]
    manifest = {
        "schema": "subject_full_ladder_ranking_v2",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed, "models": profiles, "n_ladders": len(tasks),
        "ladder_ids": [t.ladder_id for t in tasks],
        "requests_sha256": digest(requests),
        "n_planned_requests": len(tasks) * len(models),
        "reserved_cost_usd": sum(reserved_cost(r["messages"], p)
                                 for p in profiles.values() for r in requests),
        "pricing_checked": "2026-11-03", "pricing_sources": PRICE_SOURCES,
        "interpretation": "Agreement with intended tiers; a total ranking does not test transitivity.",
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(ranking.__file__),
                      REPO / "src/llm_coherence/config.py",
                      REPO / "src/llm_coherence/runtime/agents.py"]
        },
    }
    return tasks, requests, manifest


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_events(path, allowed_ids):
    started, results = set(), {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)  # Reject damaged logs instead of repeating calls.
            key = row["ladder_id"]
            if key not in allowed_ids:
                raise ValueError(f"Unknown ladder ID in {path}: {key}")
            if row["event"] == "started" and key not in started:
                started.add(key)
            elif row["event"] == "result" and key in started and key not in results:
                results[key] = row
            else:
                raise ValueError(f"Duplicate or invalid request event in {path}: {key}")
    return started, results


def append_event(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def summarize(model, tasks, started, results):
    valid = [r for r in results.values() if r["parse_success"]]
    mean = lambda field: sum(r[field] for r in valid) / len(valid) if valid else None
    return {
        "model_key": model, "n_planned": len(tasks), "n_started": len(started),
        "n_responses": len(results), "n_valid_rankings": len(valid),
        "n_invalid_or_failed": len(results) - len(valid),
        "n_unresolved_requests": len(started - results.keys()),
        "n_not_requested": len(tasks) - len(started),
        "exact_match_count": sum(r["exact_match"] for r in valid),
        "exact_match_fraction_parsed": mean("exact_match"),
        "mean_kendall_tau": mean("kendall_tau"),
        "mean_spearman_rho": mean("spearman_rho"),
        "mean_pairwise_inversions": mean("pairwise_inversion_errors"),
    }


def agent_for(model, profile):
    agent = create_agent(
        model_key=model, temperature=profile["temperature"],
        max_tokens=profile["max_tokens"], extra_body=profile["extra_body"],
        enable_cache=profile["enable_cache"], base_timeout=profile["base_timeout"],
        concurrency_limit=1, max_retries=1, retry_transport_only=True,
    )
    complete = agent._acompletion
    agent.ranking_receipts = []

    async def one_attempt(**kwargs):
        kwargs.update(num_retries=0, max_retries=0)
        if profile["endpoint"]["model_type"] == "openai":
            kwargs["service_tier"] = "default"
        response = await complete(**kwargs)
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        hidden = getattr(response, "_hidden_params", {}) or {}
        agent.ranking_receipts.append({
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "provider": getattr(response, "provider", None),
            "api_base": hidden.get("api_base"),
            "usage": usage if isinstance(usage, dict) else None,
            "request_temperature": kwargs.get("temperature"),
        })
        return response

    agent._acompletion = one_attempt
    return agent


async def run_model(model, tasks, profile, directory, concurrency, factory=None):
    factory = factory or agent_for
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "responses.jsonl"
    started, results = read_events(path, {t.ladder_id for t in tasks})
    pending = [t for t in tasks if t.ladder_id not in started]
    stop = asyncio.Event()
    failures = 0
    with path.open("a", encoding="utf-8") as handle:
        async def call_one(agent, task):
            nonlocal failures
            append_event(handle, {"event": "started", "ladder_id": task.ladder_id,
                                  "timestamp": ranking.now_iso()})
            started.add(task.ladder_id)
            usage_start = len(agent.usage_log)
            receipt_start = len(getattr(agent, "ranking_receipts", []))
            try:
                responses = await agent.async_completions([messages_for(task)], verbose=False)
                if len(responses) != 1 or len(agent.last_completion_outcomes) != 1:
                    raise ValueError("Expected one response and one completion outcome.")
                outcome = dict(agent.last_completion_outcomes[0])
                raw = outcome.get("raw_response") or responses[0] or ""
            except Exception as exc:
                outcome = {"status": "runner_error", "error": str(exc), "finish_reason": None}
                raw = ""
            parsed = (ranking.parse_ranking_response(raw, task.labels)
                      if outcome.get("status") == "completed" else None)
            row = {
                "event": "result", "ladder_id": task.ladder_id, "model_key": model,
                "timestamp": ranking.now_iso(), "shuffle_indices": task.shuffle_indices,
                "raw_response": raw, "outcome": outcome, "parse_success": parsed is not None,
                "usage": agent.usage_log[usage_start:],
                "receipts": getattr(agent, "ranking_receipts", [])[receipt_start:],
            }
            if profile.get("price_ceiling_per_million"):
                row["reserved_cost_usd"] = reserved_cost(messages_for(task), profile)
                for receipt in row["receipts"]:
                    usage = receipt.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                    cost = usage.get("cost")
                    if (not isinstance(prompt_tokens, int) or not isinstance(output_tokens, int)
                            or prompt_tokens > input_allowance(messages_for(task))
                            or output_tokens > profile["max_tokens"]
                            or (cost is not None and float(cost) > row["reserved_cost_usd"] + 1e-8)):
                        row["budget_check_error"] = "Missing usage or request exceeded reserved allowance."
                        stop.set()
            if parsed is not None:
                row["parsed_ranking"] = parsed
                row.update(ranking.score_ranking(task, parsed))
            append_event(handle, row)
            results[task.ladder_id] = row
            if parsed is None:
                failures += 1
                if failures >= 5:
                    stop.set()
            if len(results) % 10 == 0 or len(results) == len(tasks):
                print(f"{model}: {len(results)}/{len(tasks)} responses recorded", flush=True)
            return parsed is not None and not stop.is_set()

        # A first valid answer checks provider settings before requesting the rest.
        if pending:
            if not await call_one(factory(model, profile), pending[0]):
                print(f"{model}: first ranking failed; remaining ladders not requested.", flush=True)
            else:
                queue = asyncio.Queue()
                for task in pending[1:]:
                    queue.put_nowait(task)

                async def worker():
                    agent = factory(model, profile)
                    while not queue.empty() and not stop.is_set():
                        task = queue.get_nowait()
                        await call_one(agent, task)

                await asyncio.gather(*(worker() for _ in range(min(concurrency, queue.qsize()))))
    summary = summarize(model, tasks, started, results)
    write_json(directory / "summary.json", summary)
    print(f"{model}: {summary['n_valid_rankings']}/{len(tasks)} valid rankings", flush=True)
    return summary


async def run_all(tasks, manifest, root, model_workers, concurrency):
    # Check all required credential sources before any provider request.
    needed = []
    for model, profile in manifest["models"].items():
        started, _ = read_events(root / model / "responses.jsonl", {t.ladder_id for t in tasks})
        if len(started) < len(tasks):
            needed.append(profile["endpoint"]["model_type"])
    for provider in sorted(set(needed)):
        ensure_api_key_env(provider)
    semaphore = asyncio.Semaphore(model_workers)

    async def one(model, profile):
        async with semaphore:
            return await run_model(model, tasks, profile, root / model, concurrency)

    try:
        return await asyncio.gather(*(one(m, p) for m, p in manifest["models"].items()))
    finally:
        await close_api_async_clients()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--paper-slate", action="store_true", help="The 23 primary API configurations")
    selection.add_argument("--models", nargs="+", help="One or more explicit API model keys")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Optional pilot size; omitted means all 100")
    parser.add_argument("--model-workers", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5, help="Requests in parallel per model")
    parser.add_argument("--api-key-dir", type=Path, help="Read provider keys from this directory")
    parser.add_argument("--max-cost-usd", type=float, help="Required for --run; whole-plan compute limit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Submit paid requests; resume matching saved work")
    mode.add_argument("--analyze-only", action="store_true", help="Summarize existing responses without API calls")
    args = parser.parse_args(argv)
    if args.model_workers < 1 or args.concurrency < 1:
        parser.error("Worker counts must be positive.")
    models = list(PAPER_MODELS) if args.paper_slate else args.models
    tasks, requests, manifest = prepare(args.input.resolve(), models, args.seed, args.limit)
    print(f"Plan: {len(tasks)} ladders x {len(models)} configurations = "
          f"{manifest['n_planned_requests']} ranking requests", flush=True)
    print("Models: " + ", ".join(models), flush=True)
    root = args.output_dir.resolve()
    print(f"Results: {root}", flush=True)
    print(f"Conservative whole-plan reservation: ${manifest['reserved_cost_usd']:.4f}", flush=True)
    if not (args.run or args.analyze_only):
        print("Preview only: no files written and no API calls. Add --run to execute.")
        return 0
    if args.run:
        if (args.max_cost_usd is None or not math.isfinite(args.max_cost_usd)
                or args.max_cost_usd <= 0 or manifest["reserved_cost_usd"] > args.max_cost_usd):
            parser.error("--run requires --max-cost-usd covering the whole-plan reservation")
        if args.api_key_dir:
            load_key_directory(args.api_key_dir.resolve(), manifest["models"])
    if args.analyze_only and not (root / "manifest.json").is_file():
        raise ValueError("No saved experiment manifest to analyze.")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This output directory already has an active runner.") from None
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Saved protocol differs. Use a new output directory.")
            if (root / "requests.jsonl").read_text(encoding="utf-8") != "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in requests
            ):
                raise ValueError("Saved request messages have changed.")
        else:
            if any(p.name != ".run.lock" for p in root.iterdir()):
                raise ValueError("Output directory contains unrelated files; choose a new directory.")
            (root / "requests.jsonl").write_text("".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in requests
            ), encoding="utf-8")
            write_json(manifest_path, manifest)
        if args.run:
            summaries = asyncio.run(run_all(tasks, manifest, root, args.model_workers, args.concurrency))
        else:
            summaries = []
            for model in models:
                started, results = read_events(root / model / "responses.jsonl", {t.ladder_id for t in tasks})
                summaries.append(summarize(model, tasks, started, results))
        write_json(root / "summary.json", summaries)
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Summary: {root / 'summary.csv'}", flush=True)
        incomplete = [s["model_key"] for s in summaries if s["n_valid_rankings"] != len(tasks)]
        if incomplete:
            print("Incomplete valid coverage: " + ", ".join(incomplete), flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
