"""A separate, paired temperature pilot using the paper's forced-choice task.

Prepare and analyze are offline. Run requires an explicit compute ceiling. The
primary comparison uses curves with all planned responses at EVERY temperature,
so changes in parsed sample size cannot silently change the scoring resolution.
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
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from llm_coherence.runtime.agents import AsyncRequestLimiter
from llm_coherence.runtime.api_keys import require_api_key
from llm_coherence.runtime.templates import comparison_prompt_template_default
from llm_coherence.runtime.utils import parse_responses_forced_choice


REPO = Path(__file__).resolve().parents[3]
MODEL = "mistralai/ministral-3b-2512"
ENDPOINT = "mistral/zdr"
BASE_URL = "https://openrouter.ai/api/v1"
PRICE_PER_TOKEN = 0.0000001  # Verified endpoint metadata, 2026-11-03.
MAX_OUTPUT_TOKENS = 10
DEFAULT_ROOT = REPO / "outputs/temperature_sensitivity_20261103"
LADDERS = REPO / "data/05_ladder_validation/phase6b_variations_pruned_final.json"
COMPARISONS = REPO / "data/06_forced_choice_inputs/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json"


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def source_hashes():
    paths = [Path(__file__), REPO / "src/llm_coherence/runtime/templates.py",
             REPO / "src/llm_coherence/runtime/utils.py"]
    return {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in paths}


def build_plan(ladders, references, seed=20261103, temperatures=(0.0, 0.3, 0.7, 1.0)):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for ladder in ladders:
        groups[ladder["category"]].append(ladder)
    selected = [rng.choice(sorted(groups[c], key=lambda x: x["original_statement_id"]))
                for c in sorted(groups)]
    selected_refs = sorted(rng.sample(references, 3), key=lambda x: x["sample_index"])
    prompts, base_tasks = [], []
    for ladder in selected:
        tiers = sorted(ladder["variations"], key=lambda t: t["tier"])
        if [t["tier"] for t in tiers] != list(range(1, 8)):
            raise ValueError("Every selected ladder must contain T1 through T7.")
        for reference in selected_refs:
            for tier in tiers:
                for order in ("AB", "BA"):
                    a, b = tier["text"], reference["text"]
                    if order == "BA":
                        a, b = b, a
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": comparison_prompt_template_default.format(option_A=a, option_B=b)},
                    ]
                    # Conservative planning allowance: one token per UTF-8 byte,
                    # plus 512 framing tokens and the fixed output-token cap.
                    token_allowance = sum(len(m["content"].encode()) for m in messages) + 512 + MAX_OUTPUT_TOKENS
                    prompt_id = len(prompts)
                    prompts.append({"messages": messages, "token_allowance": token_allowance})
                    for repeat in range(10):
                        base_tasks.append({
                            "ladder_id": str(ladder["original_statement_id"]),
                            "category": ladder["category"], "comparison_id": reference["sample_index"],
                            "tier": tier["tier"], "order": order, "repeat": repeat,
                            "prompt_id": prompt_id,
                        })
    rng.shuffle(base_tasks)
    tasks = []
    for base in base_tasks:
        for temperature in rng.sample(list(temperatures), len(temperatures)):
            task = dict(base, temperature=temperature)
            task["request_id"] = sha(encoded(task))[:24]
            tasks.append(task)
    if len({t["request_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate request identifiers.")
    manifest = {
        "schema": "paired_temperature_pilot_v1", "model": MODEL, "provider_endpoint": ENDPOINT,
        "temperatures": list(temperatures), "selection_seed": seed,
        "selection": "One randomly selected ladder per category; three shared references; no outcome-based selection.",
        "ladder_ids": [str(x["original_statement_id"]) for x in selected],
        "reference_ids": [x["sample_index"] for x in selected_refs],
        "n_ladders": len(selected), "n_curves_per_temperature": len(selected) * 3,
        "trials_per_order": 10, "trials_per_cell": 20, "max_tokens": MAX_OUTPUT_TOKENS,
        "n_requests": len(tasks), "max_price_per_million_tokens": PRICE_PER_TOKEN * 1e6,
        "reserved_cost_usd": sum(prompts[t["prompt_id"]]["token_allowance"] * PRICE_PER_TOKEN for t in tasks),
        "primary_analysis": "Same fully observed seven-tier curves at every temperature; descriptive pilot.",
        "interpretation": "A change in sampled monotonicity alone does not establish a change in latent value coherence.",
        "prompts_sha256": sha(encoded(prompts)), "tasks_sha256": sha(encoded(tasks)),
        "source_sha256": source_hashes(),
    }
    return manifest, prompts, tasks


def prepare(root):
    ladders = json.loads(LADDERS.read_text(encoding="utf-8"))
    references = json.loads(COMPARISONS.read_text(encoding="utf-8"))["comparison_sample"]
    if len(ladders) != 100 or len({x["original_statement_id"] for x in ladders}) != 100 or len(references) != 30:
        raise ValueError("Expected the fixed 100-ladder / 30-reference instrument.")
    manifest, prompts, tasks = build_plan(ladders, references)
    manifest["input_sha256"] = {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in (LADDERS, COMPARISONS)}
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        if json.loads((root / "manifest.json").read_text()) != manifest:
            raise ValueError("A different pilot already occupies this directory.")
        load_plan(root)
    else:
        if any(p.name != ".run.lock" for p in root.iterdir()):
            raise ValueError("Choose an empty output directory.")
        (root / "prompts.json").write_bytes(encoded(prompts))
        (root / "tasks.json").write_bytes(encoded(tasks))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["model", "temperatures", "n_ladders", "n_curves_per_temperature", "n_requests", "reserved_cost_usd"]}, indent=2))
    print(f"Prepared {root}; no model calls made.")


def load_plan(root, check_source=True):
    manifest = json.loads((root / "manifest.json").read_text())
    prompts = json.loads((root / "prompts.json").read_text())
    tasks = json.loads((root / "tasks.json").read_text())
    if (manifest["schema"] != "paired_temperature_pilot_v1"
            or sha(encoded(prompts)) != manifest["prompts_sha256"]
            or sha(encoded(tasks)) != manifest["tasks_sha256"]):
        raise ValueError("Prepared request files have changed.")
    if check_source and manifest["source_sha256"] != source_hashes():
        raise ValueError("Experiment code changed since preparation. Prepare a new output directory.")
    return manifest, prompts, tasks


def read_journal(root, tasks):
    allowed = {t["request_id"] for t in tasks}
    started, results = set(), {}
    path = root / "responses.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["request_id"]
            if key not in allowed:
                raise ValueError("Saved response belongs to a different pilot.")
            if row["event"] == "started" and key not in started:
                started.add(key)
            elif row["event"] == "result" and key in started and key not in results:
                results[key] = row
            else:
                raise ValueError("Duplicate or invalid journal event; refusing to repeat requests.")
    return started, results


def append(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def request_body(task, prompts):
    return {
        "model": MODEL, "messages": prompts[task["prompt_id"]]["messages"],
        "temperature": task["temperature"], "top_p": 1, "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
        "provider": {"only": [ENDPOINT], "allow_fallbacks": False, "require_parameters": True,
                     "max_price": {"prompt": PRICE_PER_TOKEN * 1e6, "completion": PRICE_PER_TOKEN * 1e6}},
    }


def parse_reply(payload):
    choices = payload.get("choices") or []
    if not choices or choices[0].get("finish_reason") != "stop":
        return None
    raw = (choices[0].get("message") or {}).get("content")
    if not isinstance(raw, str):
        return None
    answer = parse_responses_forced_choice({0: [raw]}, verbose=False)[0][0]
    return answer if answer in ("A", "B") else None


async def run(root, max_cost_usd, rate=10.0, concurrency=8):
    import httpx
    manifest, prompts, tasks = load_plan(root)
    if not math.isfinite(max_cost_usd) or max_cost_usd <= 0 or manifest["reserved_cost_usd"] > max_cost_usd:
        raise ValueError("The compute ceiling must cover the conservative allowance in manifest.json.")
    started, results = read_journal(root, tasks)
    pending = [t for t in tasks if t["request_id"] not in started]
    if not pending:
        print("No unattempted requests remain. Existing failures and unresolved requests are not replayed.")
        return
    key = require_api_key("openrouter")
    limiter = AsyncRequestLimiter(max_concurrency=concurrency, requests_per_second=rate)
    queue = asyncio.Queue()
    for task in pending:
        queue.put_nowait(task)
    stopped = asyncio.Event()
    errors = []
    async with httpx.AsyncClient(timeout=90.0, headers={"Authorization": f"Bearer {key}"}) as client:
        metadata = await client.get(f"{BASE_URL}/models/{MODEL}/endpoints")
        metadata.raise_for_status()
        data = metadata.json()
        endpoints = data.get("data", {}).get("endpoints", [])
        endpoint = next((e for e in endpoints if e.get("tag") == ENDPOINT), None)
        if endpoint is None or "temperature" not in endpoint.get("supported_parameters", []):
            raise ValueError("The pinned endpoint is unavailable or does not advertise temperature support.")
        for field in ("prompt", "completion"):
            if float(endpoint["pricing"][field]) > PRICE_PER_TOKEN:
                raise ValueError("Endpoint prices exceed the prepared ceiling.")
        (root / "endpoint_at_run.json").write_text(json.dumps(data, indent=2))
        with (root / "responses.jsonl").open("a", encoding="utf-8") as log:
            async def worker():
                while not queue.empty() and not stopped.is_set():
                    task = queue.get_nowait()
                    key_id = task["request_id"]
                    async with limiter.slot():
                        if stopped.is_set():
                            return
                        append(log, {"event": "started", "request_id": key_id, "time": time.time()})
                        started.add(key_id)
                        row = {"event": "result", "request_id": key_id, "time": time.time(), "answer": None}
                        try:
                            response = await client.post(f"{BASE_URL}/chat/completions", json=request_body(task, prompts))
                            row["http_status"] = response.status_code
                            try:
                                payload = response.json()
                            except ValueError:
                                payload = {"non_json_body": response.text[:2000]}
                            row["response"] = payload
                            if response.status_code == 200:
                                if payload.get("provider") != "Mistral":
                                    raise ValueError("Returned provider does not match the pinned Mistral provider.")
                                row["answer"] = parse_reply(payload)
                                usage = payload.get("usage") or {}
                                tokens = float(usage.get("prompt_tokens", 0)) + float(usage.get("completion_tokens", 0))
                                cost = usage.get("cost")
                                if (tokens > prompts[task["prompt_id"]]["token_allowance"]
                                        or (cost is not None and float(cost) > prompts[task["prompt_id"]]["token_allowance"] * PRICE_PER_TOKEN)):
                                    raise ValueError("Reported usage exceeded the reserved request allowance.")
                            else:
                                row["error"] = f"HTTP {response.status_code}"
                                errors.append(row["error"])
                                if response.status_code in (401, 402, 403, 404) or len(errors) >= 5:
                                    stopped.set()
                        except Exception as exc:
                            row["error"] = str(exc)
                            errors.append(type(exc).__name__)
                            # Settings/provider/allowance mismatches require review.
                            if isinstance(exc, ValueError) or len(errors) >= 5:
                                stopped.set()
                        append(log, row)
                        results[key_id] = row
                        if len(results) % 100 == 0 or queue.empty():
                            print(f"Saved {len(results)}/{len(tasks)} responses", flush=True)
            await asyncio.gather(*(worker() for _ in range(concurrency)))
    if stopped.is_set():
        print("Stopped after provider/errors/allowance checks. Inspect saved responses before resuming.")


def analyze(root):
    manifest, _, tasks = load_plan(root, check_source=False)
    started, results = read_journal(root, tasks)
    counts, coverage = defaultdict(lambda: [0, 0]), defaultdict(Counter)
    for task in tasks:
        temperature = task["temperature"]
        key = task["request_id"]
        coverage[temperature]["planned"] += 1
        coverage[temperature]["started"] += key in started
        coverage[temperature]["responses"] += key in results
        answer = results.get(key, {}).get("answer")
        if answer not in ("A", "B"):
            continue
        coverage[temperature]["parsed"] += 1
        cell = (temperature, task["ladder_id"], task["comparison_id"], task["tier"])
        counts[cell][1] += 1
        counts[cell][0] += answer == ("A" if task["order"] == "AB" else "B")
    curve_ids = sorted({(t["ladder_id"], t["comparison_id"]) for t in tasks})
    temperatures = manifest["temperatures"]
    common = {key for key in curve_ids if all(
        counts[(temperature, *key, tier)][1] == 20
        for temperature in temperatures for tier in range(1, 8)
    )}
    curves, summaries = [], []
    for temperature in temperatures:
        for ladder, reference in curve_ids:
            cells = [counts[(temperature, ladder, reference, tier)] for tier in range(1, 8)]
            complete = all(n > 0 for _, n in cells)
            p = [a / n for a, n in cells] if complete else None
            decreases = [max(0, p[i] - p[i + 1]) for i in range(6)] if p else None
            curves.append({
                "temperature": temperature, "ladder_id": ladder, "comparison_id": reference,
                "paired_full_response_curve": (ladder, reference) in common,
                "parsed_responses": sum(n for _, n in cells), "nonempty_at_all_tiers": complete,
                "strict_monotonicity": all(x == 0 for x in decreases) if decreases is not None else None,
                "mean_adjacent_decrease": sum(decreases) / 6 if decreases is not None else None,
                "largest_decrease": max(decreases) if decreases is not None else None,
                "flat_curve": len(set(p)) == 1 if p is not None else None,
                "saturated_curve": all(x == 0 for x in p) or all(x == 1 for x in p) if p is not None else None,
                **{f"p{tier}": p[tier - 1] if p is not None else None for tier in range(1, 8)},
            })
        selected = [c for c in curves if c["temperature"] == temperature and c["paired_full_response_curve"]]
        mean = lambda name: sum(c[name] for c in selected) / len(selected) if selected else None
        summaries.append({
            "temperature": temperature,
            **{name: coverage[temperature][name] for name in ("planned", "started", "responses", "parsed")},
            "unresolved_requests": coverage[temperature]["started"] - coverage[temperature]["responses"],
            "invalid_or_failed_responses": coverage[temperature]["responses"] - coverage[temperature]["parsed"],
            "common_full_response_curves": len(common), "planned_curves": len(curve_ids),
            "monotonicity_percent": 100 * mean("strict_monotonicity") if selected else None,
            "mean_adjacent_decrease": mean("mean_adjacent_decrease"),
            "mean_largest_decrease": mean("largest_decrease"),
            "flat_curve_fraction": mean("flat_curve"), "saturated_curve_fraction": mean("saturated_curve"),
        })
    baseline = summaries[0]["monotonicity_percent"]
    for summary in summaries:
        value = summary["monotonicity_percent"]
        summary["change_from_temperature_0_pp"] = value - baseline if baseline is not None and value is not None else None
    for name, rows in [("summary", summaries), ("curves", curves)]:
        with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summaries, indent=2))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-cost-usd", type=float, help="Required for run; total prepared-pilot compute ceiling")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            prepare(root)
        elif args.action == "analyze":
            analyze(root)
        else:
            if args.max_cost_usd is None:
                parser.error("run requires --max-cost-usd")
            asyncio.run(run(root, args.max_cost_usd))
            summaries = analyze(root)
            return 0 if all(s["parsed"] == s["planned"] for s in summaries) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
