"""Batch-API path for the full-ladder ranking (OpenAI and Anthropic configurations).

Same tasks, prompt and seeded shuffle as full_ladder_ranking, and the same native
request bodies as the recovery runner. OpenRouter has no batch API, so open models
stay on the live runner. Results are written in the live runner's responses.jsonl
event format, so read_events/summarize and the audit read them unchanged.

prepare is offline. submit is the only paid step (one batch per model, never resent).
status and fetch only read. Not used for the November 3 results; those came from
the live runner plus recovery.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path

import httpx

from llm_coherence.experiments import full_ladder_ranking as original
from llm_coherence.experiments.full_ladder_ranking_recovery import native_request, response_fields

OPENAI = "https://api.openai.com"
ANTHROPIC = "https://api.anthropic.com"
OPENAI_DONE = {"completed", "failed", "expired", "cancelled"}
# error codes on OpenAI error-file lines for requests the batch never ran
# (not yet checked against real output; anything else is a provider_error)
OPENAI_NEVER_RAN = {"batch_expired", "batch_cancelled"}


def prepare(output, models, input_path=original.DEFAULT_INPUT, seed=42, limit=None):
    tasks, requests, live = original.prepare(Path(input_path).resolve(), list(models), seed, limit)
    for model, profile in live["models"].items():
        if profile["endpoint"]["model_type"] not in ("openai", "anthropic"):
            raise ValueError(f"{model}: OpenRouter has no batch API; use the live runner.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory.")
    rows = []
    for model, profile in live["models"].items():
        for request in requests:
            _, body = native_request(profile, request["messages"])
            body.pop("service_tier", None)  # batch requests have their own tier
            rows.append({"custom_id": original.digest([model, request["ladder_id"]]),
                         "model_key": model, "ladder_id": request["ladder_id"],
                         "provider": profile["endpoint"]["model_type"], "body": body,
                         "reserved_cost_usd": original.reserved_cost(request["messages"], profile),
                         "input_token_allowance": original.input_allowance(request["messages"])})
    manifest = {
        "schema": "full_ladder_ranking_batch_v1", "source_manifest": live,
        "input_path": str(Path(input_path).resolve()), "limit": limit,
        "requests_sha256": original.digest(rows), "n_requests": len(rows),
        # list-price ceilings; batch pricing is lower, so this stays conservative
        "reserved_cost_usd": sum(r["reserved_cost_usd"] for r in rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    original.write_json(output / "requests.json", rows)
    original.write_json(output / "manifest.json", manifest)
    for model in live["models"]:
        mine = [r for r in rows if r["model_key"] == model]
        (output / model).mkdir()
        if mine[0]["provider"] == "openai":
            lines = [{"custom_id": r["custom_id"], "method": "POST", "url": "/v1/responses", "body": r["body"]}
                     for r in mine]
            (output / model / "batch_input.jsonl").write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
        else:
            original.write_json(output / model / "batch_input.json",
                                {"requests": [{"custom_id": r["custom_id"], "params": r["body"]} for r in mine]})
    print(f"Prepared {len(rows)} batch requests for {len(live['models'])} configurations; "
          f"reservation ${manifest['reserved_cost_usd']:.4f}. Nothing submitted.")
    return manifest


def load(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    rows = json.loads((output / "requests.json").read_text())
    if original.digest(rows) != manifest["requests_sha256"]:
        raise ValueError("Saved batch requests changed since prepare.")
    path = output / "batches.json"
    batches = json.loads(path.read_text()) if path.exists() else {}
    return manifest, rows, batches


def headers(provider):
    key = os.environ[original.API_KEY_ENV_BY_TYPE[provider]]
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": "Bearer " + key}


def checked(response):
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} from {response.request.url.path}: {response.text[:300]}")
    return response


@contextmanager
def locked(output):
    with (Path(output) / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This output directory already has an active runner.") from None
        yield


def submit(output, max_cost, client):
    output = Path(output)
    with locked(output):
        return _submit(output, max_cost, client)


def _submit(output, max_cost, client):
    manifest, _, batches = load(output)
    if not math.isfinite(max_cost) or max_cost < manifest["reserved_cost_usd"]:
        raise ValueError(f"--max-cost-usd must cover the reservation (${manifest['reserved_cost_usd']:.4f}).")
    # a create whose reply never came back may still have made a billed batch
    unconfirmed = [m for m, e in batches.items() if e.get("create_attempted_at") and not e.get("batch_id")]
    if unconfirmed:
        raise ValueError(
            "A batch create may have reached the provider without being recorded for: "
            + ", ".join(unconfirmed) + ". Check the provider's batch list, then save that id as "
            "batch_id in batches.json, or delete create_attempted_at to retry.")
    for model, profile in manifest["source_manifest"]["models"].items():
        entry = batches.get(model, {})
        if entry.get("batch_id"):
            print(f"{model}: already submitted as {entry['batch_id']}; not resending.")
            continue
        provider = profile["endpoint"]["model_type"]
        if provider == "openai":
            if not entry.get("input_file_id"):
                content = (output / model / "batch_input.jsonl").read_bytes()
                uploaded = checked(client.post(f"{OPENAI}/v1/files", headers=headers(provider),
                                               data={"purpose": "batch"},
                                               files={"file": ("batch_input.jsonl", content, "application/jsonl")})).json()
                entry["input_file_id"] = uploaded["id"]
                batches[model] = entry
                original.write_json(output / "batches.json", batches)
        entry["create_attempted_at"] = original.ranking.now_iso()
        batches[model] = entry
        original.write_json(output / "batches.json", batches)
        if provider == "openai":
            created = checked(client.post(f"{OPENAI}/v1/batches", headers=headers(provider), json={
                "input_file_id": entry["input_file_id"], "endpoint": "/v1/responses",
                "completion_window": "24h", "metadata": {"model_key": model}})).json()
        else:
            payload = json.loads((output / model / "batch_input.json").read_text())
            created = checked(client.post(f"{ANTHROPIC}/v1/messages/batches",
                                          headers=headers(provider), json=payload)).json()
        entry.update(provider=provider, batch_id=created["id"], submitted_at=original.ranking.now_iso())
        batches[model] = entry
        original.write_json(output / "batches.json", batches)  # record before the next model
        print(f"{model}: submitted {created['id']}")
    return batches


def status(output, client):
    _, _, batches = load(output)
    report = {model: status_one(entry, client) for model, entry in batches.items() if entry.get("batch_id")}
    for model, state in report.items():
        print(f"{model}: {state['state']}")
    return report


def download(entry, body, client, directory):
    """Raw result lines keyed by custom_id; raw text is kept next to the converted log."""
    texts = []
    if entry["provider"] == "openai":
        for field in ("output_file_id", "error_file_id"):
            if body.get(field):
                texts.append(checked(client.get(f"{OPENAI}/v1/files/{body[field]}/content",
                                                headers=headers("openai"))).text)
    elif body.get("results_url"):
        texts.append(checked(client.get(body["results_url"], headers=headers("anthropic"))).text)
    text = "".join(t if t.endswith("\n") or not t else t + "\n" for t in texts)
    (directory / "batch_results.raw.jsonl").write_text(text, encoding="utf-8")
    return {line["custom_id"]: line for line in map(json.loads, filter(str.strip, text.splitlines()))}


def result_row(request, line, task, batch_id):
    provider = request["provider"]
    row = {"event": "result", "ladder_id": task.ladder_id, "model_key": request["model_key"],
           "timestamp": original.ranking.now_iso(), "shuffle_indices": task.shuffle_indices,
           "raw_response": "", "parse_success": False, "usage": [], "receipts": [],
           "batch_id": batch_id, "custom_id": request["custom_id"],
           "reserved_cost_usd": request["reserved_cost_usd"]}
    if provider == "openai":
        response = line.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code") != 200:
            if (line.get("error") or {}).get("code") in OPENAI_NEVER_RAN:
                return None
            row["outcome"] = {"status": "provider_error", "http_status": response.get("status_code"),
                              "error": json.dumps(line.get("error") or body.get("error"))[:500]}
            return row
    else:
        result = line["result"]
        if result["type"] in ("expired", "canceled"):
            return None  # never ran: unresolved, not a bad answer
        if result["type"] != "succeeded":
            row["outcome"] = {"status": "provider_error", "error": json.dumps(result.get("error"))[:500]}
            return row
        body = result["message"]
    raw, state, reason, (inp, out) = response_fields(provider, body)
    row.update(raw_response=raw, outcome={"status": state, "finish_reason": reason},
               receipts=[{"response_id": body.get("id"), "model": body.get("model"), "usage": body.get("usage")}])
    parsed = original.ranking.parse_ranking_response(raw, task.labels) if state == "completed" else None
    if parsed is not None:
        row.update(parse_success=True, parsed_ranking=parsed)
        row.update(original.ranking.score_ranking(task, parsed))
    # Same checks as the live and recovery runners. Copied, not shared: the saved
    # Nov 3 recovery run pins the recovery file's hash, so moving code into it
    # would break `recovery analyze` on those results.
    cost = (body.get("usage") or {}).get("cost")
    if (not isinstance(inp, int) or not isinstance(out, int)
            or inp > request["input_token_allowance"] or out > request["body"].get(
                "max_output_tokens", request["body"].get("max_tokens", math.inf))
            or (cost is not None and float(cost) > request["reserved_cost_usd"] + 1e-8)):
        # already billed; record it so the audit can see it, nothing to stop
        row["budget_check_error"] = "Missing usage or exceeded reservation."
    return row


def has_results(provider, body):
    if provider == "openai":
        return bool(body.get("output_file_id") or body.get("error_file_id"))
    return bool(body.get("results_url"))


def fetch(output, client):
    output = Path(output)
    with locked(output):
        return _fetch(output, client)


def _fetch(output, client):
    manifest, rows, batches = load(output)
    live = manifest["source_manifest"]
    tasks, _, regenerated = original.prepare(Path(manifest["input_path"]), list(live["models"]),
                                             live["seed"], manifest["limit"])
    # instrument, selected profiles and prompts. Not whole-file hashes of config.py
    # or agents.py, which change whenever any other model is added.
    if any(regenerated[k] != live[k] for k in ("input_sha256", "models", "requests_sha256")):
        raise ValueError("Ranking protocol changed since prepare (instrument, model profiles or prompts).")
    task_by_id = {t.ladder_id: t for t in tasks}
    summaries = []
    for model in live["models"]:
        path = output / model / "responses.jsonl"
        entry = batches.get(model) or {}
        if not path.exists() and entry.get("batch_id"):
            state = status_one(entry, client)
            if not state["done"]:
                print(f"{model}: {state['state']}; not fetched yet.")
                continue
            original.write_json(output / model / "batch_status.json", state["body"])
            if not has_results(entry["provider"], state["body"]):
                # e.g. failed validation: keep the reason, don't write 100 empty requests
                print(f"{model}: batch {state['state']} with no results; see batch_status.json")
                continue
            lines = download(entry, state["body"], client, output / model)
            events = []
            for request in (r for r in rows if r["model_key"] == model):
                task = task_by_id[request["ladder_id"]]
                events.append({"event": "started", "ladder_id": task.ladder_id, "timestamp": entry["submitted_at"]})
                line = lines.get(request["custom_id"])
                row = result_row(request, line, task, entry["batch_id"]) if line else None
                if row is not None:  # None: never ran, stays unresolved like the live runner
                    events.append(row)
            # whole log or nothing, so a bad line can't leave a partial file that later fetches skip
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                                 encoding="utf-8")
            temporary.replace(path)
        if path.exists():
            started, results = original.read_events(path, set(task_by_id))
            summaries.append(original.summarize(model, tasks, started, results))
            original.write_json(output / model / "summary.json", summaries[-1])
    if summaries:
        original.write_json(output / "summary.json", summaries)
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
    return summaries


def status_one(entry, client):
    if entry["provider"] == "openai":
        body = checked(client.get(f"{OPENAI}/v1/batches/{entry['batch_id']}", headers=headers("openai"))).json()
        return {"state": body["status"], "done": body["status"] in OPENAI_DONE, "body": body}
    body = checked(client.get(f"{ANTHROPIC}/v1/messages/batches/{entry['batch_id']}",
                              headers=headers("anthropic"))).json()
    return {"state": body["processing_status"], "done": body["processing_status"] == "ended", "body": body}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "submit", "status", "fetch"])
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--models", nargs="+", help="OpenAI/Anthropic model keys")
    selection.add_argument("--paper-batch-slate", action="store_true",
                           help="The 14 OpenAI and Anthropic configurations from the paper slate")
    parser.add_argument("--input", type=Path, default=original.DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-key-dir", type=Path)
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.mode == "prepare":
        if args.paper_batch_slate:
            models = [m for m in original.PAPER_MODELS
                      if original.MODEL_SPECS[m].model_type in ("openai", "anthropic")]
        elif args.models:
            models = args.models
        else:
            parser.error("prepare needs --models or --paper-batch-slate")
        prepare(output, models, args.input, args.seed, args.limit)
        return 0
    if args.api_key_dir is None:
        parser.error(f"{args.mode} needs --api-key-dir")
    manifest, _, _ = load(output)
    original.load_key_directory(args.api_key_dir.resolve(), manifest["source_manifest"]["models"])
    with httpx.Client(timeout=300.0) as client:
        if args.mode == "submit":
            if args.max_cost_usd is None:
                parser.error("submit needs --max-cost-usd")
            submit(output, args.max_cost_usd, client)
        elif args.mode == "status":
            status(output, client)
        else:
            fetch(output, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
