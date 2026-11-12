"""Offline audit; retain strict parsing and report unambiguous formatting recovery separately."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shlex

from llm_coherence.experiments import full_ladder_ranking as ranking
from llm_coherence.experiments.full_ladder_ranking_recovery import eligible_retry, read_log


def remove_line_comments(text):
    result = []; quoted = False; escaped = False; index = 0
    while index < len(text):
        char = text[index]
        if not quoted and text[index:index+2] == "//":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
            continue
        result.append(char)
        if quoted and escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        index += 1
    return "".join(result)


def formatting_only(raw, task):
    """Require seven explicit, unique labels; never infer order from prose."""
    text = raw.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
        if not match:
            return None
        text = match.group(1)
    text = remove_line_comments(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.fullmatch(r'\s*\{\s*"ranking"\s*:\s*\[\s*([A-G](?:\s*,\s*[A-G]){6})\s*\]\s*\}\s*', text)
        if not match:
            return None
        data = {"ranking": [s.strip() for s in match.group(1).split(",")]}
    if not isinstance(data, dict) or set(data) != {"ranking"} or not isinstance(data["ranking"], list):
        return None
    labels = []
    statements = {label: task.statements[index] for label, index in zip(task.labels, task.shuffle_indices)}
    normalize = lambda value: " ".join(value.split())
    for value in data["ranking"]:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if value in task.labels:
            labels.append(value); continue
        match = re.fullmatch(r"\[([A-G])\](?:\s+(.*))?", value, re.S)
        if not match:
            return None
        label, statement = match.groups()
        if statement and normalize(statement) != normalize(statements[label]):
            return None
        labels.append(label)
    return labels if len(labels) == 7 and set(labels) == set(task.labels) else None


def cost_audit(paths, manifest):
    reported = token_upper = unknown_reserved = 0.0
    counts = Counter()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row["event"] != "result":
                continue
            counts["physical_request_results"] += 1
            profile = manifest["models"][row["model_key"]]
            receipts = row.get("receipts") or []
            usage = ((row.get("response") or {}).get("usage")
                     if "response" in row else ((receipts[0].get("usage") or {}) if receipts else {})) or {}
            cost = usage.get("cost")
            if isinstance(cost, (float, int)):
                reported += cost; counts["provider_reported_cost_results"] += 1; continue
            inp = usage.get("prompt_tokens", usage.get("input_tokens"))
            out = usage.get("completion_tokens", usage.get("output_tokens"))
            if isinstance(inp, int) and isinstance(out, int):
                if "prompt_tokens" not in usage and profile["endpoint"]["model_type"] == "anthropic":
                    inp += (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
                rates = profile["price_ceiling_per_million"]
                token_upper += (inp * rates["input"] + out * rates["output"]) / 1_000_000
                counts["token_based_cost_results"] += 1
            else:
                unknown_reserved += row.get("reserved_cost_usd", 0)
                counts["results_without_usage"] += 1
    return dict(counts, provider_reported_usd=reported,
                conservative_token_based_usd=token_upper,
                reserved_for_results_without_usage_usd=unknown_reserved,
                conservative_total_usd=reported + token_upper + unknown_reserved,
                interpretation="Not a billing statement. OpenRouter reported charges are combined with conservative token-based OpenAI/Anthropic estimates and full reservations for results lacking usage.")


def audit(source, recovery, output):
    manifest = json.loads((source / "manifest.json").read_text())
    tasks, _, current = ranking.prepare(ranking.DEFAULT_INPUT, list(manifest["models"]), manifest["seed"])
    if current != manifest:
        raise ValueError("Source protocol changed.")
    allowed = {t.ladder_id for t in tasks}
    pending = json.loads((recovery / "requests.json").read_text())
    new_started, new_results = read_log(recovery / "responses.jsonl", {r["request_id"] for r in pending})
    details, summaries, sources = [], [], {}
    for model in manifest["models"]:
        path = source / model / "responses.jsonl"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        started, results = ranking.read_events(path, allowed)
        for request in pending:
            if request["model_key"] != model or request["request_id"] not in new_started:
                continue
            lid = request["ladder_id"]
            if lid in results and not eligible_retry(results[lid]):
                raise ValueError("Recovery replaced a model outcome.")
            started.add(lid); results.pop(lid, None)
            if request["request_id"] in new_results:
                results[lid] = new_results[request["request_id"]]
        secondary = {}; format_count = 0
        for task in tasks:
            row = results.get(task.ladder_id)
            parsed = None
            if row is not None and row["outcome"]["status"] == "completed":
                parsed = ranking.ranking.parse_ranking_response(row["raw_response"], task.labels)
                if (parsed is not None) != row["parse_success"]:
                    raise ValueError("Stored and reconstructed strict parsing disagree.")
                if parsed is None:
                    parsed = formatting_only(row["raw_response"], task)
            recovered = parsed is not None and not row["parse_success"]
            format_count += recovered
            extra = {}
            if parsed is not None:
                extra = ranking.ranking.score_ranking(task, parsed)
                secondary[task.ladder_id] = dict(row, parse_success=True, **extra)
            elif row is not None:
                secondary[task.ladder_id] = row
            details.append({"model_key": model, "ladder_id": task.ladder_id,
                            "requested": task.ladder_id in started,
                            "status": row["outcome"]["status"] if row else "unrequested_or_unresolved",
                            "strict_parse_valid": row["parse_success"] if row else False,
                            "formatting_only_recovered": recovered,
                            "explicit_ranking": "".join(parsed) if parsed else "",
                            "raw_response_sha256": hashlib.sha256(row["raw_response"].encode()).hexdigest() if row else "",
                            "exact_match_with_formatting_recovery": extra.get("exact_match"),
                            "kendall_tau_with_formatting_recovery": extra.get("kendall_tau")})
        strict = ranking.summarize(model, tasks, started, results)
        relaxed = ranking.summarize(model, tasks, started, secondary)
        strict.update(formatting_only_recovered_count=format_count,
                      valid_including_formatting_recovery=relaxed["n_valid_rankings"],
                      exact_match_fraction_including_formatting_recovery=relaxed["exact_match_fraction_parsed"],
                      mean_tau_including_formatting_recovery=relaxed["mean_kendall_tau"],
                      provider_error_count=sum(r["outcome"]["status"] in ("provider_error", "transport_failure") for r in results.values()),
                      token_capped_count=sum(r["outcome"]["status"] == "token_capped" for r in results.values()))
        summaries.append(strict)
    sources[str(recovery / "responses.jsonl")] = hashlib.sha256((recovery / "responses.jsonl").read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    ranking.write_json(output / "cost_audit.json", cost_audit(sources, manifest))
    for name, rows in [("summary", summaries), ("ladder_details", details)]:
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ranking.write_json(output / "provenance.json", {
        "source": str(source), "recovery": str(recovery), "source_sha256": sources,
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "23 primary configurations, fixed 100 ladders; GLM Base excluded.",
        "formatting_analysis": "Separate post-hoc check: remove // comments outside strings, accept seven bare labels, or strip brackets around labels. A label followed by text is accepted only if the text matches its stimulus. Duplicate/missing labels and token-capped outputs remain invalid.",
        "interpretation": "Agreement of an explicit total ranking with intended tier order; not a transitivity test or estimate of latent coherence.",
        "reproduce_command": shlex.join([".venv/bin/python", "-B", "-m", "llm_coherence.analysis.audit_full_ladder_rankings",
                                         "--source", str(source), "--recovery", str(recovery), "--output", str(output)]),
    })
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source.resolve(), args.recovery.resolve(), args.output.resolve())
