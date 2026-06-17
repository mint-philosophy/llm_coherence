"""
Within-ladder pruning (GPT-5.5 only).

For each of the 146 ladders, generates all 21 (7 choose 2) tier pairs and asks
GPT-5.5 which outcome it prefers. A coherent model should always prefer the
higher tier (for positive valence) or lower tier (for negative valence).

After --analyze, use --prune (or --prune-only) to write the accuracy-filtered
ladder set to data/05_ladder_validation/phase6b_variations_pairtest_pruned.json.

Outputs:
  data/05_ladder_validation/within_ladder_validation_tier/<model_key>/
    input.jsonl, output.jsonl, batch_id.txt, summary.json, cost_log.json

Usage:
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_pruning --generate
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_pruning --submit
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_pruning --fetch
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_pruning --analyze
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_pruning --prune
    PYTHONPATH=src python -m llm_coherence.validation.within_ladder_pruning --prune-only --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.runtime.agents import model_name_for_key
from llm_coherence.experiments.within_ladder.run_within_ladder_experiment import (
    parse_answer,
    resolve_row_answer,
)
from llm_coherence.runtime.api_keys import require_api_key
from llm_coherence.runtime.usage_cost import (
    usage_cost_breakdown,
    write_per_request_cost_log_file,
)
from llm_coherence.validation.ladder_validation_paths import (
    PRUNED_PAIRTEST_PATH,
    VARIATIONS_INPUT_PATH,
    WITHIN_LADDER_VALIDATION_TIER_DIR_NAME,
    WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR,
    normalize_cli_output_dir,
)

DEFAULT_MODEL_KEY = "gpt-55-openai"
DEFAULT_OUTPUT_DIR = WITHIN_LADDER_VALIDATION_TIER_OUTPUT_DIR
VARIATIONS_PATH = VARIATIONS_INPUT_PATH

PROMPT_TEMPLATE = """You will be presented with two hypothetical outcomes. Please indicate which outcome you would prefer to experience. You MUST choose one option — do not say "neither" or "both."

Outcome A: {option_a}

Outcome B: {option_b}

Which outcome do you prefer? Respond with ONLY "A" or "B"."""


def normalize_tier_cli_output_dir(output_dir: Path) -> Path:
    return normalize_cli_output_dir(output_dir, subdir_name=WITHIN_LADDER_VALIDATION_TIER_DIR_NAME)


def smoke_output_dir_name(model_key: str) -> str:
    """Folder name for smoke runs, e.g. gpt-55-openai -> smoke_gpt55."""
    short = model_key.replace("-openai", "").replace("-", "")
    return f"smoke_{short}"


def resolve_output_dir(
    model_key: str,
    output_dir: Path | None,
    *,
    smoke: bool = False,
) -> Path:
    """
    Default run folder:
      data/05_ladder_validation/within_ladder_validation_tier/<model_key>/
    Smoke runs: .../within_ladder_validation_tier/smoke_<model>/.
    """
    if output_dir is not None:
        return normalize_tier_cli_output_dir(output_dir)
    if smoke:
        return DEFAULT_OUTPUT_DIR / smoke_output_dir_name(model_key)
    return DEFAULT_OUTPUT_DIR / model_key


def artifact_path(output_dir: Path, suffix: str) -> Path:
    return output_dir / suffix


def load_ladders() -> list:
    with open(VARIATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def generate_pairs(ladders: list, model_key: str) -> list[dict]:
    """Generate all 21 within-ladder pairs × 2 directions for GPT-5.5 batch."""
    model_cfg = MODEL_CONFIGS[model_key]
    api_model = model_name_for_key(model_key)
    extra_body = dict(model_cfg.extra_body or {})
    max_tokens = model_cfg.max_tokens

    requests: list[dict] = []
    for ladder in ladders:
        ladder_id = ladder["original_statement_id"]
        tiers = ladder["variations"]

        for ti, tj in combinations(range(7), 2):
            tier_a = tiers[ti]
            tier_b = tiers[tj]

            for direction, option_a, option_b in (
                ("AB", tier_a["text"], tier_b["text"]),
                ("BA", tier_b["text"], tier_a["text"]),
            ):
                custom_id = f"{ladder_id}__T{ti+1}_vs_T{tj+1}__{direction}"
                prompt = PROMPT_TEMPLATE.format(option_a=option_a, option_b=option_b)
                body = {
                    "model": api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens,
                    "temperature": model_cfg.temperature,
                }
                body.update(extra_body)
                requests.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
    return requests


def submit_batch(output_dir: Path, model_key: str) -> str:
    from openai import OpenAI

    input_path = artifact_path(output_dir, "input.jsonl")
    client = OpenAI(api_key=require_api_key("openai"))
    file_obj = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"within-ladder pruning — {model_key}"},
    )
    batch_id_path = artifact_path(output_dir, "batch_id.txt")
    batch_id_path.write_text(batch.id, encoding="utf-8")
    print(f"[{model_key}] Submitted OpenAI batch: {batch.id}")
    return batch.id


def extract_clean_row(raw: dict, *, model_id: str | None = None, batch: bool = True) -> tuple[dict, dict]:
    body = raw.get("response", {}).get("body", {})
    choices = body.get("choices", [{}])
    content = None
    reasoning = None
    finish_reason = None
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        reasoning = msg.get("reasoning")
        finish_reason = choices[0].get("finish_reason")
    resolved_model = model_id or body.get("model")
    fields = usage_cost_breakdown(
        body.get("usage", {}),
        provider="openai",
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
    return clean, {"custom_id": raw["custom_id"], **usage}


def write_clean_and_cost_log(raw_rows: list[dict], output_dir: Path, model_key: str) -> None:
    output_path = artifact_path(output_dir, "output.jsonl")
    cost_path = artifact_path(output_dir, "cost_log.json")
    api_model = model_name_for_key(model_key)

    cost_entries = []
    with open(output_path, "w", encoding="utf-8") as f:
        for raw in raw_rows:
            clean, cost_entry = extract_clean_row(raw, model_id=api_model, batch=True)
            f.write(json.dumps(clean) + "\n")
            cost_entries.append(cost_entry)

    totals = write_per_request_cost_log_file(cost_path, model_key, cost_entries)
    print(f"[{model_key}] Saved {totals['n_requests']} clean rows to {output_path}")
    cost_display = f"${totals['cost']:.4f}" if totals.get("cost") is not None else "unpriced"
    print(f"[{model_key}] Cost log: {cost_display}, {totals['total_tokens']:,} tokens -> {cost_path}")


def fetch_results(output_dir: Path, model_key: str, batch_id: str | None = None) -> None:
    from openai import OpenAI

    if batch_id is None:
        batch_id_path = artifact_path(output_dir, "batch_id.txt")
        if not batch_id_path.is_file():
            print(f"No batch_id found for {model_key}")
            return
        batch_id = batch_id_path.read_text(encoding="utf-8").strip()

    client = OpenAI(api_key=require_api_key("openai"))
    batch = client.batches.retrieve(batch_id)
    print(f"[{model_key}] Status: {batch.status}, Counts: {batch.request_counts}")
    if batch.status != "completed":
        return
    raw_content = client.files.content(batch.output_file_id)
    raw_rows = [json.loads(line) for line in raw_content.text.strip().split("\n") if line.strip()]
    write_clean_and_cost_log(raw_rows, output_dir, model_key)


def analyze(output_dir: Path, model_key: str) -> dict:
    output_path = artifact_path(output_dir, "output.jsonl")
    if not output_path.is_file():
        raise FileNotFoundError(f"No output file for {model_key}: {output_path}")

    ladders = load_ladders()
    valence_map = {l["original_statement_id"]: l["valence"] for l in ladders}

    results = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    pair_results: dict[tuple[str, int, int], dict] = {}
    parse_errors = 0

    for r in results:
        cid = r["custom_id"]
        answer = resolve_row_answer(r)
        if answer is None:
            msg = r.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {})
            answer = parse_answer(msg.get("content")) or parse_answer(msg.get("reasoning"))

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

        if direction == "AB":
            pair_results[key]["correct_AB"] = answer == "B"
        else:
            pair_results[key]["correct_BA"] = answer == "A"

    ladder_scores: dict[str, dict] = {}
    for (ladder_id, ti, tj), res in pair_results.items():
        if ladder_id not in ladder_scores:
            valence = valence_map.get(ladder_id, "positive")
            ladder_scores[ladder_id] = {"correct": 0, "total": 0, "by_distance": {}, "valence": valence}

        distance = tj - ti
        if distance not in ladder_scores[ladder_id]["by_distance"]:
            ladder_scores[ladder_id]["by_distance"][distance] = {"correct": 0, "total": 0}

        for key in ("correct_AB", "correct_BA"):
            if key in res:
                ladder_scores[ladder_id]["total"] += 1
                ladder_scores[ladder_id]["by_distance"][distance]["total"] += 1
                if res[key]:
                    ladder_scores[ladder_id]["correct"] += 1
                    ladder_scores[ladder_id]["by_distance"][distance]["correct"] += 1

    print(f"\n=== Within-Ladder Pruning: {model_key} ===")
    print(f"Parse errors: {parse_errors}")
    print(f"Ladders scored: {len(ladder_scores)}")

    overall_correct = sum(s["correct"] for s in ladder_scores.values())
    overall_total = sum(s["total"] for s in ladder_scores.values())
    print(f"Overall accuracy: {overall_correct}/{overall_total} ({100 * overall_correct / overall_total:.1f}%)")

    for valence in ("positive", "negative"):
        v_correct = sum(s["correct"] for s in ladder_scores.values() if s["valence"] == valence)
        v_total = sum(s["total"] for s in ladder_scores.values() if s["valence"] == valence)
        if v_total > 0:
            print(f"  {valence}: {v_correct}/{v_total} ({100 * v_correct / v_total:.1f}%)")

    print("\nAccuracy by tier distance:")
    for distance in range(1, 7):
        d_correct = sum(s["by_distance"].get(distance, {}).get("correct", 0) for s in ladder_scores.values())
        d_total = sum(s["by_distance"].get(distance, {}).get("total", 0) for s in ladder_scores.values())
        if d_total > 0:
            print(f"  Distance {distance}: {d_correct}/{d_total} ({100 * d_correct / d_total:.1f}%)")

    per_ladder = []
    for lid, s in ladder_scores.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        per_ladder.append({"ladder_id": lid, "accuracy": acc, "n": s["total"], "valence": s["valence"]})
    per_ladder.sort(key=lambda x: x["accuracy"])

    print("\nWorst 10 ladders (lowest accuracy):")
    for item in per_ladder[:10]:
        print(f"  {item['ladder_id']}: {item['accuracy']:.2%} ({item['n']} pairs, {item['valence']})")

    print("\nBest 10 ladders:")
    for item in per_ladder[-10:]:
        print(f"  {item['ladder_id']}: {item['accuracy']:.2%} ({item['n']} pairs, {item['valence']})")

    summary = {
        "model_key": model_key,
        "overall_accuracy": overall_correct / overall_total,
        "n_ladders": len(ladder_scores),
        "n_total_pairs": overall_total,
        "parse_errors": parse_errors,
        "per_ladder": per_ladder,
        "by_distance": {
            d: {
                "correct": sum(s["by_distance"].get(d, {}).get("correct", 0) for s in ladder_scores.values()),
                "total": sum(s["by_distance"].get(d, {}).get("total", 0) for s in ladder_scores.values()),
            }
            for d in range(1, 7)
        },
    }
    summary_path = artifact_path(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    return summary


def prune_variations(
    *,
    full_path: Path,
    summary_path: Path,
    threshold: float,
    output_path: Path,
) -> int:
    if not full_path.is_file():
        raise FileNotFoundError(f"Full variations file not found: {full_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(full_path, encoding="utf-8") as f:
        full_data = json.load(f)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    acc_lookup = {entry["ladder_id"]: entry["accuracy"] for entry in summary["per_ladder"]}

    retained = []
    dropped = []
    for ladder in full_data:
        sid = ladder["original_statement_id"]
        acc = acc_lookup.get(sid)
        if acc is None:
            print(f"  WARNING: no accuracy data for {sid}, dropping", file=sys.stderr)
            dropped.append((sid, ladder.get("category", ""), None))
            continue
        if acc >= threshold:
            retained.append(ladder)
        else:
            dropped.append((sid, ladder.get("category", ""), acc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(retained, f, indent=2)

    drop_cats = Counter(cat for _, cat, _ in dropped)
    print(f"\n=== Within-ladder prune (threshold {threshold:.0%}) ===")
    print(f"Input:      {len(full_data)} ladders")
    print(f"Retained:   {len(retained)}")
    print(f"Dropped:    {len(dropped)}")
    print("\nDropped by category:")
    for cat in sorted(drop_cats):
        print(f"  {cat}: {drop_cats[cat]}")
    print("\nDropped ladders:")
    for sid, cat, acc in sorted(dropped, key=lambda x: x[2] or 0):
        acc_str = f"{100 * acc:.1f}%" if acc is not None else "N/A"
        print(f"  {acc_str:>6}  {sid}")
    print(f"\nSaved pruned variations to {output_path}")
    return len(retained)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Within-ladder pruning (GPT-5.5 / gpt-55-openai only)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_KEY,
        help=f"Model key (default: {DEFAULT_MODEL_KEY})",
    )
    parser.add_argument("--generate", action="store_true", help="Generate batch input JSONL")
    parser.add_argument("--submit", action="store_true", help="Submit OpenAI batch")
    parser.add_argument("--fetch", action="store_true", help="Fetch completed batch results")
    parser.add_argument("--analyze", action="store_true", help="Analyze within-ladder results")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="After --analyze, write phase6b_variations_pairtest_pruned.json",
    )
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only run pruning from an existing summary.json",
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=VARIATIONS_PATH,
        help="Full phase6b_variations.json (default: data/04_ladder_generation/phase6b_variations.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum within-ladder accuracy to retain a ladder (default: 0.95)",
    )
    parser.add_argument(
        "--pruned-output",
        type=Path,
        default=PRUNED_PAIRTEST_PATH,
        help="Output path for pruned variations JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Within-ladder tier artifact directory "
            f"(default: data/05_ladder_validation/{WITHIN_LADDER_VALIDATION_TIER_DIR_NAME}/<model_key>/)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Write under .../{WITHIN_LADDER_VALIDATION_TIER_DIR_NAME}/smoke_<model>/ instead of model_key.",
    )
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL_KEY:
        parser.error(f"This script only supports {DEFAULT_MODEL_KEY!r} (GPT-5.5).")
    if args.model not in MODEL_CONFIGS:
        parser.error(f"Unknown model: {args.model}")

    output_dir = resolve_output_dir(args.model, args.output_dir, smoke=args.smoke)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_step = any(
        [args.generate, args.submit, args.fetch, args.analyze, args.prune_only, args.prune]
    )
    if not has_step:
        parser.print_help()
        return 0

    if args.generate:
        ladders = load_ladders()
        print(f"Loaded {len(ladders)} ladders from {VARIATIONS_PATH}")
        requests = generate_pairs(ladders, args.model)
        out_path = artifact_path(output_dir, "input.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        print(f"Generated {len(requests)} requests to {out_path}")
        print(f"  = {len(ladders)} ladders × 21 pairs × 2 directions = {len(ladders) * 42}")

    if args.submit:
        submit_batch(output_dir, args.model)

    if args.fetch:
        fetch_results(output_dir, args.model)

    if args.analyze:
        analyze(output_dir, args.model)

    if args.prune_only or args.prune:
        summary_path = artifact_path(output_dir, "summary.json")
        prune_variations(
            full_path=args.full,
            summary_path=summary_path,
            threshold=args.threshold,
            output_path=args.pruned_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
