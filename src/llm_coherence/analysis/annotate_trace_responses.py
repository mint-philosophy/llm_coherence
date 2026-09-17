"""Prepare, annotate, and validate visible-response labels across model traces."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from llm_coherence.analysis.trace_annotation import (
    FIELDS,
    JUDGE_PROTOCOL,
    RUBRIC,
    RUBRIC_SHA256,
    VERSION,
    digest,
    decode_judge_annotation,
    evidence_passages,
    evaluate,
    load_bundle,
    prepare,
    write_json,
)


def judge_messages(entry: dict) -> list[dict]:
    """Hide model/source identities and lexical suggestions from the judge."""
    blocks = ["Annotate these two separate text channels using the system rubric."]
    for channel, value in entry["text"].items():
        text = value or ""
        boundary = f"{channel}_{digest(text)[:24]}"
        while boundary in text:
            boundary += "_"
        passages = evidence_passages(text, channel)
        rendered = "\n".join(f"[{p['id']}]\n{p['text']}" for p in passages)
        blocks.append(
            f"Channel: {channel}; characters: {len(text)}; "
            f"present: {bool(text.strip())}\nBEGIN_{boundary}\n{rendered}\nEND_{boundary}"
        )
    return [
        {
            "role": "system",
            "content": RUBRIC
            + "\nAllowed labels:\n"
            + json.dumps(FIELDS)
            + "\n\n"
            + JUDGE_PROTOCOL,
        },
        {
            "role": "user",
            "content": "\n\n".join(blocks),
        },
    ]


async def annotate(
    bundle: Path,
    output: Path,
    *,
    model: str,
    split: str,
    max_entries: int,
    max_tokens: int,
    execute: bool = False,
    agent_factory: Callable | None = None,
) -> dict:
    manifest, entries = load_bundle(bundle)
    if split not in ("all", "pilot", "validation", "triage"):
        raise ValueError("Unknown annotation split")
    if not model.strip() or max_entries < 1 or max_tokens < 1:
        raise ValueError("Model and positive entry/token limits are required")
    ids = sorted(entries) if split == "all" else manifest["splits"][split]
    if not ids or len(ids) > max_entries:
        raise ValueError(
            "Selected split is empty or exceeds max_entries; no requests sent"
        )
    messages = [judge_messages(entries[i]) for i in ids]
    plan = {
        "version": VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "model": model,
        "split": split,
        "selected_entries": len(ids),
        "max_entries": max_entries,
        "max_output_tokens_per_request": max_tokens,
        "input_characters": sum(len(m[1]["content"]) for m in messages),
        "temperature": 0.0,
        "max_attempts_per_entry": 3,
        "retry_policy": "transport_only",
        "max_consecutive_invalid_annotations": 3,
        "unit": "trace_log_entry",
        "unique_trial_verified": False,
        "bundle_manifest_sha256": digest(manifest),
        "entry_ids": ids,
        "network_execution_requested": execute,
    }
    if not execute:
        return plan
    # Import/API-key checks happen only when execution is explicitly selected.
    if agent_factory is None:
        from llm_coherence.runtime.agents import create_agent
        from llm_coherence.config import canonical_model_key
        from llm_coherence.runtime.agents import MODEL_SPECS

        spec = MODEL_SPECS.get(canonical_model_key(model))
        if spec is None or spec.model_type == "vllm_base_model_logprobs":
            raise ValueError("Choose an API-backed model key supported by the runtime")
        agent_factory = create_agent
    if output.exists():
        raise FileExistsError(output)
    agent = agent_factory(
        model_key=model,
        temperature=0.0,
        max_tokens=max_tokens,
        concurrency_limit=1,
        max_retries=3,
        base_timeout=60.0,
        retry_transport_only=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", plan)
    valid, invalid, consecutive_invalid = 0, 0, 0
    try:
        with (
            (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions,
            (output / "attempts.jsonl").open("w", encoding="utf-8") as attempts,
        ):
            for entry_id, request in zip(ids, messages):
                entry = entries[entry_id]
                responses = await agent.async_completions([request], verbose=False)
                if len(responses) != 1:
                    raise ValueError("Runtime returned an unexpected response count")
                response = responses[0]
                outcomes = getattr(agent, "last_completion_outcomes", [])
                outcome = outcomes[0] if outcomes else {}
                finish = outcome.get("finish_reason")
                provider_status = outcome.get("status")
                capped = provider_status == "token_capped" or str(
                    finish
                ).strip().lower() in (
                    "length",
                    "max_tokens",
                    "max_output_tokens",
                    "max_completion_tokens",
                )
                record = {
                    "entry_id": entry_id,
                    "entry_sha256": entry["entry_sha256"],
                    "rubric_sha256": RUBRIC_SHA256,
                    "annotator": {"kind": "model", "id": model},
                    "request_sha256": digest(request),
                }
                attempt: dict[str, Any] = {
                    **record,
                    "raw_response": outcome.get("raw_response", response),
                    "finish_reason": finish,
                    "provider_outcome": outcome,
                }
                try:
                    if capped:
                        raise ValueError("Judge response reached token cap")
                    if provider_status == "empty_response":
                        raise ValueError("Judge returned empty content")
                    if response is None:
                        raise ValueError(
                            "No usable judge response; inspect provider outcome"
                        )
                    annotation, evidence_spans = decode_judge_annotation(
                        json.loads(response), entry
                    )
                except (ValueError, TypeError) as exc:
                    attempt.update(
                        status="invalid_annotation", validation_error=str(exc)
                    )
                    invalid += 1
                    consecutive_invalid += 1
                else:
                    consecutive_invalid = 0
                    record["annotation"] = annotation
                    record["evidence_spans"] = evidence_spans
                    predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions.flush()
                    attempt["status"] = "valid_annotation"
                    valid += 1
                attempts.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                attempts.flush()
                if consecutive_invalid >= 3:
                    write_json(
                        output / "incomplete.json",
                        {
                            "reason": "three_consecutive_invalid_annotations",
                            "attempted": valid + invalid,
                            "valid": valid,
                            "invalid": invalid,
                            "usage": getattr(agent, "usage_log", []),
                        },
                    )
                    raise RuntimeError(
                        "Three consecutive invalid annotations; inspect attempts before resuming"
                    )
                if (
                    response is None
                    and not capped
                    and provider_status != "empty_response"
                ):
                    # Stop infrastructure/configuration failures instead of spending
                    # through the rest of the corpus on unusable responses.
                    raise RuntimeError(
                        "Judge returned no response; run is incomplete (see attempts.jsonl)"
                    )
    finally:
        if agent_factory.__module__ == "llm_coherence.runtime.agents":
            from llm_coherence.runtime.agents import close_api_async_clients

            await close_api_async_clients()
    summary = {
        **plan,
        "valid_annotations": valid,
        "invalid_annotations": invalid,
        "execution_complete": True,
        "annotation_coverage_complete": invalid == 0,
        "semantic_accuracy_validated": False,
        "usage": getattr(agent, "usage_log", []),
        "interpretation": "Automated suggestions only; invalid outputs are missing labels, not refusals.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "prepare", help="Inventory traces and freeze review samples offline"
    )
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pilot-size", type=int, default=20)
    p.add_argument("--validation-size", type=int, default=100)
    p.add_argument("--triage-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260917)
    p = commands.add_parser(
        "annotate", help="Plan a judge run; --execute sends API requests"
    )
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model",
        required=True,
        help="Existing API model key from project configuration",
    )
    p.add_argument(
        "--split", choices=("all", "pilot", "validation", "triage"), required=True
    )
    p.add_argument("--max-entries", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--execute", action="store_true")
    p = commands.add_parser(
        "evaluate", help="Describe predictions and validate against human labels"
    )
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    p.add_argument("--human", type=Path, nargs=2)
    p.add_argument("--adjudications", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.inputs,
            args.output_dir,
            pilot_size=args.pilot_size,
            validation_size=args.validation_size,
            triage_size=args.triage_size,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "entry_count": result["entry_count"],
                    "source_files": len(result["sources"]),
                    "sample_sizes": {k: len(v) for k, v in result["splits"].items()},
                    "output_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
    elif args.command == "annotate":
        result = asyncio.run(
            annotate(
                args.bundle,
                args.output_dir,
                model=args.model,
                split=args.split,
                max_entries=args.max_entries,
                max_tokens=args.max_tokens,
                execute=args.execute,
            )
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("entry_ids", "usage")},
                indent=2,
            )
        )
    else:
        result = evaluate(
            args.bundle,
            args.predictions,
            args.output_dir,
            args.human,
            args.adjudications,
        )
        print(
            json.dumps(
                {
                    "reference_status": result["reference_status"],
                    "pending_adjudication": len(result["pending_adjudication"]),
                    "review_queue_entries": result["review_queue_entries"],
                    "report": str(args.output_dir / "report.json"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
