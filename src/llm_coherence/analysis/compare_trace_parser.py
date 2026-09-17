"""Compare final-answer parsing with provisional text annotations, offline."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from llm_coherence.analysis.screen_trace_responses import screen_text
from llm_coherence.analysis.trace_annotation import (
    RUBRIC_SHA256,
    VERSION,
    label_vector,
    load_annotations,
    load_bundle,
    write_json,
    write_jsonl,
)


PARSER_MODES = {"forced-choice": False, "answer-marker": True}
COMPARISON_SPLITS = ("development", "pilot", "triage", "validation", "all")
SELECTIONS = {"A", "B", "conditional_A", "conditional_B"}
MISSING = "missing_annotation"


def comparison_flags(parsed: str, screen: dict, annotation: dict | None) -> list[str]:
    """Identify review candidates, never decide which classifier is correct."""
    if annotation is None:
        return [MISSING]
    final = annotation["final_response"]
    choice = final["expressed_choice"]["label"]
    response = final["comparison_response"]["label"]
    flags = []
    if annotation["needs_review"]:
        flags.append("judge_flagged")
    if parsed == "unparseable" and choice in SELECTIONS:
        flags.append("parser_unparseable_with_annotated_selection")
    if parsed in ("A", "B"):
        if choice in ("A", "B") and choice != parsed:
            flags.append("parser_and_annotated_choices_differ")
        elif choice in ("conditional_A", "conditional_B"):
            flags.append("parser_selection_with_conditional_annotation")
        elif choice not in SELECTIONS:
            flags.append("parser_selection_without_annotated_selection")
        if response in ("rejects", "mixed"):
            flags.append("parser_selection_with_comparison_objection")
    if choice in SELECTIONS:
        if response in ("rejects", "mixed"):
            flags.append("annotated_selection_with_comparison_objection")
        if any(
            evidence["kind"] == "refusal_or_abstention_cue"
            for evidence in screen["evidence"]
        ):
            flags.append("annotated_selection_with_lexical_abstention_cue")
    return flags


def compare_parser(
    bundle: Path,
    prediction_paths: list[Path],
    output: Path,
    *,
    split: str,
    parser_mode: str,
) -> dict:
    """Replay the installed parser without changing votes, labels, or samples."""
    if parser_mode not in PARSER_MODES:
        raise ValueError("Choose forced-choice or answer-marker parser mode")
    if split not in COMPARISON_SPLITS:
        raise ValueError("Unknown parser-comparison split")
    if not prediction_paths:
        raise ValueError("At least one prediction file is required")
    manifest, entries = load_bundle(bundle)
    if split == "all":
        selected = sorted(entries)
    elif split == "development":
        selected = manifest["splits"]["pilot"] + manifest["splits"]["triage"]
    else:
        selected = manifest["splits"][split]
    if not selected:
        raise ValueError("Selected parser-comparison split is empty")
    judges = [load_annotations(path, entries, "model") for path in prediction_paths]
    if any(not judge for judge in judges):
        raise ValueError("Prediction files must not be empty")
    names = [next(iter(judge.values()))["annotator"]["id"] for judge in judges]
    if len(names) != len(set(names)):
        raise ValueError("Judge identities must be distinct")

    # Importing the runtime constructs no API client and reads no credentials.
    from llm_coherence.runtime import utils

    parsed = utils.parse_responses_forced_choice(
        {
            index: [entries[i]["text"]["final_response"]]
            for index, i in enumerate(selected)
        },
        with_reasoning=PARSER_MODES[parser_mode],
        verbose=False,
    )
    parser_info = {
        "callable": "llm_coherence.runtime.utils.parse_responses_forced_choice",
        "module_sha256": hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest(),
        "mode": parser_mode,
        "with_reasoning": PARSER_MODES[parser_mode],
        "choices": ["A", "B"],
        "input_channel": "final_response",
        "historical_trial_outcome_verified": False,
    }
    groups = {i: group for group, ids in manifest["splits"].items() for i in ids}
    records = []
    for index, entry_id in enumerate(selected):
        entry = entries[entry_id]
        final_screen = screen_text(entry["text"]["final_response"] or "")
        # Retain reasoning cues separately; they never affect parser replay.
        reasoning = entry["text"]["visible_reasoning"]
        reasoning_screen = screen_text(reasoning) if reasoning else None
        for name, judge in zip(names, judges):
            prediction = judge.get(entry_id)
            annotation = prediction["annotation"] if prediction else None
            records.append(
                {
                    "entry_id": entry_id,
                    "entry_sha256": entry["entry_sha256"],
                    "source_path": entry["source_path"],
                    "source_sha256": entry["source_sha256"],
                    "source_line": entry["source_line"],
                    "split": groups.get(entry_id, "unsampled"),
                    "judge": name,
                    "parser_result": parsed[index][0],
                    "annotation_status": "available" if prediction else MISSING,
                    "semantic_labels": label_vector(prediction) if prediction else None,
                    "annotation": annotation,
                    "final_response_screen": final_screen,
                    "visible_reasoning_screen": reasoning_screen,
                    "review_flags": comparison_flags(
                        parsed[index][0], final_screen, annotation
                    ),
                    "text": entry["text"],
                }
            )
    reports = []
    for name, predictions in zip(names, judges):
        own = [row for row in records if row["judge"] == name]
        counts = Counter(row["annotation_status"] for row in own)
        fields = list(label_vector(next(iter(predictions.values()))))
        reports.append(
            {
                "judge": name,
                "selected_entries": len(own),
                "annotations_available": counts["available"],
                "annotations_missing": counts[MISSING],
                "predictions_outside_selected_split": len(
                    set(predictions) - set(selected)
                ),
                "review_entries": sum(bool(row["review_flags"]) for row in own),
                "cross_tabs": {
                    field: {
                        outcome: dict(
                            sorted(
                                Counter(
                                    row["semantic_labels"][field]
                                    if row["semantic_labels"]
                                    else MISSING
                                    for row in own
                                    if row["parser_result"] == outcome
                                ).items()
                            )
                        )
                        for outcome in ("A", "B", "unparseable")
                    }
                    for field in fields
                },
                "semantic_accuracy": None,
            }
        )
    queue = [row for row in records if row["review_flags"]]
    report = {
        "comparison_version": "1.0",
        "rubric_version": VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "unit": "trace_log_entry",
        "unique_trial_verified": False,
        "split": split,
        "selected_entries": len(selected),
        "selected_entry_ids": selected,
        "selected_split_counts": dict(
            sorted(Counter(groups.get(i, "unsampled") for i in selected).items())
        ),
        "includes_validation_entries": bool(
            set(selected) & set(manifest["splits"]["validation"])
        ),
        "parser": parser_info,
        # Count each selected log entry once, even when comparing several judges.
        "parser_counts": dict(
            sorted(Counter(parsed[index][0] for index in range(len(selected))).items())
        ),
        "judges": reports,
        "review_queue_records": len(queue),
        "human_validation_performed": False,
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in [bundle / "manifest.json", *prediction_paths]
        ],
        "limitations": [
            "Parser replay uses the installed implementation and caller-selected mode; it does not prove a historical retained vote.",
            "Unparseable means no label was extracted under that parser mode, not an established refusal.",
            "Semantic labels and lexical cues remain provisional; cross-tabs are not accuracy measurements.",
            "A choice and a personal-preference disclaimer or framing objection can coexist without incoherence.",
            "Missing annotations remain missing, including when the original answer parses successfully.",
            "Selected log entries may contain retries; targeted development counts are not population refusal rates.",
            "No original vote, semantic label, rubric, or validation sample is changed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "entries.jsonl", records)
    write_jsonl(output / "review_queue.jsonl", queue)
    fields = list(label_vector(next(iter(judges[0].values()))))
    columns = [
        "entry_id",
        "source_path",
        "source_line",
        "split",
        "judge",
        "parser_result",
        "annotation_status",
        *fields,
        "judge_needs_review",
        "review_flags",
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {key: row[key] for key in columns[:7]}
                | (row["semantic_labels"] or {field: "" for field in fields})
                | {
                    "judge_needs_review": row["annotation"]["needs_review"]
                    if row["annotation"]
                    else "",
                    "review_flags": ";".join(row["review_flags"]),
                }
            )
    # Completion marker is last; failed exports must not be treated as complete.
    write_json(output / "report.json", report)
    return report
