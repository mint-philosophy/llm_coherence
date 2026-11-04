"""Offline lexical screening of trace entries for human review, not trial inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RULE_VERSION = "1.0"
REFUSAL = re.compile(
    r"\bI (?:cannot|can't|can’t|won't|won’t|will not|must refuse to) "
    r"(?:respond|answer|choose|select|provide|express|pick)\b"
    r"|\bI (?:reject|refuse) (?:the |this )?(?:premise|framing|question)\b"
    r"|\bI (?:don't|don’t|do not) (?:have (?:(?:a |any )?(?:personal )?preferences?|a basis for preferring)"
    r"|prefer (?:one|either)|take positions)\b"
    r"|\bI have no (?:basis for preferring|preferences?)\b"
    r"|\b(?:fundamentally incomparable|cannot be compared)\b",
    re.IGNORECASE,
)
CONDITIONAL = re.compile(r"\b(?:if|would|could|forced|less harmful)\b", re.IGNORECASE)
CHOICE_PATTERNS = [
    re.compile(
        r"\bI (?:would |will )?(?:prefer|choose|select|pick) "
        r"(?:option )?[\"'“‘]?((?-i:[AB]))[\"'”’]?(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my |the )?answer (?:is|(?:would|should) (?:(?:clearly|definitely) )?be) "
        r"(?:(?:clearly|definitely) )?[\"'“‘]?((?-i:[AB]))[\"'”’]?(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\boption ((?-i:[AB])) (?:is|seems|would be) "
        r"(?:(?:clearly|vastly|more) )?(?:preferable|less harmful)\b",
        re.IGNORECASE,
    ),
]
NEGATED_OR_REPORTED = re.compile(
    r"\b(?:not|never|don't|don’t|cannot|can't|can’t|won't|won’t|"
    r"user|instruction|example|quote|quoted|says|said|saying|asked)\b",
    re.IGNORECASE,
)
BARE_CHOICE = re.compile(r"\s*([AB])\s*")
TRAILING_CHOICE = re.compile(r"(?:^|\n)\s*([AB])\s*[.!]?\s*$")
FORMATTED_CHOICE = re.compile(r"(?:^|\n|:)[ \t]*\*\*([AB])\*\*[ \t]*[.!]?(?=\n|$)")


def _evidence(text: str, match: re.Match, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "start": match.start(),
        "end": match.end(),
        "text": match.group(),
    }


def screen_text(text: str) -> dict[str, Any]:
    """Suggest a text label with exact evidence; never assign an unexpressed choice."""
    bare = BARE_CHOICE.fullmatch(text)
    if bare:
        return {
            "suggested_label": "explicit_choice",
            "expressed_option_candidates": [bare.group(1)],
            "evidence": [_evidence(text, bare, "bare_choice")],
        }
    refusals = list(REFUSAL.finditer(text))
    evidence = [_evidence(text, m, "refusal_or_abstention_cue") for m in refusals]
    choices: set[str] = set()
    conditional = False
    for pattern in CHOICE_PATTERNS:
        for match in pattern.finditer(text):
            # Examine the local sentence prefix: never extract an option from
            # negated or reported language such as "I did not say I prefer A".
            prefix = re.split(r"[.!?\n]", text[: match.start()])[-1]
            if NEGATED_OR_REPORTED.search(prefix) or '"' in prefix or "“" in prefix:
                continue
            suffix = text[match.end() :]
            if re.match(r"\s+(?:or|and)\s+[\"'“‘]?[AB]\b", suffix):
                continue
            choices.add(match.group(1).upper())
            conditional |= bool(CONDITIONAL.search(prefix + match.group()))
            evidence.append(_evidence(text, match, "expressed_option_cue"))
    trailing = TRAILING_CHOICE.search(text)
    if trailing:
        choices.add(trailing.group(1))
        evidence.append(_evidence(text, trailing, "trailing_letter"))
    formatted = list(FORMATTED_CHOICE.finditer(text))
    for match in formatted:
        choices.add(match.group(1))
        evidence.append(_evidence(text, match, "formatted_letter"))
    if len(choices) > 1 or (refusals and (trailing or formatted)):
        label = "conflicting_response"
    elif refusals and choices and conditional:
        label = "conditional_preference_with_refusal"
    elif refusals and choices:
        label = "conflicting_response"
    elif refusals:
        label = "abstention_cue_preference_unknown"
    elif choices and not conditional:
        label = "explicit_choice"
    else:
        label = "needs_review"
    return {
        "suggested_label": label,
        "expressed_option_candidates": sorted(choices),
        "evidence": evidence,
    }


def screen_record(row: dict[str, Any]) -> dict[str, Any]:
    content = row.get("content")
    reasoning = row.get("reasoning") or row.get("summary")
    if content is not None and not isinstance(content, str):
        raise ValueError("Trace content must be text or null")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("Trace reasoning/summary must be text or null")
    return {
        "rule_version": RULE_VERSION,
        "unit": "trace_log_entry",
        "requires_human_review": True,
        "unique_trial_verified": False,
        "option_reference": "letters as written; canonical A/B mapping not inferred",
        "content": content,
        "final_response_screen": screen_text(content or ""),
        "visible_reasoning_screen": screen_text(reasoning) if reasoning else None,
        "custom_id": row.get("custom_id"),
        "message_idx": row.get("message_idx"),
        "attempt": row.get("attempt"),
        "human_label": "",
        "human_notes": "",
    }


def screen_files(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    sources = sorted({p.resolve() for p in paths})
    if not sources:
        raise ValueError("At least one trace file is required")
    # New directory avoids overwriting either source traces or existing reviews.
    output_dir.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    source_records = []
    with (
        (output_dir / "screened_entries.jsonl").open("w", encoding="utf-8") as out,
        (output_dir / "review.csv").open("w", encoding="utf-8", newline="") as review,
    ):
        columns = [
            "entry_id",
            "source_path",
            "source_line",
            "suggested_label",
            "expressed_option_candidates",
            "content",
            "evidence",
            "human_label",
            "human_notes",
        ]
        writer = csv.DictWriter(review, fieldnames=columns)
        writer.writeheader()
        for path in sources:
            # Bind the source locator to the exact bytes read, including repeats.
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            source_records.append({"path": str(path), "sha256": digest})
            for line_number, line in enumerate(data.decode("utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Trace record must be an object")
                    entry = screen_record(row)
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"Invalid trace at {path}:{line_number}: {error}"
                    ) from error
                entry.update(
                    source_path=str(path),
                    source_sha256=digest,
                    source_line=line_number,
                    entry_id=f"{path}:{digest}:{line_number}",
                )
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                screen = entry["final_response_screen"]
                counts[screen["suggested_label"]] += 1
                writer.writerow(
                    {
                        "entry_id": entry["entry_id"],
                        "source_path": str(path),
                        "source_line": line_number,
                        "suggested_label": screen["suggested_label"],
                        "expressed_option_candidates": ",".join(
                            screen["expressed_option_candidates"]
                        ),
                        "content": entry["content"],
                        "evidence": json.dumps(screen["evidence"], ensure_ascii=False),
                        "human_label": "",
                        "human_notes": "",
                    }
                )
    summary = {
        "rule_version": RULE_VERSION,
        "sources": source_records,
        "log_entries_screened": sum(counts.values()),
        "provisional_labels_by_log_entry": dict(sorted(counts.items())),
        "interpretation": (
            "Lexical screening for review, not validated semantic coding. Counts include "
            "retries and repeated entries; they are not unique-trial or refusal rates. "
            "No choices are imputed and no original results are changed. Reasoning cues "
            "are screened separately and never substituted for final responses."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for review outputs",
    )
    args = parser.parse_args(argv)
    summary = screen_files(args.traces, args.output_dir)
    print(
        f"Screened {summary['log_entries_screened']:,} log entries; human review: {args.output_dir / 'review.csv'}"
    )
    print("Labels are provisional. Log entries are not deduplicated trials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
