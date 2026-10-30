"""Recover archived per-order choices without calling any inference service.

Only choices that reconcile to the selected result's A/B totals are exposed.
Unkeyed traces and conflicting duplicate trials never become additional trials.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from llm_coherence.runtime.utils import parse_responses_forced_choice
from llm_coherence.experiments.ladder_statement_pair.openai_batch_runner import (
    decode_custom_id, _response_text_and_diagnostics,
)
from .rebuttal_inputs import digest, read_verified_json

ABSENT, UNPARSEABLE, CONFLICT = -2, -1, -3
LEGACY_BATCH_DIRS = {
    "gpt-54": "batch_processing_neurips_std",
    "gpt-54-thinking": "batch_processing_neurips_std_on",
    "gpt-54-mini": "batch_processing_neurips_mini",
    "gpt-54-mini-thinking": "batch_processing_neurips_mini_on",
    "gpt-54-nano": "batch_processing_neurips",
    "gpt-54-nano-thinking": "batch_processing_neurips_nano_on",
}
LEGACY_ID = re.compile(r"(.+)__(\d+)__trial(\d+)__dir([AB])$")


def parse_labels(texts, with_reasoning, direction):
    labels = parse_responses_forced_choice({0: texts}, with_reasoning=with_reasoning, verbose=False)[0]
    target = "A" if direction == 0 else "B"
    return np.array([UNPARSEABLE if x not in {"A", "B"} else int(x == target) for x in labels], dtype=np.int8)


def reconcile(labels, preference):
    """Require all slots documented and counts exactly equal the selected data."""
    return (not np.any(labels < UNPARSEABLE) and
            int((labels == 1).sum()) == preference["count_prefer_a"] and
            int((labels == 0).sum()) == preference["count_prefer_b"])


def iter_hashed_jsonl(path, provenance):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line in stream:
            checksum.update(line)
            if line.strip():
                yield json.loads(line)
    provenance[str(Path(path).resolve())] = checksum.hexdigest()


def merge_trial(destination, index, value):
    """Same-choice duplicates collapse; disagreements remain ambiguous."""
    old = destination[index]
    if old == ABSENT:
        destination[index] = value
    elif old != value:
        destination[index] = CONFLICT


def keyed_sidecar(path, raw, provenance):
    count, trials = len(raw["preferences"]), raw["config"]["num_trials"]
    slots = np.full((count, 2, trials), ABSENT, dtype=np.int8)
    custom_ids, skipped = {}, Counter()
    for row in iter_hashed_jsonl(path, provenance):
        if not all(k in row for k in ("custom_id", "pair_idx", "trial", "direction", "content")):
            skipped["unkeyed_rows"] += 1
            continue
        pair, trial = row["pair_idx"], row["trial"]
        direction = {"A": 0, "AB": 0, "ab": 0, "B": 1, "BA": 1, "ba": 1}.get(row["direction"])
        if (not row["custom_id"] or direction is None or not isinstance(pair, int) or
                not isinstance(trial, int) or not 0 <= pair < count or not 0 <= trial < trials):
            raise ValueError(f"Invalid keyed trial metadata: {path}")
        if row.get("test_name", raw["config"]["test_name"]) != raw["config"]["test_name"]:
            raise ValueError(f"Wrong ladder in sidecar: {path}")
        key = (pair, direction, trial)
        cid = row["custom_id"]
        if cid in custom_ids and custom_ids[cid] != key:
            raise ValueError(f"Custom ID maps to multiple trials: {path}")
        custom_ids[cid] = key
        value = parse_labels([row["content"]], raw["config"]["with_reasoning"], direction)[0]
        merge_trial(slots, key, value)
    return slots, dict(skipped)


def modern_batch_slots(run_dir, raws, provenance):
    manifest_path, jobs_path = run_dir / "batch_manifest.json", run_dir / "batch_jobs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    provenance[str(manifest_path)] = digest(manifest_path)
    provenance[str(jobs_path)] = digest(jobs_path)
    lookup = {raw["config"]["test_name"]: (li, raw) for li, raw in raws.items()}
    sets, result = {}, {}
    for si, record in enumerate(manifest["sets"]):
        if record["test_name"] not in lookup:
            continue
        li, raw = lookup[record["test_name"]]
        if raw["config"]["model_key"] != manifest["model_key"]:
            raise ValueError("Batch model does not match selected results")
        if raw["metadata"].get("batch_run_id") != run_dir.name:
            continue
        if record["comparison_file_sha256"] != raw["metadata"]["comparison_file_sha256"]:
            raise ValueError("Batch comparison fingerprint mismatch")
        cp = (run_dir / record["comparison_path"]).resolve()
        if digest(cp) != record["comparison_file_sha256"]:
            raise ValueError(f"Changed batch comparison inputs: {cp}")
        provenance[str(cp)] = digest(cp)
        comparisons = json.loads(cp.read_text(encoding="utf-8"))["comparisons"]
        if [(p["outcome_a"], p["outcome_b"]) for p in comparisons] != [
                (p["outcome_a"], p["outcome_b"]) for p in raw["preferences"]]:
            raise ValueError("Batch pair ordering differs from saved results")
        trials = raw["config"]["num_trials"]
        if trials != manifest["num_trials"] or not manifest["include_flipped"]:
            raise ValueError("Batch trial protocol mismatch")
        sets[si] = li
        result[li] = np.full((len(comparisons), 2, trials), ABSENT, dtype=np.int8)
    for shard in manifest["shards"]:
        p = run_dir / shard["input_file"]
        actual = digest(p)
        if actual != shard["sha256"]:
            raise ValueError(f"Batch request file hash mismatch: {p}")
        provenance[str(p)] = actual
    for job in jobs["jobs"]:
        for name in dict.fromkeys(x for x in (job.get("output_file"), job.get("error_file")) if x):
            for row in iter_hashed_jsonl(run_dir / name, provenance):
                si, pair, direction, trial = decode_custom_id(row["custom_id"])
                if si not in sets:
                    continue
                li, di = sets[si], int(direction == "ba")
                raw = raws[li]
                response = row.get("response") or {}
                if response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
                    continue
                d = _response_text_and_diagnostics(response["body"])
                text = d["text"] if d["response_status"] == "completed" and not d["incomplete_reason"] else None
                value = parse_labels([text], raw["config"]["with_reasoning"], di)[0]
                # Exact duplicate choices collapse. Conflicting attempts are
                # excluded, even if the production aggregator selected one.
                merge_trial(result[li], (pair, di, trial), value)
    return result


def prompt_matches(body, preference, direction):
    """Verify both displayed outcome texts in an archived legacy request."""
    text = body.get("input")
    if not isinstance(text, str):
        users = [m.get("content") for m in body.get("messages", []) if m.get("role") == "user"]
        text = users[-1] if users else None
    if not isinstance(text, str) or "Option A:\n" not in text or "\n\nOption B:\n" not in text:
        return False
    a, rest = text.split("Option A:\n", 1)[1].split("\n\nOption B:\n", 1)
    b = rest.split("\n\n", 1)[0]
    expected = [preference["outcome_a"]["text"], preference["outcome_b"]["text"]]
    if direction:
        expected.reverse()
    return [a, b] == expected


def legacy_batch_slots(folder, raws, provenance):
    lookup = {raw["config"]["test_name"]: (li, raw) for li, raw in raws.items()}
    model_ids = {raw.get("metadata", {}).get("model_name_full") for raw in raws.values()}
    model_ids.discard(None)
    allowed, conflicts = set(), set()
    result = {li: np.full((len(raw["preferences"]), 2, raw["config"]["num_trials"]), ABSENT, dtype=np.int8)
              for li, raw in raws.items()}
    for p in sorted(folder.rglob("*.jsonl")):
        if "output" in p.name:
            continue
        for row in iter_hashed_jsonl(p, provenance):
            match = LEGACY_ID.fullmatch(str(row.get("custom_id", "")))
            if not match or match[1] not in lookup or "body" not in row:
                continue
            li, raw = lookup[match[1]]
            pair, trial, di = int(match[2]), int(match[3]), int(match[4] == "B")
            if pair >= len(raw["preferences"]) or trial >= raw["config"]["num_trials"]:
                raise ValueError("Legacy request trial index out of range")
            key = (li, pair, di, trial)
            body = row["body"]
            expected_model = raw.get("metadata", {}).get("model_name_full")
            if expected_model is None and len(model_ids) == 1:
                expected_model = next(iter(model_ids))
            aliases = {expected_model, re.sub(r"-\d{4}-\d{2}-\d{2}$", "", expected_model or "")}
            if (body.get("model") not in aliases or
                    not prompt_matches(body, raw["preferences"][pair], di)):
                conflicts.add(key)
            else:
                allowed.add(key)
    allowed -= conflicts
    for p in sorted(folder.rglob("*output*.jsonl")):
        for row in iter_hashed_jsonl(p, provenance):
            match = LEGACY_ID.fullmatch(str(row.get("custom_id", "")))
            if not match or match[1] not in lookup:
                continue
            li, raw = lookup[match[1]]
            pair, trial, di = int(match[2]), int(match[3]), int(match[4] == "B")
            if (li, pair, di, trial) not in allowed:
                continue
            response = row.get("response") or {}
            if response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
                continue
            text = _response_text_and_diagnostics(response["body"])["text"]
            # These legacy batch_smoke requests ask for a bare A/B even with
            # internal reasoning enabled. Its with_reasoning metadata does not
            # imply the later "Answer: A" output format. Use unambiguous letter
            # parsing, then require exact reconciliation to the archived totals.
            value = parse_labels([text], False, di)[0]
            merge_trial(result[li], (pair, di, trial), value)
    return result


def recover_trials(panel, sources, *, legacy_root=None):
    reader_hash = digest(Path(__file__))
    trials = panel.trials_per_direction
    labels = np.full(panel.probabilities.shape + (2, trials), ABSENT, dtype=np.int8)
    source_codes = np.zeros(panel.probabilities.shape, dtype=np.int8)
    references = {ref: i for i, ref in enumerate(panel.reference_ids)}
    raws, paths, coordinates = {}, {}, {}
    provenance, audit = {}, Counter()
    details = []
    for li, lid in enumerate(panel.ladder_ids):
        record = sources[lid]
        raw = read_verified_json(record)
        raws[li], paths[li] = raw, Path(record["path"])
        provenance[str(paths[li])] = record["sha256"]
        coordinates[li] = [(li, references[p["outcome_b"]["comparison_id"]], p["outcome_a"]["tier"] - 1)
                           for p in raw["preferences"]]

    def accept(li, slots, source):
        for pi, pref in enumerate(raws[li]["preferences"]):
            key = coordinates[li][pi]
            if source_codes[key]:
                continue
            if reconcile(slots[pi], pref):
                labels[key], source_codes[key] = slots[pi], source
                audit[f"cells_source_{source}"] += 1
            elif np.any(slots[pi] != ABSENT):
                audit[f"unreconciled_candidates_source_{source}"] += 1

    for li, raw in raws.items():
        slots = np.full((len(raw["preferences"]), 2, trials), ABSENT, dtype=np.int8)
        for pi, pref in enumerate(raw["preferences"]):
            if "raw_responses_original" in pref and "raw_responses_flipped" in pref:
                for di, field in enumerate(("raw_responses_original", "raw_responses_flipped")):
                    if len(pref[field]) != trials:
                        raise ValueError("Embedded trial array has wrong length")
                    slots[pi, di] = parse_labels(pref[field], raw["config"]["with_reasoning"], di)
        accept(li, slots, 1)
        if source_codes[li].all():
            continue
        if legacy_root is not None:
            name = raw["config"]["test_name"]
            prefix = Path(legacy_root) / "results" / name / f"{name}_{panel.model_key}"
            legacy_result = prefix.with_name(prefix.name + "_results.json")
            response = prefix.with_name(prefix.name + "_responses.jsonl")
            if legacy_result.exists() and response.exists() and digest(legacy_result) == digest(paths[li]):
                provenance[str(legacy_result)] = digest(legacy_result)
                slots, skipped = keyed_sidecar(response, raw, provenance)
                audit.update(skipped)
                accept(li, slots, 2)
        if source_codes[li].all():
            continue
        trace = paths[li].parent / "reasoning_traces.jsonl"
        if trace.exists():
            slots, skipped = keyed_sidecar(trace, raw, provenance)
            audit.update(skipped)
            accept(li, slots, 5)
    run_dirs = {}
    for li, raw in raws.items():
        run_id = raw.get("metadata", {}).get("batch_run_id")
        if not source_codes[li].all() and run_id:
            run_dir = paths[li].parent.parent / "batch_runs" / run_id
            if (run_dir / "batch_manifest.json").exists():
                run_dirs.setdefault(run_dir, {})[li] = raw
    for run_dir, chosen in run_dirs.items():
        for li, slots in modern_batch_slots(run_dir, chosen, provenance).items():
            accept(li, slots, 3)
    if legacy_root and panel.model_key in LEGACY_BATCH_DIRS and not source_codes.all():
        folder = Path(legacy_root) / LEGACY_BATCH_DIRS[panel.model_key]
        chosen = {li: raw for li, raw in raws.items() if not source_codes[li].all()}
        if folder.exists():
            for li, slots in legacy_batch_slots(folder, chosen, provenance).items():
                accept(li, slots, 4)
    for li, lid in enumerate(panel.ladder_ids):
        details.append({"ladder_id": lid, "reconciled_cells": int((source_codes[li] > 0).sum()),
                        "expected_cells": int(source_codes[li].size)})
    audit.update({"cells_total": int(source_codes.size), "reconciled_cells": int((source_codes > 0).sum()),
                  "cells_without_verified_trials": int((source_codes == 0).sum())})
    return labels, source_codes, {"model_key": panel.model_key, "coverage": dict(audit), "ladders": details,
                                  "ladder_ids": panel.ladder_ids, "reference_ids": panel.reference_ids,
                                  "reader_code_sha256": reader_hash,
                                  "source_files": provenance,
                                  "source_codes": {1: "embedded responses", 2: "legacy response sidecar with exact result hash",
                                                   3: "modern batch run with manifest", 4: "legacy batch with matching prompts and totals",
                                                   5: "keyed trace sidecar"}}
