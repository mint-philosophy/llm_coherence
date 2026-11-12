"""Count which runner produced each audited full-ladder ranking.

Uses the audit's replacement rule: a started recovery request replaces the
original outcome for that ladder. Original requests went through litellm;
recovery requests went straight to the endpoint recorded in requests.json.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from llm_coherence.experiments import full_ladder_ranking as ranking
from llm_coherence.experiments.full_ladder_ranking_recovery import read_log


def request_paths(source, recovery):
    manifest = json.loads((source / "manifest.json").read_text())
    tasks, _, current = ranking.prepare(ranking.DEFAULT_INPUT, list(manifest["models"]), manifest["seed"])
    if current != manifest:
        raise ValueError("Source protocol changed.")
    allowed = {t.ladder_id for t in tasks}
    pending = json.loads((recovery / "requests.json").read_text())
    new_started, _ = read_log(recovery / "responses.jsonl", {r["request_id"] for r in pending})
    rows = []
    for model in manifest["models"]:
        started, _ = ranking.read_events(source / model / "responses.jsonl", allowed)
        recovered = [r for r in pending if r["model_key"] == model and r["request_id"] in new_started]
        recovered_ids = {r["ladder_id"] for r in recovered}
        if len(recovered_ids) != len(recovered):
            raise ValueError(f"Two recovery requests for one ladder: {model}")
        reasons = Counter(r["reason"] for r in recovered)
        rows.append({
            "model_key": model,
            "requested_ladders": len(started | recovered_ids),
            "original_runner": len(started - recovered_ids),
            "recovery_runner": len(recovered_ids),
            "recovery_endpoint": ";".join(sorted({r["url"] for r in recovered})),
            "recovery_reasons": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = request_paths(args.source.resolve(), args.recovery.resolve())
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"{len(rows)} configurations, {sum(r['recovery_runner'] for r in rows)} rankings from recovery")
