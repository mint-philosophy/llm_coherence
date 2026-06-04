"""
Generate pairwise-accuracy-pruned ladder set by removing ladders
below an accuracy threshold from within-ladder validation results.

Inputs:
  --full       Path to full phase6b_variations.json (146 ladders)
  --summary    Path to a model's within-ladder *_summary.json
  --threshold  Minimum accuracy to retain (default: 0.95)

Output:
  data/02_validation/ladder_validation/variations_pruned/phase6b_variations_pairtest_pruned.json

Usage:
    python -m llm_coherence.validation.generate_pairtest_pruned
    python -m llm_coherence.validation.generate_pairtest_pruned --threshold 0.90
    python -m llm_coherence.validation.generate_pairtest_pruned --summary outputs/02_validation/ladder_validation/gpt-55-openai_summary.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from llm_coherence.paths import (  # noqa: E402
    DATA_DIR,
    DEFAULT_VARIATIONS_INPUT,
    PRUNED_PAIRTEST_PATH,
    REPO_ROOT,
    WITHIN_LADDER_OUTPUTS_DIR,
)

BASE_DIR = REPO_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Prune ladders by within-ladder pairwise accuracy"
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=DEFAULT_VARIATIONS_INPUT,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=WITHIN_LADDER_OUTPUTS_DIR / "gpt-55-openai_summary.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum accuracy to retain a ladder (default: 0.95)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PRUNED_PAIRTEST_PATH,
    )
    args = parser.parse_args()

    if not args.full.exists():
        print(f"Full variations file not found: {args.full}", file=sys.stderr)
        sys.exit(1)
    if not args.summary.exists():
        print(f"Summary file not found: {args.summary}", file=sys.stderr)
        sys.exit(1)

    with open(args.full, encoding="utf-8") as f:
        full_data = json.load(f)
    with open(args.summary, encoding="utf-8") as f:
        summary = json.load(f)

    acc_lookup = {e["ladder_id"]: e["accuracy"] for e in summary["per_ladder"]}

    retained = []
    dropped = []
    for ladder in full_data:
        sid = ladder["original_statement_id"]
        acc = acc_lookup.get(sid)
        if acc is None:
            print(f"  WARNING: no accuracy data for {sid}, dropping", file=sys.stderr)
            dropped.append((sid, ladder.get("category", ""), None))
            continue
        if acc >= args.threshold:
            retained.append(ladder)
        else:
            dropped.append((sid, ladder.get("category", ""), acc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(retained, f, indent=2)

    drop_cats = Counter(cat for _, cat, _ in dropped)

    print(f"Threshold:  {args.threshold:.0%}")
    print(f"Input:      {len(full_data)} ladders")
    print(f"Retained:   {len(retained)}")
    print(f"Dropped:    {len(dropped)}")
    print()
    print("Dropped by category:")
    for cat in sorted(drop_cats):
        print(f"  {cat}: {drop_cats[cat]}")
    print()
    print("Dropped ladders:")
    for sid, cat, acc in sorted(dropped, key=lambda x: x[2] or 0):
        acc_str = f"{100*acc:.1f}%" if acc is not None else "N/A"
        print(f"  {acc_str:>6}  {sid}")
    print()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
