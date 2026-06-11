#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intersect ladders across property-, pairtest-, and ranking-pruned phase6b files.

Writes:
  data/05_ladder_validation/phase6b_variations_pruned_final.json
  data/05_ladder_validation/phase6b_variations_pruned_final_report.json

Usage (from parametric_variations/):
  PYTHONPATH=src python -m llm_coherence.validation.build_final_pruned_variations
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_coherence.paths import (
    PRUNED_FINAL_PATH,
    PRUNED_FINAL_REPORT_PATH,
    PRUNED_PAIRTEST_PATH,
    PRUNED_PROPERTY_PATH,
    PRUNED_RANKING_PATH,
)

DEFAULT_PAIRTEST = PRUNED_PAIRTEST_PATH
DEFAULT_PROPERTY = PRUNED_PROPERTY_PATH
DEFAULT_RANKING = PRUNED_RANKING_PATH
DEFAULT_OUTPUT = PRUNED_FINAL_PATH
DEFAULT_REPORT = PRUNED_FINAL_REPORT_PATH


@dataclass(frozen=True)
class PrunedSource:
    key: str
    label: str
    path: Path


def load_ladders(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def ladder_id(ladder: dict[str, Any]) -> str:
    return str(ladder.get("original_statement_id", ""))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_intersection_report(
    sources: list[PrunedSource],
    *,
    output_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    id_sets: dict[str, set[str]] = {}

    for src in sources:
        ladders = load_ladders(src.path)
        loaded[src.key] = ladders
        ids = {ladder_id(l) for l in ladders}
        if "" in ids:
            raise ValueError(f"Missing original_statement_id in {src.path}")
        id_sets[src.key] = ids

    common_ids = set.intersection(*id_sets.values()) if id_sets else set()
    all_ids = set.union(*id_sets.values()) if id_sets else set()

    ladder_by_id: dict[str, dict[str, Any]] = {}
    for src in sources:
        for ladder in loaded[src.key]:
            lid = ladder_id(ladder)
            if lid in common_ids and lid not in ladder_by_id:
                ladder_by_id[lid] = ladder

    final_ladders = [ladder_by_id[lid] for lid in sorted(common_ids)]

    per_file: list[dict[str, Any]] = []
    for src in sources:
        ids = id_sets[src.key]
        n_in = len(ids)
        n_kept = len(ids & common_ids)
        n_dropped = n_in - n_kept
        dropped_ids = sorted(ids - common_ids)
        per_file.append(
            {
                "file": src.label,
                "path": str(src.path),
                "n_input_ladders": n_in,
                "n_selected_for_final": n_kept,
                "n_dropped_vs_final": n_dropped,
                "selection_fraction": round(n_kept / n_in, 6) if n_in else None,
                "drop_fraction": round(n_dropped / n_in, 6) if n_in else None,
                "dropped_ladder_ids": dropped_ids,
            }
        )

    pairwise: list[dict[str, Any]] = []
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            overlap = id_sets[a.key] & id_sets[b.key]
            pairwise.append(
                {
                    "files": [a.label, b.label],
                    "n_overlap": len(overlap),
                    "only_in_first": len(id_sets[a.key] - id_sets[b.key]),
                    "only_in_second": len(id_sets[b.key] - id_sets[a.key]),
                }
            )

    report = {
        "sources": [
            {"key": s.key, "label": s.label, "path": str(s.path)} for s in sources
        ],
        "counts": {
            "n_pairtest_pruned": len(id_sets.get("pairtest", set())),
            "n_property_pruned": len(id_sets.get("property", set())),
            "n_ranking_pruned": len(id_sets.get("ranking", set())),
            "n_union_all_sources": len(all_ids),
            "n_intersection_all_three": len(common_ids),
        },
        "per_file_selection": per_file,
        "pairwise_overlap": pairwise,
        "final_output": {
            "path": str(output_path),
            "n_ladders": len(final_ladders),
            "rule": "Keep ladder iff original_statement_id appears in all three pruned inputs",
        },
    }
    return report, final_ladders


def print_report(report: dict[str, Any]) -> None:
    c = report["counts"]
    print("=" * 72)
    print("PHASE6B PRUNED INTERSECTION")
    print("=" * 72)
    print(f"Pairtest pruned:  {c['n_pairtest_pruned']}")
    print(f"Property pruned:  {c['n_property_pruned']}")
    print(f"Ranking pruned:   {c['n_ranking_pruned']}")
    print(f"Union (any file): {c['n_union_all_sources']}")
    print(f"Intersection (all three): {c['n_intersection_all_three']}")
    print()
    for row in report["per_file_selection"]:
        print(f"{row['file']}:")
        print(
            f"  input={row['n_input_ladders']}  kept={row['n_selected_for_final']}  "
            f"dropped={row['n_dropped_vs_final']}  "
            f"({100 * row['selection_fraction']:.1f}% kept)"
        )
    print()
    for row in report["pairwise_overlap"]:
        print(f"Overlap {row['files'][0]} & {row['files'][1]}: {row['n_overlap']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build phase6b_variations_pruned_final.json")
    p.add_argument("--pairtest", type=Path, default=DEFAULT_PAIRTEST)
    p.add_argument("--property", type=Path, default=DEFAULT_PROPERTY)
    p.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sources = [
        PrunedSource("pairtest", "pairtest_pruned", args.pairtest),
        PrunedSource("property", "property_pruned", args.property),
        PrunedSource("ranking", "ranking_pruned", args.ranking),
    ]
    for src in sources:
        if not src.path.exists():
            raise FileNotFoundError(f"Missing input: {src.path}")

    report, final_ladders = build_intersection_report(sources, output_path=args.output)
    report["final_output"]["path"] = str(args.output)

    write_json(args.output, final_ladders)
    write_json(args.report, report)
    print_report(report)
    print(f"\nWrote final ladders: {args.output} ({len(final_ladders)} ladders)")
    print(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
