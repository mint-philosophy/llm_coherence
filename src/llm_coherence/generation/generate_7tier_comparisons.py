#!/usr/bin/env python3
"""
Generate forced-choice comparisons for 7-tier variation sets.

Reuses the 30 cross-ladder reference statements from
data/03_comparisons/comparison_sample.json.
7 tiers x 30 comparisons = 210 comparisons per variation set.

Default input: data/01_ladders/phase6b_variations_pruned_final.json

Default output directory: data/03_comparisons/phase6b_variations_pruned/

Outputs are named from the variations file stem, e.g. for
phase6b_variations_pruned_final.json:
  - data/03_comparisons/phase6b_variations_pruned/phase6b_variations_pruned_final_manifest.json
  - data/03_comparisons/phase6b_variations_pruned/phase6b_variations_pruned_final_{id}_comparisons.json

Usage:
    PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons
    PYTHONPATH=src python -m llm_coherence.generation.generate_7tier_comparisons --variations data/01_ladders/phase6b_variations_pruned_final.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llm_coherence.paths import (  # noqa: E402
    COMPARISON_SAMPLE_PATH,
    COMPARISONS_DIR,
    PRUNED_FINAL_PATH,
    REPO_ROOT,
)

DEFAULT_COMPARISONS_OUTPUT_DIR = COMPARISONS_DIR
DEFAULT_COMPARISON_SAMPLE_PATH = COMPARISON_SAMPLE_PATH

USABLE_VARIATION_STATUSES = frozenset({"success", "fixed_via_critique_rewrite_v2"})


def load_comparison_sample(path: Path) -> list[dict]:
    """Load reference comparison statements (30 by default).

    Accepts:
      - data/comparison_sample.json: object with key \"comparison_sample\"
      - legacy phase5_manifest.json: object with key \"comparison_sample\"
      - a bare JSON list of statement objects
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "comparison_sample" in data:
        sample = data["comparison_sample"]
        if not isinstance(sample, list):
            raise TypeError(f"{path}: \"comparison_sample\" must be a list")
        return sample
    raise ValueError(
        f"{path}: expected a JSON list or an object with \"comparison_sample\" key"
    )


def sanitize_id(variation_id: str) -> str:
    return variation_id.replace(" ", "_").replace("/", "_")


def variations_stem(variations_path: Path) -> str:
    """e.g. phase6b_variations_pruned_final.json -> phase6b_variations_pruned_final."""
    return variations_path.stem


def comparison_basename(variations_path: Path, ladder_id: str) -> str:
    """Per-ladder comparison filename (no directory)."""
    return f"{variations_stem(variations_path)}_{sanitize_id(ladder_id)}_comparisons.json"


def manifest_basename(variations_path: Path) -> str:
    return f"{variations_stem(variations_path)}_manifest.json"


def test_name_for_ladder(variations_path: Path, ladder_id: str) -> str:
    """test_name field inside each comparison file (matches file stem before _comparisons)."""
    return f"{variations_stem(variations_path)}_{sanitize_id(ladder_id)}"


def generate_comparisons_for_variation(
    variation: dict,
    comparison_statements: list[dict],
) -> list[dict]:
    """Generate 7 * len(comparison_statements) comparisons for one variation set."""
    tiers_sorted = sorted(variation["variations"], key=lambda t: t["tier"])
    comparisons = []

    for comp_stmt in comparison_statements:
        for tier in tiers_sorted:
            comparisons.append({
                "outcome_a": {
                    "text": tier["text"],
                    "variation_id": variation["original_statement_id"],
                    "tier": tier["tier"],
                    "tier_label": tier["label"],
                    "category": variation["category"],
                    "identified_property": variation["identified_property"],
                    "valence": variation.get("valence", "positive"),
                },
                "outcome_b": {
                    "text": comp_stmt["text"],
                    "comparison_id": comp_stmt.get("sample_index", comp_stmt.get("pool_index", 0)),
                    "comparison_category": comp_stmt["category"],
                },
            })

    return comparisons


def filter_usable_variations(variations: list[dict]) -> list[dict]:
    kept = [v for v in variations if v.get("status") in USABLE_VARIATION_STATUSES]
    skipped = len(variations) - len(kept)
    if skipped:
        print(
            f"Skipped {skipped} ladders with status not in "
            f"{sorted(USABLE_VARIATION_STATUSES)}"
        )
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 6b comparisons for 7-tier variation sets "
            "(local JSON only; no LLM or network calls)."
        )
    )
    default_variations = PRUNED_FINAL_PATH.relative_to(REPO_ROOT)
    parser.add_argument(
        "--variations",
        type=Path,
        default=default_variations,
        help=(
            "Path to variations JSON "
            f"(default: {default_variations.as_posix()})"
        ),
    )
    default_sample = DEFAULT_COMPARISON_SAMPLE_PATH.relative_to(REPO_ROOT)
    parser.add_argument(
        "--comparison-sample",
        type=Path,
        default=default_sample,
        help=(
            "JSON file with cross-ladder reference statements "
            f"(default: {default_sample.as_posix()}; object with \"comparison_sample\" list, "
            "or a bare JSON list)"
        ),
    )
    parser.add_argument(
        "--phase5-manifest",
        type=Path,
        default=None,
        help=(
            "Deprecated: use --comparison-sample. If set, overrides --comparison-sample "
            "(read comparison_sample from a full phase5_manifest.json)."
        ),
    )
    default_output = DEFAULT_COMPARISONS_OUTPUT_DIR.relative_to(REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=(
            "Output directory "
            f"(default: {default_output.as_posix()})"
        ),
    )
    args = parser.parse_args()

    base = REPO_ROOT
    var_path = (base / args.variations).resolve() if not args.variations.is_absolute() else args.variations
    if not var_path.exists():
        raise FileNotFoundError(f"Variations file not found: {var_path}")

    out = (base / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    stem = variations_stem(var_path)

    print("\n" + "=" * 70)
    print("PHASE 6b: GENERATE COMPARISONS FOR 7-TIER VARIATION SETS")
    print("=" * 70 + "\n")
    print(f"Variations source: {var_path}")
    print(f"Output directory:  {out}")
    print(f"Output stem:       {stem}\n")

    with open(var_path, encoding="utf-8") as f:
        variations = json.load(f)
    if not isinstance(variations, list):
        raise ValueError(f"Expected a JSON list in {var_path}")

    variations = filter_usable_variations(variations)
    print(f"Loaded {len(variations)} usable variation sets (7 tiers each)")

    if args.phase5_manifest is not None:
        sample_path = (
            args.phase5_manifest.resolve()
            if args.phase5_manifest.is_absolute()
            else (base / args.phase5_manifest).resolve()
        )
    else:
        sample_path = (
            args.comparison_sample.resolve()
            if args.comparison_sample.is_absolute()
            else (base / args.comparison_sample).resolve()
        )
    if not sample_path.exists():
        raise FileNotFoundError(f"Comparison sample file not found: {sample_path}")
    sampled = load_comparison_sample(sample_path)
    print(f"Reusing {len(sampled)} comparison statements from {sample_path}")

    n_tiers = 7
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "6b",
        "description": "7-tier forced-choice comparisons (valence-aware variation sets)",
        "variations_source": str(var_path),
        "variations_stem": stem,
        "n_variations": len(variations),
        "n_comparison_samples": len(sampled),
        "n_tiers": n_tiers,
        "total_comparisons": len(variations) * len(sampled) * n_tiers,
        "comparison_sample_source": str(sample_path),
        "comparison_sample": sampled,
        "variation_files": [],
    }

    for var in variations:
        ladder_id = var["original_statement_id"]
        test_name = test_name_for_ladder(var_path, ladder_id)
        comps = generate_comparisons_for_variation(var, sampled)

        fname = comparison_basename(var_path, ladder_id)
        payload = {
            "test_name": test_name,
            "test_type": "phase6b_monotonicity",
            "variations_source": str(var_path),
            "variation_id": ladder_id,
            "variation_category": var["category"],
            "identified_property": var["identified_property"],
            "valence": var.get("valence", "positive"),
            "n_comparison_statements": len(sampled),
            "n_tiers": n_tiers,
            "num_comparisons": len(comps),
            "comparisons": comps,
        }
        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        manifest["variation_files"].append(fname)

    manifest_path = out / manifest_basename(var_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(variations)} comparison files")
    print(f"Manifest: {manifest_path}")
    print(f"Total comparisons: {manifest['total_comparisons']}")
    # Rough budget for a separate experiment runner (not used by this script).
    downstream_calls = manifest["total_comparisons"] * 2 * 10
    print(
        "Downstream estimate only (this script made 0 API calls): "
        f"if you run elicitation with flipped prompts and 10 trials, "
        f"expect on the order of {downstream_calls:,} model calls."
    )
    print()


if __name__ == "__main__":
    main()
