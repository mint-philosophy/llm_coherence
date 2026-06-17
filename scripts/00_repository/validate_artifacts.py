#!/usr/bin/env python3
"""Validate and refresh lightweight artifact indexes.

This script keeps browsable indexes in sync with the generated artifacts
without moving the artifacts themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "llm_coherence").exists():
            return path
    raise RuntimeError("Could not find llm_coherence repository root")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

LIGHTWEIGHT_OUTPUT_NAMES = {".gitkeep", "README.md", "model_run_index.json"}

COMPARISON_PREFIX = "phase6b_variations_pruned_final_"
COMPARISON_SUFFIX = "_comparisons.json"


try:
    from llm_coherence.config import MODEL_CONFIGS
    from llm_coherence.paths import COMPARISONS_DIR, MODEL_RUN_INDEX_PATH
    from llm_coherence.runtime.model_run_index import (
        build_model_run_index,
        model_run_payloads_present,
        write_model_run_index,
    )
except Exception:
    MODEL_CONFIGS = {}
    COMPARISONS_DIR = REPO_ROOT / "data" / "06_forced_choice_inputs" / "phase6b_variations_pruned"
    MODEL_RUN_INDEX_PATH = REPO_ROOT / "results" / "model_run_index.json"

    def model_run_payloads_present() -> bool:  # type: ignore[misc]
        return False

    def build_model_run_index(**_kwargs):  # type: ignore[misc]
        return {"schema_version": "1.1", "models": []}

    def write_model_run_index(payload, **kwargs):  # type: ignore[misc]
        write_json(MODEL_RUN_INDEX_PATH, payload)

COMPARISON_DIR = COMPARISONS_DIR
COMPARISON_MANIFEST = COMPARISON_DIR / "phase6b_variations_pruned_final_manifest.json"
CATEGORY_INDEX = COMPARISON_DIR / "category_index.json"
FINAL_LADDERS_PATH = REPO_ROOT / "data" / "05_ladder_validation" / "phase6b_variations_pruned_final.json"
MODEL_RUN_INDEX = MODEL_RUN_INDEX_PATH


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def comparison_short_name(filename: str) -> tuple[str, str, str]:
    test_name = Path(filename).name.removesuffix(COMPARISON_SUFFIX)
    short = test_name.removeprefix(COMPARISON_PREFIX)
    category, ladder_id = short.rsplit("_", 1)
    return test_name, category, ladder_id


def build_category_index() -> dict[str, Any]:
    manifest = load_json(COMPARISON_MANIFEST)
    groups: dict[str, list[dict[str, str]]] = {}
    for filename in manifest["variation_files"]:
        test_name, category, ladder_id = comparison_short_name(filename)
        groups.setdefault(category, []).append(
            {
                "ladder_id": ladder_id,
                "test_name": test_name,
                "file": filename,
            }
        )

    categories = []
    for category in sorted(groups):
        entries = sorted(groups[category], key=lambda item: int(item["ladder_id"]))
        categories.append(
            {
                "category": category,
                "label": category.replace("_", " "),
                "count": len(entries),
                "variation_files": entries,
            }
        )

    return {
        "schema_version": "1.0",
        "layout": "category_directories",
        "generated_from": COMPARISON_MANIFEST.name,
        "note": (
            "Comparison files are grouped by category for browsing. The "
            "manifest stores category-relative file paths; runners still use "
            "the flat test_name stored inside each JSON payload."
        ),
        "total_variation_sets": sum(category["count"] for category in categories),
        "categories": categories,
    }


def validate_comparison_files(errors: list[str]) -> None:
    manifest = load_json(COMPARISON_MANIFEST)
    for filename in manifest["variation_files"]:
        path = COMPARISON_DIR / filename
        if not path.exists():
            errors.append(f"missing comparison file: {rel(path)}")


def validate_final_ladders(errors: list[str]) -> int:
    if not FINAL_LADDERS_PATH.exists():
        errors.append(f"missing final ladders: {rel(FINAL_LADDERS_PATH)}")
        return 0
    return len(load_json(FINAL_LADDERS_PATH))


def validate_index(path: Path, expected: dict[str, Any], errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing index: {rel(path)}")
        return
    actual = load_json(path)
    if actual != expected:
        errors.append(f"stale index: {rel(path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated artifact indexes.")
    parser.add_argument(
        "--write-indexes",
        action="store_true",
        help="Refresh category_index.json and results/model_run_index.json before validating.",
    )
    args = parser.parse_args()

    category_index = build_category_index()
    has_model_run_payloads = model_run_payloads_present()
    model_run_index = build_model_run_index() if has_model_run_payloads else None

    if args.write_indexes:
        write_json(CATEGORY_INDEX, category_index)
        if model_run_index is not None:
            write_model_run_index(model_run_index, output_path=MODEL_RUN_INDEX)
        else:
            print(f"Skipping {rel(MODEL_RUN_INDEX)} refresh; no model-run payloads are present.")

    errors: list[str] = []
    final_ladder_count = validate_final_ladders(errors)
    validate_comparison_files(errors)
    validate_index(CATEGORY_INDEX, category_index, errors)
    if model_run_index is not None:
        validate_index(MODEL_RUN_INDEX, model_run_index, errors)
    elif not MODEL_RUN_INDEX.exists():
        errors.append(f"missing snapshot index: {rel(MODEL_RUN_INDEX)}")

    if errors:
        print("Artifact validation failed:")
        for error in errors:
            print(f"  - {error}")
        if not args.write_indexes:
            print(
                "\nRun `PYTHONPATH=src python scripts/00_repository/validate_artifacts.py "
                "--write-indexes` to refresh indexes."
            )
        return 1

    print("Artifact validation passed.")
    print(f"  final ladders: {final_ladder_count}")
    print(f"  comparison sets: {category_index['total_variation_sets']}")
    if model_run_index is None:
        print("  model-run payloads: absent; using snapshot index only")
    else:
        print(f"  model folders: {len(model_run_index['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
