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


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

COMPARISON_DIR = REPO_ROOT / "data/03_comparisons/phase6b_variations_pruned"
COMPARISON_MANIFEST = COMPARISON_DIR / "phase6b_variations_pruned_final_manifest.json"
CATEGORY_INDEX = COMPARISON_DIR / "category_index.json"
MODEL_RUNS_DIR = REPO_ROOT / "outputs/04_model_runs"
MODEL_RUN_INDEX = MODEL_RUNS_DIR / "model_run_index.json"
ANALYSIS_DIR = REPO_ROOT / "outputs/05_analysis"
LIGHTWEIGHT_OUTPUT_NAMES = {".gitkeep", "README.md", "model_run_index.json"}

COMPARISON_PREFIX = "phase6b_variations_pruned_final_"
COMPARISON_SUFFIX = "_comparisons.json"


try:
    from llm_coherence.config import MODEL_CONFIGS
except Exception:
    MODEL_CONFIGS = {}


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


def model_run_payloads_present() -> bool:
    if not MODEL_RUNS_DIR.exists():
        return False
    return any(
        path.is_file() and path.name not in LIGHTWEIGHT_OUTPUT_NAMES
        for path in MODEL_RUNS_DIR.rglob("*")
    )


def comparison_short_name(filename: str) -> tuple[str, str, str]:
    test_name = filename.removesuffix(COMPARISON_SUFFIX)
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
        "layout": "flat",
        "generated_from": COMPARISON_MANIFEST.name,
        "note": (
            "Comparison files remain flat so existing runners can resolve "
            "manifest filenames relative to this data directory. Use this "
            "index for category browsing without changing paths."
        ),
        "total_variation_sets": sum(category["count"] for category in categories),
        "categories": categories,
    }


def model_reasoning_mode(model_key: str) -> str:
    if model_key not in MODEL_CONFIGS:
        return "not_configured"
    if "logprobs" in model_key:
        return "logprob_scored"
    cfg = MODEL_CONFIGS[model_key]
    extra_body = cfg.extra_body or {}
    if model_key.endswith("-thinking"):
        return "thinking_on"
    if extra_body.get("thinking", {}).get("type") == "enabled":
        return "thinking_on"
    if extra_body.get("reasoning", {}).get("enabled") is True:
        return "thinking_on"
    if extra_body.get("reasoning_effort") not in (None, "none"):
        return "thinking_on"
    return "thinking_off"


def count_analysis_outputs(model_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not ANALYSIS_DIR.exists():
        return counts
    for stage_dir in sorted(path for path in ANALYSIS_DIR.iterdir() if path.is_dir()):
        counts[stage_dir.name] = sum(1 for path in stage_dir.rglob(f"{model_key}.json"))
    return counts


def build_model_run_index() -> dict[str, Any]:
    expected_variation_sets = len(load_json(COMPARISON_MANIFEST)["variation_files"])
    models = []

    for model_dir in sorted(path for path in MODEL_RUNS_DIR.iterdir() if path.is_dir()):
        model_key = model_dir.name
        files = [path for path in model_dir.rglob("*") if path.is_file()]
        payload_files = [path for path in files if path.name != ".gitkeep"]
        result_files = [
            path
            for path in payload_files
            if path.name == "results.json" or path.name.endswith("_results.json")
        ]
        ladder_dirs = [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("phase6b_ladder_")]
        category_summary_dirs = [
            path
            for path in model_dir.iterdir()
            if path.is_dir() and path.name.startswith("phase6b_by_category_")
        ]
        reasoning_trace_files = [path for path in payload_files if path.name == "reasoning_traces.jsonl"]
        cost_files = [
            path
            for path in payload_files
            if path.name in {"cost_summary.json", "phase6b_cost_log.json"}
        ]

        if model_key in MODEL_CONFIGS:
            kind = "model"
        elif result_files:
            kind = "unconfigured_model"
        else:
            kind = "utility"

        if len(result_files) >= expected_variation_sets:
            completeness = "complete_or_extra"
        elif result_files:
            completeness = "partial"
        else:
            completeness = "no_results"

        models.append(
            {
                "model_key": model_key,
                "kind": kind,
                "reasoning_mode": model_reasoning_mode(model_key),
                "configured": model_key in MODEL_CONFIGS,
                "expected_variation_sets": expected_variation_sets if kind != "utility" else 0,
                "result_files": len(result_files),
                "ladder_result_dirs": len(ladder_dirs),
                "reasoning_trace_files": len(reasoning_trace_files),
                "category_summary_dirs": len(category_summary_dirs),
                "cost_files": len(cost_files),
                "payload_files": len(payload_files),
                "completeness": completeness,
                "analysis_outputs": count_analysis_outputs(model_key),
                "path": rel(model_dir),
            }
        )

    return {
        "schema_version": "1.0",
        "source": rel(MODEL_RUNS_DIR),
        "expected_variation_sets": expected_variation_sets,
        "note": (
            "Snapshot inventory generated from local publication output "
            "payloads. Raw output payloads are excluded from Git so the public "
            "repository stays browsable. Top-level model-run folders are model "
            "keys; thinking-on variants use separate keys and folders, usually "
            "with the -thinking suffix."
        ),
        "models": models,
    }


def validate_comparison_files(errors: list[str]) -> None:
    manifest = load_json(COMPARISON_MANIFEST)
    for filename in manifest["variation_files"]:
        path = COMPARISON_DIR / filename
        if not path.exists():
            errors.append(f"missing comparison file: {rel(path)}")


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
        help="Refresh category_index.json and model_run_index.json before validating.",
    )
    args = parser.parse_args()

    category_index = build_category_index()
    has_model_run_payloads = model_run_payloads_present()
    model_run_index = build_model_run_index() if has_model_run_payloads else None

    if args.write_indexes:
        write_json(CATEGORY_INDEX, category_index)
        if model_run_index is not None:
            write_json(MODEL_RUN_INDEX, model_run_index)
        else:
            print(f"Skipping {rel(MODEL_RUN_INDEX)} refresh; no model-run payloads are present.")

    errors: list[str] = []
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
            print("\nRun `PYTHONPATH=src python scripts/validate_artifacts.py --write-indexes` to refresh indexes.")
        return 1

    print("Artifact validation passed.")
    print(f"  comparison sets: {category_index['total_variation_sets']}")
    if model_run_index is None:
        print("  model-run payloads: absent; using snapshot index only")
    else:
        print(f"  model folders: {len(model_run_index['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
