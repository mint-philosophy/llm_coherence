"""Build and write the public ``results/model_run_index.json`` snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_coherence.config import MODEL_CONFIGS
from llm_coherence.paths import (
    ANALYSIS_OUTPUTS_DIR,
    COHERENCE_TEST_SUBDIR,
    COMPARISONS_DIR,
    LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR,
    LADDER_VS_COMPARISON_SUBDIR,
    MODEL_RUN_INDEX_PATH,
    REPO_ROOT,
    WITHIN_LADDER_SUBDIR,
    WITHIN_LADDER_RUNS_OUTPUT_DIR,
)

COMPARISON_MANIFEST = COMPARISONS_DIR / "phase6b_variations_pruned_final_manifest.json"

_SKIP_OUTPUT_ROOTS = frozenset(
    {
        "checkpoints",
        "04_ladder_generation",
        "figures",
        "tables",
    }
)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


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


def _count_analysis_outputs(model_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not ANALYSIS_OUTPUTS_DIR.is_dir():
        return counts
    for stage_dir in sorted(path for path in ANALYSIS_OUTPUTS_DIR.iterdir() if path.is_dir()):
        counts[stage_dir.name] = sum(1 for path in stage_dir.rglob(f"{model_key}.json"))
    return counts


def _is_model_output_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir() or model_dir.name in _SKIP_OUTPUT_ROOTS:
        return False
    return any(
        (model_dir / subdir).is_dir()
        for subdir in (WITHIN_LADDER_SUBDIR, LADDER_VS_COMPARISON_SUBDIR)
    )


def model_run_payloads_present(
    runs_root: Path | None = None,
) -> bool:
    """Return True when ``outputs/<model>/`` contains experiment payloads."""
    root = runs_root if runs_root is not None else LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR
    if not root.is_dir():
        return False
    for model_dir in root.iterdir():
        if not _is_model_output_dir(model_dir):
            continue
        lvc = model_dir / LADDER_VS_COMPARISON_SUBDIR
        if any(lvc.rglob("results.json")):
            return True
        if (model_dir / WITHIN_LADDER_SUBDIR / "summary.json").is_file():
            return True
    return False


def build_model_run_index(
    *,
    runs_root: Path | None = None,
    manifest_path: Path = COMPARISON_MANIFEST,
) -> dict[str, Any]:
    """Inventory model-scoped trees under ``outputs/<model_key>/``."""
    root = runs_root if runs_root is not None else LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR
    with open(manifest_path, encoding="utf-8") as fh:
        expected_variation_sets = len(json.load(fh)["variation_files"])

    models: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in root.iterdir() if _is_model_output_dir(path)):
        model_key = model_dir.name
        lvc = model_dir / LADDER_VS_COMPARISON_SUBDIR
        within = model_dir / WITHIN_LADDER_SUBDIR
        coherence_dir = lvc / COHERENCE_TEST_SUBDIR if lvc.is_dir() else None

        payload_files = [
            path
            for path in model_dir.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        result_files = [
            path
            for path in payload_files
            if path.name == "results.json"
            or path.name.endswith("_results.json")
        ]
        ladder_dirs = []
        if lvc.is_dir():
            ladder_dirs = [
                path
                for path in lvc.iterdir()
                if path.is_dir()
                and (
                    path.name.startswith("phase6b_variations_prune_")
                    or path.name.startswith("phase6b_ladder_")
                )
            ]
        category_summary_dirs = []
        if coherence_dir and coherence_dir.is_dir():
            category_summary_dirs = [
                path
                for path in coherence_dir.iterdir()
                if path.is_dir() and path.name.startswith("phase6b_by_category_")
            ]
        reasoning_trace_files = [
            path for path in payload_files if path.name == "reasoning_traces.jsonl"
        ]
        cost_files = [
            path
            for path in payload_files
            if path.name in {"cost_summary.json", "phase6b_cost_log.json", "cost_log.json"}
        ]
        coherence_files = []
        if coherence_dir and coherence_dir.is_dir():
            coherence_files = list(coherence_dir.glob("phase6b_coherence_*.json"))

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
        elif (within / "summary.json").is_file():
            completeness = "within_ladder_only"
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
                "coherence_files": len(coherence_files),
                "within_ladder_summary": (within / "summary.json").is_file(),
                "cost_files": len(cost_files),
                "payload_files": len(payload_files),
                "completeness": completeness,
                "analysis_outputs": _count_analysis_outputs(model_key),
                "path": _rel(model_dir),
                "ladder_vs_comparison_path": _rel(lvc) if lvc.is_dir() else None,
                "within_ladder_path": _rel(within) if within.is_dir() else None,
            }
        )

    return {
        "schema_version": "1.1",
        "source": _rel(root),
        "within_ladder_source": _rel(WITHIN_LADDER_RUNS_OUTPUT_DIR),
        "expected_variation_sets": expected_variation_sets,
        "note": (
            "Snapshot inventory generated from local model-run payloads under "
            "outputs/<model_key>/. Instance~1 artifacts live in within_ladder/; "
            "Instance~2 ladder-vs-comparison runs live in "
            "ladder_vs_comparison_statements/ (coherence analysis under "
            "coherence_test/). Raw payloads are excluded from Git; this index is "
            "written to results/model_run_index.json."
        ),
        "models": models,
    }


def write_model_run_index(
    payload: dict[str, Any] | None = None,
    *,
    runs_root: Path | None = None,
    output_path: Path = MODEL_RUN_INDEX_PATH,
) -> Path:
    """Write the model-run index JSON under ``results/model_run_index.json``."""
    data = payload if payload is not None else build_model_run_index(runs_root=runs_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return output_path
