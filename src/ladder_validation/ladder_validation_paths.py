"""Shared paths for phase6b ladder validation pipelines (llm_coherence layout)."""

from __future__ import annotations

from pathlib import Path

# This file lives in src/ladder_validation/; three parents up is the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Aliases kept for downstream imports that still reference these names.
BASE_DIR = REPO_ROOT
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Pipeline data directories.
LADDER_VALIDATION_DIR = DATA_DIR / "ladder_validation"
VARIATIONS_PRUNED_DIR = LADDER_VALIDATION_DIR / "variations_pruned"
GENERATE_VARIATIONS_DIR = DATA_DIR / "generate_variations"

# Within-ladder validation summaries land under outputs/.
WITHIN_LADDER_OUTPUTS_DIR = OUTPUTS_DIR / "ladder_validation"

PAIRTEST_DIR_NAME = "within_ladder_validation_pairtest"
PROPERTY_DIR_NAME = "within_ladder_validation_property"
RANKING_DIR_NAME = "within_ladder_validation_ranking"

PAIRTEST_OUTPUT_DIR = LADDER_VALIDATION_DIR / PAIRTEST_DIR_NAME
PROPERTY_OUTPUT_DIR = LADDER_VALIDATION_DIR / PROPERTY_DIR_NAME
RANKING_OUTPUT_DIR = LADDER_VALIDATION_DIR / RANKING_DIR_NAME

RUN_DIR_BY_NAME: dict[str, Path] = {
    PAIRTEST_DIR_NAME: PAIRTEST_OUTPUT_DIR,
    PROPERTY_DIR_NAME: PROPERTY_OUTPUT_DIR,
    RANKING_DIR_NAME: RANKING_OUTPUT_DIR,
}

PRUNED_PAIRTEST_FILENAME = "phase6b_variations_pairtest_pruned.json"
PRUNED_PROPERTY_FILENAME = "phase6b_variations_prop_pruned.json"
PRUNED_RANKING_FILENAME = "phase6b_variations_ranking_pruned.json"
PRUNED_FINAL_FILENAME = "phase6b_variations_pruned_final.json"
PRUNED_FINAL_REPORT_FILENAME = "phase6b_variations_pruned_final_report.json"

# Intermediate prunings live under data/ladder_validation/variations_pruned/.
PRUNED_PAIRTEST_PATH = VARIATIONS_PRUNED_DIR / PRUNED_PAIRTEST_FILENAME
PRUNED_PROPERTY_PATH = VARIATIONS_PRUNED_DIR / PRUNED_PROPERTY_FILENAME
PRUNED_RANKING_PATH = VARIATIONS_PRUNED_DIR / PRUNED_RANKING_FILENAME
# The final intersection (canonical 100-ladder set) lives under data/generate_variations/.
PRUNED_FINAL_PATH = GENERATE_VARIATIONS_DIR / PRUNED_FINAL_FILENAME
PRUNED_FINAL_REPORT_PATH = GENERATE_VARIATIONS_DIR / PRUNED_FINAL_REPORT_FILENAME

# Default input shared across validation/pruning/experiment scripts.
DEFAULT_VARIATIONS_INPUT = PRUNED_FINAL_PATH


def run_output_dir(subdir_name: str) -> Path:
    if subdir_name not in RUN_DIR_BY_NAME:
        raise ValueError(f"Unknown validation subdir: {subdir_name!r}")
    return RUN_DIR_BY_NAME[subdir_name]


def validation_run_dir_relative(subdir_name: str) -> str:
    return str(run_output_dir(subdir_name).relative_to(REPO_ROOT)).replace("\\", "/")


def pairtest_output_dir_relative() -> str:
    return validation_run_dir_relative(PAIRTEST_DIR_NAME)


def property_output_dir_relative() -> str:
    return validation_run_dir_relative(PROPERTY_DIR_NAME)


def ranking_output_dir_relative() -> str:
    return validation_run_dir_relative(RANKING_DIR_NAME)
