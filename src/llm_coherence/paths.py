"""Shared paths for the ordered llm_coherence experiment pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Aliases kept for downstream imports that still reference these names.
BASE_DIR = REPO_ROOT
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Ordered pipeline data directories.
LADDERS_DATA_DIR = DATA_DIR / "01_ladders"
VALIDATION_DATA_DIR = DATA_DIR / "02_validation"
COMPARISONS_DATA_DIR = DATA_DIR / "03_comparisons"

LADDER_VALIDATION_DIR = VALIDATION_DATA_DIR / "ladder_validation"
VARIATIONS_PRUNED_DIR = LADDER_VALIDATION_DIR / "variations_pruned"
GENERATE_VARIATIONS_DIR = LADDERS_DATA_DIR
COMPARISONS_DIR = COMPARISONS_DATA_DIR / "phase6b_variations_pruned"
COMPARISON_SAMPLE_PATH = COMPARISONS_DATA_DIR / "comparison_sample.json"

# Ordered pipeline output directories.
LADDER_OUTPUTS_DIR = OUTPUTS_DIR / "01_ladders"
VALIDATION_OUTPUTS_DIR = OUTPUTS_DIR / "02_validation"
COMPARISON_OUTPUTS_DIR = OUTPUTS_DIR / "03_comparisons"
MODEL_RUNS_OUTPUT_DIR = OUTPUTS_DIR / "04_model_runs"
ANALYSIS_OUTPUTS_DIR = OUTPUTS_DIR / "05_analysis"
REPORT_OUTPUTS_DIR = OUTPUTS_DIR / "06_figures_tables"

# Within-ladder validation summaries land under outputs/.
WITHIN_LADDER_OUTPUTS_DIR = VALIDATION_OUTPUTS_DIR / "ladder_validation"

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

# Intermediate prunings live under data/02_validation/ladder_validation/variations_pruned/.
PRUNED_PAIRTEST_PATH = VARIATIONS_PRUNED_DIR / PRUNED_PAIRTEST_FILENAME
PRUNED_PROPERTY_PATH = VARIATIONS_PRUNED_DIR / PRUNED_PROPERTY_FILENAME
PRUNED_RANKING_PATH = VARIATIONS_PRUNED_DIR / PRUNED_RANKING_FILENAME
# The final intersection (canonical 100-ladder set) lives under data/01_ladders/.
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
