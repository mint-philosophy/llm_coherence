"""Shared paths for the ordered llm_coherence experiment pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Aliases kept for downstream imports that still reference these names.
BASE_DIR = REPO_ROOT
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Ordered pipeline data directories. These follow the methodology sequence in
# the paper, from source outcomes through final forced-choice experiment inputs.
SOURCE_OPTIONS_DATA_DIR = DATA_DIR / "01_source_outcomes"
CATEGORY_FILTERING_DATA_DIR = DATA_DIR / "02_category_filtering"
OUTCOME_SCREENING_DATA_DIR = DATA_DIR / "03_outcome_screening"
LADDER_GENERATION_DATA_DIR = DATA_DIR / "04_ladder_generation"
VALIDATION_DATA_DIR = DATA_DIR / "05_ladder_validation"
FORCED_CHOICE_INPUTS_DATA_DIR = DATA_DIR / "06_forced_choice_inputs"

LADDER_VALIDATION_DIR = VALIDATION_DATA_DIR / "ladder_validation"
VARIATIONS_PRUNED_DIR = LADDER_VALIDATION_DIR / "variations_pruned"
GENERATE_VARIATIONS_DIR = LADDER_GENERATION_DATA_DIR
COMPARISONS_DIR = FORCED_CHOICE_INPUTS_DATA_DIR / "phase6b_variations_pruned"
COMPARISON_SAMPLE_PATH = FORCED_CHOICE_INPUTS_DATA_DIR / "comparison_sample.json"

SOURCE_OPTIONS_PATH = SOURCE_OPTIONS_DATA_DIR / "options_hierarchical.json"
FILTERED_OPTIONS_PATH = CATEGORY_FILTERING_DATA_DIR / "options_hierarchical_filtered_phase1.json"
PHASE1_FILTERING_REPORT_PATH = CATEGORY_FILTERING_DATA_DIR / "phase1_filtering_report.json"
PHASE2_FILTERING_RESULTS_PATH = OUTCOME_SCREENING_DATA_DIR / "phase2_filtering_results.json"
PHASE3_VARIATIONS_PATH = LADDER_GENERATION_DATA_DIR / "phase3_variations.json"
PHASE6B_VARIATIONS_PATH = LADDER_GENERATION_DATA_DIR / "phase6b_variations.json"

# Ordered pipeline output directories. Canonical generated inputs live under
# data/; generated run, analysis, validation-summary, figure, and table outputs
# live under results/. outputs/ is reserved for scratch files and checkpoints.
LADDER_OUTPUTS_DIR = OUTPUTS_DIR / "04_ladder_generation"
VALIDATION_OUTPUTS_DIR = RESULTS_DIR / "05_ladder_validation"
COMPARISON_OUTPUTS_DIR = FORCED_CHOICE_INPUTS_DATA_DIR
MODEL_RUNS_OUTPUT_DIR = RESULTS_DIR / "07_model_runs"
ANALYSIS_OUTPUTS_DIR = RESULTS_DIR / "08_analysis"
REPORT_OUTPUTS_DIR = RESULTS_DIR / "09_figures_tables"

# Within-ladder validation summaries land under results/.
WITHIN_LADDER_OUTPUTS_DIR = VALIDATION_OUTPUTS_DIR / "ladder_validation"

PAIRTEST_DIR_NAME = "within_ladder_validation_pairtest"
PROPERTY_DIR_NAME = "within_ladder_validation_property"
RANKING_DIR_NAME = "within_ladder_validation_ranking"

PAIRTEST_OUTPUT_DIR = VALIDATION_OUTPUTS_DIR / PAIRTEST_DIR_NAME
PROPERTY_OUTPUT_DIR = VALIDATION_OUTPUTS_DIR / PROPERTY_DIR_NAME
RANKING_OUTPUT_DIR = VALIDATION_OUTPUTS_DIR / RANKING_DIR_NAME

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

# Intermediate prunings live under data/05_ladder_validation/ladder_validation/variations_pruned/.
PRUNED_PAIRTEST_PATH = VARIATIONS_PRUNED_DIR / PRUNED_PAIRTEST_FILENAME
PRUNED_PROPERTY_PATH = VARIATIONS_PRUNED_DIR / PRUNED_PROPERTY_FILENAME
PRUNED_RANKING_PATH = VARIATIONS_PRUNED_DIR / PRUNED_RANKING_FILENAME
# The final intersection (canonical 100-ladder set) lives under data/05_ladder_validation/.
PRUNED_FINAL_PATH = VALIDATION_DATA_DIR / PRUNED_FINAL_FILENAME
PRUNED_FINAL_REPORT_PATH = VALIDATION_DATA_DIR / PRUNED_FINAL_REPORT_FILENAME

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
