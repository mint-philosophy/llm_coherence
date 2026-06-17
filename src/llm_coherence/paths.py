"""Shared paths for the ordered llm_coherence experiment pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Aliases kept for downstream imports that still reference these names.
BASE_DIR = REPO_ROOT
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUTS_DIR = REPO_ROOT / "outputs"
API_KEYS_DIR = REPO_ROOT / "api_keys"

WITHIN_LADDER_SUBDIR = "within_ladder"
LADDER_VS_COMPARISON_SUBDIR = "ladder_vs_comparison_statements"
COHERENCE_TEST_SUBDIR = "coherence_test"

# Ordered pipeline data directories. These follow the methodology sequence in
# the paper, from source outcomes through final forced-choice experiment inputs.
SOURCE_OPTIONS_DATA_DIR = DATA_DIR / "01_source_outcomes"
CATEGORY_FILTERING_DATA_DIR = DATA_DIR / "02_category_filtering"
OUTCOME_SCREENING_DATA_DIR = DATA_DIR / "03_outcome_screening"
LADDER_GENERATION_DATA_DIR = DATA_DIR / "04_ladder_generation"
FORCED_CHOICE_INPUTS_DATA_DIR = DATA_DIR / "06_forced_choice_inputs"

from llm_coherence.validation.ladder_validation_paths import (  # noqa: E402
    DEFAULT_VARIATIONS_INPUT,
    LADDER_VALIDATION_OUTPUTS_DIR,
    PAIRTEST_DIR_NAME,
    PAIRTEST_OUTPUT_DIR,
    PRUNED_FINAL_FILENAME,
    PRUNED_FINAL_PATH,
    PRUNED_FINAL_REPORT_FILENAME,
    PRUNED_FINAL_REPORT_PATH,
    PRUNED_PAIRTEST_FILENAME,
    PRUNED_PAIRTEST_PATH,
    PRUNED_PROPERTY_FILENAME,
    PRUNED_PROPERTY_PATH,
    PRUNED_RANKING_FILENAME,
    PRUNED_RANKING_PATH,
    PROPERTY_DIR_NAME,
    PROPERTY_OUTPUT_DIR,
    RANKING_DIR_NAME,
    RANKING_OUTPUT_DIR,
    RUN_DIR_BY_NAME,
    TIER_DIR_NAME,
    TIER_OUTPUT_DIR,
    VALIDATION_DATA_DIR,
    VARIATIONS_INPUT_PATH,
    WITHIN_LADDER_OUTPUTS_DIR,
    WITHIN_LADDER_VALIDATION_TIER_DIR_NAME,
    normalize_cli_output_dir,
    pairtest_output_dir_relative,
    property_output_dir_relative,
    ranking_output_dir_relative,
    resolve_pruned_variations_output,
    run_output_dir,
    tier_output_dir_relative,
    validation_run_dir_relative,
    within_ladder_validation_tier_output_dir_relative,
)

GENERATE_VARIATIONS_DIR = LADDER_GENERATION_DATA_DIR
COMPARISONS_DIR = FORCED_CHOICE_INPUTS_DATA_DIR / "phase6b_variations_pruned"
COMPARISON_SAMPLE_PATH = FORCED_CHOICE_INPUTS_DATA_DIR / "comparison_sample.json"

SOURCE_OPTIONS_PATH = SOURCE_OPTIONS_DATA_DIR / "options_hierarchical.json"
FILTERED_OPTIONS_PATH = CATEGORY_FILTERING_DATA_DIR / "options_hierarchical_filtered_phase1.json"
PHASE1_FILTERING_REPORT_PATH = CATEGORY_FILTERING_DATA_DIR / "phase1_filtering_report.json"
PHASE2_FILTERING_RESULTS_PATH = OUTCOME_SCREENING_DATA_DIR / "phase2_filtering_results.json"
PHASE3_VARIATIONS_PATH = LADDER_GENERATION_DATA_DIR / "phase3_variations.json"
PHASE6B_VARIATIONS_PATH = VARIATIONS_INPUT_PATH

# Ordered pipeline output directories. Canonical generated inputs live under
# data/; ladder-vs-comparison model runs, analysis, figures, and tables live
# under results/. Instance~1 within-ladder runs write under outputs/<model>/.
LADDER_OUTPUTS_DIR = OUTPUTS_DIR / "04_ladder_generation"
COMPARISON_OUTPUTS_DIR = FORCED_CHOICE_INPUTS_DATA_DIR
WITHIN_LADDER_RUNS_OUTPUT_DIR = OUTPUTS_DIR
LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR = OUTPUTS_DIR
CHECKPOINTS_OUTPUT_DIR = OUTPUTS_DIR / "checkpoints"
MODEL_RUNS_OUTPUT_DIR = RESULTS_DIR / "07_model_runs"
ANALYSIS_OUTPUTS_DIR = RESULTS_DIR / "08_analysis"
FIGURES_OUTPUT_DIR = RESULTS_DIR / "figures"
TABLES_OUTPUT_DIR = RESULTS_DIR / "tables"
MODEL_RUN_INDEX_PATH = RESULTS_DIR / "model_run_index.json"

# Deprecated names kept for imports; all point at the flat data/05_ladder_validation/ tree.
LADDER_VALIDATION_DIR = VALIDATION_DATA_DIR
VARIATIONS_PRUNED_DIR = VALIDATION_DATA_DIR
VALIDATION_OUTPUTS_DIR = VALIDATION_DATA_DIR


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repo-relative or absolute path under the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def model_results_dir(model_key: str, results_root: Path | None = None) -> Path:
    """Resolve the model-scoped directory under ``results/07_model_runs/``."""
    from llm_coherence.config import resolve_model_results_dir

    root = results_root if results_root is not None else MODEL_RUNS_OUTPUT_DIR
    return resolve_model_results_dir(model_key, root)


def model_ladder_vs_comparison_dir(
    model_key: str, runs_root: Path | None = None
) -> Path:
    """Directory for Instance~2 ladder-vs-comparison subject-model artifacts."""
    root = runs_root if runs_root is not None else LADDER_VS_COMPARISON_RUNS_OUTPUT_DIR
    return model_results_dir(model_key, root) / LADDER_VS_COMPARISON_SUBDIR


def model_coherence_test_dir(
    model_key: str, runs_root: Path | None = None
) -> Path:
    """Directory for step-11 coherence analysis outputs (Instance~2)."""
    return model_ladder_vs_comparison_dir(model_key, runs_root) / COHERENCE_TEST_SUBDIR


def phase6b_coherence_json_path(
    model_key: str, runs_root: Path | None = None
) -> Path:
    """Path to aggregated coherence metrics JSON for a model."""
    return model_coherence_test_dir(model_key, runs_root) / f"phase6b_coherence_{model_key}.json"


def phase6b_justification_analysis_json_path(
    model_key: str, runs_root: Path | None = None
) -> Path:
    """Path to non-monotonic justification report JSON for a model."""
    return (
        model_coherence_test_dir(model_key, runs_root)
        / f"phase6b_justification_analysis_{model_key}.json"
    )


def model_within_ladder_dir(model_key: str, results_root: Path | None = None) -> Path:
    """Directory for Instance~1 within-ladder subject-model artifacts."""
    return model_results_dir(model_key, results_root) / WITHIN_LADDER_SUBDIR
